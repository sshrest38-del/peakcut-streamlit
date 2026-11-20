#!/usr/bin/env python3
"""
PeakCut Telegram Bot
Turn long videos into captioned vertical clips via Telegram.
"""
import os
import sys
import json
import time
import tempfile
import subprocess
import threading
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import uuid

import numpy as np
import wave

# Add the original PeakCut studio to path
sys.path.insert(0, str(Path(__file__).parent.parent / "PeakCut"))

# Telegram imports
from telegram import Update, InputFile, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler,
    ContextTypes, filters
)
from telegram.constants import ParseMode

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# ============================================================
# Configuration
# ============================================================

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN not set in environment")

# AI Provider configs (can be overridden per-user)
DEFAULT_AI_PROVIDER = os.getenv("PEAKCUT_AI_PROVIDER", "gemini")
DEFAULT_AI_MODELS = os.getenv("PEAKCUT_AI_MODELS", "gemini-2.5-flash")
DEFAULT_AI_KEY = os.getenv("PEAKCUT_AI_KEY", "")  # Optional default

# Processing limits
MAX_VIDEO_HOURS = 8
MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2GB Telegram limit
JOB_WORKERS = int(os.getenv("PEAKCUT_JOB_WORKERS", "2"))

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ============================================================
# Data Models (same as Streamlit app)
# ============================================================

@dataclass
class VideoSource:
    type: str  # "url" or "file"
    url: str = ""
    file_path: str = ""
    file_id: str = ""
    name: str = ""
    duration: float = 0
    title: str = ""
    uploader: str = ""
    thumbnail: str = ""

@dataclass
class ClipOptions:
    max_clips: int = 6
    min_len: int = 15
    max_len: int = 60
    style: str = "bounce"
    layout: str = "blur"
    quality: int = 720
    captions: bool = True
    range_start: Optional[float] = None
    range_end: Optional[float] = None

@dataclass
class AIConfig:
    provider: str = DEFAULT_AI_PROVIDER
    models: str = DEFAULT_AI_MODELS
    api_key: str = DEFAULT_AI_KEY
    base_url: str = ""

@dataclass
class Clip:
    id: str
    start: float
    end: float
    score: int
    category: str
    hook: str
    title: str
    description: str
    hashtags: List[str]
    highlight_words: List[str]
    reason: str
    style: str
    layout: str
    captions: bool
    status: str = "queued"
    file: str = ""
    cover: str = ""
    error: str = ""

@dataclass
class Job:
    id: str
    user_id: int
    chat_id: int
    message_id: int
    kind: str = "video"
    created: float = field(default_factory=time.time)
    source: Dict = field(default_factory=dict)
    options: Dict = field(default_factory=dict)
    ai: Dict = field(default_factory=dict)
    status: str = "queued"
    stage: str = "Waiting"
    progress: float = 0.0
    title: str = ""
    uploader: str = ""
    duration: float = 0
    clips: List[Dict] = field(default_factory=list)
    error: Optional[str] = None
    # Progress tracking
    start_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    eta_seconds: Optional[float] = None
    speed_mbps: float = 0.0

# ============================================================
# Global State
# ============================================================

jobs: Dict[str, Job] = {}
user_states: Dict[int, Dict] = {}  # user_id -> {step, data}
user_configs: Dict[int, Dict] = {}  # user_id -> {provider, models, api_key}

# ============================================================
# Utility Functions
# ============================================================

def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS"""
    td = timedelta(seconds=int(seconds))
    return str(td)

def format_speed(mbps: float) -> str:
    """Format speed as MB/s"""
    if mbps < 1:
        return f"{mbps*1024:.0f} KB/s"
    return f"{mbps:.1f} MB/s"

def format_progress_bar(progress: float, width: int = 20) -> str:
    """Create a text progress bar"""
    filled = int(width * progress)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {progress*100:.1f}%"

async def update_job_message(job: Job, context: ContextTypes.DEFAULT_TYPE, text: str = None):
    """Update the job progress message in Telegram"""
    if not text:
        # Build progress message
        elapsed = time.time() - job.start_time
        eta = f" | ETA: {format_time(job.eta_seconds)}" if job.eta_seconds else ""
        speed = f" | Speed: {format_speed(job.speed_mbps)}" if job.speed_mbps else ""
        
        text = (
            f"🔄 <b>Processing Job</b> `{job.id[:12]}`\n\n"
            f"📹 <b>{job.title or 'Video'}</b>\n"
            f"📊 Status: <b>{job.stage}</b>\n"
            f"{format_progress_bar(job.progress)}\n"
            f"⏱️ Elapsed: {format_time(elapsed)}{eta}{speed}\n"
            f"✂️ Clips found: {len(job.clips)}"
        )
    
    try:
        await context.bot.edit_message_text(
            chat_id=job.chat_id,
            message_id=job.message_id,
            text=text,
            parse_mode=ParseMode.HTML
        )
    except Exception as e:
        logger.debug(f"Could not update message: {e}")

def update_progress(job: Job, progress: float, stage: str, bytes_processed: int = 0):
    """Update job progress with ETA calculation"""
    job.progress = max(0.0, min(1.0, progress))
    job.stage = stage
    job.last_update = time.time()
    
    # Calculate ETA
    elapsed = job.last_update - job.start_time
    if job.progress > 0.01:
        total_estimated = elapsed / job.progress
        job.eta_seconds = max(0, total_estimated - elapsed)
    
    # Calculate speed
    if bytes_processed > 0 and elapsed > 0:
        job.speed_mbps = (bytes_processed / (1024 * 1024)) / elapsed

# ============================================================
# Video Processing Functions (same as Streamlit app)
# ============================================================

PROVIDERS = {
    "gemini": ("Gemini", "https://generativelanguage.googleapis.com/v1beta/models", "https://aistudio.google.com/apikey", "gemini-2.5-flash"),
    "openai": ("OpenAI", "https://api.openai.com/v1", "https://platform.openai.com/api-keys", "gpt-4o-mini"),
    "xai": ("Grok", "https://api.x.ai/v1", "https://console.x.ai", "grok-beta"),
    "groq": ("Groq", "https://api.groq.com/openai/v1", "https://console.groq.com/keys", "llama-3.3-70b-versatile"),
    "openrouter": ("OpenRouter", "https://openrouter.ai/api/v1", "https://openrouter.ai/keys", "google/gemini-2.5-flash"),
    "anthropic": ("Claude", "https://api.anthropic.com/v1", "https://console.anthropic.com", "claude-3-5-sonnet-20241022"),
    "ollama": ("Ollama (local)", "http://localhost:11434/v1", "http://localhost:11434", "llama3.2"),
    "custom": ("Custom (OpenAI format)", "", "", "model-name"),
}

LAYOUTS = {
    "blur": "Blur background with video centered",
    "gameplay": "Gameplay layout with facecam corner",
    "fit": "Fit video to 9:16 with black bars",
}
STYLES = {
    "bounce": "Bounce - Poppy, energetic captions",
    "hype": "Hype - Bold, high-impact captions",
    "clean": "Clean - Minimal, readable captions",
}

def run_ffmpeg(cmd: List[str], cwd: Optional[Path] = None, timeout: int = 3600) -> str:
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr[-1500:]}")
        return result.stdout
    except subprocess.TimeoutExpired:
        raise RuntimeError("ffmpeg took too long and was stopped.")

def probe_video(path: Path) -> Dict:
    out = run_ffmpeg(["ffprobe", "-v", "error", "-print_format", "json",
                      "-show_format", "-show_streams", str(path)])
    data = json.loads(out)
    v = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    if not v: raise RuntimeError("No video track.")
    if not any(s["codec_type"] == "audio" for s in data["streams"]):
        raise RuntimeError("No audio track.")
    w, h = int(v["width"]), int(v["height"])
    rot = str((v.get("side_data_list") or [{}])[0].get("rotation", "0"))
    if rot.lstrip("-") in ("90", "270"): w, h = h, w
    return {"width": w, "height": h, "duration": float(data["format"]["duration"])}

def extract_audio(src: Path, wav: Path, max_hours: float = 8):
    run_ffmpeg(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
                "-t", str(int(max_hours * 3600) + 60), "-af", "aresample=async=1:first_pts=0",
                "-c:a", "pcm_s16le", str(wav)])

def load_audio(wav: Path) -> np.ndarray:
    with wave.open(str(wav), 'rb') as wf:
        assert wf.getframerate() == 16000
        assert wf.getsampwidth() == 2
        assert wf.getnchannels() == 1
        data = wf.readframes(wf.getnframes())
    return np.frombuffer(data, dtype=np.int16)

def loudness_per_second(audio: np.ndarray) -> np.ndarray:
    n = len(audio) // 16000
    if n == 0: return np.zeros(1, dtype=np.float32)
    x = audio[:n * 16000].reshape(n, 16000).astype(np.float32) / 32768.0
    rms = np.sqrt((x ** 2).mean(axis=1) + 1e-10)
    return (20 * np.log10(rms)).astype(np.float32)

def make_transcript_lines(words: List[Dict], db: np.ndarray) -> List[Dict]:
    lines, cur = [], []
    z = (db - np.median(db)) / (db.std() + 1e-6) if len(db) > 5 else np.zeros(len(db))
    def flush():
        if cur:
            s, e = cur[0]["s"], cur[-1]["e"]
            is_loud = False
            if len(z):
                seg = z[int(s): int(e) + 1]
                is_loud = bool(len(seg) and seg.max() > 1.5)
            lines.append({"s": s, "e": e, "text": " ".join(w["w"] for w in cur), "loud": is_loud})
            cur.clear()
    for w in words:
        if cur and (w["s"] - cur[-1]["e"] > 0.7 or w["s"] - cur[0]["s"] > 10
                    or (cur[-1]["w"][-1:] in ".?!" and len(cur) >= 4)):
            flush()
        cur.append(w)
    flush()
    return lines

async def transcribe_audio(audio: np.ndarray, job: Job, context: ContextTypes.DEFAULT_TYPE) -> List[Dict]:
    """Transcribe using sherpa-onnx if available, else return empty for loudness-only"""
    try:
        import sherpa_onnx
        model_dir = Path.home() / ".cache" / "peakcut" / "parakeet"
        if not model_dir.exists():
            return []
        encoder = model_dir / "encoder.int8.onnx"
        decoder = model_dir / "decoder.int8.onnx"
        joiner = model_dir / "joiner.int8.onnx"
        tokens = model_dir / "tokens.txt"
        if not all(f.exists() for f in [encoder, decoder, joiner, tokens]):
            return []
        
        recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(encoder), decoder=str(decoder), joiner=str(joiner),
            tokens=str(tokens), num_threads=4, sample_rate=16000,
            feature_dim=80, decoding_method="greedy_search",
            model_type="nemo_transducer", provider="cpu"
        )
        
        words = []
        sr = 16000
        chunk_duration = 20
        chunk_samples = chunk_duration * sr
        
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i + chunk_samples].astype(np.float32) / 32768.0
            if len(chunk) < 4800: continue
            stream = recognizer.create_stream()
            stream.accept_waveform(16000, chunk)
            recognizer.decode_stream(stream)
            result = stream.result
            offset = i / sr
            for token, timestamp in zip(result.tokens, result.timestamps):
                word = token.replace("▁", " ").strip()
                if word:
                    words.append({"w": word, "s": round(offset + timestamp, 2),
                                  "e": round(offset + timestamp + 0.12, 2)})
            # Update progress
            pct = min(1.0, (i + chunk_samples) / len(audio))
            update_progress(job, 0.25 + 0.5 * pct, f"Transcribing... {int(pct*100)}%")
            await update_job_message(job, context)
        
        # Fix word endings
        for i, w in enumerate(words):
            nxt = words[i + 1]["s"] if i + 1 < len(words) else w["e"] + 0.4
            w["e"] = round(max(w["s"] + 0.08, min(nxt, max(w["e"], w["s"] + 0.6))), 2)
        return words
    except ImportError:
        return []
    except Exception as e:
        logger.warning(f"Transcription failed: {e}")
        return []

async def find_clips_ai(words: List[Dict], db: np.ndarray, duration: float,
                        title: str, uploader: str, options: ClipOptions,
                        ai_config: AIConfig) -> List[Dict]:
    if not words:
        return []
    
    if ai_config.api_key and ai_config.provider in PROVIDERS:
        try:
            return await find_clips_with_ai(words, db, duration, title, uploader, options, ai_config)
        except Exception as e:
            logger.warning(f"AI clip finding failed: {e}. Using loudness fallback.")
    
    return find_clips_by_loudness(words, db, duration, options)

async def find_clips_with_ai(words: List[Dict], db: np.ndarray, duration: float,
                             title: str, uploader: str, options: ClipOptions,
                             ai_config: AIConfig) -> List[Dict]:
    import httpx
    provider_info = PROVIDERS[ai_config.provider]
    base_url = ai_config.base_url or provider_info[1]
    model = ai_config.models.split(",")[0].strip()
    
    lines = make_transcript_lines(words, db)
    transcript_text = "\n".join(
        f"[{l['s']:.1f}] {'(LOUD) ' if l['loud'] else ''}{l['text']}"
        for l in lines
    )
    
    prompt = f"""You are a clip scout for short-form videos (YouTube Shorts, Reels, TikTok).
Video title: {title}
Channel: {uploader}

Pick up to {options.max_clips} moments that would work as standalone clips for someone who has never seen the video:
funny moments, fails, clutch plays, rage, shock, hot takes, or a short story with a payoff.

Rules:
- Each clip is {options.min_len} to {options.max_len} seconds. Use the [seconds] values for start and end.
- Start on the setup line, not mid-sentence. End right after the payoff or reaction.
- Do not pick intros, sponsor reads, "like and subscribe", or dead air.
- Skip weak moments. Fewer strong clips beat many weak ones. Return [] if nothing is worth clipping.
- score: 0-100, how likely a stranger keeps watching.
- hook: at most 5 words, ALL CAPS, shown as big text on the video.
- title: at most 70 characters, for the post.
- description: one or two sentences.
- hashtags: 5 tags without the # sign.
- highlight_words: 3 to 6 single words spoken in the clip that carry the punch.
- category: one of funny, fail, clutch, rage, shock, take, story, other.
- reason: one short sentence on why it works.

<transcript>
{transcript_text}
</transcript>

Return JSON array of clips."""
    
    headers = {"Content-Type": "application/json"}
    if ai_config.provider == "anthropic":
        headers["x-api-key"] = ai_config.api_key
        headers["anthropic-version"] = "2023-06-01"
        body = {"model": model, "max_tokens": 8192, "messages": [{"role": "user", "content": prompt}]}
        url = f"{base_url}/messages"
    elif ai_config.provider == "gemini":
        url = f"{base_url}/{model}:generateContent?key={ai_config.api_key}"
        body = {"contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"responseMimeType": "application/json"}}
    else:
        headers["Authorization"] = f"Bearer {ai_config.api_key}"
        body = {"model": model, "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"}, "temperature": 0.4}
        url = f"{base_url}/chat/completions"
    
    async with httpx.AsyncClient(timeout=180) as client:
        resp = await client.post(url, json=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    
    if ai_config.provider == "anthropic":
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    elif ai_config.provider == "gemini":
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    else:
        text = data["choices"][0]["message"]["content"]
    
    clips = json.loads(text)
    if not isinstance(clips, list):
        clips = clips.get("clips", [])
    
    processed = []
    for c in clips:
        if not isinstance(c, dict): continue
        s, e = float(c.get("start", 0)), float(c.get("end", 0))
        if e - s < options.min_len - 1 or e - s > options.max_len + 1: continue
        processed.append({
            "start": round(s, 2), "end": round(e, 2),
            "score": max(0, min(100, int(c.get("score", 50)))),
            "category": c.get("category", "other"),
            "hook": c.get("hook", "WAIT FOR IT").upper()[:60],
            "title": c.get("title", "Clip")[:100],
            "description": c.get("description", "")[:300],
            "hashtags": c.get("hashtags", [])[:8],
            "highlight_words": c.get("highlight_words", [])[:8],
            "reason": c.get("reason", "")[:200],
        })
    processed.sort(key=lambda c: -c["score"])
    return processed[:options.max_clips]

def find_clips_by_loudness(words: List[Dict], db: np.ndarray, duration: float,
                           options: ClipOptions) -> List[Dict]:
    z = (db - np.median(db)) / (db.std() + 1e-6)
    picks, order = [], np.argsort(-z)
    for sec in order:
        s = max(0.0, float(sec) - options.max_len * 0.6)
        e = min(duration, s + options.max_len)
        if any(not (e <= p["start"] or s >= p["end"]) for p in picks): continue
        if not any(s <= w["s"] <= e for w in words): continue
        picks.append({
            "start": round(s, 2), "end": round(e, 2), "score": 50,
            "category": "other", "hook": "WAIT FOR IT",
            "title": "Loud moment", "description": "",
            "hashtags": [], "highlight_words": [],
            "reason": "Picked by loudness because the AI was unavailable.",
        })
        if len(picks) >= options.max_clips: break
    return picks

def words_to_ass(words: List[Dict], style_name: str, video_w: int, video_h: int) -> str:
    style_configs = {
        "bounce": {"font": "Poppins ExtraBold", "size": 132, "color": "&HFFFFFF", "outline": 7},
        "hype": {"font": "Anton", "size": 152, "color": "&HFFFFFF", "outline": 7},
        "clean": {"font": "Poppins ExtraBold", "size": 100, "color": "&HFFFFFF", "outline": 5},
    }
    style = style_configs.get(style_name, style_configs["bounce"])
    ass = f"""[Script Info]
Title: PeakCut Captions
ScriptType: v4.00+
PlayResX: {video_w}
PlayResY: {video_h}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{style['font']},{style['size']},{style['color']},&H000000,&H000000,&H80000000,-1,0,0,0,100,100,0,0,1,{style['outline']},0,2,20,20,20,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    for w in words:
        start = format_ass_time(w["s"])
        end = format_ass_time(w["e"])
        text = w["w"].replace("{", "\\{").replace("}", "\\}")
        ass += f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n"
    return ass

def format_ass_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

async def render_clip(src_path: Path, out_dir: Path, clip: Clip, words: List[Dict],
                      job_width: int, job_height: int, job: Job, context: ContextTypes.DEFAULT_TYPE) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    clip_id = clip.id
    
    if clip.captions and words:
        clip_words = [w for w in words if clip.start - 1 <= w["s"] <= clip.end + 1]
        clip_words = [{**w, "s": w["s"] - clip.start, "e": w["e"] - clip.start} for w in clip_words]
        ass_content = words_to_ass(clip_words, clip.style, 1080, 1920)
        ass_path = out_dir / f"{clip_id}.ass"
        ass_path.write_text(ass_content, encoding="utf-8")
    else:
        ass_path = None
    
    layout = clip.layout
    if layout == "blur":
        filter_complex = (
            f"[0:v]scale=1080:1920:force_original_aspect_ratio=decrease,"
            f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black[v0];"
            f"[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
            f"crop=1080:1920,boxblur=40:1[bg];"
            f"[bg][v0]overlay=(W-w)/2:(H-h)/2[v]"
        )
    elif layout == "gameplay":
        filter_complex = (
            f"[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
            f"crop=1080:1920[v]"
        )
    else:
        filter_complex = (
            f"[0:v]scale=1080:1920:force_original_aspect_ratio=decrease,"
            f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black[v]"
        )
    
    if ass_path:
        filter_complex += f";[v]ass={ass_path.name}[vout]"
        map_video = "[vout]"
    else:
        map_video = "[v]"
    
    encoders = [
        ["-c:v", "h264_nvenc", "-preset", "fast", "-cq", "23"],
        ["-c:v", "h264_qsv", "-preset", "fast", "-global_quality", "23"],
        ["-c:v", "libx264", "-crf", "21", "-preset", "veryfast"],
    ]
    
    for enc in encoders:
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-ss", f"{clip.start:.2f}", "-t", f"{clip.end - clip.start:.2f}",
            "-i", str(src_path), "-filter_complex", filter_complex,
            "-map", map_video, "-map", "0:a:0",
            "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
            "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
            "-movflags", "+faststart", "-pix_fmt", "yuv420p",
        ] + enc + [str(out_dir / f"{clip_id}.mp4")]
        
        try:
            run_ffmpeg(cmd, cwd=out_dir)
            break
        except RuntimeError:
            continue
    else:
        raise RuntimeError("All encoders failed")
    
    # Cover image
    cover_t = min(1.2, (clip.end - clip.start) / 2)
    run_ffmpeg(["ffmpeg", "-y", "-v", "error", "-ss", f"{cover_t:.2f}",
                "-i", str(out_dir / f"{clip_id}.mp4"), "-frames:v", "1",
                "-q:v", "3", str(out_dir / f"{clip_id}.jpg")])
    
    return {"file": f"{clip_id}.mp4", "cover": f"{clip_id}.jpg", "layout": layout}

# ============================================================
# Telegram Bot Handlers
# ============================================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_states[user_id] = {"step": "main"}
    
    keyboard = [
        [InlineKeyboardButton("📥 From URL", callback_data="input_url"),
         InlineKeyboardButton("📁 Upload File", callback_data="input_file")],
        [InlineKeyboardButton("🤖 AI Settings", callback_data="ai_settings"),
         InlineKeyboardButton("⚙️ Clip Options", callback_data="clip_options")],
        [InlineKeyboardButton("✂️ Start Processing", callback_data="start_process")],
        [InlineKeyboardButton("ℹ️ Help", callback_data="help")]
    ]
    
    await update.message.reply_text(
        "🎬 <b>PeakCut Bot</b>\n\n"
        "Turn long videos into captioned vertical clips for Shorts, Reels & TikTok.\n\n"
        "Choose an option below:",
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 <b>PeakCut Bot Help</b>\n\n"
        "<b>Quick Start:</b>\n"
        "1. Send a YouTube/Twitch/Kick link OR upload a video file\n"
        "2. Configure AI (optional - uses loudness if no key)\n"
        "3. Set clip options (style, layout, length)\n"
        "4. Click Start Processing\n"
        "5. Get clips delivered as videos!\n\n"
        "<b>Commands:</b>\n"
        "/start - Main menu\n"
        "/config - AI configuration\n"
        "/options - Clip options\n"
        "/status - Check job status\n"
        "/cancel - Cancel current job\n"
        "/help - This message\n\n"
        "<b>AI Providers:</b> Gemini, OpenAI, Grok, Groq, OpenRouter, Claude, Ollama\n"
        "<b>Caption Styles:</b> Bounce, Hype, Clean\n"
        "<b>Layouts:</b> Blur, Gameplay, Fit",
        parse_mode=ParseMode.HTML
    )

async def config_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    cfg = user_configs.get(user_id, {})
    provider = cfg.get("provider", DEFAULT_AI_PROVIDER)
    models = cfg.get("models", DEFAULT_AI_MODELS)
    has_key = "✅" if cfg.get("api_key") else "❌"
    
    keyboard = [
        [InlineKeyboardButton(f"Provider: {PROVIDERS[provider][0]}", callback_data="cfg_provider")],
        [InlineKeyboardButton(f"Model: {models}", callback_data="cfg_model")],
        [InlineKeyboardButton(f"API Key: {has_key}", callback_data="cfg_key")],
        [InlineKeyboardButton("🔍 Test Connection", callback_data="cfg_test")],
        [InlineKeyboardButton("🔙 Back", callback_data="main_menu")]
    ]
    
    await update.message.reply_text(
        f"🤖 <b>AI Configuration</b>\n\n"
        f"Provider: <b>{PROVIDERS[provider][0]}</b>\n"
        f"Model: <b>{models}</b>\n"
        f"API Key: {has_key}\n\n"
        f"Click buttons to change:",
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def options_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    state = user_states.get(user_id, {})
    opts = state.get("options", ClipOptions())
    
    keyboard = [
        [InlineKeyboardButton(f"Max Clips: {opts.max_clips}", callback_data="opt_max_clips"),
         InlineKeyboardButton(f"Length: {opts.min_len}-{opts.max_len}s", callback_data="opt_length")],
        [InlineKeyboardButton(f"Style: {STYLES[opts.style].split(' - ')[0]}", callback_data="opt_style"),
         InlineKeyboardButton(f"Layout: {LAYOUTS[opts.layout].split(' - ')[0]}", callback_data="opt_layout")],
        [InlineKeyboardButton(f"Quality: {opts.quality}p", callback_data="opt_quality"),
         InlineKeyboardButton(f"Captions: {'On' if opts.captions else 'Off'}", callback_data="opt_captions")],
        [InlineKeyboardButton("🔙 Back", callback_data="main_menu")]
    ]
    
    await update.message.reply_text(
        f"⚙️ <b>Clip Options</b>\n\n"
        f"Max Clips: {opts.max_clips}\n"
        f"Min Length: {opts.min_len}s\n"
        f"Max Length: {opts.max_len}s\n"
        f"Style: {STYLES[opts.style]}\n"
        f"Layout: {LAYOUTS[opts.layout]}\n"
        f"Quality: {opts.quality}p\n"
        f"Captions: {'Enabled' if opts.captions else 'Disabled'}",
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_jobs = [j for j in jobs.values() if j.user_id == user_id]
    
    if not user_jobs:
        await update.message.reply_text("📭 No active jobs.")
        return
    
    text = "📋 <b>Your Jobs:</b>\n\n"
    for job in user_jobs[-5:]:  # Last 5 jobs
        elapsed = time.time() - job.start_time
        text += f"🔸 <code>{job.id[:12]}</code> | {job.status.upper()} | {job.stage}\n"
        text += f"   ⏱️ {format_time(elapsed)} | ✂️ {len(job.clips)} clips\n\n"
    
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)

async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_jobs = [j for j in jobs.values() if j.user_id == user_id and j.status in ("queued", "running")]
    
    if not user_jobs:
        await update.message.reply_text("❌ No running jobs to cancel.")
        return
    
    for job in user_jobs:
        job.status = "cancelled"
        job.stage = "Cancelled by user"
    
    await update.message.reply_text("✅ Cancelled running jobs.")

# ============================================================
# Callback Query Handlers
# ============================================================

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    data = query.data
    
    if data == "main_menu":
        await show_main_menu(query, context)
    elif data == "input_url":
        await ask_for_url(query, context)
    elif data == "input_file":
        await ask_for_file(query, context)
    elif data == "ai_settings":
        await show_ai_settings(query, context)
    elif data == "clip_options":
        await show_clip_options(query, context)
    elif data == "start_process":
        await start_processing(query, context)
    elif data == "help":
        await show_help(query, context)
    # Config callbacks
    elif data == "cfg_provider":
        await choose_provider(query, context)
    elif data.startswith("set_provider_"):
        await set_provider(query, context)
    elif data == "cfg_model":
        await ask_model(query, context)
    elif data == "cfg_key":
        await ask_api_key(query, context)
    elif data == "cfg_test":
        await test_connection(query, context)
    # Options callbacks
    elif data == "opt_max_clips":
        await ask_max_clips(query, context)
    elif data == "opt_length":
        await ask_length(query, context)
    elif data == "opt_style":
        await choose_style(query, context)
    elif data == "opt_layout":
        await choose_layout(query, context)
    elif data == "opt_quality":
        await choose_quality(query, context)
    elif data == "opt_captions":
        await toggle_captions(query, context)
    elif data.startswith("set_style_") or data.startswith("set_layout_") or data.startswith("set_quality_"):
        await set_option(query, context)

async def show_main_menu(query, context):
    keyboard = [
        [InlineKeyboardButton("📥 From URL", callback_data="input_url"),
         InlineKeyboardButton("📁 Upload File", callback_data="input_file")],
        [InlineKeyboardButton("🤖 AI Settings", callback_data="ai_settings"),
         InlineKeyboardButton("⚙️ Clip Options", callback_data="clip_options")],
        [InlineKeyboardButton("✂️ Start Processing", callback_data="start_process")],
        [InlineKeyboardButton("ℹ️ Help", callback_data="help")]
    ]
    await query.edit_message_text(
        "🎬 <b>PeakCut Bot</b>\n\nChoose an option:",
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def ask_for_url(query, context):
    user_id = query.from_user.id
    user_states[user_id] = {"step": "waiting_url"}
    await query.edit_message_text(
        "🔗 <b>Send Video URL</b>\n\n"
        "Supported: YouTube, Twitch, Kick (videos, VODs, clips)\n\n"
        "Just send the link as a message.",
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data="main_menu")]])
    )

async def ask_for_file(query, context):
    user_id = query.from_user.id
    user_states[user_id] = {"step": "waiting_file"}
    await query.edit_message_text(
        "📁 <b>Upload Video File</b>\n\n"
        "Send a video file (mp4, mov, mkv, webm, etc.)\n"
        "Max 2GB, max 8 hours.\n\n"
        "Just send the file as a message.",
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data="main_menu")]])
    )

async def show_ai_settings(query, context):
    user_id = query.from_user.id
    cfg = user_configs.get(user_id, {})
    provider = cfg.get("provider", DEFAULT_AI_PROVIDER)
    models = cfg.get("models", DEFAULT_AI_MODELS)
    has_key = "✅ Set" if cfg.get("api_key") else "❌ Not set"
    
    keyboard = [
        [InlineKeyboardButton(f"Provider: {PROVIDERS[provider][0]}", callback_data="cfg_provider")],
        [InlineKeyboardButton(f"Model: {models}", callback_data="cfg_model")],
        [InlineKeyboardButton(f"API Key: {has_key}", callback_data="cfg_key")],
        [InlineKeyboardButton("🔍 Test Connection", callback_data="cfg_test")],
        [InlineKeyboardButton("🔙 Back", callback_data="main_menu")]
    ]
    
    await query.edit_message_text(
        f"🤖 <b>AI Configuration</b>\n\n"
        f"Provider: <b>{PROVIDERS[provider][0]}</b>\n"
        f"Model: <b>{models}</b>\n"
        f"API Key: {has_key}",
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def choose_provider(query, context):
    keyboard = []
    for key, (label, _, _, _) in PROVIDERS.items():
        keyboard.append([InlineKeyboardButton(label, callback_data=f"set_provider_{key}")])
    keyboard.append([InlineKeyboardButton("🔙 Back", callback_data="ai_settings")])
    await query.edit_message_text(
        "🤖 <b>Choose AI Provider</b>",
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def set_provider(query, context):
    user_id = query.from_user.id
    provider = query.data.replace("set_provider_", "")
    if user_id not in user_configs:
        user_configs[user_id] = {}
    user_configs[user_id]["provider"] = provider
    user_configs[user_id]["models"] = PROVIDERS[provider][3]
    await show_ai_settings(query, context)

async def ask_model(query, context):
    await query.edit_message_text(
        "🤖 <b>Enter Model Name</b>\n\n"
        "Send the model name as a message.\n"
        "Examples:\n"
        "• Gemini: <code>gemini-2.5-flash</code>, <code>gemini-2.5-pro</code>\n"
        "• OpenAI: <code>gpt-4o-mini</code>, <code>gpt-4o</code>\n"
        "• Grok: <code>grok-beta</code>\n"
        "• Groq: <code>llama-3.3-70b-versatile</code>\n"
        "• Ollama: <code>llama3.2</code>",
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data="ai_settings")]])
    )
    user_id = query.from_user.id
    user_states[user_id] = {"step": "waiting_model"}

async def ask_api_key(query, context):
    await query.edit_message_text(
        "🔑 <b>Enter API Key</b>\n\n"
        "Send your API key as a message.\n"
        "It will be stored only for this session.\n\n"
        "<b>Get keys:</b>\n"
        "• Gemini: <a href='https://aistudio.google.com/apikey'>AI Studio</a>\n"
        "• OpenAI: <a href='https://platform.openai.com/api-keys'>Platform</a>\n"
        "• Grok: <a href='https://console.x.ai'>X Console</a>\n"
        "• Groq: <a href='https://console.groq.com/keys'>Groq Console</a>",
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data="ai_settings")]])
    )
    user_id = query.from_user.id
    user_states[user_id] = {"step": "waiting_api_key"}

async def test_connection(query, context):
    user_id = query.from_user.id
    cfg = user_configs.get(user_id, {})
    
    if not cfg.get("api_key") and cfg.get("provider") != "ollama":
        await query.answer("❌ No API key set!", show_alert=True)
        return
    
    await query.edit_message_text("🔍 Testing connection...")
    
    ai_config = AIConfig(
        provider=cfg.get("provider", DEFAULT_AI_PROVIDER),
        models=cfg.get("models", DEFAULT_AI_MODELS),
        api_key=cfg.get("api_key", ""),
        base_url=cfg.get("base_url", "")
    )
    
    try:
        import httpx
        provider = ai_config.provider
        provider_info = PROVIDERS[provider]
        base_url = ai_config.base_url or provider_info[1]
        model = ai_config.models.split(",")[0].strip()
        
        async with httpx.AsyncClient(timeout=30) as client:
            if provider == "gemini":
                url = f"{base_url}/{model}:generateContent?key={ai_config.api_key}"
                resp = await client.post(url, json={
                    "contents": [{"parts": [{"text": "Hi"}]}],
                    "generationConfig": {"responseMimeType": "application/json"}
                })
            elif provider == "anthropic":
                resp = await client.post(f"{base_url}/messages",
                    json={"model": model, "max_tokens": 10,
                          "messages": [{"role": "user", "content": "Hi"}]},
                    headers={"x-api-key": ai_config.api_key, "anthropic-version": "2023-06-01"})
            elif provider == "ollama":
                resp = await client.post(f"{base_url}/chat/completions",
                    json={"model": ai_config.models, "messages": [{"role": "user", "content": "Hi"}]})
            else:
                resp = await client.post(f"{base_url}/chat/completions",
                    json={"model": model, "messages": [{"role": "user", "content": "Hi"}]},
                    headers={"Authorization": f"Bearer {ai_config.api_key}"})
            
            if resp.status_code == 200:
                await query.edit_message_text(
                    "✅ <b>Connection Successful!</b>\n\nAPI key is working.",
                    parse_mode=ParseMode.HTML,
                    reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data="ai_settings")]])
                )
            else:
                await query.edit_message_text(
                    f"❌ <b>Failed:</b> {resp.status_code}\n{resp.text[:200]}",
                    parse_mode=ParseMode.HTML,
                    reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data="ai_settings")]])
                )
    except Exception as e:
        await query.edit_message_text(
            f"❌ <b>Error:</b> {str(e)[:200]}",
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data="ai_settings")]])
        )

async def show_clip_options(query, context):
    user_id = query.from_user.id
    state = user_states.get(user_id, {})
    opts = state.get("options", ClipOptions())
    
    keyboard = [
        [InlineKeyboardButton(f"Max Clips: {opts.max_clips}", callback_data="opt_max_clips"),
         InlineKeyboardButton(f"Length: {opts.min_len}-{opts.max_len}s", callback_data="opt_length")],
        [InlineKeyboardButton(f"Style: {STYLES[opts.style].split(' - ')[0]}", callback_data="opt_style"),
         InlineKeyboardButton(f"Layout: {LAYOUTS[opts.layout].split(' - ')[0]}", callback_data="opt_layout")],
        [InlineKeyboardButton(f"Quality: {opts.quality}p", callback_data="opt_quality"),
         InlineKeyboardButton(f"Captions: {'On' if opts.captions else 'Off'}", callback_data="opt_captions")],
        [InlineKeyboardButton("🔙 Back", callback_data="main_menu")]
    ]
    
    await query.edit_message_text(
        f"⚙️ <b>Clip Options</b>\n\n"
        f"Max Clips: {opts.max_clips}\n"
        f"Min Length: {opts.min_len}s\n"
        f"Max Length: {opts.max_len}s\n"
        f"Style: {STYLES[opts.style]}\n"
        f"Layout: {LAYOUTS[opts.layout]}\n"
        f"Quality: {opts.quality}p\n"
        f"Captions: {'Enabled' if opts.captions else 'Disabled'}",
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def ask_max_clips(query, context):
    await query.edit_message_text(
        "🔢 <b>Max Clips (1-20)</b>\n\nSend a number as message.",
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data="clip_options")]])
    )
    user_id = query.from_user.id
    user_states[user_id] = {"step": "waiting_max_clips"}

async def ask_length(query, context):
    await query.edit_message_text(
        "⏱️ <b>Clip Length Range</b>\n\nSend as: <code>min max</code> (e.g. <code>15 60</code>)\nMin: 5s, Max: 180s",
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data="clip_options")]])
    )
    user_id = query.from_user.id
    user_states[user_id] = {"step": "waiting_length"}

async def choose_style(query, context):
    keyboard = []
    for key, label in STYLES.items():
        keyboard.append([InlineKeyboardButton(label, callback_data=f"set_style_{key}")])
    keyboard.append([InlineKeyboardButton("🔙 Back", callback_data="clip_options")])
    await query.edit_message_text(
        "🎨 <b>Choose Caption Style</b>",
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def choose_layout(query, context):
    keyboard = []
    for key, label in LAYOUTS.items():
        keyboard.append([InlineKeyboardButton(label, callback_data=f"set_layout_{key}")])
    keyboard.append([InlineKeyboardButton("🔙 Back", callback_data="clip_options")])
    await query.edit_message_text(
        "📐 <b>Choose Layout</b>",
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def choose_quality(query, context):
    keyboard = []
    for q in [360, 480, 720, 1080]:
        keyboard.append([InlineKeyboardButton(f"{q}p", callback_data=f"set_quality_{q}")])
    keyboard.append([InlineKeyboardButton("🔙 Back", callback_data="clip_options")])
    await query.edit_message_text(
        "📺 <b>Choose Download Quality</b>",
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def toggle_captions(query, context):
    user_id = query.from_user.id
    state = user_states.get(user_id, {})
    opts = state.get("options", ClipOptions())
    opts.captions = not opts.captions
    state["options"] = opts
    user_states[user_id] = state
    await show_clip_options(query, context)

async def set_option(query, context):
    user_id = query.from_user.id
    state = user_states.get(user_id, {})
    opts = state.get("options", ClipOptions())
    
    if query.data.startswith("set_style_"):
        opts.style = query.data.replace("set_style_", "")
    elif query.data.startswith("set_layout_"):
        opts.layout = query.data.replace("set_layout_", "")
    elif query.data.startswith("set_quality_"):
        opts.quality = int(query.data.replace("set_quality_", ""))
    
    state["options"] = opts
    user_states[user_id] = state
    await show_clip_options(query, context)

async def start_processing(query, context):
    user_id = query.from_user.id
    state = user_states.get(user_id, {})
    source = state.get("source")
    
    if not source:
        await query.answer("❌ No video selected! Add a URL or upload a file first.", show_alert=True)
        return
    
    # Create job
    job_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    work_dir = Path(tempfile.gettempdir()) / "peakcut_jobs" / job_id
    work_dir.mkdir(parents=True)
    clips_dir = work_dir / "clips"
    clips_dir.mkdir()
    
    opts = state.get("options", ClipOptions())
    cfg = user_configs.get(user_id, {})
    ai_config = AIConfig(
        provider=cfg.get("provider", DEFAULT_AI_PROVIDER),
        models=cfg.get("models", DEFAULT_AI_MODELS),
        api_key=cfg.get("api_key", ""),
        base_url=cfg.get("base_url", "")
    )
    
    job = Job(
        id=job_id,
        user_id=user_id,
        chat_id=query.message.chat_id,
        message_id=query.message.message_id,
        source={"type": source.type, "url": source.url, "path": source.file_path, "file_id": source.file_id, "name": source.name},
        options=asdict(opts),
        ai={"provider": ai_config.provider, "model": ai_config.models.split(",")[0].strip()},
        title=source.title or source.name,
        uploader=source.uploader or "",
        duration=source.duration
    )
    jobs[job_id] = job
    
    # Send initial progress message
    msg = await context.bot.send_message(
        chat_id=query.message.chat_id,
        text=f"🔄 Starting job <code>{job_id[:12]}</code>...",
        parse_mode=ParseMode.HTML
    )
    job.message_id = msg.message_id
    
    # Start processing in background
    asyncio.create_task(process_job(job, context, source, opts, ai_config, clips_dir))

async def process_job(job: Job, context: ContextTypes.DEFAULT_TYPE, source: VideoSource, 
                      opts: ClipOptions, ai_config: AIConfig, clips_dir: Path):
    try:
        job.status = "running"
        await update_job_message(job, context)
        
        # Step 1: Get source video/audio
        update_progress(job, 0.05, "Downloading video...")
        await update_job_message(job, context)
        
        if source.type == "url":
            video_path = await download_video_section_async(source.url, clips_dir, "source", 0, job.duration or 8*3600, opts.quality, job, context)
            # Also download audio separately
            audio_path = await download_audio_async(source.url, clips_dir, job, context)
        else:
            video_path = Path(source.file_path)
            if not video_path.exists():
                # Download from Telegram
                video_path = await download_telegram_file(source.file_id, clips_dir / source.name, job, context)
            audio_path = await extract_audio_async(video_path, clips_dir / "audio.wav", job, context)
        
        if not job.duration:
            meta = probe_video(video_path)
            job.duration = meta["duration"]
            job.width, job.height = meta["width"], meta["height"]
        
        # Step 2: Transcribe
        update_progress(job, 0.2, "Extracting audio...")
        await update_job_message(job, context)
        
        audio = load_audio(audio_path)
        db = loudness_per_second(audio)
        
        update_progress(job, 0.25, "Transcribing...")
        await update_job_message(job, context)
        
        words = await transcribe_audio(audio, job, context)
        
        # Step 3: Find clips
        update_progress(job, 0.75, "Finding best moments...")
        await update_job_message(job, context)
        
        clips_data = await find_clips_ai(words, db, job.duration, job.title, job.uploader, opts, ai_config)
        
        job.clips = []
        for i, c in enumerate(clips_data):
            clip = Clip(
                id=f"clip{i+1:02d}", start=c["start"], end=c["end"],
                score=c["score"], category=c["category"], hook=c["hook"],
                title=c["title"], description=c["description"],
                hashtags=c["hashtags"], highlight_words=c["highlight_words"],
                reason=c["reason"], style=opts.style, layout=opts.layout,
                captions=opts.captions
            )
            job.clips.append(asdict(clip))
        
        # Step 4: Render clips
        if job.clips:
            for i, clip_dict in enumerate(job.clips):
                update_progress(job, 0.8 + 0.15 * (i / len(job.clips)),
                                f"Rendering clip {i+1}/{len(job.clips)}...")
                await update_job_message(job, context)
                
                clip_obj = Clip(**clip_dict)
                clip_obj.status = "rendering"
                result = await render_clip(video_path, clips_dir, clip_obj, words,
                                           job.width, job.height, job, context)
                clip_obj.__dict__.update(result)
                clip_obj.status = "done"
                job.clips[i] = asdict(clip_obj)
        
        job.status = "done"
        job.stage = f"Done - {len(job.clips)} clips ready"
        job.progress = 1.0
        await update_job_message(job, context)
        
        # Send clips to Telegram
        await send_clips_to_telegram(job, context, clips_dir)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        job.status = "error"
        job.error = str(e)
        job.stage = "Failed"
        await update_job_message(job, context, f"❌ <b>Job Failed</b>\n\nError: {str(e)[:500]}")

# ============================================================
# Telegram-specific download functions
# ============================================================

async def download_telegram_file(file_id: str, out_path: Path, job: Job, context: ContextTypes.DEFAULT_TYPE) -> Path:
    """Download file from Telegram with progress"""
    file = await context.bot.get_file(file_id)
    file_size = file.file_size
    
    downloaded = 0
    async def progress_callback(current, total):
        nonlocal downloaded
        downloaded = current
        pct = current / total
        update_progress(job, 0.05 + 0.1 * pct, f"Downloading... {pct*100:.0f}%", current)
        await update_job_message(job, context)
    
    await file.download_to_drive(custom_path=str(out_path), progress=progress_callback)
    return out_path

async def download_video_section_async(url: str, out_dir: Path, stem: str, start: float, end: float, height: int, job: Job, context: ContextTypes.DEFAULT_TYPE) -> Path:
    import yt_dlp
    from yt_dlp.utils import download_range_func
    
    def hook(d):
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if total:
                pct = d.get("downloaded_bytes", 0) / total
                update_progress(job, 0.05 + 0.1 * pct, f"Downloading video... {pct*100:.0f}%", d.get("downloaded_bytes", 0))
                asyncio.create_task(update_job_message(job, context))
    
    ydl_opts = {
        "format": f"bv*[height<={height}][vcodec^=avc1]+ba[ext=m4a]/bv*[height<={height}]+ba/b[height<={height}]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / f"{stem}.%(ext)s"),
        "download_ranges": download_range_func(None, [(max(0.0, start), end)]),
        "noplaylist": True, "quiet": True, "no_warnings": True,
        "socket_timeout": 20,
        "progress_hooks": [hook],
    }
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: yt_dlp.YoutubeDL(ydl_opts).download([url]))
    
    files = sorted(out_dir.glob(f"{stem}.*"), key=lambda p: p.stat().st_size, reverse=True)
    if not files:
        raise RuntimeError("Download finished but no file produced.")
    return files[0]

async def download_audio_async(url: str, out_dir: Path, job: Job, context: ContextTypes.DEFAULT_TYPE) -> Path:
    import yt_dlp
    
    def hook(d):
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if total:
                pct = d.get("downloaded_bytes", 0) / total
                update_progress(job, 0.15 + 0.1 * pct, f"Downloading audio... {pct*100:.0f}%", d.get("downloaded_bytes", 0))
                asyncio.create_task(update_job_message(job, context))
    
    ydl_opts = {
        "format": "ba[ext=m4a]/ba/worst",
        "outtmpl": str(out_dir / "audio_src.%(ext)s"),
        "progress_hooks": [hook],
        "noplaylist": True, "quiet": True, "no_warnings": True,
        "socket_timeout": 20,
    }
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: yt_dlp.YoutubeDL(ydl_opts).download([url]))
    
    files = sorted(out_dir.glob("audio_src.*"), key=lambda p: p.stat().st_size, reverse=True)
    if not files:
        raise RuntimeError("Audio download failed.")
    return files[0]

async def extract_audio_async(src: Path, wav: Path, job: Job, context: ContextTypes.DEFAULT_TYPE) -> Path:
    update_progress(job, 0.15, "Extracting audio...")
    await update_job_message(job, context)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: extract_audio(src, wav))
    return wav

# ============================================================
# Send results to Telegram
# ============================================================

async def send_clips_to_telegram(job: Job, context: ContextTypes.DEFAULT_TYPE, clips_dir: Path):
    """Send all clips as video files and documents"""
    
    # Send summary first
    summary = f"✅ <b>Job Complete!</b> `{job.id[:12]}`\n\n"
    summary += f"📹 <b>{job.title}</b>\n"
    summary += f"✂️ <b>{len(job.clips)} clips generated</b>\n\n"
    
    for clip in job.clips:
        summary += f"🔸 <b>{clip['title']}</b> ({clip['end']-clip['start']:.1f}s)\n"
        summary += f"   🎯 {clip['hook']} | Score: {clip['score']}/100\n"
        if clip.get('hashtags'):
            summary += f"   #{' #'.join(clip['hashtags'][:5])}\n"
    
    await context.bot.send_message(
        chat_id=job.chat_id,
        text=summary,
        parse_mode=ParseMode.HTML
    )
    
    # Send each clip
    for clip in job.clips:
        if not clip.get("file"):
            continue
        
        video_path = clips_dir / clip["file"]
        cover_path = clips_dir / clip.get("cover", "")
        
        if not video_path.exists():
            continue
        
        # Prepare caption
        caption = f"🎬 <b>{clip['title']}</b>\n"
        caption += f"⏱️ {clip['start']:.1f}s - {clip['end']:.1f}s ({clip['end']-clip['start']:.1f}s)\n"
        caption += f"🎯 {clip['hook']} | Score: {clip['score']}/100\n"
        if clip.get('description'):
            caption += f"\n{clip['description']}\n"
        if clip.get('hashtags'):
            caption += f"\n#{' #'.join(clip['hashtags'])}"
        
        try:
            # Send as video (plays in Telegram)
            with open(video_path, "rb") as f:
                await context.bot.send_video(
                    chat_id=job.chat_id,
                    video=f,
                    caption=caption,
                    parse_mode=ParseMode.HTML,
                    supports_streaming=True,
                    thumbnail=open(cover_path, "rb") if cover_path.exists() else None
                )
            
            # Also send as document (downloadable file)
            with open(video_path, "rb") as f:
                await context.bot.send_document(
                    chat_id=job.chat_id,
                    document=f,
                    filename=clip["file"],
                    caption=f"📥 Download: {clip['title']}"
                )
                
        except Exception as e:
            logger.error(f"Failed to send clip {clip['id']}: {e}")
            await context.bot.send_message(
                chat_id=job.chat_id,
                text=f"❌ Failed to send clip: {clip['title']}\nError: {str(e)[:200]}"
            )
    
    # Send metadata JSON
    meta = {
        "job_id": job.id,
        "title": job.title,
        "source": job.source,
        "clips": [
            {k: v for k, v in c.items() if k not in ("src", "fixed_words", "fixed_for")}
            for c in job.clips
        ]
    }
    meta_path = clips_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    
    with open(meta_path, "rb") as f:
        await context.bot.send_document(
            chat_id=job.chat_id,
            document=f,
            filename=f"peakcut_{job.id}_metadata.json",
            caption="📋 Job metadata (JSON)"
        )

# ============================================================
# Message Handler for URL/File input
# ============================================================

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    state = user_states.get(user_id, {})
    step = state.get("step")
    
    if step == "waiting_url" and update.message.text:
        url = update.message.text.strip()
        # Validate URL
        from urllib.parse import urlparse
        try:
            u = urlparse(url)
            host = (u.hostname or "").lower()
            allowed = ("youtube.com", "youtu.be", "twitch.tv", "kick.com", "clips.twitch.tv", "vod.twitch.tv")
            if not (u.scheme in ("http", "https") and any(host == s or host.endswith("." + s) for s in allowed)):
                await update.message.reply_text("❌ Invalid URL. Use YouTube, Twitch, or Kick.")
                return
        except:
            await update.message.reply_text("❌ Invalid URL format.")
            return
        
        # Get video info
        msg = await update.message.reply_text("🔍 Fetching video info...")
        try:
            import yt_dlp
            ydl_opts = {"noplaylist": True, "quiet": True, "no_warnings": True, "extract_flat": "in_playlist", "playlistend": 1}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
            
            if info.get("_type") != "video":
                await msg.edit_text("❌ That's a channel/playlist, not a video.")
                return
            if info.get("is_live"):
                await msg.edit_text("❌ Live streams not supported yet.")
                return
            
            source = VideoSource(
                type="url", url=url, name=info.get("title") or "",
                duration=float(info.get("duration") or 0),
                title=info.get("title") or "",
                uploader=info.get("uploader") or info.get("channel") or "",
                thumbnail=info.get("thumbnail") or ""
            )
            
            state["source"] = source
            state["step"] = "main"
            user_states[user_id] = state
            
            await msg.edit_text(
                f"✅ <b>Video Loaded</b>\n\n"
                f"📹 <b>{source.title}</b>\n"
                f"👤 {source.uploader}\n"
                f"⏱️ {source.duration:.0f}s ({source.duration/60:.1f} min)\n\n"
                f"Configure AI & options, then click Start Processing.",
                parse_mode=ParseMode.HTML,
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🤖 AI Settings", callback_data="ai_settings"),
                     InlineKeyboardButton("⚙️ Clip Options", callback_data="clip_options")],
                    [InlineKeyboardButton("✂️ Start Processing", callback_data="start_process")]
                ])
            )
        except Exception as e:
            await msg.edit_text(f"❌ Failed to fetch video: {str(e)[:200]}")
    
    elif step == "waiting_file" and update.message.video:
        video = update.message.video
        if video.duration > 8 * 3600:
            await update.message.reply_text("❌ Video too long (max 8 hours).")
            return
        if video.file_size > MAX_FILE_SIZE:
            await update.message.reply_text("❌ File too large (max 2GB).")
            return
        
        source = VideoSource(
            type="file", file_id=video.file_id, name=video.file_name or "video.mp4",
            duration=video.duration, title=video.file_name or "video.mp4"
        )
        state["source"] = source
        state["step"] = "main"
        user_states[user_id] = state
        
        await update.message.reply_text(
            f"✅ <b>Video Received</b>\n\n"
            f"📹 <b>{source.name}</b>\n"
            f"⏱️ {source.duration:.0f}s ({source.duration/60:.1f} min)\n"
            f"📦 {video.file_size / (1024*1024):.1f} MB\n\n"
            f"Configure AI & options, then click Start Processing.",
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🤖 AI Settings", callback_data="ai_settings"),
                 InlineKeyboardButton("⚙️ Clip Options", callback_data="clip_options")],
                [InlineKeyboardButton("✂️ Start Processing", callback_data="start_process")]
            ])
        )
    
    elif step == "waiting_model" and update.message.text:
        model = update.message.text.strip()
        if user_id not in user_configs:
            user_configs[user_id] = {}
        user_configs[user_id]["models"] = model
        state["step"] = "main"
        user_states[user_id] = state
        await update.message.reply_text(
            f"✅ Model set to: <code>{model}</code>",
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back to AI Settings", callback_data="ai_settings")]])
        )
    
    elif step == "waiting_api_key" and update.message.text:
        api_key = update.message.text.strip()
        if user_id not in user_configs:
            user_configs[user_id] = {}
        user_configs[user_id]["api_key"] = api_key
        state["step"] = "main"
        user_states[user_id] = state
        # Delete the message with API key for security
        await update.message.delete()
        await update.message.reply_text(
            "✅ API Key saved (message deleted for security)",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back to AI Settings", callback_data="ai_settings")]])
        )
    
    elif step == "waiting_max_clips" and update.message.text:
        try:
            val = int(update.message.text.strip())
            if 1 <= val <= 20:
                opts = state.get("options", ClipOptions())
                opts.max_clips = val
                state["options"] = opts
                state["step"] = "main"
                user_states[user_id] = state
                await update.message.reply_text(f"✅ Max clips set to {val}")
            else:
                await update.message.reply_text("❌ Enter a number between 1 and 20.")
        except ValueError:
            await update.message.reply_text("❌ Invalid number.")
    
    elif step == "waiting_length" and update.message.text:
        parts = update.message.text.strip().split()
        if len(parts) == 2:
            try:
                min_len = int(parts[0])
                max_len = int(parts[1])
                if 5 <= min_len < max_len <= 180:
                    opts = state.get("options", ClipOptions())
                    opts.min_len = min_len
                    opts.max_len = max_len
                    state["options"] = opts
                    state["step"] = "main"
                    user_states[user_id] = state
                    await update.message.reply_text(f"✅ Length range set to {min_len}-{max_len}s")
                else:
                    await update.message.reply_text("❌ Invalid range. Min 5s, max 180s, min < max.")
            except ValueError:
                await update.message.reply_text("❌ Enter two numbers: min max")
        else:
            await update.message.reply_text("❌ Format: <code>min max</code> (e.g. <code>15 60</code>)", parse_mode=ParseMode.HTML)

# ============================================================
# Main
# ============================================================

async def main():
    """Run the bot"""
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Commands
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("config", config_command))
    application.add_handler(CommandHandler("options", options_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("cancel", cancel_command))
    
    # Callbacks
    application.add_handler(CallbackQueryHandler(button_callback))
    
    # Messages
    application.add_handler(MessageHandler(filters.TEXT | filters.VIDEO, handle_message))
    
    # Start
    logger.info("Starting PeakCut Telegram Bot...")
    await application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    asyncio.run(main())
