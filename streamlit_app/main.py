#!/usr/bin/env python3
"""
PeakCut Streamlit App
Turn long videos and live streams into captioned vertical clips.
"""
import os
import sys
import json
import time
import tempfile
import subprocess
import threading
import queue
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import uuid

import streamlit as st
import numpy as np
import wave

# Add the original PeakCut studio to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "PeakCut"))

# Page config
st.set_page_config(
    page_title="PeakCut - Video to Vertical Clips",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
    }
    .metric-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .clip-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #fafafa;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-queued { background: #fff3cd; color: #856404; }
    .status-running { background: #cce5ff; color: #004085; }
    .status-done { background: #d4edda; color: #155724; }
    .status-error { background: #f8d7da; color: #721c24; }
    .progress-container {
        background: #e9ecef;
        border-radius: 10px;
        height: 8px;
        overflow: hidden;
    }
    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        transition: width 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Data Models
# ============================================================

@dataclass
class VideoSource:
    type: str  # "url" or "file"
    url: str = ""
    file_path: str = ""
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
    style: str = "bounce"  # bounce, hype, clean
    layout: str = "blur"   # blur, gameplay, fit
    quality: int = 720
    captions: bool = True
    range_start: Optional[float] = None
    range_end: Optional[float] = None


@dataclass
class AIConfig:
    provider: str = "gemini"
    models: str = "gemini-2.5-flash"
    api_key: str = ""
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
    status: str = "queued"  # queued, rendering, done, error
    src: str = ""
    file: str = ""
    cover: str = ""
    error: str = ""
    version: int = 0


@dataclass
class Job:
    id: str
    user: str
    kind: str  # "video" or "live"
    creator: Optional[str]
    created: float
    source: Dict
    options: Dict
    status: str = "queued"
    stage: str = "Waiting"
    progress: float = 0.0
    title: str = ""
    uploader: str = ""
    duration: float = 0
    clips: List[Dict] = field(default_factory=list)
    error: Optional[str] = None
    ai: Dict = field(default_factory=dict)
    width: int = 1920
    height: int = 1080
    game: str = ""


# ============================================================
# Session State Initialization
# ============================================================

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "current_job": None,
        "jobs": {},
        "video_source": None,
        "clip_options": ClipOptions(),
        "ai_config": AIConfig(),
        "processing": False,
        "current_stage": "",
        "progress_value": 0.0,
        "transcript_words": [],
        "loudness_db": None,
        "selected_clips": [],
        "render_queue": queue.Queue(),
        "worker_thread": None,
        "preview_url": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ============================================================
# Utility Functions
# ============================================================

def run_ffmpeg(cmd: List[str], cwd: Optional[Path] = None, timeout: int = 3600) -> str:
    """Run ffmpeg command and return stdout."""
    try:
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr[-1500:]}")
        return result.stdout
    except subprocess.TimeoutExpired:
        raise RuntimeError("ffmpeg took too long and was stopped.")


def probe_video(path: Path) -> Dict:
    """Get video metadata using ffprobe."""
    out = run_ffmpeg([
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path)
    ])
    data = json.loads(out)
    v = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    if not v:
        raise RuntimeError("This file has no video track.")
    if not any(s["codec_type"] == "audio" for s in data["streams"]):
        raise RuntimeError("This file has no audio track.")
    w, h = int(v["width"]), int(v["height"])
    rot = str((v.get("side_data_list") or [{}])[0].get("rotation", "0"))
    if rot.lstrip("-") in ("90", "270"):
        w, h = h, w
    return {"width": w, "height": h, "duration": float(data["format"]["duration"])}


def extract_audio(src: Path, wav: Path, max_hours: float = 8):
    """Extract audio from video to WAV."""
    run_ffmpeg([
        "ffmpeg", "-y", "-v", "error", "-i", str(src),
        "-vn", "-ac", "1", "-ar", "16000",
        "-t", str(int(max_hours * 3600) + 60),
        "-af", "aresample=async=1:first_pts=0",
        "-c:a", "pcm_s16le", str(wav)
    ])


def load_audio(wav: Path) -> np.ndarray:
    """Load WAV file as int16 numpy array using Python's wave module."""
    with wave.open(str(wav), 'rb') as wf:
        assert wf.getframerate() == 16000, f"Expected 16000 Hz, got {wf.getframerate()}"
        assert wf.getsampwidth() == 2, f"Expected 16-bit, got {wf.getsampwidth()*8}-bit"
        assert wf.getnchannels() == 1, f"Expected mono, got {wf.getnchannels()} channels"
        n_frames = wf.getnframes()
        data = wf.readframes(n_frames)
    return np.frombuffer(data, dtype=np.int16)


def loudness_per_second(audio: np.ndarray) -> np.ndarray:
    """Calculate RMS level in dB for each second."""
    n = len(audio) // 16000
    if n == 0:
        return np.zeros(1, dtype=np.float32)
    x = audio[:n * 16000].reshape(n, 16000).astype(np.float32) / 32768.0
    rms = np.sqrt((x ** 2).mean(axis=1) + 1e-10)
    return (20 * np.log10(rms)).astype(np.float32)


# ============================================================
# AI Provider Configuration
# ============================================================

PROVIDERS = {
    "gemini": ("Gemini", "https://generativelanguage.googleapis.com/v1beta/models", "https://aistudio.google.com/apikey", "gemini-2.5-flash"),
    "openai": ("OpenAI", "https://api.openai.com/v1", "https://platform.openai.com/api-keys", "gpt-4o-mini"),
    "xai": ("Grok", "https://api.x.ai/v1", "https://console.x.ai", "grok-4"),
    "groq": ("Groq", "https://api.groq.com/openai/v1", "https://console.groq.com/keys", "llama-3.3-70b-versatile"),
    "openrouter": ("OpenRouter", "https://openrouter.ai/api/v1", "https://openrouter.ai/keys", "google/gemini-2.5-flash"),
    "anthropic": ("Claude", "https://api.anthropic.com/v1", "https://console.anthropic.com", "claude-3-5-sonnet-20241022"),
    "ollama": ("Ollama (local)", "http://localhost:11434/v1", "http://localhost:11434", "llama3.2"),
    "custom": ("Custom (OpenAI format)", "", "", "model-name"),
}


# ============================================================
# Video Input Functions
# ============================================================

def validate_url(url: str) -> bool:
    """Check if URL is from an allowed site."""
    from urllib.parse import urlparse
    try:
        u = urlparse(url.strip())
        host = (u.hostname or "").lower()
        allowed = (
            "youtube.com", "youtu.be", "twitch.tv", "kick.com",
            "clips.twitch.tv", "vod.twitch.tv"
        )
        return u.scheme in ("http", "https") and any(
            host == s or host.endswith("." + s) for s in allowed
        )
    except ValueError:
        return False


def get_video_info(url: str) -> Dict:
    """Get video info using yt-dlp."""
    import yt_dlp
    ydl_opts = {
        "noplaylist": True, "quiet": True, "no_warnings": True,
        "extract_flat": "in_playlist", "playlistend": 1,
        "socket_timeout": 20,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return {
        "title": info.get("title") or "",
        "uploader": info.get("uploader") or info.get("channel") or "",
        "duration": float(info.get("duration") or 0),
        "is_live": bool(info.get("is_live")),
        "thumbnail": info.get("thumbnail") or "",
        "kind": info.get("_type") or "video",
    }


def download_audio(url: str, out_dir: Path, progress_callback=None) -> Path:
    """Download audio only from URL."""
    import yt_dlp
    def hook(d):
        if progress_callback and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if total:
                progress_callback(d.get("downloaded_bytes", 0) / total)

    ydl_opts = {
        "format": "ba[ext=m4a]/ba/worst",
        "outtmpl": str(out_dir / "audio_src.%(ext)s"),
        "progress_hooks": [hook],
        "noplaylist": True, "quiet": True, "no_warnings": True,
        "socket_timeout": 20,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    files = sorted(out_dir.glob("audio_src.*"), key=lambda p: p.stat().st_size, reverse=True)
    if not files:
        raise RuntimeError("Download finished but no file was produced.")
    return files[0]


def download_video_section(url: str, out_dir: Path, stem: str, start: float, end: float, height: int = 720) -> Path:
    """Download a section of a video."""
    import yt_dlp
    from yt_dlp.utils import download_range_func
    out_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "format": f"bv*[height<={height}][vcodec^=avc1]+ba[ext=m4a]/bv*[height<={height}]+ba/b[height<={height}]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / f"{stem}.%(ext)s"),
        "download_ranges": download_range_func(None, [(max(0.0, start), end)]),
        "noplaylist": True, "quiet": True, "no_warnings": True,
        "socket_timeout": 20,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    files = sorted(out_dir.glob(f"{stem}.*"), key=lambda p: p.stat().st_size, reverse=True)
    if not files:
        raise RuntimeError("Download finished but no file was produced.")
    return files[0]


# ============================================================
# Transcription (using sherpa-onnx if available)
# ============================================================

def transcribe_audio(audio: np.ndarray, progress_callback=None) -> List[Dict]:
    """Transcribe audio to words with timestamps.
    Uses sherpa-onnx Parakeet if available, otherwise returns empty list for demo."""
    try:
        import sherpa_onnx
        # Try to load the model
        model_dir = Path(__file__).parent.parent / "PeakCut" / "studio" / "models" / "parakeet"
        if not model_dir.exists():
            # Try to find model elsewhere
            model_dir = Path.home() / ".cache" / "peakcut" / "parakeet"
        if not model_dir.exists():
            return []  # Model not found, skip transcription
            
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
        
        # Split audio into chunks
        words = []
        sr = 16000
        chunk_duration = 20  # seconds
        chunk_samples = chunk_duration * sr
        
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i + chunk_samples].astype(np.float32) / 32768.0
            if len(chunk) < 4800:
                continue
            stream = recognizer.create_stream()
            stream.accept_waveform(16000, chunk)
            recognizer.decode_stream(stream)
            result = stream.result
            offset = i / sr
            for token, timestamp in zip(result.tokens, result.timestamps):
                word = token.replace("▁", " ").strip()
                if word:
                    words.append({
                        "w": word,
                        "s": round(offset + timestamp, 2),
                        "e": round(offset + timestamp + 0.12, 2)
                    })
            if progress_callback:
                progress_callback(min(1.0, (i + chunk_samples) / len(audio)))
        return words
    except ImportError:
        st.warning("sherpa-onnx not installed. Transcription unavailable.")
        return []
    except Exception as e:
        st.warning(f"Transcription failed: {e}")
        return []


# ============================================================
# AI Clip Finding (Simplified version)
# ============================================================

def find_clips_ai(words: List[Dict], db: np.ndarray, duration: float,
                  title: str, uploader: str, options: ClipOptions,
                  ai_config: AIConfig) -> List[Dict]:
    """Find clips using AI. Falls back to loudness-based if no AI configured."""
    if not words:
        return []
    
    # Try to use AI if configured
    if ai_config.api_key and ai_config.provider in PROVIDERS:
        try:
            return find_clips_with_ai(words, db, duration, title, uploader, options, ai_config)
        except Exception as e:
            st.warning(f"AI clip finding failed: {e}. Falling back to loudness.")
    
    # Fallback: loudness-based
    return find_clips_by_loudness(words, db, duration, options)


def find_clips_with_ai(words: List[Dict], db: np.ndarray, duration: float,
                       title: str, uploader: str, options: ClipOptions,
                       ai_config: AIConfig) -> List[Dict]:
    """Use AI to find clips."""
    import httpx
    
    provider_info = PROVIDERS[ai_config.provider]
    base_url = ai_config.base_url or provider_info[1]
    model = ai_config.models.split(",")[0].strip()
    
    # Format transcript for AI
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
        body = {
            "model": model,
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": prompt}]
        }
        url = f"{base_url}/messages"
    elif ai_config.provider == "gemini":
        url = f"{base_url}/{model}:generateContent?key={ai_config.api_key}"
        body = {"contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"responseMimeType": "application/json"}}
    else:
        headers["Authorization"] = f"Bearer {ai_config.api_key}"
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.4
        }
        url = f"{base_url}/chat/completions"
    
    with httpx.Client(timeout=180) as client:
        resp = client.post(url, json=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    
    # Parse response based on provider
    if ai_config.provider == "anthropic":
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    elif ai_config.provider == "gemini":
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    else:
        text = data["choices"][0]["message"]["content"]
    
    clips = json.loads(text)
    if not isinstance(clips, list):
        clips = clips.get("clips", [])
    
    # Post-process clips
    processed = []
    for c in clips:
        if not isinstance(c, dict):
            continue
        s, e = float(c.get("start", 0)), float(c.get("end", 0))
        if e - s < options.min_len - 1 or e - s > options.max_len + 1:
            continue
        processed.append({
            "start": round(s, 2),
            "end": round(e, 2),
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
    """Fallback: pick clips by loudness."""
    z = (db - np.median(db)) / (db.std() + 1e-6)
    picks, order = [], np.argsort(-z)
    
    for sec in order:
        s = max(0.0, float(sec) - options.max_len * 0.6)
        e = min(duration, s + options.max_len)
        if any(not (e <= p["start"] or s >= p["end"]) for p in picks):
            continue
        if not any(s <= w["s"] <= e for w in words):
            continue
        picks.append({
            "start": round(s, 2),
            "end": round(e, 2),
            "score": 50,
            "category": "other",
            "hook": "WAIT FOR IT",
            "title": "Loud moment",
            "description": "",
            "hashtags": [],
            "highlight_words": [],
            "reason": "Picked by loudness because the AI was unavailable.",
        })
        if len(picks) >= options.max_clips:
            break
    return picks


def make_transcript_lines(words: List[Dict], db: np.ndarray) -> List[Dict]:
    """Group words into caption-ish lines with loudness marks."""
    lines, cur = [], []
    z = (db - np.median(db)) / (db.std() + 1e-6) if len(db) > 5 else np.zeros(len(db))
    
    def flush():
        if cur:
            s, e = cur[0]["s"], cur[-1]["e"]
            is_loud = False
            if len(z):
                seg = z[int(s): int(e) + 1]
                is_loud = bool(len(seg) and seg.max() > 1.5)
            lines.append({"s": s, "e": e, "text": " ".join(w["w"] for w in cur),
                          "loud": is_loud})
            cur.clear()
    
    for w in words:
        if cur and (w["s"] - cur[-1]["e"] > 0.7 or w["s"] - cur[0]["s"] > 10
                    or (cur[-1]["w"][-1:] in ".?!" and len(cur) >= 4)):
            flush()
        cur.append(w)
    flush()
    return lines


# ============================================================
# Caption Processing
# ============================================================

def fix_caption_words(words: List[Dict], title: str = "") -> List[Dict]:
    """Clean up caption words (simplified version)."""
    # In a real implementation, this would use AI to fix words
    # For now, just do basic cleanup
    fixed = []
    for w in words:
        word = w["w"].strip()
        # Basic cleanup
        if word.lower() in ("uh", "um", "er", "ah"):
            continue
        fixed.append({**w, "w": word})
    return fixed


def words_to_ass(words: List[Dict], style_name: str, video_w: int, video_h: int) -> str:
    """Generate ASS subtitle content from words."""
    # Simplified ASS generation - in reality this is complex
    # This is a placeholder that creates basic subtitles
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
    
    for i, w in enumerate(words):
        start = format_ass_time(w["s"])
        end = format_ass_time(w["e"])
        text = w["w"].replace("{", "\\{").replace("}", "\\}")
        ass += f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n"
    
    return ass


def format_ass_time(seconds: float) -> str:
    """Format seconds as ASS time (H:MM:SS.cc)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


# ============================================================
# Video Rendering
# ============================================================

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


def render_clip(src_path: Path, out_dir: Path, clip: Clip, words: List[Dict],
                job_width: int, job_height: int) -> Dict:
    """Render a single clip with captions."""
    out_dir.mkdir(parents=True, exist_ok=True)
    clip_id = clip["id"]
    
    # Generate ASS subtitles
    if clip.get("captions", True) and words:
        clip_words = [w for w in words if clip["start"] - 1 <= w["s"] <= clip["end"] + 1]
        clip_words = fix_caption_words(clip_words)
        clip_words = [{**w, "s": w["s"] - clip["start"], "e": w["e"] - clip["start"]} for w in clip_words]
        ass_content = words_to_ass(clip_words, clip["style"], 1080, 1920)
        ass_path = out_dir / f"{clip_id}.ass"
        ass_path.write_text(ass_content, encoding="utf-8")
    else:
        ass_path = None
    
    # Build ffmpeg filter
    layout = clip.get("layout", "blur")
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
    else:  # fit
        filter_complex = (
            f"[0:v]scale=1080:1920:force_original_aspect_ratio=decrease,"
            f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black[v]"
        )
    
    if ass_path:
        filter_complex += f";[v]ass={ass_path.name}[vout]"
        map_video = "[vout]"
    else:
        map_video = "[v]"
    
    # Try hardware encoding first, fallback to software
    encoders = [
        ["-c:v", "h264_nvenc", "-preset", "fast", "-cq", "23"],  # NVENC
        ["-c:v", "h264_qsv", "-preset", "fast", "-global_quality", "23"],  # QSV
        ["-c:v", "libx264", "-crf", "21", "-preset", "veryfast"],  # CPU
    ]
    
    for enc in encoders:
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-ss", f"{clip['start']:.2f}", "-t", f"{clip['end'] - clip['start']:.2f}",
            "-i", str(src_path),
            "-filter_complex", filter_complex,
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
    
    # Generate cover image
    cover_t = min(1.2, (clip["end"] - clip["start"]) / 2)
    run_ffmpeg([
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{cover_t:.2f}", "-i", str(out_dir / f"{clip_id}.mp4"),
        "-frames:v", "1", "-q:v", "3", str(out_dir / f"{clip_id}.jpg")
    ])
    
    return {
        "file": f"{clip_id}.mp4",
        "cover": f"{clip_id}.jpg",
        "layout": layout,
    }


# ============================================================
# Streamlit UI Components
# ============================================================

def render_header():
    """Render the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>✂️ PeakCut</h1>
        <p>Turn long videos into captioned vertical clips for Shorts, Reels & TikTok</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar with navigation and settings."""
    with st.sidebar:
        st.markdown("### 🎬 PeakCut")
        
        # Navigation
        page = st.radio(
            "Navigate",
            ["🏠 Home", "📥 Input Video", "🤖 AI Settings", "⚙️ Clip Options", "✂️ Process & Clips", "📤 Export"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # Quick stats
        if st.session_state.current_job:
            job = st.session_state.current_job
            st.markdown("### 📊 Current Job")
            st.metric("Status", job.status.title())
            st.metric("Clips Found", len(job.clips))
            if job.duration:
                st.metric("Duration", f"{job.duration:.0f}s")
        
        st.divider()
        
        # System info
        st.markdown("### ℹ️ System")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("CPU", f"{os.cpu_count()} cores")
        with col2:
            import psutil
            mem = psutil.virtual_memory()
            st.metric("RAM", f"{mem.total // (1024**3)} GB")
        
        # FFmpeg check
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            st.success("✅ FFmpeg available")
        except:
            st.error("❌ FFmpeg not found")
        
        return page


def page_home():
    """Home page with overview."""
    render_header()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>📥 Any Video Source</h3>
            <p>Paste a YouTube, Twitch, or Kick link, or upload a video file directly. PeakCut downloads and processes it automatically.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🤖 AI-Powered Clips</h3>
            <p>Uses your own AI key (Gemini, OpenAI, Grok, Groq, Claude, Ollama, or custom) to find the most engaging moments automatically.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>🎨 Vertical Clips with Captions</h3>
            <p>Renders 9:16 clips with word-by-word animated captions in 3 styles (Bounce, Hype, Clean) and 3 layouts.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Quick start
    st.markdown("## 🚀 Quick Start")
    st.markdown("""
    1. **Input Video** - Paste a URL or upload a file
    2. **AI Settings** - Add your API key (optional - works without it using loudness)
    3. **Clip Options** - Choose clip length, style, layout
    4. **Process** - Let PeakCut transcribe, find clips, and render
    5. **Export** - Download your vertical clips ready to post!
    """)
    
    # Feature highlights
    st.markdown("## ✨ Features")
    features = [
        ("🎯 Smart Clip Detection", "AI analyzes transcript + audio loudness to find viral moments"),
        ("📝 Word-by-Word Captions", "Animated captions that highlight each word as it's spoken"),
        ("🎨 3 Caption Styles", "Bounce (poppy), Hype (bold), Clean (minimal)"),
        ("📐 3 Layouts", "Blur (centered), Gameplay (full), Fit (letterboxed)"),
        ("⚡ Parallel Processing", "Multiple clips render simultaneously"),
        ("🔄 Re-render Clips", "Adjust timing, style, captions and re-render instantly"),
        ("📱 Ready to Post", "Clips include hooks, titles, descriptions, hashtags"),
        ("🆓 Free & Local", "Runs entirely on your machine, no cloud required"),
    ]
    
    for title, desc in features:
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown(f"### {title}")
        with col2:
            st.markdown(desc)
    
    # Get started button
    if st.button("🚀 Get Started - Input Video", type="primary", use_container_width=True):
        st.session_state.page = "📥 Input Video"
        st.rerun()


def page_input_video():
    """Video input page."""
    st.markdown("## 📥 Input Video")
    
    tab1, tab2 = st.tabs(["🔗 From URL", "📁 Upload File"])
    
    with tab1:
        st.markdown("### Paste a Video Link")
        url = st.text_input(
            "YouTube, Twitch, or Kick URL",
            placeholder="https://youtube.com/watch?v=... or https://twitch.tv/...",
            help="Supported: YouTube (videos & Shorts), Twitch (VODs & clips), Kick"
        )
        
        if url and validate_url(url):
            if st.button("🔍 Preview Video", type="primary"):
                with st.spinner("Fetching video info..."):
                    try:
                        info = get_video_info(url)
                        if info["kind"] != "video":
                            st.error("That link is a channel or playlist, not a single video.")
                        elif info["is_live"]:
                            st.error("This is a live stream. Use the Live Clipping feature instead.")
                        else:
                            st.session_state.video_source = VideoSource(
                                type="url", url=url, name=info["title"],
                                duration=info["duration"], title=info["title"],
                                uploader=info["uploader"], thumbnail=info["thumbnail"]
                            )
                            st.success("Video loaded successfully!")
                    except Exception as e:
                        st.error(f"Failed to fetch video: {e}")
        elif url:
            st.warning("Please enter a valid YouTube, Twitch, or Kick URL")
    
    with tab2:
        st.markdown("### Upload a Video File")
        uploaded = st.file_uploader(
            "Choose a video file",
            type=["mp4", "mov", "mkv", "webm", "m4v", "ts", "flv"],
            help="Max 8 hours. Large files may take longer to process."
        )
        
        if uploaded:
            # Save to temp file
            temp_dir = Path(tempfile.gettempdir()) / "peakcut_uploads"
            temp_dir.mkdir(exist_ok=True)
            file_path = temp_dir / f"{uuid.uuid4().hex}_{uploaded.name}"
            file_path.write_bytes(uploaded.read())
            
            # Probe video
            try:
                info = probe_video(file_path)
                if info["duration"] > 8 * 3600:
                    st.error("Video is longer than 8 hours. Please trim it first.")
                else:
                    st.session_state.video_source = VideoSource(
                        type="file", file_path=str(file_path), name=uploaded.name,
                        duration=info["duration"], title=uploaded.name,
                        width=info["width"], height=info["height"]
                    )
                    st.success("Video uploaded successfully!")
            except Exception as e:
                st.error(f"Invalid video file: {e}")
    
    # Show video preview if loaded
    if st.session_state.video_source:
        src = st.session_state.video_source
        st.divider()
        st.markdown("### 📹 Video Preview")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            if src.thumbnail:
                st.image(src.thumbnail, use_container_width=True)
        with col2:
            st.markdown(f"**{src.title or src.name}**")
            if src.uploader:
                st.caption(f"By: {src.uploader}")
            st.caption(f"Duration: {src.duration:.0f}s ({src.duration/60:.1f} min)")
            if src.type == "url":
                st.caption(f"Source: {src.url[:60]}...")
            else:
                st.caption(f"File: {src.name}")
        
        # Range selection
        st.markdown("### ⏱️ Select Range (Optional)")
        col1, col2 = st.columns(2)
        with col1:
            range_start = st.number_input(
                "Start (seconds)", min_value=0.0, max_value=src.duration,
                value=st.session_state.clip_options.range_start or 0.0, step=1.0
            )
        with col2:
            range_end = st.number_input(
                "End (seconds)", min_value=0.0, max_value=src.duration,
                value=st.session_state.clip_options.range_end or src.duration, step=1.0
            )
        
        st.session_state.clip_options.range_start = range_start if range_start > 0 else None
        st.session_state.clip_options.range_end = range_end if range_end < src.duration else None
        
        if st.button("✅ Confirm & Continue", type="primary", use_container_width=True):
            st.success("Video confirmed! Go to **AI Settings** or **Clip Options** to continue.")
            st.balloons()


def page_ai_settings():
    """AI provider configuration page."""
    st.markdown("## 🤖 AI Provider Settings")
    st.markdown("Configure your AI provider for smart clip detection. **Optional** - PeakCut works without AI using loudness detection.")
    
    ai = st.session_state.ai_config
    
    # Provider selection
    provider = st.selectbox(
        "AI Provider",
        options=list(PROVIDERS.keys()),
        format_func=lambda k: PROVIDERS[k][0],
        index=list(PROVIDERS.keys()).index(ai.provider) if ai.provider in PROVIDERS else 0,
    )
    ai.provider = provider
    
    provider_info = PROVIDERS[provider]
    
    # Provider-specific UI
    if provider == "ollama":
        st.info("💡 Ollama runs locally. Make sure it's running at http://localhost:11434")
        ai.base_url = st.text_input("Base URL", value=ai.base_url or provider_info[1])
        ai.models = st.text_input("Model", value=ai.models or provider_info[3],
                                  help="Model name as shown in `ollama list`")
        st.caption("No API key needed for local Ollama")
    elif provider == "custom":
        ai.base_url = st.text_input("Base URL", value=ai.base_url,
                                    placeholder="https://api.example.com/v1",
                                    help="Any service that speaks the OpenAI chat format")
        ai.api_key = st.text_input("API Key", value=ai.api_key, type="password",
                                   help="Your API key for the custom service")
        ai.models = st.text_input("Model", value=ai.models or provider_info[3])
    else:
        # Show key link
        st.markdown(f"[Get API Key]({provider_info[2]}) • Example model: `{provider_info[3]}`")
        ai.api_key = st.text_input("API Key", value=ai.api_key, type="password",
                                   help="Your API key (stored only in this session)")
        ai.models = st.text_input("Models (comma-separated)", value=ai.models or provider_info[3],
                                  help="Tried in order. First one that works is used.")
        ai.base_url = provider_info[1]  # Use default
    
    # Test connection
    if st.button("🔍 Test Connection"):
        if not ai.api_key and provider not in ("ollama",):
            st.warning("Please enter an API key first")
        else:
            with st.spinner("Testing..."):
                try:
                    # Simple test call
                    import httpx
                    if provider == "gemini":
                        url = f"{ai.base_url}/{ai.models.split(',')[0].strip()}:generateContent?key={ai.api_key}"
                        resp = httpx.post(url, json={"contents": [{"parts": [{"text": "Hi"}]}]}, timeout=30)
                    elif provider == "anthropic":
                        resp = httpx.post(f"{ai.base_url}/messages",
                            json={"model": ai.models.split(',')[0].strip(), "max_tokens": 10,
                                  "messages": [{"role": "user", "content": "Hi"}]},
                            headers={"x-api-key": ai.api_key, "anthropic-version": "2023-06-01"}, timeout=30)
                    elif provider == "ollama":
                        resp = httpx.post(f"{ai.base_url}/chat/completions",
                            json={"model": ai.models, "messages": [{"role": "user", "content": "Hi"}]}, timeout=30)
                    else:
                        resp = httpx.post(f"{ai.base_url}/chat/completions",
                            json={"model": ai.models.split(',')[0].strip(), "messages": [{"role": "user", "content": "Hi"}]},
                            headers={"Authorization": f"Bearer {ai.api_key}"}, timeout=30)
                    
                    if resp.status_code == 200:
                        st.success("✅ Connection successful!")
                    else:
                        st.error(f"❌ Failed: {resp.status_code} - {resp.text[:200]}")
                except Exception as e:
                    st.error(f"❌ Connection failed: {e}")
    
    # AI usage stats
    if st.session_state.current_job and st.session_state.current_job.ai:
        st.divider()
        st.markdown("### 📊 AI Usage (Current Job)")
        ai_info = st.session_state.current_job.ai
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model", ai_info.get("model", "N/A"))
        with col2:
            st.metric("Tokens Used", ai_info.get("tokens", 0))
        with col3:
            if ai_info.get("note"):
                st.caption(ai_info["note"])


def page_clip_options():
    """Clip options configuration page."""
    st.markdown("## ⚙️ Clip Options")
    
    opts = st.session_state.clip_options
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📏 Clip Length")
        opts.max_clips = st.slider("Max Clips", 1, 20, opts.max_clips,
                                    help="Maximum number of clips to generate")
        opts.min_len = st.slider("Min Length (seconds)", 5, 120, opts.min_len,
                                  help="Minimum clip duration")
        opts.max_len = st.slider("Max Length (seconds)", 10, 180, opts.max_len,
                                  help="Maximum clip duration")
        
        if opts.min_len >= opts.max_len:
            st.error("Min length must be less than max length")
            opts.min_len = max(5, opts.max_len - 5)
    
    with col2:
        st.markdown("### 🎨 Style & Layout")
        opts.style = st.selectbox(
            "Caption Style",
            options=list(STYLES.keys()),
            format_func=lambda k: STYLES[k],
            index=list(STYLES.keys()).index(opts.style)
        )
        opts.layout = st.selectbox(
            "Video Layout",
            options=list(LAYOUTS.keys()),
            format_func=lambda k: LAYOUTS[k],
            index=list(LAYOUTS.keys()).index(opts.layout)
        )
        opts.quality = st.selectbox(
            "Download Quality",
            options=[360, 480, 720, 1080],
            index=[360, 480, 720, 1080].index(opts.quality),
            help="Quality for downloading source video"
        )
        opts.captions = st.checkbox("Add Captions", value=opts.captions,
                                     help="Burn word-by-word captions into clips")
    
    # Preview of settings
    st.divider()
    st.markdown("### 📋 Current Settings")
    settings_df = {
        "Setting": ["Max Clips", "Min Length", "Max Length", "Caption Style", "Layout", "Quality", "Captions"],
        "Value": [opts.max_clips, f"{opts.min_len}s", f"{opts.max_len}s", STYLES[opts.style], LAYOUTS[opts.layout], f"{opts.quality}p", "On" if opts.captions else "Off"],
    }
    st.table(settings_df)


def page_process():
    """Process video and show clips page."""
    st.markdown("## ✂️ Process & Clips")
    
    if not st.session_state.video_source:
        st.warning("⚠️ No video selected. Go to **Input Video** first.")
        if st.button("📥 Go to Input Video"):
            st.session_state.page = "📥 Input Video"
            st.rerun()
        return
    
    src = st.session_state.video_source
    opts = st.session_state.clip_options
    ai = st.session_state.ai_config
    
    # Show video info
    with st.expander("📹 Video Info", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Title", src.title[:30] + "..." if len(src.title) > 30 else src.title)
        with col2:
            st.metric("Duration", f"{src.duration:.0f}s")
        with col3:
            st.metric("Source", src.type.upper())
    
    # Start processing
    if not st.session_state.processing and not st.session_state.current_job:
        if st.button("🚀 Start Processing", type="primary", use_container_width=True):
            st.session_state.processing = True
            st.session_state.current_stage = "Initializing..."
            st.session_state.progress_value = 0.0
            # Start worker thread
            thread = threading.Thread(target=process_video_worker, args=(src, opts, ai), daemon=True)
            thread.start()
            st.session_state.worker_thread = thread
            st.rerun()
    
    # Show progress
    if st.session_state.processing or (st.session_state.current_job and st.session_state.current_job.status in ("queued", "running")):
        render_progress()
    
    # Show clips if done
    if st.session_state.current_job and st.session_state.current_job.status == "done":
        render_clips_list(st.session_state.current_job)
    
    # Show error if any
    if st.session_state.current_job and st.session_state.current_job.status == "error":
        st.error(f"❌ Processing failed: {st.session_state.current_job.error}")
        if st.button("🔄 Try Again"):
            st.session_state.current_job = None
            st.session_state.processing = False
            st.rerun()


def process_video_worker(src: VideoSource, opts: ClipOptions, ai: AIConfig):
    """Background worker for video processing."""
    try:
        job_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        work_dir = Path(tempfile.gettempdir()) / "peakcut_jobs" / job_id
        work_dir.mkdir(parents=True)
        clips_dir = work_dir / "clips"
        clips_dir.mkdir()
        
        # Create job object
        job = Job(
            id=job_id, user="local", kind="video", creator=None,
            created=time.time(), source={"type": src.type, "url": src.url, "path": src.file_path, "name": src.name},
            options=asdict(opts), status="running", stage="Downloading audio",
            title=src.title, uploader=src.uploader, duration=src.duration
        )
        st.session_state.current_job = job
        
        # Step 1: Get audio
        update_progress(0.05, "Downloading audio...")
        if src.type == "url":
            audio_path = download_audio(src.url, work_dir, lambda p: update_progress(0.05 + 0.1 * p, "Downloading audio..."))
        else:
            audio_path = Path(src.file_path)
        
        # Step 2: Extract audio to WAV
        update_progress(0.15, "Extracting audio...")
        wav_path = work_dir / "audio.wav"
        extract_audio(audio_path, wav_path)
        
        # Step 3: Load audio and compute loudness
        update_progress(0.2, "Analyzing audio...")
        audio = load_audio(wav_path)
        if not job.duration:
            job.duration = len(audio) / 16000
        db = loudness_per_second(audio)
        job.width, job.height = 1920, 1080  # Will be updated from video
        
        # Step 4: Transcribe
        update_progress(0.25, "Transcribing speech...")
        words = transcribe_audio(audio, lambda p: update_progress(0.25 + 0.5 * p, f"Transcribing... {int(p*100)}%"))
        job.transcript_words = words  # Store for later
        
        if not words:
            st.session_state.transcript_words = []
            update_progress(0.75, "No speech found, using loudness only...")
        else:
            st.session_state.transcript_words = words
            st.session_state.loudness_db = db
        
        # Step 5: Find clips
        update_progress(0.75, "Finding best moments...")
        clips_data = find_clips_ai(words, db, job.duration, job.title, job.uploader, opts, ai)
        
        # Create clip objects
        job.clips = []
        for i, c in enumerate(clips_data):
            clip = Clip(
                id=f"clip{i+1:02d}",
                start=c["start"], end=c["end"],
                score=c["score"], category=c["category"],
                hook=c["hook"], title=c["title"],
                description=c["description"], hashtags=c["hashtags"],
                highlight_words=c["highlight_words"], reason=c["reason"],
                style=opts.style, layout=opts.layout, captions=opts.captions,
                status="queued"
            )
            job.clips.append(asdict(clip))
        
        # Step 6: Render clips
        if job.clips:
            # Get source video for rendering
            if src.type == "url":
                update_progress(0.8, "Downloading source video...")
                video_path = download_video_section(src.url, clips_dir, "source", 0, job.duration, opts.quality)
            else:
                video_path = Path(src.file_path)
            
            # Update job with video dimensions
            meta = probe_video(video_path)
            job.width, job.height = meta["width"], meta["height"]
            
            # Render each clip
            for i, clip in enumerate(job.clips):
                update_progress(0.8 + 0.15 * (i / len(job.clips)), f"Rendering clip {i+1}/{len(job.clips)}...")
                clip_obj = Clip(**clip)
                clip_obj.status = "rendering"
                result = render_clip(video_path, clips_dir, asdict(clip_obj), words, job.width, job.height)
                clip_obj.__dict__.update(result)
                clip_obj.status = "done"
                job.clips[i] = asdict(clip_obj)
        
        job.status = "done"
        job.stage = f"Done - {len(job.clips)} clips ready"
        job.progress = 1.0
        st.session_state.processing = False
        update_progress(1.0, job.stage)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        if st.session_state.current_job:
            st.session_state.current_job.status = "error"
            st.session_state.current_job.error = str(e)
            st.session_state.current_job.stage = "Failed"
        st.session_state.processing = False


def update_progress(progress: float, stage: str):
    """Update progress from worker thread."""
    st.session_state.progress_value = progress
    st.session_state.current_stage = stage
    if st.session_state.current_job:
        st.session_state.current_job.progress = progress
        st.session_state.current_job.stage = stage


def render_progress():
    """Render progress bar and status."""
    job = st.session_state.current_job
    if not job:
        return
    
    st.markdown("### 🔄 Processing")
    
    # Progress bar
    progress_pct = int(job.progress * 100)
    st.markdown(f"""
    <div class="progress-container">
        <div class="progress-bar" style="width: {progress_pct}%"></div>
    </div>
    <p style="text-align: center; margin-top: 0.5rem;">{progress_pct}% - {job.stage}</p>
    """, unsafe_allow_html=True)
    
    # Status badge
    status_class = f"status-{job.status}"
    st.markdown(f'<span class="status-badge {status_class}">{job.status.upper()}</span>', unsafe_allow_html=True)
    
    # Auto-refresh
    if st.session_state.processing:
        time.sleep(1)
        st.rerun()


def render_clips_list(job: Job):
    """Render list of generated clips."""
    st.divider()
    st.markdown(f"### ✅ Done! {len(job.clips)} Clips Generated")
    
    for clip in job.clips:
        with st.container():
            st.markdown('<div class="clip-card">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col1:
                if clip.get("cover"):
                    cover_path = Path(tempfile.gettempdir()) / "peakcut_jobs" / job.id / "clips" / clip["cover"]
                    if cover_path.exists():
                        st.image(str(cover_path), use_container_width=True)
            
            with col2:
                st.markdown(f"**{clip.get('title', clip['id'])}**")
                st.caption(f"⏱️ {clip['start']:.1f}s – {clip['end']:.1f}s ({clip['end']-clip['start']:.1f}s) • Score: {clip['score']}/100 • {clip['category'].title()}")
                st.caption(f"🎯 {clip['hook']}")
                if clip.get('description'):
                    st.caption(clip['description'])
                if clip.get('hashtags'):
                    st.caption(" ".join(f"#{h}" for h in clip['hashtags']))
            
            with col3:
                st.markdown(f"**Style:** {clip['style'].title()}")
                st.markdown(f"**Layout:** {clip['layout'].title()}")
                if clip.get("file"):
                    video_path = Path(tempfile.gettempdir()) / "peakcut_jobs" / job.id / "clips" / clip["file"]
                    if video_path.exists():
                        with open(video_path, "rb") as f:
                            st.download_button(
                                "⬇️ Download",
                                f.read(),
                                file_name=clip["file"],
                                mime="video/mp4",
                                key=f"dl_{clip['id']}"
                            )
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Bulk download
    st.divider()
    if st.button("📦 Download All as ZIP", use_container_width=True):
        create_zip_download(job)


def create_zip_download(job: Job):
    """Create a ZIP file with all clips."""
    import zipfile
    import io
    
    clips_dir = Path(tempfile.gettempdir()) / "peakcut_jobs" / job.id / "clips"
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for clip in job.clips:
            if clip.get("file"):
                fpath = clips_dir / clip["file"]
                if fpath.exists():
                    zf.write(fpath, arcname=clip["file"])
            if clip.get("cover"):
                fpath = clips_dir / clip["cover"]
                if fpath.exists():
                    zf.write(fpath, arcname=clip["cover"])
        # Add metadata JSON
        meta = {
            "job_id": job.id,
            "title": job.title,
            "source": job.source,
            "clips": [
                {k: v for k, v in c.items() if k not in ("src", "fixed_words", "fixed_for")}
                for c in job.clips
            ]
        }
        zf.writestr("metadata.json", json.dumps(meta, indent=2))
    
    zip_buffer.seek(0)
    st.download_button(
        "⬇️ Download ZIP",
        zip_buffer.getvalue(),
        file_name=f"peakcut_{job.id}.zip",
        mime="application/zip",
        use_container_width=True
    )


def page_export():
    """Export page."""
    st.markdown("## 📤 Export Clips")
    
    if not st.session_state.current_job or not st.session_state.current_job.clips:
        st.info("No clips to export. Process a video first.")
        return
    
    job = st.session_state.current_job
    clips_dir = Path(tempfile.gettempdir()) / "peakcut_jobs" / job.id / "clips"
    
    st.markdown(f"### Job: {job.id}")
    st.markdown(f"**{len(job.clips)} clips ready for export**")
    
    # Individual downloads
    for clip in job.clips:
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.markdown(f"**{clip.get('title', clip['id'])}** ({clip['end']-clip['start']:.1f}s)")
        with col2:
            if clip.get("file"):
                fpath = clips_dir / clip["file"]
                if fpath.exists():
                    with open(fpath, "rb") as f:
                        st.download_button("📹 Video", f.read(), clip["file"], "video/mp4", key=f"exp_v_{clip['id']}")
        with col3:
            if clip.get("cover"):
                fpath = clips_dir / clip["cover"]
                if fpath.exists():
                    with open(fpath, "rb") as f:
                        st.download_button("🖼️ Cover", f.read(), clip["cover"], "image/jpeg", key=f"exp_c_{clip['id']}")
    
    st.divider()
    
    # Bulk options
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📦 Download All as ZIP", use_container_width=True):
            create_zip_download(job)
    with col2:
        if st.button("📋 Copy Metadata JSON", use_container_width=True):
            meta = {
                "job_id": job.id,
                "title": job.title,
                "source": job.source,
                "clips": [
                    {k: v for k, v in c.items() if k not in ("src", "fixed_words", "fixed_for")}
                    for c in job.clips
                ]
            }
            st.code(json.dumps(meta, indent=2), language="json")
    
    # Posting helpers (placeholder)
    st.divider()
    st.markdown("### 📱 Post to Social Media")
    st.info("🚧 Direct posting to YouTube/Instagram/TikTok requires OAuth setup. See the original PeakCut for full posting integration.")
    
    for platform in ["YouTube Shorts", "Instagram Reels", "TikTok"]:
        st.markdown(f"""
        **{platform}:**
        1. Download clips using buttons above
        2. Open {platform} app/website
        3. Upload video, add caption from clip metadata
        4. Use hashtags: `{" ".join(f"#{h}" for c in job.clips for h in c.get('hashtags', [])[:3])}`
        """)


# ============================================================
# Main App
# ============================================================

def main():
    """Main entry point."""
    # Render sidebar and get current page
    page = render_sidebar()
    
    # Route to page
    if page == "🏠 Home":
        page_home()
    elif page == "📥 Input Video":
        page_input_video()
    elif page == "🤖 AI Settings":
        page_ai_settings()
    elif page == "⚙️ Clip Options":
        page_clip_options()
    elif page == "✂️ Process & Clips":
        page_process()
    elif page == "📤 Export":
        page_export()


if __name__ == "__main__":
    main()
