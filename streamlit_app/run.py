#!/usr/bin/env python3
"""Launch script for PeakCut Streamlit app."""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    app_path = Path(__file__).parent / "main.py"
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port", "8501",
        "--server.headless", "true",
        "--server.address", "0.0.0.0",
    ])
