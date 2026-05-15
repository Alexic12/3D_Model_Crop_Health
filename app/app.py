#!/usr/bin/env python3
"""
app.py – Bootstrap the full gateway with FastAPI and Streamlit worker supervisor.

Usage:
$ python app.py            # Runs FastAPI gateway and starts Streamlit workers

Requirements:
- PYTHONPATH is set to the app root for proper absolute imports
- uvicorn must be installed in the active environment
"""

import os
import platform
import subprocess
import sys
from pathlib import Path

from app.config.config import settings

def is_windows() -> bool:
    return platform.system().lower() == "windows"

def run_uvicorn():
    # Set PYTHONPATH so "from app.api.main" works
    base_dir = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = str(base_dir)

    # Command to run FastAPI via uvicorn
    cmd = [
        sys.executable,
        "-m", "uvicorn",
        "app.api.main:app",
        "--host", settings.API_HOST,
        "--port", str(settings.API_PORT),
        "--reload"
    ]

    print(f"🚀 Starting FastAPI gateway on http://localhost:{settings.API_PORT}")
    print(f"📡 Redirects: /desktop → {settings.DESKTOP_PORT} | /mobile → {settings.MOBILE_PORT}")
    print("🧠 Streamlit workers auto-managed by ProcessManager\n")

    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start uvicorn: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_uvicorn()
