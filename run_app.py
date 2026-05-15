"""Repository entrypoint for the Crop Health gateway.

Starts the FastAPI gateway, which supervises the desktop and mobile Streamlit
workers through app.api.process_manager.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    """Return the project root for both source and PyInstaller layouts."""
    here = Path(__file__).resolve()
    if (here.parent / "_internal" / "app").exists():
        return here.parent / "_internal"
    return here.parent


def is_port_available(host: str, port: int) -> bool:
    bind_host = "" if host in {"0.0.0.0", "::"} else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((bind_host, port))
        except OSError:
            return False
    return True


def main() -> None:
    root = repo_root()
    os.chdir(root)

    if not (root / "app" / "api" / "main.py").exists():
        print(f"[ERROR] Could not find FastAPI gateway under {root / 'app' / 'api' / 'main.py'}.")
        sys.exit(1)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from app.config.config import settings

    host = env.get("HOST", env.get("CHV_API_HOST", settings.API_HOST))
    port = env.get("PORT", env.get("CHV_API_PORT", str(settings.API_PORT)))
    log_level = env.get("LOG_LEVEL", env.get("CHV_LOG_LEVEL", settings.LOG_LEVEL.lower()))
    workers = env.get("UVICORN_WORKERS")
    required_ports = {
        "gateway": int(port),
        "desktop": int(settings.DESKTOP_PORT),
        "mobile": int(settings.MOBILE_PORT),
    }
    busy_ports = [
        f"{name}:{port_}"
        for name, port_ in required_ports.items()
        if not is_port_available(host, port_)
    ]
    if busy_ports:
        print(f"[ERROR] Required port(s) already in use: {', '.join(busy_ports)}")
        print("Close the existing app instance or set CHV_API_PORT, CHV_DESKTOP_PORT, and CHV_MOBILE_PORT.")
        sys.exit(1)

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.api.main:app",
        "--host",
        host,
        "--port",
        str(port),
        "--log-level",
        log_level,
    ]
    if workers:
        cmd.extend(["--workers", str(workers)])

    print("Running Crop Health gateway:", " ".join(cmd))
    try:
        subprocess.run(cmd, env=env, check=True)
    except FileNotFoundError:
        print("[ERROR] uvicorn not found. Install dependencies with: pip install -r requirements.txt")
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] Gateway exited with code {exc.returncode}")
        sys.exit(exc.returncode)


if __name__ == "__main__":
    main()
