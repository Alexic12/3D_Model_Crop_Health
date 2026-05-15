# app/run_app_uvicorn.py  (can live in app/ or repo root)
import os, socket, sys, subprocess
from pathlib import Path

def repo_root() -> Path:
    here = Path(__file__).resolve()
    return here.parent.parent if here.parent.name == "app" else here.parent

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
    os.chdir(root)  # make repo root the CWD

    main_py = root / "app" / "api" / "main.py"
    if not main_py.exists():
        print(f"[ERROR] Couldn't find {main_py}.")
        sys.exit(1)

    # Ensure package marker exists
    init_file = root / "app" / "__init__.py"
    if not init_file.exists():
        init_file.touch()

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from app.config.config import settings

    host = os.getenv("HOST", os.getenv("CHV_API_HOST", settings.API_HOST))
    port = os.getenv("PORT", os.getenv("CHV_API_PORT", str(settings.API_PORT)))
    log_level = os.getenv("LOG_LEVEL", os.getenv("CHV_LOG_LEVEL", settings.LOG_LEVEL.lower()))
    workers = os.getenv("UVICORN_WORKERS")
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

    cmd = [sys.executable, "-m", "uvicorn", "app.api.main:app",
           "--host", host, "--port", str(port), "--log-level", log_level]
    if workers:
        cmd += ["--workers", str(workers)]

    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("[ERROR] uvicorn not found. pip install 'uvicorn[standard]'")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Uvicorn exited with code {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
