# app/run_app_uvicorn.py  (can live in app/ or repo root)
import os, sys, subprocess
from pathlib import Path

def repo_root() -> Path:
    here = Path(__file__).resolve()
    return here.parent.parent if here.parent.name == "app" else here.parent

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

    host = os.getenv("HOST", "0.0.0.0")
    port = os.getenv("PORT", "8000")
    log_level = os.getenv("LOG_LEVEL", "info")
    workers = os.getenv("UVICORN_WORKERS")

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