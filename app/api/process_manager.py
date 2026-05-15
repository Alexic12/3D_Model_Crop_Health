"""
Robust supervisor for Streamlit workers.
• Starts each app in its own subprocess using threads.
• Restarts automatically if the worker crashes.
• Cross-platform (works on Windows, Linux, macOS).
"""

import logging
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

from app.config.config import settings

logger = logging.getLogger(__name__)

@dataclass
class WorkerSpec:
    name: str
    script: str
    port: int
    proc: Optional[subprocess.Popen] = None

class ProcessManager:
    def __init__(self) -> None:
        self._shutdown = threading.Event()
        self._workers = [
            WorkerSpec(name="desktop", script=settings.DESKTOP_SCRIPT, port=settings.DESKTOP_PORT),
            WorkerSpec(name="mobile",  script=settings.MOBILE_SCRIPT,  port=settings.MOBILE_PORT),
        ]

    def start_all(self) -> None:
        for spec in self._workers:
            thread = threading.Thread(target=self._launch_worker, args=(spec,), daemon=True)
            thread.start()
            logger.info(f"🧵 Worker thread started for {spec.name}")

    def _launch_worker(self, spec: WorkerSpec) -> None:
        while not self._shutdown.is_set():
            try:
                cmd = [
                    sys.executable, "-m", "streamlit", "run", spec.script,
                    "--server.port", str(spec.port),
                    "--server.address", "0.0.0.0",
                    "--server.headless", "true",
                    "--browser.gatherUsageStats", "false"
                ]
                logger.info(f"🚀 Launching {spec.name} → {' '.join(cmd)}")
                spec.proc = subprocess.Popen(cmd)
                spec.proc.wait()
                if self._shutdown.is_set():
                    break
                logger.warning(f"⚠️ {spec.name} exited unexpectedly. Restarting in 2 seconds...")
                time.sleep(2)
            except Exception as e:
                logger.exception(f"❌ Failed to launch {spec.name}: {e}")
                time.sleep(5)

    def stop_all(self) -> None:
        self._shutdown.set()
        for spec in self._workers:
            if spec.proc and spec.proc.poll() is None:
                logger.info(f"🛑 Terminating {spec.name}")
                spec.proc.terminate()
                try:
                    spec.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning(f"⚠️ Force killing unresponsive {spec.name}")
                    spec.proc.kill()

def build_default_manager() -> ProcessManager:
    return ProcessManager()
