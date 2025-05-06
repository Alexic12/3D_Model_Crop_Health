"""
FastAPI gateway – exposes /desktop and /mobile.
Run with:
    uvicorn app.api.main:app --host 0.0.0.0 --port 8000
This module also starts the Streamlit workers on start‑up.
"""

from __future__ import annotations
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse
from starlette.middleware.cors import CORSMiddleware
import socket

from app.config.config import settings
from app.api.process_manager import build_default_manager, ProcessManager

logger = logging.getLogger(__name__)
manager: ProcessManager | None = None

# ── Lifespan handler for startup and shutdown ─────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global manager
    manager = build_default_manager()
    manager.start_all()  # Start the background Streamlit workers (via threads)
    logger.info("✅ ProcessManager started.")
    yield
    if manager:
        manager._shutdown.set()
        manager.stop_all()
        logger.info("🛑 ProcessManager shutdown complete.")

app = FastAPI(
    title="Crop Health Visualizer – Gateway",
    description="Redirects /desktop and /mobile → Streamlit workers.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS configuration ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ── Helper function for redirects ─────────────────────────────────────
def _make_redirect(port: int) -> RedirectResponse:
    host_ip = socket.gethostbyname(socket.gethostname())
    url = f"http://{host_ip}:{port}"
    return RedirectResponse(url=url, status_code=307)

# ── Landing page with styled buttons ─────────────────────────────
@app.get("/", include_in_schema=False, response_class=HTMLResponse)
async def landing_page():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Crop Health Gateway</title>
        <style>
            body {
                margin: 0;
                font-family: 'Segoe UI', sans-serif;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                height: 100vh;
                background-color: #0b0c2a;
                color: #fff;
                text-align: center;
            }
            h1 {
                font-size: 1.8rem;
                margin-bottom: 2rem;
                color: #ffffffcc;
            }
            .button {
                display: block;
                margin: 10px auto;
                padding: 14px 28px;
                font-size: 1.2rem;
                background-color: #1f1fff;
                color: white;
                border: none;
                border-radius: 12px;
                cursor: pointer;
                transition: background-color 0.3s ease;
                width: 80%;
                max-width: 300px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.3);
            }
            .button:hover {
                background-color: #3939ff;
            }
            footer {
                position: absolute;
                bottom: 1rem;
                font-size: 0.9rem;
                color: #888;
            }
        </style>
    </head>
    <body>
        <h1>🌿 Crop Health Visualizer</h1>
        <a href="/desktop"><button class="button">📊 Desktop Version</button></a>
        <a href="/mobile"><button class="button">📱 Mobile Version</button></a>
        <footer>© 2025 F2X</footer>
    </body>
    </html>
    """

# ── Redirect routes ─────────────────────────────────────────────
@app.get("/desktop", tags=["UI"])
async def serve_desktop():
    logger.info(f"🔁 Redirecting to desktop app on port {settings.DESKTOP_PORT}")
    return _make_redirect(settings.DESKTOP_PORT)

@app.get("/mobile", tags=["UI"])
async def serve_mobile():
    logger.info(f"🔁 Redirecting to mobile app on port {settings.MOBILE_PORT}")
    return _make_redirect(settings.MOBILE_PORT)

# ── Catch‑all for undefined routes ─────────────────────────────────────
@app.get("/{path:path}", include_in_schema=False)
async def catch_all(path: str):
    raise HTTPException(status_code=404, detail=f"No route /{path}")
