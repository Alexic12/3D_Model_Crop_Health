"""
Centralised configuration & logging.
Edit ports / paths here – nothing hard‑coded elsewhere.
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    # ── Streamlit workers ──────────────────────────────────────────────
    DESKTOP_SCRIPT: str = Field(default=str(BASE_DIR / "desktop_app/ui_desktop.py"))
    MOBILE_SCRIPT:  str = Field(default=str(BASE_DIR / "mobile_app/ui_mobile.py"))
    DESKTOP_PORT:   int = Field(default=8504)
    MOBILE_PORT:    int = Field(default=8505)

    # ── FastAPI server ─────────────────────────────────────────────────
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)

    # ── Misc ───────────────────────────────────────────────────────────
    LOG_LEVEL: str = Field(default="INFO")
    
    # ── Google Maps API Key (for AWS Fargate deployment) ──────────────
    GOOGLE_MAPS_API_KEY: str = Field(default="AIzaSyB1Vv2XMsTy1AxEowrzOaI5Sn96ffC6HNY")

    class Config:
        env_prefix = "CHV_"
        case_sensitive = False

settings = Settings()

# ── Logging setup ─────────────────────────────────────────────────────
import logging

LOG_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format=LOG_FMT,
    datefmt="%Y-%m-%d %H:%M:%S",
)