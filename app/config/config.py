"""Centralised configuration and logging for the Crop Health application."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AliasChoices, Field

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CHV_",
        case_sensitive=False,
    )

    # ── Streamlit workers ──────────────────────────────────────────────
    DESKTOP_SCRIPT: str = Field(default=str(BASE_DIR / "desktop_app/ui_desktop.py"))
    MOBILE_SCRIPT:  str = Field(default=str(BASE_DIR / "mobile_app/ui_mobile.py"))
    DESKTOP_PORT:   int = Field(default=8501)
    MOBILE_PORT:    int = Field(default=8502)

    # ── FastAPI server ─────────────────────────────────────────────────
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)

    # ── Misc ───────────────────────────────────────────────────────────
    LOG_LEVEL: str = Field(default="INFO")
    
    # ── Google Maps API Key (optional; set in environment or .env) ─────
    GOOGLE_MAPS_API_KEY: str = Field(
        default="",
        validation_alias=AliasChoices("CHV_GOOGLE_MAPS_API_KEY", "GOOGLE_MAPS_API_KEY"),
    )

settings = Settings()

# ── Logging setup ─────────────────────────────────────────────────────
import logging

LOG_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format=LOG_FMT,
    datefmt="%Y-%m-%d %H:%M:%S",
)
