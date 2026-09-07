"""
app/core/config.py
──────────────────
Typed settings loaded from environment / .env file.
All other modules import `settings` from here — never read
os.environ directly.
"""
from functools import lru_cache
from typing import Annotated, List

from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ────────────────────────────────────────────────────
    app_name: str = "CIELO·TUC"
    app_version: str = "0.1.0"
    debug: bool = False
    secret_key: str

    # ── Database ───────────────────────────────────────────────
    database_url: str           # async (asyncpg)
    database_url_sync: str      # sync (psycopg2) — for Alembic

    # ── Scheduler ─────────────────────────────────────────────
    scheduler_enabled: bool = True  # set False to disable background tasks

    # ── External APIs ──────────────────────────────────────────
    smn_api_url: str = "https://ws.smn.gob.ar/map_items/weather"
    nasa_gpm_base_url: str = "https://gpm.nasa.gov/api/v1"
    nasa_power_base_url: str = "https://power.larc.nasa.gov/api/temporal"
    era5_api_key: str = ""

    # ── Windy API ─────────────────────────────────────────────
    # Point Forecast API — ECMWF/GFS multi-level. Key: https://api.windy.com
    windy_api_key: str = ""
    windy_model: str = "ecmwf"
    windy_point_forecast_url: str = "https://api.windy.com/api/point-forecast/v2"
    windy_time_step_hours: int = 3

    # ── Twilio ─────────────────────────────────────────────────
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from_phone: str = ""
    twilio_whatsapp_from: str = ""
    twilio_alert_recipients: Annotated[List[str], NoDecode] = []
    # Comma-separated list of E.164 phone numbers for emergency SMS alerts

    # ── MLflow ─────────────────────────────────────────────────
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "cielotuc-weather-lstm"

    # ── CORS ───────────────────────────────────────────────────
    # Comma-separated list. NoDecode keeps the raw env string so the
    # validator below can split it (avoids pydantic-settings JSON parse).
    cors_origins: Annotated[List[str], NoDecode] = [
        "http://localhost:5173",
        "https://cielo-tuc.vercel.app",
    ]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors(cls, v):
        if isinstance(v, str):
            return [o.strip() for o in v.split(",")]
        return v

    @field_validator("twilio_alert_recipients", mode="before")
    @classmethod
    def parse_recipients(cls, v):
        if isinstance(v, str):
            return [p.strip() for p in v.split(",") if p.strip()]
        return v

    # ── FLOOD·TUC ──────────────────────────────────────────────
    floodtuc_api_url: str = "http://localhost:8001"
    floodtuc_api_key: str = ""
    floodtuc_alert_threshold_mm: float = 70.0

    # ── ML Model ───────────────────────────────────────────────
    model_retrain_interval_days: int = 30
    model_lookback_hours: int = 72      # input window
    model_forecast_horizons: List[int] = [3, 6, 12, 24, 48, 168]  # hours ahead

    # ── Province constants ─────────────────────────────────────
    tucuman_lat: float = -26.82
    tucuman_lng: float = -65.22
    tucuman_bbox: List[float] = [-28.0, -66.5, -26.0, -64.5]  # S,W,N,E


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
