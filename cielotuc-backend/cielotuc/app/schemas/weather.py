"""
app/schemas/weather.py
───────────────────────
Pydantic v2 schemas — request bodies and response models.
Kept separate from ORM models so the API contract is explicit.
"""
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# ── Zone ───────────────────────────────────────────────────────
class ZoneBase(BaseModel):
    name: str
    department: str
    latitude: float
    longitude: float
    altitude_m: Optional[float] = None
    population: Optional[int] = None
    is_mountain: bool = False


class ZoneOut(ZoneBase):
    id: int
    impermeable_pct: Optional[float] = None

    model_config = {"from_attributes": True}


# ── Current conditions (real-time) ────────────────────────────
class CurrentConditions(BaseModel):
    zone_id: int
    zone_name: str
    timestamp: datetime

    temperature_c: float
    feels_like_c: Optional[float] = None
    humidity_pct: float
    pressure_hpa: float
    wind_speed_kmh: float
    wind_direction_deg: Optional[float] = None
    wind_gust_kmh: Optional[float] = None
    precip_1h_mm: float = 0.0
    precip_24h_mm: float = 0.0
    visibility_km: Optional[float] = None
    uv_index: Optional[float] = None
    cloud_cover_pct: Optional[float] = None

    condition: str = "unknown"
    # e.g. "sunny" | "cloudy" | "rain" | "storm"

    # AI overlay
    ai_confidence: Optional[float] = None
    zonda_risk_score: Optional[float] = None   # 0–100

    model_config = {"from_attributes": True}


# ── Hourly forecast item ───────────────────────────────────────
class HourlyForecast(BaseModel):
    target_time: datetime
    horizon_hours: int
    temperature_c: float
    rain_probability: float = Field(ge=0.0, le=1.0)
    precip_mm_expected: float = 0.0
    wind_speed_kmh: Optional[float] = None
    condition: str
    confidence: float = Field(ge=0.0, le=1.0)


# ── Daily forecast item ───────────────────────────────────────
class DailyForecast(BaseModel):
    date: str           # "2026-04-20"
    day_name: str       # "Lunes"
    condition: str
    temp_min_c: float
    temp_max_c: float
    rain_probability: float = Field(ge=0.0, le=1.0)
    precip_mm_expected: float = 0.0
    ai_confidence: float = Field(ge=0.0, le=1.0)

    # Extreme event probabilities for this day
    hail_risk: float = 0.0
    zonda_risk: float = 0.0
    storm_risk: float = 0.0


# ── Full forecast response ────────────────────────────────────
class ForecastResponse(BaseModel):
    zone_id: int
    zone_name: str
    generated_at: datetime
    model_version: str

    current: CurrentConditions
    hourly: List[HourlyForecast]    # next 12h
    daily: List[DailyForecast]      # next 7 days


# ── AI Prediction out ─────────────────────────────────────────
class PredictionOut(BaseModel):
    id: int
    zone_id: int
    zone_name: Optional[str] = None
    generated_at: datetime
    target_time: datetime
    horizon_hours: int
    rain_probability: float
    precip_mm_expected: Optional[float] = None
    temperature_c: Optional[float] = None
    temperature_min_c: Optional[float] = None
    temperature_max_c: Optional[float] = None
    condition: Optional[str] = None
    confidence: float
    zonda_risk: Optional[float] = None
    hail_risk: Optional[float] = None
    storm_risk: Optional[float] = None
    heatwave_risk: Optional[float] = None

    model_config = {"from_attributes": True}


# ── Comparison with SMN / Weather.com ─────────────────────────
class ComparisonRow(BaseModel):
    variable: str
    cielotuc: str
    smn: str
    weather_com: str


class ComparisonResponse(BaseModel):
    zone_id: int
    generated_at: datetime
    rows: List[ComparisonRow]

    # Accuracy leaderboard (last 30 days)
    cielotuc_accuracy: float
    smn_accuracy: float
    weathercom_accuracy: float

    # Wins table
    notable_wins: List[dict]


# ── Zonda index ───────────────────────────────────────────────
class ZondaIndex(BaseModel):
    timestamp: datetime
    risk_score: float = Field(ge=0.0, le=100.0)
    risk_level: str     # "none" | "low" | "medium" | "high" | "active"
    cordillera_pressure_hpa: Optional[float] = None
    thermal_differential_c: Optional[float] = None
    descending_speed_kmh: Optional[float] = None
    forecast_24h_probability: float = Field(ge=0.0, le=1.0)


# ── Model metrics ─────────────────────────────────────────────
class ModelMetrics(BaseModel):
    version: str
    trained_at: datetime
    training_samples: Optional[int] = None
    accuracy_overall: Optional[float] = None
    precision_rain: Optional[float] = None
    recall_rain: Optional[float] = None
    rmse_temperature: Optional[float] = None
    avg_lead_time_hours: Optional[float] = None
    false_positive_rate: Optional[float] = None
    is_active: bool

    # Accuracy trend (last N retraining cycles)
    accuracy_history: List[dict] = []


# ── FLOOD·TUC alert payload ───────────────────────────────────
class FloodAlertCreate(BaseModel):
    zone_id: int
    severity: str = "moderate"   # preventive | moderate | critical
    rain_probability: float
    expected_precip_mm: Optional[float] = None
    trigger_type: str = "automatic"
    notes: Optional[str] = None


class FloodAlertOut(FloodAlertCreate):
    id: int
    triggered_at: datetime
    delivered: bool
    floodtuc_response_code: Optional[int] = None
    outcome: Optional[str] = None

    model_config = {"from_attributes": True}


# ── Sensor reading ingest ─────────────────────────────────────
class SensorReadingCreate(BaseModel):
    station_code: str
    timestamp: datetime
    temperature_c: Optional[float] = None
    humidity_pct: Optional[float] = None
    pressure_hpa: Optional[float] = None
    precip_mm: Optional[float] = None
    wind_speed_kmh: Optional[float] = None
    wind_direction_deg: Optional[float] = None
    wind_gust_kmh: Optional[float] = None

    @field_validator("temperature_c")
    @classmethod
    def sane_temperature(cls, v):
        if v is not None and not (-30 <= v <= 55):
            raise ValueError(f"Temperature {v}°C out of plausible range for Tucumán")
        return v
