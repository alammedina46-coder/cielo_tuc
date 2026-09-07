"""
app/models/weather.py
──────────────────────
All ORM models for CIELO·TUC.

Tables:
  zones            — Tucumán's 17 departments + neighbourhoods
  weather_stations — Physical / virtual sensor locations
  sensor_readings  — Raw time-series data from every station
  weather_events   — Historical extreme weather events (labelled)
  ai_predictions   — Every prediction the model has made
  model_versions   — Registry of trained model versions
  flood_alerts     — Alerts sent downstream to FLOOD·TUC
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


# ── Zones (17 departments + extra neighbourhoods) ──────────────
class Zone(Base):
    __tablename__ = "zones"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    department: Mapped[str] = mapped_column(String(100), nullable=False)
    zone_type: Mapped[str] = mapped_column(String(30), default="department")
    # "department" | "neighbourhood" | "rural"

    # Centroid coordinates
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    altitude_m: Mapped[Optional[float]] = mapped_column(Float)

    # Terrain / land-use features (used as ML inputs)
    impermeable_pct: Mapped[Optional[float]] = mapped_column(Float)
    # % of area covered by impermeable surfaces (asphalt, concrete)
    population: Mapped[Optional[int]] = mapped_column(Integer)
    is_mountain: Mapped[bool] = mapped_column(Boolean, default=False)
    # True for Tafí del Valle, Tafí Viejo, sierra zones

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    stations: Mapped[list["WeatherStation"]] = relationship(back_populates="zone")
    predictions: Mapped[list["AIPrediction"]] = relationship(back_populates="zone")
    events: Mapped[list["WeatherEvent"]] = relationship(back_populates="zone")


# ── Weather stations ───────────────────────────────────────────
class WeatherStation(Base):
    __tablename__ = "weather_stations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    zone_id: Mapped[int] = mapped_column(ForeignKey("zones.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    station_code: Mapped[str] = mapped_column(String(20), unique=True, index=True)

    source: Mapped[str] = mapped_column(String(30), nullable=False)
    # "own_iot" | "smn" | "eeaoc" | "weather_underground" | "virtual"

    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    altitude_m: Mapped[Optional[float]] = mapped_column(Float)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    battery_pct: Mapped[Optional[float]] = mapped_column(Float)
    last_seen: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    zone: Mapped["Zone"] = relationship(back_populates="stations")
    readings: Mapped[list["SensorReading"]] = relationship(back_populates="station")


# ── Sensor readings (time-series core table) ───────────────────
class SensorReading(Base):
    __tablename__ = "sensor_readings"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True)
    station_id: Mapped[int] = mapped_column(
        ForeignKey("weather_stations.id"), nullable=False, index=True
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    # ── Atmospheric variables (the 34 ML features) ────────────
    temperature_c: Mapped[Optional[float]] = mapped_column(Float)
    temperature_min_c: Mapped[Optional[float]] = mapped_column(Float)
    temperature_max_c: Mapped[Optional[float]] = mapped_column(Float)
    feels_like_c: Mapped[Optional[float]] = mapped_column(Float)

    pressure_hpa: Mapped[Optional[float]] = mapped_column(Float)
    pressure_sea_level_hpa: Mapped[Optional[float]] = mapped_column(Float)
    humidity_pct: Mapped[Optional[float]] = mapped_column(Float)
    dew_point_c: Mapped[Optional[float]] = mapped_column(Float)

    precip_mm: Mapped[Optional[float]] = mapped_column(Float)
    precip_3h_mm: Mapped[Optional[float]] = mapped_column(Float)
    precip_6h_mm: Mapped[Optional[float]] = mapped_column(Float)
    precip_24h_mm: Mapped[Optional[float]] = mapped_column(Float)

    wind_speed_kmh: Mapped[Optional[float]] = mapped_column(Float)
    wind_direction_deg: Mapped[Optional[float]] = mapped_column(Float)
    wind_gust_kmh: Mapped[Optional[float]] = mapped_column(Float)

    visibility_km: Mapped[Optional[float]] = mapped_column(Float)
    cloud_cover_pct: Mapped[Optional[float]] = mapped_column(Float)
    uv_index: Mapped[Optional[float]] = mapped_column(Float)

    # ── Derived / satellite variables ─────────────────────────
    cape_j_kg: Mapped[Optional[float]] = mapped_column(Float)
    # Convective Available Potential Energy — storm predictor
    k_index: Mapped[Optional[float]] = mapped_column(Float)
    # Atmospheric instability index
    ndvi: Mapped[Optional[float]] = mapped_column(Float)
    # Normalized Difference Vegetation Index from satellite
    soil_temp_c: Mapped[Optional[float]] = mapped_column(Float)
    precipitable_water_mm: Mapped[Optional[float]] = mapped_column(Float)

    # ── Zonda-specific variables ──────────────────────────────
    cordillera_pressure_hpa: Mapped[Optional[float]] = mapped_column(Float)
    thermal_differential_c: Mapped[Optional[float]] = mapped_column(Float)
    # temp difference Andes summit vs Tucumán plain

    # ── ENSO / large-scale context ────────────────────────────
    enso_index: Mapped[Optional[float]] = mapped_column(Float)
    # Oceanic Niño Index: negative=La Niña, positive=El Niño

    # ── Quality flags ─────────────────────────────────────────
    is_validated: Mapped[bool] = mapped_column(Boolean, default=False)
    quality_flag: Mapped[Optional[str]] = mapped_column(String(10))
    # "ok" | "suspect" | "missing" | "interpolated"

    station: Mapped["WeatherStation"] = relationship(back_populates="readings")


# ── Historical weather events (labelled for ML training) ───────
class WeatherEvent(Base):
    __tablename__ = "weather_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    zone_id: Mapped[int] = mapped_column(ForeignKey("zones.id"), nullable=False)

    event_type: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    # "storm" | "hail" | "zonda" | "heatwave" | "frost" | "flood_rain"

    severity: Mapped[str] = mapped_column(String(20), nullable=False)
    # "low" | "moderate" | "high" | "extreme"

    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    peak_precip_mm: Mapped[Optional[float]] = mapped_column(Float)
    peak_wind_kmh: Mapped[Optional[float]] = mapped_column(Float)
    peak_temp_c: Mapped[Optional[float]] = mapped_column(Float)

    # Was this event predicted by CIELO·TUC?
    was_predicted: Mapped[Optional[bool]] = mapped_column(Boolean)
    prediction_lead_hours: Mapped[Optional[float]] = mapped_column(Float)
    # How many hours before the model issued a correct alert

    source: Mapped[str] = mapped_column(String(50), default="manual")
    # "manual" | "smn_report" | "defensa_civil" | "news"
    notes: Mapped[Optional[str]] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    zone: Mapped["Zone"] = relationship(back_populates="events")


# ── AI predictions ─────────────────────────────────────────────
class AIPrediction(Base):
    __tablename__ = "ai_predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    zone_id: Mapped[int] = mapped_column(ForeignKey("zones.id"), nullable=False, index=True)
    model_version_id: Mapped[int] = mapped_column(
        ForeignKey("model_versions.id"), nullable=False
    )

    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )
    target_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    horizon_hours: Mapped[int] = mapped_column(Integer, nullable=False)
    # 3 | 6 | 12 | 24 | 48 | 168

    # ── Predictions ───────────────────────────────────────────
    rain_probability: Mapped[float] = mapped_column(Float, nullable=False)
    # 0.0 – 1.0
    precip_mm_expected: Mapped[Optional[float]] = mapped_column(Float)
    temperature_c: Mapped[Optional[float]] = mapped_column(Float)
    temperature_min_c: Mapped[Optional[float]] = mapped_column(Float)
    temperature_max_c: Mapped[Optional[float]] = mapped_column(Float)
    wind_speed_kmh: Mapped[Optional[float]] = mapped_column(Float)
    condition: Mapped[Optional[str]] = mapped_column(String(30))
    # "sunny" | "cloudy" | "rain" | "storm" | "hail" | "zonda"

    # ── Confidence & validation ───────────────────────────────
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    # model's internal confidence score 0–1
    was_confirmed: Mapped[Optional[bool]] = mapped_column(Boolean)
    # set after the event window passes
    actual_precip_mm: Mapped[Optional[float]] = mapped_column(Float)
    # filled in by the validation service

    # ── Extreme event flags ───────────────────────────────────
    zonda_risk: Mapped[Optional[float]] = mapped_column(Float)
    hail_risk: Mapped[Optional[float]] = mapped_column(Float)
    storm_risk: Mapped[Optional[float]] = mapped_column(Float)
    heatwave_risk: Mapped[Optional[float]] = mapped_column(Float)

    zone: Mapped["Zone"] = relationship(back_populates="predictions")
    model_version: Mapped["ModelVersion"] = relationship(back_populates="predictions")


# ── Model versions ─────────────────────────────────────────────
class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    version: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    # e.g. "3.2"
    mlflow_run_id: Mapped[Optional[str]] = mapped_column(String(64), unique=True)
    architecture: Mapped[str] = mapped_column(String(50), default="CNN-LSTM")

    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    training_samples: Mapped[Optional[int]] = mapped_column(Integer)
    training_start_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    training_end_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # ── Performance metrics ───────────────────────────────────
    accuracy_overall: Mapped[Optional[float]] = mapped_column(Float)
    precision_rain: Mapped[Optional[float]] = mapped_column(Float)
    recall_rain: Mapped[Optional[float]] = mapped_column(Float)
    rmse_temperature: Mapped[Optional[float]] = mapped_column(Float)
    avg_lead_time_hours: Mapped[Optional[float]] = mapped_column(Float)
    false_positive_rate: Mapped[Optional[float]] = mapped_column(Float)

    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    # Only one version is active at a time
    notes: Mapped[Optional[str]] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    predictions: Mapped[list["AIPrediction"]] = relationship(back_populates="model_version")


# ── FLOOD·TUC outbound alerts ──────────────────────────────────
class FloodAlert(Base):
    __tablename__ = "flood_alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    zone_id: Mapped[int] = mapped_column(ForeignKey("zones.id"), nullable=False)
    prediction_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("ai_predictions.id")
    )

    triggered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    trigger_type: Mapped[str] = mapped_column(String(20), default="automatic")
    # "automatic" | "manual"

    rain_probability: Mapped[float] = mapped_column(Float)
    expected_precip_mm: Mapped[Optional[float]] = mapped_column(Float)
    severity: Mapped[str] = mapped_column(String(20))
    # "preventive" | "moderate" | "critical"

    floodtuc_response_code: Mapped[Optional[int]] = mapped_column(Integer)
    floodtuc_response_body: Mapped[Optional[str]] = mapped_column(Text)
    delivered: Mapped[bool] = mapped_column(Boolean, default=False)

    outcome: Mapped[Optional[str]] = mapped_column(String(30))
    # "confirmed" | "false_positive" | "pending"
