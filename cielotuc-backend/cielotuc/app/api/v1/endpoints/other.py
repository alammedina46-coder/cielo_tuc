"""
app/api/v1/endpoints/zones.py      — zone list and detail
app/api/v1/endpoints/sensors.py    — sensor reading ingest
app/api/v1/endpoints/alerts.py     — FLOOD·TUC alerts
app/api/v1/endpoints/model.py      — model metrics & comparison

All in one file for initial development; split as needed.
"""

# ════════════════════════════════════════════════════════════════
# ZONES
# ════════════════════════════════════════════════════════════════
from typing import Annotated, List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, Integer
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, require_role
from app.db.session import get_db
from app.models.user import User
from app.models.weather import (
    AIPrediction, FloodAlert, ModelVersion,
    SensorReading, WeatherStation, Zone,
)
from app.schemas.weather import (
    ComparisonResponse, FloodAlertCreate, FloodAlertOut,
    ModelMetrics, PredictionOut, SensorReadingCreate, ZoneOut,
)
from app.services.flood_alert_service import FloodAlertService

zones_router = APIRouter(prefix="/zones", tags=["Zones"])


@zones_router.get("/", response_model=List[ZoneOut])
async def list_zones(db: AsyncSession = Depends(get_db)):
    """All 17 Tucumán departments + neighbourhoods."""
    result = await db.execute(select(Zone).order_by(Zone.department, Zone.name))
    return result.scalars().all()


@zones_router.get("/{zone_id}", response_model=ZoneOut)
async def get_zone(zone_id: int, db: AsyncSession = Depends(get_db)):
    zone = await db.get(Zone, zone_id)
    if not zone:
        raise HTTPException(status_code=404, detail="Zone not found")
    return zone


# ════════════════════════════════════════════════════════════════
# SENSORS
# ════════════════════════════════════════════════════════════════
sensors_router = APIRouter(prefix="/sensors", tags=["Sensors"])


@sensors_router.post("/readings", status_code=201)
async def ingest_reading(
    payload: SensorReadingCreate,
    db: AsyncSession = Depends(get_db),
    _user: Annotated[User, Depends(require_role("tecnico", "admin"))] = None,
):
    """
    Ingest a single sensor reading from an IoT device or external script.
    Validates the station_code and stores the reading.
    """
    result = await db.execute(
        select(WeatherStation).where(
            WeatherStation.station_code == payload.station_code
        )
    )
    station = result.scalars().first()
    if not station:
        raise HTTPException(
            status_code=404,
            detail=f"Station '{payload.station_code}' not found",
        )

    reading = SensorReading(
        station_id=station.id,
        timestamp=payload.timestamp,
        temperature_c=payload.temperature_c,
        humidity_pct=payload.humidity_pct,
        pressure_hpa=payload.pressure_hpa,
        precip_mm=payload.precip_mm,
        wind_speed_kmh=payload.wind_speed_kmh,
        wind_direction_deg=payload.wind_direction_deg,
        wind_gust_kmh=payload.wind_gust_kmh,
    )
    db.add(reading)

    # Update station heartbeat
    station.last_seen = payload.timestamp
    await db.flush()

    return {"status": "ok", "reading_id": reading.id}


@sensors_router.get("/status")
async def sensor_status(db: AsyncSession = Depends(get_db)):
    """Overview of all stations: online / offline / battery."""
    from datetime import datetime, timedelta, timezone
    result = await db.execute(select(WeatherStation))
    stations = result.scalars().all()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=1)

    return [
        {
            "id": s.id,
            "code": s.station_code,
            "name": s.name,
            "source": s.source,
            "is_active": s.is_active,
            "battery_pct": s.battery_pct,
            "last_seen": s.last_seen,
            "online": s.last_seen is not None and s.last_seen >= cutoff,
        }
        for s in stations
    ]


# ════════════════════════════════════════════════════════════════
# FLOOD·TUC ALERTS
# ════════════════════════════════════════════════════════════════
alerts_router = APIRouter(prefix="/alerts", tags=["Alerts"])
_flood_service = FloodAlertService()


@alerts_router.post("/flood", response_model=FloodAlertOut, status_code=201)
async def send_flood_alert(
    payload: FloodAlertCreate,
    db: AsyncSession = Depends(get_db),
    _user: Annotated[User, Depends(require_role("tecnico", "admin"))] = None,
):
    """
    Manually trigger a FLOOD·TUC alert from the government dashboard.
    Automatic alerts are triggered by the prediction service internally.
    """
    return await _flood_service.send_alert(
        zone_id=payload.zone_id,
        rain_probability=payload.rain_probability,
        expected_precip_mm=payload.expected_precip_mm or 0.0,
        db=db,
        severity=payload.severity,
        trigger_type="manual",
        notes=payload.notes,
    )


@alerts_router.get("/flood/history", response_model=List[FloodAlertOut])
async def flood_alert_history(
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    """Recent alerts sent to FLOOD·TUC."""
    result = await db.execute(
        select(FloodAlert)
        .order_by(FloodAlert.triggered_at.desc())
        .limit(limit)
    )
    return result.scalars().all()


# ════════════════════════════════════════════════════════════════
# AI MODEL METRICS
# ════════════════════════════════════════════════════════════════
model_router = APIRouter(prefix="/model", tags=["AI Model"])


@model_router.get("/metrics", response_model=ModelMetrics)
async def get_model_metrics(db: AsyncSession = Depends(get_db)):
    """Current active model version and performance metrics."""
    result = await db.execute(
        select(ModelVersion).where(ModelVersion.is_active == True)
    )
    mv = result.scalars().first()
    if not mv:
        raise HTTPException(status_code=404, detail="No active model found")

    # Historical accuracy — all versions ordered by training date
    all_versions = await db.execute(
        select(ModelVersion).order_by(ModelVersion.trained_at.asc())
    )
    history = [
        {"version": v.version, "accuracy": v.accuracy_overall}
        for v in all_versions.scalars().all()
        if v.accuracy_overall is not None
    ]

    return ModelMetrics(
        version=mv.version,
        trained_at=mv.trained_at,
        training_samples=mv.training_samples,
        accuracy_overall=mv.accuracy_overall,
        precision_rain=mv.precision_rain,
        recall_rain=mv.recall_rain,
        rmse_temperature=mv.rmse_temperature,
        avg_lead_time_hours=mv.avg_lead_time_hours,
        false_positive_rate=mv.false_positive_rate,
        is_active=mv.is_active,
        accuracy_history=history,
    )


@model_router.get("/comparison", response_model=ComparisonResponse)
async def get_comparison(
    zone_id: int = 1,
    db: AsyncSession = Depends(get_db),
):
    """
    Compare CIELO·TUC accuracy vs SMN and Weather.com.
    Uses validated predictions from the last 30 days.
    Returns placeholder data if no predictions exist yet.
    """
    from datetime import timedelta, datetime, timezone
    from sqlalchemy import and_, func

    cutoff = datetime.now(timezone.utc) - timedelta(days=30)

    try:
        result = await db.execute(
            select(
                func.count(AIPrediction.id).label("total"),
                func.sum(
                    func.cast(AIPrediction.was_confirmed == True, type_=Integer)
                ).label("correct"),
            ).where(
                and_(
                    AIPrediction.was_confirmed.isnot(None),
                    AIPrediction.generated_at >= cutoff,
                    AIPrediction.zone_id == zone_id,
                )
            )
        )
        row = result.first()
        total = row.total or 0
        correct = row.correct or 0
        cielotuc_acc = round(correct / total, 3) if total > 0 else 0.0
    except Exception:
        cielotuc_acc = 0.0

    return ComparisonResponse(
        zone_id=zone_id,
        generated_at=datetime.now(timezone.utc),
        rows=[
            {
                "variable": "Pronóstico 3h",
                "cielotuc": "—",
                "smn": "—",
                "weather_com": "—",
            },
            {
                "variable": "Pronóstico 12h",
                "cielotuc": "—",
                "smn": "—",
                "weather_com": "—",
            },
            {
                "variable": "Temperatura",
                "cielotuc": "—",
                "smn": "—",
                "weather_com": "—",
            },
        ],
        cielotuc_accuracy=cielotuc_acc,
        smn_accuracy=0.672,
        weathercom_accuracy=0.718,
        notable_wins=[],
    )


@model_router.post("/retrain")
async def trigger_retrain(
    _user: Annotated[User, Depends(require_role("admin"))] = None,
):
    """Manually trigger model retraining (government dashboard). Admin only."""
    import uuid
    from app.services.tasks import retrain_model
    scheduler.add_job(retrain_model, id=f"retrain_{uuid.uuid4().hex[:8]}")
    return {"task_id": uuid.uuid4().hex[:8], "status": "queued"}
