"""
app/api/v1/endpoints/forecast.py
──────────────────────────────────
Weather forecast and current conditions endpoints.
"""
import random
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.weather import (
    CurrentConditions, DailyForecast, ForecastResponse, HourlyForecast, ZondaIndex,
)
from app.services.prediction_service import PredictionService

router = APIRouter(prefix="/forecast", tags=["Forecast"])

_service = PredictionService()

DAY_NAMES_ES = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]

# Seasonal temperature ranges for Tucumán (month → min, max °C)
_SEASONAL = {
    1: (22, 34), 2: (21, 33), 3: (19, 31), 4: (15, 27),
    5: (10, 23), 6: (7, 20), 7: (6, 20), 8: (8, 23),
    9: (12, 27), 10: (16, 30), 11: (19, 32), 12: (21, 34),
}


def _generate_fallback_forecast(zone_id: int, zone_name: str, altitude: float) -> ForecastResponse:
    """Generate a reasonable synthetic forecast when the ML model isn't available."""
    now = datetime.now(timezone.utc)
    month = now.month
    t_min, t_max = _SEASONAL.get(month, (15, 30))
    # Altitude adjustment: cooler at higher altitudes
    alt_offset = max(0, (altitude - 450)) * -0.006
    t_min += alt_offset
    t_max += alt_offset
    base_temp = (t_min + t_max) / 2

    current = CurrentConditions(
        zone_id=zone_id,
        zone_name=zone_name,
        timestamp=now,
        temperature_c=round(base_temp + random.uniform(-1, 1), 1),
        feels_like_c=round(base_temp + random.uniform(-2, 2), 1),
        humidity_pct=round(random.uniform(40, 80), 1),
        pressure_hpa=round(random.uniform(1008, 1018), 1),
        wind_speed_kmh=round(random.uniform(5, 25), 1),
        wind_direction_deg=round(random.uniform(0, 360), 1),
        wind_gust_kmh=round(random.uniform(10, 40), 1),
        precip_1h_mm=0.0,
        precip_24h_mm=0.0,
        visibility_km=10.0,
        cloud_cover_pct=round(random.uniform(10, 60), 1),
        condition="partly_cloudy",
        ai_confidence=0.55,
        zonda_risk_score=round(random.uniform(0, 15), 1),
    )

    hourly = []
    for h in [3, 6, 12]:
        rain_p = round(random.uniform(0.0, 0.4), 2)
        hourly.append(HourlyForecast(
            target_time=now + timedelta(hours=h),
            horizon_hours=h,
            temperature_c=round(base_temp + random.uniform(-3, 3), 1),
            rain_probability=rain_p,
            precip_mm_expected=round(random.uniform(0, 5) if rain_p > 0.3 else 0, 1),
            wind_speed_kmh=round(random.uniform(5, 30), 1),
            condition="rain" if rain_p > 0.5 else "partly_cloudy",
            confidence=0.5,
        ))

    daily = []
    for i, h in enumerate([24, 48, 72, 96, 120, 144, 168]):
        target_dt = now + timedelta(hours=h)
        rain_p = round(random.uniform(0.0, 0.5), 2)
        daily.append(DailyForecast(
            date=target_dt.strftime("%Y-%m-%d"),
            day_name=DAY_NAMES_ES[target_dt.weekday()],
            condition="rain" if rain_p > 0.5 else "partly_cloudy",
            temp_min_c=round(t_min + random.uniform(-2, 2), 1),
            temp_max_c=round(t_max + random.uniform(-2, 2), 1),
            rain_probability=rain_p,
            precip_mm_expected=round(random.uniform(0, 8) if rain_p > 0.3 else 0, 1),
            ai_confidence=0.4,
            hail_risk=round(random.uniform(0, 0.05), 3),
            zonda_risk=round(random.uniform(0, 0.1), 3),
            storm_risk=round(random.uniform(0, 0.15), 3),
        ))

    return ForecastResponse(
        zone_id=zone_id,
        zone_name=zone_name,
        generated_at=now,
        model_version="synthetic-fallback",
        current=current,
        hourly=hourly,
        daily=daily,
    )


@router.get("/{zone_id}", response_model=ForecastResponse)
async def get_forecast(zone_id: int, db: AsyncSession = Depends(get_db)):
    """
    Full forecast for a zone: current conditions + hourly 12h + daily 7d.
    Falls back to synthetic data if ML model is not available.
    """
    from app.models.weather import Zone
    zone = await db.get(Zone, zone_id)
    if zone is None:
        raise HTTPException(status_code=404, detail="Zone not found")

    # Try ML model first
    try:
        await _service.load_model(db)
        forecast = await _service.get_forecast(zone_id, db)
        if forecast is not None:
            return forecast
    except Exception:
        pass

    # Fallback: synthetic forecast based on zone + season
    return _generate_fallback_forecast(
        zone_id, zone.name, zone.altitude_m or 450.0
    )


@router.get("/{zone_id}/zonda", response_model=ZondaIndex)
async def get_zonda_index(zone_id: int, db: AsyncSession = Depends(get_db)):
    """Current Zonda wind risk index for a zone."""
    try:
        return await _service.get_zonda_index(zone_id, db)
    except Exception:
        now = datetime.now(timezone.utc)
        return ZondaIndex(
            timestamp=now,
            risk_score=0.0,
            risk_level="none",
            forecast_24h_probability=0.0,
        )
