"""
app/api/v1/endpoints/forecast.py
──────────────────────────────────
Weather forecast and current conditions endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.weather import ForecastResponse, ZondaIndex, CurrentConditions
from app.services.prediction_service import PredictionService

router = APIRouter(prefix="/forecast", tags=["Forecast"])

# Module-level service instance (loaded at startup)
_service = PredictionService()


@router.get("/{zone_id}", response_model=ForecastResponse)
async def get_forecast(zone_id: int, db: AsyncSession = Depends(get_db)):
    """
    Full forecast for a zone: current conditions + hourly 12h + daily 7d.
    Used by both the citizen and government views.
    """
    await _service.load_model(db)
    forecast = await _service.get_forecast(zone_id, db)
    if forecast is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not available or insufficient data for this zone",
        )
    return forecast


@router.get("/{zone_id}/zonda", response_model=ZondaIndex)
async def get_zonda_index(zone_id: int, db: AsyncSession = Depends(get_db)):
    """Current Zonda wind risk index for a zone."""
    return await _service.get_zonda_index(zone_id, db)
