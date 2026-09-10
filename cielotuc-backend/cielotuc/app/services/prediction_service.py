"""
app/services/prediction_service.py
────────────────────────────────────
Inference service — bridges the trained model and the API layer.

Responsibilities:
  - Load and cache the active model version
  - Build the input window from the latest sensor readings
  - Run forward pass and post-process outputs
  - Compute the Zonda risk index
  - Store predictions in the database
  - Trigger FLOOD·TUC alerts when thresholds are exceeded
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

import torch
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.ml.models.cnn_lstm import CnnLstmWeatherModel, FastWeatherModel, FastWeatherModelV5, WeatherModelV3
from app.ml.pipeline.data_pipeline import WeatherDataPipeline, FEATURE_COLS
from app.models.weather import (
    AIPrediction, ModelVersion, SensorReading, WeatherStation, Zone
)
from app.schemas.weather import (
    CurrentConditions, DailyForecast, ForecastResponse,
    HourlyForecast, ZondaIndex, PredictionOut
)
from app.services.flood_alert_service import FloodAlertService


CONDITION_MAP = {
    (0.0, 0.2): "sunny",
    (0.2, 0.4): "partly_cloudy",
    (0.4, 0.6): "cloudy",
    (0.6, 0.75): "rain",
    (0.75, 1.01): "storm",
}

DAY_NAMES_ES = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]


def rain_prob_to_condition(prob: float) -> str:
    for (lo, hi), cond in CONDITION_MAP.items():
        if lo <= prob < hi:
            return cond
    return "storm"


class PredictionService:
    """
    Singleton-style service. Instantiate once at app startup,
    reuse across requests.
    """

    def __init__(self):
        self._model: Optional[CnnLstmWeatherModel] = None
        self._pipeline = WeatherDataPipeline()
        self._pipeline.load_scaler()
        self._flood_service = FloodAlertService()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._active_version: Optional[str] = None
        self._feat_cols: Optional[list[str]] = None

    async def load_model(self, db: AsyncSession) -> None:
        """Load the currently active model version. Tries local checkpoint first, then MLflow."""
        from pathlib import Path

        result = await db.execute(
            select(ModelVersion).where(ModelVersion.is_active == True)
        )
        mv = result.scalars().first()
        if mv is None:
            logger.warning("No active model version found — predictions unavailable")
            return

        self._active_version = mv.version
        version_str = mv.version

        # Try local checkpoint first (for v2.0+ models)
        models_dir = Path("models")
        ckpt_path = models_dir / f"cielotuc_v{version_str}.pt"
        if ckpt_path.exists():
            try:
                ckpt = torch.load(ckpt_path, map_location=self._device, weights_only=False)
                n_features = ckpt.get("n_features", 34)
                n_timesteps = ckpt.get("n_timesteps", settings.model_lookback_hours)
                horizons = ckpt.get("horizons", settings.model_forecast_horizons)

                # Choose model class based on saved metadata
                sd_keys = set(ckpt.get("model_state_dict", {}).keys())
                model_class = ckpt.get("model_class", "")
                is_v3 = "se_expand" in sd_keys or model_class == "WeatherModelV3"
                is_v5 = model_class == "FastWeatherModelV5"

                # Fallback: detect v5.0 vs v2.0 by head layer output size (56 vs 48)
                if not is_v3 and not is_v5:
                    head_keys = [k for k in sd_keys if k.startswith("heads.") and k.endswith(".0.weight")]
                    if head_keys:
                        is_v5 = ckpt["model_state_dict"][head_keys[0]].shape[0] == 56

                if is_v3 and n_timesteps <= 24:
                    self._model = WeatherModelV3(
                        n_features=n_features, n_timesteps=n_timesteps, horizons=horizons,
                    ).to(self._device)
                elif is_v5:
                    self._model = FastWeatherModelV5(
                        n_features=n_features, n_timesteps=n_timesteps, horizons=horizons,
                    ).to(self._device)
                elif n_features > 34 or n_timesteps <= 24:
                    self._model = FastWeatherModel(
                        n_features=n_features, n_timesteps=n_timesteps, horizons=horizons,
                    ).to(self._device)
                else:
                    self._model = CnnLstmWeatherModel(
                        n_features=n_features, n_timesteps=n_timesteps, horizons=horizons,
                    ).to(self._device)

                self._model.load_state_dict(ckpt["model_state_dict"])
                self._model.eval()

                # Save feat_cols for exact column ordering at inference
                self._feat_cols = ckpt.get("feat_cols")

                # Update pipeline lookback to match model
                self._pipeline = WeatherDataPipeline(lookback_hours=n_timesteps)
                self._pipeline.load_scaler()

                logger.info(f"Loaded model v{version_str} from {ckpt_path.name} "
                           f"({n_features} features, {n_timesteps}h lookback, "
                           f"{self._model.n_parameters:,} params)")
                return
            except Exception as e:
                logger.error(f"Local checkpoint load failed: {e}")

        # Fallback: try MLflow
        try:
            import mlflow.pytorch
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            self._model = CnnLstmWeatherModel(
                n_features=34,
                n_timesteps=settings.model_lookback_hours,
                horizons=settings.model_forecast_horizons,
            ).to(self._device)
            self._model = mlflow.pytorch.load_model(
                f"runs:/{mv.mlflow_run_id}/model",
                map_location=self._device,
            )
            logger.info(f"Loaded model v{version_str} from MLflow")
        except Exception as e:
            logger.error(f"MLflow load failed: {e}. Model not available.")
            self._model = None

    async def get_forecast(
        self,
        zone_id: int,
        db: AsyncSession,
    ) -> Optional[ForecastResponse]:
        """
        Generate the full forecast for a zone.
        Returns None if the model is not loaded or data is insufficient.
        """
        if self._model is None:
            return None

        # ── 1. Fetch last 72h of readings for this zone ──────
        zone = await db.get(Zone, zone_id)
        if zone is None:
            return None

        readings = await self._get_recent_readings(zone_id, db)
        if len(readings) < settings.model_lookback_hours:
            logger.warning(
                f"Zone {zone_id}: only {len(readings)} readings, need "
                f"{settings.model_lookback_hours}. Using synthetic padding."
            )
            readings = self._pad_readings(readings)

        # ── 2. Preprocess ─────────────────────────────────────
        import pandas as pd
        df = pd.DataFrame([
            {
                "timestamp": r.timestamp,
                "temperature_c": r.temperature_c or 20.0,
                "humidity_pct": r.humidity_pct or 60.0,
                "pressure_hpa": r.pressure_hpa or 1010.0,
                "precip_mm": r.precip_mm or 0.0,
                "wind_speed_kmh": r.wind_speed_kmh or 10.0,
                "wind_direction_deg": r.wind_direction_deg or 90.0,
                "wind_gust_kmh": r.wind_gust_kmh or 15.0,
                "cape_j_kg": r.cape_j_kg or 0.0,
                "cordillera_pressure_hpa": r.cordillera_pressure_hpa or 890.0,
                "thermal_differential_c": r.thermal_differential_c or 3.0,
                "enso_index": r.enso_index or 0.0,
            }
            for r in readings
        ])

        df = self._pipeline.preprocess(
            df,
            altitude_m=zone.altitude_m or 450.0,
            impermeable_pct=zone.impermeable_pct or 0.5,
            is_mountain=zone.is_mountain,
        )

        # ── 3. Run model inference ─────────────────────────────
        # Add enhanced features if model uses v2.0+ (48 features)
        n_expected = self._model.cnn[0].in_channels if hasattr(self._model, 'cnn') else 34
        use_scale = n_expected <= 34  # v1.0 was trained scaled, v2.0 on raw data
        if n_expected > 34:
            df = self._pipeline.add_enhanced_features(df)

        x = self._pipeline.build_inference_window(
            df, scale=use_scale, columns=self._feat_cols,
        )
        x = x.to(self._device)

        self._model.eval()
        with torch.no_grad():
            raw_preds = self._model(x)

        # ── 4. Post-process output ─────────────────────────────
        now = datetime.now(timezone.utc)
        latest_row = readings[-1]

        current = CurrentConditions(
            zone_id=zone_id,
            zone_name=zone.name,
            timestamp=now,
            temperature_c=latest_row.temperature_c or 20.0,
            feels_like_c=latest_row.feels_like_c,
            humidity_pct=latest_row.humidity_pct or 60.0,
            pressure_hpa=latest_row.pressure_hpa or 1010.0,
            wind_speed_kmh=latest_row.wind_speed_kmh or 10.0,
            wind_direction_deg=latest_row.wind_direction_deg,
            wind_gust_kmh=latest_row.wind_gust_kmh,
            precip_1h_mm=latest_row.precip_mm or 0.0,
            precip_24h_mm=latest_row.precip_24h_mm or 0.0,
            condition=rain_prob_to_condition(
                float(raw_preds["3"]["rain_probability"][0].cpu())
            ),
            ai_confidence=float(raw_preds["3"]["rain_probability"][0].cpu()),
            zonda_risk_score=float(raw_preds["3"]["zonda_risk"][0].cpu()) * 100,
        )

        hourly = []
        for h in [3, 6, 12]:
            key = str(h)
            p = raw_preds[key]
            rain_prob = float(p["rain_probability"][0].cpu())
            hourly.append(HourlyForecast(
                target_time=now + timedelta(hours=h),
                horizon_hours=h,
                temperature_c=float(p["temperature_c"][0].cpu()),
                rain_probability=rain_prob,
                precip_mm_expected=float(p["precip_mm"][0].cpu()),
                wind_speed_kmh=float(p["wind_speed_kmh"][0].cpu()),
                condition=rain_prob_to_condition(rain_prob),
                confidence=0.9,
            ))

        daily = []
        for i, h in enumerate([24, 48, 72, 96, 120, 144, 168]):
            key = str(min(h, max(settings.model_forecast_horizons)))
            p = raw_preds[key]
            rain_prob = float(p["rain_probability"][0].cpu())
            target_dt = now + timedelta(hours=h)
            daily.append(DailyForecast(
                date=target_dt.strftime("%Y-%m-%d"),
                day_name=DAY_NAMES_ES[target_dt.weekday()],
                condition=rain_prob_to_condition(rain_prob),
                temp_min_c=float(p["temperature_c"][0].cpu()) - 4,
                temp_max_c=float(p["temperature_c"][0].cpu()) + 4,
                rain_probability=rain_prob,
                precip_mm_expected=float(p["precip_mm"][0].cpu()),
                ai_confidence=0.85,
                hail_risk=float(p["hail_risk"][0].cpu()),
                zonda_risk=float(p["zonda_risk"][0].cpu()),
                storm_risk=float(p["storm_risk"][0].cpu()),
            ))

        # ── 5. Persist predictions ────────────────────────────
        await self._save_predictions(zone_id, raw_preds, now, db)

        # ── 6. Check FLOOD·TUC threshold ──────────────────────
        max_precip = max(
            float(raw_preds[str(h)]["precip_mm"][0].cpu())
            for h in settings.model_forecast_horizons
        )
        if max_precip >= settings.floodtuc_alert_threshold_mm:
            await self._flood_service.send_alert(
                zone_id=zone_id,
                rain_probability=float(
                    raw_preds["24"]["rain_probability"][0].cpu()
                ),
                expected_precip_mm=max_precip,
                db=db,
            )

        result = await db.execute(
            select(ModelVersion).where(ModelVersion.is_active == True)
        )
        mv = result.scalars().first()

        return ForecastResponse(
            zone_id=zone_id,
            zone_name=zone.name,
            generated_at=now,
            model_version=mv.version if mv else "unknown",
            current=current,
            hourly=hourly,
            daily=daily,
        )

    async def get_zonda_index(
        self, zone_id: int, db: AsyncSession
    ) -> ZondaIndex:
        """Calculate the current Zonda risk index for a zone."""
        readings = await self._get_recent_readings(zone_id, db, hours=6)
        if not readings:
            return ZondaIndex(
                timestamp=datetime.now(timezone.utc),
                risk_score=0.0,
                risk_level="none",
                forecast_24h_probability=0.0,
            )

        latest = readings[-1]
        diff = latest.thermal_differential_c or 3.0
        wind = latest.wind_speed_kmh or 10.0
        humidity = latest.humidity_pct or 60.0

        # Heuristic score: high temp differential, high wind, low humidity
        score = (
            min(diff / 20.0, 1.0) * 40
            + min(wind / 100.0, 1.0) * 35
            + (1 - min(humidity / 100.0, 1.0)) * 25
        )

        if score < 20:
            level = "none"
        elif score < 40:
            level = "low"
        elif score < 60:
            level = "medium"
        elif score < 80:
            level = "high"
        else:
            level = "active"

        return ZondaIndex(
            timestamp=datetime.now(timezone.utc),
            risk_score=round(score, 1),
            risk_level=level,
            cordillera_pressure_hpa=latest.cordillera_pressure_hpa,
            thermal_differential_c=latest.thermal_differential_c,
            descending_speed_kmh=latest.wind_speed_kmh,
            forecast_24h_probability=min(score / 100.0, 1.0),
        )

    # ── Helpers ───────────────────────────────────────────────

    async def _get_recent_readings(
        self,
        zone_id: int,
        db: AsyncSession,
        hours: int | None = None,
    ) -> list[SensorReading]:
        hours = hours or settings.model_lookback_hours
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        result = await db.execute(
            select(SensorReading)
            .join(WeatherStation)
            .where(
                WeatherStation.zone_id == zone_id,
                SensorReading.timestamp >= cutoff,
            )
            .order_by(SensorReading.timestamp.asc())
        )
        return result.scalars().all()

    def _pad_readings(self, readings: list) -> list:
        """Repeat the oldest available reading to fill the lookback window."""
        if not readings:
            return readings
        needed = settings.model_lookback_hours - len(readings)
        return [readings[0]] * needed + list(readings)

    async def _save_predictions(
        self,
        zone_id: int,
        raw_preds: dict,
        generated_at: datetime,
        db: AsyncSession,
    ) -> None:
        result = await db.execute(
            select(ModelVersion).where(ModelVersion.is_active == True)
        )
        mv = result.scalars().first()
        if mv is None:
            return

        for h in settings.model_forecast_horizons:
            key = str(h)
            p = raw_preds[key]
            pred = AIPrediction(
                zone_id=zone_id,
                model_version_id=mv.id,
                generated_at=generated_at,
                target_time=generated_at + timedelta(hours=h),
                horizon_hours=h,
                rain_probability=float(p["rain_probability"][0].cpu()),
                precip_mm_expected=float(p["precip_mm"][0].cpu()),
                temperature_c=float(p["temperature_c"][0].cpu()),
                wind_speed_kmh=float(p["wind_speed_kmh"][0].cpu()),
                condition=rain_prob_to_condition(
                    float(p["rain_probability"][0].cpu())
                ),
                confidence=0.88,
                zonda_risk=float(p["zonda_risk"][0].cpu()),
                storm_risk=float(p["storm_risk"][0].cpu()),
                hail_risk=float(p["hail_risk"][0].cpu()),
            )
            db.add(pred)
        await db.flush()
