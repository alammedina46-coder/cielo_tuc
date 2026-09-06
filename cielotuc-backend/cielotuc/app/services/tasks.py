"""
app/services/tasks.py
──────────────────────
Background tasks using APScheduler (no Redis/Celery required).

Runs inside the FastAPI lifespan — zero external infra needed.
"""

import asyncio
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger

scheduler = AsyncIOScheduler(timezone="America/Argentina/Tucuman")


def _run_async(coro):
    """Run an async coroutine from a sync context."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── Task implementations ────────────────────────────────────────


async def fetch_weather_data():
    """Pull fresh observations from SMN + NASA POWER + Windy."""
    import pandas as pd
    from app.db.session import AsyncSessionLocal
    from app.ml.pipeline.data_pipeline import WeatherDataPipeline
    from app.models.weather import SensorReading, WeatherStation
    from app.services.windy_client import WindyClient
    from sqlalchemy import select

    logger.info("Task: fetch_weather_data started")
    pipeline = WeatherDataPipeline()

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(WeatherStation).where(WeatherStation.is_active == True)
        )
        stations = result.scalars().all()

        for station in stations:
            try:
                df = pd.DataFrame()
                if station.source == "smn":
                    df = await pipeline.fetch_smn()
                else:
                    df = await pipeline.fetch_nasa_power(
                        station.latitude, station.longitude
                    )

                windy = await pipeline.fetch_windy(
                    station.latitude, station.longitude
                )
                df = WindyClient.merge_into(df, windy)

                if df.empty:
                    continue

                for _, row in df.iterrows():
                    reading = SensorReading(
                        station_id=station.id,
                        timestamp=row.get("timestamp"),
                        temperature_c=row.get("temperature_c"),
                        humidity_pct=row.get("humidity_pct"),
                        pressure_hpa=row.get("pressure_hpa"),
                        precip_mm=row.get("precip_mm", 0.0),
                        wind_speed_kmh=row.get("wind_speed_kmh"),
                        wind_direction_deg=row.get("wind_direction_deg"),
                        cape_j_kg=row.get("cape_j_kg"),
                        cordillera_pressure_hpa=row.get("cordillera_pressure_hpa"),
                        thermal_differential_c=row.get("thermal_differential_c"),
                    )
                    db.add(reading)

                station.last_seen = datetime.now(timezone.utc)
                await db.flush()

            except Exception as e:
                logger.warning(f"Station {station.station_code}: {e}")

        await db.commit()
        logger.info(f"Ingested data for {len(stations)} stations")


async def run_predictions():
    """Run the AI model for all active zones."""
    from app.db.session import AsyncSessionLocal
    from app.models.weather import Zone
    from app.services.prediction_service import PredictionService
    from sqlalchemy import select

    logger.info("Task: run_predictions started")
    service = PredictionService()
    async with AsyncSessionLocal() as db:
        await service.load_model(db)
        if service._model is None:
            logger.warning("No model loaded — skipping predictions")
            return

        result = await db.execute(select(Zone))
        zones = result.scalars().all()
        logger.info(f"Running predictions for {len(zones)} zones")

        for zone in zones:
            try:
                await service.get_forecast(zone.id, db)
            except Exception as e:
                logger.warning(f"Prediction failed for zone {zone.id}: {e}")

        await db.commit()


async def validate_predictions():
    """Compare past predictions vs actual sensor readings."""
    from app.db.session import AsyncSessionLocal
    from app.models.weather import AIPrediction, SensorReading, WeatherStation
    from sqlalchemy import and_, select

    now = datetime.now(timezone.utc)
    logger.info("Task: validate_predictions started")

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(AIPrediction).where(
                and_(
                    AIPrediction.was_confirmed == None,
                    AIPrediction.target_time <= now,
                    AIPrediction.target_time >= now - timedelta(hours=48),
                )
            )
        )
        preds = result.scalars().all()
        logger.info(f"Validating {len(preds)} predictions")

        for pred in preds:
            result = await db.execute(
                select(SensorReading)
                .join(WeatherStation)
                .where(
                    and_(
                        WeatherStation.zone_id == pred.zone_id,
                        SensorReading.timestamp >= pred.target_time - timedelta(hours=1),
                        SensorReading.timestamp <= pred.target_time + timedelta(hours=1),
                    )
                )
                .order_by(SensorReading.timestamp.asc())
                .limit(1)
            )
            actual = result.scalars().first()
            if actual:
                actual_rain = (actual.precip_mm or 0.0) > 1.0
                pred.was_confirmed = actual_rain
                pred.actual_precip_mm = actual.precip_mm

        await db.commit()
        logger.info("Prediction validation complete")


async def retrain_model():
    """Monthly model retraining."""
    from app.db.session import AsyncSessionLocal
    from app.ml.pipeline.data_pipeline import WeatherDataPipeline
    from app.ml.training.trainer import ModelTrainer
    from app.models.weather import ModelVersion, SensorReading
    from sqlalchemy import select

    logger.info("Task: retrain_model started")
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(SensorReading)
            .where(SensorReading.is_validated == True)
            .order_by(SensorReading.timestamp.asc())
        )
        readings = result.scalars().all()

    if len(readings) < 1000:
        logger.warning(f"Only {len(readings)} validated readings — skipping retrain")
        return

    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "timestamp": r.timestamp,
                "temperature_c": r.temperature_c or 20.0,
                "humidity_pct": r.humidity_pct or 60.0,
                "pressure_hpa": r.pressure_hpa or 1010.0,
                "precip_mm": r.precip_mm or 0.0,
                "wind_speed_kmh": r.wind_speed_kmh or 10.0,
                "wind_direction_deg": r.wind_direction_deg or 90.0,
                "cape_j_kg": r.cape_j_kg or 0.0,
                "cordillera_pressure_hpa": r.cordillera_pressure_hpa or 890.0,
                "thermal_differential_c": r.thermal_differential_c or 3.0,
                "enso_index": r.enso_index or 0.0,
            }
            for r in readings
        ]
    )

    pipeline = WeatherDataPipeline()
    df = pipeline.preprocess(df)
    pipeline.fit_scaler(df)
    pipeline.save_scaler()
    X, y = pipeline.build_training_tensors(df)

    trainer = ModelTrainer(max_epochs=80)
    version_str = datetime.now(timezone.utc).strftime("%Y%m")
    metrics = trainer.train(X, y, version=version_str)

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(ModelVersion).where(ModelVersion.is_active == True)
        )
        current = result.scalars().first()
        current_accuracy = current.accuracy_overall if current else 0.0
        new_accuracy = 1.0 - metrics.get("best_val_loss", 1.0)

        if new_accuracy > (current_accuracy or 0.0):
            logger.info(f"New model ({new_accuracy:.3f}) better than current ({current_accuracy:.3f})")
            if current:
                current.is_active = False
            new_mv = ModelVersion(
                version=version_str,
                mlflow_run_id=metrics.get("run_id", "local"),
                trained_at=datetime.now(timezone.utc),
                training_samples=len(X),
                accuracy_overall=new_accuracy,
                is_active=True,
                notes=f"Auto-retrain. Val loss: {metrics['best_val_loss']:.4f}",
            )
            db.add(new_mv)
            await db.commit()
            logger.info(f"New model v{version_str} is now active")


def start_scheduler():
    """Register all periodic jobs and start the scheduler."""
    scheduler.add_job(
        fetch_weather_data,
        IntervalTrigger(minutes=15),
        id="fetch_weather_data",
        name="Fetch weather data (15min)",
        replace_existing=True,
    )
    scheduler.add_job(
        run_predictions,
        IntervalTrigger(minutes=30),
        id="run_predictions",
        name="Run predictions (30min)",
        replace_existing=True,
    )
    scheduler.add_job(
        validate_predictions,
        IntervalTrigger(minutes=30),
        id="validate_predictions",
        name="Validate predictions (30min)",
        replace_existing=True,
    )
    scheduler.add_job(
        retrain_model,
        CronTrigger(day=1, hour=2, minute=0),
        id="retrain_model",
        name="Monthly retrain (1st 02:00)",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("APScheduler started — 4 jobs registered")


def stop_scheduler():
    """Gracefully shut down the scheduler."""
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("APScheduler stopped")
