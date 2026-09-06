"""
scripts/train_initial_model.py
────────────────────────────────
Downloads historical data and trains the first CIELO·TUC model.

What it does:
  1. Downloads ERA5 reanalysis data for Tucumán (1981–2026)
     via the Copernicus CDS API (free with registration)
  2. Downloads NASA POWER hourly data for each station
  3. Combines into a unified training DataFrame
  4. Trains the CNN-LSTM model and logs to MLflow
  5. Updates the active ModelVersion in the DB

Prerequisites:
  pip install cdsapi  (for ERA5)
  ~/.cdsapirc file with your Copernicus API key

Usage:
  python scripts/train_initial_model.py [--years 10] [--zone capital]
"""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.ml.pipeline.data_pipeline import WeatherDataPipeline
from app.ml.training.trainer import ModelTrainer


# ── Tucumán centroid for bulk data download ────────────────────
TUC_LAT = -26.82
TUC_LNG = -65.22


async def download_nasa_power(years_back: int = 10) -> pd.DataFrame:
    """Download historical hourly data from NASA POWER (free, no key)."""
    import httpx
    from datetime import timedelta

    end = datetime.utcnow()
    start = end.replace(year=end.year - years_back)

    logger.info(f"Downloading NASA POWER: {start.date()} → {end.date()}")
    params = {
        "parameters": "T2M,T2M_MIN,T2M_MAX,RH2M,WS10M,WD10M,PS,PRECTOTCORR,ALLSKY_SFC_SW_DWN",
        "community": "RE",
        "longitude": TUC_LNG,
        "latitude": TUC_LAT,
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "format": "JSON",
        "time-standard": "UTC",
        "temporal-api": "hourly",
    }
    url = f"{settings.nasa_power_base_url}/hourly/point"

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()["properties"]["parameter"]

    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df.index, format="%Y%m%d%H")
    df.index.name = "timestamp"
    df = df.reset_index()

    df = df.rename(columns={
        "T2M": "temperature_c",
        "T2M_MIN": "temperature_min_c",
        "T2M_MAX": "temperature_max_c",
        "RH2M": "humidity_pct",
        "WS10M": "wind_speed_kmh",
        "WD10M": "wind_direction_deg",
        "PS": "pressure_hpa",
        "PRECTOTCORR": "precip_mm",
    })

    # Replace NASA fill value (-999) with NaN
    df = df.replace(-999.0, np.nan)
    logger.info(f"NASA POWER: {len(df):,} rows downloaded")
    return df


def download_era5_fallback(years_back: int = 10) -> pd.DataFrame:
    """
    Download ERA5 reanalysis via CDS API.
    Requires ~/.cdsapirc with API key from:
    https://cds.climate.copernicus.eu/user/register

    Falls back to NASA POWER data if CDS not configured.
    """
    try:
        import cdsapi
        c = cdsapi.Client()

        logger.info("Downloading ERA5 data via CDS API...")
        end_year = datetime.utcnow().year
        start_year = end_year - years_back

        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": [
                    "2m_temperature", "2m_dewpoint_temperature",
                    "surface_pressure", "10m_u_component_of_wind",
                    "10m_v_component_of_wind", "total_precipitation",
                    "convective_available_potential_energy",
                    "total_column_water_vapour",
                ],
                "year": [str(y) for y in range(start_year, end_year + 1)],
                "month": [f"{m:02d}" for m in range(1, 13)],
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": settings.tucuman_bbox,  # N, W, S, E
                "format": "netcdf",
            },
            "/tmp/era5_tucuman.nc",
        )
        logger.info("ERA5 download complete: /tmp/era5_tucuman.nc")

        # Parse netCDF → DataFrame
        import xarray as xr
        ds = xr.open_dataset("/tmp/era5_tucuman.nc")
        df = ds.to_dataframe().reset_index()
        df = df.rename(columns={
            "t2m": "temperature_c",
            "d2m": "dew_point_c",
            "sp": "pressure_hpa",
            "tp": "precip_mm",
            "cape": "cape_j_kg",
            "tcwv": "precipitable_water_mm",
        })
        # Convert Kelvin to Celsius
        for col in ["temperature_c", "dew_point_c"]:
            if col in df.columns:
                df[col] = df[col] - 273.15
        df["precip_mm"] = df["precip_mm"] * 1000  # m → mm
        df["pressure_hpa"] = df["pressure_hpa"] / 100  # Pa → hPa
        return df

    except ImportError:
        logger.warning("cdsapi not installed — ERA5 unavailable")
        return pd.DataFrame()
    except Exception as e:
        logger.warning(f"ERA5 download failed: {e} — falling back to NASA POWER")
        return pd.DataFrame()


def generate_synthetic_extremes(n_events: int = 200) -> pd.DataFrame:
    """
    Generate synthetic extreme weather events to augment training data.
    Critical for rare events like Zonda and hail that are
    underrepresented in historical records.

    Each event spans 6–24 hours and has realistic Tucumán values.
    """
    logger.info(f"Generating {n_events} synthetic extreme events for augmentation")
    rows = []
    rng = np.random.default_rng(42)

    for _ in range(n_events):
        event_type = rng.choice(
            ["zonda", "hail_storm", "heat_wave", "extreme_rain"],
            p=[0.25, 0.25, 0.25, 0.25],
        )
        duration = rng.integers(6, 25)
        start = pd.Timestamp("2000-01-01") + pd.Timedelta(
            days=int(rng.integers(0, 365 * 24))
        )

        for h in range(duration):
            t = start + pd.Timedelta(hours=h)
            if event_type == "zonda":
                row = {
                    "timestamp": t,
                    "temperature_c": rng.uniform(35, 45),
                    "humidity_pct": rng.uniform(5, 20),
                    "wind_speed_kmh": rng.uniform(60, 110),
                    "pressure_hpa": rng.uniform(990, 1000),
                    "precip_mm": 0.0,
                    "cape_j_kg": 0.0,
                    "cordillera_pressure_hpa": rng.uniform(870, 885),
                    "thermal_differential_c": rng.uniform(15, 28),
                    "_label_zonda": 1.0,
                    "_label_storm": 0.0,
                    "_label_hail": 0.0,
                }
            elif event_type == "hail_storm":
                row = {
                    "timestamp": t,
                    "temperature_c": rng.uniform(18, 28),
                    "humidity_pct": rng.uniform(70, 95),
                    "wind_speed_kmh": rng.uniform(40, 80),
                    "pressure_hpa": rng.uniform(995, 1008),
                    "precip_mm": rng.uniform(10, 50) if h > duration // 2 else 0,
                    "cape_j_kg": rng.uniform(2000, 5000),
                    "cordillera_pressure_hpa": rng.uniform(895, 910),
                    "thermal_differential_c": rng.uniform(5, 12),
                    "_label_zonda": 0.0,
                    "_label_storm": 1.0,
                    "_label_hail": 1.0,
                }
            elif event_type == "heat_wave":
                row = {
                    "timestamp": t,
                    "temperature_c": rng.uniform(38, 44),
                    "humidity_pct": rng.uniform(15, 35),
                    "wind_speed_kmh": rng.uniform(5, 25),
                    "pressure_hpa": rng.uniform(1005, 1015),
                    "precip_mm": 0.0,
                    "cape_j_kg": rng.uniform(100, 800),
                    "cordillera_pressure_hpa": rng.uniform(895, 908),
                    "thermal_differential_c": rng.uniform(8, 16),
                    "_label_zonda": 0.0,
                    "_label_storm": 0.0,
                    "_label_hail": 0.0,
                }
            else:  # extreme_rain
                row = {
                    "timestamp": t,
                    "temperature_c": rng.uniform(20, 30),
                    "humidity_pct": rng.uniform(85, 100),
                    "wind_speed_kmh": rng.uniform(20, 55),
                    "pressure_hpa": rng.uniform(998, 1010),
                    "precip_mm": rng.uniform(20, 80),
                    "cape_j_kg": rng.uniform(1500, 4000),
                    "cordillera_pressure_hpa": rng.uniform(900, 915),
                    "thermal_differential_c": rng.uniform(3, 8),
                    "_label_zonda": 0.0,
                    "_label_storm": 1.0,
                    "_label_hail": 0.0,
                }
            rows.append(row)

    return pd.DataFrame(rows)


async def update_active_model(version: str, run_id: str, metrics: dict):
    """Swap the active model version in the DB."""
    from app.db.session import AsyncSessionLocal
    from app.models.weather import ModelVersion
    from sqlalchemy import select, update

    async with AsyncSessionLocal() as db:
        # Deactivate current
        await db.execute(
            update(ModelVersion).where(ModelVersion.is_active == True)
            .values(is_active=False)
        )
        # Insert new
        mv = ModelVersion(
            version=version,
            mlflow_run_id=run_id,
            trained_at=datetime.utcnow(),
            architecture="CNN-LSTM",
            training_samples=metrics.get("training_samples", 0),
            accuracy_overall=1.0 - metrics.get("best_val_loss", 0.5),
            is_active=True,
            notes=f"Initial training. Val loss: {metrics.get('best_val_loss', '?')}",
        )
        db.add(mv)
        await db.commit()
        logger.info(f"✅ Model v{version} is now active in the DB")


async def main(years_back: int = 10, use_era5: bool = False):
    logger.info("=" * 60)
    logger.info("CIELO·TUC — Initial Model Training")
    logger.info("=" * 60)

    # ── 1. Download data ──────────────────────────────────────
    logger.info(f"Step 1/5: Downloading {years_back} years of historical data")

    if use_era5:
        df_main = download_era5_fallback(years_back)
    else:
        df_main = pd.DataFrame()

    if df_main.empty:
        logger.info("Using NASA POWER as primary data source")
        df_main = await download_nasa_power(years_back)

    logger.info(f"Historical data: {len(df_main):,} rows")

    # ── 2. Augment with synthetic extremes ────────────────────
    logger.info("Step 2/5: Augmenting with synthetic extreme events")
    df_synthetic = generate_synthetic_extremes(n_events=500)
    df_combined = pd.concat([df_main, df_synthetic], ignore_index=True)
    df_combined = df_combined.sort_values("timestamp").reset_index(drop=True)
    logger.info(f"Combined dataset: {len(df_combined):,} rows")

    # ── 3. Preprocess ─────────────────────────────────────────
    logger.info("Step 3/5: Feature engineering and preprocessing")
    pipeline = WeatherDataPipeline()
    df_processed = pipeline.preprocess(df_combined)
    pipeline.fit_scaler(df_processed)

    # Save scaler for inference
    pipeline.save_scaler()
    logger.info("Scaler saved to models/scaler.pkl")

    # ── 4. Build tensors ──────────────────────────────────────
    logger.info("Step 4/5: Building training tensors")
    X, y = pipeline.build_training_tensors(df_processed)
    logger.info(f"X shape: {X.shape} | y horizons: {list(y.keys())}")

    # ── 5. Train ──────────────────────────────────────────────
    logger.info("Step 5/5: Training CNN-LSTM model")
    trainer = ModelTrainer(
        max_epochs=100,
        batch_size=128,
        lr=1e-3,
    )
    version = f"1.0-{datetime.utcnow().strftime('%Y%m')}"
    metrics = trainer.train(X, y, version=version)
    metrics["training_samples"] = len(X)

    logger.info(f"Training complete: {metrics}")

    # ── Update DB ─────────────────────────────────────────────
    await update_active_model(version, metrics["run_id"], metrics)

    logger.info("\n" + "=" * 60)
    logger.info("✅ Initial training complete!")
    logger.info(f"   Model version: {version}")
    logger.info(f"   MLflow run: {metrics['run_id']}")
    logger.info(f"   Best val loss: {metrics.get('best_val_loss', '?'):.4f}")
    logger.info("   View in MLflow: http://localhost:5000")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train initial CIELO·TUC model")
    parser.add_argument(
        "--years", type=int, default=10,
        help="Years of historical data to use (default: 10)"
    )
    parser.add_argument(
        "--era5", action="store_true",
        help="Use ERA5 data if CDS API key is configured"
    )
    args = parser.parse_args()
    asyncio.run(main(years_back=args.years, use_era5=args.era5))
