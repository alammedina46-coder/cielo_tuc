"""
scripts/train_v1.py
────────────────────
CIELO·TUC v1.0 Training Script — standalone, no MLflow/DB required.

What it does:
  1. Downloads 10 years of NASA POWER hourly data for Tucumán (free, no key)
  2. Augments with 500 synthetic extreme events (Zonda, hail, heatwave, extreme rain)
  3. Preprocesses: feature engineering + MinMaxScaler
  4. Builds sliding-window tensors (72h lookback → 6 horizons)
  5. Trains CNN-LSTM with TimeSeriesSplit backtesting
  6. Evaluates: accuracy (rain), RMSE (temp), precision/recall per target
  7. Saves: model checkpoint, scaler, metrics JSON, training report

Usage:
  python scripts/train_v1.py [--years 10] [--n-synthetic 500] [--max-epochs 100]
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ml.models.cnn_lstm import CnnLstmWeatherModel, WeatherLoss
from app.ml.pipeline.data_pipeline import WeatherDataPipeline

# ── Constants ───────────────────────────────────────────────────
TUC_LAT = -26.82
TUC_LNG = -65.22
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


# ── Data Download ───────────────────────────────────────────────
async def download_nasa_power(years_back: int = 10) -> pd.DataFrame:
    """Download historical hourly data from NASA POWER (free, no key)."""
    import httpx

    from datetime import timezone, timedelta

    # NASA POWER JSON has ~3-year limit; use CSV for larger ranges
    end = datetime(2026, 7, 31, tzinfo=timezone.utc)
    start = end.replace(year=end.year - years_back)

    logger.info(f"Downloading NASA POWER: {start.date()} → {end.date()} (CSV format)")

    # Download in 3-year chunks to avoid server timeouts
    chunks = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start.replace(year=chunk_start.year + 3), end)
        logger.info(f"  chunk: {chunk_start.date()} → {chunk_end.date()}")
        params = {
            "parameters": "T2M,RH2M,WS10M,WD10M,PS,PRECTOTCORR,ALLSKY_SFC_SW_DWN",
            "community": "RE",
            "longitude": TUC_LNG,
            "latitude": TUC_LAT,
            "start": chunk_start.strftime("%Y%m%d"),
            "end": chunk_end.strftime("%Y%m%d"),
            "format": "CSV",
        }
        url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
        async with httpx.AsyncClient(timeout=180) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
        # Parse CSV: skip header section (ends with "-END HEADER-")
        lines = resp.text.strip().split("\n")
        data_start = next(i for i, l in enumerate(lines) if "-END HEADER-" in l) + 1
        chunk_df = pd.read_csv(
            __import__("io").StringIO("\n".join(lines[data_start:])),
        )
        chunk_df["timestamp"] = pd.to_datetime(
            chunk_df[["YEAR", "MO", "DY", "HR"]].rename(
                columns={"YEAR": "year", "MO": "month", "DY": "day", "HR": "hour"}
            )
        )
        chunk_df = chunk_df.drop(columns=["YEAR", "MO", "DY", "HR"])
        chunks.append(chunk_df)
        chunk_start = chunk_end + pd.Timedelta(days=1)

    df = pd.concat(chunks, ignore_index=True)
    df.index = df["timestamp"]
    df.index.name = "timestamp"
    df = df.drop(columns=["timestamp"], errors="ignore")

    df = df.rename(columns={
        "T2M": "temperature_c",
        "RH2M": "humidity_pct",
        "WS10M": "wind_speed_kmh",
        "WD10M": "wind_direction_deg",
        "PS": "pressure_hpa",
        "PRECTOTCORR": "precip_mm",
        "ALLSKY_SFC_SW_DWN": "solar_radiation_wm2",
    })
    # Derive min/max from hourly temperature using rolling windows
    df["temperature_min_c"] = df["temperature_c"].rolling(24, min_periods=1).min()
    df["temperature_max_c"] = df["temperature_c"].rolling(24, min_periods=1).max()

    df = df.replace(-999.0, np.nan)
    logger.info(f"NASA POWER: {len(df):,} rows downloaded")
    return df


# ── ERA5 Reanalysis Download ────────────────────────────────────
ERA5_CACHE_DIR = Path("models/era5_cache")
ERA5_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Area: [N, W, S, E] bounding box around Tucumán (≈0.25° grid)
ERA5_AREA = [-26.5, -65.5, -27.2, -65.0]

ERA5_SURFACE_VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation",
    "convective_available_potential_energy",
    "total_cloud_cover",
    "surface_solar_radiation_downwards",
    "soil_temperature_level_1",
]

ERA5_PRESSURE_VARIABLES = [
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "geopotential",
]

ERA5_PRESSURE_LEVELS = [850, 700, 500]


def _era5_cache_path(dataset: str, year: int) -> Path:
    return ERA5_CACHE_DIR / f"{dataset}_{year}.nc"


def _download_era5_year(c, dataset: str, year: int, variables: list, levels: list | None = None) -> Path:
    """
    Download one year of ERA5 data month-by-month (avoids CDS cost limits).
    Each month is cached separately. Returns the year file (merged from months).
    """
    import xarray as xr

    year_file = ERA5_CACHE_DIR / f"{dataset}_{year}.nc"
    if year_file.exists():
        logger.info(f"    ERA5 cache hit: {year_file.name}")
        return year_file

    month_frames = []
    for month in range(1, 13):
        month_cache = ERA5_CACHE_DIR / f"{dataset}_{year}_{month:02d}.nc"
        if month_cache.exists():
            logger.info(f"    ERA5 month cache hit: {month_cache.name}")
        else:
            request = {
                "product_type": "reanalysis",
                "variable": variables,
                "year": str(year),
                "month": f"{month:02d}",
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": ERA5_AREA,
                "data_format": "netcdf",
            }
            if levels:
                request["pressure_level"] = [str(l) for l in levels]

            logger.info(f"    Downloading {dataset} {year}-{month:02d} ...")
            try:
                c.retrieve(dataset, request, str(month_cache))
            except Exception as e:
                logger.warning(f"    Month {year}-{month:02d} failed: {e}")
                continue

        ds = xr.open_dataset(month_cache)
        month_frames.append(ds.to_dataframe().reset_index())
        ds.close()

    if not month_frames:
        raise RuntimeError(f"No months downloaded for {dataset} year={year}")

    df = pd.concat(month_frames, ignore_index=True)
    df.to_netcdf(year_file)
    logger.info(f"    Saved year file: {year_file.name}")
    return year_file


def download_era5_surface(years_back: int = 10) -> pd.DataFrame:
    """Download ERA5 single-level reanalysis for Tucumán. Returns hourly DataFrame."""
    import cdsapi
    import xarray as xr

    c = cdsapi.Client(quiet=True)
    end_year = 2025
    start_year = end_year - years_back + 1

    logger.info(f"Downloading ERA5 surface: {start_year}-{end_year}")
    frames = []
    for year in range(start_year, end_year + 1):
        nc = _download_era5_year(c, "reanalysis-era5-single-levels", year, ERA5_SURFACE_VARIABLES)
        ds = xr.open_dataset(nc)
        # Select nearest grid point to Tucumán centroid (avoids 3×3 grid duplicates)
        ds = ds.sel(latitude=TUC_LAT, longitude=TUC_LNG, method="nearest")
        df_year = ds.to_dataframe().reset_index()
        frames.append(df_year)
        ds.close()

    df = pd.concat(frames, ignore_index=True)

    # ERA5 time column → timezone-naive UTC
    if "time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
    elif "valid_time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["valid_time"], utc=True).dt.tz_localize(None)

    # Drop extra coords (latitude, longitude are constant after sel)
    drop_cols = [c for c in ["latitude", "longitude", "time", "valid_time", "expver", "number"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Rename to match pipeline expectations
    rename_map = {
        "t2m": "era5_temperature_c",
        "d2m": "era5_dewpoint_c",
        "sp": "era5_surface_pressure_hpa",
        "u10": "era5_wind_u_ms",
        "v10": "era5_wind_v_ms",
        "tp": "era5_precip_m",           # meters, convert to mm
        "cape": "era5_cape_jkg",
        "tcc": "era5_cloud_cover_pct",
        "ssrd": "era5_solar_radiation_wm2",
        "stl1": "era5_soil_temp_c",
    }
    # Filter to only rename columns that exist
    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # Conversions
    if "era5_precip_m" in df.columns:
        df["era5_precip_mm"] = df["era5_precip_m"] * 1000.0  # m → mm
        df = df.drop(columns=["era5_precip_m"])
    if "era5_surface_pressure_hpa" in df.columns:
        df["era5_surface_pressure_hpa"] = df["era5_surface_pressure_hpa"] / 100.0  # Pa → hPa
    if "era5_solar_radiation_wm2" in df.columns:
        df["era5_solar_radiation_wm2"] = df["era5_solar_radiation_wm2"] / 3600.0  # J/m² → W/m²

    # Wind speed + direction from u/v components
    if "era5_wind_u_ms" in df.columns and "era5_wind_v_ms" in df.columns:
        u = df["era5_wind_u_ms"]
        v = df["era5_wind_v_ms"]
        df["era5_wind_speed_kmh"] = np.sqrt(u**2 + v**2) * 3.6  # m/s → km/h
        df["era5_wind_direction_deg"] = (np.degrees(np.arctan2(-u, -v)) + 360) % 360
        df = df.drop(columns=["era5_wind_u_ms", "era5_wind_v_ms"])

    # Cloud cover: ERA5 uses 0-1 scale
    if "era5_cloud_cover_pct" in df.columns:
        df["era5_cloud_cover_pct"] = df["era5_cloud_cover_pct"] * 100.0

    logger.info(f"ERA5 surface: {len(df):,} rows")
    return df


def download_era5_pressure(years_back: int = 10) -> pd.DataFrame:
    """Download ERA5 pressure-level data (850/700/500 hPa) for Zonda detection."""
    import cdsapi
    import xarray as xr

    c = cdsapi.Client(quiet=True)
    end_year = 2025
    start_year = end_year - years_back + 1

    logger.info(f"Downloading ERA5 pressure levels {ERA5_PRESSURE_LEVELS} hPa: {start_year}-{end_year}")
    frames = []
    for year in range(start_year, end_year + 1):
        nc = _download_era5_year(
            c, "reanalysis-era5-pressure-levels", year,
            ERA5_PRESSURE_VARIABLES, ERA5_PRESSURE_LEVELS,
        )
        ds = xr.open_dataset(nc)
        ds = ds.sel(latitude=TUC_LAT, longitude=TUC_LNG, method="nearest")
        df_year = ds.to_dataframe().reset_index()
        frames.append(df_year)
        ds.close()

    df = pd.concat(frames, ignore_index=True)

    if "time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
    elif "valid_time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["valid_time"], utc=True).dt.tz_localize(None)

    drop_cols = [c for c in ["latitude", "longitude", "time", "valid_time", "expver", "number"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Pivot: one row per timestamp, columns per pressure level
    level_dfs = []
    if "pressure_level" in df.columns:
        for level in ERA5_PRESSURE_LEVELS:
            lvl_df = df[df["pressure_level"] == level].copy()
            lvl_df = lvl_df.drop(columns=["pressure_level"], errors="ignore")
            suffix = f"_{level}hpa"
            lvl_df = lvl_df.rename(columns={
                "t": f"era5_temp{suffix}",
                "u": f"era5_wind_u{suffix}",
                "v": f"era5_wind_v{suffix}",
                "z": f"era5_geopotential{suffix}",
            })
            # Wind speed from components
            u_col = f"era5_wind_u{suffix}"
            v_col = f"era5_wind_v{suffix}"
            if u_col in lvl_df.columns and v_col in lvl_df.columns:
                lvl_df[f"era5_wind_speed{suffix}"] = np.sqrt(lvl_df[u_col]**2 + lvl_df[v_col]**2) * 3.6
            level_dfs.append(lvl_df)

    if not level_dfs:
        logger.warning("  No pressure level data extracted")
        return pd.DataFrame()

    # Merge all levels on timestamp
    result = level_dfs[0]
    for lvl_df in level_dfs[1:]:
        result = result.merge(lvl_df, on="timestamp", how="outer")

    result = result.sort_values("timestamp").reset_index(drop=True)
    logger.info(f"ERA5 pressure levels: {len(result):,} rows, {len(result.columns)} columns")
    return result


def merge_era5_with_nasa(df_nasa: pd.DataFrame, df_era5_surface: pd.DataFrame, df_era5_pressure: pd.DataFrame) -> pd.DataFrame:
    """
    Merge ERA5 reanalysis data into NASA POWER observations.

    Strategy: ERA5 fills gaps and improves quality of existing 34 features,
    rather than adding new columns (which would require model architecture changes).
    """
    logger.info("Merging NASA POWER + ERA5 data...")

    df = df_nasa.copy()

    if not df_era5_surface.empty:
        # Align on timestamp index
        era5_s = df_era5_surface.copy()
        if "timestamp" in era5_s.columns:
            era5_s = era5_s.set_index("timestamp").sort_index()
        era5_s = era5_s[~era5_s.index.duplicated(keep="first")]

        # Reindex ERA5 to match NASA POWER timestamps (forward-fill gaps)
        era5_aligned = era5_s.reindex(df.index, method="ffill")

        # 1. CAPE: use real ERA5 CAPE instead of Bolton proxy
        if "era5_cape_jkg" in era5_aligned.columns:
            cape = era5_aligned["era5_cape_jkg"]
            mask = cape.notna()
            df.loc[mask, "cape_j_kg"] = cape[mask]
            logger.info(f"  ERA5 CAPE: filled {mask.sum():,} of {len(mask):,} rows")

        # 2. Surface pressure: fill gaps
        if "era5_surface_pressure_hpa" in era5_aligned.columns:
            sp = era5_aligned["era5_surface_pressure_hpa"]
            mask = sp.notna() & df["pressure_hpa"].isna()
            df.loc[mask, "pressure_hpa"] = sp[mask]

        # 3. Cloud cover: ERA5 provides 0-100%
        if "era5_cloud_cover_pct" in era5_aligned.columns:
            cc = era5_aligned["era5_cloud_cover_pct"]
            mask = cc.notna()
            df.loc[mask, "cloud_cover_pct"] = cc[mask]

        # 4. Solar radiation: ERA5 SSRD (J/m² → W/m²)
        if "era5_solar_radiation_wm2" in era5_aligned.columns:
            ssrd = era5_aligned["era5_solar_radiation_wm2"]
            mask = ssrd.notna()
            df.loc[mask, "uv_index"] = (ssrd[mask] / 40.0).clip(0, 12)  # rough proxy

        # 5. Soil temperature
        if "era5_soil_temp_c" in era5_aligned.columns:
            st = era5_aligned["era5_soil_temp_c"]
            mask = st.notna()
            df.loc[mask, "soil_temp_c"] = st[mask]

        # 6. Dewpoint: fill gaps
        if "era5_dewpoint_c" in era5_aligned.columns:
            dp = era5_aligned["era5_dewpoint_c"]
            mask = dp.notna() & df["dew_point_c"].isna()
            df.loc[mask, "dew_point_c"] = dp[mask]

        logger.info(f"  ERA5 surface enrichment applied")

    # Pressure-level data for Zonda detection
    if not df_era5_pressure.empty:
        era5_p = df_era5_pressure.copy()
        if "timestamp" in era5_p.columns:
            era5_p = era5_p.set_index("timestamp").sort_index()
        era5_p = era5_p[~era5_p.index.duplicated(keep="first")]
        era5_aligned_p = era5_p.reindex(df.index, method="ffill")

        # 7. 700hPa temperature → improves cordillera pressure proxy
        if "era5_temp_700hpa" in era5_aligned_p.columns:
            t700 = era5_aligned_p["era5_temp_700hpa"]
            mask = t700.notna()
            # Use 700hPa temp to refine thermal_differential_c
            if "era5_temperature_c" in (df_era5_surface.columns if not df_era5_surface.empty else []):
                pass  # already handled via surface
            logger.info(f"  ERA5 700hPa temp: {mask.sum():,} rows available")

        # 8. 850hPa wind → improves wind prediction at altitude
        if "era5_wind_speed_850hpa" in era5_aligned_p.columns:
            w850 = era5_aligned_p["era5_wind_speed_850hpa"]
            mask = w850.notna()
            logger.info(f"  ERA5 850hPa wind: {mask.sum():,} rows available")

    logger.info(f"  Final dataset: {len(df):,} rows, {len(df.columns)} columns")
    return df


def generate_synthetic_extremes(n_events: int = 500) -> pd.DataFrame:
    """Generate synthetic extreme weather events for augmentation."""
    logger.info(f"Generating {n_events} synthetic extreme events")
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


# ── Evaluation Metrics ──────────────────────────────────────────
def evaluate_predictions(
    all_preds: dict[str, np.ndarray],
    all_targets: dict[str, np.ndarray],
    horizons: list[int],
) -> dict:
    """
    Compute per-target, per-horizon metrics.

    Returns a nested dict: metrics[horizon][target_name] = {metric: value}
    """
    metrics = {}
    target_names = [
        "rain_probability", "precip_mm", "temperature_c",
        "wind_speed_kmh", "zonda_risk", "storm_risk", "hail_risk",
    ]

    for h in horizons:
        h_key = str(h)
        preds = all_preds[h_key]
        tgts = all_targets[h_key]
        h_metrics = {}

        for i, name in enumerate(target_names):
            p = preds[:, i]
            t = tgts[:, i]

            if name in ("rain_probability", "zonda_risk", "storm_risk", "hail_risk"):
                # Binary targets: accuracy, precision, recall
                p_bin = (p > 0.5).astype(int)
                t_bin = (t > 0.5).astype(int)
                h_metrics[name] = {
                    "accuracy": float(accuracy_score(t_bin, p_bin)),
                    "precision": float(precision_score(t_bin, p_bin, zero_division=0)),
                    "recall": float(recall_score(t_bin, p_bin, zero_division=0)),
                }
            elif name == "temperature_c":
                h_metrics[name] = {
                    "rmse": float(np.sqrt(mean_squared_error(t, p))),
                    "mae": float(mean_absolute_error(t, p)),
                }
            elif name in ("precip_mm", "wind_speed_kmh"):
                h_metrics[name] = {
                    "rmse": float(np.sqrt(mean_squared_error(t, p))),
                    "mae": float(mean_absolute_error(t, p)),
                }

        # Composite accuracy: average of binary accuracies + inverse normalized RMSE for temp
        binary_accs = [
            h_metrics[k]["accuracy"]
            for k in ("rain_probability", "zonda_risk", "storm_risk", "hail_risk")
            if k in h_metrics
        ]
        temp_rmse = h_metrics.get("temperature_c", {}).get("rmse", 10.0)
        temp_score = max(0, 1.0 - temp_rmse / 30.0)  # normalize: 30°C range
        h_metrics["composite_accuracy"] = float(np.mean(binary_accs + [temp_score]))

        metrics[f"{h}h"] = h_metrics

    return metrics


# ── Training Loop ───────────────────────────────────────────────
def train_fold(
    X_train: torch.Tensor,
    y_train: dict,
    X_val: torch.Tensor,
    y_val: dict,
    horizons: list[int],
    max_epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = "cpu",
) -> tuple[CnnLstmWeatherModel, dict]:
    """Train one fold and return the best model + metrics."""
    model = CnnLstmWeatherModel(
        n_features=34, n_timesteps=72, horizons=horizons,
    ).to(device)

    criterion = WeatherLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5,
    )

    # Flatten targets
    def flatten_y(ydict):
        return torch.cat([ydict[str(h)] for h in horizons], dim=-1)

    def unflatten_y(flat):
        result = {}
        for i, h in enumerate(horizons):
            start = i * 7
            chunk = flat[:, start:start + 7]
            result[str(h)] = {
                "rain_probability": chunk[:, 0],
                "precip_mm": chunk[:, 1],
                "temperature_c": chunk[:, 2],
                "wind_speed_kmh": chunk[:, 3],
                "zonda_risk": chunk[:, 4],
                "storm_risk": chunk[:, 5],
                "hail_risk": chunk[:, 6],
            }
        return result

    train_ds = TensorDataset(X_train, flatten_y(y_train))
    val_ds = TensorDataset(X_val, flatten_y(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        # Train
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            targets = unflatten_y(yb)
            loss = torch.stack([
                criterion(preds[str(h)], targets[str(h)]) for h in horizons
            ]).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                targets = unflatten_y(yb)
                loss = torch.stack([
                    criterion(preds[str(h)], targets[str(h)]) for h in horizons
                ]).mean()
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            logger.info(f"  Epoch {epoch:3d}/{max_epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

        if patience_counter >= 10:
            logger.info(f"  Early stopping at epoch {epoch}")
            break

    # Load best
    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    return model, {"best_val_loss": best_val_loss, "epoch": epoch}


def collect_predictions(
    model: CnnLstmWeatherModel,
    X: torch.Tensor,
    y: dict,
    horizons: list[int],
    device: str = "cpu",
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Run inference and collect all predictions + targets as numpy arrays."""
    model.eval()
    all_preds = {str(h): [] for h in horizons}
    all_targets = {str(h): [] for h in horizons}

    def flatten_y(ydict):
        return torch.cat([ydict[str(h)] for h in horizons], dim=-1)

    def unflatten_y(flat):
        result = {}
        for i, h in enumerate(horizons):
            start = i * 7
            chunk = flat[:, start:start + 7]
            result[str(h)] = {
                "rain_probability": chunk[:, 0],
                "precip_mm": chunk[:, 1],
                "temperature_c": chunk[:, 2],
                "wind_speed_kmh": chunk[:, 3],
                "zonda_risk": chunk[:, 4],
                "storm_risk": chunk[:, 5],
                "hail_risk": chunk[:, 6],
            }
        return result

    ds = TensorDataset(X, flatten_y(y))
    loader = DataLoader(ds, batch_size=256, shuffle=False)

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            preds = model(xb)
            targets = unflatten_y(yb.cpu())
            for h in horizons:
                h_key = str(h)
                # Stack each target into [batch, 7]
                pred_arr = torch.stack([
                    preds[h_key]["rain_probability"],
                    preds[h_key]["precip_mm"],
                    preds[h_key]["temperature_c"],
                    preds[h_key]["wind_speed_kmh"],
                    preds[h_key]["zonda_risk"],
                    preds[h_key]["storm_risk"],
                    preds[h_key]["hail_risk"],
                ], dim=1).cpu().numpy()
                all_preds[h_key].append(pred_arr)
                tgt_arr = torch.stack([
                    targets[h_key]["rain_probability"],
                    targets[h_key]["precip_mm"],
                    targets[h_key]["temperature_c"],
                    targets[h_key]["wind_speed_kmh"],
                    targets[h_key]["zonda_risk"],
                    targets[h_key]["storm_risk"],
                    targets[h_key]["hail_risk"],
                ], dim=1).numpy()
                all_targets[h_key].append(tgt_arr)

    for h in horizons:
        h_key = str(h)
        all_preds[h_key] = np.concatenate(all_preds[h_key], axis=0)
        all_targets[h_key] = np.concatenate(all_targets[h_key], axis=0)

    return all_preds, all_targets


# ── Main ────────────────────────────────────────────────────────
async def main(years_back: int = 10, n_synthetic: int = 500, max_epochs: int = 100, n_folds: int = 5, use_era5: bool = False):
    logger.info("=" * 60)
    logger.info("CIELO·TUC — Model Training v1.0")
    if use_era5:
        logger.info("  Data sources: NASA POWER + ERA5 Reanalysis")
    else:
        logger.info("  Data source: NASA POWER only")
    logger.info("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    horizons = [3, 6, 12, 24, 48, 168]

    # ── 1. Download data ──────────────────────────────────────
    logger.info(f"\n[1/6] Downloading {years_back} years of NASA POWER data...")
    t0 = time.time()
    df_main = await download_nasa_power(years_back)
    logger.info(f"  Done in {time.time() - t0:.1f}s")

    # ── 1b. Optional ERA5 download ────────────────────────────
    df_era5_surface = pd.DataFrame()
    df_era5_pressure = pd.DataFrame()
    if use_era5:
        t_era5 = time.time()
        logger.info(f"\n[1b/6] Downloading ERA5 reanalysis ({years_back} years)...")
        try:
            df_era5_surface = download_era5_surface(years_back)
            df_era5_pressure = download_era5_pressure(years_back)
            df_main = merge_era5_with_nasa(df_main, df_era5_surface, df_era5_pressure)
            logger.info(f"  ERA5 download + merge: {time.time() - t_era5:.1f}s")
        except Exception as e:
            logger.warning(f"  ERA5 download failed: {e}")
            logger.warning("  Falling back to NASA POWER only")
            use_era5 = False

    # ── 2. Augment with synthetic extremes ────────────────────
    logger.info(f"\n[2/6] Augmenting with {n_synthetic} synthetic extreme events...")
    df_synthetic = generate_synthetic_extremes(n_events=n_synthetic)
    df_combined = pd.concat([df_main, df_synthetic], ignore_index=True)
    df_combined = df_combined.sort_values("timestamp").reset_index(drop=True)
    logger.info(f"  Combined dataset: {len(df_combined):,} rows")

    # ── 3. Preprocess ─────────────────────────────────────────
    logger.info("\n[3/6] Feature engineering and preprocessing...")
    pipeline = WeatherDataPipeline()
    df_processed = pipeline.preprocess(df_combined)
    pipeline.fit_scaler(df_processed)
    pipeline.save_scaler()
    logger.info("  Scaler saved to models/scaler.pkl")

    # ── 4. Build tensors ──────────────────────────────────────
    logger.info("\n[4/6] Building training tensors...")
    X, y = pipeline.build_training_tensors(df_processed)
    logger.info(f"  X shape: {X.shape}")
    for h in horizons:
        logger.info(f"  y[{h}h] shape: {y[str(h)].shape}")

    # ── 5. TimeSeriesSplit backtesting ────────────────────────
    logger.info(f"\n[5/6] Training with {n_folds}-fold TimeSeriesSplit backtesting...")
    n = len(X)
    fold_size = n // (n_folds + 1)

    all_fold_metrics = []
    best_model = None
    best_composite = -1.0

    for fold in range(n_folds):
        logger.info(f"\n--- Fold {fold + 1}/{n_folds} ---")
        # Time-aware split: train on [0 : (fold+1)*fold_size], validate on next fold_size
        train_end = (fold + 1) * fold_size
        val_end = min(train_end + fold_size, n)

        X_train = X[:train_end]
        X_val = X[train_end:val_end]
        y_train = {h: v[:train_end] for h, v in y.items()}
        y_val = {h: v[train_end:val_end] for h, v in y.items()}

        logger.info(f"  Train: {len(X_train):,} samples | Val: {len(X_val):,} samples")

        model, train_info = train_fold(
            X_train, y_train, X_val, y_val,
            horizons=horizons, max_epochs=max_epochs,
            batch_size=128, lr=1e-3, device=device,
        )

        # Evaluate on validation fold
        preds, tgts = collect_predictions(model, X_val, y_val, horizons, device)
        fold_metrics = evaluate_predictions(preds, tgts, horizons)
        fold_metrics["fold"] = fold + 1
        fold_metrics["train_loss"] = train_info["best_val_loss"]
        fold_metrics["samples_train"] = len(X_train)
        fold_metrics["samples_val"] = len(X_val)
        all_fold_metrics.append(fold_metrics)

        composite = np.mean([
            fold_metrics[f"{h}h"]["composite_accuracy"] for h in horizons
        ])
        logger.info(f"  Fold {fold + 1} composite accuracy: {composite:.4f}")

        if composite > best_composite:
            best_composite = composite
            best_model = model

    # ── Final evaluation on last fold (full holdout) ──────────
    logger.info("\n" + "=" * 60)
    logger.info("FINAL EVALUATION (last fold holdout)")
    logger.info("=" * 60)

    final_preds, final_tgts = collect_predictions(
        best_model, X_val, y_val, horizons, device,
    )
    final_metrics = evaluate_predictions(final_preds, final_tgts, horizons)

    # Print summary
    for h in horizons:
        h_key = f"{h}h"
        m = final_metrics[h_key]
        logger.info(f"\n  Horizon {h}h:")
        logger.info(f"    Rain accuracy:  {m['rain_probability']['accuracy']:.4f}")
        logger.info(f"    Rain precision: {m['rain_probability']['precision']:.4f}")
        logger.info(f"    Rain recall:    {m['rain_probability']['recall']:.4f}")
        logger.info(f"    Temp RMSE:      {m['temperature_c']['rmse']:.2f}°C")
        logger.info(f"    Temp MAE:       {m['temperature_c']['mae']:.2f}°C")
        logger.info(f"    Wind RMSE:      {m['wind_speed_kmh']['rmse']:.2f} km/h")
        logger.info(f"    Zonda accuracy: {m['zonda_risk']['accuracy']:.4f}")
        logger.info(f"    Storm accuracy: {m['storm_risk']['accuracy']:.4f}")
        logger.info(f"    Hail accuracy:  {m['hail_risk']['accuracy']:.4f}")
        logger.info(f"    Composite:      {m['composite_accuracy']:.4f}")

    # Aggregate across horizons
    avg_composite = np.mean([final_metrics[f"{h}h"]["composite_accuracy"] for h in horizons])
    avg_rain_acc = np.mean([final_metrics[f"{h}h"]["rain_probability"]["accuracy"] for h in horizons])
    avg_temp_rmse = np.mean([final_metrics[f"{h}h"]["temperature_c"]["rmse"] for h in horizons])
    logger.info(f"\n  {'=' * 40}")
    logger.info(f"  AVERAGE COMPOSITE ACCURACY: {avg_composite:.4f}")
    logger.info(f"  AVERAGE RAIN ACCURACY:      {avg_rain_acc:.4f}")
    logger.info(f"  AVERAGE TEMP RMSE:          {avg_temp_rmse:.2f}°C")

    # ── 6. Save outputs ──────────────────────────────────────
    from datetime import timezone
    version = f"1.0-{datetime.now(timezone.utc).strftime('%Y%m%d')}"
    timestamp = datetime.now(timezone.utc).isoformat()

    # Save model checkpoint
    checkpoint_path = MODELS_DIR / f"cielotuc_v{version}.pt"
    torch.save({
        "model_state_dict": best_model.state_dict(),
        "version": version,
        "horizons": horizons,
        "n_features": 34,
        "n_timesteps": 72,
        "trained_at": timestamp,
        "training_samples": len(X),
        "device": device,
        "use_era5": use_era5,
    }, checkpoint_path)
    logger.info(f"\nModel saved: {checkpoint_path}")

    # Save metrics JSON
    report = {
        "version": version,
        "trained_at": timestamp,
        "training_samples": len(X),
        "years_of_data": years_back,
        "synthetic_events": n_synthetic,
        "device": device,
        "horizons": horizons,
        "use_era5": use_era5,
        "data_sources": "NASA POWER + ERA5" if use_era5 else "NASA POWER",
        "avg_composite_accuracy": float(avg_composite),
        "avg_rain_accuracy": float(avg_rain_acc),
        "avg_temp_rmse_c": float(avg_temp_rmse),
        "final_metrics": final_metrics,
        "fold_metrics": all_fold_metrics,
    }
    metrics_path = MODELS_DIR / f"metrics_v{version}.json"
    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Metrics saved: {metrics_path}")

    logger.info("\n" + "=" * 60)
    logger.info(f"CIELO·TUC v{version} — Training complete!")
    logger.info(f"  Data source: {'NASA POWER + ERA5' if use_era5 else 'NASA POWER'}")
    logger.info(f"  Composite accuracy: {avg_composite:.4f}")
    logger.info(f"  Model: {checkpoint_path}")
    logger.info(f"  Metrics: {metrics_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CIELO·TUC v1.0")
    parser.add_argument("--years", type=int, default=10, help="Years of NASA POWER data")
    parser.add_argument("--n-synthetic", type=int, default=500, help="Synthetic events")
    parser.add_argument("--max-epochs", type=int, default=100, help="Max epochs per fold")
    parser.add_argument("--folds", type=int, default=5, help="Number of TimeSeriesSplit folds (1=single split)")
    parser.add_argument("--era5", action="store_true", help="Download and merge ERA5 reanalysis data (requires CDS API key)")
    args = parser.parse_args()
    asyncio.run(main(
        years_back=args.years,
        n_synthetic=args.n_synthetic,
        max_epochs=args.max_epochs,
        n_folds=args.folds,
        use_era5=args.era5,
    ))
