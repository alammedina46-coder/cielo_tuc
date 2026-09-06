"""
app/ml/pipeline/data_pipeline.py
──────────────────────────────────
End-to-end data pipeline:
  1. Fetch raw data from external APIs (SMN, NASA GPM, ERA5)
  2. Merge into a unified hourly DataFrame per station
  3. Compute derived features (CAPE proxy, Zonda index, etc.)
  4. Normalize with a fitted MinMaxScaler
  5. Build sliding-window tensors ready for the CNN-LSTM

Usage
-----
  pipeline = WeatherDataPipeline()
  await pipeline.fetch_and_update()       # call from Celery task
  X, y = pipeline.build_training_tensors()
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import MinMaxScaler
import torch

from app.core.config import settings
from app.services.windy_client import WindyClient


# ── Feature columns (order matters — same as model input) ─────
FEATURE_COLS = [
    # Atmospheric basics
    "temperature_c", "temperature_min_c", "temperature_max_c", "feels_like_c",
    "pressure_hpa", "pressure_sea_level_hpa", "humidity_pct", "dew_point_c",
    # Precipitation
    "precip_mm", "precip_3h_mm", "precip_6h_mm", "precip_24h_mm",
    # Wind
    "wind_speed_kmh", "wind_direction_deg", "wind_gust_kmh",
    # Visibility & radiation
    "visibility_km", "cloud_cover_pct", "uv_index",
    # Derived instability indices
    "cape_j_kg", "k_index",
    # Satellite
    "ndvi", "soil_temp_c", "precipitable_water_mm",
    # Zonda-specific
    "cordillera_pressure_hpa", "thermal_differential_c",
    # Temporal encodings (cyclic)
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    # Large-scale context
    "enso_index",
    # Zone features (static per station, repeated)
    "altitude_m", "impermeable_pct", "is_mountain",
    # Neighbor pressure differential (Zonda proxy)
    "andes_plain_pressure_diff",
]

assert len(FEATURE_COLS) == 34, f"Expected 34 features, got {len(FEATURE_COLS)}"

# ── Target columns per horizon ────────────────────────────────
TARGET_COLS = [
    "rain_probability", "precip_mm",
    "temperature_c", "wind_speed_kmh",
    "zonda_risk", "storm_risk", "hail_risk",
]


class WeatherDataPipeline:
    """
    Manages data fetching, feature engineering, scaling and
    tensor construction for training and inference.
    """

    def __init__(
        self,
        lookback_hours: int = 72,
        horizons: list[int] | None = None,
        scaler_path: Path | None = None,
    ):
        self.lookback = lookback_hours
        self.horizons = horizons or settings.model_forecast_horizons
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scaler_path = scaler_path or Path("models/scaler.pkl")
        self._df: Optional[pd.DataFrame] = None

    # ── 1. Data fetching ──────────────────────────────────────

    # ── 1. Data fetching ──────────────────────────────────────

    async def fetch_windy(
        self,
        lat: float,
        lon: float,
        model: str | None = None,
    ) -> pd.DataFrame:
        """
        Fetch multi-level ECMWF/GFS forecast from the Windy Point
        Forecast API for a given coordinate.

        The result includes the upper-level wind fields (850/700 hPa)
        that drive the Zonda index, and the Andes-to-plain pressure
        differential used by the model.
        """
        client = WindyClient(
            model=model,
            time_step=settings.windy_time_step_hours,
        )
        return await client.fetch_forecast(lat=lat, lon=lon)

    async def fetch_smn(self, hours_back: int = 48) -> pd.DataFrame:
        """
        Fetch latest observations from SMN public API.
        Returns a DataFrame with columns matching FEATURE_COLS (partial).
        """
        url = settings.smn_api_url
        async with httpx.AsyncClient(timeout=20) as client:
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()
                rows = []
                for item in data:
                    if "tucuman" in item.get("name", "").lower():
                        rows.append({
                            "timestamp": pd.to_datetime(item.get("datetime")),
                            "temperature_c": item.get("temp"),
                            "humidity_pct": item.get("humidity"),
                            "pressure_hpa": item.get("pressure"),
                            "wind_speed_kmh": item.get("wind_speed"),
                            "wind_direction_deg": item.get("wind_dir"),
                            "precip_mm": item.get("precip", 0.0),
                        })
                return pd.DataFrame(rows)
            except Exception as e:
                logger.warning(f"SMN fetch failed: {e} — using cached data")
                return pd.DataFrame()

    async def fetch_nasa_gpm(
        self, lat: float, lng: float, days_back: int = 7
    ) -> pd.DataFrame:
        """
        Fetch half-hourly precipitation from NASA GPM IMERG.
        Aggregates to hourly.
        """
        end = datetime.utcnow()
        start = end - timedelta(days=days_back)
        url = (
            f"{settings.nasa_gpm_base_url}/precipitation"
            f"?lat={lat}&lon={lng}"
            f"&start={start.strftime('%Y%m%d')}"
            f"&end={end.strftime('%Y%m%d')}"
        )
        async with httpx.AsyncClient(timeout=30) as client:
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                raw = resp.json()
                rows = [
                    {"timestamp": pd.to_datetime(r["time"]), "precip_mm": r["precip"]}
                    for r in raw.get("data", [])
                ]
                df = pd.DataFrame(rows)
                if not df.empty:
                    df = df.set_index("timestamp").resample("1h").sum().reset_index()
                return df
            except Exception as e:
                logger.warning(f"NASA GPM fetch failed: {e}")
                return pd.DataFrame()

    async def fetch_nasa_power(
        self, lat: float, lng: float, days_back: int = 7
    ) -> pd.DataFrame:
        """
        Fetch hourly temperature, humidity, wind from NASA POWER.
        Free — no API key required.
        """
        end = datetime.utcnow()
        start = end - timedelta(days=days_back)
        params = {
            "parameters": "T2M,RH2M,WS10M,WD10M,PS,ALLSKY_SFC_SW_DWN",
            "community": "RE",
            "longitude": lng,
            "latitude": lat,
            "start": start.strftime("%Y%m%d"),
            "end": end.strftime("%Y%m%d"),
            "format": "JSON",
            "time-standard": "UTC",
            "temporal-api": "hourly",
        }
        url = f"{settings.nasa_power_base_url}/hourly/point"
        async with httpx.AsyncClient(timeout=30) as client:
            try:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()["properties"]["parameter"]
                df = pd.DataFrame(data).reset_index()
                df.rename(columns={
                    "index": "timestamp",
                    "T2M": "temperature_c",
                    "RH2M": "humidity_pct",
                    "WS10M": "wind_speed_kmh",
                    "WD10M": "wind_direction_deg",
                    "PS": "pressure_hpa",
                }, inplace=True)
                df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y%m%d%H")
                return df
            except Exception as e:
                logger.warning(f"NASA POWER fetch failed: {e}")
                return pd.DataFrame()

    # ── 2. Feature engineering ────────────────────────────────

    def add_cyclic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode hour-of-day and day-of-year as sine/cosine pairs."""
        df = df.copy()
        hour = df["timestamp"].dt.hour
        doy = df["timestamp"].dt.day_of_year
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        df["doy_sin"] = np.sin(2 * np.pi * doy / 365)
        df["doy_cos"] = np.cos(2 * np.pi * doy / 365)
        return df

    def add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling precipitation windows."""
        df = df.sort_values("timestamp").copy()
        df["precip_3h_mm"] = df["precip_mm"].rolling(3, min_periods=1).sum()
        df["precip_6h_mm"] = df["precip_mm"].rolling(6, min_periods=1).sum()
        df["precip_24h_mm"] = df["precip_mm"].rolling(24, min_periods=1).sum()
        return df

    def add_zonda_proxy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Approximate Zonda risk from pressure gradient.
        Prefers provided cordillera readings (synthetic events, Windy,
        real stations); falls back to a heuristic estimate otherwise.
        """
        df = df.copy()

        if (
            "cordillera_pressure_hpa" not in df.columns
            or df["cordillera_pressure_hpa"].isna().all()
        ):
            # Placeholder — in production, join cordillera station readings
            df["cordillera_pressure_hpa"] = (
                df["pressure_hpa"].fillna(1013) * 0.88
            )

        if (
            "thermal_differential_c" not in df.columns
            or df["thermal_differential_c"].isna().all()
        ):
            df["thermal_differential_c"] = (
                df["temperature_c"].fillna(20) - 8  # approx summit temp
            )

        df["andes_plain_pressure_diff"] = (
            df["pressure_hpa"].fillna(1013)
            - df["cordillera_pressure_hpa"].fillna(891)
        )
        return df

    def add_cape_proxy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Proxy CAPE from temperature and dew point.
        Prefers real CAPE values (synthetic events, ERA5, Windy) and
        only estimates where missing. Real CAPE requires radiosonde data;
        this approximation covers observability gaps.
        """
        df = df.copy()

        if (
            "cape_j_kg" not in df.columns
            or df["cape_j_kg"].isna().all()
        ):
            # Simple Bolton approximation
            T = df["temperature_c"].fillna(20)
            Td = df.get("dew_point_c", T - 5).fillna(T - 5)
            df["cape_j_kg"] = np.maximum(0, (T - Td) * 50)

        if "k_index" not in df.columns or df["k_index"].isna().all():
            T = df["temperature_c"].fillna(20)
            Td = df.get("dew_point_c", T - 5).fillna(T - 5)
            df["k_index"] = T - Td  # simplified K-index proxy

        return df

    def add_static_zone_features(
        self,
        df: pd.DataFrame,
        altitude_m: float = 450.0,
        impermeable_pct: float = 0.6,
        is_mountain: bool = False,
    ) -> pd.DataFrame:
        """Fill in zone-level static features (same for all rows of a station)."""
        df = df.copy()
        df["altitude_m"] = altitude_m
        df["impermeable_pct"] = impermeable_pct
        df["is_mountain"] = float(is_mountain)
        return df

    def fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Linear interpolation → forward fill → backward fill."""
        if "timestamp" in df.columns:
            df = df.dropna(subset=["timestamp"])
            df = df.set_index("timestamp").sort_index()
            df = df[~df.index.duplicated(keep="first")]
        df = df.interpolate(method="time", limit=6)
        df = df.ffill(limit=12).bfill(limit=12)
        df = df.fillna(0.0)
        return df.reset_index()

    # ── 3. Full preprocessing pipeline ───────────────────────

    def preprocess(
        self,
        df: pd.DataFrame,
        altitude_m: float = 450.0,
        impermeable_pct: float = 0.6,
        is_mountain: bool = False,
        enso_index: float = 0.0,
        ndvi: float = 0.5,
        soil_temp_c: float = 20.0,
        precipitable_water_mm: float = 25.0,
        uv_index: float = 6.0,
    ) -> pd.DataFrame:
        """Apply all feature engineering steps to a raw DataFrame."""
        df = self.add_cyclic_features(df)
        df = self.add_rolling_features(df)
        df = self.add_zonda_proxy(df)
        df = self.add_cape_proxy(df)
        df = self.add_static_zone_features(df, altitude_m, impermeable_pct, is_mountain)
        df["enso_index"] = enso_index
        df["ndvi"] = ndvi
        df["soil_temp_c"] = soil_temp_c
        df["precipitable_water_mm"] = precipitable_water_mm
        df["uv_index"] = uv_index
        df["visibility_km"] = df.get("visibility_km", pd.Series([10.0] * len(df)))
        df["cloud_cover_pct"] = df.get("cloud_cover_pct", pd.Series([50.0] * len(df)))
        df["feels_like_c"] = df.get("feels_like_c", df["temperature_c"])
        df["pressure_sea_level_hpa"] = df.get(
            "pressure_sea_level_hpa", df["pressure_hpa"]
        )
        df["dew_point_c"] = df.get("dew_point_c", df["temperature_c"] - 5)
        df["temperature_min_c"] = df.get("temperature_min_c", df["temperature_c"] - 2)
        df["temperature_max_c"] = df.get("temperature_max_c", df["temperature_c"] + 2)
        df["wind_gust_kmh"] = df.get(
            "wind_gust_kmh", df["wind_speed_kmh"] * 1.3
        )
        df = self.fill_missing(df)
        # Ensure all 34 features exist
        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0
        # Preserve event labels (real events / synthetic `_label_*`) so the
        # tensor builder can use them as supervised targets.
        label_cols = [c for c in df.columns if c.startswith("_label_")]
        return df[["timestamp"] + FEATURE_COLS + label_cols]

    # ── 4. Scaling ────────────────────────────────────────────

    def fit_scaler(self, df: pd.DataFrame) -> None:
        """Fit MinMax scaler on the full training dataset."""
        self.scaler.fit(df[FEATURE_COLS].values)

    def save_scaler(self, path: Path | None = None) -> None:
        """Persist the fitted scaler so inference can reuse it."""
        import pickle
        path = path or self.scaler_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Scaler saved to {path}")

    def load_scaler(self, path: Path | None = None) -> bool:
        """
        Load a previously fitted scaler (required before transform /
        build_inference_window in a live process). Returns True on success.
        """
        import pickle
        path = path or self.scaler_path
        if not path.exists():
            logger.warning(f"Scaler not found at {path} — must fit first")
            return False
        with open(path, "rb") as f:
            self.scaler = pickle.load(f)
        return True

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(df[FEATURE_COLS].values)

    def inverse_transform_temp(self, scaled: np.ndarray) -> np.ndarray:
        """Inverse transform only temperature column."""
        temp_idx = FEATURE_COLS.index("temperature_c")
        dummy = np.zeros((len(scaled), len(FEATURE_COLS)))
        dummy[:, temp_idx] = scaled
        return self.scaler.inverse_transform(dummy)[:, temp_idx]

    # ── 5. Sliding window tensor builder ──────────────────────

    def build_training_tensors(
        self, df: pd.DataFrame
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Build input/target tensors for training.

        X shape: [N, lookback, n_features]
        y shape: {horizon_str: [N, n_targets]} for each horizon

        Target construction: shift the precip/temp/wind columns forward
        by `horizon` hours to create the label for that window.

        Targets are kept in RAW units (°C, mm, km/h) — matching how the
        prediction service interprets model outputs — except that
        rain_probability and the extreme-risk flags (Zonda/storm/hail)
        are binary 0/1. When the preprocessed frame carries `_label_*`
        columns (real events or synthetic augmentation), those override
        the heuristic risk labels.
        """
        scaled = self.transform(df)
        # Raw (unscaled) series for building targets in physical units.
        draw_p = df["precip_mm"].to_numpy(dtype=np.float64)
        draw_temp = df["temperature_c"].to_numpy(dtype=np.float64)
        draw_wind = df["wind_speed_kmh"].to_numpy(dtype=np.float64)

        has_labels = any(c.startswith("_label_") for c in df.columns)
        label_cols = [c for c in df.columns if c.startswith("_label_")]
        label_arrays = {
            c: df[c].to_numpy(dtype=np.float64) for c in label_cols
        }
        zonda_label = label_arrays.get("_label_zonda")
        storm_label = label_arrays.get("_label_storm")
        hail_label = label_arrays.get("_label_hail")

        # Data-driven heuristic thresholds (used only when no labels exist)
        draw_diff = df["andes_plain_pressure_diff"].to_numpy(dtype=np.float64)
        draw_cape = df["cape_j_kg"].to_numpy(dtype=np.float64)
        zonda_thr = float(np.quantile(draw_diff, 0.9))
        storm_thr = float(np.quantile(draw_cape, 0.9))

        X_list = []
        y_dict: dict[str, list] = {str(h): [] for h in self.horizons}
        n = len(scaled)

        max_horizon = max(self.horizons)
        for i in range(self.lookback, n - max_horizon):
            window = scaled[i - self.lookback: i]   # [lookback, features]
            X_list.append(window)
            for h in self.horizons:
                future_idx = i + h
                # ── Targets in raw units ──────────────────────
                p = draw_p[future_idx]
                t = draw_temp[future_idx]
                w = draw_wind[future_idx]
                raw_precip = p
                rain = float(raw_precip > 1.0)    # binarize

                # ── Extreme-risk labels ───────────────────────
                if zonda_label is not None:
                    zonda = float(zonda_label[future_idx])
                else:
                    zonda = float(draw_diff[future_idx] > zonda_thr)

                if storm_label is not None:
                    storm = float(storm_label[future_idx])
                else:
                    storm = float(draw_cape[future_idx] > storm_thr)

                if hail_label is not None:
                    hail = float(hail_label[future_idx])
                else:
                    hail = float(storm and raw_precip > 10.0)

                y_dict[str(h)].append([rain, p, t, w, zonda, storm, hail])

        X = torch.tensor(np.array(X_list), dtype=torch.float32)
        y = {
            h: torch.tensor(np.array(vals), dtype=torch.float32)
            for h, vals in y_dict.items()
        }
        return X, y

    def build_inference_window(
        self, df: pd.DataFrame
    ) -> torch.Tensor:
        """
        Build a single input window for live inference.
        Uses the last `lookback` rows of the preprocessed DataFrame.
        Returns shape: [1, lookback, n_features]
        """
        assert len(df) >= self.lookback, (
            f"Need at least {self.lookback} rows for inference, got {len(df)}"
        )
        df_tail = df.tail(self.lookback)
        scaled = self.transform(df_tail)
        return torch.tensor(scaled[np.newaxis, :, :], dtype=torch.float32)
