#!/usr/bin/env python3
"""
scripts/train_v6_light.py
─────────────────────────
CIELO·TUC v6.0-light Training — Conservative improvement over v5.0

v5.0 baseline: 83.8% composite, 67.1% rain, 7.3°C temp RMSE
v6.0 (transformer) was WORSE at 70.8% — too much complexity for synthetic data.
Strategy: "do less, better."

Key improvements over v5.0:
  1. FastWeatherModelV5Light: FeatureDropout (10%) + multi-scale CNN (kernel 3 + 5)
  2. 2 extra features for long horizons: temp_3day_avg, pressure_3day_avg
  3. CosineAnnealingWarmRestarts scheduler (restarts at 15, 45, 105)
  4. Label smoothing 0.02 for binary targets (FocalBCE)
  5. 3500 synthetic events (up from 3000)
  6. 150 max epochs, patience 15
  7. MixUp augmentation (α=0.2)

Usage:
  python scripts/train_v6_light.py --years 20 --n-synthetic 3500 --max-epochs 150 --folds 3 --batch-size 512
"""

import argparse
import asyncio
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

def _ensure_deps():
    """Auto-install missing packages (useful in Colab)."""
    required = {"httpx": "httpx", "loguru": "loguru", "sklearn": "scikit-learn"}
    missing = []
    for mod, pkg in required.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Installing missing packages: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)

_ensure_deps()

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# loguru is optional — fallback to print-based logger for Colab
try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    logger = logging.getLogger("train_v6_light")
    # Duck-type logger to match loguru API
    class _Logger:
        def info(self, msg, *a, **kw): logger.info(msg, *a)
        def warning(self, msg, *a, **kw): logger.warning(msg, *a)
        def error(self, msg, *a, **kw): logger.error(msg, *a)
    logger = _Logger()

from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    f1_score,
)
from torch.utils.data import DataLoader, TensorDataset

# ── Constants ───────────────────────────────────────────────────
TUC_LAT = -26.82
TUC_LNG = -65.22

# 10 zones across Tucumán province
ZONE_POINTS = [
    {"lat": -26.82, "lng": -65.22, "altitude_m": 450.0, "is_mountain": 0.0, "zone_name": "Capital"},
    {"lat": -26.81, "lng": -65.32, "altitude_m": 520.0, "is_mountain": 0.0, "zone_name": "Yerba Buena"},
    {"lat": -26.73, "lng": -65.27, "altitude_m": 600.0, "is_mountain": 1.0, "zone_name": "Tafí Viejo"},
    {"lat": -27.35, "lng": -65.60, "altitude_m": 350.0, "is_mountain": 0.0, "zone_name": "Concepción"},
    {"lat": -26.90, "lng": -64.95, "altitude_m": 300.0, "is_mountain": 0.0, "zone_name": "Cruz Alta"},
    {"lat": -26.85, "lng": -65.19, "altitude_m": 460.0, "is_mountain": 0.0, "zone_name": "Bella Vista"},
    {"lat": -26.79, "lng": -65.30, "altitude_m": 480.0, "is_mountain": 0.0, "zone_name": "Lules"},
    {"lat": -27.17, "lng": -65.50, "altitude_m": 400.0, "is_mountain": 0.0, "zone_name": "Monteros"},
    {"lat": -27.50, "lng": -65.50, "altitude_m": 700.0, "is_mountain": 1.0, "zone_name": "Chicligasta"},
    {"lat": -26.80, "lng": -65.20, "altitude_m": 470.0, "is_mountain": 0.0, "zone_name": "Capital Norte"},
]

LOOKBACK = 24
HORIZONS = [3, 6, 12, 24, 48, 168]
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

SEASONAL_TEMP_RANGE = 35.0  # Tucumán: ~5°C (winter night) to ~40°C (summer day)

HORIZON_WEIGHTS = {3: 2.0, 6: 1.8, 12: 1.5, 24: 1.2, 48: 1.0, 168: 0.8}


# ── FeatureDropout (new in v6) ─────────────────────────────────
class FeatureDropout(nn.Module):
    """Randomly zero out entire input features during training only.
    Acts as regularization without adding external data."""
    def __init__(self, drop_rate: float = 0.10):
        super().__init__()
        self.drop_rate = drop_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_rate <= 0:
            return x
        # x shape: (batch, timesteps, features)
        mask = (torch.rand(x.shape[0], 1, x.shape[2], device=x.device) > self.drop_rate).float()
        return x * mask


# ── Model v6-light (~280K params) ──────────────────────────────
class FastWeatherModelV5Light(nn.Module):
    """
    v6.0-light model — same LSTM backbone as v5 with two additions:
    1. FeatureDropout (10%) as regularization
    2. Multi-scale CNN: kernel_size=3 branch + kernel_size=5 branch → concat → 128→64

    ~280K parameters (up from 232K).
    """
    def __init__(self, n_features: int, n_timesteps: int = LOOKBACK, horizons: list = None):
        super().__init__()
        self.horizons = horizons or HORIZONS
        self.hidden_size = 112

        # Feature dropout — only active during training
        self.feature_dropout = FeatureDropout(drop_rate=0.10)

        # CNN branch 1: kernel_size=3 (captures 3-hour patterns)
        self.cnn3 = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )

        # CNN branch 2: kernel_size=5 (captures 5-hour patterns — multi-scale)
        self.cnn5 = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )

        # Fusion: concat 64+64=128 → Linear 64
        self.cnn_fusion = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
        )

        # 2-layer LSTM — 112 hidden (same as v5)
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=self.hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.15,
        )

        # Attention
        self.attn = nn.Linear(self.hidden_size, 1)

        # Per-horizon heads — 56 units (same as v5)
        self.heads = nn.ModuleDict()
        for h in self.horizons:
            self.heads[str(h)] = nn.Sequential(
                nn.Linear(self.hidden_size, 56),
                nn.BatchNorm1d(56),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(56, 7),
            )

    def forward(self, x: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        # Feature dropout (training only)
        x = self.feature_dropout(x)

        # Multi-scale CNN
        cnn_in = x.permute(0, 2, 1)  # (B, F, T)
        out3 = self.cnn3(cnn_in)  # (B, 64, T)
        out5 = self.cnn5(cnn_in)  # (B, 64, T)
        cnn_cat = torch.cat([out3, out5], dim=1)  # (B, 128, T)
        cnn_fused = self.cnn_fusion(cnn_cat.permute(0, 2, 1))  # (B, T, 64)

        # LSTM
        lstm_out, _ = self.lstm(cnn_fused)

        # Attention
        scores = self.attn(lstm_out)
        weights = torch.softmax(scores, dim=1)
        context = (weights * lstm_out).sum(dim=1)

        result = {}
        for h in self.horizons:
            raw = self.heads[str(h)](context)
            result[str(h)] = {
                "rain_probability": torch.sigmoid(raw[:, 0]),
                "precip_mm": torch.relu(raw[:, 1]),
                "temperature_c": raw[:, 2],
                "wind_speed_kmh": torch.relu(raw[:, 3]),
                "zonda_risk": torch.sigmoid(raw[:, 4]),
                "storm_risk": torch.sigmoid(raw[:, 5]),
                "hail_risk": torch.sigmoid(raw[:, 6]),
            }
        return result

    @property
    def n_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Focal BCE with label smoothing ─────────────────────────────
class FocalBCE(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: float = 3.0, label_smoothing: float = 0.02):
        super().__init__()
        self.gamma = gamma
        self.pw = pos_weight
        self.label_smoothing = label_smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Disable autocast — BCE is unsafe under AMP even with float32 cast
        with torch.amp.autocast(pred.device.type, enabled=False):
            pred_f32 = pred.float()
            target_f32 = target.float()
            # Label smoothing: prevent overconfident predictions on rare events
            target_smooth = target_f32 * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
            bce = F.binary_cross_entropy(pred_f32, target_smooth, reduction="none")
            pt = torch.where(target_smooth == 1, pred_f32, 1 - pred_f32)
            focal = (1 - pt) ** self.gamma
            weight = torch.where(target_smooth == 1, self.pw, 1.0)
            return (focal * weight * bce).mean()


# ── Horizon-Weighted Loss ───────────────────────────────────────
class WeatherLossV4(nn.Module):
    """
    Loss that applies horizon weights during training.
    Short horizons (3h, 6h) get higher weight than long (48h, 168h).
    Uses Huber loss with delta=5.0 for temperature.
    """
    def __init__(self, horizons: list = None, horizon_weights: dict = None):
        super().__init__()
        self.horizons = horizons or HORIZONS
        self.horizon_weights = horizon_weights or HORIZON_WEIGHTS
        self.focal_rain = FocalBCE(gamma=2.0, pos_weight=3.0, label_smoothing=0.02)
        self.focal_extreme = FocalBCE(gamma=2.0, pos_weight=2.0, label_smoothing=0.02)
        self.mse = nn.MSELoss()
        self.huber_temp = nn.HuberLoss(delta=5.0)

    def forward(self, preds: Dict[str, Dict[str, torch.Tensor]],
                targets: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        device = preds[str(self.horizons[0])]["rain_probability"].device
        total_loss = torch.zeros(1, device=device, requires_grad=True)
        weight_sum = 0.0

        for h in self.horizons:
            hk = str(h)
            hw = self.horizon_weights.get(h, 1.0)
            p = preds[hk]
            t = targets[hk]

            loss_h = torch.zeros(1, device=device, requires_grad=True)
            if "rain_probability" in t:
                loss_h = loss_h + 1.5 * self.focal_rain(p["rain_probability"], t["rain_probability"])
            if "precip_mm" in t:
                loss_h = loss_h + 0.4 * self.mse(p["precip_mm"], t["precip_mm"])
            if "temperature_c" in t:
                loss_h = loss_h + 0.8 * self.huber_temp(p["temperature_c"], t["temperature_c"])
            if "wind_speed_kmh" in t:
                loss_h = loss_h + 0.3 * self.mse(p["wind_speed_kmh"], t["wind_speed_kmh"])
            for key in ("zonda_risk", "storm_risk", "hail_risk"):
                if key in t:
                    loss_h = loss_h + 1.0 * self.focal_extreme(p[key], t[key])

            total_loss = total_loss + hw * loss_h
            weight_sum += hw

        return total_loss / weight_sum


# ── NASA POWER Download (single zone) ──────────────────────────
async def download_nasa_power_zone(lat: float, lng: float, years_back: int) -> pd.DataFrame:
    """Download NASA POWER hourly data for a single coordinate point."""
    import httpx

    end = datetime(2026, 7, 31, tzinfo=timezone.utc)
    start = end.replace(year=end.year - years_back)

    chunks = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start.replace(year=chunk_start.year + 3), end)
        params = {
            "parameters": "T2M,RH2M,WS10M,WD10M,PS,PRECTOTCORR,ALLSKY_SFC_SW_DWN",
            "community": "RE",
            "longitude": lng,
            "latitude": lat,
            "start": chunk_start.strftime("%Y%m%d"),
            "end": chunk_end.strftime("%Y%m%d"),
            "format": "CSV",
        }
        url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=180) as client:
                    resp = await client.get(url, params=params)
                    resp.raise_for_status()
                break
            except Exception as e:
                if attempt == 2:
                    logger.warning(f"  Chunk failed for ({lat},{lng}): {e}, skipping")
                    chunk_start = chunk_end + pd.Timedelta(days=1)
                    break
                await asyncio.sleep(5)
        else:
            chunk_start = chunk_end + pd.Timedelta(days=1)
            continue

        lines = resp.text.strip().split("\n")
        data_start = next(i for i, l in enumerate(lines) if "-END HEADER-" in l) + 1
        chunk_df = pd.read_csv(__import__("io").StringIO("\n".join(lines[data_start:])))
        chunk_df["timestamp"] = pd.to_datetime(
            chunk_df[["YEAR", "MO", "DY", "HR"]].rename(
                columns={"YEAR": "year", "MO": "month", "DY": "day", "HR": "hour"}
            )
        )
        chunk_df = chunk_df.drop(columns=["YEAR", "MO", "DY", "HR"])
        chunks.append(chunk_df)
        chunk_start = chunk_end + pd.Timedelta(days=1)

    if not chunks:
        logger.warning(f"  No data downloaded for ({lat},{lng})")
        return pd.DataFrame()

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
    df["temperature_min_c"] = df["temperature_c"].rolling(24, min_periods=1).min()
    df["temperature_max_c"] = df["temperature_c"].rolling(24, min_periods=1).max()
    df = df.replace(-999.0, np.nan)
    logger.info(f"  Zone ({lat},{lng}): {len(df):,} rows")
    return df


# ── Download all zones (serial to avoid rate-limiting) ──────────
async def download_all_zones(years_back: int) -> Dict[int, pd.DataFrame]:
    """Download NASA POWER for all 10 zones. Returns dict of zone_id → DataFrame."""
    logger.info(f"Downloading NASA POWER for {len(ZONE_POINTS)} zones ({years_back} years)...")
    dfs = {}
    for i, zp in enumerate(ZONE_POINTS):
        logger.info(f"  [{i+1}/{len(ZONE_POINTS)}] {zp['zone_name']} ({zp['lat']}, {zp['lng']})...")
        df = await download_nasa_power_zone(zp["lat"], zp["lng"], years_back)
        if not df.empty:
            dfs[i] = df
    return dfs


# ── Zone classification helpers ────────────────────────────────
def _is_mountain_zone(zone_id: int) -> bool:
    """Check if zone is a mountain zone (Tafí Viejo, Chicligasta)."""
    return ZONE_POINTS[zone_id].get("is_mountain", 0.0) == 1.0


def _is_valley_zone(zone_id: int) -> bool:
    """Check if zone is a valley zone (Concepción, Monteros)."""
    name = ZONE_POINTS[zone_id].get("zone_name", "")
    return name in ("Concepción", "Monteros")


def _is_urban_zone(zone_id: int) -> bool:
    """Check if zone is an urban zone (Capital, Yerba Buena, Cruz Alta, Capital Norte, Bella Vista)."""
    name = ZONE_POINTS[zone_id].get("zone_name", "")
    return name in ("Capital", "Yerba Buena", "Cruz Alta", "Capital Norte", "Bella Vista")


# ── Gaussian bell curve intensity ──────────────────────────────
def _gaussian_intensity(p: float, sigma: float = 0.28) -> float:
    """
    Gaussian bell curve intensity profile. Peak at p=0.5 (middle of main phase).
    Replaces the sine curve from v4.0 for more realistic event shapes.
    """
    center = 0.5
    return float(np.exp(-0.5 * ((p - center) / sigma) ** 2))


# ── Improved Synthetic Events (v5) ─────────────────────────────
def generate_synthetic_extremes_v5(
    n_events: int = 3000,
    n_zones: int = 10,
) -> pd.DataFrame:
    """
    Generate synthetic extreme weather events with realistic structure:
    - Pre-storm buildup (gradual pressure drop over 6h before peak)
    - Post-storm recovery (gradual clearing over 6h)
    - Mixed events: zonda→storm transition (20% chance)
    - Autocorrelated noise (random walk, not uniform)
    - NEW: cold_front events (10%) — temperature drops, high wind, moderate precip
    - NEW: multi_day_storm events (5%) — 48-72h intermittent heavy rain
    - Gaussian bell curve intensity profiles (replaces sine)
    - Zone-specific base temperatures based on altitude

    Each event is tagged with a zone_id so it can be distributed across zones.
    """
    logger.info(f"Generating {n_events} synthetic extreme events (v5) across {n_zones} zones")
    rows = []
    rng = np.random.default_rng(42)

    # Event type distribution — 7 types now
    event_weights = {
        "zonda": 0.18,
        "hail_storm": 0.18,
        "heat_wave": 0.12,
        "extreme_rain": 0.18,
        "cold_front": 0.10,
        "multi_day_storm": 0.05,
        "normal_volatile": 0.19,
    }

    # Persistent noise state for autocorrelated noise
    noise_temp = 0.0
    noise_humid = 0.0
    noise_wind = 0.0

    # Zone-specific altitude-based base temperature adjustments
    # Capital (450m) is the reference at 28°C
    # Each 100m of altitude ≈ -0.65°C (lapse rate)
    def _zone_base_temp(zone_id: int) -> float:
        alt = ZONE_POINTS[zone_id]["altitude_m"]
        return 28.0 - 0.65 * (alt - 450.0) / 100.0

    for event_idx in range(n_events):
        # Assign each event to a zone — uniform distribution across all zones
        zone_id = int(event_idx % n_zones)

        event_type = rng.choice(
            list(event_weights.keys()), p=list(event_weights.values())
        )

        # Duration depends on event type
        if event_type == "multi_day_storm":
            duration = rng.integers(48, 73)  # 48-72 hours
        elif event_type == "cold_front":
            duration = rng.integers(12, 36)  # 12-36 hours
        elif event_type == "heat_wave":
            duration = rng.integers(24, 72)  # 24-72 hours
        elif event_type == "zonda":
            duration = rng.integers(8, 24)  # 8-24 hours
        else:
            duration = rng.integers(8, 48)

        start = pd.Timestamp("2001-01-01") + pd.Timedelta(
            days=int(rng.integers(0, 365 * 20))
        )

        # Pre-event buildup (6h before peak) + main event + post-event decay (6h)
        buildup_h = 6
        decay_h = 6
        total_h = buildup_h + duration + decay_h

        is_mixed = (event_type == "zonda") and (rng.random() < 0.20)

        # Zone-specific base temperature
        base_temp = _zone_base_temp(zone_id)

        for h in range(total_h):
            t = start + pd.Timedelta(hours=h)
            # Normalized progress within each phase
            if h < buildup_h:
                phase = "buildup"
                p = h / buildup_h  # 0→1
            elif h < buildup_h + duration:
                phase = "main"
                p = (h - buildup_h) / max(duration - 1, 1)  # 0→1
            else:
                phase = "decay"
                p = (h - buildup_h - duration) / decay_h  # 0→1

            # Autocorrelated noise (random walk)
            noise_temp += rng.normal(0, 0.15)
            noise_temp *= 0.95  # mean-revert
            noise_humid += rng.normal(0, 0.2)
            noise_humid *= 0.95
            noise_wind += rng.normal(0, 0.1)
            noise_wind *= 0.95

            # ── Zonda ──
            if event_type == "zonda":
                if phase == "main":
                    intensity = _gaussian_intensity(p)
                elif phase == "buildup":
                    intensity = 0.3 * p
                else:
                    intensity = 0.3 * (1 - p)

                if is_mixed and phase == "main" and p > 0.6:
                    # Zonda → storm transition (real in Tucumán)
                    storm_t = (p - 0.6) / 0.4
                    row = {
                        "timestamp": t,
                        "temperature_c": (base_temp + 17 * intensity * (1 - storm_t)
                                          - 5 * storm_t + noise_temp),
                        "humidity_pct": (15 + 50 * (1 - intensity) * (1 - storm_t)
                                        + 60 * storm_t + noise_humid),
                        "wind_speed_kmh": (30 + 80 * intensity * (1 - storm_t * 0.5)
                                          + noise_wind),
                        "pressure_hpa": (1000 - 10 * intensity * (1 - storm_t)
                                        - 12 * storm_t + rng.normal(0, 0.3)),
                        "precip_mm": max(0, 20 * storm_t + rng.normal(0, 2)),
                        "cape_j_kg": 100 * intensity * (1 - storm_t) + 3000 * storm_t,
                        "cordillera_pressure_hpa": 890 - 20 * intensity,
                        "thermal_differential_c": 5 + 23 * intensity * (1 - storm_t),
                        "wind_direction_deg": 270 if storm_t < 0.5 else 180,
                        "_label_zonda": 1.0 if intensity > 0.4 and storm_t < 0.5 else 0.0,
                        "_label_storm": 1.0 if storm_t > 0.3 else 0.0,
                        "_label_hail": 1.0 if storm_t > 0.6 else 0.0,
                    }
                else:
                    row = {
                        "timestamp": t,
                        "temperature_c": base_temp + 17 * intensity + noise_temp,
                        "humidity_pct": 15 + 50 * (1 - intensity) + noise_humid,
                        "wind_speed_kmh": 30 + 80 * intensity + noise_wind,
                        "pressure_hpa": 1000 - 10 * intensity + rng.normal(0, 0.3),
                        "precip_mm": 0.0,
                        "cape_j_kg": 100 * intensity,
                        "cordillera_pressure_hpa": 890 - 20 * intensity,
                        "thermal_differential_c": 5 + 23 * intensity,
                        "wind_direction_deg": 270 + rng.uniform(-15, 15),
                        "_label_zonda": 1.0 if intensity > 0.4 else 0.0,
                        "_label_storm": 0.0,
                        "_label_hail": 0.0,
                    }

            # ── Hail Storm ──
            elif event_type == "hail_storm":
                if phase == "main":
                    intensity = _gaussian_intensity(p)
                elif phase == "buildup":
                    intensity = 0.2 * p
                else:
                    intensity = 0.2 * (1 - p)

                # Mountain zones get more hail (+30% intensity boost)
                if _is_mountain_zone(zone_id):
                    intensity = min(1.0, intensity * 1.3)

                row = {
                    "timestamp": t,
                    "temperature_c": base_temp - 5 * intensity + noise_temp,
                    "humidity_pct": 80 + 15 * intensity + noise_humid,
                    "wind_speed_kmh": 25 + 55 * intensity + noise_wind,
                    "pressure_hpa": 1005 - 12 * intensity + rng.normal(0, 0.3),
                    "precip_mm": (15 + 35 * intensity + rng.normal(0, 2)
                                  if intensity > 0.3 else rng.uniform(0, 3)),
                    "cape_j_kg": 1000 + 4000 * intensity,
                    "cordillera_pressure_hpa": 900 + rng.uniform(-3, 3),
                    "thermal_differential_c": 5 + 7 * intensity,
                    "wind_direction_deg": 180 + rng.uniform(-30, 30),
                    "_label_zonda": 0.0,
                    "_label_storm": 1.0,
                    "_label_hail": 1.0 if intensity > 0.55 else 0.0,
                }

            # ── Heat Wave ──
            elif event_type == "heat_wave":
                if phase == "main":
                    intensity = _gaussian_intensity(p, sigma=0.35)
                elif phase == "buildup":
                    intensity = 0.4 * p
                else:
                    intensity = 0.4 * (1 - p)

                row = {
                    "timestamp": t,
                    "temperature_c": 33 + 11 * intensity + noise_temp,
                    "humidity_pct": 25 + 10 * (1 - intensity) + noise_humid,
                    "wind_speed_kmh": 8 + 5 * intensity + noise_wind,
                    "pressure_hpa": 1010 + 3 * (1 - intensity) + rng.normal(0, 0.3),
                    "precip_mm": 0.0,
                    "cape_j_kg": 200 + 600 * intensity,
                    "cordillera_pressure_hpa": 898 + rng.uniform(-2, 2),
                    "thermal_differential_c": 8 + 10 * intensity,
                    "_label_zonda": 0.0,
                    "_label_storm": 0.0,
                    "_label_hail": 0.0,
                }

            # ── Extreme Rain ──
            elif event_type == "extreme_rain":
                if phase == "main":
                    intensity = _gaussian_intensity(p)
                elif phase == "buildup":
                    intensity = 0.15 * p
                else:
                    intensity = 0.15 * (1 - p)

                # Valley zones get more extreme rain (+20%)
                if _is_valley_zone(zone_id):
                    intensity = min(1.0, intensity * 1.2)

                row = {
                    "timestamp": t,
                    "temperature_c": base_temp - 4 * intensity + noise_temp,
                    "humidity_pct": 90 + 10 * intensity + noise_humid,
                    "wind_speed_kmh": 15 + 30 * intensity + noise_wind,
                    "pressure_hpa": 1002 - 8 * intensity + rng.normal(0, 0.3),
                    "precip_mm": (20 + 60 * intensity + rng.normal(0, 3)
                                  if intensity > 0.2 else rng.uniform(0, 5)),
                    "cape_j_kg": 800 + 3200 * intensity,
                    "cordillera_pressure_hpa": 902 + rng.uniform(-2, 2),
                    "thermal_differential_c": 3 + 4 * intensity,
                    "_label_zonda": 0.0,
                    "_label_storm": 1.0,
                    "_label_hail": 0.0,
                }

            # ── Cold Front (NEW in v5) ──
            elif event_type == "cold_front":
                if phase == "main":
                    intensity = _gaussian_intensity(p, sigma=0.30)
                elif phase == "buildup":
                    intensity = 0.25 * p
                else:
                    intensity = 0.25 * (1 - p)

                row = {
                    "timestamp": t,
                    "temperature_c": base_temp - 12 * intensity + noise_temp,
                    "humidity_pct": 60 + 20 * intensity + noise_humid,
                    "wind_speed_kmh": 35 + 40 * intensity + noise_wind,
                    "pressure_hpa": 1008 - 15 * intensity + rng.normal(0, 0.4),
                    "precip_mm": (8 + 25 * intensity + rng.normal(0, 2)
                                  if intensity > 0.2 else rng.uniform(0, 2)),
                    "cape_j_kg": 200 + 800 * intensity,
                    "cordillera_pressure_hpa": 895 + rng.uniform(-3, 3),
                    "thermal_differential_c": 2 + 3 * intensity,
                    "wind_direction_deg": 180 + rng.uniform(-45, 45),  # S-SE wind
                    "_label_zonda": 0.0,
                    "_label_storm": 1.0 if intensity > 0.5 else 0.0,
                    "_label_hail": 0.0,
                }

            # ── Multi-Day Storm (NEW in v5) ──
            elif event_type == "multi_day_storm":
                # Multiple peaks within the long duration
                # Use a modulated Gaussian to create intermittent heavy rain
                n_peaks = max(2, duration // 12)
                peak_spacing = duration / n_peaks
                closest_dist = min(
                    abs(p - (k * peak_spacing) / max(duration - 1, 1))
                    for k in range(n_peaks)
                )
                intensity = float(np.exp(-0.5 * (closest_dist / 0.12) ** 2))
                # Add a slow baseline
                intensity = max(intensity * 0.9, 0.15 * np.sin(p * np.pi))

                # Valley zones get more intense multi-day storms
                if _is_valley_zone(zone_id):
                    intensity = min(1.0, intensity * 1.15)

                row = {
                    "timestamp": t,
                    "temperature_c": base_temp - 3 * intensity + noise_temp,
                    "humidity_pct": 85 + 15 * intensity + noise_humid,
                    "wind_speed_kmh": 10 + 35 * intensity + noise_wind,
                    "pressure_hpa": 1000 - 8 * intensity + rng.normal(0, 0.3),
                    "precip_mm": (15 + 50 * intensity + rng.normal(0, 3)
                                  if intensity > 0.25 else rng.uniform(0, 5)),
                    "cape_j_kg": 500 + 2500 * intensity,
                    "cordillera_pressure_hpa": 900 + rng.uniform(-4, 4),
                    "thermal_differential_c": 2 + 3 * intensity,
                    "_label_zonda": 0.0,
                    "_label_storm": 1.0,
                    "_label_hail": 1.0 if intensity > 0.7 and rng.random() < 0.15 else 0.0,
                }

            # ── Normal Volatile ──
            else:  # normal_volatile
                row = {
                    "timestamp": t,
                    "temperature_c": 15 + 15 * np.sin(p * np.pi * 2) + noise_temp,
                    "humidity_pct": 40 + 30 * rng.uniform(0, 1) + noise_humid,
                    "wind_speed_kmh": 5 + 20 * rng.uniform(0, 1) + noise_wind,
                    "pressure_hpa": 1008 + rng.normal(0, 2),
                    "precip_mm": rng.choice([0, 0, 0, 0, rng.uniform(1, 15)]),
                    "cape_j_kg": rng.uniform(50, 1500),
                    "cordillera_pressure_hpa": 900 + rng.uniform(-5, 5),
                    "thermal_differential_c": rng.uniform(1, 10),
                    "_label_zonda": 0.0,
                    "_label_storm": 0.0,
                    "_label_hail": 0.0,
                }

            row["_zone_id"] = zone_id
            rows.append(row)

    return pd.DataFrame(rows)


# ── Distribute synthetic events across zones ───────────────────
def distribute_synthetic_events(
    df_synth: pd.DataFrame,
    dfs_by_zone: Dict[int, pd.DataFrame],
) -> Dict[int, pd.DataFrame]:
    """
    Split synthetic events into per-zone chunks and merge each into the
    correct zone's DataFrame. This is the critical fix over v4.0 which
    only added all synthetic events to Capital (zone 0).
    """
    logger.info("Distributing synthetic events across all zones...")

    # Group synthetic events by zone_id
    zone_groups = df_synth.groupby("_zone_id")

    for zone_id in sorted(dfs_by_zone.keys()):
        if zone_id in zone_groups.groups:
            zone_synth = zone_groups.get_group(zone_id).copy()
            # Drop the _zone_id column before merging into the zone DataFrame
            zone_synth = zone_synth.drop(columns=["_zone_id"], errors="ignore")
            dfs_by_zone[zone_id] = pd.concat(
                [dfs_by_zone[zone_id], zone_synth], ignore_index=True
            ).sort_index()
            logger.info(
                f"  Zone {zone_id} ({ZONE_POINTS[zone_id]['zone_name']}): "
                f"+{len(zone_synth):,} synthetic events"
            )
        else:
            logger.warning(f"  Zone {zone_id}: no synthetic events assigned")

    return dfs_by_zone


# ── Preprocessing (inlined from data_pipeline.py) ───────────────
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Inline preprocessing — matches app/ml/pipeline/data_pipeline.py."""
    if "timestamp" not in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        return df

    if isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
    elif "timestamp" in df.columns:
        df = df.set_index("timestamp").sort_index()

    df.index.name = "timestamp"
    df = df[~df.index.duplicated(keep="first")]
    df = df[df.index.notnull()]

    # Interpolate
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method="time", limit=6)
    df[numeric_cols] = df[numeric_cols].ffill(limit=12)
    df[numeric_cols] = df[numeric_cols].bfill(limit=12)
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Cyclic features
    hour = df.index.hour + df.index.minute / 60
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    doy = df.index.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365)

    # Derived
    if "solar_radiation_wm2" in df.columns:
        df["uv_index"] = (df["solar_radiation_wm2"] / 1000 * 11).clip(0, 12)
    if "temperature_c" in df.columns:
        df["dew_point_c"] = df["temperature_c"] - (100 - df.get("humidity_pct", 50)) / 5
        df["feels_like_c"] = df["temperature_c"]

    # CAPE proxy
    if "cape_j_kg" not in df.columns:
        if "temperature_c" in df.columns and "dew_point_c" in df.columns:
            df["cape_j_kg"] = ((df["temperature_c"] - df["dew_point_c"]) * 50).clip(0)
        else:
            df["cape_j_kg"] = 0.0

    # Zonda proxy
    if "cordillera_pressure_hpa" not in df.columns:
        df["cordillera_pressure_hpa"] = df.get("pressure_hpa", 1013) * 0.88
    if "thermal_differential_c" not in df.columns:
        df["thermal_differential_c"] = 5.0

    # Static zone features (defaults)
    for col, default in [
        ("altitude_m", 450), ("impermeable_pct", 0.6), ("is_mountain", 0),
        ("enso_index", 0), ("ndvi", 0.5), ("soil_temp_c", 20),
        ("precipitable_water_mm", 25),
    ]:
        if col not in df.columns:
            df[col] = default

    # Fill remaining
    for col, default in [
        ("visibility_km", 10), ("cloud_cover_pct", 50),
        ("precip_3h_mm", 0), ("precip_6h_mm", 0), ("precip_24h_mm", 0),
        ("k_index", 0), ("pressure_sea_level_hpa", 1013),
    ]:
        if col not in df.columns:
            df[col] = default

    return df


# ── Enhanced Features v4 (22 new features) ─────────────────────
def add_enhanced_features_v4(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 22 enhanced features: 14 from v2.0 + 6 from v4.0 + 2 new in v6.0.
    The 6 v4.0 features:
      - pressure_change_6h: 6h pressure tendency (storm prediction)
      - humid_dewpoint_spread: T-Td (air mass dryness)
      - temp_amplitude_12h: 12h diurnal range
      - wind_shear_approx: surface vs derived upper wind
      - cape_lag_3h: CAPE tendency
      - is_daytime: binary hour 6-18
    The 2 new v6.0 features:
      - temp_3day_avg: 72h rolling average temperature (multi-day trends)
      - pressure_3day_avg: 72h rolling average pressure (multi-day trends)
    """
    # ── v2.0 existing 14 features ──
    if "temperature_c" in df.columns:
        df["temp_lag_1h"] = df["temperature_c"].shift(1)
        df["temp_lag_3h"] = df["temperature_c"].shift(3)
        df["temp_tendency"] = df["temperature_c"].diff(3)
        df["temp_range_24h"] = (
            df["temperature_c"].rolling(24, min_periods=1).max()
            - df["temperature_c"].rolling(24, min_periods=1).min()
        )

    if "humidity_pct" in df.columns:
        df["humid_lag_1h"] = df["humidity_pct"].shift(1)
        df["humid_tendency"] = df["humidity_pct"].diff(3)

    if "pressure_hpa" in df.columns:
        df["pressure_tendency_3h"] = df["pressure_hpa"].diff(3)
        df["pressure_tendency_24h"] = df["pressure_hpa"].diff(24)

    if "wind_speed_kmh" in df.columns:
        df["wind_lag_1h"] = df["wind_speed_kmh"].shift(1)

    if "precip_mm" in df.columns:
        df["precip_lag_1h"] = df["precip_mm"].shift(1)
        df["precip_tendency"] = df["precip_mm"].diff(3)

    if "temperature_c" in df.columns and "humidity_pct" in df.columns:
        df["temp_humid_index"] = df["temperature_c"] * df["humidity_pct"] / 100

    if "cape_j_kg" in df.columns and "humidity_pct" in df.columns:
        df["instability_index"] = df["cape_j_kg"] * df["humidity_pct"] / 10000

    if "wind_speed_kmh" in df.columns and "pressure_hpa" in df.columns:
        df["wind_pressure_ratio"] = df["wind_speed_kmh"] / df["pressure_hpa"].clip(lower=900)

    # ── v4.0 new 6 features ──
    if "pressure_hpa" in df.columns:
        df["pressure_change_6h"] = df["pressure_hpa"].diff(6)

    if "temperature_c" in df.columns and "dew_point_c" in df.columns:
        df["humid_dewpoint_spread"] = df["temperature_c"] - df["dew_point_c"]

    if "temperature_c" in df.columns:
        df["temp_amplitude_12h"] = (
            df["temperature_c"].rolling(12, min_periods=1).max()
            - df["temperature_c"].rolling(12, min_periods=1).min()
        )
        df["is_daytime"] = ((df.index.hour >= 6) & (df.index.hour <= 18)).astype(float)

    if "cape_j_kg" in df.columns:
        df["cape_lag_3h"] = df["cape_j_kg"].shift(3)

    if "wind_speed_kmh" in df.columns:
        df["wind_shear_approx"] = (
            df["wind_speed_kmh"].rolling(6, min_periods=1).max()
            - df["wind_speed_kmh"]
        )

    # ── v6.0 new 2 features: 72h rolling averages ──
    if "temperature_c" in df.columns:
        df["temp_3day_avg"] = df["temperature_c"].rolling(72, min_periods=1).mean()

    if "pressure_hpa" in df.columns:
        df["pressure_3day_avg"] = df["pressure_hpa"].rolling(72, min_periods=1).mean()

    # Fill NaN from lag features
    df = df.ffill(limit=6).bfill(limit=3).fillna(0)
    return df


# ── Build Tensors from Multi-Zone Data ─────────────────────────
def build_tensors_multi_zone(
    dfs_by_zone: Dict[int, pd.DataFrame],
    lookback: int = LOOKBACK,
    horizons: list = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], list, list, dict]:
    """
    Build training tensors from multiple zone DataFrames.

    Each zone's data gets a zone_id column (0-9).
    Static features altitude_m and is_mountain vary per zone.
    Returns: X, y, feat_cols, zone_ids_per_sample, sample_info
    """
    horizons = horizons or HORIZONS
    max_h = max(horizons)

    # First, preprocess and enhance each zone independently
    zone_data = {}
    for zone_id, df in dfs_by_zone.items():
        zp = ZONE_POINTS[zone_id]
        df_processed = preprocess(df.copy())
        df_processed["altitude_m"] = zp["altitude_m"]
        df_processed["is_mountain"] = zp["is_mountain"]
        df_processed = add_enhanced_features_v4(df_processed)
        df_processed["zone_id"] = float(zone_id)
        zone_data[zone_id] = df_processed

    # Get feature columns (same across zones after preprocessing)
    sample_df = list(zone_data.values())[0]
    skip_cols = {
        "_label_zonda", "_label_storm", "_label_hail",
        "timestamp", "year", "month", "day", "zone_id",
    }
    feat_cols = [c for c in sample_df.columns if c not in skip_cols]

    # Build tensors per zone, then concatenate
    all_X = []
    all_y = {str(h): [] for h in horizons}
    all_zone_ids = []
    samples_per_zone = {}

    for zone_id, df in zone_data.items():
        data = df[feat_cols].values.astype(np.float32)
        n_samples = len(data) - lookback - max_h
        if n_samples <= 0:
            logger.warning(f"  Zone {zone_id}: not enough data ({len(data)} rows), skipping")
            continue

        X_zone = np.zeros((n_samples, lookback, len(feat_cols)), dtype=np.float32)
        y_zone = {str(h): np.zeros((n_samples, 7), dtype=np.float32) for h in horizons}

        temp_idx = feat_cols.index("temperature_c") if "temperature_c" in feat_cols else 0
        precip_idx = feat_cols.index("precip_mm") if "precip_mm" in feat_cols else 0
        wind_idx = feat_cols.index("wind_speed_kmh") if "wind_speed_kmh" in feat_cols else 0

        for i in range(n_samples):
            X_zone[i] = data[i:i + lookback]
            for hi, h in enumerate(horizons):
                target_row = data[i + lookback + h - 1]
                label_z = df.iloc[i + lookback + h - 1].get("_label_zonda", 0.0)
                label_s = df.iloc[i + lookback + h - 1].get("_label_storm", 0.0)
                label_h = df.iloc[i + lookback + h - 1].get("_label_hail", 0.0)
                y_zone[str(h)][i] = [
                    1.0 if target_row[precip_idx] > 1.0 else 0.0,
                    max(0, float(target_row[precip_idx])),
                    float(target_row[temp_idx]),
                    max(0, float(target_row[wind_idx])),
                    float(label_z),
                    float(label_s),
                    float(label_h),
                ]

        all_X.append(X_zone)
        for h in horizons:
            all_y[str(h)].append(y_zone[str(h)])
        all_zone_ids.extend([zone_id] * n_samples)
        samples_per_zone[zone_id] = n_samples
        logger.info(f"  Zone {zone_id} ({ZONE_POINTS[zone_id]['zone_name']}): {n_samples:,} samples")

    if not all_X:
        raise ValueError("No zones had enough data to build tensors")

    X = np.concatenate(all_X, axis=0)
    y = {h: np.concatenate(arrs, axis=0) for h, arrs in all_y.items()}
    zone_ids = np.array(all_zone_ids, dtype=np.int64)

    logger.info(f"  Total: {len(X):,} samples, {len(feat_cols)} features")
    return (
        torch.tensor(X, dtype=torch.float32),
        {h: torch.tensor(v, dtype=torch.float32) for h, v in y.items()},
        feat_cols,
        zone_ids,
        samples_per_zone,
    )


# ── Temporal K-Fold CV ─────────────────────────────────────────
def temporal_kfold_cv(
    n_total: int, n_folds: int = 3
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Split the full timeline into n_folds equal parts.
    For fold k: train on all parts except k, validate on part k.
    This ensures temporal ordering is preserved.
    """
    fold_size = n_total // n_folds
    folds = []
    for k in range(n_folds):
        val_start = k * fold_size
        val_end = (k + 1) * fold_size if k < n_folds - 1 else n_total
        val_idx = np.arange(val_start, val_end)
        train_idx = np.concatenate([
            np.arange(0, val_start),
            np.arange(val_end, n_total),
        ])
        folds.append((train_idx, val_idx))
    return folds


# ── Evaluation v4 ──────────────────────────────────────────────
def evaluate_v4(
    all_preds: Dict[str, np.ndarray],
    all_targets: Dict[str, np.ndarray],
    horizons: list,
    horizon_weights: dict = None,
) -> dict:
    """
    Evaluate predictions with horizon-weighted composite.
    Temperature score uses seasonal normalization: RMSE / SEASONAL_TEMP_RANGE.
    Also computes directional accuracy for temperature.
    """
    horizon_weights = horizon_weights or HORIZON_WEIGHTS
    metrics = {}
    names = [
        "rain_probability", "precip_mm", "temperature_c", "wind_speed_kmh",
        "zonda_risk", "storm_risk", "hail_risk",
    ]

    for h in horizons:
        hk = str(h)
        p, t = all_preds[hk], all_targets[hk]
        hm = {}

        for i, name in enumerate(names):
            pv, tv = p[:, i], t[:, i]
            if name in ("rain_probability", "zonda_risk", "storm_risk", "hail_risk"):
                pb = (pv > 0.5).astype(int)
                tb = (tv > 0.5).astype(int)
                hm[name] = {
                    "accuracy": float(accuracy_score(tb, pb)),
                    "precision": float(precision_score(tb, pb, zero_division=0)),
                    "recall": float(recall_score(tb, pb, zero_division=0)),
                    "f1": float(f1_score(tb, pb, zero_division=0)),
                }
            else:
                rmse = float(np.sqrt(mean_squared_error(tv, pv)))
                mae = float(mean_absolute_error(tv, pv))
                hm[name] = {"rmse": rmse, "mae": mae}

        # Seasonal temperature score
        temp_rmse = hm.get("temperature_c", {}).get("rmse", 10.0)
        temp_score = max(0.0, 1.0 - temp_rmse / SEASONAL_TEMP_RANGE)

        # Directional accuracy for temperature
        temp_p = all_preds[hk][:, names.index("temperature_c")]
        temp_t = all_targets[hk][:, names.index("temperature_c")]
        if len(temp_p) > 1:
            pred_dir = np.sign(np.diff(temp_p))
            true_dir = np.sign(np.diff(temp_t))
            directional_acc = float(np.mean(pred_dir == true_dir))
        else:
            directional_acc = 0.5
        hm["temp_directional_accuracy"] = directional_acc

        # Composite
        binary_accs = [
            hm[k]["accuracy"]
            for k in ("rain_probability", "zonda_risk", "storm_risk", "hail_risk")
        ]
        hm["composite_accuracy"] = float(np.mean(binary_accs + [temp_score]))
        hm["temp_seasonal_score"] = temp_score
        metrics[f"{h}h"] = hm

    return metrics


# ── Collect Predictions ─────────────────────────────────────────
def collect_preds(
    model: nn.Module,
    X: torch.Tensor,
    y: Dict[str, torch.Tensor],
    horizons: list,
    device: str = "cpu",
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Collect predictions and targets for evaluation."""
    model.eval()
    all_p = {str(h): [] for h in horizons}
    all_t = {str(h): [] for h in horizons}

    def flatten(d):
        return torch.cat([d[str(h)] for h in horizons], dim=-1)

    def unflatten(flat):
        r = {}
        for i, h in enumerate(horizons):
            c = flat[:, i * 7:(i + 1) * 7]
            r[str(h)] = {
                "rain_probability": c[:, 0], "precip_mm": c[:, 1],
                "temperature_c": c[:, 2], "wind_speed_kmh": c[:, 3],
                "zonda_risk": c[:, 4], "storm_risk": c[:, 5],
                "hail_risk": c[:, 6],
            }
        return r

    ds = TensorDataset(X, flatten(y))
    loader = DataLoader(ds, batch_size=512, shuffle=False)

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            preds = model(xb)
            tgts = unflatten(yb.cpu())
            for h in horizons:
                hk = str(h)
                pa = torch.stack([
                    preds[hk]["rain_probability"], preds[hk]["precip_mm"],
                    preds[hk]["temperature_c"], preds[hk]["wind_speed_kmh"],
                    preds[hk]["zonda_risk"], preds[hk]["storm_risk"],
                    preds[hk]["hail_risk"],
                ], dim=1).cpu().numpy()
                ta = torch.stack([
                    tgts[hk]["rain_probability"], tgts[hk]["precip_mm"],
                    tgts[hk]["temperature_c"], tgts[hk]["wind_speed_kmh"],
                    tgts[hk]["zonda_risk"], tgts[hk]["storm_risk"],
                    tgts[hk]["hail_risk"],
                ], dim=1).numpy()
                all_p[hk].append(pa)
                all_t[hk].append(ta)

    for h in horizons:
        hk = str(h)
        all_p[hk] = np.concatenate(all_p[hk])
        all_t[hk] = np.concatenate(all_t[hk])
    return all_p, all_t


# ── MixUp Augmentation (new in v6) ────────────────────────────
def mixup_data(X: torch.Tensor, y_flat: torch.Tensor, alpha: float = 0.2):
    """
    Apply MixUp augmentation: blend random pairs of samples.
    Returns mixed X, mixed y, and the lambda used (for loss scaling).
    During training only — regularization without adding data.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = X.size(0)
    index = torch.randperm(batch_size, device=X.device)

    X_mixed = lam * X + (1 - lam) * X[index]
    y_mixed = lam * y_flat + (1 - lam) * y_flat[index]

    return X_mixed, y_mixed, lam


# ── CosineAnnealingWarmRestarts with warmup wrapper ────────────
class WarmupCosineAnnealingWarmRestarts(torch.optim.lr_scheduler.LambdaLR):
    """
    CosineAnnealingWarmRestarts with first 5 epochs warmup.
    Restarts at epochs 15, 45, 105 (T_0=15, T_mult=2).
    """
    pass


# ── Single Fold Training ───────────────────────────────────────
def train_v5_fold(
    X_train: torch.Tensor,
    y_train: Dict[str, torch.Tensor],
    X_val: torch.Tensor,
    y_val: Dict[str, torch.Tensor],
    horizons: list,
    max_epochs: int = 150,
    batch_size: int = 512,
    lr: float = 1.5e-3,
    device: str = "cpu",
    fold_num: int = 0,
) -> nn.Module:
    """Train a single fold of the temporal CV using FastWeatherModelV5Light."""
    model = FastWeatherModelV5Light(
        n_features=X_train.shape[2],
        n_timesteps=LOOKBACK,
        horizons=horizons,
    ).to(device)
    logger.info(f"  Fold {fold_num}: {model.n_parameters:,} parameters")

    def flatten(d):
        return torch.cat([d[str(h)] for h in horizons], dim=-1)

    def unflatten(flat):
        r = {}
        for i, h in enumerate(horizons):
            c = flat[:, i * 7:(i + 1) * 7]
            r[str(h)] = {
                "rain_probability": c[:, 0], "precip_mm": c[:, 1],
                "temperature_c": c[:, 2], "wind_speed_kmh": c[:, 3],
                "zonda_risk": c[:, 4], "storm_risk": c[:, 5],
                "hail_risk": c[:, 6],
            }
        return r

    train_ds = TensorDataset(X_train, flatten(y_train))
    val_ds = TensorDataset(X_val, flatten(y_val))
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    criterion = WeatherLossV4(horizons=horizons, horizon_weights=HORIZON_WEIGHTS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # CosineAnnealingWarmRestarts: restarts at 15, 45, 105
    # T_0=15 (first restart after 15 epochs), T_mult=2 (doubles period)
    warmup_epochs = 5
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=lr * 0.01
    )

    accum_steps = 2  # effective batch = 1024
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val = float("inf")
    best_state = None
    patience = 0
    max_patience = 15
    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        # Train
        model.train()
        optimizer.zero_grad()
        t_losses = []
        for step, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)

            # MixUp augmentation (training only)
            if epoch > warmup_epochs:
                xb, yb_mixed, lam = mixup_data(xb, yb, alpha=0.2)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    preds = model(xb)
                    unflat_yb = unflatten(yb_mixed)
                    loss = torch.stack([
                        criterion.forward(preds, unflat_yb)
                    ]).mean() / accum_steps
            else:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    preds = model(xb)
                    unflat_yb = unflatten(yb)
                    loss = torch.stack([
                        criterion.forward(preds, unflat_yb)
                    ]).mean() / accum_steps

            scaler.scale(loss).backward()
            if (step + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            t_losses.append(loss.item() * accum_steps)

        # Validate
        model.eval()
        v_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    preds = model(xb)
                    unflat_yb = unflatten(yb)
                    loss = torch.stack([
                        criterion.forward(preds, unflat_yb)
                    ]).mean()
                v_losses.append(loss.item())

        train_loss = float(np.mean(t_losses))
        val_loss = float(np.mean(v_losses))

        # Warmup + CosineAnnealingWarmRestarts
        if epoch <= warmup_epochs:
            warmup_lr = lr * (epoch / warmup_epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr
        else:
            base_scheduler.step(epoch - warmup_epochs)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if epoch % 10 == 0 or epoch <= 3:
            elapsed = time.time() - t0
            lr_now = optimizer.param_groups[0]["lr"]
            logger.info(
                f"    Epoch {epoch:3d}/{max_epochs}  train={train_loss:.4f}  "
                f"val={val_loss:.4f}  lr={lr_now:.6f}  [{elapsed:.0f}s]"
                f"{'  [AMP]' if use_amp else ''}"
            )

        if patience >= max_patience:
            logger.info(f"    Early stopping at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    logger.info(
        f"  Fold {fold_num} done in {time.time()-t0:.1f}s, best val={best_val:.4f}"
    )
    return model


# ── Per-Zone Evaluation Breakdown ──────────────────────────────
def evaluate_per_zone(
    model: nn.Module,
    X: torch.Tensor,
    y: Dict[str, torch.Tensor],
    zone_ids: np.ndarray,
    horizons: list,
    device: str = "cpu",
) -> Dict[int, dict]:
    """
    Evaluate the model on each zone independently.
    Returns a dict of zone_id → metrics dict.
    This helps identify which zones are strong/weak (e.g., mountain underperformance).
    """
    logger.info("Running per-zone evaluation breakdown...")
    zone_metrics = {}

    for zone_id in sorted(set(zone_ids)):
        mask = zone_ids == zone_id
        if mask.sum() < 100:
            logger.warning(f"  Zone {zone_id}: only {mask.sum()} samples, skipping per-zone eval")
            continue

        X_zone = X[mask]
        y_zone = {h: v[mask] for h, v in y.items()}
        zone_name = ZONE_POINTS[zone_id]["zone_name"]

        preds, tgts = collect_preds(model, X_zone, y_zone, horizons, device)
        metrics = evaluate_v4(preds, tgts, horizons, HORIZON_WEIGHTS)

        # Compute weighted composite for this zone
        weight_sum = sum(HORIZON_WEIGHTS.values())
        weighted_composite = sum(
            HORIZON_WEIGHTS[h] * metrics[f"{h}h"]["composite_accuracy"]
            for h in horizons
        ) / weight_sum

        zone_metrics[zone_id] = {
            "zone_name": zone_name,
            "n_samples": int(mask.sum()),
            "weighted_composite": weighted_composite,
            "metrics": metrics,
        }

        logger.info(
            f"  Zone {zone_id} ({zone_name}): composite={weighted_composite:.4f} "
            f"({mask.sum():,} samples)"
        )

    return zone_metrics


# ── Main ────────────────────────────────────────────────────────
async def main(
    years_back: int = 20,
    n_synthetic: int = 3500,
    max_epochs: int = 150,
    n_folds: int = 3,
    batch_size: int = 512,
):
    logger.info("=" * 60)
    logger.info("CIELO·TUC v6.0-light Training")
    logger.info("  Conservative improvement over v5.0")
    logger.info("  Multi-scale CNN + FeatureDropout + MixUp + CosineAnnealingWarmRestarts")
    logger.info(f"  Lookback: {LOOKBACK}h | Horizons: {HORIZONS}")
    logger.info(f"  Zones: {len(ZONE_POINTS)} | Folds: {n_folds}")
    logger.info(f"  Synthetic events: {n_synthetic}")
    logger.info(f"  Years of data: {years_back}")
    logger.info(f"  Horizon weights: {HORIZON_WEIGHTS}")
    logger.info(f"  Max epochs: {max_epochs} | Patience: 15")
    logger.info("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # ── 1. Download multi-zone data (10 zones) ──
    logger.info(f"\n[1/7] Downloading NASA POWER data ({len(ZONE_POINTS)} zones)...")
    t0 = time.time()
    dfs_by_zone = await download_all_zones(years_back)
    logger.info(f"  Download done in {time.time()-t0:.1f}s")
    logger.info(f"  Got data for {len(dfs_by_zone)}/{len(ZONE_POINTS)} zones")

    # ── 2. Generate synthetic events (distributed across ALL zones) ──
    logger.info(f"\n[2/7] Generating {n_synthetic} synthetic events (v5, all zones)...")
    df_synth = generate_synthetic_extremes_v5(
        n_events=n_synthetic,
        n_zones=len(ZONE_POINTS),
    )

    # CRITICAL FIX: distribute synthetic events across ALL zones
    dfs_by_zone = distribute_synthetic_events(df_synth, dfs_by_zone)

    # ── 3. Build multi-zone tensors ──
    logger.info(f"\n[3/7] Building multi-zone tensors (lookback={LOOKBACK}h)...")
    X, y, feat_cols, zone_ids, samples_per_zone = build_tensors_multi_zone(
        dfs_by_zone, lookback=LOOKBACK, horizons=HORIZONS
    )
    logger.info(f"  X shape: {X.shape}")
    for h in HORIZONS:
        logger.info(f"  y[{h}h] shape: {y[str(h)].shape}")

    # ── 4. Temporal K-Fold CV ──
    logger.info(f"\n[4/7] Temporal {n_folds}-Fold Cross-Validation...")
    folds = temporal_kfold_cv(len(X), n_folds=n_folds)
    all_fold_metrics = []
    all_fold_composites = []
    best_model = None
    best_composite = -1.0

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        logger.info(
            f"\n  ── Fold {fold_idx+1}/{n_folds} "
            f"(train={len(train_idx):,}, val={len(val_idx):,}) ──"
        )

        X_train, X_val = X[train_idx], X[val_idx]
        y_train = {h: v[train_idx] for h, v in y.items()}
        y_val = {h: v[val_idx] for h, v in y.items()}

        # Check if validation has zone data
        val_zones = set(zone_ids[val_idx].tolist())
        logger.info(f"  Validation zones: {sorted(val_zones)}")

        model = train_v5_fold(
            X_train, y_train, X_val, y_val,
            HORIZONS, max_epochs=max_epochs,
            batch_size=batch_size, device=device,
            fold_num=fold_idx + 1,
        )

        # Evaluate this fold
        preds, tgts = collect_preds(model, X_val, y_val, HORIZONS, device)
        metrics = evaluate_v4(preds, tgts, HORIZONS, HORIZON_WEIGHTS)

        # Weighted composite for this fold
        weight_sum = sum(HORIZON_WEIGHTS.values())
        weighted_composite = sum(
            HORIZON_WEIGHTS[h] * metrics[f"{h}h"]["composite_accuracy"]
            for h in HORIZONS
        ) / weight_sum

        all_fold_metrics.append(metrics)
        all_fold_composites.append(weighted_composite)

        logger.info(f"  Fold {fold_idx+1} weighted composite: {weighted_composite:.4f}")

        # Track best model
        if weighted_composite > best_composite:
            best_composite = weighted_composite
            best_model = model

    # ── 5. Aggregate fold results ──
    logger.info(f"\n[5/7] Aggregating {n_folds}-fold results...")
    avg_metrics = {}
    for h in HORIZONS:
        hk = f"{h}h"
        avg_m = {}
        for key in all_fold_metrics[0][hk]:
            if isinstance(all_fold_metrics[0][hk][key], dict):
                avg_m[key] = {
                    k2: float(np.mean([fm[hk][key][k2] for fm in all_fold_metrics]))
                    for k2 in all_fold_metrics[0][hk][key]
                }
            else:
                avg_m[key] = float(np.mean([fm[hk][key] for fm in all_fold_metrics]))
        avg_metrics[hk] = avg_m

    # ── 6. Per-zone evaluation breakdown (NEW in v5) ──
    logger.info(f"\n[6/7] Per-zone evaluation breakdown...")
    zone_breakdown = evaluate_per_zone(
        best_model, X, y, zone_ids, HORIZONS, device
    )

    # ── 7. Print results ──
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS (averaged across folds)")
    logger.info("=" * 60)

    for h in HORIZONS:
        m = avg_metrics[f"{h}h"]
        logger.info(f"\n  {h}h:")
        logger.info(
            f"    Rain acc: {m['rain_probability']['accuracy']:.4f} "
            f"| F1: {m['rain_probability']['f1']:.4f}"
        )
        logger.info(
            f"    Temp RMSE: {m['temperature_c']['rmse']:.2f}°C "
            f"| MAE: {m['temperature_c']['mae']:.2f}°C"
            f"| Seasonal: {m['temp_seasonal_score']:.4f}"
            f"| DirAcc: {m['temp_directional_accuracy']:.4f}"
        )
        logger.info(
            f"    Zonda: {m['zonda_risk']['accuracy']:.4f} "
            f"(P:{m['zonda_risk']['precision']:.4f} "
            f"R:{m['zonda_risk']['recall']:.4f}) "
            f"| Storm: {m['storm_risk']['accuracy']:.4f} "
            f"(P:{m['storm_risk']['precision']:.4f} "
            f"R:{m['storm_risk']['recall']:.4f}) "
            f"| Hail: {m['hail_risk']['accuracy']:.4f} "
            f"(P:{m['hail_risk']['precision']:.4f} "
            f"R:{m['hail_risk']['recall']:.4f})"
        )
        logger.info(f"    Composite: {m['composite_accuracy']:.4f}")

    weight_sum = sum(HORIZON_WEIGHTS.values())
    weighted_composite = sum(
        HORIZON_WEIGHTS[h] * avg_metrics[f"{h}h"]["composite_accuracy"]
        for h in HORIZONS
    ) / weight_sum
    avg_rain = float(np.mean(
        [avg_metrics[f"{h}h"]["rain_probability"]["accuracy"] for h in HORIZONS]
    ))
    avg_temp = float(np.mean(
        [avg_metrics[f"{h}h"]["temperature_c"]["rmse"] for h in HORIZONS]
    ))
    avg_directional = float(np.mean(
        [avg_metrics[f"{h}h"]["temp_directional_accuracy"] for h in HORIZONS]
    ))

    logger.info(f"\n  {'='*40}")
    logger.info(f"  WEIGHTED COMPOSITE:     {weighted_composite:.4f} ({weighted_composite*100:.1f}%)")
    logger.info(f"  AVG RAIN ACC:           {avg_rain:.4f}")
    logger.info(f"  AVG TEMP RMSE:          {avg_temp:.2f}°C")
    logger.info(f"  AVG TEMP DIR ACC:       {avg_directional:.4f}")
    logger.info(f"  FOLD SCORES:            {[f'{c:.4f}' for c in all_fold_composites]}")

    # ── Per-zone breakdown table (NEW in v5) ──
    logger.info(f"\n  {'='*60}")
    logger.info("  PER-ZONE BREAKDOWN")
    logger.info(f"  {'='*60}")
    header = f"  {'Zone':<20} {'Samples':>8} {'Composite':>10} {'Status':<10}"
    logger.info(header)
    logger.info(f"  {'-'*48}")

    # Determine thresholds from zone composites
    zone_composites = {
        zid: zm["weighted_composite"]
        for zid, zm in zone_breakdown.items()
    }
    if zone_composites:
        avg_zone_composite = np.mean(list(zone_composites.values()))
    else:
        avg_zone_composite = 0.5

    for zone_id in sorted(zone_breakdown.keys()):
        zm = zone_breakdown[zone_id]
        composite = zm["weighted_composite"]
        n_samp = zm["n_samples"]
        if composite >= avg_zone_composite * 1.05:
            status = "STRONG"
        elif composite <= avg_zone_composite * 0.95:
            status = "WEAK"
        else:
            status = "OK"
        logger.info(
            f"  {zm['zone_name']:<20} {n_samp:>8,} {composite:>10.4f} {status:<10}"
        )

    # Flag weak zones
    weak_zones = [
        zone_breakdown[zid]["zone_name"]
        for zid, zm in zone_breakdown.items()
        if zm["weighted_composite"] < avg_zone_composite * 0.95
    ]
    if weak_zones:
        logger.info(f"\n  ⚠ Weak zones: {', '.join(weak_zones)}")
    else:
        logger.info(f"\n  ✓ All zones perform within expected range")

    # ── Save model ──
    version = f"6.0-{datetime.now(timezone.utc).strftime('%Y%m%d')}"
    ts = datetime.now(timezone.utc).isoformat()

    ckpt = MODELS_DIR / f"cielotuc_v{version}.pt"
    torch.save({
        "model_state_dict": best_model.state_dict(),
        "version": version,
        "model_class": "FastWeatherModelV5Light",
        "horizons": HORIZONS,
        "n_features": X.shape[2],
        "n_timesteps": LOOKBACK,
        "trained_at": ts,
        "training_samples": len(X),
        "device": device,
        "feat_cols": feat_cols,
        "zone_count": len(ZONE_POINTS),
        "horizon_weights": HORIZON_WEIGHTS,
    }, ckpt)
    logger.info(f"\nModel saved: {ckpt}")

    report = {
        "version": version,
        "trained_at": ts,
        "training_samples": len(X),
        "years_of_data": years_back,
        "synthetic_events": n_synthetic,
        "device": device,
        "lookback_hours": LOOKBACK,
        "horizons": HORIZONS,
        "horizon_weights": HORIZON_WEIGHTS,
        "zone_count": len(ZONE_POINTS),
        "cv_folds": n_folds,
        "fold_scores": all_fold_composites,
        "weighted_composite_accuracy": weighted_composite,
        "avg_rain_accuracy": avg_rain,
        "avg_temp_rmse_c": avg_temp,
        "avg_temp_directional_accuracy": avg_directional,
        "final_metrics": avg_metrics,
        "n_parameters": best_model.n_parameters,
        "feat_cols": feat_cols,
        "samples_per_zone": samples_per_zone,
        "per_zone_breakdown": {
            str(zid): {
                "zone_name": zm["zone_name"],
                "n_samples": zm["n_samples"],
                "weighted_composite": zm["weighted_composite"],
            }
            for zid, zm in zone_breakdown.items()
        },
    }
    mp = MODELS_DIR / f"metrics_v{version}.json"
    with open(mp, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Metrics saved: {mp}")

    logger.info("\n" + "=" * 60)
    logger.info(f"CIELO·TUC v{version} — Training complete!")
    logger.info(f"  Weighted composite: {weighted_composite:.4f} ({weighted_composite*100:.1f}%)")
    logger.info(f"  Parameters:         {best_model.n_parameters:,}")
    logger.info(f"  Features:           {len(feat_cols)}")
    logger.info(f"  Zones:              {len(ZONE_POINTS)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CIELO·TUC v6.0-light")
    parser.add_argument("--years", type=int, default=20)
    parser.add_argument("--n-synthetic", type=int, default=3500)
    parser.add_argument("--max-epochs", type=int, default=150)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    args, _ = parser.parse_known_args()
    asyncio.run(main(
        years_back=args.years,
        n_synthetic=args.n_synthetic,
        max_epochs=args.max_epochs,
        n_folds=args.folds,
        batch_size=args.batch_size,
    ))
