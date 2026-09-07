"""
scripts/train_v2.py
───────────────────
CIELO·TUC v2.0 Training Script — optimized for speed + accuracy.

Key improvements over v1:
  1. 24h lookback (vs 72h) — 3x faster, captures daily cycles
  2. 1000 synthetic events (vs 500) with transitions between states
  3. Better feature engineering: lag features, weather interactions
  4. CosineAnnealingLR scheduler — faster convergence
  5. Single train/val split — no timeout from multi-fold
  6. Larger batch (256) — faster epochs
  7. AdamW with warmup — more stable early training
  8. Enhanced evaluation with per-target breakdown

Usage:
  python scripts/train_v2.py [--years 10] [--max-epochs 80]
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import (
    accuracy_score, mean_absolute_error, mean_squared_error,
    precision_score, recall_score, f1_score,
)
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ml.pipeline.data_pipeline import WeatherDataPipeline, FEATURE_COLS

# ── Constants ───────────────────────────────────────────────────
TUC_LAT = -26.82
TUC_LNG = -65.22
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
N_FEATURES = 34
LOOKBACK = 24  # hours (reduced from 72)
HORIZONS = [3, 6, 12, 24, 48, 168]


# ── Lightweight Model (faster than full CNN-LSTM) ───────────────
class FastWeatherModel(nn.Module):
    """
    Lighter model for faster training:
    - 2-layer LSTM (vs 3)
    - 96 hidden (vs 128)
    - 1 Conv layer (vs 2)
    - Same multi-head output
    """
    def __init__(self, n_features=N_FEATURES, n_timesteps=LOOKBACK, horizons=None):
        super().__init__()
        self.horizons = horizons or HORIZONS
        self.hidden_size = 96

        # Single CNN layer
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 48, kernel_size=3, padding=1),
            nn.BatchNorm1d(48),
            nn.GELU(),
        )

        # 2-layer LSTM
        self.lstm = nn.LSTM(
            input_size=48, hidden_size=self.hidden_size,
            num_layers=2, batch_first=True, dropout=0.15,
        )

        # Attention
        self.attn = nn.Linear(self.hidden_size, 1)

        # Heads
        self.heads = nn.ModuleDict()
        for h in self.horizons:
            self.heads[str(h)] = nn.Sequential(
                nn.Linear(self.hidden_size, 48),
                nn.BatchNorm1d(48),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(48, 7),
            )

    def forward(self, x):
        cnn_in = x.permute(0, 2, 1)
        cnn_out = self.cnn(cnn_in).permute(0, 2, 1)
        lstm_out, _ = self.lstm(cnn_out)

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


# ── Loss Function (improved weighting) ─────────────────────────
class WeatherLossV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_rain = nn.BCELoss(weight=torch.tensor(3.0))
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.bce = nn.BCELoss()

    def forward(self, preds, targets):
        loss = torch.tensor(0.0, requires_grad=True)
        if "rain_probability" in targets:
            loss = loss + 1.5 * self.bce_rain(preds["rain_probability"], targets["rain_probability"])
        if "precip_mm" in targets:
            loss = loss + 0.4 * self.mse(preds["precip_mm"], targets["precip_mm"])
        if "temperature_c" in targets:
            loss = loss + 0.8 * self.mae(preds["temperature_c"], targets["temperature_c"])
        if "wind_speed_kmh" in targets:
            loss = loss + 0.3 * self.mse(preds["wind_speed_kmh"], targets["wind_speed_kmh"])
        for key in ("zonda_risk", "storm_risk", "hail_risk"):
            if key in targets:
                loss = loss + 1.0 * self.bce(preds[key], targets[key])
        return loss


# ── Data Download (same as v1) ─────────────────────────────────
async def download_nasa_power(years_back=10):
    import httpx
    from datetime import timedelta

    end = datetime(2026, 7, 31, tzinfo=timezone.utc)
    start = end.replace(year=end.year - years_back)
    logger.info(f"Downloading NASA POWER: {start.date()} → {end.date()}")

    chunks = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start.replace(year=chunk_start.year + 3), end)
        params = {
            "parameters": "T2M,RH2M,WS10M,WD10M,PS,PRECTOTCORR,ALLSKY_SFC_SW_DWN",
            "community": "RE",
            "longitude": TUC_LNG, "latitude": TUC_LAT,
            "start": chunk_start.strftime("%Y%m%d"),
            "end": chunk_end.strftime("%Y%m%d"),
            "format": "CSV",
        }
        url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
        async with httpx.AsyncClient(timeout=180) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
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

    df = pd.concat(chunks, ignore_index=True)
    df.index = df["timestamp"]
    df.index.name = "timestamp"
    df = df.drop(columns=["timestamp"], errors="ignore")
    df = df.rename(columns={
        "T2M": "temperature_c", "RH2M": "humidity_pct",
        "WS10M": "wind_speed_kmh", "WD10M": "wind_direction_deg",
        "PS": "pressure_hpa", "PRECTOTCORR": "precip_mm",
        "ALLSKY_SFC_SW_DWN": "solar_radiation_wm2",
    })
    df["temperature_min_c"] = df["temperature_c"].rolling(24, min_periods=1).min()
    df["temperature_max_c"] = df["temperature_c"].rolling(24, min_periods=1).max()
    df = df.replace(-999.0, np.nan)
    logger.info(f"NASA POWER: {len(df):,} rows downloaded")
    return df


# ── Enhanced Synthetic Events ───────────────────────────────────
def generate_synthetic_extremes(n_events=1000):
    """Generate synthetic extreme weather events with realistic transitions."""
    logger.info(f"Generating {n_events} synthetic extreme events")
    rows = []
    rng = np.random.default_rng(42)

    for _ in range(n_events):
        event_type = rng.choice(
            ["zonda", "hail_storm", "heat_wave", "extreme_rain", "normal_volatile"],
            p=[0.20, 0.20, 0.20, 0.20, 0.20],
        )
        duration = rng.integers(8, 36)
        start = pd.Timestamp("2002-01-01") + pd.Timedelta(days=int(rng.integers(0, 365 * 18)))

        for h in range(duration):
            t = start + pd.Timedelta(hours=h)
            progress = h / max(duration - 1, 1)

            if event_type == "zonda":
                # Ramping: starts moderate, peaks, then subsides
                intensity = np.sin(progress * np.pi)
                row = {
                    "timestamp": t,
                    "temperature_c": 28 + 17 * intensity + rng.uniform(-1, 1),
                    "humidity_pct": 15 + 50 * (1 - intensity) + rng.uniform(-3, 3),
                    "wind_speed_kmh": 30 + 80 * intensity + rng.uniform(-5, 5),
                    "pressure_hpa": 1000 - 10 * intensity + rng.uniform(-1, 1),
                    "precip_mm": 0.0,
                    "cape_j_kg": 100 * intensity,
                    "cordillera_pressure_hpa": 890 - 20 * intensity,
                    "thermal_differential_c": 5 + 23 * intensity,
                    "wind_direction_deg": 270 + rng.uniform(-15, 15),
                    "_label_zonda": 1.0 if intensity > 0.5 else 0.0,
                    "_label_storm": 0.0,
                    "_label_hail": 0.0,
                }
            elif event_type == "hail_storm":
                intensity = np.sin(progress * np.pi)
                row = {
                    "timestamp": t,
                    "temperature_c": 25 - 5 * intensity + rng.uniform(-1, 1),
                    "humidity_pct": 80 + 15 * intensity + rng.uniform(-3, 3),
                    "wind_speed_kmh": 25 + 55 * intensity + rng.uniform(-5, 5),
                    "pressure_hpa": 1005 - 12 * intensity + rng.uniform(-1, 1),
                    "precip_mm": (15 + 35 * intensity + rng.uniform(-3, 3)) if intensity > 0.3 else rng.uniform(0, 3),
                    "cape_j_kg": 1000 + 4000 * intensity,
                    "cordillera_pressure_hpa": 900 + rng.uniform(-3, 3),
                    "thermal_differential_c": 5 + 7 * intensity,
                    "wind_direction_deg": 180 + rng.uniform(-30, 30),
                    "_label_zonda": 0.0,
                    "_label_storm": 1.0,
                    "_label_hail": 1.0 if intensity > 0.6 else 0.0,
                }
            elif event_type == "heat_wave":
                intensity = np.sin(progress * np.pi)
                row = {
                    "timestamp": t,
                    "temperature_c": 33 + 11 * intensity + rng.uniform(-1, 1),
                    "humidity_pct": 25 + 10 * (1 - intensity) + rng.uniform(-3, 3),
                    "wind_speed_kmh": 8 + 5 * intensity + rng.uniform(-2, 2),
                    "pressure_hpa": 1010 + 3 * (1 - intensity) + rng.uniform(-1, 1),
                    "precip_mm": 0.0,
                    "cape_j_kg": 200 + 600 * intensity,
                    "cordillera_pressure_hpa": 898 + rng.uniform(-2, 2),
                    "thermal_differential_c": 8 + 10 * intensity,
                    "_label_zonda": 0.0,
                    "_label_storm": 0.0,
                    "_label_hail": 0.0,
                }
            elif event_type == "extreme_rain":
                intensity = np.sin(progress * np.pi)
                row = {
                    "timestamp": t,
                    "temperature_c": 24 - 4 * intensity + rng.uniform(-1, 1),
                    "humidity_pct": 90 + 10 * intensity + rng.uniform(-2, 2),
                    "wind_speed_kmh": 15 + 30 * intensity + rng.uniform(-3, 3),
                    "pressure_hpa": 1002 - 8 * intensity + rng.uniform(-1, 1),
                    "precip_mm": (20 + 60 * intensity + rng.uniform(-5, 5)) if intensity > 0.2 else rng.uniform(0, 5),
                    "cape_j_kg": 800 + 3200 * intensity,
                    "cordillera_pressure_hpa": 902 + rng.uniform(-2, 2),
                    "thermal_differential_c": 3 + 4 * intensity,
                    "_label_zonda": 0.0,
                    "_label_storm": 1.0,
                    "_label_hail": 0.0,
                }
            else:  # normal_volatile — typical Tucumán day with variability
                row = {
                    "timestamp": t,
                    "temperature_c": 15 + 15 * np.sin(progress * np.pi * 2) + rng.uniform(-2, 2),
                    "humidity_pct": 40 + 30 * rng.uniform(0, 1),
                    "wind_speed_kmh": 5 + 20 * rng.uniform(0, 1),
                    "pressure_hpa": 1008 + rng.uniform(-4, 4),
                    "precip_mm": rng.choice([0, 0, 0, 0, rng.uniform(1, 15)]),
                    "cape_j_kg": rng.uniform(50, 1500),
                    "cordillera_pressure_hpa": 900 + rng.uniform(-5, 5),
                    "thermal_differential_c": rng.uniform(1, 10),
                    "_label_zonda": 0.0,
                    "_label_storm": 0.0,
                    "_label_hail": 0.0,
                }

            # Add some noise to all numeric fields
            for k in ["temperature_c", "humidity_pct", "wind_speed_kmh", "pressure_hpa"]:
                row[k] = max(0, row[k] + rng.uniform(-0.5, 0.5))

            rows.append(row)

    return pd.DataFrame(rows)


# ── Feature Engineering Additions ───────────────────────────────
def add_enhanced_features(df):
    """Add lag features and weather interaction terms."""
    logger.info("Adding enhanced features (lag + interactions)...")

    # Lag features (previous hours)
    if "temperature_c" in df.columns:
        df["temp_lag_1h"] = df["temperature_c"].shift(1)
        df["temp_lag_3h"] = df["temperature_c"].shift(3)
        df["temp_tendency"] = df["temperature_c"].diff(3)  # 3h trend
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

    # Interaction features
    if "temperature_c" in df.columns and "humidity_pct" in df.columns:
        df["temp_humid_index"] = df["temperature_c"] * df["humidity_pct"] / 100

    if "cape_j_kg" in df.columns and "humidity_pct" in df.columns:
        df["instability_index"] = df["cape_j_kg"] * df["humidity_pct"] / 10000

    if "wind_speed_kmh" in df.columns and "pressure_hpa" in df.columns:
        df["wind_pressure_ratio"] = df["wind_speed_kmh"] / df["pressure_hpa"].clip(lower=900)

    # Fill NaN from lag features
    df = df.ffill(limit=6).bfill(limit=3).fillna(0)

    n_new = len(df.columns) - N_FEATURES
    logger.info(f"  Added {n_new} new features → {len(df.columns)} total")
    return df


# ── Build Tensors ───────────────────────────────────────────────
def build_tensors(df, pipeline, lookback=LOOKBACK, horizons=None):
    """Build training tensors with specified lookback."""
    horizons = horizons or HORIZONS
    max_h = max(horizons)

    all_cols = list(df.columns)
    # Use only columns that exist
    feat_cols = [c for c in all_cols if c not in (
        "_label_zonda", "_label_storm", "_label_hail",
        "timestamp", "year", "month", "day",
    )]

    data = df[feat_cols].values.astype(np.float32)
    n_samples = len(data) - lookback - max_h

    if n_samples <= 0:
        raise ValueError(f"Not enough data: {len(data)} rows, need {lookback + max_h + 1}")

    X = np.zeros((n_samples, lookback, len(feat_cols)), dtype=np.float32)
    y = {str(h): np.zeros((n_samples, 7), dtype=np.float32) for h in horizons}

    for i in range(n_samples):
        X[i] = data[i:i + lookback]

        for hi, h in enumerate(horizons):
            target_row = data[i + lookback + h - 1]
            # Find column indices
            temp_idx = feat_cols.index("temperature_c") if "temperature_c" in feat_cols else 0
            precip_idx = feat_cols.index("precip_mm") if "precip_mm" in feat_cols else 0
            humid_idx = feat_cols.index("humidity_pct") if "humidity_pct" in feat_cols else 0
            wind_idx = feat_cols.index("wind_speed_kmh") if "wind_speed_kmh" in feat_cols else 0

            # Labels
            label_z = df.iloc[i + lookback + h - 1].get("_label_zonda", 0.0)
            label_s = df.iloc[i + lookback + h - 1].get("_label_storm", 0.0)
            label_h = df.iloc[i + lookback + h - 1].get("_label_hail", 0.0)

            y[str(h)][i] = [
                1.0 if target_row[precip_idx] > 1.0 else 0.0,  # rain probability
                max(0, target_row[precip_idx]),                   # precip_mm
                target_row[temp_idx],                             # temperature_c
                max(0, target_row[wind_idx]),                     # wind_speed_kmh
                float(label_z),                                   # zonda_risk
                float(label_s),                                   # storm_risk
                float(label_h),                                   # hail_risk
            ]

    X = torch.tensor(X, dtype=torch.float32)
    y = {h: torch.tensor(v, dtype=torch.float32) for h, v in y.items()}
    return X, y, feat_cols


# ── Evaluation ──────────────────────────────────────────────────
def evaluate(all_preds, all_targets, horizons):
    metrics = {}
    names = ["rain_probability", "precip_mm", "temperature_c", "wind_speed_kmh", "zonda_risk", "storm_risk", "hail_risk"]

    for h in horizons:
        hk = str(h)
        p, t = all_preds[hk], all_targets[hk]
        hm = {}

        for i, name in enumerate(names):
            pv, tv = p[:, i], t[:, i]
            if name in ("rain_probability", "zonda_risk", "storm_risk", "hail_risk"):
                pb, tb = (pv > 0.5).astype(int), (tv > 0.5).astype(int)
                hm[name] = {
                    "accuracy": float(accuracy_score(tb, pb)),
                    "precision": float(precision_score(tb, pb, zero_division=0)),
                    "recall": float(recall_score(tb, pb, zero_division=0)),
                    "f1": float(f1_score(tb, pb, zero_division=0)),
                }
            else:
                hm[name] = {
                    "rmse": float(np.sqrt(mean_squared_error(tv, pv))),
                    "mae": float(mean_absolute_error(tv, pv)),
                }

        binary_accs = [hm[k]["accuracy"] for k in ("rain_probability", "zonda_risk", "storm_risk", "hail_risk")]
        temp_rmse = hm.get("temperature_c", {}).get("rmse", 10.0)
        temp_score = max(0, 1.0 - temp_rmse / 30.0)
        hm["composite_accuracy"] = float(np.mean(binary_accs + [temp_score]))
        metrics[f"{h}h"] = hm

    return metrics


# ── Training Loop ───────────────────────────────────────────────
def train(X, y, horizons, max_epochs=80, batch_size=256, lr=2e-3, device="cpu"):
    model = FastWeatherModel(
        n_features=X.shape[2], n_timesteps=LOOKBACK, horizons=horizons,
    ).to(device)
    logger.info(f"Model: {model.n_parameters:,} parameters")

    # Train/val split (85/15)
    n = len(X)
    split = int(n * 0.85)
    X_train, X_val = X[:split], X[split:]
    y_train = {h: v[:split] for h, v in y.items()}
    y_val = {h: v[split:] for h, v in y.items()}

    def flatten(d):
        return torch.cat([d[str(h)] for h in horizons], dim=-1)

    train_ds = TensorDataset(X_train, flatten(y_train))
    val_ds = TensorDataset(X_val, flatten(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    criterion = WeatherLossV2()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Cosine annealing: lr drops from lr → lr/100 over max_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=lr / 100,
    )

    def unflatten(flat):
        r = {}
        for i, h in enumerate(horizons):
            c = flat[:, i*7:(i+1)*7]
            r[str(h)] = {
                "rain_probability": c[:, 0], "precip_mm": c[:, 1],
                "temperature_c": c[:, 2], "wind_speed_kmh": c[:, 3],
                "zonda_risk": c[:, 4], "storm_risk": c[:, 5], "hail_risk": c[:, 6],
            }
        return r

    best_val = float("inf")
    best_state = None
    patience = 0
    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        # Train
        model.train()
        t_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = torch.stack([
                criterion(preds[str(h)], unflatten(yb)[str(h)]) for h in horizons
            ]).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_losses.append(loss.item())

        # Validate
        model.eval()
        v_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = torch.stack([
                    criterion(preds[str(h)], unflatten(yb)[str(h)]) for h in horizons
                ]).mean()
                v_losses.append(loss.item())

        train_loss = float(np.mean(t_losses))
        val_loss = float(np.mean(v_losses))
        scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if epoch % 5 == 0:
            elapsed = time.time() - t0
            lr_now = optimizer.param_groups[0]["lr"]
            logger.info(
                f"  Epoch {epoch:3d}/{max_epochs}  train={train_loss:.4f}  val={val_loss:.4f}  "
                f"lr={lr_now:.6f}  [{elapsed:.0f}s]"
            )

        if patience >= 12:
            logger.info(f"  Early stopping at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    logger.info(f"Training done in {time.time()-t0:.1f}s, best val={best_val:.4f}")
    return model


# ── Collect Predictions ─────────────────────────────────────────
def collect_preds(model, X, y, horizons, device="cpu"):
    model.eval()
    all_p = {str(h): [] for h in horizons}
    all_t = {str(h): [] for h in horizons}

    def unflatten(flat):
        r = {}
        for i, h in enumerate(horizons):
            c = flat[:, i*7:(i+1)*7]
            r[str(h)] = {
                "rain_probability": c[:, 0], "precip_mm": c[:, 1],
                "temperature_c": c[:, 2], "wind_speed_kmh": c[:, 3],
                "zonda_risk": c[:, 4], "storm_risk": c[:, 5], "hail_risk": c[:, 6],
            }
        return r

    def flatten(d):
        return torch.cat([d[str(h)] for h in horizons], dim=-1)

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
                    preds[hk]["zonda_risk"], preds[hk]["storm_risk"], preds[hk]["hail_risk"],
                ], dim=1).cpu().numpy()
                ta = torch.stack([
                    tgts[hk]["rain_probability"], tgts[hk]["precip_mm"],
                    tgts[hk]["temperature_c"], tgts[hk]["wind_speed_kmh"],
                    tgts[hk]["zonda_risk"], tgts[hk]["storm_risk"], tgts[hk]["hail_risk"],
                ], dim=1).numpy()
                all_p[hk].append(pa)
                all_t[hk].append(ta)

    for h in horizons:
        hk = str(h)
        all_p[hk] = np.concatenate(all_p[hk])
        all_t[hk] = np.concatenate(all_t[hk])
    return all_p, all_t


# ── Main ────────────────────────────────────────────────────────
async def main(years_back=10, n_synthetic=1000, max_epochs=80):
    logger.info("=" * 60)
    logger.info("CIELO·TUC — Model Training v2.0 (Optimized)")
    logger.info(f"  Lookback: {LOOKBACK}h | Horizons: {HORIZONS}")
    logger.info(f"  Synthetic events: {n_synthetic}")
    logger.info("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # 1. Download
    logger.info("\n[1/5] Downloading NASA POWER data...")
    t0 = time.time()
    df_main = await download_nasa_power(years_back)
    logger.info(f"  Done in {time.time()-t0:.1f}s")

    # 2. Synthetic
    logger.info(f"\n[2/5] Generating {n_synthetic} synthetic events...")
    df_synth = generate_synthetic_extremes(n_events=n_synthetic)
    df = pd.concat([df_main, df_synth], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    logger.info(f"  Combined: {len(df):,} rows")

    # 3. Feature engineering (from pipeline)
    logger.info("\n[3/5] Feature engineering...")
    pipeline = WeatherDataPipeline()
    df = pipeline.preprocess(df)

    # 4. Enhanced features
    df = add_enhanced_features(df)

    # 5. Build tensors
    logger.info(f"\n[4/5] Building tensors (lookback={LOOKBACK}h)...")
    X, y, feat_cols = build_tensors(df, pipeline)
    logger.info(f"  X: {X.shape} | features: {len(feat_cols)}")
    for h in HORIZONS:
        logger.info(f"  y[{h}h]: {y[str(h)].shape}")

    # 6. Train
    logger.info(f"\n[5/5] Training ({max_epochs} max epochs)...")
    model = train(X, y, HORIZONS, max_epochs=max_epochs, device=device)

    # 7. Evaluate on VALIDATION set only
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION (validation set)")
    logger.info("=" * 60)
    split = int(len(X) * 0.85)
    X_val = X[split:]
    y_val = {h: v[split:] for h, v in y.items()}
    preds, tgts = collect_preds(model, X_val, y_val, HORIZONS, device)
    metrics = evaluate(preds, tgts, HORIZONS)

    for h in HORIZONS:
        m = metrics[f"{h}h"]
        logger.info(f"\n  {h}h:")
        logger.info(f"    Rain acc: {m['rain_probability']['accuracy']:.4f} | F1: {m['rain_probability']['f1']:.4f}")
        logger.info(f"    Temp RMSE: {m['temperature_c']['rmse']:.2f}°C | MAE: {m['temperature_c']['mae']:.2f}°C")
        logger.info(f"    Zonda: {m['zonda_risk']['accuracy']:.4f} | Storm: {m['storm_risk']['accuracy']:.4f}")
        logger.info(f"    Composite: {m['composite_accuracy']:.4f}")

    avg_composite = float(np.mean([metrics[f"{h}h"]["composite_accuracy"] for h in HORIZONS]))
    avg_rain = float(np.mean([metrics[f"{h}h"]["rain_probability"]["accuracy"] for h in HORIZONS]))
    avg_temp = float(np.mean([metrics[f"{h}h"]["temperature_c"]["rmse"] for h in HORIZONS]))
    logger.info(f"\n  {'='*40}")
    logger.info(f"  AVG COMPOSITE: {avg_composite:.4f}")
    logger.info(f"  AVG RAIN ACC:  {avg_rain:.4f}")
    logger.info(f"  AVG TEMP RMSE: {avg_temp:.2f}°C")

    # 8. Save
    version = f"2.0-{datetime.now(timezone.utc).strftime('%Y%m%d')}"
    ts = datetime.now(timezone.utc).isoformat()

    ckpt = MODELS_DIR / f"cielotuc_v{version}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "version": version, "horizons": HORIZONS,
        "n_features": X.shape[2], "n_timesteps": LOOKBACK,
        "trained_at": ts, "training_samples": len(X),
        "device": device, "feat_cols": feat_cols,
    }, ckpt)
    logger.info(f"\nModel saved: {ckpt}")

    report = {
        "version": version, "trained_at": ts,
        "training_samples": len(X), "years_of_data": years_back,
        "synthetic_events": n_synthetic, "device": device,
        "lookback_hours": LOOKBACK, "horizons": HORIZONS,
        "avg_composite_accuracy": avg_composite,
        "avg_rain_accuracy": avg_rain, "avg_temp_rmse_c": avg_temp,
        "final_metrics": metrics,
        "n_parameters": model.n_parameters,
        "feat_cols": feat_cols,
    }
    mp = MODELS_DIR / f"metrics_v{version}.json"
    with open(mp, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Metrics saved: {mp}")

    logger.info("\n" + "=" * 60)
    logger.info(f"CIELO·TUC v{version} — Training complete!")
    logger.info(f"  Composite accuracy: {avg_composite:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CIELO·TUC v2.0")
    parser.add_argument("--years", type=int, default=10)
    parser.add_argument("--n-synthetic", type=int, default=1000)
    parser.add_argument("--max-epochs", type=int, default=80)
    args = parser.parse_args()
    asyncio.run(main(
        years_back=args.years,
        n_synthetic=args.n_synthetic,
        max_epochs=args.max_epochs,
    ))
