"""
tests/unit/test_ml.py
──────────────────────
Unit tests for the CNN-LSTM model and data pipeline.
Run with: pytest tests/unit/test_ml.py -v
"""

import numpy as np
import pandas as pd
import pytest
import torch

from app.ml.models.cnn_lstm import CnnLstmWeatherModel, WeatherLoss
from app.ml.pipeline.data_pipeline import (
    WeatherDataPipeline, FEATURE_COLS, TARGET_COLS
)


# ── Fixtures ──────────────────────────────────────────────────
@pytest.fixture
def small_model():
    """Small model for fast tests."""
    return CnnLstmWeatherModel(
        n_features=34,
        n_timesteps=24,  # smaller window for speed
        horizons=[6, 24],
        cnn_channels=16,
        lstm_hidden=32,
        lstm_layers=2,
    )


@pytest.fixture
def pipeline():
    return WeatherDataPipeline(lookback_hours=24, horizons=[6, 24])


@pytest.fixture
def sample_df():
    """72 rows of realistic Tucumán weather data."""
    n = 72
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="1h"),
        "temperature_c": rng.uniform(15, 35, n),
        "humidity_pct": rng.uniform(40, 95, n),
        "pressure_hpa": rng.uniform(1000, 1020, n),
        "precip_mm": np.where(rng.random(n) > 0.8, rng.uniform(0, 30, n), 0),
        "wind_speed_kmh": rng.uniform(5, 60, n),
        "wind_direction_deg": rng.uniform(0, 360, n),
        "wind_gust_kmh": rng.uniform(10, 80, n),
        "cape_j_kg": rng.uniform(0, 3000, n),
        "cordillera_pressure_hpa": rng.uniform(880, 900, n),
        "thermal_differential_c": rng.uniform(2, 20, n),
        "enso_index": np.zeros(n),
    })


# ── Feature columns test ───────────────────────────────────────
def test_feature_cols_count():
    assert len(FEATURE_COLS) == 34, f"Expected 34, got {len(FEATURE_COLS)}"


def test_no_duplicate_features():
    assert len(FEATURE_COLS) == len(set(FEATURE_COLS))


# ── Pipeline tests ─────────────────────────────────────────────
def test_preprocess_output_shape(pipeline, sample_df):
    df = pipeline.preprocess(sample_df)
    assert "timestamp" in df.columns
    for col in FEATURE_COLS:
        assert col in df.columns, f"Missing feature: {col}"
    assert len(df) == len(sample_df)


def test_preprocess_no_nulls(pipeline, sample_df):
    df = pipeline.preprocess(sample_df)
    nulls = df[FEATURE_COLS].isnull().sum().sum()
    assert nulls == 0, f"Found {nulls} null values after preprocessing"


def test_cyclic_features_bounded(pipeline, sample_df):
    df = pipeline.preprocess(sample_df)
    for col in ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]:
        assert df[col].between(-1.0, 1.0).all(), f"{col} out of [-1, 1]"


def test_rolling_precip_monotone(pipeline, sample_df):
    """3h precip should always be ≤ 24h precip."""
    df = pipeline.preprocess(sample_df)
    assert (df["precip_3h_mm"] <= df["precip_24h_mm"] + 1e-6).all()


def test_scaler_fit_transform(pipeline, sample_df):
    df = pipeline.preprocess(sample_df)
    pipeline.fit_scaler(df)
    scaled = pipeline.transform(df)
    assert scaled.shape == (len(df), 34)
    # After MinMax scaling values should be mostly in [-1, 1]
    assert scaled.min() >= -1.1
    assert scaled.max() <= 1.1


def test_build_inference_window(pipeline, sample_df):
    df = pipeline.preprocess(sample_df)
    pipeline.fit_scaler(df)
    window = pipeline.build_inference_window(df)
    assert window.shape == (1, pipeline.lookback, 34)
    assert window.dtype == torch.float32


def test_build_training_tensors_shapes(pipeline, sample_df):
    df = pipeline.preprocess(sample_df)
    pipeline.fit_scaler(df)
    X, y = pipeline.build_training_tensors(df)
    assert X.dim() == 3
    assert X.shape[1] == pipeline.lookback
    assert X.shape[2] == 34
    for h in pipeline.horizons:
        assert str(h) in y
        assert y[str(h)].shape[1] == 7


# ── Model architecture tests ───────────────────────────────────
def test_model_forward_shapes(small_model):
    batch_size = 4
    x = torch.randn(batch_size, 24, 34)
    with torch.no_grad():
        preds = small_model(x)
    for h in [6, 24]:
        key = str(h)
        assert key in preds
        for output_key in [
            "rain_probability", "precip_mm", "temperature_c",
            "wind_speed_kmh", "zonda_risk", "storm_risk", "hail_risk"
        ]:
            assert output_key in preds[key]
            assert preds[key][output_key].shape == (batch_size,)


def test_probabilities_in_range(small_model):
    x = torch.randn(8, 24, 34)
    with torch.no_grad():
        preds = small_model(x)
    for h in [6, 24]:
        for prob_key in ["rain_probability", "zonda_risk", "storm_risk", "hail_risk"]:
            p = preds[str(h)][prob_key]
            assert (p >= 0).all() and (p <= 1).all(), \
                f"{prob_key} h={h} out of [0,1]"


def test_nonnegative_outputs(small_model):
    x = torch.randn(8, 24, 34)
    with torch.no_grad():
        preds = small_model(x)
    for h in [6, 24]:
        for nonneg_key in ["precip_mm", "wind_speed_kmh"]:
            assert (preds[str(h)][nonneg_key] >= 0).all(), \
                f"{nonneg_key} h={h} contains negative values"


def test_model_parameter_count(small_model):
    n = small_model.n_parameters
    assert n > 0
    print(f"\nSmall model parameters: {n:,}")


def test_different_inputs_different_outputs(small_model):
    """Model should not return the same output for different inputs."""
    x1 = torch.randn(2, 24, 34)
    x2 = torch.randn(2, 24, 34)
    with torch.no_grad():
        p1 = small_model(x1)
        p2 = small_model(x2)
    assert not torch.allclose(
        p1["6"]["rain_probability"],
        p2["6"]["rain_probability"]
    )


# ── Loss function tests ───────────────────────────────────────
def test_loss_positive(small_model):
    criterion = WeatherLoss()
    x = torch.randn(4, 24, 34)
    with torch.no_grad():
        preds = small_model(x)
    targets = {
        "rain_probability": torch.rand(4),
        "precip_mm": torch.rand(4) * 20,
        "temperature_c": torch.randn(4) * 5 + 25,
        "wind_speed_kmh": torch.rand(4) * 40,
        "zonda_risk": torch.rand(4),
        "storm_risk": torch.rand(4),
        "hail_risk": torch.rand(4),
    }
    loss = criterion(preds["6"], targets)
    assert loss.item() > 0
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


# ── Zonda heuristic tests ─────────────────────────────────────
def test_zonda_proxy_computed(pipeline, sample_df):
    df = pipeline.add_zonda_proxy(sample_df.copy())
    assert "cordillera_pressure_hpa" in df.columns
    assert "thermal_differential_c" in df.columns
    assert "andes_plain_pressure_diff" in df.columns
    assert df["andes_plain_pressure_diff"].notna().all()


def test_zonda_proxy_respects_provided_values(pipeline, sample_df):
    """Real cordillera readings (synthetic/Windy) must not be overwritten."""
    df = pipeline.add_zonda_proxy(sample_df.copy())
    df["cordillera_pressure_hpa"] = 870.0
    df["thermal_differential_c"] = 18.0
    out = pipeline.add_zonda_proxy(df)
    assert (out["cordillera_pressure_hpa"] == 870.0).all()
    assert (out["thermal_differential_c"] == 18.0).all()


def test_cape_proxy_respects_provided_values(pipeline, sample_df):
    df = pipeline.add_cape_proxy(sample_df.copy())
    df["cape_j_kg"] = 3200.0
    out = pipeline.add_cape_proxy(df)
    assert (out["cape_j_kg"] == 3200.0).all()


def test_preprocess_preserves_labels(pipeline, sample_df):
    df = sample_df.copy()
    df["_label_zonda"] = 1.0
    df["_label_storm"] = 0.0
    out = pipeline.preprocess(df)
    assert "_label_zonda" in out.columns
    assert "_label_storm" in out.columns
    assert (out["_label_zonda"] == 1.0).all()


def test_build_training_tensors_label_override(pipeline, sample_df):
    """When _label_* present, extreme-risk targets follow the labels."""
    df = pipeline.preprocess(sample_df)
    df["_label_zonda"] = 1.0
    df["_label_storm"] = 1.0
    df["_label_hail"] = 1.0
    pipeline.fit_scaler(df)
    _, y = pipeline.build_training_tensors(df)

    # order per TARGET_COLS: rain, precip, temp, wind, zonda, storm, hail
    for h in pipeline.horizons:
        assert (y[str(h)][:, 4] == 1.0).all()   # zonda
        assert (y[str(h)][:, 5] == 1.0).all()   # storm
        assert (y[str(h)][:, 6] == 1.0).all()   # hail


def test_build_training_tensors_raw_targets(pipeline, sample_df):
    """Precip/temp/wind targets must be in raw units (mm, °C, km/h)."""
    df = pipeline.preprocess(sample_df)
    pipeline.fit_scaler(df)
    _, y = pipeline.build_training_tensors(df)

    temp_targets = y["6"][:, 2].numpy()
    # sample_df temperatures are in 15–35 °C
    assert temp_targets.min() >= 10.0
    assert temp_targets.max() <= 40.0
    # rain target is binary
    rain = y["6"][:, 0].numpy()
    assert set(rain) <= {0.0, 1.0}
