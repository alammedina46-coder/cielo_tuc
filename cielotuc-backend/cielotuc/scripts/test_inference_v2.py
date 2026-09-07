"""Quick smoke test: load v2.0 model, build inference window, run forward pass."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
from app.ml.pipeline.data_pipeline import WeatherDataPipeline, FEATURE_COLS
from app.ml.models.cnn_lstm import FastWeatherModel

# 1. Load checkpoint
ckpt = torch.load("models/cielotuc_v2.0-20260907.pt", map_location="cpu", weights_only=False)
n_features = ckpt["n_features"]
n_timesteps = ckpt["n_timesteps"]
horizons = ckpt["horizons"]
feat_cols = ckpt.get("feat_cols", [])
print(f"Model: {n_features} features, {n_timesteps}h lookback")
print(f"Expected feat_cols: {len(feat_cols)}")

# 2. Create model and load weights
model = FastWeatherModel(n_features=n_features, n_timesteps=n_timesteps, horizons=horizons)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Model loaded OK, params: {model.n_parameters:,}")

# 3. Build a fake preprocessed DataFrame (simulating what prediction_service does)
pipeline = WeatherDataPipeline(lookback_hours=n_timesteps)
np.random.seed(42)
n_rows = n_timesteps + 5
df_data = {col: np.random.randn(n_rows) * 10 for col in FEATURE_COLS}
df_data["timestamp"] = pd.date_range("2026-01-01", periods=n_rows, freq="h")
df = pd.DataFrame(df_data)
df = pipeline.add_enhanced_features(df)
print(f"DataFrame columns after enhanced: {len(df.columns)}")

# 4. Build inference window (no scale, with feat_cols)
x = pipeline.build_inference_window(df, scale=False, columns=feat_cols)
print(f"Inference tensor shape: {x.shape}")
expected = (1, n_timesteps, n_features)
assert x.shape == expected, f"Shape mismatch: {x.shape} != {expected}"
print("Shape OK!")

# 5. Forward pass
with torch.no_grad():
    preds = model(x)
print(f"Forward pass OK, outputs: {list(preds.keys())}")
for h_str, v in preds.items():
    rain = v["rain_probability"][0].item()
    temp = v["temperature_c"][0].item()
    zonda = v["zonda_risk"][0].item()
    print(f"  {h_str}h: rain={rain:.3f} temp={temp:.1f}C zonda={zonda:.3f}")

print("\n=== FULL v2.0 INFERENCE PATH VERIFIED OK ===")
