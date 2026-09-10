"""
app/ml/models/cnn_lstm.py
──────────────────────────
CNN-LSTM hybrid model for weather prediction.

Architecture:
  Input  → [batch, timesteps=72, features=34]
  CNN 1D → extract local temporal patterns (last few hours)
  LSTM   → learn long-range dependencies (days back)
  Attention → weight the most important timesteps
  Dense heads → one per forecast horizon (3/6/12/24/48/168h)

Each head predicts:
  - rain_probability  (sigmoid → 0–1)
  - precip_mm         (relu → ≥0)
  - temperature_c     (linear)
  - wind_speed_kmh    (relu → ≥0)
  - zonda_risk        (sigmoid → 0–1)
  - storm_risk        (sigmoid → 0–1)
  - hail_risk         (sigmoid → 0–1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """
    Soft attention over the LSTM hidden states sequence.
    Lets the model learn which timesteps matter most
    for each forecast horizon.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        # lstm_out: [batch, seq_len, hidden_size]
        scores = self.attn(lstm_out)          # [batch, seq_len, 1]
        weights = F.softmax(scores, dim=1)    # normalize over time
        context = (weights * lstm_out).sum(dim=1)  # [batch, hidden_size]
        return context


class ForecastHead(nn.Module):
    """
    Per-horizon prediction head.
    Outputs 7 values: rain_prob, precip, temp, wind,
    zonda_risk, storm_risk, hail_risk.
    """

    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(hidden_size, 7)

    def forward(self, x: torch.Tensor) -> dict:
        h = F.gelu(self.bn1(self.fc1(x)))
        h = self.drop(h)
        out = self.fc2(h)   # [batch, 7]

        return {
            "rain_probability": torch.sigmoid(out[:, 0]),
            "precip_mm":        F.relu(out[:, 1]),
            "temperature_c":    out[:, 2],            # unbounded
            "wind_speed_kmh":   F.relu(out[:, 3]),
            "zonda_risk":       torch.sigmoid(out[:, 4]),
            "storm_risk":       torch.sigmoid(out[:, 5]),
            "hail_risk":        torch.sigmoid(out[:, 6]),
        }


class CnnLstmWeatherModel(nn.Module):
    """
    CIELO·TUC main model.

    Parameters
    ----------
    n_features : int
        Number of input variables per timestep (default 34).
    n_timesteps : int
        Lookback window in hours (default 72).
    horizons : list[int]
        Forecast horizons in hours, e.g. [3, 6, 12, 24, 48, 168].
    cnn_channels : int
        Number of CNN output channels.
    lstm_hidden : int
        LSTM hidden state size.
    lstm_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout rate applied between LSTM layers.
    """

    def __init__(
        self,
        n_features: int = 34,
        n_timesteps: int = 72,
        horizons: list[int] | None = None,
        cnn_channels: int = 64,
        lstm_hidden: int = 128,
        lstm_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.horizons = horizons or [3, 6, 12, 24, 48, 168]
        self.n_timesteps = n_timesteps
        self.lstm_hidden = lstm_hidden

        # ── 1. CNN — local pattern extractor ──────────────────
        # kernel_size=3: looks at 3-hour windows
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=n_features,
                out_channels=cnn_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
        )

        # ── 2. LSTM — temporal dependency learner ─────────────
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=False,
        )

        # ── 3. Attention ──────────────────────────────────────
        self.attention = TemporalAttention(lstm_hidden)

        # ── 4. One forecast head per horizon ──────────────────
        self.heads = nn.ModuleDict(
            {str(h): ForecastHead(lstm_hidden) for h in self.horizons}
        )

    def forward(
        self, x: torch.Tensor
    ) -> dict[str, dict[str, torch.Tensor]]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape [batch, timesteps, features]

        Returns
        -------
        dict mapping horizon (str) → dict of prediction tensors
        """
        # CNN expects [batch, features, timesteps]
        cnn_in = x.permute(0, 2, 1)
        cnn_out = self.cnn(cnn_in)                    # [B, channels, T]
        lstm_in = cnn_out.permute(0, 2, 1)            # [B, T, channels]

        lstm_out, _ = self.lstm(lstm_in)              # [B, T, hidden]
        context = self.attention(lstm_out)             # [B, hidden]

        return {str(h): self.heads[str(h)](context) for h in self.horizons}

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Composite loss function ────────────────────────────────────
class WeatherLoss(nn.Module):
    """
    Multi-task loss combining:
      - Binary cross-entropy for rain probability (weighted for class imbalance)
      - MSE for precipitation amount
      - MAE for temperature
      - MSE for wind speed
      - BCE for extreme event risks (Zonda, storm, hail)

    The weights reflect relative importance and scale differences.
    """

    def __init__(
        self,
        rain_pos_weight: float = 3.0,  # rain is less frequent → upweight
        w_rain: float = 1.0,
        w_precip: float = 0.3,
        w_temp: float = 0.5,
        w_wind: float = 0.2,
        w_extremes: float = 0.8,
    ):
        super().__init__()
        self.bce_rain = nn.BCELoss(
            weight=torch.tensor(rain_pos_weight)
        )
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.bce = nn.BCELoss()

        self.w_rain = w_rain
        self.w_precip = w_precip
        self.w_temp = w_temp
        self.w_wind = w_wind
        self.w_extremes = w_extremes

    def forward(self, preds: dict, targets: dict) -> torch.Tensor:
        loss = torch.tensor(0.0, requires_grad=True)

        if "rain_probability" in targets:
            loss = loss + self.w_rain * self.bce_rain(
                preds["rain_probability"], targets["rain_probability"]
            )
        if "precip_mm" in targets:
            loss = loss + self.w_precip * self.mse(
                preds["precip_mm"], targets["precip_mm"]
            )
        if "temperature_c" in targets:
            loss = loss + self.w_temp * self.mae(
                preds["temperature_c"], targets["temperature_c"]
            )
        if "wind_speed_kmh" in targets:
            loss = loss + self.w_wind * self.mse(
                preds["wind_speed_kmh"], targets["wind_speed_kmh"]
            )
        for key in ("zonda_risk", "storm_risk", "hail_risk"):
            if key in targets:
                loss = loss + self.w_extremes * self.bce(
                    preds[key], targets[key]
                )

        return loss


# ── v2.0: Lightweight fast model ───────────────────────────────
class FastWeatherModel(nn.Module):
    """
    CIELO·TUC v2.0 — optimized for speed + accuracy.
    2-layer LSTM, 1 CNN layer, 96 hidden, supports dynamic n_features/lookback.
    Same output format as CnnLstmWeatherModel (7 targets per horizon).
    """

    def __init__(
        self,
        n_features: int = 48,
        n_timesteps: int = 24,
        horizons: list[int] | None = None,
    ):
        super().__init__()
        self.horizons = horizons or [3, 6, 12, 24, 48, 168]
        self.n_timesteps = n_timesteps
        self.hidden_size = 96

        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 48, kernel_size=3, padding=1),
            nn.BatchNorm1d(48),
            nn.GELU(),
        )

        self.lstm = nn.LSTM(
            input_size=48, hidden_size=self.hidden_size,
            num_layers=2, batch_first=True, dropout=0.15,
        )

        self.attn = nn.Linear(self.hidden_size, 1)

        self.heads = nn.ModuleDict()
        for h in self.horizons:
            self.heads[str(h)] = nn.Sequential(
                nn.Linear(self.hidden_size, 48),
                nn.BatchNorm1d(48),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(48, 7),
            )

    def forward(self, x: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        cnn_in = x.permute(0, 2, 1)
        cnn_out = self.cnn(cnn_in).permute(0, 2, 1)
        lstm_out, _ = self.lstm(cnn_out)

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
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── v5.0: Increased capacity over v2.0 ───────────────────────
class FastWeatherModelV5(nn.Module):
    """
    CIELO·TUC v5.0 — 10-zone trained, increased capacity.
    1-layer CNN (64 channels) + 2-layer LSTM (112 hidden) + Attention + heads (56).
    ~232K params, 53 features, 24h lookback.
    """

    def __init__(
        self,
        n_features: int = 53,
        n_timesteps: int = 24,
        horizons: list[int] | None = None,
    ):
        super().__init__()
        self.horizons = horizons or [3, 6, 12, 24, 48, 168]
        self.hidden_size = 112

        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )

        self.lstm = nn.LSTM(
            input_size=64, hidden_size=self.hidden_size,
            num_layers=2, batch_first=True, dropout=0.15,
        )

        self.attn = nn.Linear(self.hidden_size, 1)

        self.heads = nn.ModuleDict()
        for h in self.horizons:
            self.heads[str(h)] = nn.Sequential(
                nn.Linear(self.hidden_size, 56),
                nn.BatchNorm1d(56),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(56, 7),
            )

    def forward(self, x: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        cnn_in = x.permute(0, 2, 1)
        cnn_out = self.cnn(cnn_in).permute(0, 2, 1)
        lstm_out, _ = self.lstm(cnn_out)

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
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── v3.0: GPU-trained model with SE attention + residual ───────
class WeatherModelV3(nn.Module):
    """
    CIELO·TUC v3.0 — Colab GPU trained.
    3-layer LSTM (128 hidden) with residual skip, 2 CNN layers (64ch),
    Squeeze-and-Excitation channel attention, deeper forecast heads.
    517K params, 47 features, 24h lookback.
    """

    def __init__(
        self,
        n_features: int = 47,
        n_timesteps: int = 24,
        horizons: list[int] | None = None,
    ):
        super().__init__()
        self.horizons = horizons or [3, 6, 12, 24, 48, 168]
        self.hidden = 128

        self.cnn1 = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.GELU(),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.GELU(),
        )

        self.se_pool = nn.AdaptiveAvgPool1d(1)
        self.se_fc = nn.Sequential(nn.Linear(64, 16), nn.GELU(), nn.Linear(16, 64), nn.Sigmoid())

        self.lstm_proj = nn.Linear(64, self.hidden)
        self.lstm = nn.LSTM(
            input_size=self.hidden, hidden_size=self.hidden,
            num_layers=3, batch_first=True, dropout=0.2,
        )
        self.residual_proj = nn.Linear(64, self.hidden)

        self.attn = nn.Sequential(
            nn.Linear(self.hidden, 64), nn.Tanh(), nn.Linear(64, 1),
        )

        self.heads = nn.ModuleDict()
        for h in self.horizons:
            self.heads[str(h)] = nn.Sequential(
                nn.Linear(self.hidden, 64),
                nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(64, 32), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(32, 7),
            )

    def forward(self, x: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        c1 = self.cnn1(x.permute(0, 2, 1))
        c2 = self.cnn2(c1)
        c_out = c1 + c2

        se = self.se_pool(c_out).squeeze(-1)
        se = self.se_fc(se).unsqueeze(-1)
        c_out = c_out * se

        lstm_in = self.lstm_proj(c_out.permute(0, 2, 1))
        skip = self.residual_proj(c_out.permute(0, 2, 1))
        lstm_out, _ = self.lstm(lstm_in)
        lstm_out = lstm_out + skip

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
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
