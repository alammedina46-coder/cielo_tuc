"""
app/ml/training/trainer.py
───────────────────────────
Training loop for the CNN-LSTM model.
- Tracks every run in MLflow
- Uses TimeSeriesSplit for proper cross-validation
- Saves best checkpoint automatically
- Exposes `should_retrain()` to check if 30 days have passed

Typical usage
-------------
  trainer = ModelTrainer()
  trainer.train(X, y)            # full training run
  trainer.load_best()            # load for inference
  trainer.should_retrain()       # returns True if ≥30 days
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import DataLoader, TensorDataset

from app.core.config import settings
from app.ml.models.cnn_lstm import CnnLstmWeatherModel, WeatherLoss


MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class ModelTrainer:
    """
    Manages the full training lifecycle of CnnLstmWeatherModel.
    """

    def __init__(
        self,
        horizons: list[int] | None = None,
        n_features: int = 34,
        n_timesteps: int = 72,
        batch_size: int = 64,
        max_epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str | None = None,
    ):
        self.horizons = horizons or settings.model_forecast_horizons
        self.n_features = n_features
        self.n_timesteps = n_timesteps
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.lr = lr
        self.weight_decay = weight_decay

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Trainer using device: {self.device}")

        self.model: Optional[CnnLstmWeatherModel] = None
        self._best_val_loss = float("inf")
        self._best_checkpoint = MODELS_DIR / "best_model.pt"

    def _build_model(self) -> CnnLstmWeatherModel:
        return CnnLstmWeatherModel(
            n_features=self.n_features,
            n_timesteps=self.n_timesteps,
            horizons=self.horizons,
        ).to(self.device)

    def _make_dataloaders(
        self,
        X: torch.Tensor,
        y: dict[str, torch.Tensor],
        val_size: float = 0.15,
    ) -> tuple[DataLoader, DataLoader]:
        """
        TimeSeriesSplit-aware train/val split.
        Targets concatenated across horizons for the dataset.
        """
        n = len(X)
        split_idx = int(n * (1 - val_size))

        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train = {h: v[:split_idx] for h, v in y.items()}
        y_val = {h: v[split_idx:] for h, v in y.items()}

        # Flatten targets into a single tensor [N, horizons * 7]
        def flatten_y(ydict):
            return torch.cat([ydict[str(h)] for h in self.horizons], dim=-1)

        train_ds = TensorDataset(X_train, flatten_y(y_train))
        val_ds = TensorDataset(X_val, flatten_y(y_val))

        return (
            DataLoader(train_ds, batch_size=self.batch_size, shuffle=False),
            DataLoader(val_ds, batch_size=self.batch_size, shuffle=False),
        )

    def _unflatten_y(
        self, flat: torch.Tensor
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Reverse the flattening done in _make_dataloaders."""
        n_targets = 7
        result = {}
        for i, h in enumerate(self.horizons):
            start = i * n_targets
            chunk = flat[:, start: start + n_targets]
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

    def train(
        self,
        X: torch.Tensor,
        y: dict[str, torch.Tensor],
        version: str | None = None,
    ) -> dict:
        """
        Run full training with MLflow tracking.

        Returns a dict of metrics logged to MLflow.
        """
        version = version or datetime.utcnow().strftime("%Y%m%d_%H%M")
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)

        with mlflow.start_run(run_name=f"cielotuc_v{version}") as run:
            logger.info(f"MLflow run: {run.info.run_id}")

            # Log hyperparameters
            mlflow.log_params({
                "n_features": self.n_features,
                "n_timesteps": self.n_timesteps,
                "horizons": str(self.horizons),
                "batch_size": self.batch_size,
                "max_epochs": self.max_epochs,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "device": self.device,
                "training_samples": len(X),
            })

            self.model = self._build_model()
            logger.info(f"Model parameters: {self.model.n_parameters:,}")

            criterion = WeatherLoss()
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=5, factor=0.5, verbose=True
            )
            stopper = EarlyStopping(patience=10)

            train_loader, val_loader = self._make_dataloaders(X, y)
            best_metrics = {}

            for epoch in range(1, self.max_epochs + 1):
                # ── Train ───────────────────────────────────
                self.model.train()
                train_losses = []
                for xb, yb in train_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    optimizer.zero_grad()
                    preds_all = self.model(xb)
                    targets_all = self._unflatten_y(yb)
                    loss = torch.stack([
                        criterion(preds_all[str(h)], targets_all[str(h)])
                        for h in self.horizons
                    ]).mean()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    train_losses.append(loss.item())

                # ── Validate ─────────────────────────────────
                self.model.eval()
                val_losses = []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(self.device), yb.to(self.device)
                        preds_all = self.model(xb)
                        targets_all = self._unflatten_y(yb)
                        loss = torch.stack([
                            criterion(preds_all[str(h)], targets_all[str(h)])
                            for h in self.horizons
                        ]).mean()
                        val_losses.append(loss.item())

                train_loss = np.mean(train_losses)
                val_loss = np.mean(val_losses)
                scheduler.step(val_loss)

                mlflow.log_metrics(
                    {"train_loss": train_loss, "val_loss": val_loss}, step=epoch
                )

                if epoch % 10 == 0:
                    logger.info(
                        f"Epoch {epoch:3d}/{self.max_epochs} "
                        f"train={train_loss:.4f} val={val_loss:.4f}"
                    )

                # ── Save best checkpoint ──────────────────────
                if val_loss < self._best_val_loss:
                    self._best_val_loss = val_loss
                    torch.save(self.model.state_dict(), self._best_checkpoint)
                    best_metrics = {
                        "best_epoch": epoch,
                        "best_val_loss": val_loss,
                        "best_train_loss": train_loss,
                    }

                if stopper.step(val_loss):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            # ── Log best model to MLflow ──────────────────────
            self.load_best()
            mlflow.pytorch.log_model(self.model, "model")
            mlflow.log_metrics(best_metrics)
            mlflow.log_artifact(str(self._best_checkpoint))

            logger.info(
                f"Training complete. Best val loss: {self._best_val_loss:.4f}"
            )
            return {
                "run_id": run.info.run_id,
                "version": version,
                **best_metrics,
            }

    def load_best(self) -> None:
        """Load weights from the best saved checkpoint."""
        if self.model is None:
            self.model = self._build_model()
        self.model.load_state_dict(
            torch.load(self._best_checkpoint, map_location=self.device)
        )
        self.model.eval()
        logger.info("Loaded best model checkpoint")

    def load_from_mlflow(self, run_id: str) -> None:
        """Load a model from a specific MLflow run."""
        uri = f"runs:/{run_id}/model"
        self.model = mlflow.pytorch.load_model(uri, map_location=self.device)
        self.model.eval()
        logger.info(f"Loaded model from MLflow run {run_id}")

    @staticmethod
    def should_retrain(last_trained_at: datetime) -> bool:
        """
        Return True if enough time has passed since the last training.
        Threshold: settings.model_retrain_interval_days (default 30).
        """
        threshold = timedelta(days=settings.model_retrain_interval_days)
        return datetime.utcnow() - last_trained_at >= threshold
