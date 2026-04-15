"""PyTorch DNN+Attention model implementation.

Binary classification for in-hospital mortality prediction on MIMIC-III sepsis data.
Architecture: MLP with multi-head attention residual block.

Consumes build artifacts via the data_loader module; evaluation is handled
externally by the evaluator module.
"""

import importlib
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .explainer import (
    ShapResult,
    normalize_shap_values,
    normalize_expected_value,
)
from .model import Model

logger = logging.getLogger(__name__)

# Export file name constants
_MODEL_FILE = "dnn_model.pt"
_MODEL_CONFIG_FILE = "model_config.json"
_FEATURE_NAMES_OUT_FILE = "feature_names.txt"


# ---------------------------------------------------------------------------
# Network architecture
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Network architecture and training hyperparameters."""

    input_dim: int = 0
    heads: int = 8
    key_dim: int = 64
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-4
    patience: int = 10
    lr_factor: float = 0.7
    lr_patience: int = 3
    val_split: float = 0.2


class CustomAttentionLayer(nn.Module):
    """Multi-head attention operating on the feature dimension.

    Input shape : (batch, features)
    Output shape: (batch, features)
    """

    def __init__(self, input_dim: int, heads: int = 4, key_dim: int = 32):
        super().__init__()
        self.heads = heads
        self.key_dim = key_dim
        inner_dim = heads * key_dim

        self.query_dense = nn.Linear(input_dim, inner_dim)
        self.key_dense = nn.Linear(input_dim, inner_dim)
        self.value_dense = nn.Linear(input_dim, inner_dim)
        self.output_dense = nn.Linear(inner_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self.query_dense(x)
        key = self.key_dense(x)
        value = self.value_dense(x)

        scale = self.key_dim**0.5
        scores = torch.matmul(query.unsqueeze(-1), key.unsqueeze(-2))
        attention_weights = torch.softmax(scores / scale, dim=-1)
        attended = torch.matmul(attention_weights, value.unsqueeze(-1)).squeeze(
            -1
        )
        return self.output_dense(attended)


class AdvancedMortalityModel(nn.Module):
    """MLP + multi-head attention residual block for binary mortality prediction.

    Architecture: 256 → 128 → attention residual → 64 → 32 → 1
    """

    def __init__(self, input_dim: int, heads: int = 8, key_dim: int = 64):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
        )
        self.block2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
        )
        self.attention = CustomAttentionLayer(128, heads=heads, key_dim=key_dim)
        self.block3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
        )
        self.block4 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
        )
        self.head = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = x + self.attention(x)  # residual connection
        x = self.block3(x)
        x = self.block4(x)
        return self.head(x)  # raw logit


# ---------------------------------------------------------------------------
# DNNModel — smep Model implementation
# ---------------------------------------------------------------------------


class DNNModel(Model):
    """PyTorch MLP+Attention model for in-hospital mortality prediction.

    Trains a binary classifier using AdvancedMortalityModel on processed
    MIMIC-III features, with early stopping and learning rate scheduling.
    """

    def __init__(self) -> None:
        self._model: Optional[AdvancedMortalityModel] = None
        self._config: ModelConfig = ModelConfig()
        self._feature_names: list[str] = []
        self._is_trained: bool = False
        self._source_path: Path | None = None
        self._device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        tuning: dict[str, Any] | None = None,
    ) -> None:
        logger.info(
            "Training DNN: %d samples, %d features, positive rate %.2f%%",
            X_train.shape[0],
            X_train.shape[1],
            y_train.mean() * 100,
        )

        if tuning is not None:
            strategy = str(tuning.get("strategy", "none")).strip().lower()
            if strategy != "none":
                logger.warning(
                    "DNN model does not support tuning strategy '%s'; "
                    "using default hyperparameters.",
                    strategy,
                )

        # Use provided validation data; fall back to holdout if not provided
        if X_val is None or y_val is None:
            n_val = max(1, int(len(X_train) * self._config.val_split))
            rng = np.random.RandomState(42)
            indices = rng.permutation(len(X_train))
            val_idx = indices[:n_val]
            train_idx = indices[n_val:]
            X_val, y_val = X_train[val_idx], y_train[val_idx]
            X_train, y_train = X_train[train_idx], y_train[train_idx]
            logger.info(
                "No val data provided; using %.0f%% holdout: %d train, %d val",
                self._config.val_split * 100,
                len(X_train),
                len(X_val),
            )
        else:
            logger.info(
                "Using provided validation data: %d samples", len(X_val)
            )

        self._config.input_dim = X_train.shape[1]
        self._model = AdvancedMortalityModel(
            input_dim=self._config.input_dim,
            heads=self._config.heads,
            key_dim=self._config.key_dim,
        ).to(self._device)

        self._train_loop(X_train, y_train, X_val, y_val)
        self._is_trained = True
        logger.info("Training complete.")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._is_trained or self._model is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")

        self._model.eval()
        batch_size = self._config.batch_size
        results: list[np.ndarray] = []
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=(self._device.type == "cuda"),
        )
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self._device, non_blocking=True)
                proba = torch.sigmoid(self._model(xb)).cpu().numpy().flatten()
                results.append(proba)
        return np.concatenate(results)

    def save(self, output_path: Path) -> dict[str, Any]:
        if not self._is_trained or self._model is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save state dict
        model_file = output_path / _MODEL_FILE
        torch.save(self._model.state_dict(), model_file)
        logger.info("Model saved to %s", model_file)

        # Save model config for reconstruction
        config_file = output_path / _MODEL_CONFIG_FILE
        config_file.write_text(
            json.dumps(asdict(self._config), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Model config saved to %s", config_file)

        return {
            "model": "dnn",
            "model_file": _MODEL_FILE,
            "model_config_file": _MODEL_CONFIG_FILE,
            "architecture": "MLP+Attention",
            "training_config": self._to_json_compatible(asdict(self._config)),
            "device": str(self._device),
        }

    def load(self, weight_path: Path) -> None:
        weight_path = Path(weight_path)
        self._source_path = weight_path

        if not weight_path.exists() or not weight_path.is_dir():
            raise FileNotFoundError(
                f"Weight directory not found: {weight_path}"
            )

        # Load model config
        config_file = weight_path / _MODEL_CONFIG_FILE
        if not config_file.exists():
            raise FileNotFoundError(
                f"Model config file not found: {config_file}"
            )
        config_data = json.loads(config_file.read_text(encoding="utf-8"))
        self._config = ModelConfig(**config_data)

        # Rebuild and load model
        model_file = weight_path / _MODEL_FILE
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        self._model = AdvancedMortalityModel(
            input_dim=self._config.input_dim,
            heads=self._config.heads,
            key_dim=self._config.key_dim,
        ).to(self._device)

        self._model.load_state_dict(
            torch.load(model_file, map_location=self._device, weights_only=True)
        )
        self._model.eval()
        logger.info("Model loaded from %s", model_file)

        feature_file = weight_path / _FEATURE_NAMES_OUT_FILE
        if feature_file.exists():
            self._feature_names = [
                line.strip()
                for line in feature_file.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]

        self._is_trained = True

    def compute_shap(
        self,
        X: pd.DataFrame,
        max_samples: int = 500,
    ) -> ShapResult:
        if not self._is_trained or self._model is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")

        shap_module = importlib.import_module("shap")

        self._model.eval()
        bg_size = min(100, len(X))
        background = torch.tensor(
            X.values[:bg_size].astype(np.float32),
            device=self._device,
        )

        explainer = shap_module.GradientExplainer(self._model, background)  # type: ignore[attr-defined]
        test_tensor = torch.tensor(
            X.values.astype(np.float32), device=self._device
        )
        raw_shap_values = explainer.shap_values(test_tensor)

        shap_values = normalize_shap_values(
            raw_shap_values,
            n_samples=len(X),
            n_features=X.shape[1],
        )
        expected = normalize_expected_value(
            getattr(explainer, "expected_value", None)
        )

        return ShapResult(
            shap_values=shap_values,
            expected_value=expected,
            explainer_type="GradientExplainer",
            shap_version=str(getattr(shap_module, "__version__", "unknown")),
        )

    # ------------------------------------------------------------------
    # Private: training loop
    # ------------------------------------------------------------------

    def _train_loop(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        assert self._model is not None
        cfg = self._config

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self._model.parameters(), lr=cfg.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=cfg.lr_factor, patience=cfg.lr_patience
        )

        # Keep tensors on CPU; stream batches to GPU via DataLoader
        use_pin = self._device.type == "cuda"
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32).unsqueeze(1),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            pin_memory=use_pin,
            drop_last=(len(train_dataset) % cfg.batch_size == 1),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            pin_memory=use_pin,
        )

        best_val_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        epochs_no_improve = 0

        for epoch in range(cfg.epochs):
            self._model.train()
            epoch_loss = 0.0
            n_batches = 0

            for xb, yb in train_loader:
                if len(xb) < 2:
                    continue  # BatchNorm1d requires batch size >= 2
                xb = xb.to(self._device, non_blocking=True)
                yb = yb.to(self._device, non_blocking=True)

                optimizer.zero_grad()
                loss = criterion(self._model(xb), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            # Validation (batched to avoid OOM)
            self._model.eval()
            val_loss_sum = 0.0
            val_n = 0
            with torch.no_grad():
                for xb_v, yb_v in val_loader:
                    xb_v = xb_v.to(self._device, non_blocking=True)
                    yb_v = yb_v.to(self._device, non_blocking=True)
                    val_loss_sum += criterion(
                        self._model(xb_v), yb_v
                    ).item() * len(xb_v)
                    val_n += len(xb_v)
            val_loss = val_loss_sum / max(val_n, 1)

            scheduler.step(val_loss)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                avg_train = epoch_loss / max(n_batches, 1)
                logger.info(
                    "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f",
                    epoch + 1,
                    cfg.epochs,
                    avg_train,
                    val_loss,
                )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    k: v.cpu().clone()
                    for k, v in self._model.state_dict().items()
                }
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= cfg.patience:
                    logger.info(
                        "Early stopping at epoch %d (patience=%d)",
                        epoch + 1,
                        cfg.patience,
                    )
                    break

        # Restore best weights
        if best_state is not None:
            self._model.load_state_dict(best_state)
            self._model.to(self._device)

    # ------------------------------------------------------------------
    # Private: JSON utilities
    # ------------------------------------------------------------------

    def _to_json_compatible(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(k): self._to_json_compatible(v) for k, v in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [self._to_json_compatible(item) for item in value]
        if isinstance(value, np.generic):
            return self._to_json_compatible(value.item())
        if isinstance(value, torch.Tensor):
            return self._to_json_compatible(value.item())
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        return value
