"""PyTorch DNN+Attention model implementation.

Binary classification task for in-hospital mortality prediction on MIMIC-III sepsis data.
Architecture: MLP with multi-head attention residual block, converted from TensorFlow.

Data format (same as xgboost):
  - X.csv: Feature matrix (n_samples x n_features)
  - y.csv: Label column (hospital_expire_flag, 0/1)
  - X_train.csv / y_train.csv: Optional training split files
  - X_test.csv / y_test.csv: Optional test split files
  - feature_names.txt: Feature name list (one per line)
  - metadata.json: Dataset metadata
"""

import importlib
import json
import logging
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .model import Model

logger = logging.getLogger(__name__)

# Data file name constants
_X_FILE = "X.csv"
_Y_FILE = "y.csv"
_X_TRAIN_FILE = "X_train.csv"
_Y_TRAIN_FILE = "y_train.csv"
_X_TEST_FILE = "X_test.csv"
_Y_TEST_FILE = "y_test.csv"
_FEATURE_NAMES_FILE = "feature_names.txt"
_METADATA_FILE = "metadata.json"

# Export file name constants
_MODEL_FILE = "dnn_model.pt"
_MODEL_CONFIG_FILE = "model_config.json"
_FEATURE_NAMES_OUT_FILE = "feature_names.txt"
_METADATA_OUT_FILE = "metadata.json"
_METRICS_OUT_FILE = "metrics.json"
_CURVE_POINTS_FILE = "curve_points.json"
_ROC_PLOT_FILE = "roc_curve.png"
_PR_PLOT_FILE = "pr_curve.png"

# Explain file name constants
_EXPLAIN_METADATA_FILE = "explain_metadata.json"
_SHAP_SUMMARY_BAR_FILE = "shap_summary_bar.png"
_SHAP_SUMMARY_BEESWARM_FILE = "shap_summary_beeswarm.png"
_SHAP_VALUES_SAMPLE_FILE = "shap_values_sample.csv"
_SHAP_EXPECTED_VALUE_FILE = "shap_expected_value.json"
_TOP_FEATURES_FILE = "top_features.json"


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
        self._metadata: dict[str, Any] = {}
        self._is_trained: bool = False
        self._source_path: Path | None = None
        self._training_data_files: dict[str, str] = {
            "X": _X_FILE,
            "y": _Y_FILE,
        }
        self._evaluation_metrics: dict[str, Any] = {
            "evaluated": False,
            "reason": "evaluation not run yet",
        }
        self._curve_points: dict[str, Any] | None = None
        self._device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def train(
        self,
        source_path: Path,
        tuning: dict[str, Any] | None = None,
    ) -> None:
        source_path = Path(source_path)
        self._source_path = source_path
        logger.info("Loading data from %s...", source_path)

        X_train, y_train = self._load_training_data(source_path)
        self._feature_names = self._load_feature_names(source_path)
        self._metadata = self._load_metadata(source_path)
        self._metadata["training_data_files"] = self._training_data_files.copy()

        logger.info(
            "Data loaded: %d samples, %d features, positive rate %.2f%%",
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

        # Determine validation data
        test_data = self._load_test_data(source_path)
        if test_data is not None:
            X_val, y_val = test_data
            logger.info(
                "Using test split for validation: %d samples", len(X_val)
            )
        else:
            # Split training data for validation
            n_val = max(1, int(len(X_train) * self._config.val_split))
            rng = np.random.RandomState(42)
            indices = rng.permutation(len(X_train))
            val_idx = indices[:n_val]
            train_idx = indices[n_val:]
            X_val, y_val = X_train[val_idx], y_train[val_idx]
            X_train, y_train = X_train[train_idx], y_train[train_idx]
            logger.info(
                "No test split found; using %.0f%% holdout: %d train, %d val",
                self._config.val_split * 100,
                len(X_train),
                len(X_val),
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

        # Evaluate on test data if available
        self._evaluation_metrics = self._evaluate(X_val, y_val)

    def export(self, output_path: Path) -> None:
        if not self._is_trained or self._model is None:
            raise RuntimeError(
                "Model has not been trained. Call train() first."
            )

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save state dict
        model_file = output_path / _MODEL_FILE
        torch.save(self._model.state_dict(), model_file)
        logger.info("Model exported to %s", model_file)

        # Save model config for reconstruction
        config_file = output_path / _MODEL_CONFIG_FILE
        config_file.write_text(
            json.dumps(asdict(self._config), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Model config exported to %s", config_file)

        # Feature names
        feature_file = output_path / _FEATURE_NAMES_OUT_FILE
        feature_file.write_text(
            "\n".join(self._feature_names), encoding="utf-8"
        )

        # Copy explain source file
        self._export_explain_source_file(output_path)

        # Metadata
        export_metadata = {
            **self._metadata,
            "model": "dnn",
            "model_file": _MODEL_FILE,
            "model_config_file": _MODEL_CONFIG_FILE,
            "n_features": len(self._feature_names),
            "architecture": "MLP+Attention",
            "training_config": asdict(self._config),
            "device": str(self._device),
        }
        export_metadata = self._to_json_compatible(export_metadata)
        meta_file = output_path / _METADATA_OUT_FILE
        meta_file.write_text(
            json.dumps(
                export_metadata,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        # Metrics and curves
        export_eval_payload = dict(self._evaluation_metrics)
        curve_points_file = output_path / _CURVE_POINTS_FILE

        if self._curve_points is not None:
            curve_points_file.write_text(
                json.dumps(self._curve_points, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            curve_rendering = self._render_curves(
                self._curve_points, output_path
            )
            export_eval_payload["curve_rendering"] = curve_rendering

            curve_files: dict[str, str] = {"points": _CURVE_POINTS_FILE}
            if curve_rendering.get("roc", {}).get("rendered"):
                curve_files["roc"] = _ROC_PLOT_FILE
            if curve_rendering.get("pr", {}).get("rendered"):
                curve_files["pr"] = _PR_PLOT_FILE
            export_eval_payload["curve_files"] = curve_files

        metrics_file = output_path / _METRICS_OUT_FILE
        export_eval_payload = self._to_json_compatible(export_eval_payload)
        metrics_file.write_text(
            json.dumps(
                export_eval_payload,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        self._evaluation_metrics = export_eval_payload
        logger.info("Metrics exported to %s", metrics_file)

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

        self._feature_names = self._load_feature_names(weight_path)
        self._metadata = self._load_metadata(weight_path)
        self._is_trained = True

    def infer(self, source_path: Path) -> np.ndarray:
        if not self._is_trained or self._model is None:
            raise RuntimeError("Model has not been loaded. Call load() first.")

        source_path = Path(source_path)
        x_file = source_path / _X_FILE
        if not x_file.exists():
            raise FileNotFoundError(f"Feature file not found: {x_file}")

        X_df = pd.read_csv(x_file)
        if X_df.empty:
            raise ValueError("Feature file X.csv is empty.")

        X = X_df.values.astype(np.float32)
        logger.info("Running inference on %d samples...", X.shape[0])

        X_t = torch.tensor(X, dtype=torch.float32, device=self._device)
        self._model.eval()
        with torch.no_grad():
            proba = torch.sigmoid(self._model(X_t)).cpu().numpy().flatten()

        logger.info("Inference complete.")
        return proba

    def explain(
        self,
        source_path: Path,
        output_path: Path,
        max_samples: int = 500,
    ) -> dict[str, Any]:
        if max_samples < 1:
            raise ValueError("max_samples must be >= 1")

        source_path = Path(source_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        summary: dict[str, Any] = {
            "method": "shap",
            "explainer": "GradientExplainer",
            "model": "dnn",
            "sample_size": 0,
            "source_file": None,
            "outputs": {},
            "status": "failed",
            "reason": None,
        }

        try:
            if not self._is_trained or self._model is None:
                self.load(source_path)

            feature_file = self._resolve_explain_source_file(source_path)
            summary["source_file"] = feature_file.name

            X_df = pd.read_csv(feature_file)
            if X_df.empty:
                raise ValueError(
                    f"Explain source file is empty: {feature_file}"
                )

            sample_size = min(len(X_df), max_samples)
            if sample_size < len(X_df):
                X_sample = X_df.sample(n=sample_size, random_state=42)
            else:
                X_sample = X_df
            X_sample = X_sample.reset_index(drop=True)
            summary["sample_size"] = int(len(X_sample))

            outputs = self._build_explanations(X_sample, output_path)
            summary["outputs"] = outputs
            summary["status"] = "success"
        except Exception as error:
            summary["status"] = "failed"
            summary["reason"] = str(error)
            logger.exception("Failed to generate explainability artifacts")

        summary_file = output_path / _EXPLAIN_METADATA_FILE
        summary_file.write_text(
            json.dumps(
                self._to_json_compatible(summary),
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        if summary["status"] != "success":
            reason = summary.get("reason") or "unknown explain error"
            raise RuntimeError(f"Explainability failed: {reason}")

        return summary

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

        X_train_t = torch.tensor(
            X_train, dtype=torch.float32, device=self._device
        )
        y_train_t = torch.tensor(
            y_train, dtype=torch.float32, device=self._device
        ).unsqueeze(1)
        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=self._device)
        y_val_t = torch.tensor(
            y_val, dtype=torch.float32, device=self._device
        ).unsqueeze(1)

        best_val_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        epochs_no_improve = 0

        for epoch in range(cfg.epochs):
            self._model.train()
            indices = torch.randperm(len(X_train_t), device=self._device)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(X_train_t), cfg.batch_size):
                batch_idx = indices[start : start + cfg.batch_size]
                if len(batch_idx) < 2:
                    continue  # BatchNorm1d requires batch size >= 2
                xb, yb = X_train_t[batch_idx], y_train_t[batch_idx]

                optimizer.zero_grad()
                loss = criterion(self._model(xb), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            # Validation
            self._model.eval()
            with torch.no_grad():
                val_loss = criterion(self._model(X_val_t), y_val_t).item()

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
    # Private: evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, X_val: np.ndarray, y_val: np.ndarray) -> dict[str, Any]:
        if not self._is_trained or self._model is None:
            return {"evaluated": False, "reason": "model not trained"}

        self._model.eval()
        X_t = torch.tensor(X_val, dtype=torch.float32, device=self._device)
        with torch.no_grad():
            y_score = torch.sigmoid(self._model(X_t)).cpu().numpy().flatten()

        y_pred = (y_score >= 0.5).astype(np.int32)
        self._curve_points = self._build_curve_points(y_val, y_score)

        metrics: dict[str, float | None] = {
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "precision": float(precision_score(y_val, y_pred, zero_division=0)),
            "recall": float(recall_score(y_val, y_pred, zero_division=0)),
            "f1": float(f1_score(y_val, y_pred, zero_division=0)),
            "roc_auc": None,
            "pr_auc": None,
        }

        try:
            metrics["roc_auc"] = float(roc_auc_score(y_val, y_score))
        except ValueError:
            logger.warning("ROC-AUC undefined for current label distribution")

        try:
            metrics["pr_auc"] = float(average_precision_score(y_val, y_score))
        except ValueError:
            logger.warning("PR-AUC undefined for current label distribution")

        matrix = confusion_matrix(y_val, y_pred, labels=[0, 1])
        tn, fp, fn, tp = matrix.ravel()

        payload: dict[str, Any] = {
            "evaluated": True,
            "model": "dnn",
            "threshold": 0.5,
            "test_set": {
                "n_samples": int(len(y_val)),
                "positive_rate": float(y_val.mean()),
            },
            "metrics": metrics,
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
            "training_data_files": self._training_data_files.copy(),
        }
        logger.info("Evaluation complete: %s", payload["metrics"])
        return payload

    # ------------------------------------------------------------------
    # Private: explainability
    # ------------------------------------------------------------------

    def _resolve_explain_source_file(self, source_path: Path) -> Path:
        test_file = source_path / _X_TEST_FILE
        if test_file.exists():
            return test_file
        full_file = source_path / _X_FILE
        if full_file.exists():
            return full_file
        raise FileNotFoundError(
            f"Explain input file not found: {test_file} or {full_file}"
        )

    def _export_explain_source_file(self, output_path: Path) -> None:
        if self._source_path is None:
            return
        try:
            source_file = self._resolve_explain_source_file(self._source_path)
            target_file = output_path / source_file.name
            shutil.copy2(source_file, target_file)
            logger.info("Explain source features exported to %s", target_file)
        except FileNotFoundError:
            pass

    def _build_explanations(
        self, X_sample: pd.DataFrame, output_path: Path
    ) -> dict[str, str]:
        if self._model is None:
            raise RuntimeError("Model is unavailable for explainability")

        try:
            shap_module = cast(Any, importlib.import_module("shap"))
        except Exception as error:
            raise RuntimeError(f"shap unavailable: {error}") from error

        try:
            matplotlib_module = cast(Any, importlib.import_module("matplotlib"))
            matplotlib_module.use("Agg")
            plt = cast(Any, importlib.import_module("matplotlib.pyplot"))
        except Exception as error:
            raise RuntimeError(f"matplotlib unavailable: {error}") from error

        # Use GradientExplainer for PyTorch models
        self._model.eval()
        background = torch.tensor(
            X_sample.values[: min(100, len(X_sample))].astype(np.float32),
            device=self._device,
        )

        explainer = shap_module.GradientExplainer(self._model, background)
        test_tensor = torch.tensor(
            X_sample.values.astype(np.float32), device=self._device
        )
        raw_shap_values = explainer.shap_values(test_tensor)

        shap_values = self._normalize_shap_values(
            raw_shap_values,
            n_samples=len(X_sample),
            n_features=X_sample.shape[1],
        )

        # Summary bar plot
        shap_module.summary_plot(
            shap_values,
            X_sample,
            plot_type="bar",
            show=False,
            max_display=20,
        )
        bar_plot_file = output_path / _SHAP_SUMMARY_BAR_FILE
        bar_fig = plt.gcf()
        bar_fig.tight_layout()
        bar_fig.savefig(str(bar_plot_file), dpi=150)
        plt.close(bar_fig)

        # Beeswarm plot
        shap_module.summary_plot(
            shap_values,
            X_sample,
            show=False,
            max_display=20,
        )
        beeswarm_file = output_path / _SHAP_SUMMARY_BEESWARM_FILE
        beeswarm_fig = plt.gcf()
        beeswarm_fig.tight_layout()
        beeswarm_fig.savefig(str(beeswarm_file), dpi=150)
        plt.close(beeswarm_fig)

        # SHAP values CSV
        shap_df = pd.DataFrame(shap_values, columns=X_sample.columns)
        shap_sample_file = output_path / _SHAP_VALUES_SAMPLE_FILE
        shap_df.to_csv(shap_sample_file, index=False)

        # Top features
        mean_abs = np.abs(shap_values).mean(axis=0)
        order = np.argsort(mean_abs)[::-1]
        top_features: list[dict[str, float | str]] = [
            {
                "feature": str(X_sample.columns[int(idx)]),
                "mean_abs_shap": float(mean_abs[int(idx)]),
            }
            for idx in order[: min(20, len(order))]
        ]
        top_features_file = output_path / _TOP_FEATURES_FILE
        top_features_file.write_text(
            json.dumps(top_features, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Expected value
        expected_value_payload = {
            "expected_value": self._normalize_expected_value(
                getattr(explainer, "expected_value", None)
            ),
            "n_samples": int(len(X_sample)),
            "n_features": int(X_sample.shape[1]),
            "shap_version": str(getattr(shap_module, "__version__", "unknown")),
            "explainer": "GradientExplainer",
        }
        expected_value_file = output_path / _SHAP_EXPECTED_VALUE_FILE
        expected_value_file.write_text(
            json.dumps(
                self._to_json_compatible(expected_value_payload),
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        return {
            "summary_bar": _SHAP_SUMMARY_BAR_FILE,
            "summary_beeswarm": _SHAP_SUMMARY_BEESWARM_FILE,
            "shap_values_sample": _SHAP_VALUES_SAMPLE_FILE,
            "expected_value": _SHAP_EXPECTED_VALUE_FILE,
            "top_features": _TOP_FEATURES_FILE,
            "metadata": _EXPLAIN_METADATA_FILE,
        }

    def _normalize_shap_values(
        self,
        raw_values: Any,
        n_samples: int,
        n_features: int,
    ) -> np.ndarray:
        values = raw_values
        if isinstance(values, list):
            if not values:
                raise RuntimeError("SHAP returned empty value list")
            values = values[-1] if len(values) > 1 else values[0]

        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 3:
            array = array[:, :, -1]
        if array.ndim != 2:
            raise RuntimeError(f"Unexpected SHAP value shape: {array.shape}")

        if array.shape[0] != n_samples and array.shape[1] == n_samples:
            array = array.T

        if array.shape != (n_samples, n_features):
            raise RuntimeError(
                f"Unexpected SHAP matrix shape: {array.shape}, "
                f"expected {(n_samples, n_features)}"
            )
        return array

    def _normalize_expected_value(self, value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) == 0:
                return None
            value = value[-1]
        if isinstance(value, torch.Tensor):
            value = value.item()
        number = float(value)
        return number if math.isfinite(number) else None

    # ------------------------------------------------------------------
    # Private: data loading (shared conventions with xgboost)
    # ------------------------------------------------------------------

    def _load_training_data(
        self, source_path: Path
    ) -> tuple[np.ndarray, np.ndarray]:
        train_x_file = source_path / _X_TRAIN_FILE
        train_y_file = source_path / _Y_TRAIN_FILE

        if train_x_file.exists() and train_y_file.exists():
            self._training_data_files = {
                "X": _X_TRAIN_FILE,
                "y": _Y_TRAIN_FILE,
            }
            logger.info(
                "Using train split files: %s and %s",
                train_x_file.name,
                train_y_file.name,
            )
            return self._load_data_files(train_x_file, train_y_file)

        self._training_data_files = {"X": _X_FILE, "y": _Y_FILE}
        logger.info(
            "Train split not found; falling back to %s and %s",
            _X_FILE,
            _Y_FILE,
        )
        x_file = source_path / _X_FILE
        y_file = source_path / _Y_FILE
        return self._load_data_files(x_file, y_file)

    def _load_test_data(
        self, source_path: Path
    ) -> tuple[np.ndarray, np.ndarray] | None:
        test_x = source_path / _X_TEST_FILE
        test_y = source_path / _Y_TEST_FILE
        if not test_x.exists() or not test_y.exists():
            return None
        return self._load_data_files(test_x, test_y)

    def _load_data_files(
        self, x_file: Path, y_file: Path
    ) -> tuple[np.ndarray, np.ndarray]:
        if not x_file.exists():
            raise FileNotFoundError(f"Feature file not found: {x_file}")
        if not y_file.exists():
            raise FileNotFoundError(f"Label file not found: {y_file}")

        X_df = pd.read_csv(x_file)
        y_df = pd.read_csv(y_file)

        if X_df.empty or y_df.empty:
            raise ValueError("Data files are empty.")
        if len(X_df) != len(y_df):
            raise ValueError(
                f"Sample count mismatch: X={len(X_df)}, y={len(y_df)}"
            )

        y_col = y_df.columns[0]
        X = X_df.to_numpy(dtype=np.float32)
        y = y_df[y_col].to_numpy(dtype=np.int32)
        return X, y

    def _load_feature_names(self, source_path: Path) -> list[str]:
        feature_file = source_path / _FEATURE_NAMES_FILE
        if not feature_file.exists():
            logger.warning("Feature names file not found: %s", feature_file)
            return []
        return [
            line.strip()
            for line in feature_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _load_metadata(self, source_path: Path) -> dict[str, Any]:
        meta_file = source_path / _METADATA_FILE
        if not meta_file.exists():
            logger.warning("Metadata file not found: %s", meta_file)
            return {}
        return json.loads(meta_file.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Private: curve building & rendering
    # ------------------------------------------------------------------

    def _build_curve_points(
        self, y_true: np.ndarray, y_score: np.ndarray
    ) -> dict[str, Any]:
        def _safe_float_list(values: Any) -> list[float | None]:
            output: list[float | None] = []
            for v in values:
                n = float(v)
                output.append(n if np.isfinite(n) else None)
            return output

        points: dict[str, Any] = {}
        try:
            fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
            points["roc"] = {
                "fpr": _safe_float_list(fpr),
                "tpr": _safe_float_list(tpr),
                "thresholds": _safe_float_list(roc_thresholds),
            }
        except ValueError as e:
            points["roc"] = {
                "error": str(e),
                "fpr": [],
                "tpr": [],
                "thresholds": [],
            }

        try:
            precision, recall, pr_thresholds = precision_recall_curve(
                y_true, y_score
            )
            points["pr"] = {
                "precision": _safe_float_list(precision),
                "recall": _safe_float_list(recall),
                "thresholds": _safe_float_list(pr_thresholds),
            }
        except ValueError as e:
            points["pr"] = {
                "error": str(e),
                "precision": [],
                "recall": [],
                "thresholds": [],
            }

        return points

    def _render_curves(
        self, curve_points: dict[str, Any], output_path: Path
    ) -> dict[str, Any]:
        try:
            matplotlib_module = importlib.import_module("matplotlib")
            matplotlib = cast(Any, matplotlib_module)
            matplotlib.use("Agg")
            plt = cast(Any, importlib.import_module("matplotlib.pyplot"))
        except Exception as error:
            reason = f"matplotlib unavailable: {error}"
            logger.warning("Skipping curve rendering: %s", reason)
            return {
                "roc": {"rendered": False, "skipped_reason": reason},
                "pr": {"rendered": False, "skipped_reason": reason},
            }

        rendering_status: dict[str, Any] = {
            "roc": {"rendered": False},
            "pr": {"rendered": False},
        }

        roc_points = curve_points.get("roc", {})
        if isinstance(roc_points, dict):
            fpr = cast(list[float], roc_points.get("fpr", []))
            tpr = cast(list[float], roc_points.get("tpr", []))
            if fpr and tpr:
                roc_path = output_path / _ROC_PLOT_FILE
                fig = plt.figure(figsize=(6, 6))
                plt.plot(fpr, tpr, linewidth=2, label="ROC")
                plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
                plt.xlim(0.0, 1.0)
                plt.ylim(0.0, 1.05)
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC Curve")
                plt.grid(alpha=0.3)
                plt.legend(loc="lower right")
                plt.tight_layout()
                fig.savefig(str(roc_path), dpi=150)
                plt.close(fig)
                rendering_status["roc"] = {
                    "rendered": True,
                    "file": _ROC_PLOT_FILE,
                }

        pr_points = curve_points.get("pr", {})
        if isinstance(pr_points, dict):
            precision = cast(list[float], pr_points.get("precision", []))
            recall = cast(list[float], pr_points.get("recall", []))
            if precision and recall:
                pr_path = output_path / _PR_PLOT_FILE
                fig = plt.figure(figsize=(6, 6))
                plt.plot(recall, precision, linewidth=2, label="PR")
                plt.xlim(0.0, 1.0)
                plt.ylim(0.0, 1.05)
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title("Precision-Recall Curve")
                plt.grid(alpha=0.3)
                plt.legend(loc="lower left")
                plt.tight_layout()
                fig.savefig(str(pr_path), dpi=150)
                plt.close(fig)
                rendering_status["pr"] = {
                    "rendered": True,
                    "file": _PR_PLOT_FILE,
                }

        return rendering_status

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
