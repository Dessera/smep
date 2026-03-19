"""XGBoost model implementation.

Binary classification task for in-hospital mortality prediction on MIMIC-III sepsis data.
Data format:
  - X.csv: Feature matrix (n_samples x n_features)
  - y.csv: Label column (hospital_expire_flag, 0/1)
    - X_train.csv / y_train.csv: Optional training split files
  - feature_names.txt: Feature name list (one per line)
  - metadata.json: Dataset metadata
"""

import json
import logging
import importlib
from pathlib import Path
from typing import Any, Optional, cast

import joblib
import numpy as np
import pandas as pd
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

from xgboost import XGBClassifier

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
_MODEL_FILE = "xgboost_model.joblib"
_FEATURE_NAMES_OUT_FILE = "feature_names.txt"
_METADATA_OUT_FILE = "metadata.json"
_METRICS_OUT_FILE = "metrics.json"
_CURVE_POINTS_FILE = "curve_points.json"
_ROC_PLOT_FILE = "roc_curve.png"
_PR_PLOT_FILE = "pr_curve.png"


class XGBoostModel(Model):
    """XGBoost-based in-hospital mortality prediction model for sepsis.

    Trains a binary classifier using XGBClassifier on processed MIMIC-III features,
    and exports model weights and preprocessors via joblib serialization.

    Attributes:
        _classifier: Trained XGBClassifier instance.
        _feature_names: List of feature names.
        _metadata: Dataset metadata dictionary.
        _is_trained: Whether the model has been trained.
    """

    def __init__(self) -> None:
        """Initialize XGBoostModel."""
        self._classifier: Optional[XGBClassifier] = None
        self._feature_names: list[str] = []
        self._metadata: dict[str, Any] = {}
        self._is_trained: bool = False
        self._training_data_files: dict[str, str] = {
            "X": _X_FILE,
            "y": _Y_FILE,
        }
        self._evaluation_metrics: dict[str, Any] = {
            "evaluated": False,
            "reason": "evaluation not run yet",
        }
        self._curve_points: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def train(self, source_path: Path) -> None:
        """Load processed data from source_path and train the XGBoost model.

        Training pipeline:
        1. Prefer X_train.csv / y_train.csv when available
        2. Fall back to X.csv / y.csv for backward compatibility
        3. Train the model on the selected training dataset

        Args:
            source_path: Path to the processed data directory.

        Raises:
            FileNotFoundError: If required data files are missing.
            ValueError: If data format is invalid or sample count is insufficient.
            RuntimeError: If the training process fails.
        """
        source_path = Path(source_path)
        logger.info(f"Loading data from {source_path}...")

        X, y = self._load_training_data(source_path)
        self._feature_names = self._load_feature_names(source_path)
        self._metadata = self._load_metadata(source_path)
        self._metadata["training_data_files"] = self._training_data_files.copy()

        logger.info(
            f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features, "
            f"positive rate {y.mean():.2%}"
        )

        # Build classifier (hyperparameters tuned for small datasets)
        scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
        self._classifier = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=float(scale_pos_weight),
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )

        # Train model on the selected training dataset
        logger.info(
            "Training model using %s and %s...",
            self._training_data_files["X"],
            self._training_data_files["y"],
        )
        self._classifier.fit(X, y)
        self._is_trained = True
        logger.info("Training complete.")

        self._evaluation_metrics = self._evaluate_on_available_test_data(
            source_path
        )

    def export(self, output_path: Path) -> None:
        """Export the trained model and preprocessors to output_path directory.

        Exported files:
        - xgboost_model.joblib: XGBClassifier weights
        - feature_names.txt: Feature name list
        - metadata.json: Metadata including original dataset info and training config
        - metrics.json: Post-training test-set evaluation metrics

        Args:
            output_path: Export directory path (created automatically if not exists).

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if not self._is_trained:
            raise RuntimeError(
                "Model has not been trained. Call train() first."
            )
        if self._classifier is None:
            raise RuntimeError("Classifier is unavailable after training")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export model
        model_file = output_path / _MODEL_FILE
        joblib.dump(self._classifier, model_file)
        logger.info(f"Model exported to {model_file}")

        # Export feature names
        feature_file = output_path / _FEATURE_NAMES_OUT_FILE
        feature_file.write_text(
            "\n".join(self._feature_names), encoding="utf-8"
        )
        logger.info(f"Feature names exported to {feature_file}")

        # Export metadata
        export_metadata = {
            **self._metadata,
            "model": "xgboost",
            "model_file": _MODEL_FILE,
            "n_features": len(self._feature_names),
            "classifier_params": self._classifier.get_params(),
        }
        meta_file = output_path / _METADATA_OUT_FILE
        meta_file.write_text(
            json.dumps(export_metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Metadata exported to {meta_file}")

        export_eval_payload = dict(self._evaluation_metrics)
        curve_points_file = output_path / _CURVE_POINTS_FILE

        if self._curve_points is not None:
            curve_points_file.write_text(
                json.dumps(self._curve_points, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info(f"Curve points exported to {curve_points_file}")

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
        metrics_file.write_text(
            json.dumps(export_eval_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._evaluation_metrics = export_eval_payload
        logger.info(f"Evaluation metrics exported to {metrics_file}")

    def load(self, weight_path: Path) -> None:
        """Load trained weights from weight_path directory.

        Loads the following files produced by export():
        - xgboost_model.joblib: XGBClassifier weights
        - feature_names.txt: Feature name list
        - metadata.json: Dataset and training metadata

        Args:
            weight_path: Path to the directory produced by export().

        Raises:
            FileNotFoundError: If required weight files are missing.
            RuntimeError: If loading fails.
        """
        weight_path = Path(weight_path)
        if not weight_path.exists() or not weight_path.is_dir():
            raise FileNotFoundError(
                f"Weight directory not found: {weight_path}"
            )

        model_file = weight_path / _MODEL_FILE
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        try:
            self._classifier = joblib.load(model_file)
            logger.info(f"Model loaded from {model_file}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}") from e

        self._feature_names = self._load_feature_names(weight_path)
        self._metadata = self._load_metadata(weight_path)
        self._is_trained = True
        logger.info("Model weights loaded successfully.")

    def infer(self, source_path: Path) -> np.ndarray:
        """Run inference on data located at source_path.

        Args:
            source_path: Path to the processed data directory containing X.csv.

        Returns:
            1-D numpy array of predicted mortality probabilities (float32).

        Raises:
            RuntimeError: If the model has not been loaded.
            FileNotFoundError: If X.csv is missing in source_path.
        """
        if not self._is_trained or self._classifier is None:
            raise RuntimeError("Model has not been loaded. Call load() first.")

        source_path = Path(source_path)
        x_file = source_path / _X_FILE
        if not x_file.exists():
            raise FileNotFoundError(f"Feature file not found: {x_file}")

        X_df = pd.read_csv(x_file)
        if X_df.empty:
            raise ValueError("Feature file X.csv is empty.")

        X = X_df.values.astype(np.float32)
        logger.info(f"Running inference on {X.shape[0]} samples...")

        proba = self._classifier.predict_proba(X)[:, 1]
        logger.info("Inference complete.")
        return proba

    def get_evaluation_summary(self) -> dict[str, Any]:
        """Return evaluation payload collected during the latest training."""
        return self._evaluation_metrics.copy()

    # ------------------------------------------------------------------
    # Private helper methods
    # ------------------------------------------------------------------

    def _load_training_data(
        self, source_path: Path
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load preferred training data, using split files when present."""
        train_x_file = source_path / _X_TRAIN_FILE
        train_y_file = source_path / _Y_TRAIN_FILE

        if train_x_file.exists() and train_y_file.exists():
            self._training_data_files = {
                "X": _X_TRAIN_FILE,
                "y": _Y_TRAIN_FILE,
            }
            logger.info(
                "Detected processed train split files; using %s and %s",
                train_x_file.name,
                train_y_file.name,
            )
            return self._load_data_files(train_x_file, train_y_file)

        self._training_data_files = {
            "X": _X_FILE,
            "y": _Y_FILE,
        }
        logger.info(
            "Train split files not found; falling back to %s and %s",
            _X_FILE,
            _Y_FILE,
        )
        return self._load_data(source_path)

    def _load_test_data(
        self, source_path: Path
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Load test split data when available, otherwise return None."""
        test_x_file = source_path / _X_TEST_FILE
        test_y_file = source_path / _Y_TEST_FILE

        if not test_x_file.exists() or not test_y_file.exists():
            return None

        logger.info(
            "Detected processed test split files; using %s and %s",
            test_x_file.name,
            test_y_file.name,
        )
        return self._load_data_files(test_x_file, test_y_file)

    def _evaluate_on_available_test_data(
        self, source_path: Path
    ) -> dict[str, Any]:
        """Evaluate on test split if available, otherwise return skipped payload."""
        if not self._is_trained or self._classifier is None:
            return {
                "evaluated": False,
                "reason": "model not trained",
            }

        test_data = self._load_test_data(source_path)
        if test_data is None:
            logger.info("Test split files not found; skipping evaluation")
            return {
                "evaluated": False,
                "reason": "test split files not found",
                "expected_files": [_X_TEST_FILE, _Y_TEST_FILE],
                "training_data_files": self._training_data_files.copy(),
            }

        X_test, y_test = test_data
        y_score = self._classifier.predict_proba(X_test)[:, 1]
        y_pred = (y_score >= 0.5).astype(np.int32)
        self._curve_points = self._build_curve_points(y_test, y_score)

        metrics: dict[str, float | None] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(
                precision_score(y_test, y_pred, zero_division=0)
            ),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": None,
            "pr_auc": None,
        }

        try:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_score))
        except ValueError:
            logger.warning(
                "ROC-AUC is undefined for the current test-set label distribution"
            )

        try:
            metrics["pr_auc"] = float(average_precision_score(y_test, y_score))
        except ValueError:
            logger.warning(
                "PR-AUC is undefined for the current test-set label distribution"
            )

        matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = matrix.ravel()

        payload: dict[str, Any] = {
            "evaluated": True,
            "model": "xgboost",
            "threshold": 0.5,
            "test_set": {
                "n_samples": int(len(y_test)),
                "positive_rate": float(y_test.mean()),
                "source_files": {
                    "X": _X_TEST_FILE,
                    "y": _Y_TEST_FILE,
                },
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
        logger.info("Test-set evaluation complete: %s", payload["metrics"])
        return payload

    def _build_curve_points(
        self, y_true: np.ndarray, y_score: np.ndarray
    ) -> dict[str, Any]:
        """Build serializable ROC/PR curve points."""

        def _safe_float_list(values: Any) -> list[float | None]:
            output: list[float | None] = []
            for value in values:
                number = float(value)
                if np.isfinite(number):
                    output.append(number)
                else:
                    output.append(None)
            return output

        points: dict[str, Any] = {}

        try:
            fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
            points["roc"] = {
                "fpr": _safe_float_list(fpr),
                "tpr": _safe_float_list(tpr),
                "thresholds": _safe_float_list(roc_thresholds),
            }
        except ValueError as error:
            points["roc"] = {
                "error": str(error),
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
        except ValueError as error:
            points["pr"] = {
                "error": str(error),
                "precision": [],
                "recall": [],
                "thresholds": [],
            }

        return points

    def _render_curves(
        self, curve_points: dict[str, Any], output_path: Path
    ) -> dict[str, Any]:
        """Render ROC/PR curves when plotting dependency is available."""
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
            else:
                rendering_status["roc"] = {
                    "rendered": False,
                    "skipped_reason": "ROC curve points unavailable",
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
            else:
                rendering_status["pr"] = {
                    "rendered": False,
                    "skipped_reason": "PR curve points unavailable",
                }

        return rendering_status

    def _load_data(self, source_path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Load feature matrix and label vector.

        Args:
            source_path: Path to the data directory.

        Returns:
            Tuple (X, y), both as numpy arrays.

        Raises:
            FileNotFoundError: If X.csv or y.csv does not exist.
            ValueError: If sample counts mismatch or data is empty.
        """
        x_file = source_path / _X_FILE
        y_file = source_path / _Y_FILE

        return self._load_data_files(x_file, y_file)

    def _load_data_files(
        self, x_file: Path, y_file: Path
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load feature matrix and label vector from explicit file paths."""

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
                f"Sample count mismatch: X has {len(X_df)} rows, y has {len(y_df)} rows."
            )

        # Use the first column as the label
        y_col = y_df.columns[0]
        X = X_df.to_numpy(dtype=np.float32)
        y = y_df[y_col].to_numpy(dtype=np.int32)

        return X, y

    def _load_feature_names(self, source_path: Path) -> list[str]:
        """Load the list of feature names.

        Args:
            source_path: Path to the data directory.

        Returns:
            List of feature names.
        """
        feature_file = source_path / _FEATURE_NAMES_FILE
        if not feature_file.exists():
            logger.warning(
                f"Feature names file not found: {feature_file}, using default names."
            )
            return []

        names = [
            line.strip()
            for line in feature_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return names

    def _load_metadata(self, source_path: Path) -> dict[str, Any]:
        """Load dataset metadata.

        Args:
            source_path: Path to the data directory.

        Returns:
            Metadata dictionary, or empty dict if file does not exist.
        """
        meta_file = source_path / _METADATA_FILE
        if not meta_file.exists():
            logger.warning(
                f"Metadata file not found: {meta_file}, using empty dict."
            )
            return {}

        return json.loads(meta_file.read_text(encoding="utf-8"))
