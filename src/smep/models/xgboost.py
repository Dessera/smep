"""XGBoost model implementation.

Binary classification task for in-hospital mortality prediction on MIMIC-III sepsis data.
Data format:
  - X.csv: Feature matrix (n_samples x n_features)
  - y.csv: Label column (hospital_expire_flag, 0/1)
  - feature_names.txt: Feature name list (one per line)
  - metadata.json: Dataset metadata
"""

import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from .model import Model

logger = logging.getLogger(__name__)

# Data file name constants
_X_FILE = "X.csv"
_Y_FILE = "y.csv"
_FEATURE_NAMES_FILE = "feature_names.txt"
_METADATA_FILE = "metadata.json"

# Export file name constants
_MODEL_FILE = "xgboost_model.joblib"
_FEATURE_NAMES_OUT_FILE = "feature_names.txt"
_METADATA_OUT_FILE = "metadata.json"


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
        self._metadata: dict = {}
        self._is_trained: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def train(self, source_path: Path) -> None:
        """Load processed data from source_path and train the XGBoost model.

        Training pipeline:
        1. Load X.csv / y.csv / feature_names.txt / metadata.json
        2. Train the model on the full dataset

        Args:
            source_path: Path to the processed data directory.

        Raises:
            FileNotFoundError: If required data files are missing.
            ValueError: If data format is invalid or sample count is insufficient.
            RuntimeError: If the training process fails.
        """
        source_path = Path(source_path)
        logger.info(f"Loading data from {source_path}...")

        X, y = self._load_data(source_path)
        self._feature_names = self._load_feature_names(source_path)
        self._metadata = self._load_metadata(source_path)

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

        # Train model on full dataset
        logger.info("Training model on full dataset...")
        self._classifier.fit(X, y)
        self._is_trained = True
        logger.info("Training complete.")

    def export(self, output_path: Path) -> None:
        """Export the trained model and preprocessors to output_path directory.

        Exported files:
        - xgboost_model.joblib: XGBClassifier weights
        - xgboost_scaler.joblib: StandardScaler parameters
        - feature_names.txt: Feature name list
        - metadata.json: Metadata including original dataset info and training config

        Args:
            output_path: Export directory path (created automatically if not exists).

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if not self._is_trained:
            raise RuntimeError(
                "Model has not been trained. Call train() first."
            )

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

    # ------------------------------------------------------------------
    # Private helper methods
    # ------------------------------------------------------------------

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
        X = X_df.values.astype(np.float32)
        y = y_df[y_col].values.astype(np.int32)

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

    def _load_metadata(self, source_path: Path) -> dict:
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
