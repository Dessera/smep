"""Abstract base class for models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TYPE_CHECKING
import logging

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .explainer import ShapResult

logger = logging.getLogger(__name__)


class Model(ABC):
    """Abstract base class for models.

    Subclasses implement fit / predict_proba / save / load.
    Data loading and evaluation are handled externally by data_loader
    and evaluator modules respectively.
    """

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        tuning: dict[str, Any] | None = None,
    ) -> None:
        """Train the model on provided numpy arrays.

        Args:
            X_train: Training feature matrix.
            y_train: Training labels.
            X_val: Optional validation feature matrix.
            y_val: Optional validation labels.
            tuning: Optional hyperparameter tuning configuration.
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return positive-class probabilities.

        Args:
            X: Feature matrix.

        Returns:
            1-D numpy array of predicted probabilities.
        """
        pass

    @abstractmethod
    def save(self, output_path: Path) -> dict[str, Any]:
        """Save model weights to output_path.

        Args:
            output_path: Destination directory for model files.

        Returns:
            Model-specific metadata dictionary.
        """
        pass

    @abstractmethod
    def load(self, weight_path: Path) -> None:
        """Load model weights from weight_path directory.

        Args:
            weight_path: Path to the directory produced by save().

        Raises:
            FileNotFoundError: If required weight files are missing.
            RuntimeError: If loading fails.
        """
        pass

    def compute_shap(
        self,
        X: pd.DataFrame,
        max_samples: int = 500,
    ) -> "ShapResult":
        """Compute SHAP values for the given feature data.

        This is an optional capability. Model implementations that support
        explainability should override this method.

        Args:
            X: Feature DataFrame with column names (already sampled by CLI).
            max_samples: Maximum samples for background/reference data.

        Returns:
            ShapResult containing shap_values and expected_value.

        Raises:
            RuntimeError: If not implemented by the model.
        """
        raise RuntimeError(
            f"Model '{self.__class__.__name__}' does not support compute_shap()."
        )
