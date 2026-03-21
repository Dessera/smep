"""Abstract base class for models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import logging

logger = logging.getLogger(__name__)


class Model(ABC):
    """Abstract base class for models.

    Subclasses should implement the train() and export() methods
    to provide model-specific training and export logic.
    """

    @abstractmethod
    def train(
        self,
        source_path: Path,
        tuning: dict[str, Any] | None = None,
    ) -> None:
        """Load processed data from source_path and train the model.

        Args:
            source_path: Path to the processed data directory.
            tuning: Optional hyperparameter tuning config.

        Raises:
            FileNotFoundError: If required data files do not exist.
            ValueError: If the data format is invalid.
            RuntimeError: If the training process fails.
        """
        pass

    @abstractmethod
    def export(self, output_path: Path) -> None:
        """Export trained weights/model files to output_path.

        Args:
            output_path: Destination path for exported weight files.

        Raises:
            RuntimeError: If the model has not been trained or export fails.
        """
        pass

    @abstractmethod
    def load(self, weight_path: Path) -> None:
        """Load trained weights from weight_path directory.

        Args:
            weight_path: Path to the directory produced by export().

        Raises:
            FileNotFoundError: If required weight files are missing.
            RuntimeError: If loading fails.
        """
        pass

    @abstractmethod
    def infer(self, source_path: Path) -> Any:
        """Run inference on data located at source_path.

        Args:
            source_path: Path to the processed data directory (same format as train).

        Returns:
            Inference results (model-specific, e.g. numpy array of predictions).

        Raises:
            FileNotFoundError: If required data files are missing.
            RuntimeError: If the model has not been loaded.
        """
        pass

    def explain(
        self,
        source_path: Path,
        output_path: Path,
        max_samples: int = 500,
    ) -> dict[str, Any]:
        """Generate explainability artifacts for a trained model.

        This is an optional capability. Model implementations that support
        explainability should override this method.

        Args:
            source_path: Path to training/export output directory.
            output_path: Directory where explainability artifacts are written.
            max_samples: Maximum number of samples used for explanation.

        Returns:
            A dictionary summary of explainability execution.

        Raises:
            RuntimeError: If explainability is not implemented by the model.
        """
        raise RuntimeError(
            f"Model '{self.__class__.__name__}' does not support explain()."
        )
