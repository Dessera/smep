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
