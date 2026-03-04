"""Data processing base class."""

from abc import ABC, abstractmethod
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DataProcessor(ABC):
    """Abstract base class for data processors.

    Subclasses should implement the process() method to provide
    specific data processing logic for different datasets.
    """

    @abstractmethod
    def process(self, source_path: Path, target_path: Path) -> None:
        """Process data from source and save to target path.

        Args:
            source_path: Path to source data directory.
            target_path: Path to save processed data.

        Raises:
            FileNotFoundError: If required source files are missing.
            ValueError: If data format is invalid.
            Exception: If processing fails for any reason.
        """
        pass

    def _ensure_target_directory(self, target_path: Path) -> None:
        """Ensure the target directory exists.

        Args:
            target_path: Path to verify/create.

        Raises:
            PermissionError: If the directory cannot be created.
        """
        try:
            target_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Target directory ready: {target_path}")
        except PermissionError:
            logger.error(
                f"Permission denied when creating target directory: {target_path}"
            )
            raise
