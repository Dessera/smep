"""Data exporter base class."""

from abc import ABC, abstractmethod
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DataExporter(ABC):
    """Abstract base class for data exporters.

    Exporters produce a base table (initial wide table) from raw data
    sources. They handle cohort selection, feature extraction, and
    quality reporting but do NOT perform train/test splitting,
    imputation, or standardization.
    """

    @abstractmethod
    def export(self, source_path: Path, target_path: Path) -> None:
        """Export raw data into a base table.

        Args:
            source_path: Path to raw data directory.
            target_path: Path to write output files.
        """
        ...
