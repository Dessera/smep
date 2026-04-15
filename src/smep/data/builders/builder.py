"""Dataset builder base class."""

from abc import ABC, abstractmethod
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DatasetBuilder(ABC):
    """Abstract base class for dataset builders.

    Builders take a base table (produced by an exporter) and produce
    train/val/test splits with preprocessing artifacts.
    """

    @abstractmethod
    def build(self, base_table_path: Path, output_path: Path) -> None:
        """Build a training dataset from a base table.

        Args:
            base_table_path: Path to base_table.csv or directory containing it.
            output_path: Path to write output files.
        """
        ...
