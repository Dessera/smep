"""Base class for data fetchers."""

from abc import ABC, abstractmethod
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DataFetcher(ABC):
    """Abstract base class for data fetchers.

    Subclasses should implement the fetch() method to provide
    specific data source fetching logic.
    """

    @abstractmethod
    def fetch(self, output_path: Path) -> None:
        """Fetch data from the source and save to the specified path.

        Args:
            output_path: Destination path where data will be saved.

        Raises:
            Exception: If fetching fails for any reason.
        """
        pass
