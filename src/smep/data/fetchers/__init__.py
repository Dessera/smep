"""Data fetchers module."""

from typing import Any, Dict, List, Type
import logging

from .fetcher import DataFetcher
from .kaggle import KaggleDownloadError, KaggleFetcher
from .mimic import MIMIC3DemoFetcher, MIMIC310KFetcher

logger = logging.getLogger(__name__)


class DataFetcherRegistry:
    """Registry for managing available data fetchers.

    This class maintains a registry of available data sources with their
    metadata and provides methods to access fetcher instances.
    """

    def __init__(self):
        """Initialize the registry with default fetchers."""
        self._registry: Dict[str, Dict[str, Any]] = {}
        self._register_builtin_fetchers()

    def _register_builtin_fetchers(self) -> None:
        """Register the built-in data fetchers."""
        self.register(
            name="mimic3-demo",
            description="MIMIC-III Clinical Database Demo",
            fetcher_class=MIMIC3DemoFetcher,
        )
        self.register(
            name="mimic3-10k",
            description="MIMIC-III 10K Subset",
            fetcher_class=MIMIC310KFetcher,
        )

    def register(
        self, name: str, description: str, fetcher_class: Type[DataFetcher]
    ) -> None:
        """Register a new data fetcher.

        Args:
            name: Identifier for the data source (used in CLI).
            description: Human-readable description of the data source.
            fetcher_class: The DataFetcher subclass to register.

        Raises:
            ValueError: If the name is already registered.
        """
        if name in self._registry:
            raise ValueError(f"Fetcher '{name}' is already registered")

        self._registry[name] = {
            "name": name,
            "description": description,
            "fetcher_class": fetcher_class,
        }
        logger.debug(f"Registered fetcher: {name}")

    def get_fetcher(self, name: str) -> DataFetcher:
        """Get a fetcher instance by name.

        Args:
            name: The name of the fetcher to retrieve.

        Returns:
            An instance of the requested DataFetcher.

        Raises:
            KeyError: If the fetcher name is not found.
        """
        if name not in self._registry:
            available = ", ".join(self.list_fetchers())
            raise KeyError(
                f"Fetcher '{name}' not found. Available fetchers: {available}"
            )

        fetcher_class = self._registry[name]["fetcher_class"]
        return fetcher_class()

    def list_fetchers(self) -> List[str]:
        """List all available fetcher names.

        Returns:
            A list of available fetcher names.
        """
        return list(self._registry.keys())

    def get_fetcher_info(self, name: str) -> Dict[str, str]:
        """Get information about a specific fetcher.

        Args:
            name: The name of the fetcher.

        Returns:
            A dictionary containing name and description.

        Raises:
            KeyError: If the fetcher name is not found.
        """
        if name not in self._registry:
            raise KeyError(f"Fetcher '{name}' not found")

        entry = self._registry[name]
        return {"name": entry["name"], "description": entry["description"]}

    def get_all_fetcher_info(self) -> List[Dict[str, str]]:
        """Get information about all available fetchers.

        Returns:
            A list of dictionaries containing fetcher metadata.
        """
        return [
            {"name": entry["name"], "description": entry["description"]}
            for entry in self._registry.values()
        ]


# Global registry instance
_global_registry = DataFetcherRegistry()


def get_registry() -> DataFetcherRegistry:
    """Get the global data fetcher registry.

    Returns:
        The global DataFetcherRegistry instance.
    """
    return _global_registry


__all__ = [
    "DataFetcher",
    "KaggleFetcher",
    "KaggleDownloadError",
    "MIMIC3DemoFetcher",
    "MIMIC310KFetcher",
    "DataFetcherRegistry",
    "get_registry",
]
