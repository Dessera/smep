"""Data processors module."""

from typing import Any, Dict, List, Type
import logging

from .processor import DataProcessor
from .mimic import MIMIC3Processor

logger = logging.getLogger(__name__)


class DataProcessorRegistry:
    """Registry for managing data processors.

    This class maintains a registry of available data processors with their
    metadata and provides methods to access processor instances.
    """

    def __init__(self):
        """Initialize the registry with default processors."""
        self._registry: Dict[str, Dict[str, Any]] = {}
        self._register_builtin_processors()

    def _register_builtin_processors(self) -> None:
        """Register the built-in data processors."""
        self.register(
            name="mimic3",
            description="MIMIC-III Feature Engineering Pipeline",
            processor_class=MIMIC3Processor,
        )

    def register(
        self, name: str, description: str, processor_class: Type[DataProcessor]
    ) -> None:
        """Register a new data processor.

        Args:
            name: Identifier for the processor (used in CLI).
            description: Human-readable description.
            processor_class: The DataProcessor subclass to register.

        Raises:
            ValueError: If the name is already registered.
        """
        if name in self._registry:
            raise ValueError(f"Processor '{name}' is already registered")

        self._registry[name] = {
            "name": name,
            "description": description,
            "processor_class": processor_class,
        }
        logger.debug(f"Registered processor: {name}")

    def get_processor(self, name: str) -> DataProcessor:
        """Get a processor instance by name.

        Args:
            name: The name of the processor to retrieve.

        Returns:
            An instance of the requested DataProcessor.

        Raises:
            KeyError: If the processor name is not found.
        """
        if name not in self._registry:
            available = ", ".join(self.list_processors())
            raise KeyError(
                f"Processor '{name}' not found. Available processors: {available}"
            )

        processor_class = self._registry[name]["processor_class"]
        return processor_class()

    def list_processors(self) -> List[str]:
        """List all available processor names.

        Returns:
            A list of available processor names.
        """
        return list(self._registry.keys())

    def get_processor_info(self, name: str) -> Dict[str, str]:
        """Get information about a specific processor.

        Args:
            name: The name of the processor.

        Returns:
            A dictionary containing name and description.

        Raises:
            KeyError: If the processor name is not found.
        """
        if name not in self._registry:
            raise KeyError(f"Processor '{name}' not found")

        entry = self._registry[name]
        return {"name": entry["name"], "description": entry["description"]}

    def get_all_processor_info(self) -> List[Dict[str, str]]:
        """Get information about all available processors.

        Returns:
            A list of dictionaries containing processor metadata.
        """
        return [
            {"name": entry["name"], "description": entry["description"]}
            for entry in self._registry.values()
        ]


# Global processor registry instance
_global_processor_registry = DataProcessorRegistry()


def get_processor_registry() -> DataProcessorRegistry:
    """Get the global data processor registry.

    Returns:
        The global DataProcessorRegistry instance.
    """
    return _global_processor_registry


__all__ = [
    "DataProcessor",
    "MIMIC3Processor",
    "DataProcessorRegistry",
    "get_processor_registry",
]
