"""Dataset builders module."""

from typing import Any
import logging

from .builder import DatasetBuilder
from .default import DefaultDatasetBuilder

logger = logging.getLogger(__name__)


class DatasetBuilderRegistry:
    """Registry for dataset builders."""

    def __init__(self) -> None:
        self._registry: dict[str, dict[str, Any]] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        self.register(
            name="default",
            description="Default dataset builder (split + impute + encode + scale)",
            builder_class=DefaultDatasetBuilder,
        )

    def register(
        self,
        name: str,
        description: str,
        builder_class: type[DatasetBuilder],
    ) -> None:
        if name in self._registry:
            raise ValueError(f"Builder '{name}' is already registered")
        self._registry[name] = {
            "name": name,
            "description": description,
            "builder_class": builder_class,
        }

    def get_builder(self, name: str, **kwargs: Any) -> DatasetBuilder:
        if name not in self._registry:
            available = ", ".join(self._registry)
            raise KeyError(
                f"Builder '{name}' not found. Available: {available}"
            )
        cls = self._registry[name]["builder_class"]
        return cls(**kwargs)

    def get_all_builder_info(self) -> list[dict[str, str]]:
        return [
            {
                "name": b["name"],
                "description": b["description"],
            }
            for b in self._registry.values()
        ]


_global_builder_registry = DatasetBuilderRegistry()


def get_builder_registry() -> DatasetBuilderRegistry:
    return _global_builder_registry


__all__ = [
    "DatasetBuilder",
    "DatasetBuilderRegistry",
    "DefaultDatasetBuilder",
    "get_builder_registry",
]
