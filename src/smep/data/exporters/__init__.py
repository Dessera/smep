"""Data exporters module."""

from typing import Any
import logging

from .base import DataExporter
from .mimic import MIMIC3Exporter

logger = logging.getLogger(__name__)


class DataExporterRegistry:
    """Registry for data exporters."""

    def __init__(self) -> None:
        self._registry: dict[str, dict[str, Any]] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        self.register(
            name="mimic3",
            description=(
                "MIMIC-III base table exporter "
                "(raw CSV → extractedMimic-style wide table)"
            ),
            exporter_class=MIMIC3Exporter,
        )

    def register(
        self,
        name: str,
        description: str,
        exporter_class: type[DataExporter],
    ) -> None:
        if name in self._registry:
            raise ValueError(f"Exporter '{name}' is already registered")
        self._registry[name] = {
            "name": name,
            "description": description,
            "exporter_class": exporter_class,
        }

    def get_exporter(self, name: str, **kwargs: Any) -> DataExporter:
        if name not in self._registry:
            available = ", ".join(self._registry)
            raise KeyError(
                f"Exporter '{name}' not found. Available: {available}"
            )
        cls = self._registry[name]["exporter_class"]
        return cls(**kwargs)

    def get_all_exporter_info(self) -> list[dict[str, str]]:
        return [
            {
                "name": e["name"],
                "description": e["description"],
            }
            for e in self._registry.values()
        ]


_global_exporter_registry = DataExporterRegistry()


def get_exporter_registry() -> DataExporterRegistry:
    return _global_exporter_registry


__all__ = [
    "DataExporter",
    "DataExporterRegistry",
    "MIMIC3Exporter",
    "get_exporter_registry",
]
