"""Output writer for base table exports."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def write_outputs(
    target_path: Path,
    base_table: pd.DataFrame,
    schema: dict[str, Any],
    quality: dict[str, Any],
    export_config: dict[str, Any],
) -> None:
    """Persist all export artefacts to *target_path*.

    Files written:
        base_table.csv
        base_table_schema.json
        base_table_metadata.json
        base_table_quality.json
    """
    target_path.mkdir(parents=True, exist_ok=True)

    # 1. CSV
    csv_path = target_path / "base_table.csv"
    base_table.to_csv(csv_path, index=False)
    logger.info("Wrote %s (%s rows)", csv_path, len(base_table))

    # 2. Schema
    _write_json(target_path / "base_table_schema.json", schema)

    # 3. Metadata
    mortality_rate = None
    if "hospital_expire_flag" in base_table.columns:
        mortality_rate = round(
            float(base_table["hospital_expire_flag"].mean()), 4
        )
    metadata: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_samples": len(base_table),
        "total_columns": len(base_table.columns),
        "hospital_mortality_rate": mortality_rate,
        "export_config": export_config,
    }
    _write_json(target_path / "base_table_metadata.json", metadata)

    # 4. Quality
    _write_json(target_path / "base_table_quality.json", quality)


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively replace float NaN/Inf with None for valid JSON."""
    if isinstance(obj, float) and (
        obj != obj or obj == float("inf") or obj == float("-inf")
    ):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _write_json(path: Path, data: dict[str, Any]) -> None:
    sanitized = _sanitize_for_json(data)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            sanitized,
            fh,
            indent=2,
            ensure_ascii=False,
            default=str,
            allow_nan=False,
        )
    logger.info("Wrote %s", path)
