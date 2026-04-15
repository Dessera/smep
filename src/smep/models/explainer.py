"""Unified SHAP explainability module.

Provides common SHAP value normalization, output writing (plots, CSV, JSON),
and the ShapResult container. Model implementations only compute raw SHAP
values; this module handles all file I/O and visualization.
"""

import importlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Output file name constants
_EXPLAIN_METADATA_FILE = "explain_metadata.json"
_SHAP_SUMMARY_BAR_FILE = "shap_summary_bar.png"
_SHAP_SUMMARY_BEESWARM_FILE = "shap_summary_beeswarm.png"
_SHAP_VALUES_SAMPLE_FILE = "shap_values_sample.csv"
_SHAP_EXPECTED_VALUE_FILE = "shap_expected_value.json"
_TOP_FEATURES_FILE = "top_features.json"


@dataclass
class ShapResult:
    """Container for SHAP computation results."""

    shap_values: np.ndarray  # shape (n_samples, n_features)
    expected_value: float | None  # baseline prediction value
    explainer_type: str  # "TreeExplainer" / "GradientExplainer"
    shap_version: str  # shap.__version__


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def normalize_shap_values(
    raw_values: Any,
    n_samples: int,
    n_features: int,
) -> np.ndarray:
    """Normalize raw SHAP output into a (n_samples, n_features) array.

    Handles list-wrapped values (multi-class), 3-D arrays, and transposed
    matrices that different SHAP explainer versions may produce.
    """
    values = raw_values
    if isinstance(values, list):
        if not values:
            raise RuntimeError("SHAP returned empty value list")
        values = values[-1] if len(values) > 1 else values[0]

    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 3:
        array = array[:, :, -1]
    if array.ndim != 2:
        raise RuntimeError(f"Unexpected SHAP value shape: {array.shape}")

    if array.shape[0] != n_samples and array.shape[1] == n_samples:
        array = array.T

    if array.shape != (n_samples, n_features):
        raise RuntimeError(
            f"Unexpected normalized SHAP matrix shape: "
            f"{array.shape}, expected {(n_samples, n_features)}"
        )
    return array


def normalize_expected_value(value: Any) -> float | None:
    """Extract a scalar expected value from various SHAP explainer outputs."""
    if value is None:
        return None
    # Handle torch.Tensor without importing torch at module level
    if hasattr(value, "item") and callable(value.item):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return None
        value = value[-1]
    if hasattr(value, "item") and callable(value.item):
        try:
            value = value.item()
        except Exception:
            pass
    number = float(value)
    return number if math.isfinite(number) else None


def write_explain_outputs(
    output_path: Path,
    shap_result: ShapResult,
    X_sample: pd.DataFrame,
) -> dict[str, str]:
    """Write all SHAP visualization and data artifacts.

    Args:
        output_path: Directory to write output files.
        shap_result: Computed SHAP values and metadata.
        X_sample: Feature DataFrame with column names used for plots.

    Returns:
        Mapping of artifact keys to file names.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        shap_module = importlib.import_module("shap")
    except Exception as error:
        raise RuntimeError(f"shap unavailable: {error}") from error

    try:
        matplotlib_module = importlib.import_module("matplotlib")
        matplotlib_module.use("Agg")  # type: ignore[attr-defined]
        plt = importlib.import_module("matplotlib.pyplot")
    except Exception as error:
        raise RuntimeError(f"matplotlib unavailable: {error}") from error

    shap_values = shap_result.shap_values

    # 1. SHAP bar summary plot
    shap_module.summary_plot(  # type: ignore[attr-defined]
        shap_values, X_sample, plot_type="bar", show=False, max_display=20
    )
    bar_plot_file = output_path / _SHAP_SUMMARY_BAR_FILE
    bar_fig = plt.gcf()  # type: ignore[attr-defined]
    bar_fig.tight_layout()
    bar_fig.savefig(str(bar_plot_file), dpi=150)
    plt.close(bar_fig)  # type: ignore[attr-defined]

    # 2. SHAP beeswarm plot
    shap_module.summary_plot(  # type: ignore[attr-defined]
        shap_values, X_sample, show=False, max_display=20
    )
    beeswarm_file = output_path / _SHAP_SUMMARY_BEESWARM_FILE
    beeswarm_fig = plt.gcf()  # type: ignore[attr-defined]
    beeswarm_fig.tight_layout()
    beeswarm_fig.savefig(str(beeswarm_file), dpi=150)
    plt.close(beeswarm_fig)  # type: ignore[attr-defined]

    # 3. SHAP values sample CSV
    shap_df = pd.DataFrame(shap_values, columns=X_sample.columns)
    shap_sample_file = output_path / _SHAP_VALUES_SAMPLE_FILE
    shap_df.to_csv(shap_sample_file, index=False)

    # 4. Top features JSON
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    top_features: list[dict[str, float | str]] = [
        {
            "feature": str(X_sample.columns[int(idx)]),
            "mean_abs_shap": float(mean_abs[int(idx)]),
        }
        for idx in order[: min(20, len(order))]
    ]
    top_features_file = output_path / _TOP_FEATURES_FILE
    top_features_file.write_text(
        json.dumps(top_features, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 5. Expected value JSON
    expected_value_payload = {
        "expected_value": shap_result.expected_value,
        "n_samples": int(len(X_sample)),
        "n_features": int(X_sample.shape[1]),
        "shap_version": shap_result.shap_version,
        "explainer": shap_result.explainer_type,
    }
    expected_value_file = output_path / _SHAP_EXPECTED_VALUE_FILE
    expected_value_file.write_text(
        json.dumps(
            to_json_compatible(expected_value_payload),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    return {
        "summary_bar": _SHAP_SUMMARY_BAR_FILE,
        "summary_beeswarm": _SHAP_SUMMARY_BEESWARM_FILE,
        "shap_values_sample": _SHAP_VALUES_SAMPLE_FILE,
        "expected_value": _SHAP_EXPECTED_VALUE_FILE,
        "top_features": _TOP_FEATURES_FILE,
    }


def to_json_compatible(value: Any) -> Any:
    """Recursively convert numpy/torch scalars and non-finite floats for JSON."""
    if isinstance(value, dict):
        return {str(key): to_json_compatible(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_json_compatible(item) for item in value]
    if isinstance(value, np.generic):
        return to_json_compatible(value.item())
    if hasattr(value, "item") and callable(value.item):
        try:
            return to_json_compatible(value.item())
        except Exception:
            pass
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value
