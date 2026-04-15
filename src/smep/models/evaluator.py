"""Unified evaluation module for binary classification models.

Provides metrics computation, curve point extraction, curve rendering,
and file output for train/val/test splits.
"""

import importlib
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

_ROC_PLOT_FILE = "roc_curve.png"
_PR_PLOT_FILE = "pr_curve.png"


@dataclass
class EvaluationResult:
    """Evaluation metrics for a single data split."""

    split: str
    n_samples: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    pr_auc: float | None
    confusion_matrix: list[list[int]]
    positive_rate: float


def evaluate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    split: str,
) -> EvaluationResult:
    """Compute binary classification evaluation metrics.

    Args:
        y_true: Ground truth labels (0/1).
        y_prob: Predicted positive-class probabilities.
        split: Name of the data split (e.g. "train", "val", "test").

    Returns:
        EvaluationResult with all computed metrics.
    """
    y_pred = (y_prob >= 0.5).astype(np.int32)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1_val = float(f1_score(y_true, y_pred, zero_division=0))

    roc_auc_val: float | None = None
    try:
        roc_auc_val = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        logger.warning(
            "ROC-AUC undefined for %s split label distribution", split
        )

    pr_auc_val: float | None = None
    try:
        pr_auc_val = float(average_precision_score(y_true, y_prob))
    except ValueError:
        logger.warning(
            "PR-AUC undefined for %s split label distribution", split
        )

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_list = matrix.tolist()

    positive_rate = float(y_true.mean())

    return EvaluationResult(
        split=split,
        n_samples=int(len(y_true)),
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1_val,
        roc_auc=roc_auc_val,
        pr_auc=pr_auc_val,
        confusion_matrix=cm_list,
        positive_rate=positive_rate,
    )


def compute_curve_points(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, Any]:
    """Compute ROC and PR curve coordinate points.

    Args:
        y_true: Ground truth labels (0/1).
        y_prob: Predicted positive-class probabilities.

    Returns:
        Dictionary with "roc" and "pr" keys containing curve coordinates.
    """

    def _safe_float_list(values: Any) -> list[float | None]:
        output: list[float | None] = []
        for v in values:
            n = float(v)
            output.append(n if math.isfinite(n) else None)
        return output

    points: dict[str, Any] = {}

    try:
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
        points["roc"] = {
            "fpr": _safe_float_list(fpr),
            "tpr": _safe_float_list(tpr),
            "thresholds": _safe_float_list(roc_thresholds),
        }
    except ValueError as e:
        points["roc"] = {
            "error": str(e),
            "fpr": [],
            "tpr": [],
            "thresholds": [],
        }

    try:
        prec_arr, rec_arr, pr_thresholds = precision_recall_curve(
            y_true, y_prob
        )
        points["pr"] = {
            "precision": _safe_float_list(prec_arr),
            "recall": _safe_float_list(rec_arr),
            "thresholds": _safe_float_list(pr_thresholds),
        }
    except ValueError as e:
        points["pr"] = {
            "error": str(e),
            "precision": [],
            "recall": [],
            "thresholds": [],
        }

    return points


def render_curves(
    curve_points: dict[str, Any],
    output_path: Path,
) -> None:
    """Render ROC and PR curve plots to PNG files.

    Args:
        curve_points: Output of compute_curve_points().
        output_path: Directory to write plot files into.
    """
    try:
        matplotlib_module = importlib.import_module("matplotlib")
        matplotlib = cast(Any, matplotlib_module)
        matplotlib.use("Agg")
        plt = cast(Any, importlib.import_module("matplotlib.pyplot"))
    except Exception as error:
        logger.warning(
            "Skipping curve rendering: matplotlib unavailable: %s", error
        )
        return

    roc_data = curve_points.get("roc", {})
    if isinstance(roc_data, dict):
        fpr = roc_data.get("fpr", [])
        tpr = roc_data.get("tpr", [])
        if fpr and tpr:
            fig = plt.figure(figsize=(6, 6))
            plt.plot(fpr, tpr, linewidth=2, label="ROC")
            plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
            plt.xlim(0.0, 1.0)
            plt.ylim(0.0, 1.05)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.grid(alpha=0.3)
            plt.legend(loc="lower right")
            plt.tight_layout()
            fig.savefig(str(output_path / _ROC_PLOT_FILE), dpi=150)
            plt.close(fig)
            logger.info("ROC curve saved to %s", output_path / _ROC_PLOT_FILE)

    pr_data = curve_points.get("pr", {})
    if isinstance(pr_data, dict):
        precision = pr_data.get("precision", [])
        recall = pr_data.get("recall", [])
        if precision and recall:
            fig = plt.figure(figsize=(6, 6))
            plt.plot(recall, precision, linewidth=2, label="PR")
            plt.xlim(0.0, 1.0)
            plt.ylim(0.0, 1.05)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall Curve")
            plt.grid(alpha=0.3)
            plt.legend(loc="lower left")
            plt.tight_layout()
            fig.savefig(str(output_path / _PR_PLOT_FILE), dpi=150)
            plt.close(fig)
            logger.info("PR curve saved to %s", output_path / _PR_PLOT_FILE)


def _to_json_compatible(value: Any) -> Any:
    """Recursively convert values to strict JSON-compatible types."""
    if isinstance(value, dict):
        return {str(k): _to_json_compatible(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_compatible(item) for item in value]
    if isinstance(value, np.generic):
        return _to_json_compatible(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_evaluation_outputs(
    output_path: Path,
    train_result: EvaluationResult,
    val_result: EvaluationResult,
    test_result: EvaluationResult,
    curve_points: dict[str, Any],
) -> None:
    """Write all evaluation artifacts to the output directory.

    Writes:
        - train_metrics.json
        - val_metrics.json
        - test_metrics.json
        - curve_points.json
        - roc_curve.png (if matplotlib available)
        - pr_curve.png (if matplotlib available)

    Args:
        output_path: Directory to write evaluation files into.
        train_result: Evaluation result for training split.
        val_result: Evaluation result for validation split.
        test_result: Evaluation result for test split.
        curve_points: ROC/PR curve coordinates (from test split).
    """
    output_path.mkdir(parents=True, exist_ok=True)

    for result in [train_result, val_result, test_result]:
        fname = f"{result.split}_metrics.json"
        payload = _to_json_compatible(asdict(result))
        fpath = output_path / fname
        fpath.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        logger.info("Metrics written to %s", fpath)

    # Curve points
    curve_file = output_path / "curve_points.json"
    curve_file.write_text(
        json.dumps(
            _to_json_compatible(curve_points),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    logger.info("Curve points written to %s", curve_file)

    # Render curve plots
    render_curves(curve_points, output_path)
