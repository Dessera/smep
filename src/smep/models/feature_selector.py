"""Feature importance evaluation module.

Provides permutation importance analysis to identify features that
genuinely contribute to model predictions vs. noise features.
"""

import importlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.inspection import permutation_importance

logger = logging.getLogger(__name__)

# Output file name constants
_IMPORTANCE_REPORT_FILE = "feature_importance_report.json"
_IMPORTANCE_PLOT_FILE = "feature_importance.png"


@dataclass
class FeatureImportanceResult:
    """Result of feature importance evaluation."""

    feature: str
    importance_mean: float
    importance_std: float
    rank: int


@dataclass
class FeatureImportanceReport:
    """Full report of feature importance analysis."""

    method: str
    scoring: str
    n_repeats: int
    n_samples: int
    n_features: int
    results: list[FeatureImportanceResult]
    noise_features: list[str]
    signal_features: list[str]


class _ModelScoringWrapper(ClassifierMixin, BaseEstimator):
    """Wraps a smep Model into a sklearn-compatible estimator for
    permutation_importance, which requires predict_proba(X) → (n, 2)."""

    _estimator_type = "classifier"

    def __init__(self, model: Any = None) -> None:
        self.model = model
        self.classes_ = np.array([0, 1])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_ModelScoringWrapper":
        """No-op: model is already trained."""
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p1 = self.model.predict_proba(X)
        return np.column_stack([1 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        p1 = self.model.predict_proba(X)
        return (p1 >= 0.5).astype(int)


def evaluate_feature_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    scoring: str = "roc_auc",
    n_repeats: int = 10,
    random_state: int = 42,
    n_jobs: int = -1,
) -> FeatureImportanceReport:
    """Evaluate feature importance using permutation importance.

    Args:
        model: A trained smep Model instance with predict_proba().
        X: Feature matrix (n_samples, n_features).
        y: True labels.
        feature_names: List of feature names matching X columns.
        scoring: Scoring metric for sklearn (e.g. 'roc_auc', 'f1', 'accuracy').
        n_repeats: Number of permutation repeats per feature.
        random_state: Random seed for reproducibility.
        n_jobs: Number of parallel jobs (-1 for all cores).

    Returns:
        FeatureImportanceReport with ranked features and noise/signal lists.
    """
    wrapper = _ModelScoringWrapper(model)

    logger.info(
        "Computing permutation importance: %d samples, %d features, "
        "scoring=%s, n_repeats=%d",
        X.shape[0],
        X.shape[1],
        scoring,
        n_repeats,
    )

    result = permutation_importance(
        wrapper,
        X,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    importances_mean = result.importances_mean
    importances_std = result.importances_std

    # Rank features by mean importance (descending)
    order = np.argsort(importances_mean)[::-1]

    results: list[FeatureImportanceResult] = []
    noise_features: list[str] = []
    signal_features: list[str] = []

    for rank, idx in enumerate(order, start=1):
        idx = int(idx)
        name = (
            feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        )
        mean_val = float(importances_mean[idx])
        std_val = float(importances_std[idx])

        results.append(
            FeatureImportanceResult(
                feature=name,
                importance_mean=mean_val,
                importance_std=std_val,
                rank=rank,
            )
        )

        if mean_val <= 0:
            noise_features.append(name)
        else:
            signal_features.append(name)

    logger.info(
        "Feature importance complete: %d signal, %d noise",
        len(signal_features),
        len(noise_features),
    )

    return FeatureImportanceReport(
        method="permutation",
        scoring=scoring,
        n_repeats=n_repeats,
        n_samples=X.shape[0],
        n_features=X.shape[1],
        results=results,
        noise_features=noise_features,
        signal_features=signal_features,
    )


def render_feature_importance(
    report: FeatureImportanceReport,
    output_path: Path,
    max_display: int = 30,
) -> str:
    """Render a horizontal bar chart of feature importances.

    Args:
        report: The feature importance report.
        output_path: Directory to write the plot file.
        max_display: Maximum number of features to show.

    Returns:
        File name of the rendered plot.
    """
    try:
        matplotlib_module = importlib.import_module("matplotlib")
        matplotlib_module.use("Agg")  # type: ignore[attr-defined]
        plt = importlib.import_module("matplotlib.pyplot")
    except Exception as error:
        raise RuntimeError(f"matplotlib unavailable: {error}") from error

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Take top features by absolute importance
    display_results = report.results[:max_display]
    display_results = list(reversed(display_results))  # lowest at top for barh

    names = [r.feature for r in display_results]
    means = [r.importance_mean for r in display_results]
    stds = [r.importance_std for r in display_results]
    colors = ["#2ecc71" if m > 0 else "#e74c3c" for m in means]

    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.3)))  # type: ignore[attr-defined]
    ax.barh(names, means, xerr=stds, color=colors, edgecolor="none", alpha=0.85)
    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel(f"Permutation Importance ({report.scoring})")
    ax.set_title(f"Feature Importance (top {len(display_results)})")

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#2ecc71", label="Signal (positive)"),
        Patch(facecolor="#e74c3c", label="Noise (≤ 0)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    fig.tight_layout()
    plot_file = output_path / _IMPORTANCE_PLOT_FILE
    fig.savefig(str(plot_file), dpi=150)
    plt.close(fig)  # type: ignore[attr-defined]

    return _IMPORTANCE_PLOT_FILE


def write_feature_importance_outputs(
    output_path: Path,
    report: FeatureImportanceReport,
) -> dict[str, str]:
    """Write feature importance report JSON and visualization.

    Args:
        output_path: Directory to write outputs.
        report: The feature importance report.

    Returns:
        Mapping of artifact keys to file names.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Write report JSON
    report_data = {
        "method": report.method,
        "scoring": report.scoring,
        "n_repeats": report.n_repeats,
        "n_samples": report.n_samples,
        "n_features": report.n_features,
        "n_signal_features": len(report.signal_features),
        "n_noise_features": len(report.noise_features),
        "noise_features": report.noise_features,
        "signal_features": report.signal_features,
        "features": [asdict(r) for r in report.results],
    }
    report_file = output_path / _IMPORTANCE_REPORT_FILE
    report_file.write_text(
        json.dumps(report_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Render plot
    plot_file = render_feature_importance(report, output_path)

    return {
        "report": _IMPORTANCE_REPORT_FILE,
        "plot": plot_file,
    }
