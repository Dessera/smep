"""Feature importance evaluation module.

Provides permutation importance and LASSO-based analysis to identify
features that genuinely contribute to model predictions vs. noise features.
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
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

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
    confidence_level: float
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
    n_repeats: int = 30,
    random_state: int = 42,
    n_jobs: int = -1,
    confidence_level: float = 0.8,
) -> FeatureImportanceReport:
    """Evaluate feature importance using permutation importance.

    Args:
        model: A trained smep Model instance with predict_proba().
        X: Feature matrix (n_samples, n_features).
        y: True labels.
        feature_names: List of feature names matching X columns.
        scoring: Scoring metric for sklearn (e.g. 'roc_auc', 'f1', 'accuracy').
        n_repeats: Number of permutation repeats per feature.
            Higher values improve bootstrap confidence interval stability;
            at least 30 is recommended for reliable results.
        random_state: Random seed for reproducibility.
        n_jobs: Number of parallel jobs (-1 for all cores).
        confidence_level: Confidence level for the bootstrap lower bound.
            A feature is classified as noise when the lower bound of its
            permutation importance distribution is <= 0.
            E.g. 0.95 uses the 5th percentile as the lower bound.

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
    # result.importances has shape (n_features, n_repeats)
    lower_percentile = (1.0 - confidence_level) * 100.0

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
        # Bootstrap lower bound: if the lower tail of the empirical
        # distribution is still positive, the feature is a signal.
        lower_bound = float(
            np.percentile(result.importances[idx], lower_percentile)
        )

        results.append(
            FeatureImportanceResult(
                feature=name,
                importance_mean=mean_val,
                importance_std=std_val,
                rank=rank,
            )
        )

        if lower_bound <= 0:
            noise_features.append(name)
        else:
            signal_features.append(name)

    logger.info(
        "Feature importance complete: %d signal, %d noise "
        "(confidence_level=%.2f, lower_percentile=%.1f%%)",
        len(signal_features),
        len(noise_features),
        confidence_level,
        lower_percentile,
    )

    return FeatureImportanceReport(
        method="permutation",
        scoring=scoring,
        n_repeats=n_repeats,
        n_samples=X.shape[0],
        n_features=X.shape[1],
        confidence_level=confidence_level,
        results=results,
        noise_features=noise_features,
        signal_features=signal_features,
    )


def evaluate_lasso_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    cv: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
    max_iter: int = 10000,
) -> FeatureImportanceReport:
    """Evaluate feature importance using LASSO (L1-regularised linear model).

    Features whose coefficients are driven to zero by the L1 penalty are
    classified as noise; remaining features are signal.  The regularisation
    strength (alpha) is chosen automatically via cross-validation.

    Args:
        X: Feature matrix (n_samples, n_features).
        y: True labels (binary 0/1).
        feature_names: List of feature names matching X columns.
        cv: Number of cross-validation folds for LassoCV.
        random_state: Random seed for reproducibility.
        n_jobs: Number of parallel jobs (-1 for all cores).
        max_iter: Maximum iterations for the LASSO solver.

    Returns:
        FeatureImportanceReport with ranked features and noise/signal lists.
    """
    logger.info(
        "Computing LASSO importance: %d samples, %d features, cv=%d",
        X.shape[0],
        X.shape[1],
        cv,
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lasso = LassoCV(
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        max_iter=max_iter,
    )
    lasso.fit(X_scaled, y)

    coefs = lasso.coef_
    abs_coefs = np.abs(coefs)

    logger.info(
        "LASSO fit complete: alpha=%.6f, non-zero coefficients=%d/%d",
        lasso.alpha_,
        int(np.sum(abs_coefs > 0)),
        len(coefs),
    )

    # Rank by absolute coefficient (descending)
    order = np.argsort(abs_coefs)[::-1]

    results: list[FeatureImportanceResult] = []
    noise_features: list[str] = []
    signal_features: list[str] = []

    for rank, idx in enumerate(order, start=1):
        idx = int(idx)
        name = (
            feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        )
        coef_val = float(coefs[idx])
        abs_val = float(abs_coefs[idx])

        results.append(
            FeatureImportanceResult(
                feature=name,
                importance_mean=coef_val,
                importance_std=0.0,  # LASSO gives point estimates
                rank=rank,
            )
        )

        if abs_val == 0.0:
            noise_features.append(name)
        else:
            signal_features.append(name)

    logger.info(
        "LASSO importance complete: %d signal, %d noise (alpha=%.6f)",
        len(signal_features),
        len(noise_features),
        lasso.alpha_,
    )

    return FeatureImportanceReport(
        method="lasso",
        scoring=f"lasso_cv (alpha={lasso.alpha_:.6f})",
        n_repeats=cv,
        n_samples=X.shape[0],
        n_features=X.shape[1],
        confidence_level=0.0,
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

    if report.method == "lasso":
        ax.set_xlabel("LASSO Coefficient")
    else:
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
        "confidence_level": report.confidence_level,
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
