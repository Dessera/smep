"""XGBoost model implementation.

Binary classification for in-hospital mortality prediction on MIMIC-III sepsis data.
Consumes build artifacts via the data_loader module; evaluation is handled
externally by the evaluator module.
"""

import importlib
import json
import logging
import math
from pathlib import Path
from typing import Any, Optional, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier

from .explainer import (
    ShapResult,
    normalize_shap_values,
    normalize_expected_value,
)
from .model import Model

logger = logging.getLogger(__name__)

# Export file name constants
_MODEL_FILE = "xgboost_model.joblib"
_FEATURE_NAMES_OUT_FILE = "feature_names.txt"


class XGBoostModel(Model):
    """XGBoost-based in-hospital mortality prediction model for sepsis.

    Attributes:
        _classifier: Trained XGBClassifier instance.
        _feature_names: List of feature names.
        _is_trained: Whether the model has been trained.
    """

    def __init__(self) -> None:
        self._classifier: Optional[XGBClassifier] = None
        self._feature_names: list[str] = []
        self._is_trained: bool = False
        self._source_path: Path | None = None
        self._tuning_summary: dict[str, Any] = {
            "enabled": False,
            "strategy": "none",
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        tuning: dict[str, Any] | None = None,
    ) -> None:
        logger.info(
            "Training XGBoost: %d samples, %d features, positive rate %.2f%%",
            X_train.shape[0],
            X_train.shape[1],
            y_train.mean() * 100,
        )

        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        use_early_stopping = X_val is not None and y_val is not None

        base_classifier = XGBClassifier(
            n_estimators=500,
            max_depth=10,
            learning_rate=0.01,
            subsample=0.7,
            min_child_weight=0.3,
            gamma=1,
            reg_alpha=1,
            reg_lambda=1,
            scale_pos_weight=float(scale_pos_weight),
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )

        tuning_config = self._normalize_tuning_config(tuning)
        self._tuning_summary = {
            "enabled": tuning_config["strategy"] != "none",
            "strategy": tuning_config["strategy"],
            "cv": tuning_config["cv"],
            "scoring": tuning_config["scoring"],
            "search_space_source": tuning_config["search_space_source"],
            "best_params": None,
            "best_cv_score": None,
            "n_candidates": None,
        }

        strategy = tuning_config["strategy"]
        fit_params: dict[str, Any] = {"verbose": False}
        if use_early_stopping:
            fit_params["eval_set"] = [(X_val, y_val)]

        if strategy == "none":
            self._classifier = base_classifier
            self._classifier.fit(X_train, y_train, **fit_params)
        elif strategy == "grid":
            # Don't pass eval_set into CV — it leaks validation data into
            # each fold.  Early stopping is only used in strategy=none.
            base_classifier.set_params(
                n_estimators=200, early_stopping_rounds=None
            )
            search = GridSearchCV(
                estimator=base_classifier,
                param_grid=tuning_config["param_grid"],
                cv=tuning_config["cv"],
                scoring=tuning_config["scoring"],
                n_jobs=tuning_config["n_jobs"],
                refit=True,
            )
            search.fit(X_train, y_train)
            self._classifier = cast(XGBClassifier, search.best_estimator_)
            self._tuning_summary["best_params"] = cast(
                dict[str, Any], search.best_params_
            )
            self._tuning_summary["best_cv_score"] = float(search.best_score_)
            self._tuning_summary["n_candidates"] = len(
                cast(dict[str, Any], search.cv_results_)["params"]
            )
        else:
            base_classifier.set_params(
                n_estimators=200, early_stopping_rounds=None
            )
            search = RandomizedSearchCV(
                estimator=base_classifier,
                param_distributions=tuning_config["param_distributions"],
                n_iter=tuning_config["n_iter"],
                cv=tuning_config["cv"],
                scoring=tuning_config["scoring"],
                n_jobs=tuning_config["n_jobs"],
                random_state=tuning_config["random_state"],
                refit=True,
            )
            search.fit(X_train, y_train)
            self._classifier = cast(XGBClassifier, search.best_estimator_)
            self._tuning_summary["best_params"] = cast(
                dict[str, Any], search.best_params_
            )
            self._tuning_summary["best_cv_score"] = float(search.best_score_)
            self._tuning_summary["n_candidates"] = len(
                cast(dict[str, Any], search.cv_results_)["params"]
            )

        self._is_trained = True
        logger.info("Training complete.")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._is_trained or self._classifier is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")
        return self._classifier.predict_proba(X)[:, 1]

    def save(self, output_path: Path) -> dict[str, Any]:
        if not self._is_trained or self._classifier is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        model_file = output_path / _MODEL_FILE
        joblib.dump(self._classifier, model_file)
        logger.info("Model saved to %s", model_file)

        return {
            "model": "xgboost",
            "model_file": _MODEL_FILE,
            "classifier_params": self._to_json_compatible(
                self._classifier.get_params()
            ),
            "tuning_summary": self._tuning_summary.copy(),
        }

    def load(self, weight_path: Path) -> None:
        weight_path = Path(weight_path)
        self._source_path = weight_path
        if not weight_path.exists() or not weight_path.is_dir():
            raise FileNotFoundError(
                f"Weight directory not found: {weight_path}"
            )

        model_file = weight_path / _MODEL_FILE
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        try:
            self._classifier = joblib.load(model_file)
            logger.info("Model loaded from %s", model_file)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}") from e

        feature_file = weight_path / _FEATURE_NAMES_OUT_FILE
        if feature_file.exists():
            self._feature_names = [
                line.strip()
                for line in feature_file.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]

        self._is_trained = True
        logger.info("Model weights loaded successfully.")

    def compute_shap(
        self,
        X: pd.DataFrame,
        max_samples: int = 500,
    ) -> ShapResult:
        if not self._is_trained or self._classifier is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")

        shap_module = importlib.import_module("shap")
        explainer = shap_module.TreeExplainer(self._classifier)  # type: ignore[attr-defined]
        raw_shap_values = explainer.shap_values(X)
        shap_values = normalize_shap_values(
            raw_shap_values,
            n_samples=len(X),
            n_features=X.shape[1],
        )
        expected = normalize_expected_value(explainer.expected_value)

        return ShapResult(
            shap_values=shap_values,
            expected_value=expected,
            explainer_type="TreeExplainer",
            shap_version=str(getattr(shap_module, "__version__", "unknown")),
        )

    def get_tuning_summary(self) -> dict[str, Any]:
        return self._tuning_summary.copy()

    # ------------------------------------------------------------------
    # Private helper methods
    # ------------------------------------------------------------------

    def _normalize_tuning_config(
        self, tuning: dict[str, Any] | None
    ) -> dict[str, Any]:
        default_grid = {
            "max_depth": [3, 5, 7, 10, 20, 40, 60],
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [50, 100, 200, 300, 500, 1000],
            "subsample": [0.1, 0.3, 0.5, 0.7, 0.8, 1.0],
            "min_child_weight": [0.1, 0.3, 0.5, 1, 3, 5],
            # "gamma": [0.0, 0.1, 0.3, 0.5, 1, 3],
            # "reg_alpha": [0.0, 0.1, 0.3, 0.5, 1, 3],
            # "reg_lambda": [0.0, 0.1, 0.3, 0.5, 1, 3],
        }

        config = tuning.copy() if tuning is not None else {}
        strategy = str(config.get("strategy", "none")).strip().lower()
        if strategy not in {"none", "grid", "random"}:
            raise ValueError(
                "Invalid tuning strategy. Expected one of: none, grid, random"
            )

        cv = int(config.get("cv", 5))
        if cv < 2:
            raise ValueError("cv must be >= 2")

        scoring = str(config.get("scoring", "roc_auc"))
        n_jobs = int(config.get("n_jobs", -1))
        n_iter = int(config.get("n_iter", 30))
        random_state = int(config.get("random_state", 42))

        normalized: dict[str, Any] = {
            "strategy": strategy,
            "cv": cv,
            "scoring": scoring,
            "n_jobs": n_jobs,
            "n_iter": n_iter,
            "random_state": random_state,
            "search_space_source": "none",
            "param_grid": None,
            "param_distributions": None,
        }

        if strategy == "grid":
            param_grid = config.get("param_grid")
            if param_grid is None:
                param_grid = default_grid
                normalized["search_space_source"] = "default"
            else:
                if not isinstance(param_grid, dict):
                    raise ValueError("param_grid must be a dict")
                normalized["search_space_source"] = "provided"
            normalized["param_grid"] = param_grid

        if strategy == "random":
            param_distributions = config.get("param_distributions")
            if param_distributions is None:
                param_distributions = default_grid
                normalized["search_space_source"] = "default"
            else:
                if not isinstance(param_distributions, dict):
                    raise ValueError("param_distributions must be a dict")
                normalized["search_space_source"] = "provided"
            if n_iter < 1:
                raise ValueError("n_iter must be >= 1")
            normalized["param_distributions"] = param_distributions

        return normalized

    def _to_json_compatible(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): self._to_json_compatible(val)
                for key, val in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [self._to_json_compatible(item) for item in value]
        if isinstance(value, np.generic):
            return self._to_json_compatible(value.item())
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        return value
