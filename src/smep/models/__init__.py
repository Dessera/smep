"""Model registry module."""

from typing import Any, Dict, List, Type
import logging

from .model import Model
from .xgboost import XGBoostModel
from .dnn import DNNModel
from .data_loader import TrainingData, load_training_data
from .evaluator import (
    EvaluationResult,
    ThresholdResult,
    evaluate,
    find_optimal_threshold,
    compute_curve_points,
    render_curves,
    write_evaluation_outputs,
)
from .explainer import (
    ShapResult,
    normalize_shap_values,
    normalize_expected_value,
    write_explain_outputs,
    to_json_compatible,
)
from .feature_selector import (
    FeatureImportanceResult,
    FeatureImportanceReport,
    evaluate_feature_importance,
    write_feature_importance_outputs,
)

logger = logging.getLogger(__name__)


class ModelRegistry:
    """模型注册表，用于管理可用的训练模型。

    维护已注册模型的元数据，并提供按名称获取模型实例的方法。
    """

    def __init__(self):
        """初始化注册表并注册内置模型。"""
        self._registry: Dict[str, Dict[str, Any]] = {}
        self._register_builtin_models()

    def _register_builtin_models(self) -> None:
        """注册内置模型。"""
        self.register(
            name="xgboost",
            description="XGBoost in-hospital mortality prediction for sepsis (MIMIC-III binary classification)",
            model_class=XGBoostModel,
        )
        self.register(
            name="dnn",
            description="PyTorch MLP+Attention in-hospital mortality prediction for sepsis (MIMIC-III binary classification)",
            model_class=DNNModel,
        )

    def register(
        self, name: str, description: str, model_class: Type[Model]
    ) -> None:
        """注册一个新模型。

        Args:
            name: 模型标识符（用于 CLI）。
            description: 人类可读的描述。
            model_class: 要注册的 Model 子类。

        Raises:
            ValueError: 如果该名称已被注册。
        """
        if name in self._registry:
            raise ValueError(f"Model '{name}' is already registered")

        self._registry[name] = {
            "name": name,
            "description": description,
            "model_class": model_class,
        }
        logger.debug(f"已注册模型: {name}")

    def get_model(self, name: str) -> Model:
        """按名称获取模型实例。

        Args:
            name: 要获取的模型名称。

        Returns:
            请求的 Model 实例。

        Raises:
            KeyError: 如果模型名称未找到。
        """
        if name not in self._registry:
            available = ", ".join(self.list_models())
            raise KeyError(
                f"Model '{name}' not found. Available models: {available}"
            )

        model_class = self._registry[name]["model_class"]
        return model_class()

    def list_models(self) -> List[str]:
        """列出所有可用模型名称。

        Returns:
            可用模型名称列表。
        """
        return list(self._registry.keys())

    def get_model_info(self, name: str) -> Dict[str, str]:
        """获取指定模型的元信息。

        Args:
            name: 模型名称。

        Returns:
            包含 name 和 description 的字典。

        Raises:
            KeyError: 如果模型名称未找到。
        """
        if name not in self._registry:
            raise KeyError(f"Model '{name}' not found")

        entry = self._registry[name]
        return {"name": entry["name"], "description": entry["description"]}

    def get_all_model_info(self) -> List[Dict[str, str]]:
        """获取所有可用模型的元信息。

        Returns:
            包含模型元数据的字典列表。
        """
        return [
            {"name": entry["name"], "description": entry["description"]}
            for entry in self._registry.values()
        ]


# 全局模型注册表单例
_global_model_registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    """获取全局模型注册表。

    Returns:
        全局 ModelRegistry 实例。
    """
    return _global_model_registry


__all__ = [
    "Model",
    "XGBoostModel",
    "DNNModel",
    "ModelRegistry",
    "get_registry",
    "TrainingData",
    "load_training_data",
    "EvaluationResult",
    "ThresholdResult",
    "evaluate",
    "find_optimal_threshold",
    "compute_curve_points",
    "render_curves",
    "write_evaluation_outputs",
    "ShapResult",
    "normalize_shap_values",
    "normalize_expected_value",
    "write_explain_outputs",
    "to_json_compatible",
    "FeatureImportanceResult",
    "FeatureImportanceReport",
    "evaluate_feature_importance",
    "write_feature_importance_outputs",
]
