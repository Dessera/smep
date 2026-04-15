"""Unified data loading for training pipeline.

Reads build artifacts produced by `smep data build` and returns
a single TrainingData object consumed by model.fit().
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Expected file names from `smep data build`
_X_TRAIN_FILE = "X_train.csv"
_Y_TRAIN_FILE = "y_train.csv"
_X_VAL_FILE = "X_val.csv"
_Y_VAL_FILE = "y_val.csv"
_X_TEST_FILE = "X_test.csv"
_Y_TEST_FILE = "y_test.csv"
_FEATURE_NAMES_FILE = "feature_names.txt"
_DATASET_MANIFEST_FILE = "dataset_manifest.json"


@dataclass
class TrainingData:
    """Container for all training data loaded from build artifacts."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    manifest: dict[str, Any] | None


def load_training_data(source_path: Path) -> TrainingData:
    """Load all training data from a build artifact directory.

    Args:
        source_path: Path to the directory containing build artifacts.

    Returns:
        TrainingData with all splits loaded as numpy arrays.

    Raises:
        FileNotFoundError: If any required file is missing.
        ValueError: If data dimensions are inconsistent.
    """
    source_path = Path(source_path)

    required_files = [
        _X_TRAIN_FILE,
        _Y_TRAIN_FILE,
        _X_VAL_FILE,
        _Y_VAL_FILE,
        _X_TEST_FILE,
        _Y_TEST_FILE,
        _FEATURE_NAMES_FILE,
    ]
    for fname in required_files:
        fpath = source_path / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Required file not found: {fpath}")

    X_train, y_train = _load_split(
        source_path / _X_TRAIN_FILE, source_path / _Y_TRAIN_FILE
    )
    X_val, y_val = _load_split(
        source_path / _X_VAL_FILE, source_path / _Y_VAL_FILE
    )
    X_test, y_test = _load_split(
        source_path / _X_TEST_FILE, source_path / _Y_TEST_FILE
    )

    feature_names = _load_feature_names(source_path / _FEATURE_NAMES_FILE)

    # Validate feature dimensions
    n_features = X_train.shape[1]
    for name, arr in [("X_val", X_val), ("X_test", X_test)]:
        if arr.shape[1] != n_features:
            raise ValueError(
                f"Feature dimension mismatch: X_train has {n_features} features, "
                f"{name} has {arr.shape[1]} features."
            )

    if feature_names and len(feature_names) != n_features:
        raise ValueError(
            f"Feature names count ({len(feature_names)}) does not match "
            f"feature dimension ({n_features})."
        )

    manifest = _load_manifest(source_path / _DATASET_MANIFEST_FILE)

    logger.info(
        "Data loaded: train=%d, val=%d, test=%d, features=%d",
        len(X_train),
        len(X_val),
        len(X_test),
        n_features,
    )

    return TrainingData(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        manifest=manifest,
    )


def _load_split(x_file: Path, y_file: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a single data split (X, y) from CSV files."""
    X_df = pd.read_csv(x_file)
    y_df = pd.read_csv(y_file)

    if X_df.empty or y_df.empty:
        raise ValueError(f"Data files are empty: {x_file.name}, {y_file.name}")

    if len(X_df) != len(y_df):
        raise ValueError(
            f"Sample count mismatch in {x_file.name}/{y_file.name}: "
            f"X={len(X_df)}, y={len(y_df)}"
        )

    y_col = y_df.columns[0]
    X = X_df.to_numpy(dtype=np.float32)
    y = y_df[y_col].to_numpy(dtype=np.int32)
    return X, y


def _load_feature_names(feature_file: Path) -> list[str]:
    """Load feature names from a text file (one per line)."""
    return [
        line.strip()
        for line in feature_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_manifest(manifest_file: Path) -> dict[str, Any] | None:
    """Load optional dataset manifest JSON."""
    if not manifest_file.exists():
        logger.info("No dataset manifest found at %s", manifest_file)
        return None
    return json.loads(manifest_file.read_text(encoding="utf-8"))
