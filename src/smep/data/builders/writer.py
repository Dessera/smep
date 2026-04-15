"""Output writer for the dataset builder."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

logger = logging.getLogger(__name__)

# Output file names
X_TRAIN_FILE = "X_train.csv"
Y_TRAIN_FILE = "y_train.csv"
X_VAL_FILE = "X_val.csv"
Y_VAL_FILE = "y_val.csv"
X_TEST_FILE = "X_test.csv"
Y_TEST_FILE = "y_test.csv"
FEATURE_NAMES_FILE = "feature_names.txt"
ARTIFACTS_FILE = "preprocessing_artifacts.joblib"
MANIFEST_FILE = "dataset_manifest.json"


def write_dataset_outputs(
    output_path: Path,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    artifacts: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    """Write all dataset builder output files.

    Args:
        output_path: Directory to write files into.
        X_train: Training feature matrix.
        y_train: Training labels.
        X_val: Validation feature matrix.
        y_val: Validation labels.
        X_test: Test feature matrix.
        y_test: Test labels.
        artifacts: Preprocessing artifacts dict to serialize.
        manifest: Dataset manifest dict.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    # X/y CSVs
    X_train.to_csv(output_path / X_TRAIN_FILE, index=False)
    y_train.to_frame().to_csv(output_path / Y_TRAIN_FILE, index=False)
    X_val.to_csv(output_path / X_VAL_FILE, index=False)
    y_val.to_frame().to_csv(output_path / Y_VAL_FILE, index=False)
    X_test.to_csv(output_path / X_TEST_FILE, index=False)
    y_test.to_frame().to_csv(output_path / Y_TEST_FILE, index=False)

    # Feature names
    with open(output_path / FEATURE_NAMES_FILE, "w") as f:
        for name in X_train.columns:
            f.write(f"{name}\n")

    # Preprocessing artifacts
    joblib.dump(artifacts, output_path / ARTIFACTS_FILE)

    # Manifest
    manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    with open(output_path / MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info("Wrote %d files to %s", 9, output_path)
