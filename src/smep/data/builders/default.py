"""Default dataset builder implementation."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)

from .builder import DatasetBuilder
from .schema import (
    EXCLUDE_COLUMNS,
    CATEGORICAL_COLUMNS,
    BINARY_COLUMNS,
    DEFAULT_DROP_COLUMNS,
)
from .writer import write_dataset_outputs

logger = logging.getLogger(__name__)

_BASE_TABLE_FILE = "base_table.csv"


class DefaultDatasetBuilder(DatasetBuilder):
    """Default dataset builder that converts a base table into
    train/val/test splits with preprocessing artifacts.

    Args:
        label: Label column name.
        split: Tuple of (train, val, test) ratios.
        random_state: Random seed for reproducibility.
        stratify: Whether to stratify splits by label.
        imputer: Imputation strategy name.
        scaler: Scaling strategy name.
        min_coverage: Minimum non-missing rate to keep a column.
        drop_columns: Columns to manually exclude.
        keep_columns: Columns to manually keep (mutually exclusive with drop_columns).
    """

    def __init__(
        self,
        *,
        label: str = "hospital_expire_flag",
        split: tuple[float, float, float] = (0.8, 0.1, 0.1),
        random_state: int = 42,
        stratify: bool = True,
        imputer: str = "median",
        scaler: str = "standard",
        min_coverage: float = 0.0,
        drop_columns: list[str] | None = DEFAULT_DROP_COLUMNS,
        keep_columns: list[str] | None = None,
    ) -> None:
        self._label = label
        self._split = split
        self._random_state = random_state
        self._stratify = stratify
        self._imputer_name = imputer
        self._scaler_name = scaler
        self._min_coverage = min_coverage
        self._drop_columns = drop_columns
        self._keep_columns = keep_columns

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, base_table_path: Path, output_path: Path) -> None:
        df = self._load_and_validate(base_table_path)
        y = df[self._label]
        X, col_info = self._classify_columns(df)
        X, dropped_coverage = self._filter_coverage(X, col_info)

        (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
        ) = self._split_data(X, y)

        artifacts: dict[str, Any] = {
            "label_column": self._label,
            "exclude_columns": sorted(EXCLUDE_COLUMNS | {self._label}),
        }

        X_train, X_val, X_test, artifacts = self._impute(
            X_train,
            X_val,
            X_test,
            col_info,
            artifacts,
        )

        X_train, X_val, X_test, artifacts = self._encode(
            X_train,
            X_val,
            X_test,
            col_info,
            artifacts,
        )

        X_train, X_val, X_test, artifacts = self._scale(
            X_train,
            X_val,
            X_test,
            col_info,
            artifacts,
        )

        artifacts["feature_names"] = list(X_train.columns)
        artifacts["dropped_low_coverage"] = dropped_coverage

        manifest = self._build_manifest(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            col_info,
            dropped_coverage,
        )

        write_dataset_outputs(
            output_path,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            artifacts,
            manifest,
        )

    # ------------------------------------------------------------------
    # Step 1: Load & validate
    # ------------------------------------------------------------------

    def _load_and_validate(self, path: Path) -> pd.DataFrame:
        if path.is_dir():
            csv_path = path / _BASE_TABLE_FILE
        else:
            csv_path = path

        if not csv_path.exists():
            raise FileNotFoundError(f"Base table not found: {csv_path}")

        logger.info("Loading base table from %s", csv_path)
        df = pd.read_csv(csv_path)

        # Validate label
        if self._label not in df.columns:
            raise ValueError(
                f"Label column '{self._label}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        # Validate primary keys
        for pk in ("subject_id", "hadm_id", "icustay_id"):
            if pk not in df.columns:
                raise ValueError(f"Primary key column '{pk}' not found")

        pk_cols = ["subject_id", "hadm_id", "icustay_id"]
        dup_count = df.duplicated(subset=pk_cols).sum()
        if dup_count > 0:
            logger.warning(
                "Found %d duplicate primary key rows – dropping duplicates",
                dup_count,
            )
            df = df.drop_duplicates(subset=pk_cols, keep="first")

        # Drop rows with missing label
        missing_label = df[self._label].isna().sum()
        if missing_label > 0:
            logger.warning(
                "Dropping %d rows with missing label '%s'",
                missing_label,
                self._label,
            )
            df = df.dropna(subset=[self._label])

        logger.info("Loaded %d samples × %d columns", len(df), len(df.columns))
        return df

    # ------------------------------------------------------------------
    # Step 2: Column classification
    # ------------------------------------------------------------------

    def _classify_columns(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, list[str]]]:
        exclude = EXCLUDE_COLUMNS | {self._label}
        feature_cols = [c for c in df.columns if c not in exclude]

        # Apply keep/drop filters
        if self._keep_columns is not None:
            feature_cols = [
                c for c in feature_cols if c in set(self._keep_columns)
            ]
        elif self._drop_columns is not None:
            drop_set = set(self._drop_columns)
            feature_cols = [c for c in feature_cols if c not in drop_set]

        numeric_cols: list[str] = []
        binary_cols: list[str] = []
        categorical_cols: list[str] = []
        unknown_cols: list[str] = []

        present_binary = BINARY_COLUMNS - {self._label}
        present_categorical = CATEGORICAL_COLUMNS

        for col in feature_cols:
            if col in present_binary:
                binary_cols.append(col)
            elif col in present_categorical:
                categorical_cols.append(col)
            elif pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)
            elif pd.api.types.is_string_dtype(
                df[col]
            ) or pd.api.types.is_object_dtype(df[col]):
                unknown_cols.append(col)
            else:
                numeric_cols.append(col)

        if unknown_cols:
            logger.warning(
                "Excluding %d unrecognised string columns: %s",
                len(unknown_cols),
                unknown_cols,
            )

        keep = numeric_cols + binary_cols + categorical_cols
        X = df[keep].copy()

        col_info: dict[str, list[str]] = {
            "numeric": numeric_cols,
            "binary": binary_cols,
            "categorical": categorical_cols,
        }

        logger.info(
            "Column classification: %d numeric, %d binary, %d categorical "
            "(excluded %d unknown)",
            len(numeric_cols),
            len(binary_cols),
            len(categorical_cols),
            len(unknown_cols),
        )
        return X, col_info

    # ------------------------------------------------------------------
    # Step 3: Coverage filter
    # ------------------------------------------------------------------

    def _filter_coverage(
        self,
        X: pd.DataFrame,
        col_info: dict[str, list[str]],
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        if self._min_coverage <= 0.0:
            return X, {}

        coverage = X.notna().mean()
        drop_mask = coverage < self._min_coverage
        dropped = {
            col: round(float(coverage[col]), 4) for col in X.columns[drop_mask]
        }

        if dropped:
            logger.info(
                "Dropping %d columns below %.1f%% coverage: %s",
                len(dropped),
                self._min_coverage * 100,
                list(dropped.keys()),
            )
            X = X.drop(columns=list(dropped.keys()))

            # Update col_info
            dropped_set = set(dropped.keys())
            for key in col_info:
                col_info[key] = [
                    c for c in col_info[key] if c not in dropped_set
                ]

        return X, dropped

    # ------------------------------------------------------------------
    # Step 4: Stratified split
    # ------------------------------------------------------------------

    def _split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> tuple[
        pd.DataFrame,
        pd.Series,
        pd.DataFrame,
        pd.Series,
        pd.DataFrame,
        pd.Series,
    ]:
        train_r, val_r, test_r = self._split
        stratify_col = y if self._stratify else None

        # Check if stratification is feasible
        if self._stratify:
            min_class_count = y.value_counts().min()
            if min_class_count < 3:
                logger.warning(
                    "Minority class has only %d samples – "
                    "falling back to non-stratified split",
                    min_class_count,
                )
                stratify_col = None

        # Step 1: split off test
        X_rest, X_test, y_rest, y_test = train_test_split(
            X,
            y,
            test_size=test_r,
            random_state=self._random_state,
            stratify=stratify_col,
        )

        # Step 2: split rest into train/val
        val_fraction = val_r / (train_r + val_r)
        stratify_rest = y_rest if stratify_col is not None else None

        X_train, X_val, y_train, y_val = train_test_split(
            X_rest,
            y_rest,
            test_size=val_fraction,
            random_state=self._random_state,
            stratify=stratify_rest,
        )

        logger.info(
            "Split: train=%d, val=%d, test=%d",
            len(X_train),
            len(X_val),
            len(X_test),
        )
        return X_train, y_train, X_val, y_val, X_test, y_test

    # ------------------------------------------------------------------
    # Step 5: Imputation
    # ------------------------------------------------------------------

    def _impute(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        col_info: dict[str, list[str]],
        artifacts: dict[str, Any],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        numeric_cols = [c for c in col_info["numeric"] if c in X_train.columns]
        binary_cols = [c for c in col_info["binary"] if c in X_train.columns]
        categorical_cols = [
            c for c in col_info["categorical"] if c in X_train.columns
        ]

        # Categorical: fill with "_MISSING"
        for df in (X_train, X_val, X_test):
            for col in categorical_cols:
                if col in df.columns:
                    df[col] = df[col].fillna("_MISSING")

        artifacts["imputer_numeric"] = None
        artifacts["imputer_binary"] = None

        if self._imputer_name == "none":
            artifacts["numeric_columns"] = numeric_cols
            artifacts["binary_columns"] = binary_cols
            artifacts["categorical_columns"] = categorical_cols
            return X_train, X_val, X_test, artifacts

        # Numeric imputer
        if numeric_cols:
            imp = self._make_imputer(self._imputer_name)
            X_train[numeric_cols] = imp.fit_transform(X_train[numeric_cols])
            X_val[numeric_cols] = imp.transform(X_val[numeric_cols])
            X_test[numeric_cols] = imp.transform(X_test[numeric_cols])
            artifacts["imputer_numeric"] = imp

        # Binary imputer (always most_frequent)
        if binary_cols:
            imp_bin = SimpleImputer(strategy="most_frequent")
            X_train[binary_cols] = imp_bin.fit_transform(X_train[binary_cols])
            X_val[binary_cols] = imp_bin.transform(X_val[binary_cols])
            X_test[binary_cols] = imp_bin.transform(X_test[binary_cols])
            artifacts["imputer_binary"] = imp_bin

        artifacts["numeric_columns"] = numeric_cols
        artifacts["binary_columns"] = binary_cols
        artifacts["categorical_columns"] = categorical_cols
        return X_train, X_val, X_test, artifacts

    @staticmethod
    def _make_imputer(name: str) -> SimpleImputer | KNNImputer:
        if name == "median":
            return SimpleImputer(strategy="median")
        if name == "mean":
            return SimpleImputer(strategy="mean")
        if name == "knn":
            return KNNImputer(n_neighbors=5)
        raise ValueError(f"Unknown imputer: {name}")

    # ------------------------------------------------------------------
    # Step 6: Categorical encoding
    # ------------------------------------------------------------------

    def _encode(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        col_info: dict[str, list[str]],
        artifacts: dict[str, Any],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        cat_cols = [c for c in col_info["categorical"] if c in X_train.columns]

        if not cat_cols:
            artifacts["encoder"] = None
            return X_train, X_val, X_test, artifacts

        encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore",
            drop="if_binary",
        )
        encoded_train = encoder.fit_transform(X_train[cat_cols])
        encoded_val = encoder.transform(X_val[cat_cols])
        encoded_test = encoder.transform(X_test[cat_cols])

        encoded_names = list(encoder.get_feature_names_out(cat_cols))

        X_train = self._replace_columns(
            X_train, cat_cols, encoded_train, encoded_names
        )
        X_val = self._replace_columns(
            X_val, cat_cols, encoded_val, encoded_names
        )
        X_test = self._replace_columns(
            X_test, cat_cols, encoded_test, encoded_names
        )

        artifacts["encoder"] = encoder
        return X_train, X_val, X_test, artifacts

    @staticmethod
    def _replace_columns(
        df: pd.DataFrame,
        old_cols: list[str],
        new_data: np.ndarray,
        new_names: list[str],
    ) -> pd.DataFrame:
        df = df.drop(columns=old_cols)
        encoded_df = pd.DataFrame(new_data, columns=new_names, index=df.index)
        return pd.concat([df, encoded_df], axis=1)

    # ------------------------------------------------------------------
    # Step 7: Scaling
    # ------------------------------------------------------------------

    def _scale(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        col_info: dict[str, list[str]],
        artifacts: dict[str, Any],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        numeric_cols = [c for c in col_info["numeric"] if c in X_train.columns]

        if self._scaler_name == "none" or not numeric_cols:
            artifacts["scaler"] = None
            return X_train, X_val, X_test, artifacts

        scaler = self._make_scaler(self._scaler_name)
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

        artifacts["scaler"] = scaler
        return X_train, X_val, X_test, artifacts

    @staticmethod
    def _make_scaler(
        name: str,
    ) -> StandardScaler | MinMaxScaler | RobustScaler:
        if name == "standard":
            return StandardScaler()
        if name == "minmax":
            return MinMaxScaler()
        if name == "robust":
            return RobustScaler()
        raise ValueError(f"Unknown scaler: {name}")

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def _build_manifest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        col_info: dict[str, list[str]],
        dropped_coverage: dict[str, float],
    ) -> dict[str, Any]:
        def _label_dist(s: pd.Series) -> dict[str, int]:
            return {str(k): int(v) for k, v in s.value_counts().items()}

        n_before = (
            len(col_info["numeric"])
            + len(col_info["binary"])
            + len(col_info["categorical"])
        )

        return {
            "random_state": self._random_state,
            "split_ratio": {
                "train": self._split[0],
                "val": self._split[1],
                "test": self._split[2],
            },
            "label_column": self._label,
            "stratified": self._stratify,
            "imputer": self._imputer_name,
            "scaler": self._scaler_name,
            "min_coverage": self._min_coverage,
            "total_samples": len(X_train) + len(X_val) + len(X_test),
            "split_sizes": {
                "train": len(X_train),
                "val": len(X_val),
                "test": len(X_test),
            },
            "label_distribution": {
                "train": _label_dist(y_train),
                "val": _label_dist(y_val),
                "test": _label_dist(y_test),
            },
            "n_features_before_encoding": n_before,
            "n_features_after_encoding": X_train.shape[1],
            "dropped_low_coverage": dropped_coverage,
        }
