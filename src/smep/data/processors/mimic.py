"""MIMIC-III data processor for sepsis mortality prediction."""

from pathlib import Path
import json
import logging
from datetime import datetime
from collections.abc import Iterator
from typing import Any, cast
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .processor import DataProcessor

logger = logging.getLogger(__name__)


class MIMIC3Processor(DataProcessor):
    """Processor for MIMIC-III Clinical Database.

    Implements feature engineering pipeline for sepsis mortality prediction,
    including cohort selection, static/temporal feature extraction,
    and data cleaning.
    """

    # Sepsis ICD-9 codes
    SEPSIS_ICD9_CODES = ["99591", "99592", "78552"]

    # Time window for temporal features (hours)
    TIME_WINDOW_HOURS = 24

    # Available aggregation statistics
    VALID_AGG_STATS = {"mean", "max", "min", "std"}

    def __init__(
        self,
        agg_stats: list[str] | None = None,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
        split_enabled: bool = True,
    ):
        """Initialize the MIMIC3 processor.

        Args:
            agg_stats: List of aggregation statistics to compute for temporal
                features. Valid values: 'mean', 'max', 'min', 'std'.
                Defaults to all four statistics if None.
            test_size: Fraction of samples reserved for the test split.
            random_state: Random seed used for reproducible splitting.
            stratify: Whether train/test split should be label-stratified.
            split_enabled: Whether to write train/test split files.
        """
        if agg_stats is None:
            self.agg_stats = ["mean", "max", "min", "std"]
        else:
            invalid = set(agg_stats) - self.VALID_AGG_STATS
            if invalid:
                raise ValueError(
                    f"Invalid aggregation statistics: {invalid}. "
                    f"Valid options are: {self.VALID_AGG_STATS}"
                )
            self.agg_stats = list(agg_stats)

        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")

        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify
        self.split_enabled = split_enabled

        self.cohort_df: pd.DataFrame | None = None
        self.labels_df: pd.DataFrame | None = None
        self.static_features: pd.DataFrame | None = None
        self.temporal_features: pd.DataFrame | None = None
        self.full_df: pd.DataFrame | None = None
        self.train_df: pd.DataFrame | None = None
        self.test_df: pd.DataFrame | None = None
        self.final_df: pd.DataFrame | None = None
        self.split_metadata: dict[str, object] = {}
        self.preprocessing_metadata: dict[str, object] = {}
        self.sample_filtering_metadata: dict[str, object] = {}

    def process(self, source_path: Path, target_path: Path) -> None:
        """Process MIMIC-III data and generate ML-ready dataset.

        Args:
            source_path: Path to MIMIC-III CSV files.
            target_path: Path to save processed data.
        """
        logger.info("Starting MIMIC-III data processing...")
        logger.info(f"Source: {source_path}")
        logger.info(f"Target: {target_path}")

        # Ensure target directory exists
        self._ensure_target_directory(target_path)

        # Execute processing pipeline
        logger.info("Step 1: Filtering sepsis cohort...")
        self._filter_sepsis_cohort(source_path)

        logger.info("Step 2: Extracting labels...")
        self._extract_labels(source_path)

        logger.info("Step 3: Extracting static features...")
        self._extract_static_features(source_path)

        logger.info("Step 4: Extracting temporal features...")
        self._extract_temporal_features(source_path)

        logger.info("Step 5: Cleaning and aggregating...")
        self._clean_and_aggregate()

        logger.info("Step 6: Saving output...")
        self._save_output(target_path)

        logger.info("Processing complete!")

    def _filter_sepsis_cohort(self, source_path: Path) -> None:
        """Filter patients with sepsis diagnosis."""
        diagnoses_path = source_path / "DIAGNOSES_ICD.csv"
        diagnoses_df = self._read_csv_standard(
            diagnoses_path,
            required_columns=["subject_id", "hadm_id", "icd9_code"],
        )

        # Filter for sepsis ICD-9 codes (remove dots from codes)
        sepsis_mask = (
            diagnoses_df["icd9_code"]
            .astype(str)
            .str.replace(".", "", regex=False)
            .isin(self.SEPSIS_ICD9_CODES)
        )
        self.cohort_df = diagnoses_df[sepsis_mask][
            ["subject_id", "hadm_id"]
        ].drop_duplicates()

        logger.info(f"Identified {len(self.cohort_df)} sepsis admissions")

    def _extract_labels(self, source_path: Path) -> None:
        """Extract mortality labels."""
        admissions_path = source_path / "ADMISSIONS.csv"
        admissions_df = self._read_csv_standard(
            admissions_path,
            required_columns=[
                "subject_id",
                "hadm_id",
                "hospital_expire_flag",
            ],
        )
        cohort_df = self._require_frame(self.cohort_df, "cohort_df")

        # Merge with cohort to get labels
        self.labels_df = cohort_df.merge(
            admissions_df[["subject_id", "hadm_id", "hospital_expire_flag"]],
            on=["subject_id", "hadm_id"],
            how="left",
        )

        logger.info(
            f"Mortality rate: {self.labels_df['hospital_expire_flag'].mean():.2%}"
        )

    def _extract_static_features(self, source_path: Path) -> None:
        """Extract static patient features."""
        patients_path = source_path / "PATIENTS.csv"
        admissions_path = source_path / "ADMISSIONS.csv"

        patients_df = self._read_csv_standard(
            patients_path,
            required_columns=["subject_id", "gender", "dob"],
        )
        admissions_df = self._read_csv_standard(
            admissions_path,
            required_columns=[
                "subject_id",
                "hadm_id",
                "admittime",
                "admission_type",
                "insurance",
                "ethnicity",
            ],
        )

        # Merge with cohort
        static_df = self._require_frame(self.cohort_df, "cohort_df").copy()

        # Add patient demographics
        static_df = static_df.merge(
            patients_df[["subject_id", "gender", "dob"]],
            on="subject_id",
            how="left",
        )

        # Add admission info
        static_df = static_df.merge(
            admissions_df[
                [
                    "subject_id",
                    "hadm_id",
                    "admittime",
                    "admission_type",
                    "insurance",
                    "ethnicity",
                ]
            ],
            on=["subject_id", "hadm_id"],
            how="left",
        )

        # Calculate age
        static_df["dob"] = pd.to_datetime(static_df["dob"])
        static_df["admittime"] = pd.to_datetime(static_df["admittime"])
        static_df["age"] = (
            static_df["admittime"] - static_df["dob"]
        ).dt.days / 365.25

        # Encode categorical variables
        static_df["gender_m"] = (static_df["gender"] == "M").astype(int)

        # One-hot encode admission type, insurance, ethnicity
        static_df = pd.get_dummies(
            static_df,
            columns=["admission_type", "insurance", "ethnicity"],
            prefix=["admit", "ins", "eth"],
        )

        # Select final static features
        feature_cols = ["subject_id", "hadm_id", "age", "gender_m"] + [
            col
            for col in static_df.columns
            if col.startswith(("admit_", "ins_", "eth_"))
        ]
        self.static_features = static_df[feature_cols]

        logger.info(f"Extracted {len(feature_cols) - 2} static features")

    def _extract_temporal_features(self, source_path: Path) -> None:
        """Extract and aggregate temporal features from chart events and lab events."""
        # Get ICU stay times
        icustays_path = source_path / "ICUSTAYS.csv"
        icustays_df = self._read_csv_standard(
            icustays_path,
            required_columns=[
                "subject_id",
                "hadm_id",
                "icustay_id",
                "intime",
                "outtime",
            ],
        )
        icustays_df["intime"] = pd.to_datetime(icustays_df["intime"])
        icustays_df["outtime"] = pd.to_datetime(icustays_df["outtime"])

        # Merge with cohort
        icu_cohort = self._require_frame(self.cohort_df, "cohort_df").merge(
            icustays_df[["subject_id", "hadm_id", "icustay_id", "intime"]],
            on=["subject_id", "hadm_id"],
            how="left",
        )

        # Extract chart events (vital signs)
        temporal_features = self._extract_chartevents(source_path, icu_cohort)

        # Extract lab events
        lab_features = self._extract_labevents(source_path, icu_cohort)

        # Merge temporal and lab features
        self.temporal_features = temporal_features.merge(
            lab_features, on=["subject_id", "hadm_id"], how="outer"
        )

        logger.info(
            f"Extracted {len(self.temporal_features.columns) - 2} temporal features"
        )

    def _extract_chartevents(
        self, source_path: Path, icu_cohort: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract vital signs from CHARTEVENTS."""
        chartevents_path = source_path / "CHARTEVENTS.csv"

        # Define vital sign item IDs (common MIMIC-III item IDs)
        vital_items = {
            "HR": [211, 220045],  # Heart Rate
            "SysBP": [51, 442, 455, 6701, 220179, 220050],  # Systolic BP
            "DiasBP": [8368, 8440, 8441, 8555, 220180, 220051],  # Diastolic BP
            "Temp": [223761, 678],  # Temperature
            "RR": [618, 615, 220210, 224690],  # Respiratory Rate
            "SpO2": [646, 220277],  # Oxygen Saturation
        }

        # Collect all cohort rows from all chunks before aggregating to avoid
        # duplicate columns caused by per-chunk partial aggregations.
        raw_rows: list[pd.DataFrame] = []

        try:
            for chunk in self._read_csv_chunks_standard(
                chartevents_path,
                chunksize=100000,
                required_columns=[
                    "subject_id",
                    "hadm_id",
                    "charttime",
                    "itemid",
                    "valuenum",
                ],
            ):
                chunk["charttime"] = pd.to_datetime(chunk["charttime"])

                # Filter for cohort and time window
                chunk = chunk.merge(
                    icu_cohort[["subject_id", "hadm_id", "intime"]],
                    on=["subject_id", "hadm_id"],
                    how="inner",
                )
                chunk = chunk[
                    (chunk["charttime"] >= chunk["intime"])
                    & (
                        chunk["charttime"]
                        <= chunk["intime"]
                        + pd.Timedelta(hours=self.TIME_WINDOW_HOURS)
                    )
                ]
                if not chunk.empty:
                    raw_rows.append(
                        chunk[["subject_id", "hadm_id", "itemid", "valuenum"]]
                    )
        except Exception as e:
            logger.warning(
                f"Error processing CHARTEVENTS: {e}. Using empty features."
            )

        if not raw_rows:
            return icu_cohort[["subject_id", "hadm_id"]].copy()

        all_data = pd.concat(raw_rows, axis=0)
        aggregated_features: list[pd.DataFrame] = []
        for vital_name, item_ids in vital_items.items():
            vital_data = all_data[all_data["itemid"].isin(item_ids)]
            if not vital_data.empty:
                agg = vital_data.groupby(["subject_id", "hadm_id"])[
                    "valuenum"
                ].agg([(stat, stat) for stat in self.agg_stats])
                agg.columns = [
                    f"{vital_name}_{stat}" for stat in self.agg_stats
                ]
                aggregated_features.append(agg)

        if aggregated_features:
            result = pd.concat(aggregated_features, axis=1).reset_index()
        else:
            result = icu_cohort[["subject_id", "hadm_id"]].copy()

        return result

    def _extract_labevents(
        self, source_path: Path, icu_cohort: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract lab values from LABEVENTS."""
        labevents_path = source_path / "LABEVENTS.csv"

        # Define lab item labels to extract
        lab_items = {
            "Creatinine": [50912],
            "Platelet": [51265],
            "WBC": [51300, 51301],
            "Hemoglobin": [51222],
            "Sodium": [50824, 50983],
            "Potassium": [50822, 50971],
            "Chloride": [50806, 50902],
        }

        raw_rows: list[pd.DataFrame] = []

        try:
            for chunk in self._read_csv_chunks_standard(
                labevents_path,
                chunksize=100000,
                required_columns=[
                    "subject_id",
                    "hadm_id",
                    "charttime",
                    "itemid",
                    "valuenum",
                ],
            ):
                chunk["charttime"] = pd.to_datetime(chunk["charttime"])

                # Filter for cohort and time window
                chunk = chunk.merge(
                    icu_cohort[["subject_id", "hadm_id", "intime"]],
                    on=["subject_id", "hadm_id"],
                    how="inner",
                )
                chunk = chunk[
                    (chunk["charttime"] >= chunk["intime"])
                    & (
                        chunk["charttime"]
                        <= chunk["intime"]
                        + pd.Timedelta(hours=self.TIME_WINDOW_HOURS)
                    )
                ]
                if not chunk.empty:
                    raw_rows.append(
                        chunk[["subject_id", "hadm_id", "itemid", "valuenum"]]
                    )
        except Exception as e:
            logger.warning(
                f"Error processing LABEVENTS: {e}. Using empty features."
            )

        if not raw_rows:
            return icu_cohort[["subject_id", "hadm_id"]].copy()

        all_data = pd.concat(raw_rows, axis=0)
        aggregated_features: list[pd.DataFrame] = []
        for lab_name, item_ids in lab_items.items():
            lab_data = all_data[all_data["itemid"].isin(item_ids)]
            if not lab_data.empty:
                agg = lab_data.groupby(["subject_id", "hadm_id"])[
                    "valuenum"
                ].agg([(stat, stat) for stat in self.agg_stats])
                agg.columns = [f"{lab_name}_{stat}" for stat in self.agg_stats]
                aggregated_features.append(agg)

        if aggregated_features:
            result = pd.concat(aggregated_features, axis=1).reset_index()
        else:
            result = icu_cohort[["subject_id", "hadm_id"]].copy()

        return result

    def _clean_and_aggregate(self) -> None:
        """Clean data and merge all features."""
        merged_df = self._build_merged_dataset()
        filtered_df = self._filter_sparse_samples(merged_df)

        if filtered_df.empty:
            raise ValueError("No samples remaining after sample filtering")

        if self.split_enabled:
            train_df, test_df = self._split_dataset(filtered_df)
            processed_train_df, processed_test_df = self._preprocess_datasets(
                train_df, test_df
            )
            full_df = pd.concat(
                [processed_train_df, processed_test_df], axis=0
            ).sort_index()
        else:
            processed_train_df, processed_test_df = self._preprocess_datasets(
                filtered_df, None
            )
            full_df = processed_train_df.copy()

        self.train_df = processed_train_df
        self.test_df = processed_test_df
        self.full_df = full_df
        self.final_df = full_df

    def _build_merged_dataset(self) -> pd.DataFrame:
        """Build the merged sample table before filtering and preprocessing."""
        labels_df = self._require_frame(self.labels_df, "labels_df")
        static_features = self._require_frame(
            self.static_features, "static_features"
        )
        temporal_features = self._require_frame(
            self.temporal_features, "temporal_features"
        )

        merged_df = labels_df.merge(
            static_features, on=["subject_id", "hadm_id"], how="left"
        ).merge(temporal_features, on=["subject_id", "hadm_id"], how="left")

        logger.info(f"Merged data shape: {merged_df.shape}")
        return merged_df

    def _filter_sparse_samples(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """Remove samples whose feature vectors are mostly missing."""
        feature_columns = self._get_feature_columns(merged_df)
        if not feature_columns:
            logger.warning("No features found after initial merge")
            raise ValueError("No features found after initial merge")

        feature_df = merged_df[feature_columns]
        logger.info(
            "Initial features: %s, samples: %s",
            feature_df.shape[1],
            feature_df.shape[0],
        )

        missing_ratio = feature_df.isna().sum(axis=1) / len(feature_columns)
        valid_samples = missing_ratio <= 0.9
        removed_samples = int((~valid_samples).sum())
        filtered_df = merged_df.loc[valid_samples].copy()

        logger.info(
            "Removed %s samples with >90%% missing values", removed_samples
        )

        self.sample_filtering_metadata = {
            "removed_samples": removed_samples,
            "remaining_samples": len(filtered_df),
            "threshold": 0.9,
        }
        return filtered_df

    def _split_dataset(
        self, filtered_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the filtered dataset into train and test partitions."""
        if len(filtered_df) < 2:
            raise ValueError(
                "At least 2 samples are required to create a train/test split"
            )

        label_series = filtered_df["hospital_expire_flag"]
        stratify_labels = None
        if self.stratify:
            class_counts = label_series.value_counts(dropna=False)
            if len(class_counts) < 2:
                raise ValueError(
                    "Stratified split requires at least 2 label classes"
                )
            if int(class_counts.min()) < 2:
                raise ValueError(
                    "Stratified split requires at least 2 samples in each label class"
                )
            stratify_labels = label_series

        try:
            split_frames = cast(
                list[pd.DataFrame],
                train_test_split(
                    filtered_df,
                    test_size=self.test_size,
                    random_state=self.random_state,
                    stratify=stratify_labels,
                ),
            )
        except ValueError as error:
            raise ValueError(
                "Failed to create train/test split. Adjust test_size or disable stratify."
            ) from error

        train_df, test_df = split_frames[0], split_frames[1]
        logger.info(
            "Created train/test split before preprocessing: train=%s, test=%s",
            train_df.shape,
            test_df.shape,
        )

        self.split_metadata = {
            "enabled": True,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "stratify": self.stratify,
            "train_samples": len(train_df),
            "test_samples": len(test_df),
        }
        return train_df.copy(), test_df.copy()

    def _preprocess_datasets(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame | None
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Fit preprocessing on train data and apply it to all partitions."""
        train_feature_columns = self._get_feature_columns(train_df)
        if not train_feature_columns:
            raise ValueError("No features available for preprocessing")

        train_feature_df = train_df[train_feature_columns]
        missing_ratio = train_feature_df.isna().sum() / len(train_feature_df)
        kept_feature_columns = missing_ratio[
            missing_ratio <= 0.95
        ].index.tolist()
        removed_feature_count = len(train_feature_columns) - len(
            kept_feature_columns
        )

        logger.info(
            "Removed %s features with >95%% missing values based on training split",
            removed_feature_count,
        )

        if not kept_feature_columns:
            logger.error("No features remaining after training-only filtering")
            raise ValueError("All features were removed during filtering")

        train_features = train_df[kept_feature_columns]
        logger.info(
            "Training feature shape before imputation: %s", train_features.shape
        )

        imputer = SimpleImputer(strategy="median")
        train_imputed_values = cast(Any, imputer.fit_transform(train_features))
        train_imputed = pd.DataFrame(
            train_imputed_values,
            columns=kept_feature_columns,
            index=train_df.index,
        )

        scaler = StandardScaler()
        train_scaled_values = cast(Any, scaler.fit_transform(train_imputed))
        train_scaled = pd.DataFrame(
            train_scaled_values,
            columns=kept_feature_columns,
            index=train_df.index,
        )

        processed_train_df = train_scaled.copy()
        processed_train_df["hospital_expire_flag"] = train_df[
            "hospital_expire_flag"
        ]

        processed_test_df = None
        if test_df is not None:
            test_features = test_df[kept_feature_columns]
            test_imputed_values = cast(Any, imputer.transform(test_features))
            test_imputed = pd.DataFrame(
                test_imputed_values,
                columns=kept_feature_columns,
                index=test_df.index,
            )
            test_scaled_values = cast(Any, scaler.transform(test_imputed))
            test_scaled = pd.DataFrame(
                test_scaled_values,
                columns=kept_feature_columns,
                index=test_df.index,
            )
            processed_test_df = test_scaled.copy()
            processed_test_df["hospital_expire_flag"] = test_df[
                "hospital_expire_flag"
            ]

        fit_scope = "train_only" if test_df is not None else "all_data"
        self.preprocessing_metadata = {
            "fit_scope": fit_scope,
            "imputer": "median",
            "scaler": "standard",
            "feature_filter_source": fit_scope,
            "removed_feature_count": removed_feature_count,
            "feature_count_after_filtering": len(kept_feature_columns),
        }

        return processed_train_df, processed_test_df

    def _save_output(self, target_path: Path) -> None:
        """Save processed data to target directory."""
        full_df = self._require_frame(self.full_df, "full_df")
        train_df = self._require_frame(self.train_df, "train_df")

        full_features = full_df.drop(columns=["hospital_expire_flag"])
        full_labels = full_df["hospital_expire_flag"]

        if self.split_enabled:
            test_df = self._require_frame(self.test_df, "test_df")

            train_features = train_df.drop(columns=["hospital_expire_flag"])
            train_labels = train_df["hospital_expire_flag"]
            test_features = test_df.drop(columns=["hospital_expire_flag"])
            test_labels = test_df["hospital_expire_flag"]

            train_features.to_csv(target_path / "X_train.csv", index=False)
            test_features.to_csv(target_path / "X_test.csv", index=False)
            train_labels.to_csv(
                target_path / "y_train.csv", index=False, header=True
            )
            test_labels.to_csv(
                target_path / "y_test.csv", index=False, header=True
            )

            logger.info(
                "Saved train/test split outputs: train=%s, test=%s",
                train_features.shape,
                test_features.shape,
            )
        else:
            self.split_metadata = {
                "enabled": False,
                "test_size": None,
                "random_state": self.random_state,
                "stratify": self.stratify,
                "train_samples": len(train_df),
                "test_samples": 0,
            }
            logger.info("Train/test split disabled; exported full dataset only")

        full_features.to_csv(target_path / "X.csv", index=False)
        logger.info(f"Saved feature matrix: {full_features.shape}")

        full_labels.to_csv(target_path / "y.csv", index=False, header=True)
        logger.info(f"Saved labels: {len(full_labels)}")

        with open(target_path / "feature_names.txt", "w") as f:
            f.write("\n".join(full_features.columns))
        logger.info(f"Saved {len(full_features.columns)} feature names")

        metadata: dict[str, Any] = {
            "n_samples": len(full_features),
            "n_features": len(full_features.columns),
            "mortality_rate": float(full_labels.mean()),
            "processing_date": datetime.now().isoformat(),
            "time_window_hours": self.TIME_WINDOW_HOURS,
            "sepsis_codes": self.SEPSIS_ICD9_CODES,
            "sample_filtering": self.sample_filtering_metadata,
            "split": self.split_metadata,
            "preprocessing": self.preprocessing_metadata,
        }

        with open(target_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Saved metadata")

    def _get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Return model feature columns excluding identifiers and labels."""
        return [
            column
            for column in df.columns
            if column not in {"subject_id", "hadm_id", "hospital_expire_flag"}
        ]

    def _read_csv_standard(
        self,
        file_path: Path,
        required_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Read CSV with filename and column-name compatibility normalization."""
        resolved_path = self._resolve_csv_path(file_path)
        df = pd.read_csv(resolved_path)
        normalized = self._normalize_columns(df)

        if required_columns:
            self._validate_required_columns(
                normalized,
                required_columns,
                resolved_path,
            )

        return normalized

    def _read_csv_chunks_standard(
        self,
        file_path: Path,
        chunksize: int,
        required_columns: list[str] | None = None,
    ) -> Iterator[pd.DataFrame]:
        """Read CSV chunks with normalized column names."""
        resolved_path = self._resolve_csv_path(file_path)
        for chunk in pd.read_csv(resolved_path, chunksize=chunksize):
            normalized = self._normalize_columns(chunk)
            if required_columns:
                self._validate_required_columns(
                    normalized,
                    required_columns,
                    resolved_path,
                )
            yield normalized

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to lowercase for case-insensitive access."""
        normalized_df = df.copy()
        normalized_df.columns = [
            str(column).strip().lower() for column in normalized_df.columns
        ]
        return normalized_df

    def _resolve_csv_path(self, file_path: Path) -> Path:
        """Resolve a CSV path case-insensitively within its parent directory."""
        if file_path.exists():
            return file_path

        parent = file_path.parent
        if not parent.exists() or not parent.is_dir():
            raise FileNotFoundError(f"Required file not found: {file_path}")

        target_name = file_path.name.lower()
        for candidate in parent.iterdir():
            if candidate.is_file() and candidate.name.lower() == target_name:
                return candidate

        raise FileNotFoundError(f"Required file not found: {file_path}")

    def _validate_required_columns(
        self,
        df: pd.DataFrame,
        required_columns: list[str],
        source_path: Path,
    ) -> None:
        """Validate required columns after normalization."""
        missing = [
            column
            for column in required_columns
            if column.lower() not in df.columns
        ]
        if missing:
            available = ", ".join(map(str, df.columns))
            raise ValueError(
                f"Missing required columns in {source_path}: {missing}. "
                f"Available columns: {available}"
            )

    def _require_frame(
        self, frame: pd.DataFrame | None, name: str
    ) -> pd.DataFrame:
        """Require a DataFrame attribute to be populated before use."""
        if frame is None:
            raise RuntimeError(f"Required dataframe '{name}' has not been set")
        return frame
