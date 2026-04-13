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
    TEMPORAL_FEATURE_CONFIG_PATH = (
        Path(__file__).resolve().parent.parent
        / "resources"
        / "mimic_temporal_features.json"
    )

    def __init__(
        self,
        agg_stats: list[str] | None = None,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
        split_enabled: bool = True,
        min_age: int = 15,
        min_icu_hours: int = 12,
        first_stay_only: bool = True,
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
            min_age: Minimum patient age in years for cohort inclusion.
            min_icu_hours: Minimum ICU stay duration in hours.
            first_stay_only: Whether to keep only the first ICU stay per subject.
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
        if min_age < 0:
            raise ValueError("min_age must be non-negative")
        if min_icu_hours <= 0:
            raise ValueError("min_icu_hours must be positive")

        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify
        self.split_enabled = split_enabled
        self.min_age = min_age
        self.min_icu_hours = min_icu_hours
        self.first_stay_only = first_stay_only
        self.temporal_feature_config_path = self.TEMPORAL_FEATURE_CONFIG_PATH
        self.temporal_feature_config = self._load_temporal_feature_config()

        self.cohort_df: pd.DataFrame | None = None
        self.labels_df: pd.DataFrame | None = None
        self.static_features: pd.DataFrame | None = None
        self.temporal_hourly: pd.DataFrame | None = None
        self.temporal_features: pd.DataFrame | None = None
        self.merged_df: pd.DataFrame | None = None
        self.filtered_df: pd.DataFrame | None = None
        self.full_df: pd.DataFrame | None = None
        self.train_df: pd.DataFrame | None = None
        self.test_df: pd.DataFrame | None = None
        self.final_df: pd.DataFrame | None = None
        self.split_metadata: dict[str, object] = {}
        self.preprocessing_metadata: dict[str, object] = {}
        self.sample_filtering_metadata: dict[str, object] = {}
        self.stage_metadata: dict[str, dict[str, object]] = {}
        self.temporal_quality_metadata: dict[str, object] = {
            "config_path": str(self.temporal_feature_config_path),
            "sources": {},
        }

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
        """Filter patients with sepsis diagnosis and ICU stay constraints."""
        diagnoses_path = source_path / "DIAGNOSES_ICD.csv"
        icustays_path = source_path / "ICUSTAYS.csv"
        admissions_path = source_path / "ADMISSIONS.csv"
        patients_path = source_path / "PATIENTS.csv"

        diagnoses_df = self._read_csv_standard(
            diagnoses_path,
            required_columns=["subject_id", "hadm_id", "icd9_code"],
        )
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
        admissions_df = self._read_csv_standard(
            admissions_path,
            required_columns=["subject_id", "hadm_id", "admittime"],
        )
        patients_df = self._read_csv_standard(
            patients_path,
            required_columns=["subject_id", "dob"],
        )

        # Filter for sepsis ICD-9 codes (remove dots from codes)
        sepsis_mask = (
            diagnoses_df["icd9_code"]
            .astype(str)
            .str.replace(".", "", regex=False)
            .isin(self.SEPSIS_ICD9_CODES)
        )

        sepsis_admissions = diagnoses_df[sepsis_mask][
            ["subject_id", "hadm_id"]
        ].drop_duplicates()

        icustays_df["intime"] = pd.to_datetime(icustays_df["intime"])
        icustays_df["outtime"] = pd.to_datetime(icustays_df["outtime"])
        admissions_df["admittime"] = pd.to_datetime(admissions_df["admittime"])
        patients_df["dob"] = pd.to_datetime(patients_df["dob"])

        cohort_df = sepsis_admissions.merge(
            icustays_df,
            on=["subject_id", "hadm_id"],
            how="inner",
        )
        cohort_df = cohort_df.merge(
            admissions_df[["subject_id", "hadm_id", "admittime"]],
            on=["subject_id", "hadm_id"],
            how="left",
        )
        cohort_df = cohort_df.merge(
            patients_df[["subject_id", "dob"]],
            on="subject_id",
            how="left",
        )

        cohort_df["age"] = (
            cohort_df["admittime"] - cohort_df["dob"]
        ).dt.days / 365.25
        cohort_df["icu_los_hours"] = (
            cohort_df["outtime"] - cohort_df["intime"]
        ).dt.total_seconds() / 3600

        cohort_df = cohort_df[
            (cohort_df["age"] >= self.min_age)
            & (cohort_df["icu_los_hours"] >= self.min_icu_hours)
        ].copy()

        cohort_df = cohort_df.sort_values(
            ["subject_id", "intime", "hadm_id", "icustay_id"]
        )
        if self.first_stay_only:
            cohort_df = cohort_df.drop_duplicates(
                subset=["subject_id"], keep="first"
            )

        self.cohort_df = cohort_df[
            [
                "subject_id",
                "hadm_id",
                "icustay_id",
                "intime",
                "outtime",
                "age",
                "icu_los_hours",
            ]
        ].drop_duplicates()

        logger.info(
            "Identified %s sepsis ICU stays after cohort filters",
            len(self.cohort_df),
        )

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
        labels_df = cohort_df[["subject_id", "hadm_id", "icustay_id"]].merge(
            admissions_df[["subject_id", "hadm_id", "hospital_expire_flag"]],
            on=["subject_id", "hadm_id"],
            how="left",
        )
        self.labels_df = labels_df

        logger.info(
            f"Mortality rate: {self.labels_df['hospital_expire_flag'].mean():.2%}"
        )

    def _extract_static_features(self, source_path: Path) -> None:
        """Extract static patient features."""
        patients_path = source_path / "PATIENTS.csv"
        admissions_path = source_path / "ADMISSIONS.csv"
        icustays_path = source_path / "ICUSTAYS.csv"

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
                "admission_location",
                "insurance",
                "ethnicity",
            ],
        )
        icustays_df = self._read_csv_standard(
            icustays_path,
            required_columns=[
                "subject_id",
                "hadm_id",
                "icustay_id",
                "first_careunit",
                "intime",
            ],
        )

        admissions_df["admittime"] = pd.to_datetime(admissions_df["admittime"])
        icustays_df["intime"] = pd.to_datetime(icustays_df["intime"])

        admissions_df = admissions_df.sort_values(
            ["subject_id", "admittime", "hadm_id"]
        ).copy()
        admissions_df["hospstay_seq"] = (
            admissions_df.groupby("subject_id").cumcount() + 1
        )
        admissions_df["first_hosp_stay"] = (
            admissions_df["hospstay_seq"] == 1
        ).astype(int)

        icustays_df = icustays_df.sort_values(
            ["hadm_id", "intime", "icustay_id"]
        ).copy()
        icustays_df["icustay_seq"] = (
            icustays_df.groupby("hadm_id").cumcount() + 1
        )
        icustays_df["first_icu_stay"] = (
            icustays_df["icustay_seq"] == 1
        ).astype(int)

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
                    "admission_location",
                    "insurance",
                    "ethnicity",
                    "hospstay_seq",
                    "first_hosp_stay",
                ]
            ],
            on=["subject_id", "hadm_id"],
            how="left",
        )
        static_df = static_df.merge(
            icustays_df[
                [
                    "subject_id",
                    "hadm_id",
                    "icustay_id",
                    "first_careunit",
                    "icustay_seq",
                    "first_icu_stay",
                ]
            ],
            on=["subject_id", "hadm_id", "icustay_id"],
            how="left",
        )

        # Calculate age only if cohort did not already provide it
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
            columns=[
                "admission_type",
                "admission_location",
                "insurance",
                "ethnicity",
                "first_careunit",
            ],
            prefix=["admit", "admit_loc", "ins", "eth", "careunit"],
        )

        # Select final static features
        feature_cols = [
            "subject_id",
            "hadm_id",
            "icustay_id",
            "age",
            "gender_m",
            "hospstay_seq",
            "icustay_seq",
            "first_hosp_stay",
            "first_icu_stay",
        ] + [
            col
            for col in static_df.columns
            if col.startswith(
                ("admit_", "admit_loc_", "ins_", "eth_", "careunit_")
            )
        ]
        self.static_features = static_df[feature_cols]

        logger.info(f"Extracted {len(feature_cols) - 3} static features")

    def _extract_temporal_features(self, source_path: Path) -> None:
        """Extract hourly temporal features, then aggregate them for tabular export."""
        icu_cohort = self._require_frame(self.cohort_df, "cohort_df").copy()

        chart_hourly = self._extract_chartevents(source_path, icu_cohort)
        lab_hourly = self._extract_labevents(source_path, icu_cohort)

        temporal_hourly = pd.concat(
            [chart_hourly, lab_hourly], axis=0, ignore_index=True
        )

        if temporal_hourly.empty:
            self.temporal_hourly = pd.DataFrame(
                columns=[
                    "subject_id",
                    "hadm_id",
                    "icustay_id",
                    "hours_in",
                    "feature_name",
                    "value_mean",
                    "value_count",
                    "value_std",
                ]
            )
        else:
            self.temporal_hourly = (
                temporal_hourly.groupby(
                    [
                        "subject_id",
                        "hadm_id",
                        "icustay_id",
                        "hours_in",
                        "feature_name",
                    ],
                    as_index=False,
                )["valuenum"]
                .agg(["mean", "count", "std"])
                .reset_index()
                .rename(
                    columns={
                        "mean": "value_mean",
                        "count": "value_count",
                        "std": "value_std",
                    }
                )
            )
            self.temporal_hourly["value_std"] = self.temporal_hourly[
                "value_std"
            ].fillna(0.0)
            self.temporal_hourly = self.temporal_hourly.sort_values(
                [
                    "subject_id",
                    "hadm_id",
                    "icustay_id",
                    "hours_in",
                    "feature_name",
                ]
            ).reset_index(drop=True)

        self.temporal_features = self._aggregate_temporal_features(
            icu_cohort,
            self._require_frame(self.temporal_hourly, "temporal_hourly"),
        )

        logger.info(
            "Extracted %s hourly rows and %s aggregated temporal features",
            len(self.temporal_hourly),
            len(self.temporal_features.columns) - 3,
        )

    def _extract_chartevents(
        self, source_path: Path, icu_cohort: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract hourly vital-sign rows from CHARTEVENTS."""
        chartevents_path = source_path / "CHARTEVENTS.csv"
        return self._extract_hourly_measurements(
            file_path=chartevents_path,
            icu_cohort=icu_cohort,
            feature_specs=self._get_temporal_feature_specs("chartevents"),
            include_icustay_id=True,
            source_name="chartevents",
        )

    def _extract_labevents(
        self, source_path: Path, icu_cohort: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract hourly lab rows from LABEVENTS."""
        labevents_path = source_path / "LABEVENTS.csv"
        return self._extract_hourly_measurements(
            file_path=labevents_path,
            icu_cohort=icu_cohort,
            feature_specs=self._get_temporal_feature_specs("labevents"),
            include_icustay_id=False,
            source_name="labevents",
        )

    def _extract_hourly_measurements(
        self,
        file_path: Path,
        icu_cohort: pd.DataFrame,
        feature_specs: dict[str, dict[str, Any]],
        include_icustay_id: bool,
        source_name: str,
    ) -> pd.DataFrame:
        """Extract hourly long-format rows for a temporal source table."""
        item_lookup = {
            item_id: feature_name
            for feature_name, feature_spec in feature_specs.items()
            for item_id in feature_spec.get("itemids", [])
        }
        required_columns = [
            "subject_id",
            "hadm_id",
            "charttime",
            "itemid",
            "valuenum",
        ]
        if include_icustay_id:
            required_columns.insert(2, "icustay_id")

        raw_rows: list[pd.DataFrame] = []
        cohort_columns = [
            "subject_id",
            "hadm_id",
            "icustay_id",
            "intime",
            "outtime",
        ]
        merge_keys = ["subject_id", "hadm_id", "icustay_id"]
        if not include_icustay_id:
            merge_keys = ["subject_id", "hadm_id"]

        try:
            for chunk in self._read_csv_chunks_standard(
                file_path,
                chunksize=100000,
                required_columns=required_columns,
            ):
                chunk["charttime"] = pd.to_datetime(chunk["charttime"])
                chunk["itemid"] = pd.to_numeric(
                    chunk["itemid"], errors="coerce"
                )
                chunk["valuenum"] = pd.to_numeric(
                    chunk["valuenum"], errors="coerce"
                ).astype(float)
                chunk = chunk[chunk["itemid"].isin(item_lookup)]
                if chunk.empty:
                    continue

                chunk = chunk.merge(
                    icu_cohort[cohort_columns],
                    on=merge_keys,
                    how="inner",
                )
                if chunk.empty:
                    continue

                window_end = chunk["intime"] + pd.Timedelta(
                    hours=self.TIME_WINDOW_HOURS
                )
                chunk_window_end = pd.concat(
                    [window_end, chunk["outtime"]], axis=1
                ).min(axis=1)
                chunk = chunk[
                    chunk["valuenum"].notna()
                    & (chunk["charttime"] >= chunk["intime"])
                    & (chunk["charttime"] <= chunk_window_end)
                ].copy()
                if chunk.empty:
                    continue

                chunk["hours_in"] = (
                    (chunk["charttime"] - chunk["intime"]).dt.total_seconds()
                    // 3600
                ).astype(int)
                chunk["feature_name"] = chunk["itemid"].map(item_lookup)
                chunk = self._apply_quality_rules(
                    chunk,
                    feature_specs=feature_specs,
                    source_name=source_name,
                )
                if chunk.empty:
                    continue
                raw_rows.append(
                    chunk[
                        [
                            "subject_id",
                            "hadm_id",
                            "icustay_id",
                            "hours_in",
                            "feature_name",
                            "valuenum",
                        ]
                    ]
                )
        except Exception as error:
            logger.warning(
                "Error processing %s: %s. Using empty hourly features.",
                file_path.name,
                error,
            )

        if not raw_rows:
            return pd.DataFrame(
                columns=[
                    "subject_id",
                    "hadm_id",
                    "icustay_id",
                    "hours_in",
                    "feature_name",
                    "valuenum",
                ]
            )

        return pd.concat(raw_rows, axis=0, ignore_index=True)

    def _load_temporal_feature_config(self) -> dict[str, dict[str, Any]]:
        """Load external temporal feature mapping and quality rules."""
        if not self.temporal_feature_config_path.exists():
            raise FileNotFoundError(
                "Temporal feature configuration file not found: "
                f"{self.temporal_feature_config_path}"
            )

        with self.temporal_feature_config_path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)

        if not isinstance(loaded, dict):
            raise ValueError(
                "Temporal feature configuration must be a JSON object"
            )

        return cast(dict[str, dict[str, Any]], loaded)

    def _get_temporal_feature_specs(
        self, source_name: str
    ) -> dict[str, dict[str, Any]]:
        """Return configured temporal feature specs for a source table."""
        source_config = self.temporal_feature_config.get(source_name, {})
        if not isinstance(source_config, dict):
            raise ValueError(
                f"Temporal feature config for {source_name} must be an object"
            )
        return cast(dict[str, dict[str, Any]], source_config)

    def _apply_quality_rules(
        self,
        chunk: pd.DataFrame,
        feature_specs: dict[str, dict[str, Any]],
        source_name: str,
    ) -> pd.DataFrame:
        """Apply unit normalization and range checks to hourly rows."""
        cleaned = chunk.copy()
        if "valueuom" not in cleaned.columns:
            cleaned["valueuom"] = ""
        cleaned["valueuom"] = (
            cleaned["valueuom"].fillna("").astype(str).str.strip().str.lower()
        )

        converted_rows = 0
        clipped_rows = 0
        dropped_rows = 0

        for feature_name, feature_spec in feature_specs.items():
            feature_mask = cleaned["feature_name"] == feature_name
            if not feature_mask.any():
                continue

            for conversion in feature_spec.get("unit_conversions", []):
                conversion_kind = conversion.get("kind")
                conversion_units = {
                    str(unit).strip().lower()
                    for unit in conversion.get("units", [])
                }
                conversion_mask = feature_mask & cleaned["valueuom"].isin(
                    conversion_units
                )
                if conversion.get("infer_if_above_one"):
                    conversion_mask = conversion_mask | (
                        feature_mask & (cleaned["valuenum"] > 1)
                    )
                if not conversion_mask.any():
                    continue

                if conversion_kind == "fahrenheit_to_celsius":
                    cleaned.loc[conversion_mask, "valuenum"] = (
                        cleaned.loc[conversion_mask, "valuenum"] - 32.0
                    ) * (5.0 / 9.0)
                    converted_rows += int(conversion_mask.sum())
                elif conversion_kind == "percent_to_fraction":
                    cleaned.loc[conversion_mask, "valuenum"] = (
                        cleaned.loc[conversion_mask, "valuenum"] / 100.0
                    )
                    converted_rows += int(conversion_mask.sum())

            valid_min = feature_spec.get("valid_min")
            valid_max = feature_spec.get("valid_max")
            out_of_range = str(feature_spec.get("out_of_range", "drop"))

            if valid_min is not None:
                low_mask = feature_mask & (cleaned["valuenum"] < valid_min)
                if low_mask.any():
                    if out_of_range == "clip":
                        cleaned.loc[low_mask, "valuenum"] = valid_min
                        clipped_rows += int(low_mask.sum())
                    else:
                        cleaned.loc[low_mask, "valuenum"] = float("nan")

            if valid_max is not None:
                high_mask = feature_mask & (cleaned["valuenum"] > valid_max)
                if high_mask.any():
                    if out_of_range == "clip":
                        cleaned.loc[high_mask, "valuenum"] = valid_max
                        clipped_rows += int(high_mask.sum())
                    else:
                        cleaned.loc[high_mask, "valuenum"] = float("nan")

        valid_mask = cleaned["valuenum"].notna()
        dropped_rows += int((~valid_mask).sum())
        self._update_quality_metadata(
            source_name,
            rows_seen=len(cleaned),
            rows_kept=int(valid_mask.sum()),
            rows_dropped=dropped_rows,
            unit_converted=converted_rows,
            clipped=clipped_rows,
        )
        return cleaned.loc[valid_mask].copy()

    def _update_quality_metadata(
        self,
        source_name: str,
        rows_seen: int,
        rows_kept: int,
        rows_dropped: int,
        unit_converted: int,
        clipped: int,
    ) -> None:
        """Accumulate temporal data quality statistics by source."""
        sources = cast(
            dict[str, dict[str, int]], self.temporal_quality_metadata["sources"]
        )
        source_stats = sources.setdefault(
            source_name,
            {
                "rows_seen": 0,
                "rows_kept": 0,
                "rows_dropped": 0,
                "unit_converted": 0,
                "clipped": 0,
            },
        )
        source_stats["rows_seen"] += rows_seen
        source_stats["rows_kept"] += rows_kept
        source_stats["rows_dropped"] += rows_dropped
        source_stats["unit_converted"] += unit_converted
        source_stats["clipped"] += clipped

    def _aggregate_temporal_features(
        self, icu_cohort: pd.DataFrame, temporal_hourly: pd.DataFrame
    ) -> pd.DataFrame:
        """Aggregate hourly rows into per-stay tabular features."""
        key_columns = ["subject_id", "hadm_id", "icustay_id"]
        base = icu_cohort[key_columns].drop_duplicates().copy()

        if temporal_hourly.empty:
            return base

        aggregated = (
            temporal_hourly.groupby(key_columns + ["feature_name"])[
                "value_mean"
            ]
            .agg(self.agg_stats)
            .unstack("feature_name")
        )
        aggregated.columns = [
            f"{feature_name}_{stat}"
            for stat, feature_name in aggregated.columns.to_flat_index()
        ]
        aggregated = aggregated.reset_index()

        return base.merge(aggregated, on=key_columns, how="left")

    def _clean_and_aggregate(self) -> None:
        """Clean data and merge all features."""
        merged_df = self._build_merged_dataset()
        filtered_df = self._filter_sparse_samples(merged_df)

        self.merged_df = merged_df.copy()
        self.filtered_df = filtered_df.copy()

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
            static_features,
            on=["subject_id", "hadm_id", "icustay_id"],
            how="left",
        ).merge(
            temporal_features,
            on=["subject_id", "hadm_id", "icustay_id"],
            how="left",
        )

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
        self._save_stage_outputs(target_path)

        full_df = self._require_frame(self.full_df, "full_df")
        train_df = self._require_frame(self.train_df, "train_df")

        full_features = full_df.drop(columns=["hospital_expire_flag"])
        full_labels = full_df["hospital_expire_flag"]

        processed_dataset = full_df.copy()
        processed_dataset.to_csv(
            target_path / "processed_dataset.csv", index=False
        )
        self._record_stage_metadata("processed_dataset", processed_dataset)

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
            "cohort_rules": {
                "min_age": self.min_age,
                "min_icu_hours": self.min_icu_hours,
                "first_stay_only": self.first_stay_only,
            },
            "temporal_feature_config": {
                "path": str(self.temporal_feature_config_path),
                "sources": {
                    source_name: list(source_specs.keys())
                    for source_name, source_specs in self.temporal_feature_config.items()
                },
            },
            "temporal_quality": self.temporal_quality_metadata,
            "stage_outputs": self.stage_metadata,
            "sample_filtering": self.sample_filtering_metadata,
            "split": self.split_metadata,
            "preprocessing": self.preprocessing_metadata,
        }

        with open(target_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Saved metadata")

    def _save_stage_outputs(self, target_path: Path) -> None:
        """Persist stage-level datasets for debugging and reuse."""
        stage_frames = {
            "cohort": self._require_frame(self.cohort_df, "cohort_df"),
            "labels": self._require_frame(self.labels_df, "labels_df"),
            "static_features": self._require_frame(
                self.static_features, "static_features"
            ),
            "temporal_hourly": self._require_frame(
                self.temporal_hourly, "temporal_hourly"
            ),
            "temporal_features": self._require_frame(
                self.temporal_features, "temporal_features"
            ),
            "merged_dataset": self._require_frame(self.merged_df, "merged_df"),
            "filtered_dataset": self._require_frame(
                self.filtered_df, "filtered_df"
            ),
        }

        for stage_name, frame in stage_frames.items():
            output_path = target_path / f"{stage_name}.csv"
            frame.to_csv(output_path, index=False)
            self._record_stage_metadata(stage_name, frame)

        logger.info("Saved %s stage-level datasets", len(stage_frames))

    def _record_stage_metadata(self, stage_name: str, df: pd.DataFrame) -> None:
        """Record a compact summary for a stage output."""
        self.stage_metadata[stage_name] = {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": list(df.columns),
        }

    def _get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Return model feature columns excluding identifiers and labels."""
        return [
            column
            for column in df.columns
            if column
            not in {
                "subject_id",
                "hadm_id",
                "icustay_id",
                "hospital_expire_flag",
            }
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
