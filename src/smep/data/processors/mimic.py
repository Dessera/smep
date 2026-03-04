"""MIMIC-III data processor for sepsis mortality prediction."""

from pathlib import Path
import json
import logging
from datetime import datetime
import pandas as pd
from sklearn.impute import SimpleImputer
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

    def __init__(self):
        """Initialize the MIMIC3 processor."""
        self.cohort_df = None
        self.labels_df = None
        self.static_features = None
        self.temporal_features = None
        self.final_df = None

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
        diagnoses_df = pd.read_csv(diagnoses_path)

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
        admissions_df = pd.read_csv(admissions_path)

        # Merge with cohort to get labels
        self.labels_df = self.cohort_df.merge(
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

        patients_df = pd.read_csv(patients_path)
        admissions_df = pd.read_csv(admissions_path)

        # Merge with cohort
        static_df = self.cohort_df.copy()

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
        icustays_df = pd.read_csv(icustays_path)
        icustays_df["intime"] = pd.to_datetime(icustays_df["intime"])
        icustays_df["outtime"] = pd.to_datetime(icustays_df["outtime"])

        # Merge with cohort
        icu_cohort = self.cohort_df.merge(
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

        # Read chart events in chunks to handle large file
        aggregated_features = []

        try:
            for chunk in pd.read_csv(chartevents_path, chunksize=100000):
                chunk["charttime"] = pd.to_datetime(chunk["charttime"])

                # Filter for cohort and time window
                chunk = chunk.merge(
                    icu_cohort[["subject_id", "hadm_id", "intime"]],
                    on=["subject_id", "hadm_id"],
                    how="inner",
                )

                # Filter for first 24 hours after ICU admission
                chunk = chunk[
                    (chunk["charttime"] >= chunk["intime"])
                    & (
                        chunk["charttime"]
                        <= chunk["intime"]
                        + pd.Timedelta(hours=self.TIME_WINDOW_HOURS)
                    )
                ]

                # Extract vital signs
                for vital_name, item_ids in vital_items.items():
                    vital_data = chunk[chunk["itemid"].isin(item_ids)]
                    if not vital_data.empty:
                        # Aggregate: mean, max, min, std
                        agg = vital_data.groupby(["subject_id", "hadm_id"])[
                            "valuenum"
                        ].agg(
                            [
                                ("mean", "mean"),
                                ("max", "max"),
                                ("min", "min"),
                                ("std", "std"),
                            ]
                        )
                        agg.columns = [
                            f"{vital_name}_{stat}"
                            for stat in ["mean", "max", "min", "std"]
                        ]
                        aggregated_features.append(agg)
        except Exception as e:
            logger.warning(
                f"Error processing CHARTEVENTS: {e}. Using empty features."
            )

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

        aggregated_features = []

        try:
            for chunk in pd.read_csv(labevents_path, chunksize=100000):
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

                # Extract lab values
                for lab_name, item_ids in lab_items.items():
                    lab_data = chunk[chunk["itemid"].isin(item_ids)]
                    if not lab_data.empty:
                        agg = lab_data.groupby(["subject_id", "hadm_id"])[
                            "valuenum"
                        ].agg(
                            [
                                ("mean", "mean"),
                                ("max", "max"),
                                ("min", "min"),
                                ("std", "std"),
                            ]
                        )
                        agg.columns = [
                            f"{lab_name}_{stat}"
                            for stat in ["mean", "max", "min", "std"]
                        ]
                        aggregated_features.append(agg)
        except Exception as e:
            logger.warning(
                f"Error processing LABEVENTS: {e}. Using empty features."
            )

        if aggregated_features:
            result = pd.concat(aggregated_features, axis=1).reset_index()
        else:
            result = icu_cohort[["subject_id", "hadm_id"]].copy()

        return result

    def _clean_and_aggregate(self) -> None:
        """Clean data and merge all features."""
        # Merge all features
        self.final_df = self.labels_df.merge(
            self.static_features, on=["subject_id", "hadm_id"], how="left"
        ).merge(
            self.temporal_features, on=["subject_id", "hadm_id"], how="left"
        )

        logger.info(f"Merged data shape: {self.final_df.shape}")

        # Separate features and labels
        y = self.final_df["hospital_expire_flag"]
        X = self.final_df.drop(
            columns=["subject_id", "hadm_id", "hospital_expire_flag"]
        )

        logger.info(f"Initial features: {X.shape[1]}, samples: {X.shape[0]}")

        # Remove samples with >90% missing values
        if len(X.columns) > 0:
            missing_ratio = X.isna().sum(axis=1) / len(X.columns)
            valid_samples = missing_ratio <= 0.9
            removed_samples = (~valid_samples).sum()

            X = X[valid_samples]
            y = y[valid_samples]

            logger.info(
                f"Removed {removed_samples} samples with >90% missing values"
            )
        else:
            logger.warning("No features found after initial filtering")
            return

        # Remove features with >95% missing values
        missing_ratio_features = X.isna().sum() / len(X)
        valid_features = missing_ratio_features <= 0.95
        removed_feature_count = (~valid_features).sum()

        X = X.loc[:, valid_features]

        logger.info(
            f"Removed {removed_feature_count} features with >95% missing values"
        )
        logger.info(f"Final shape before imputation: {X.shape}")

        # Check if we have data remaining
        if X.empty or X.shape[1] == 0:
            logger.error("No features remaining after filtering")
            raise ValueError("All features were removed during filtering")

        # Impute missing values
        imputer = SimpleImputer(strategy="median")
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X), columns=X.columns, index=X.index
        )

        # Standardize features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_imputed),
            columns=X_imputed.columns,
            index=X_imputed.index,
        )

        # Store final data
        self.final_df = X_scaled
        self.final_df["hospital_expire_flag"] = y

    def _save_output(self, target_path: Path) -> None:
        """Save processed data to target directory."""
        # Separate features and labels
        y = self.final_df["hospital_expire_flag"]
        X = self.final_df.drop(columns=["hospital_expire_flag"])

        # Save feature matrix
        X.to_csv(target_path / "X.csv", index=False)
        logger.info(f"Saved feature matrix: {X.shape}")

        # Save labels
        y.to_csv(target_path / "y.csv", index=False, header=True)
        logger.info(f"Saved labels: {len(y)}")

        # Save feature names
        with open(target_path / "feature_names.txt", "w") as f:
            f.write("\n".join(X.columns))
        logger.info(f"Saved {len(X.columns)} feature names")

        # Save metadata
        metadata = {
            "n_samples": len(X),
            "n_features": len(X.columns),
            "mortality_rate": float(y.mean()),
            "processing_date": datetime.now().isoformat(),
            "time_window_hours": self.TIME_WINDOW_HOURS,
            "sepsis_codes": self.SEPSIS_ICD9_CODES,
        }

        with open(target_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Saved metadata")
