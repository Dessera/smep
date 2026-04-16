"""MIMIC-III base table exporter."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any
import logging

import numpy as np
import pandas as pd

from .base import DataExporter
from .quality import generate_quality_report
from .schema import (
    AGG_STATS,
    CHARTEVENTS_FEATURES,
    LABEVENTS_FEATURES,
    OUTPUT_COLUMNS,
    build_item_lookup,
    build_schema_dict,
)
from .scores import compute_scores_and_treatments
from .infection import compute_infection_timeline
from .sepsis import compute_sepsis_criteria
from .comorbidity import compute_comorbidity
from .writer import write_outputs

logger = logging.getLogger(__name__)

# ICD-9 codes used to identify sepsis patients
_SEPSIS_ICD9_CODES = {"99591", "99592", "78552"}

# Age ceiling for de-identified patients (>89 → 91.4)
_AGE_CEIL = 91.4
_AGE_DEIDENT_THRESHOLD = 89

_CHUNK_SIZE = 100_000


class MIMIC3Exporter(DataExporter):
    """Export a base table from raw MIMIC-III CSV files."""

    def __init__(
        self,
        *,
        min_age: int = 18,
        first_stay_only: bool = True,
        time_window_hours: int = 24,
        schema_version: str = "v1",
    ) -> None:
        if min_age < 0:
            raise ValueError("min_age must be non-negative")
        if time_window_hours <= 0:
            raise ValueError("time_window_hours must be positive")

        self.min_age = min_age
        self.first_stay_only = first_stay_only
        self.time_window_hours = time_window_hours
        self.schema_version = schema_version

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export(self, source_path: Path, target_path: Path) -> None:
        source_path = source_path.resolve()
        target_path = target_path.resolve()
        target_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            "MIMIC-III export: source=%s target=%s", source_path, target_path
        )

        # 1. Cohort
        logger.info("Step 1/8: Building cohort …")
        cohort_df = self._build_cohort(source_path)

        # 2. Static features (demographics + labels)
        logger.info("Step 2/8: Extracting static features …")
        static_df = self._extract_static_features(source_path, cohort_df)

        # 3. Temporal features (labs + vitals)
        logger.info("Step 3/8: Extracting temporal features …")
        temporal_df = self._extract_temporal_features(source_path, cohort_df)

        # 4. Severity scores & treatment features
        logger.info("Step 4/8: Computing scores & treatments …")
        scores_df = self._compute_scores_and_treatments(
            source_path, cohort_df, temporal_df
        )

        # 5. Infection timeline
        logger.info("Step 5/8: Extracting infection timeline …")
        infection_df = self._extract_infection_timeline(source_path, cohort_df)

        # 6. Sepsis diagnosis criteria
        logger.info("Step 6/8: Computing sepsis criteria …")
        sepsis_df = self._compute_sepsis_criteria(source_path, cohort_df)

        # 7. Comorbidity & baseline info
        logger.info("Step 7/8: Computing comorbidity & baseline …")
        comorbidity_df = self._compute_comorbidity(source_path, cohort_df)

        # 8. Merge, validate, write
        logger.info("Step 8/8: Merging & writing …")
        base_table = self._merge_and_finalise(
            cohort_df,
            static_df,
            temporal_df,
            scores_df,
            infection_df,
            sepsis_df,
            comorbidity_df,
        )

        schema = build_schema_dict(self.schema_version)
        quality = generate_quality_report(base_table, OUTPUT_COLUMNS)
        config = {
            "min_age": self.min_age,
            "first_stay_only": self.first_stay_only,
            "time_window_hours": self.time_window_hours,
            "schema_version": self.schema_version,
            "sepsis_icd9_codes": sorted(_SEPSIS_ICD9_CODES),
        }
        write_outputs(target_path, base_table, schema, quality, config)
        logger.info("Export finished: %s rows, %s columns", *base_table.shape)

    # ------------------------------------------------------------------
    # Step 1 – Cohort
    # ------------------------------------------------------------------

    def _build_cohort(self, source_path: Path) -> pd.DataFrame:
        diagnoses = self._read_csv(
            source_path / "DIAGNOSES_ICD.csv",
            required=["subject_id", "hadm_id", "icd9_code"],
        )
        icustays = self._read_csv(
            source_path / "ICUSTAYS.csv",
            required=[
                "subject_id",
                "hadm_id",
                "icustay_id",
                "intime",
                "outtime",
            ],
        )
        admissions = self._read_csv(
            source_path / "ADMISSIONS.csv",
            required=["subject_id", "hadm_id", "admittime", "dischtime"],
        )
        patients = self._read_csv(
            source_path / "PATIENTS.csv",
            required=["subject_id", "dob"],
        )

        # Identify sepsis admissions
        diag_codes = (
            diagnoses["icd9_code"].astype(str).str.replace(".", "", regex=False)
        )
        sepsis_pairs = diagnoses.loc[
            diag_codes.isin(_SEPSIS_ICD9_CODES),
            ["subject_id", "hadm_id"],
        ].drop_duplicates()

        # Parse datetimes
        for col in ("intime", "outtime"):
            icustays[col] = pd.to_datetime(icustays[col])
        admissions["admittime"] = pd.to_datetime(admissions["admittime"])
        admissions["dischtime"] = pd.to_datetime(admissions["dischtime"])
        patients["dob"] = pd.to_datetime(patients["dob"])

        # Join
        cohort = sepsis_pairs.merge(icustays, on=["subject_id", "hadm_id"])
        cohort = cohort.merge(
            admissions[["subject_id", "hadm_id", "admittime", "dischtime"]],
            on=["subject_id", "hadm_id"],
            how="left",
        )
        cohort = cohort.merge(
            patients[["subject_id", "dob"]],
            on="subject_id",
            how="left",
        )

        # Compute age (cap de-identified patients)
        cohort["age"] = (cohort["admittime"] - cohort["dob"]).dt.days / 365.25
        cohort.loc[cohort["age"] > _AGE_DEIDENT_THRESHOLD, "age"] = _AGE_CEIL

        # ICU length of stay (hours) for filtering
        cohort["icu_los_hours"] = (
            cohort["outtime"] - cohort["intime"]
        ).dt.total_seconds() / 3600

        # Filters
        n_before = len(cohort)
        cohort = cohort[cohort["age"] >= self.min_age].copy()
        logger.info(
            "Age filter (>= %s): %s → %s",
            self.min_age,
            n_before,
            len(cohort),
        )

        n_before = len(cohort)
        cohort = cohort[
            cohort["icu_los_hours"] >= self.time_window_hours
        ].copy()
        logger.info(
            "Time-window filter (>= %s h): %s → %s",
            self.time_window_hours,
            n_before,
            len(cohort),
        )

        cohort = cohort.sort_values(
            ["subject_id", "intime", "hadm_id", "icustay_id"]
        )
        if self.first_stay_only:
            n_before = len(cohort)
            cohort = cohort.drop_duplicates(subset=["subject_id"], keep="first")
            logger.info("First-stay-only: %s → %s", n_before, len(cohort))

        cohort = cohort[
            [
                "subject_id",
                "hadm_id",
                "icustay_id",
                "intime",
                "outtime",
                "admittime",
                "dischtime",
                "age",
            ]
        ].drop_duplicates()

        logger.info("Cohort size: %s ICU stays", len(cohort))
        if cohort.empty:
            raise ValueError("No ICU stays remaining after cohort filters")
        return cohort

    # ------------------------------------------------------------------
    # Step 2 – Static features & labels
    # ------------------------------------------------------------------

    def _extract_static_features(
        self, source_path: Path, cohort_df: pd.DataFrame
    ) -> pd.DataFrame:
        patients = self._read_csv(
            source_path / "PATIENTS.csv",
            required=["subject_id", "gender", "dod"],
        )
        admissions = self._read_csv(
            source_path / "ADMISSIONS.csv",
            required=[
                "subject_id",
                "hadm_id",
                "hospital_expire_flag",
                "ethnicity",
            ],
        )

        keys = ["subject_id", "hadm_id", "icustay_id"]
        static = cohort_df[
            keys + ["intime", "outtime", "age", "admittime", "dischtime"]
        ].copy()

        # Gender
        static = static.merge(
            patients[["subject_id", "gender", "dod"]],
            on="subject_id",
            how="left",
        )
        static["gender"] = static["gender"].fillna("").astype(str).str.upper()

        # Ethnicity & race indicators
        static = static.merge(
            admissions[
                ["subject_id", "hadm_id", "hospital_expire_flag", "ethnicity"]
            ],
            on=["subject_id", "hadm_id"],
            how="left",
        )
        static["ethnicity"] = (
            static["ethnicity"].fillna("").astype(str).str.upper()
        )

        # Hospital expire flag
        static["hospital_expire_flag"] = (
            static["hospital_expire_flag"].fillna(0).astype(int)
        )

        # 30-day mortality
        patients["dod"] = pd.to_datetime(patients["dod"], errors="coerce")
        dod_map = patients.dropna(subset=["dod"]).set_index("subject_id")["dod"]
        static["dod"] = static["subject_id"].map(dod_map)
        static["thirtyday_expire_flag"] = 0
        has_dod = static["dod"].notna() & static["dischtime"].notna()
        days_to_death = (
            static.loc[has_dod, "dod"] - static.loc[has_dod, "dischtime"]
        ).dt.total_seconds() / 86400
        static.loc[has_dod, "thirtyday_expire_flag"] = (
            days_to_death <= 30
        ).astype(int)

        # ICU & hospital LOS (days)
        static["icu_los"] = (
            (static["outtime"] - static["intime"]).dt.total_seconds() / 86400
        ).round(4)
        static["hosp_los"] = np.where(
            static["dischtime"].notna() & static["admittime"].notna(),
            (
                (static["dischtime"] - static["admittime"]).dt.total_seconds()
                / 86400
            ),
            np.nan,
        )

        keep = keys + [
            "gender",
            "ethnicity",
            "hospital_expire_flag",
            "thirtyday_expire_flag",
            "icu_los",
            "hosp_los",
        ]
        return static[keep].copy()

    # ------------------------------------------------------------------
    # Step 3 – Temporal features (labs + vitals)
    # ------------------------------------------------------------------

    def _extract_temporal_features(
        self, source_path: Path, cohort_df: pd.DataFrame
    ) -> pd.DataFrame:
        keys = ["subject_id", "hadm_id", "icustay_id"]

        chart_stats = self._extract_from_events(
            source_path / "CHARTEVENTS.csv",
            cohort_df,
            CHARTEVENTS_FEATURES,
            has_icustay_id=True,
        )
        lab_stats = self._extract_from_events(
            source_path / "LABEVENTS.csv",
            cohort_df,
            LABEVENTS_FEATURES,
            has_icustay_id=False,
        )

        all_stats = pd.concat([chart_stats, lab_stats], ignore_index=True)

        if all_stats.empty:
            logger.warning(
                "No temporal measurements found; "
                "all temporal columns will be NaN"
            )
            result = cohort_df[keys].copy()
            for feat_group in (LABEVENTS_FEATURES, CHARTEVENTS_FEATURES):
                for feat in feat_group:
                    for stat in AGG_STATS:
                        result[f"{feat}_{stat}"] = np.nan
            return result

        pivoted = all_stats.pivot_table(
            index=keys,
            columns="feature_name",
            values=["feat_min", "feat_max", "feat_mean"],
        )
        # Flatten multi-level columns: (stat, feature) → feature_stat
        pivoted.columns = [
            f"{feat}_{stat.removeprefix('feat_')}"
            for stat, feat in pivoted.columns
        ]
        pivoted = pivoted.reset_index()

        # Ensure every expected column exists
        for feat_group in (LABEVENTS_FEATURES, CHARTEVENTS_FEATURES):
            for feat in feat_group:
                for stat in AGG_STATS:
                    col = f"{feat}_{stat}"
                    if col not in pivoted.columns:
                        pivoted[col] = np.nan

        return pivoted

    def _extract_from_events(
        self,
        file_path: Path,
        cohort_df: pd.DataFrame,
        feature_specs: dict[str, dict[str, Any]],
        *,
        has_icustay_id: bool,
    ) -> pd.DataFrame:
        """Read an events table in chunks and return per-stay aggregates.

        Returns long-form DataFrame with columns:
            subject_id, hadm_id, icustay_id, feature_name,
            feat_min, feat_max, feat_mean
        """
        item_lookup = build_item_lookup(feature_specs)
        if not item_lookup:
            return pd.DataFrame()

        required = [
            "subject_id",
            "hadm_id",
            "charttime",
            "itemid",
            "valuenum",
        ]
        if has_icustay_id:
            required.insert(2, "icustay_id")

        merge_keys = ["subject_id", "hadm_id"]
        if has_icustay_id:
            merge_keys.append("icustay_id")

        cohort_cols = [
            "subject_id",
            "hadm_id",
            "icustay_id",
            "intime",
        ]
        window_td = pd.Timedelta(hours=self.time_window_hours)

        accum: list[pd.DataFrame] = []

        try:
            for chunk in self._read_csv_chunks(file_path, required=required):
                chunk["charttime"] = pd.to_datetime(
                    chunk["charttime"], errors="coerce"
                )
                chunk["itemid"] = pd.to_numeric(
                    chunk["itemid"], errors="coerce"
                )
                chunk["valuenum"] = pd.to_numeric(
                    chunk["valuenum"], errors="coerce"
                )

                # Keep only relevant items
                chunk = chunk[chunk["itemid"].isin(item_lookup)]
                if chunk.empty:
                    continue

                # Join with cohort to get intime
                chunk = chunk.merge(
                    cohort_df[cohort_cols], on=merge_keys, how="inner"
                )
                if chunk.empty:
                    continue

                # Time-window filter
                chunk = chunk[
                    chunk["valuenum"].notna()
                    & (chunk["charttime"] >= chunk["intime"])
                    & (chunk["charttime"] <= chunk["intime"] + window_td)
                ].copy()
                if chunk.empty:
                    continue

                # Map itemid → feature name
                chunk["feature_name"] = chunk["itemid"].map(item_lookup)

                # Unit conversions (Fahrenheit → Celsius)
                chunk = self._apply_unit_conversions(chunk, feature_specs)

                # Range filter
                chunk = self._apply_range_filter(chunk, feature_specs)

                if chunk.empty:
                    continue

                accum.append(
                    chunk[
                        [
                            "subject_id",
                            "hadm_id",
                            "icustay_id",
                            "feature_name",
                            "valuenum",
                        ]
                    ]
                )
        except FileNotFoundError:
            logger.warning("Events file not found: %s – skipping", file_path)
            return pd.DataFrame()

        if not accum:
            return pd.DataFrame()

        events = pd.concat(accum, ignore_index=True)
        keys = ["subject_id", "hadm_id", "icustay_id", "feature_name"]
        agg = events.groupby(keys, as_index=False)["valuenum"].agg(
            feat_min="min",
            feat_max="max",
            feat_mean="mean",
        )
        return agg

    # ------------------------------------------------------------------
    # Unit conversion & range filtering helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_unit_conversions(
        chunk: pd.DataFrame,
        feature_specs: dict[str, dict[str, Any]],
    ) -> pd.DataFrame:
        for feat_name, spec in feature_specs.items():
            # Fahrenheit → Celsius
            f_ids = spec.get("fahrenheit_itemids")
            if f_ids:
                mask = (chunk["feature_name"] == feat_name) & (
                    chunk["itemid"].isin(f_ids)
                )
                if mask.any():
                    chunk.loc[mask, "valuenum"] = (
                        chunk.loc[mask, "valuenum"] - 32.0
                    ) * (5.0 / 9.0)

            # Percent → fraction (e.g. FiO2: values > 1 are /100)
            if spec.get("normalize_to_fraction"):
                feat_mask = chunk["feature_name"] == feat_name
                pct_mask = feat_mask & (chunk["valuenum"] > 1)
                if pct_mask.any():
                    chunk.loc[pct_mask, "valuenum"] = (
                        chunk.loc[pct_mask, "valuenum"] / 100.0
                    )

            # Inches → centimetres (e.g. height)
            in_ids = spec.get("inches_itemids")
            if in_ids:
                mask = (chunk["feature_name"] == feat_name) & (
                    chunk["itemid"].isin(in_ids)
                )
                if mask.any():
                    chunk.loc[mask, "valuenum"] = (
                        chunk.loc[mask, "valuenum"] * 2.54
                    )
        return chunk

    @staticmethod
    def _apply_range_filter(
        chunk: pd.DataFrame,
        feature_specs: dict[str, dict[str, Any]],
    ) -> pd.DataFrame:
        keep_mask = pd.Series(True, index=chunk.index)
        for feat_name, spec in feature_specs.items():
            feat_mask = chunk["feature_name"] == feat_name
            if not feat_mask.any():
                continue
            v = chunk.loc[feat_mask, "valuenum"]
            valid = pd.Series(True, index=v.index)
            if "valid_min" in spec:
                valid = valid & (v >= spec["valid_min"])
            if "valid_max" in spec:
                valid = valid & (v <= spec["valid_max"])
            keep_mask.loc[v.index[~valid]] = False
        return chunk[keep_mask].copy()

    # ------------------------------------------------------------------
    # Step 4 – Severity scores & treatment features
    # ------------------------------------------------------------------

    def _compute_scores_and_treatments(
        self,
        source_path: Path,
        cohort_df: pd.DataFrame,
        temporal_df: pd.DataFrame,
    ) -> pd.DataFrame:
        return compute_scores_and_treatments(
            source_path,
            cohort_df,
            self.time_window_hours,
            temporal_df=temporal_df,
            read_csv_fn=self._read_csv,
            read_csv_chunks_fn=self._read_csv_chunks,
        )

    # ------------------------------------------------------------------
    # Step 5 – Infection timeline
    # ------------------------------------------------------------------

    def _extract_infection_timeline(
        self,
        source_path: Path,
        cohort_df: pd.DataFrame,
    ) -> pd.DataFrame:
        return compute_infection_timeline(
            source_path,
            cohort_df,
            read_csv_fn=self._read_csv,
        )

    # ------------------------------------------------------------------
    # Step 6 – Sepsis diagnosis criteria
    # ------------------------------------------------------------------

    def _compute_sepsis_criteria(
        self,
        source_path: Path,
        cohort_df: pd.DataFrame,
    ) -> pd.DataFrame:
        return compute_sepsis_criteria(
            source_path,
            cohort_df,
            read_csv_fn=self._read_csv,
        )

    # ------------------------------------------------------------------
    # Step 7 – Comorbidity & baseline info
    # ------------------------------------------------------------------

    def _compute_comorbidity(
        self,
        source_path: Path,
        cohort_df: pd.DataFrame,
    ) -> pd.DataFrame:
        return compute_comorbidity(
            source_path,
            cohort_df,
            read_csv_fn=self._read_csv,
        )

    # ------------------------------------------------------------------
    # Merge & finalise
    # ------------------------------------------------------------------

    def _merge_and_finalise(
        self,
        cohort_df: pd.DataFrame,
        static_df: pd.DataFrame,
        temporal_df: pd.DataFrame,
        scores_df: pd.DataFrame,
        infection_df: pd.DataFrame,
        sepsis_df: pd.DataFrame,
        comorbidity_df: pd.DataFrame,
    ) -> pd.DataFrame:
        keys = ["subject_id", "hadm_id", "icustay_id"]

        base = cohort_df[keys + ["intime", "outtime", "age"]].copy()
        base = base.merge(static_df, on=keys, how="left")
        base = base.merge(temporal_df, on=keys, how="left")
        base = base.merge(scores_df, on=keys, how="left")
        base = base.merge(infection_df, on=keys, how="left")
        base = base.merge(sepsis_df, on=keys, how="left")
        base = base.merge(comorbidity_df, on=keys, how="left")

        # Derived: BMI from temporal height/weight means
        h_m = base.get("height_mean", pd.Series(dtype=float))
        w_m = base.get("weight_mean", pd.Series(dtype=float))
        h_valid = h_m.notna() & (h_m > 0)
        w_valid = w_m.notna() & (w_m > 0)
        base["bmi"] = np.where(
            h_valid & w_valid,
            w_m / (h_m / 100.0) ** 2,
            np.nan,
        )

        # Derived: urineoutput normalised to mL/kg/h
        uo = base.get("urineoutput", pd.Series(dtype=float))
        base["urineoutput_per_kg_per_h"] = np.where(
            uo.notna() & w_valid & (w_m > 0),
            uo / w_m / self.time_window_hours,
            np.nan,
        )

        # Ensure all expected columns exist (in the right order)
        for col in OUTPUT_COLUMNS:
            if col not in base.columns:
                base[col] = np.nan

        base = base[OUTPUT_COLUMNS]
        return base

    # ------------------------------------------------------------------
    # CSV reading utilities
    # ------------------------------------------------------------------

    def _read_csv(
        self,
        file_path: Path,
        required: list[str] | None = None,
    ) -> pd.DataFrame:
        resolved = self._resolve_csv_path(file_path)
        df = pd.read_csv(resolved)
        df.columns = [str(c).strip().lower() for c in df.columns]
        if required:
            self._check_columns(df, required, resolved)
        return df

    def _read_csv_chunks(
        self,
        file_path: Path,
        required: list[str] | None = None,
    ) -> Iterator[pd.DataFrame]:
        resolved = self._resolve_csv_path(file_path)
        for chunk in pd.read_csv(resolved, chunksize=_CHUNK_SIZE):
            chunk.columns = [str(c).strip().lower() for c in chunk.columns]
            if required:
                self._check_columns(chunk, required, resolved)
            yield chunk

    @staticmethod
    def _resolve_csv_path(file_path: Path) -> Path:
        if file_path.exists():
            return file_path
        parent = file_path.parent
        if not parent.is_dir():
            raise FileNotFoundError(f"Directory not found: {parent}")
        target = file_path.name.lower()
        for candidate in parent.iterdir():
            if candidate.is_file() and candidate.name.lower() == target:
                return candidate
        raise FileNotFoundError(f"Required file not found: {file_path}")

    @staticmethod
    def _check_columns(
        df: pd.DataFrame, required: list[str], path: Path
    ) -> None:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing columns in {path.name}: {missing}. "
                f"Available: {list(df.columns)}"
            )
