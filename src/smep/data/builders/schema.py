"""Column classification constants for the dataset builder."""

# ---------------------------------------------------------------------------
# Columns that are always excluded from the feature matrix
# ---------------------------------------------------------------------------
EXCLUDE_COLUMNS: set[str] = {
    # Primary keys
    "subject_id",
    "hadm_id",
    "icustay_id",
    # Timestamps
    "intime",
    "outtime",
    "suspected_infection_time_poe",
    "antibiotic_time_poe",
    "blood_culture_time",
    # Day-difference (information already in other features)
    "suspected_infection_time_poe_days",
    # Outcome labels (excluded to prevent leakage; the chosen label
    # is separated into y automatically)
    "hospital_expire_flag",
    "thirtyday_expire_flag",
}

# ---------------------------------------------------------------------------
# Categorical columns that need one-hot encoding
# ---------------------------------------------------------------------------
CATEGORICAL_COLUMNS: set[str] = {
    "gender",
    "ethnicity",
    "specimen_poe",
    "first_service",
    "dbsource",
}

# ---------------------------------------------------------------------------
# Binary 0/1 columns – no scaling, impute with most-frequent
# ---------------------------------------------------------------------------
BINARY_COLUMNS: set[str] = {
    "vent",
    "rrt",
    "blood_culture_positive",
    "positiveculture_poe",
    "sepsis_angus",
    "sepsis_martin",
    "sepsis_explicit",
    "severe_sepsis_explicit",
    "septic_shock_explicit",
    "sepsis_nqf",
    "sepsis_cdc",
    "sepsis_cdc_simple",
    "diabetes",
    "metastatic_cancer",
    "tumor",
    "qsofa_gcs_score",
    "qsofa_resprate_score",
    "qsofa_sysbp_score",
}


# ---------------------------------------------------------------------------
# Default columns to drop (complement of DEFAULT_KEEP_COLUMNS)
# ---------------------------------------------------------------------------
DEFAULT_DROP_COLUMNS: set[str] = {
    # Excluded Permanently
    "sofa",
    "qsofa",
    "lods",
    "sirs",
    "hosp_los",
    "icu_los",
    # Others
    # One-hot labels
    "ethnicity",
    "specimen_poe",
    "first_service",
    "dbsource",
    "gender",
    "sepsis_cdc_simple",
    "sepsis_nqf",
    "sepsis_martin",
    "sepsis_explicit",
    "sepsis_angus",
    "sepsis_cdc",
    "septic_shock_explicit",
    "severe_sepsis_explicit",
    # waiting list
    "urineoutput_per_kg_per_h",
    "inr_min",
    "heartrate_min",
    "height_min",
    "sodium_mean",
    "resprate_max",
    "glucose_min",
    "spo2_max",
    "paco2_mean",
    "pt_min",
    "hematocrit_min",
    "fio2_max",
    "sysbp_max",
    "bmi",
    "resprate_min",
    "bicarbonate_mean",
    "gcs_verbal_max",
    "height_max",
    "meanbp_min",
    "weight_max",
    "gcs_verbal_mean",
    "hemoglobin_max",
    "pt_max",
    "paco2_min",
    "hemoglobin_min",
    "heartrate_mean",
    "wbc_min",
    "glucose_mean",
    "sysbp_min",
    "pao2_mean",
    "weight_min",
    "creatinine_max",
    "aniongap_max",
    "sodium_max",
    "platelet_min",
    "chloride_mean",
    "positiveculture_poe",
    "elixhauser_hospital",
    "inr_max",
    "gcs_eye_max",
    "bicarbonate_min",
    "diabetes",
    "height_mean",
    "rrt",
    "tumor",
    "vent",
    "weight_mean",
    "gcs_eye_mean",
    "fio2_min",
    "potassium_max",
    "bicarbonate_max",
    "bun_max",
    "chloride_min",
    "lactate_mean",
    "inr_mean",
    "tempc_min",
    "bilirubin_max",
    "bilirubin_min",
    "bilirubin_mean",
    "gcs_verbal_min",
    "spo2_mean",
    "diasbp_mean",
    "meanbp_max",
    "gcs_motor_max",
    "gcs_total_max",
    "gcs_motor_mean",
    "fio2_mean",
    "potassium_mean",
    "paco2_max",
    "diasbp_max",
    "hematocrit_mean",
    "gcs_eye_min",
}
