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
    "qsofa_gcs_score",
    "qsofa_resprate_score",
    "qsofa_sysbp_score",
}


# ---------------------------------------------------------------------------
# Default columns to drop (complement of DEFAULT_KEEP_COLUMNS)
# ---------------------------------------------------------------------------
DEFAULT_DROP_COLUMNS: set[str] = {
    "ethnicity",
    "specimen_poe",
    "first_service",
    "creatinine_min",
    "wbc_mean",
    "resprate_max",
    "meanbp_max",
    "chloride_mean",
    "sepsis_cdc_simple",
    "sepsis_nqf",
    "sepsis_martin",
    "septic_shock_explicit",
    "rrt",
    "diabetes",
    "blood_culture_positive",
    "meanbp_min",
    "sepsis_cdc",
    "hematocrit_min",
    "resprate_min",
    "sodium_min",
    "diasbp_max",
    "spo2_max",
    "bun_max",
    "sodium_mean",
    "spo2_min",
    "glucose_min",
    "severe_sepsis_explicit",
    "sysbp_max",
    "sirs",
    "sysbp_min",
    "chloride_min",
    "glucose_mean",
    "aniongap_max",
    "potassium_mean",
    "inr_mean",
    "hematocrit_mean",
    "hemoglobin_mean",
    "bicarbonate_mean",
    "inr_min",
    "hemoglobin_max",
    "resprate_mean",
    "bicarbonate_max",
    "crystalloid_bolus",
    "dbsource",
    "sodium_max",
    "inr_max",
}
