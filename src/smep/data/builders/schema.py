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
    "blood_culture_positive",
    "meanbp_min",
    "sepsis_cdc",
    "resprate_min",
    "sodium_min",
    "diasbp_max",
    "spo2_max",
    "sodium_mean",
    "spo2_min",
    "glucose_min",
    "severe_sepsis_explicit",
    "sysbp_max",
    "sirs",
    "sysbp_min",
    "chloride_min",
    "glucose_mean",
    "potassium_mean",
    "inr_mean",
    "hemoglobin_mean",
    "bicarbonate_mean",
    "inr_min",
    "hemoglobin_max",
    "bicarbonate_max",
    "crystalloid_bolus",
    "dbsource",
    "sodium_max",
    "inr_max",
    "sofa",
    "lods",
    "qsofa",
    "hosp_los",
    "icu_los",
    "gender",
    "gcs_motor_max",
    "pt_mean",
    "gcs_motor_min",
    "gcs_verbal_max",
    "fio2_max",
    "gcs_verbal_min",
    "paco2_min",
    "pao2_mean",
    "fio2_min",
    "pt_min",
    "pt_max",
    "fio2_mean",
    "wbc_min",
    "paco2_mean",
    "gcs_verbal_mean",
    "aniongap_min",
    "platelet_max",
    "gcs_eye_max",
    "gcs_motor_mean",
    "bicarbonate_min",
    "tempc_min",
    "pao2_min",
    "pao2_max",
    "height_min",
    "height_max",
    "height_mean",
    "weight_min",
    "weight_max",
    "weight_mean",
    "metastatic_cancer",
    "lactate_mean",
    "platelet_mean",
    "sepsis_explicit",
    "chloride_max",
    "gcs_eye_mean",
    "bun_mean",
    "tempc_mean",
    "lactate_min",
    "aniongap_mean",
    "meanbp_mean",
    "bilirubin_min",
    "potassium_min",
    "creatinine_mean",
    "bun_max",
    "tempc_max",
    "urineoutput_per_kg_per_h",
    "paco2_max",
    "heartrate_max",
    "bun_min",
    "creatinine_max",
    "positiveculture_poe",
    "potassium_max",
    "gcs_eye_min",
    "bilirubin_mean",
    "diasbp_min",
    "diasbp_mean",
    "hematocrit_mean",
    "vent",
    "heartrate_min",
    "hematocrit_max",
    "sepsis_angus",
    "elixhauser_hospital",
    "glucose_max",
}

# bmi
# tumor
# mean blood pressure, blood temprature, blood urea nitrogen
# albumin
# gcs
