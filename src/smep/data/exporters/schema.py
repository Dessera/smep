"""Schema definitions for the MIMIC-III base table export."""

from typing import Any

# ---------------------------------------------------------------------------
# Aggregation statistics applied to every temporal feature
# ---------------------------------------------------------------------------
AGG_STATS = ("min", "max", "mean")

# ---------------------------------------------------------------------------
# Vital-sign features extracted from CHARTEVENTS
# ---------------------------------------------------------------------------
CHARTEVENTS_FEATURES: dict[str, dict[str, Any]] = {
    "heartrate": {
        "itemids": [211, 220045],
        "valid_min": 20,
        "valid_max": 250,
        "unit": "bpm",
    },
    "sysbp": {
        "itemids": [51, 442, 455, 6701, 220179, 220050],
        "valid_min": 40,
        "valid_max": 300,
        "unit": "mmHg",
    },
    "diasbp": {
        "itemids": [8368, 8440, 8441, 8555, 220180, 220051],
        "valid_min": 20,
        "valid_max": 200,
        "unit": "mmHg",
    },
    "meanbp": {
        "itemids": [456, 52, 6702, 443, 220052, 220181, 225312],
        "valid_min": 25,
        "valid_max": 220,
        "unit": "mmHg",
    },
    "tempc": {
        "itemids": [223762, 676, 223761, 678],
        "valid_min": 30,
        "valid_max": 43,
        "unit": "°C",
        # Fahrenheit item IDs that require conversion to Celsius
        "fahrenheit_itemids": {223761, 678},
    },
    "resprate": {
        "itemids": [618, 615, 220210, 224690],
        "valid_min": 1,
        "valid_max": 80,
        "unit": "breaths/min",
    },
    "spo2": {
        "itemids": [646, 220277],
        "valid_min": 0,
        "valid_max": 100,
        "unit": "%",
    },
}

# ---------------------------------------------------------------------------
# Lab-test features extracted from LABEVENTS
# ---------------------------------------------------------------------------
LABEVENTS_FEATURES: dict[str, dict[str, Any]] = {
    "aniongap": {
        "itemids": [50868],
        "valid_min": 3,
        "valid_max": 40,
        "unit": "mEq/L",
    },
    "bicarbonate": {
        "itemids": [50882],
        "valid_min": 5,
        "valid_max": 50,
        "unit": "mEq/L",
    },
    "bun": {
        "itemids": [51006],
        "valid_min": 1,
        "valid_max": 250,
        "unit": "mg/dL",
    },
    "chloride": {
        "itemids": [50806, 50902],
        "valid_min": 60,
        "valid_max": 150,
        "unit": "mEq/L",
    },
    "creatinine": {
        "itemids": [50912],
        "valid_min": 0.1,
        "valid_max": 20,
        "unit": "mg/dL",
    },
    "glucose": {
        "itemids": [50809, 50931],
        "valid_min": 20,
        "valid_max": 1000,
        "unit": "mg/dL",
    },
    "hematocrit": {
        "itemids": [51221],
        "valid_min": 10,
        "valid_max": 65,
        "unit": "%",
    },
    "hemoglobin": {
        "itemids": [51222],
        "valid_min": 1,
        "valid_max": 25,
        "unit": "g/dL",
    },
    "inr": {
        "itemids": [51237],
        "valid_min": 0.5,
        "valid_max": 20,
        "unit": "ratio",
    },
    "lactate": {
        "itemids": [50813],
        "valid_min": 0.1,
        "valid_max": 30,
        "unit": "mmol/L",
    },
    "platelet": {
        "itemids": [51265],
        "valid_min": 1,
        "valid_max": 1500,
        "unit": "K/µL",
    },
    "potassium": {
        "itemids": [50822, 50971],
        "valid_min": 1,
        "valid_max": 10,
        "unit": "mEq/L",
    },
    "sodium": {
        "itemids": [50824, 50983],
        "valid_min": 100,
        "valid_max": 180,
        "unit": "mEq/L",
    },
    "wbc": {
        "itemids": [51300, 51301],
        "valid_min": 0.1,
        "valid_max": 200,
        "unit": "K/µL",
    },
}

# ---------------------------------------------------------------------------
# Severity score and treatment columns
# (Previously placeholders, now computed from raw data.)
# ---------------------------------------------------------------------------
SCORE_COLUMNS: list[str] = [
    "sofa",
    "lods",
    "sirs",
    "qsofa",
]

TREATMENT_COLUMNS: list[str] = [
    "vent",
    "rrt",
    "urineoutput",
    "colloid_bolus",
    "crystalloid_bolus",
]

# ---------------------------------------------------------------------------
# Infection timeline columns
# ---------------------------------------------------------------------------
INFECTION_COLUMNS: list[str] = [
    "suspected_infection_time_poe",
    "suspected_infection_time_poe_days",
    "antibiotic_time_poe",
    "blood_culture_time",
    "blood_culture_positive",
    "positiveculture_poe",
    "specimen_poe",
]

# ---------------------------------------------------------------------------
# Sepsis diagnosis criteria columns
# ---------------------------------------------------------------------------
SEPSIS_CRITERIA_COLUMNS: list[str] = [
    "sepsis_angus",
    "sepsis_martin",
    "sepsis_explicit",
    "severe_sepsis_explicit",
    "septic_shock_explicit",
    "sepsis_nqf",
    "sepsis_cdc",
    "sepsis_cdc_simple",
]

# ---------------------------------------------------------------------------
# Comorbidity & baseline columns
# ---------------------------------------------------------------------------
COMORBIDITY_COLUMNS: list[str] = [
    "elixhauser_hospital",
    "diabetes",
    "metastatic_cancer",
    "first_service",
    "dbsource",
]

# ---------------------------------------------------------------------------
# Canonical output column order
# ---------------------------------------------------------------------------

_TEMPORAL_COLUMNS: list[str] = []
for _feat in sorted(LABEVENTS_FEATURES):
    for _stat in AGG_STATS:
        _TEMPORAL_COLUMNS.append(f"{_feat}_{_stat}")
for _feat in sorted(CHARTEVENTS_FEATURES):
    for _stat in AGG_STATS:
        _TEMPORAL_COLUMNS.append(f"{_feat}_{_stat}")

OUTPUT_COLUMNS: list[str] = [
    # Primary keys
    "subject_id",
    "hadm_id",
    "icustay_id",
    # Time
    "intime",
    "outtime",
    # Demographics
    "age",
    "gender",
    "ethnicity",
    # Labels
    "hospital_expire_flag",
    "thirtyday_expire_flag",
    # Length of stay
    "icu_los",
    "hosp_los",
    # Severity scores
    "sofa",
    "lods",
    "sirs",
    "qsofa",
    # Lab & vital aggregates
    *_TEMPORAL_COLUMNS,
    # Treatment features
    "vent",
    "rrt",
    "urineoutput",
    "colloid_bolus",
    "crystalloid_bolus",
    # Infection timeline
    *INFECTION_COLUMNS,
    # Sepsis diagnosis criteria
    *SEPSIS_CRITERIA_COLUMNS,
    # Comorbidity & baseline
    *COMORBIDITY_COLUMNS,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_item_lookup(
    features: dict[str, dict[str, Any]],
) -> dict[int, str]:
    """Map every MIMIC item ID to its canonical feature name."""
    lookup: dict[int, str] = {}
    for feat_name, spec in features.items():
        for item_id in spec["itemids"]:
            lookup[item_id] = feat_name
    return lookup


def build_schema_dict(schema_version: str) -> dict[str, Any]:
    """Return a JSON-serialisable schema description."""
    fields: dict[str, dict[str, str]] = {}

    _static_meta: dict[str, tuple[str, str, str]] = {
        "subject_id": ("integer", "", "Patient unique ID"),
        "hadm_id": ("integer", "", "Hospital admission ID"),
        "icustay_id": ("integer", "", "ICU stay ID"),
        "intime": ("datetime", "", "ICU admission time"),
        "outtime": ("datetime", "", "ICU discharge time"),
        "age": ("float", "years", "Patient age at admission"),
        "gender": ("string", "", "Gender (M/F)"),
        "ethnicity": ("string", "", "Raw ethnicity text"),
        "hospital_expire_flag": (
            "binary",
            "",
            "In-hospital mortality",
        ),
        "thirtyday_expire_flag": (
            "binary",
            "",
            "30-day mortality from hospital discharge",
        ),
        "icu_los": ("float", "days", "ICU length of stay"),
        "hosp_los": ("float", "days", "Hospital length of stay"),
    }

    for col, (dtype, unit, desc) in _static_meta.items():
        fields[col] = {
            "type": dtype,
            "unit": unit,
            "description": desc,
            "status": "implemented",
        }

    _score_meta: dict[str, tuple[str, str, str]] = {
        "sofa": (
            "integer",
            "score",
            "Sequential Organ Failure Assessment (0-24)",
        ),
        "lods": (
            "integer",
            "score",
            "Logistic Organ Dysfunction Score (0-22)",
        ),
        "sirs": (
            "integer",
            "score",
            "Systemic Inflammatory Response Syndrome criteria (0-4)",
        ),
        "qsofa": (
            "integer",
            "score",
            "Quick SOFA score (0-3)",
        ),
        "vent": (
            "binary",
            "",
            "Mechanical ventilation within window (0/1)",
        ),
        "rrt": (
            "binary",
            "",
            "Renal replacement therapy within window (0/1)",
        ),
        "urineoutput": (
            "float",
            "mL",
            "Total urine output within window",
        ),
        "colloid_bolus": (
            "float",
            "mL",
            "Total colloid bolus within window (NaN if none)",
        ),
        "crystalloid_bolus": (
            "float",
            "mL",
            "Total crystalloid bolus within window (NaN if none)",
        ),
    }
    for col, (dtype, unit, desc) in _score_meta.items():
        fields[col] = {
            "type": dtype,
            "unit": unit,
            "description": desc,
            "status": "implemented",
        }

    # Infection timeline
    _infection_meta: dict[str, tuple[str, str, str]] = {
        "suspected_infection_time_poe": (
            "datetime",
            "",
            "Suspected infection time (Sepsis-3 definition)",
        ),
        "suspected_infection_time_poe_days": (
            "float",
            "days",
            "Suspected infection time relative to ICU admission",
        ),
        "antibiotic_time_poe": (
            "datetime",
            "",
            "Earliest antibiotic prescription time",
        ),
        "blood_culture_time": (
            "datetime",
            "",
            "Earliest blood culture specimen time",
        ),
        "blood_culture_positive": (
            "binary",
            "",
            "Any blood culture positive (0/1)",
        ),
        "positiveculture_poe": (
            "binary",
            "",
            "Any culture positive (0/1)",
        ),
        "specimen_poe": (
            "string",
            "",
            "Specimen type of earliest culture",
        ),
    }
    for col, (dtype, unit, desc) in _infection_meta.items():
        fields[col] = {
            "type": dtype,
            "unit": unit,
            "description": desc,
            "status": "implemented",
        }

    # Sepsis diagnosis criteria
    _sepsis_meta: dict[str, tuple[str, str, str]] = {
        "sepsis_angus": (
            "binary",
            "",
            "Angus et al. (2001) sepsis criteria",
        ),
        "sepsis_martin": (
            "binary",
            "",
            "Martin et al. (2003) septicemia criteria",
        ),
        "sepsis_explicit": (
            "binary",
            "",
            "Explicit ICD-9 sepsis code (995.91)",
        ),
        "severe_sepsis_explicit": (
            "binary",
            "",
            "Explicit ICD-9 severe sepsis (995.92)",
        ),
        "septic_shock_explicit": (
            "binary",
            "",
            "Explicit ICD-9 septic shock (785.52)",
        ),
        "sepsis_nqf": (
            "binary",
            "",
            "National Quality Forum sepsis definition",
        ),
        "sepsis_cdc": (
            "binary",
            "",
            "CDC surveillance sepsis definition",
        ),
        "sepsis_cdc_simple": (
            "binary",
            "",
            "Simplified CDC sepsis definition",
        ),
    }
    for col, (dtype, unit, desc) in _sepsis_meta.items():
        fields[col] = {
            "type": dtype,
            "unit": unit,
            "description": desc,
            "status": "implemented",
        }

    # Comorbidity & baseline
    _comorbidity_meta: dict[str, tuple[str, str, str]] = {
        "elixhauser_hospital": (
            "integer",
            "score",
            "van Walraven weighted Elixhauser comorbidity score",
        ),
        "diabetes": (
            "binary",
            "",
            "Diabetes indicator (uncomplicated or complicated)",
        ),
        "metastatic_cancer": (
            "binary",
            "",
            "Metastatic cancer indicator",
        ),
        "first_service": (
            "string",
            "",
            "First service line on admission",
        ),
        "dbsource": (
            "string",
            "",
            "Data source (CareVue/MetaVision)",
        ),
    }
    for col, (dtype, unit, desc) in _comorbidity_meta.items():
        fields[col] = {
            "type": dtype,
            "unit": unit,
            "description": desc,
            "status": "implemented",
        }

    all_features = {**LABEVENTS_FEATURES, **CHARTEVENTS_FEATURES}
    for feat_name, spec in all_features.items():
        unit = spec.get("unit", "")
        for stat in AGG_STATS:
            col = f"{feat_name}_{stat}"
            fields[col] = {
                "type": "float",
                "unit": unit,
                "description": (
                    f"{feat_name} {stat} within aggregation window"
                ),
                "status": "implemented",
            }

    return {
        "version": schema_version,
        "sample_granularity": "ICU stay",
        "primary_key": ["subject_id", "hadm_id", "icustay_id"],
        "fields": fields,
    }
