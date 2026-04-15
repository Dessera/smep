"""Sepsis diagnosis criteria from ICD-9 codes (MIMIC-III).

Implements the following binary classification criteria:
  - sepsis_angus      Angus et al. (2001) — infection + organ dysfunction
  - sepsis_martin     Martin et al. (2003) — sepsis ICD codes
  - sepsis_explicit   Explicit ICD-9 codes for sepsis (995.91)
  - severe_sepsis_explicit  Explicit ICD-9 for severe sepsis (995.92)
  - septic_shock_explicit   Explicit ICD-9 for septic shock (785.52)
  - sepsis_nqf        National Quality Forum (NQF) definition
  - sepsis_cdc        CDC surveillance definition
  - sepsis_cdc_simple Simplified CDC definition

All criteria operate on DIAGNOSES_ICD (ICD-9 codes) per hadm_id.

References
----------
- Angus DC et al. Crit Care Med 2001;29(7):1303-10
- Martin GS et al. N Engl J Med 2003;348(16):1546-54
- Iwashyna TJ et al. J Am Geriatr Soc 2014;62(1):40-46
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ======================================================================
# ICD-9 code sets
# ======================================================================

# --- Explicit sepsis codes ---
_SEPSIS_EXPLICIT = {"99591"}  # Sepsis
_SEVERE_SEPSIS_EXPLICIT = {"99592"}  # Severe sepsis
_SEPTIC_SHOCK_EXPLICIT = {"78552"}  # Septic shock

# --- Martin (2003) septicemia codes ---
_MARTIN_CODES = {
    "0031",  # Salmonella septicemia
    "0202",  # Septicemic plague
    "0223",  # Anthrax septicemia
    "0362",  # Meningococcemia
    "0380",  # Streptococcal septicemia
    "03810",  # Staphylococcal septicemia, unspecified
    "03811",  # Methicillin susceptible Staph aureus septicemia
    "03812",  # Methicillin resistant Staph aureus septicemia
    "03819",  # Other staphylococcal septicemia
    "0382",  # Pneumococcal septicemia
    "0383",  # Anaerobic septicemia
    "03840",  # Gram-negative septicemia, unspecified
    "03841",  # H. influenzae septicemia
    "03842",  # E. coli septicemia
    "03843",  # Pseudomonas septicemia
    "03844",  # Serratia septicemia
    "03849",  # Other gram-negative septicemia
    "0388",  # Other specified septicemias
    "0389",  # Unspecified septicemia
    "0545",  # Herpes simplex septicemia
    "44901",  # Infected aortic aneurysm; not classic but used in Martin
    "77181",  # Septicemia of newborn
    "99591",  # Sepsis
    "99592",  # Severe sepsis
    "78552",  # Septic shock
}

# --- Angus (2001) — Infection codes (by prefix) ---
# These are ICD-9 code *prefixes*.  An admission qualifies as having an
# infection when any of its ICD-9 codes starts with one of these.
_ANGUS_INFECTION_PREFIXES = (
    "001",
    "002",
    "003",
    "004",
    "005",
    "008",
    "009",
    "010",
    "011",
    "012",
    "013",
    "014",
    "015",
    "016",
    "017",
    "018",
    "020",
    "021",
    "022",
    "023",
    "024",
    "025",
    "026",
    "027",
    "030",
    "031",
    "032",
    "033",
    "034",
    "035",
    "036",
    "037",
    "038",
    "039",
    "040",
    "041",
    "090",
    "091",
    "092",
    "093",
    "094",
    "095",
    "096",
    "097",
    "098",
    "099",
    "100",
    "101",
    "102",
    "103",
    "104",
    "110",
    "111",
    "112",
    "114",
    "115",
    "116",
    "117",
    "118",
    "320",
    "322",
    "324",
    "325",
    "420",
    "421",
    "451",
    "461",
    "462",
    "463",
    "464",
    "465",
    "466",
    "480",
    "481",
    "482",
    "483",
    "484",
    "485",
    "486",
    "487",
    "488",
    "490",
    "491",
    "494",
    "510",
    "513",
    "540",
    "541",
    "542",
    "566",
    "567",
    "569",  # specific: 56961
    "572",  # specific: 5720, 5721
    "575",  # specific: 5750, 57510
    "590",
    "595",
    "597",
    "599",  # specific: 5990
    "601",
    "614",
    "615",
    "616",
    "670",
    "681",
    "682",
    "683",
    "686",
    "711",  # specific: 7110x
    "730",
    "790",  # specific: 7907
    "996",  # specific: 99631, 99632, 99639, 99662, 99667
    "998",  # specific: 99851, 99859
    "999",  # specific: 99931, 99932, 99933, 99939
)

# --- Angus (2001) — Organ dysfunction codes (by prefix) ---
_ANGUS_ORGAN_DYSFUNCTION_PREFIXES = (
    # Cardiovascular
    "785.5",  # shock
    "458",  # hypotension
    # Renal
    "584",  # acute kidney failure
    # Hepatic
    "570",  # acute hepatic failure
    # Hematologic
    "286",  # coagulation defects
    # Metabolic / CNS
    "348.1",  # anoxic brain damage
    "293",  # delirium due to conditions classified elsewhere
    "348.3",  # encephalopathy
    # Respiratory
    "518.81",  # acute respiratory failure
    "518.82",  # other pulmonary insufficiency
    "518.84",  # acute and chronic respiratory failure
    "518.85",  # critical illness neuromyopathy
    "799.1",  # respiratory arrest
)

# --- NQF definition ---
# NQF = explicit sepsis codes + Angus organ dysfunction
_NQF_INFECTION_CODES = (
    _SEPSIS_EXPLICIT | _SEVERE_SEPSIS_EXPLICIT | _SEPTIC_SHOCK_EXPLICIT
)

# --- CDC: infection + organ dysfunction (broader than Angus) ---
_CDC_ORGAN_PREFIXES = (
    # Same as Angus organ dysfunction but also includes:
    "995.92",  # severe sepsis
    "785.5",
    "458",
    "584",
    "570",
    "286",
    "348.1",
    "293",
    "348.3",
    "518.81",
    "518.82",
    "518.84",
    "518.85",
    "799.1",
)


# ======================================================================
# Public API
# ======================================================================


def compute_sepsis_criteria(
    source_path: Path,
    cohort_df: pd.DataFrame,
    *,
    read_csv_fn: Any,
) -> pd.DataFrame:
    """Return a DataFrame with sepsis-criteria binary columns per stay.

    Parameters
    ----------
    source_path : Path
        Root of MIMIC-III CSV directory.
    cohort_df : pd.DataFrame
        Must have subject_id, hadm_id, icustay_id.
    read_csv_fn : callable
        CSV reader (``MIMIC3Exporter._read_csv``).
    """
    keys = ["subject_id", "hadm_id", "icustay_id"]

    diagnoses = read_csv_fn(
        source_path / "DIAGNOSES_ICD.csv",
        required=["subject_id", "hadm_id", "icd9_code"],
    )
    # Normalize codes: remove dots, strip, uppercase
    diagnoses["icd9_code"] = (
        diagnoses["icd9_code"]
        .astype(str)
        .str.replace(".", "", regex=False)
        .str.strip()
        .str.upper()
    )

    # Keep only cohort admissions
    cohort_hadm = cohort_df[["subject_id", "hadm_id"]].drop_duplicates()
    diagnoses = diagnoses.merge(cohort_hadm, on=["subject_id", "hadm_id"])

    # Build per-hadm code sets
    code_sets = (
        diagnoses.groupby(["subject_id", "hadm_id"])["icd9_code"]
        .apply(frozenset)
        .reset_index()
        .rename(columns={"icd9_code": "_codes"})
    )

    result = cohort_df[keys].copy()
    result = result.merge(code_sets, on=["subject_id", "hadm_id"], how="left")
    result["_codes"] = result["_codes"].apply(
        lambda x: x if isinstance(x, frozenset) else frozenset()
    )

    # --- Explicit ---
    result["sepsis_explicit"] = result["_codes"].apply(
        lambda cs: int(bool(cs & _SEPSIS_EXPLICIT))
    )
    result["severe_sepsis_explicit"] = result["_codes"].apply(
        lambda cs: int(bool(cs & _SEVERE_SEPSIS_EXPLICIT))
    )
    result["septic_shock_explicit"] = result["_codes"].apply(
        lambda cs: int(bool(cs & _SEPTIC_SHOCK_EXPLICIT))
    )

    # --- Martin ---
    result["sepsis_martin"] = result["_codes"].apply(
        lambda cs: int(bool(cs & _MARTIN_CODES))
    )

    # --- Angus ---
    result["_has_infection"] = result["_codes"].apply(
        lambda cs: _has_prefix_match(cs, _ANGUS_INFECTION_PREFIXES)
    )
    result["_has_organ_dysf"] = result["_codes"].apply(
        lambda cs: _has_prefix_match(cs, _ANGUS_ORGAN_DYSFUNCTION_PREFIXES)
    )
    result["sepsis_angus"] = (
        (
            result["_has_infection"]
            | result["sepsis_explicit"]
            | result["severe_sepsis_explicit"]
            | result["septic_shock_explicit"]
        )
        & (
            result["_has_organ_dysf"]
            | result["severe_sepsis_explicit"]
            | result["septic_shock_explicit"]
        )
    ).astype(int)

    # --- NQF ---
    result["_nqf_infection"] = result["_codes"].apply(
        lambda cs: int(bool(cs & _NQF_INFECTION_CODES))
    )
    result["sepsis_nqf"] = (
        result["_nqf_infection"] & result["_has_organ_dysf"]
    ).astype(int)

    # --- CDC ---
    result["_cdc_organ"] = result["_codes"].apply(
        lambda cs: _has_prefix_match(cs, _CDC_ORGAN_PREFIXES)
    )
    result["sepsis_cdc"] = (
        result["_has_infection"] & result["_cdc_organ"]
    ).astype(int)

    # --- CDC Simple ---
    # Simplified: any explicit sepsis code OR (infection + organ dysf.)
    result["sepsis_cdc_simple"] = (
        result["sepsis_explicit"]
        | result["severe_sepsis_explicit"]
        | result["septic_shock_explicit"]
        | (result["_has_infection"] & result["_has_organ_dysf"])
    ).astype(int)

    # Drop internal columns
    drop_cols = [c for c in result.columns if c.startswith("_")]
    result = result.drop(columns=drop_cols)

    return result


# ======================================================================
# Helpers
# ======================================================================


def _has_prefix_match(
    code_set: frozenset[str],
    prefixes: tuple[str, ...],
) -> bool:
    """Return True if any code in the set starts with any prefix."""
    for code in code_set:
        for prefix in prefixes:
            if code.startswith(prefix.replace(".", "")):
                return True
    return False
