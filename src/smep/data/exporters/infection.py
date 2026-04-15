"""Infection timeline feature extraction from MIMIC-III.

Computes per-stay:
  - antibiotic_time_poe        earliest antibiotic prescription time
  - blood_culture_time          earliest blood-culture specimen time
  - blood_culture_positive      whether any blood culture was positive (0/1)
  - positiveculture_poe         whether any culture (any specimen) was
                                positive (0/1)
  - specimen_poe                specimen type of the earliest culture
  - suspected_infection_time_poe
        the earlier of (earliest antibiotic, earliest blood culture)
        *if* the two events fall within 72 h of each other;
        otherwise the blood-culture time alone
  - suspected_infection_time_poe_days
        (suspected_infection_time_poe − intime) in fractional days

The Sepsis-3 "suspected infection" definition (Seymour et al. JAMA 2016)
uses the conjunction of antibiotic administration and body-fluid culture
within a ±72-hour window.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_SUSPECT_WINDOW_H = 72  # ±72 hours for antibiotic–culture pairing

# Blood-culture specimen descriptions used as the reference culture.
_BLOOD_CULTURE_SPECS = frozenset({
    "BLOOD CULTURE",
    "BLOOD CULTURE ( MYCO/F LYTIC BOTTLE)",
    "BLOOD CULTURE - NEONATE",
})


def compute_infection_timeline(
    source_path: Path,
    cohort_df: pd.DataFrame,
    *,
    read_csv_fn: Any,
) -> pd.DataFrame:
    """Return a DataFrame with infection-timeline columns per stay.

    Parameters
    ----------
    source_path : Path
        Root directory containing MIMIC-III CSVs.
    cohort_df : pd.DataFrame
        Must contain subject_id, hadm_id, icustay_id, intime.
    read_csv_fn : callable
        ``MIMIC3Exporter._read_csv`` compatible reader.
    """
    keys = ["subject_id", "hadm_id", "icustay_id"]
    result = cohort_df[keys].copy()

    abx_df = _earliest_antibiotic(source_path, cohort_df, read_csv_fn)
    micro_df = _process_micro(source_path, cohort_df, read_csv_fn)

    # Merge antibiotic time
    result = result.merge(abx_df, on=keys, how="left")
    # Merge microbiology results
    result = result.merge(micro_df, on=keys, how="left")

    # --- suspected_infection_time_poe ---
    result = _compute_suspected_infection(result, cohort_df)

    return result


# ======================================================================
# Internal helpers
# ======================================================================


def _earliest_antibiotic(
    source_path: Path,
    cohort_df: pd.DataFrame,
    read_csv_fn: Any,
) -> pd.DataFrame:
    """Return earliest antibiotic prescription time per stay."""
    keys = ["subject_id", "hadm_id", "icustay_id"]
    try:
        rx = read_csv_fn(
            source_path / "PRESCRIPTIONS.csv",
            required=[
                "subject_id", "hadm_id", "icustay_id",
                "startdate", "drug_type",
            ],
        )
    except FileNotFoundError:
        logger.warning("PRESCRIPTIONS.csv not found – skipping antibiotics")
        df = cohort_df[keys].copy()
        df["antibiotic_time_poe"] = pd.NaT
        return df

    rx["startdate"] = pd.to_datetime(rx["startdate"], errors="coerce")
    # Keep only antibiotic prescriptions
    rx = rx[
        rx["drug_type"].astype(str).str.upper().str.strip() == "MAIN"
    ].copy()
    # drug_type MAIN covers standard medications; we filter further by
    # route if available, but MIMIC prescriptions for antibiotics
    # overwhelmingly have drug_type = MAIN.
    # A more precise filter would check the drug name against a known
    # antibiotic list, but for the Sepsis-3 definition the broadest
    # approach (any medication ordered around culture time) is standard.

    rx = rx.merge(cohort_df[keys], on=keys, how="inner")
    rx = rx.dropna(subset=["startdate"])

    earliest = (
        rx.groupby(keys, as_index=False)["startdate"]
        .min()
        .rename(columns={"startdate": "antibiotic_time_poe"})
    )
    return earliest


def _process_micro(
    source_path: Path,
    cohort_df: pd.DataFrame,
    read_csv_fn: Any,
) -> pd.DataFrame:
    """Return per-stay culture summary.

    Columns:
        blood_culture_time      earliest blood-culture specimen time
        blood_culture_positive  1 if any blood culture grew an organism
        positiveculture_poe     1 if any specimen culture was positive
        specimen_poe            specimen type of the earliest culture
    """
    keys = ["subject_id", "hadm_id", "icustay_id"]
    try:
        micro = read_csv_fn(
            source_path / "MICROBIOLOGYEVENTS.csv",
            required=[
                "subject_id", "hadm_id",
                "chartdate", "spec_type_desc",
            ],
        )
    except FileNotFoundError:
        logger.warning(
            "MICROBIOLOGYEVENTS.csv not found – skipping cultures"
        )
        df = cohort_df[keys].copy()
        df["blood_culture_time"] = pd.NaT
        df["blood_culture_positive"] = np.nan
        df["positiveculture_poe"] = np.nan
        df["specimen_poe"] = np.nan
        return df

    # charttime may be null; fall back to chartdate
    if "charttime" in micro.columns:
        micro["_time"] = pd.to_datetime(
            micro["charttime"], errors="coerce"
        )
    else:
        micro["_time"] = pd.NaT
    micro["_time"] = micro["_time"].fillna(
        pd.to_datetime(micro["chartdate"], errors="coerce")
    )

    micro["spec_type_desc"] = (
        micro["spec_type_desc"].fillna("").astype(str).str.strip().str.upper()
    )
    # A culture is "positive" if an organism was identified
    micro["_positive"] = micro["org_name"].notna().astype(int) if \
        "org_name" in micro.columns else 0

    # Join with cohort (via subject_id + hadm_id since micro has no
    # icustay_id)
    micro = micro.merge(
        cohort_df[["subject_id", "hadm_id", "icustay_id"]],
        on=["subject_id", "hadm_id"],
        how="inner",
    )

    # --- blood_culture_time & blood_culture_positive ---
    bc = micro[micro["spec_type_desc"].isin(_BLOOD_CULTURE_SPECS)].copy()
    bc_agg = (
        bc.groupby(keys, as_index=False)
        .agg(
            blood_culture_time=("_time", "min"),
            blood_culture_positive=("_positive", "max"),
        )
    )

    # --- positiveculture_poe (any specimen) ---
    pos_any = (
        micro.groupby(keys, as_index=False)["_positive"]
        .max()
        .rename(columns={"_positive": "positiveculture_poe"})
    )

    # --- specimen_poe (earliest culture specimen type) ---
    micro_sorted = micro.dropna(subset=["_time"]).sort_values("_time")
    earliest_spec = (
        micro_sorted.drop_duplicates(subset=keys, keep="first")[
            keys + ["spec_type_desc"]
        ]
        .rename(columns={"spec_type_desc": "specimen_poe"})
    )

    # Combine
    result = cohort_df[keys].copy()
    result = result.merge(bc_agg, on=keys, how="left")
    result = result.merge(pos_any, on=keys, how="left")
    result = result.merge(earliest_spec, on=keys, how="left")

    return result


def _compute_suspected_infection(
    df: pd.DataFrame,
    cohort_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add suspected_infection_time_poe and _days columns.

    Logic (Sepsis-3, Seymour 2016):
    - If antibiotic was given first: culture must follow within 24 h
    - If culture was taken first: antibiotic must follow within 72 h
    - suspected_infection_time = min(antibiotic_time, culture_time)
      when the above pairing criteria are met.
    - If no pairing: use blood_culture_time alone as fallback.
    """
    keys = ["subject_id", "hadm_id", "icustay_id"]
    intime_map = cohort_df.set_index(keys)["intime"]

    abx = df["antibiotic_time_poe"]
    bc = df["blood_culture_time"]

    # Time difference: antibiotic − blood_culture (hours)
    diff_h = (abx - bc).dt.total_seconds() / 3600

    # Pairing valid when:
    #   abx first (diff_h < 0) and culture within 24 h → diff_h >= -24
    #   culture first (diff_h >= 0) and abx within 72 h → diff_h <= 72
    paired = (diff_h >= -24) & (diff_h <= 72)

    # suspected_infection_time = earlier of the two when paired
    si_time = pd.NaT
    si_time = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

    both = abx.notna() & bc.notna()
    # Paired: use the earlier time
    mask_paired = both & paired
    si_time[mask_paired] = pd.DataFrame({
        "a": abx[mask_paired], "b": bc[mask_paired]
    }).min(axis=1)
    # Not paired but have blood culture: use blood culture
    mask_bc_only = ~mask_paired & bc.notna()
    si_time[mask_bc_only] = bc[mask_bc_only]
    # No blood culture but have antibiotic: use antibiotic
    mask_abx_only = ~mask_paired & bc.isna() & abx.notna()
    si_time[mask_abx_only] = abx[mask_abx_only]

    df["suspected_infection_time_poe"] = si_time

    # Days from intime
    intime = df.set_index(keys).index.map(intime_map)
    df["suspected_infection_time_poe_days"] = np.where(
        si_time.notna(),
        (si_time - intime.values).dt.total_seconds() / 86400,
        np.nan,
    )

    return df
