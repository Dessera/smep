"""Severity-score and treatment-feature computation for MIMIC-III.

Computes per-ICU-stay values for:
  - SOFA   (Sequential Organ Failure Assessment, 0-24)
  - SIRS   (Systemic Inflammatory Response Syndrome, 0-4)
  - qSOFA  (Quick SOFA, 0-3)
  - LODS   (Logistic Organ Dysfunction Score, 0-22)
  - vent   (mechanical ventilation flag, 0/1)
  - rrt    (renal replacement therapy flag, 0/1)
  - urineoutput  (total urine output in mL within window)
  - colloid_bolus   (total colloid bolus in mL within window)
  - crystalloid_bolus (total crystalloid bolus in mL within window)

All computations are scoped to the first *time_window_hours* from ICU
admission (``intime``).  The cohort DataFrame **must** contain at minimum:
``subject_id, hadm_id, icustay_id, intime``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ======================================================================
# MIMIC-III item-ID constants
# ======================================================================

# --- Vasopressor infusions (inputevents) ------------------------------
_DOPAMINE_IDS = [30043, 30307, 221662]
_DOBUTAMINE_IDS = [30042, 30306, 221653]
_NOREPINEPHRINE_IDS = [221906]
_EPINEPHRINE_IDS = [30044, 30119, 30309, 221289]
_VASOPRESSIN_IDS = [30051, 222315]
_PHENYLEPHRINE_IDS = [221749]

_ALL_VASOPRESSOR_IDS: set[int] = {
    *_DOPAMINE_IDS,
    *_DOBUTAMINE_IDS,
    *_NOREPINEPHRINE_IDS,
    *_EPINEPHRINE_IDS,
    *_VASOPRESSIN_IDS,
    *_PHENYLEPHRINE_IDS,
}

# --- Ventilation detection ---------------------------------------------
# CareVue chartevents (presence of any row → ventilated)
_VENT_CHART_IDS = [720, 722, 223849]
# MetaVision chartevents
_VENT_MECHVENT_ID = 226260  # "Mechanically Ventilated" (value=1)
# MetaVision procedureevents_mv
_VENT_PROC_IDS = [225792, 225794, 224385]

# --- RRT detection -----------------------------------------------------
_RRT_PROC_IDS = [
    225436,
    225441,
    225802,
    225803,
    225805,
    225809,
    225955,
]
_RRT_CHART_IDS = [152]  # "Dialysis Type" in CareVue
_RRT_OUTPUT_IDS = [40386, 40425, 40507, 40624, 40690, 41500, 41527]

# --- Urine output (outputevents) – common items -----------------------
_URINE_OUTPUT_IDS = [
    # CareVue
    40055,
    40056,
    40057,
    40061,
    40065,
    40069,
    40085,
    40094,
    40096,
    40288,
    40405,
    40428,
    40473,
    40651,
    40715,
    43175,
    43373,
    43431,
    43462,
    43519,
    43522,
    43537,
    43576,
    43589,
    43633,
    43811,
    43812,
    43856,
    43966,
    43987,
    44051,
    44080,
    44103,
    44132,
    44237,
    44313,
    44706,
    44911,
    44925,
    45304,
    45415,
    # MetaVision
    226559,
    226560,
    226561,
    226563,
    226564,
    226565,
    226566,
    226584,
    226627,
    226631,
    227489,
]

# --- Colloid bolus (inputevents) --------------------------------------
_COLLOID_IDS = [
    # CareVue
    30008,
    30009,
    30181,
    30102,
    30107,
    # MetaVision
    225174,
    225795,
    226365,
    226376,
]

# --- Crystalloid bolus (inputevents) ----------------------------------
_CRYSTALLOID_IDS = [
    # CareVue
    30018,
    30020,
    30021,
    30143,
    30160,
    30352,
    30353,
    # MetaVision
    220954,
    220955,
    225158,
    225159,
    225161,
    225823,
    225825,
    225827,
    225828,
    225944,
]

_CHUNK_SIZE = 100_000


# ======================================================================
# Public API
# ======================================================================


def compute_scores_and_treatments(
    source_path: Path,
    cohort_df: pd.DataFrame,
    time_window_hours: int,
    *,
    temporal_df: pd.DataFrame | None = None,
    read_csv_fn: Any,
    read_csv_chunks_fn: Any,
) -> pd.DataFrame:
    """Return a DataFrame indexed by (subject_id, hadm_id, icustay_id)
    with columns: sofa, sirs, qsofa, lods, vent, rrt, urineoutput,
    colloid_bolus, crystalloid_bolus.

    *temporal_df* supplies already-computed temporal aggregates
    (heartrate_max, tempc_min, pao2_min, fio2_max, gcs_total_min, etc.)
    from the Step 3 extraction pipeline.  GCS, FiO2, PaO2, PaCO2,
    bilirubin, and PT are now expected to come from *temporal_df*
    instead of being re-read from raw CSV files.
    """
    keys = ["subject_id", "hadm_id", "icustay_id"]
    result = cohort_df[keys].copy()
    window_td = pd.Timedelta(hours=time_window_hours)

    logger.info("  Gathering vasopressors from INPUTEVENTS …")
    vaso_flags = _gather_vasopressor_flags(
        source_path, cohort_df, window_td, read_csv_chunks_fn
    )

    logger.info("  Gathering urine output from OUTPUTEVENTS …")
    urine_df = _gather_urine_output(
        source_path, cohort_df, window_td, read_csv_chunks_fn
    )

    # --- Pre-aggregate per-stay extremes ---
    stay_data = _build_stay_data(
        cohort_df,
        vaso_flags,
        urine_df,
        temporal_df=temporal_df,
    )

    # --- Compute scores ---
    logger.info("  Computing SOFA …")
    result["sofa"] = result.apply(
        lambda r: _sofa(stay_data, r["icustay_id"]), axis=1
    )

    logger.info("  Computing SIRS …")
    result["sirs"] = result.apply(
        lambda r: _sirs(stay_data, r["icustay_id"]), axis=1
    )

    logger.info("  Computing qSOFA …")
    result["qsofa"] = result.apply(
        lambda r: _qsofa(stay_data, r["icustay_id"]), axis=1
    )

    logger.info("  Computing LODS …")
    result["lods"] = result.apply(
        lambda r: _lods(stay_data, r["icustay_id"]), axis=1
    )

    # --- Treatment flags ---
    logger.info("  Detecting ventilation …")
    vent_df = _detect_ventilation(
        source_path, cohort_df, window_td, read_csv_chunks_fn
    )
    result = result.merge(vent_df, on=keys, how="left")
    result["vent"] = result["vent"].fillna(0).astype(int)

    logger.info("  Detecting RRT …")
    rrt_df = _detect_rrt(source_path, cohort_df, window_td, read_csv_chunks_fn)
    result = result.merge(rrt_df, on=keys, how="left")
    result["rrt"] = result["rrt"].fillna(0).astype(int)

    # --- Urine output ---
    result = result.merge(
        urine_df.rename(columns={"total_ml": "urineoutput"}),
        on=keys,
        how="left",
    )

    # --- Fluid bolus ---
    logger.info("  Gathering fluid bolus from INPUTEVENTS …")
    bolus_df = _gather_fluid_bolus(
        source_path, cohort_df, window_td, read_csv_chunks_fn
    )
    result = result.merge(bolus_df, on=keys, how="left")

    return result[
        keys
        + [
            "sofa",
            "lods",
            "sirs",
            "qsofa",
            "vent",
            "rrt",
            "urineoutput",
            "colloid_bolus",
            "crystalloid_bolus",
        ]
    ]


# ======================================================================
# Raw data gathering (chunked reads)
# ======================================================================


def _gather_vasopressor_flags(
    source_path: Path,
    cohort_df: pd.DataFrame,
    window_td: pd.Timedelta,
    read_csv_chunks_fn: Any,
) -> pd.DataFrame:
    """Return per-stay vasopressor usage flags.

    Columns: subject_id, hadm_id, icustay_id,
             dopamine, dobutamine, norepinephrine, epinephrine
    """
    keys = ["subject_id", "hadm_id", "icustay_id"]
    result = cohort_df[keys].copy()
    for col in ("dopamine", "dobutamine", "norepinephrine", "epinephrine"):
        result[col] = 0

    accum: list[pd.DataFrame] = []
    cohort_cols = keys + ["intime"]

    for csv_name in ("INPUTEVENTS_CV.csv", "INPUTEVENTS_MV.csv"):
        time_col = "charttime" if "CV" in csv_name else "starttime"
        try:
            for chunk in read_csv_chunks_fn(
                source_path / csv_name,
                required=[
                    "subject_id",
                    "hadm_id",
                    "icustay_id",
                    time_col,
                    "itemid",
                ],
            ):
                chunk["_time"] = pd.to_datetime(
                    chunk[time_col], errors="coerce"
                )
                chunk["itemid"] = pd.to_numeric(
                    chunk["itemid"], errors="coerce"
                )
                chunk = chunk[chunk["itemid"].isin(_ALL_VASOPRESSOR_IDS)]
                if chunk.empty:
                    continue
                chunk = chunk.merge(
                    cohort_df[cohort_cols],
                    on=keys,
                    how="inner",
                )
                if chunk.empty:
                    continue
                chunk = chunk[
                    (chunk["_time"] >= chunk["intime"])
                    & (chunk["_time"] <= chunk["intime"] + window_td)
                ].copy()
                if not chunk.empty:
                    accum.append(chunk[keys + ["itemid"]])
        except FileNotFoundError:
            logger.debug("%s not found – skipping", csv_name)

    if accum:
        events = pd.concat(accum, ignore_index=True)
        for col, ids in [
            ("dopamine", set(_DOPAMINE_IDS)),
            ("dobutamine", set(_DOBUTAMINE_IDS)),
            ("norepinephrine", set(_NOREPINEPHRINE_IDS)),
            ("epinephrine", set(_EPINEPHRINE_IDS)),
        ]:
            flagged = events[events["itemid"].isin(ids)][keys].drop_duplicates()
            flagged[col] = 1
            result = result.merge(
                flagged, on=keys, how="left", suffixes=("", "_r")
            )
            if f"{col}_r" in result.columns:
                result[col] = (
                    result[col].fillna(result[f"{col}_r"]).fillna(0).astype(int)
                )
                result = result.drop(columns=[f"{col}_r"])
            else:
                result[col] = result[col].fillna(0).astype(int)

    return result


def _gather_urine_output(
    source_path: Path,
    cohort_df: pd.DataFrame,
    window_td: pd.Timedelta,
    read_csv_chunks_fn: Any,
) -> pd.DataFrame:
    """Return total urineoutput (mL) per stay."""
    keys = ["subject_id", "hadm_id", "icustay_id"]
    cohort_cols = keys + ["intime"]
    accum: list[pd.DataFrame] = []

    try:
        for chunk in read_csv_chunks_fn(
            source_path / "OUTPUTEVENTS.csv",
            required=[
                "subject_id",
                "hadm_id",
                "icustay_id",
                "charttime",
                "itemid",
                "value",
            ],
        ):
            chunk["charttime"] = pd.to_datetime(
                chunk["charttime"], errors="coerce"
            )
            chunk["itemid"] = pd.to_numeric(chunk["itemid"], errors="coerce")
            chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce")
            chunk = chunk[chunk["itemid"].isin(_URINE_OUTPUT_IDS)]
            if chunk.empty:
                continue
            chunk = chunk.merge(cohort_df[cohort_cols], on=keys, how="inner")
            if chunk.empty:
                continue
            chunk = chunk[
                chunk["value"].notna()
                & (chunk["value"] > 0)
                & (chunk["charttime"] >= chunk["intime"])
                & (chunk["charttime"] <= chunk["intime"] + window_td)
            ].copy()
            if not chunk.empty:
                accum.append(chunk[keys + ["value"]])
    except FileNotFoundError:
        logger.warning("OUTPUTEVENTS not found – skipping urine")

    if accum:
        events = pd.concat(accum, ignore_index=True)
        totals = (
            events.groupby(keys, as_index=False)["value"]
            .sum()
            .rename(columns={"value": "total_ml"})
        )
        return totals
    return pd.DataFrame(columns=keys + ["total_ml"])


def _detect_ventilation(
    source_path: Path,
    cohort_df: pd.DataFrame,
    window_td: pd.Timedelta,
    read_csv_chunks_fn: Any,
) -> pd.DataFrame:
    """Return per-stay vent flag (0/1)."""
    keys = ["subject_id", "hadm_id", "icustay_id"]
    cohort_cols = keys + ["intime"]
    vent_stays: set[tuple[int, int, int]] = set()

    # 1. CHARTEVENTS – ventilator mode/type items or MechVent=1
    chart_ids = set(_VENT_CHART_IDS) | {_VENT_MECHVENT_ID}
    try:
        for chunk in read_csv_chunks_fn(
            source_path / "CHARTEVENTS.csv",
            required=[
                "subject_id",
                "hadm_id",
                "icustay_id",
                "charttime",
                "itemid",
            ],
        ):
            chunk["charttime"] = pd.to_datetime(
                chunk["charttime"], errors="coerce"
            )
            chunk["itemid"] = pd.to_numeric(chunk["itemid"], errors="coerce")
            chunk = chunk[chunk["itemid"].isin(chart_ids)]
            if chunk.empty:
                continue
            chunk = chunk.merge(cohort_df[cohort_cols], on=keys, how="inner")
            if chunk.empty:
                continue
            chunk = chunk[
                (chunk["charttime"] >= chunk["intime"])
                & (chunk["charttime"] <= chunk["intime"] + window_td)
            ]
            for row in chunk[keys].drop_duplicates().itertuples(index=False):
                vent_stays.add((row.subject_id, row.hadm_id, row.icustay_id))
    except FileNotFoundError:
        pass

    # 2. PROCEDUREEVENTS_MV – invasive/non-invasive vent
    try:
        for chunk in read_csv_chunks_fn(
            source_path / "PROCEDUREEVENTS_MV.csv",
            required=[
                "subject_id",
                "hadm_id",
                "icustay_id",
                "starttime",
                "itemid",
            ],
        ):
            chunk["starttime"] = pd.to_datetime(
                chunk["starttime"], errors="coerce"
            )
            chunk["itemid"] = pd.to_numeric(chunk["itemid"], errors="coerce")
            chunk = chunk[chunk["itemid"].isin(_VENT_PROC_IDS)]
            if chunk.empty:
                continue
            chunk = chunk.merge(cohort_df[cohort_cols], on=keys, how="inner")
            if chunk.empty:
                continue
            chunk = chunk[
                (chunk["starttime"] >= chunk["intime"])
                & (chunk["starttime"] <= chunk["intime"] + window_td)
            ]
            for row in chunk[keys].drop_duplicates().itertuples(index=False):
                vent_stays.add((row.subject_id, row.hadm_id, row.icustay_id))
    except FileNotFoundError:
        pass

    result = cohort_df[keys].copy()
    result["vent"] = result.apply(
        lambda r: int(
            (r["subject_id"], r["hadm_id"], r["icustay_id"]) in vent_stays
        ),
        axis=1,
    )
    return result


def _detect_rrt(
    source_path: Path,
    cohort_df: pd.DataFrame,
    window_td: pd.Timedelta,
    read_csv_chunks_fn: Any,
) -> pd.DataFrame:
    """Return per-stay RRT flag (0/1)."""
    keys = ["subject_id", "hadm_id", "icustay_id"]
    cohort_cols = keys + ["intime"]
    rrt_stays: set[tuple[int, int, int]] = set()

    # 1. PROCEDUREEVENTS_MV – dialysis procedures
    try:
        for chunk in read_csv_chunks_fn(
            source_path / "PROCEDUREEVENTS_MV.csv",
            required=[
                "subject_id",
                "hadm_id",
                "icustay_id",
                "starttime",
                "itemid",
            ],
        ):
            chunk["starttime"] = pd.to_datetime(
                chunk["starttime"], errors="coerce"
            )
            chunk["itemid"] = pd.to_numeric(chunk["itemid"], errors="coerce")
            chunk = chunk[chunk["itemid"].isin(_RRT_PROC_IDS)]
            if chunk.empty:
                continue
            chunk = chunk.merge(cohort_df[cohort_cols], on=keys, how="inner")
            if chunk.empty:
                continue
            chunk = chunk[
                (chunk["starttime"] >= chunk["intime"])
                & (chunk["starttime"] <= chunk["intime"] + window_td)
            ]
            for row in chunk[keys].drop_duplicates().itertuples(index=False):
                rrt_stays.add((row.subject_id, row.hadm_id, row.icustay_id))
    except FileNotFoundError:
        pass

    # 2. CHARTEVENTS – Dialysis Type (CareVue)
    try:
        for chunk in read_csv_chunks_fn(
            source_path / "CHARTEVENTS.csv",
            required=[
                "subject_id",
                "hadm_id",
                "icustay_id",
                "charttime",
                "itemid",
            ],
        ):
            chunk["charttime"] = pd.to_datetime(
                chunk["charttime"], errors="coerce"
            )
            chunk["itemid"] = pd.to_numeric(chunk["itemid"], errors="coerce")
            chunk = chunk[chunk["itemid"].isin(_RRT_CHART_IDS)]
            if chunk.empty:
                continue
            chunk = chunk.merge(cohort_df[cohort_cols], on=keys, how="inner")
            if chunk.empty:
                continue
            chunk = chunk[
                (chunk["charttime"] >= chunk["intime"])
                & (chunk["charttime"] <= chunk["intime"] + window_td)
            ]
            for row in chunk[keys].drop_duplicates().itertuples(index=False):
                rrt_stays.add((row.subject_id, row.hadm_id, row.icustay_id))
    except FileNotFoundError:
        pass

    # 3. OUTPUTEVENTS – dialysis output items
    try:
        for chunk in read_csv_chunks_fn(
            source_path / "OUTPUTEVENTS.csv",
            required=[
                "subject_id",
                "hadm_id",
                "icustay_id",
                "charttime",
                "itemid",
            ],
        ):
            chunk["charttime"] = pd.to_datetime(
                chunk["charttime"], errors="coerce"
            )
            chunk["itemid"] = pd.to_numeric(chunk["itemid"], errors="coerce")
            chunk = chunk[chunk["itemid"].isin(_RRT_OUTPUT_IDS)]
            if chunk.empty:
                continue
            chunk = chunk.merge(cohort_df[cohort_cols], on=keys, how="inner")
            if chunk.empty:
                continue
            chunk = chunk[
                (chunk["charttime"] >= chunk["intime"])
                & (chunk["charttime"] <= chunk["intime"] + window_td)
            ]
            for row in chunk[keys].drop_duplicates().itertuples(index=False):
                rrt_stays.add((row.subject_id, row.hadm_id, row.icustay_id))
    except FileNotFoundError:
        pass

    result = cohort_df[keys].copy()
    result["rrt"] = result.apply(
        lambda r: int(
            (r["subject_id"], r["hadm_id"], r["icustay_id"]) in rrt_stays
        ),
        axis=1,
    )
    return result


def _gather_fluid_bolus(
    source_path: Path,
    cohort_df: pd.DataFrame,
    window_td: pd.Timedelta,
    read_csv_chunks_fn: Any,
) -> pd.DataFrame:
    """Return colloid_bolus and crystalloid_bolus (mL) per stay."""
    keys = ["subject_id", "hadm_id", "icustay_id"]
    cohort_cols = keys + ["intime"]
    all_ids = set(_COLLOID_IDS) | set(_CRYSTALLOID_IDS)
    colloid_set = set(_COLLOID_IDS)
    crystal_set = set(_CRYSTALLOID_IDS)
    accum: list[pd.DataFrame] = []

    for csv_name in ("INPUTEVENTS_CV.csv", "INPUTEVENTS_MV.csv"):
        time_col = "charttime" if "CV" in csv_name else "starttime"
        try:
            for chunk in read_csv_chunks_fn(
                source_path / csv_name,
                required=[
                    "subject_id",
                    "hadm_id",
                    "icustay_id",
                    time_col,
                    "itemid",
                    "amount",
                ],
            ):
                chunk["_time"] = pd.to_datetime(
                    chunk[time_col], errors="coerce"
                )
                chunk["itemid"] = pd.to_numeric(
                    chunk["itemid"], errors="coerce"
                )
                chunk["amount"] = pd.to_numeric(
                    chunk["amount"], errors="coerce"
                )
                chunk = chunk[chunk["itemid"].isin(all_ids)]
                if chunk.empty:
                    continue
                chunk = chunk.merge(
                    cohort_df[cohort_cols], on=keys, how="inner"
                )
                if chunk.empty:
                    continue
                chunk = chunk[
                    chunk["amount"].notna()
                    & (chunk["amount"] > 0)
                    & (chunk["_time"] >= chunk["intime"])
                    & (chunk["_time"] <= chunk["intime"] + window_td)
                ].copy()
                if not chunk.empty:
                    accum.append(chunk[keys + ["itemid", "amount"]])
        except FileNotFoundError:
            logger.debug("%s not found – skipping", csv_name)

    result = cohort_df[keys].copy()
    result["colloid_bolus"] = np.nan
    result["crystalloid_bolus"] = np.nan

    if accum:
        events = pd.concat(accum, ignore_index=True)
        # Colloid
        col_events = events[events["itemid"].isin(colloid_set)]
        if not col_events.empty:
            col_totals = (
                col_events.groupby(keys, as_index=False)["amount"]
                .sum()
                .rename(columns={"amount": "colloid_bolus"})
            )
            result = result.drop(columns=["colloid_bolus"]).merge(
                col_totals, on=keys, how="left"
            )
        # Crystalloid
        cry_events = events[events["itemid"].isin(crystal_set)]
        if not cry_events.empty:
            cry_totals = (
                cry_events.groupby(keys, as_index=False)["amount"]
                .sum()
                .rename(columns={"amount": "crystalloid_bolus"})
            )
            result = result.drop(columns=["crystalloid_bolus"]).merge(
                cry_totals, on=keys, how="left"
            )

    return result


# ======================================================================
# Per-stay data aggregation
# ======================================================================


def _build_stay_data(
    cohort_df: pd.DataFrame,
    vaso_flags: pd.DataFrame,
    urine_df: pd.DataFrame,
    *,
    temporal_df: pd.DataFrame | None = None,
) -> dict[int, dict[str, Any]]:
    """Pre-aggregate per-icustay_id data for score computation.

    Returns dict: icustay_id → {gcs_min, fio2_max, pao2_min,
    paco2_min, bilirubin_max, pt_max, dopamine, dobutamine,
    norepinephrine, epinephrine, urineoutput, ...}

    All clinical values (GCS, FiO2, PaO2, PaCO2, bilirubin, PT,
    vitals, labs) are sourced from *temporal_df* columns produced
    by the Step 3 extraction pipeline.
    """
    data: dict[int, dict[str, Any]] = {}

    # All keys that scores need from temporal_df
    _TEMPORAL_KEYS = [
        "heartrate_max",
        "tempc_min",
        "tempc_max",
        "resprate_max",
        "sysbp_min",
        "wbc_min",
        "wbc_max",
        "platelet_min",
        "creatinine_max",
        "meanbp_min",
        "bun_max",
        # Newly moved from chart/lab raw extraction:
        "pao2_min",
        "paco2_min",
        "bilirubin_max",
        "pt_max",
        "fio2_max",
        "gcs_total_min",
        "gcs_eye_min",
        "gcs_verbal_min",
        "gcs_motor_min",
    ]

    # Initialize from cohort
    for _, row in cohort_df.iterrows():
        icustay_id = int(row["icustay_id"])
        data[icustay_id] = {
            "gcs_min": np.nan,
            "fio2_max": np.nan,
            "pao2_min": np.nan,
            "paco2_min": np.nan,
            "bilirubin_max": np.nan,
            "pt_max": np.nan,
            "dopamine": 0,
            "dobutamine": 0,
            "norepinephrine": 0,
            "epinephrine": 0,
            "urineoutput": np.nan,
            "heartrate_max": np.nan,
            "tempc_min": np.nan,
            "tempc_max": np.nan,
            "resprate_max": np.nan,
            "sysbp_min": np.nan,
            "wbc_min": np.nan,
            "wbc_max": np.nan,
            "platelet_min": np.nan,
            "creatinine_max": np.nan,
            "meanbp_min": np.nan,
            "bun_max": np.nan,
        }

    # Vasopressor flags
    for _, row in vaso_flags.iterrows():
        icustay_id = int(row["icustay_id"])
        if icustay_id in data:
            for col in (
                "dopamine",
                "dobutamine",
                "norepinephrine",
                "epinephrine",
            ):
                data[icustay_id][col] = int(row[col])

    # Urine output
    if not urine_df.empty and "total_ml" in urine_df.columns:
        for _, row in urine_df.iterrows():
            icustay_id = int(row["icustay_id"])
            if icustay_id in data:
                data[icustay_id]["urineoutput"] = float(row["total_ml"])

    # Inject temporal aggregates (all clinical values from step 3)
    if temporal_df is not None:
        avail = [c for c in _TEMPORAL_KEYS if c in temporal_df.columns]
        if avail:
            for _, row in temporal_df.iterrows():
                icustay_id = int(row["icustay_id"])
                if icustay_id not in data:
                    continue
                for col in avail:
                    val = row[col]
                    if val is not None and not np.isnan(val):
                        data[icustay_id][col] = float(val)

        # Derive composite GCS minimum from temporal columns
        # GCS is recorded two ways in MIMIC-III:
        #   CareVue: gcs_total (itemid 198) – a single total score
        #   MetaVision: gcs_eye + gcs_verbal + gcs_motor – components
        # Take the minimum of both approaches.
        for icustay_id, d in data.items():
            gcs_candidates: list[float] = []

            gcs_total = d.get("gcs_total_min")
            if gcs_total is not None and not np.isnan(gcs_total):
                gcs_candidates.append(gcs_total)

            gcs_e = d.get("gcs_eye_min")
            gcs_v = d.get("gcs_verbal_min")
            gcs_m = d.get("gcs_motor_min")
            if (
                gcs_e is not None
                and not np.isnan(gcs_e)
                and gcs_v is not None
                and not np.isnan(gcs_v)
                and gcs_m is not None
                and not np.isnan(gcs_m)
            ):
                gcs_candidates.append(gcs_e + gcs_v + gcs_m)

            if gcs_candidates:
                d["gcs_min"] = min(gcs_candidates)

    return data


# ======================================================================
# Score computation functions
# ======================================================================


def _sofa(data: dict[int, dict[str, Any]], icustay_id: int) -> int:
    """SOFA score (0-24) based on worst values in window.

    Six organ systems, each scored 0-4:
    1. Respiration: PaO2/FiO2 ratio
    2. Coagulation: Platelets
    3. Liver: Bilirubin
    4. Cardiovascular: MAP / vasopressors
    5. CNS: GCS
    6. Renal: Creatinine / urine output
    """
    d = data.get(icustay_id)
    if d is None:
        return 0

    score = 0

    # 1. Respiration – PaO2/FiO2
    pao2 = d.get("pao2_min")
    fio2 = d.get("fio2_max")
    if (
        pao2 is not None
        and fio2 is not None
        and not np.isnan(pao2)
        and not np.isnan(fio2)
        and fio2 > 0
    ):
        pf_ratio = pao2 / fio2
        if pf_ratio < 100:
            score += 4
        elif pf_ratio < 200:
            score += 3
        elif pf_ratio < 300:
            score += 2
        elif pf_ratio < 400:
            score += 1

    # 2. Coagulation – Platelets (K/µL)
    plt_min = d.get("platelet_min")
    if plt_min is not None and not np.isnan(plt_min):
        if plt_min < 20:
            score += 4
        elif plt_min < 50:
            score += 3
        elif plt_min < 100:
            score += 2
        elif plt_min < 150:
            score += 1

    # 3. Liver – Bilirubin (mg/dL)
    bili = d.get("bilirubin_max")
    if bili is not None and not np.isnan(bili):
        if bili >= 12.0:
            score += 4
        elif bili >= 6.0:
            score += 3
        elif bili >= 2.0:
            score += 2
        elif bili >= 1.2:
            score += 1

    # 4. Cardiovascular – MAP and vasopressors
    map_min = d.get("meanbp_min")
    dopa = d.get("dopamine", 0)
    dobu = d.get("dobutamine", 0)
    norepi = d.get("norepinephrine", 0)
    epi = d.get("epinephrine", 0)

    if norepi or epi:
        # High-dose vasopressors → score 3 or 4
        # Simplified: presence = score 4
        score += 4
    elif dopa or dobu:
        score += 3
    elif map_min is not None and not np.isnan(map_min) and map_min < 70:
        score += 1

    # 5. CNS – GCS
    gcs = d.get("gcs_min")
    if gcs is not None and not np.isnan(gcs):
        if gcs < 6:
            score += 4
        elif gcs < 10:
            score += 3
        elif gcs < 13:
            score += 2
        elif gcs < 15:
            score += 1

    # 6. Renal – Creatinine (mg/dL) or urine output
    cr = d.get("creatinine_max")
    uo = d.get("urineoutput")

    renal_score = 0
    if cr is not None and not np.isnan(cr):
        if cr >= 5.0:
            renal_score = 4
        elif cr >= 3.5:
            renal_score = 3
        elif cr >= 2.0:
            renal_score = 2
        elif cr >= 1.2:
            renal_score = 1

    # Urine output scoring (for 24h window)
    if uo is not None and not np.isnan(uo):
        if uo < 200:
            renal_score = max(renal_score, 4)
        elif uo < 500:
            renal_score = max(renal_score, 3)

    score += renal_score

    return score


def _sirs(data: dict[int, dict[str, Any]], icustay_id: int) -> int:
    """SIRS criteria count (0-4).

    1. Temperature <36°C or >38°C
    2. Heart rate >90 bpm
    3. Respiratory rate >20 or PaCO2 <32 mmHg
    4. WBC <4 or >12 K/µL
    """
    d = data.get(icustay_id)
    if d is None:
        return 0

    score = 0

    # 1. Temperature
    t_min = d.get("tempc_min")
    t_max = d.get("tempc_max")
    if (t_min is not None and not np.isnan(t_min) and t_min < 36) or (
        t_max is not None and not np.isnan(t_max) and t_max > 38
    ):
        score += 1

    # 2. Heart rate
    hr_max = d.get("heartrate_max")
    if hr_max is not None and not np.isnan(hr_max) and hr_max > 90:
        score += 1

    # 3. Respiratory rate or PaCO2
    rr_max = d.get("resprate_max")
    paco2 = d.get("paco2_min")
    if (rr_max is not None and not np.isnan(rr_max) and rr_max > 20) or (
        paco2 is not None and not np.isnan(paco2) and paco2 < 32
    ):
        score += 1

    # 4. WBC
    wbc_min = d.get("wbc_min")
    wbc_max = d.get("wbc_max")
    if (wbc_min is not None and not np.isnan(wbc_min) and wbc_min < 4) or (
        wbc_max is not None and not np.isnan(wbc_max) and wbc_max > 12
    ):
        score += 1

    return score


def _qsofa(data: dict[int, dict[str, Any]], icustay_id: int) -> int:
    """qSOFA score (0-3).

    1. Respiratory rate ≥ 22
    2. Systolic BP ≤ 100 mmHg
    3. Altered mentation (GCS < 15)
    """
    d = data.get(icustay_id)
    if d is None:
        return 0

    score = 0

    rr_max = d.get("resprate_max")
    if rr_max is not None and not np.isnan(rr_max) and rr_max >= 22:
        score += 1

    sbp_min = d.get("sysbp_min")
    if sbp_min is not None and not np.isnan(sbp_min) and sbp_min <= 100:
        score += 1

    gcs = d.get("gcs_min")
    if gcs is not None and not np.isnan(gcs) and gcs < 15:
        score += 1

    return score


def _lods(data: dict[int, dict[str, Any]], icustay_id: int) -> int:
    """LODS – Logistic Organ Dysfunction Score (0-22).

    Six organ systems scored according to Le Gall et al. (1996):
      Neurologic (0-5), Cardiovascular (0-5), Renal (0-5),
      Pulmonary (0-3), Hematologic (0-3), Hepatic (0-1).
    Uses worst values within the time window.
    """
    d = data.get(icustay_id)
    if d is None:
        return 0

    score = 0

    # 1. Neurologic (GCS) – 0, 1, 3, 5
    gcs = d.get("gcs_min")
    if gcs is not None and not np.isnan(gcs):
        if gcs < 6:
            score += 5
        elif gcs < 9:
            score += 3
        elif gcs < 14:
            score += 1

    # 2. Cardiovascular – HR, SBP – 0, 1, 3, 5
    hr_max = d.get("heartrate_max")
    sbp_min = d.get("sysbp_min")
    cv_score = 0
    if sbp_min is not None and not np.isnan(sbp_min):
        if sbp_min < 70:
            cv_score = 5
        elif sbp_min < 90:
            cv_score = max(cv_score, 3)
        elif sbp_min < 100:
            cv_score = max(cv_score, 1)
    if hr_max is not None and not np.isnan(hr_max):
        if hr_max >= 140:
            cv_score = max(cv_score, 3)
    score += cv_score

    # 3. Renal – BUN, creatinine, urine output – 0, 1, 3, 5
    bun_max = d.get("bun_max")
    cr_max = d.get("creatinine_max")
    uo = d.get("urineoutput")
    renal_score = 0
    if cr_max is not None and not np.isnan(cr_max):
        if cr_max >= 6.0:
            renal_score = 5
        elif cr_max >= 3.5:
            renal_score = max(renal_score, 3)
        elif cr_max >= 1.8:
            renal_score = max(renal_score, 1)
    if bun_max is not None and not np.isnan(bun_max):
        if bun_max >= 56:
            renal_score = max(renal_score, 5)
        elif bun_max >= 28:
            renal_score = max(renal_score, 3)
        elif bun_max >= 17:
            renal_score = max(renal_score, 1)
    if uo is not None and not np.isnan(uo):
        if uo < 500:
            renal_score = max(renal_score, 5)
        elif uo < 750:
            renal_score = max(renal_score, 3)
    score += renal_score

    # 4. Pulmonary – PaO2/FiO2 – 0, 1, 3
    pao2 = d.get("pao2_min")
    fio2 = d.get("fio2_max")
    if (
        pao2 is not None
        and fio2 is not None
        and not np.isnan(pao2)
        and not np.isnan(fio2)
        and fio2 > 0
    ):
        pf = pao2 / fio2
        if pf < 150:
            score += 3
        elif pf < 300:
            score += 1

    # 5. Hematologic – WBC, platelet – 0, 1, 3
    wbc_min = d.get("wbc_min")
    wbc_max = d.get("wbc_max")
    plt_min = d.get("platelet_min")
    heme_score = 0
    if plt_min is not None and not np.isnan(plt_min):
        if plt_min < 50:
            heme_score = 3
        elif plt_min < 100:
            heme_score = max(heme_score, 1)
    if wbc_min is not None and not np.isnan(wbc_min):
        if wbc_min < 1.0:
            heme_score = max(heme_score, 3)
        elif wbc_min < 2.5:
            heme_score = max(heme_score, 1)
    if wbc_max is not None and not np.isnan(wbc_max):
        if wbc_max >= 25:
            heme_score = max(heme_score, 1)
    score += heme_score

    # 6. Hepatic – bilirubin, PT – 0, 1
    bili = d.get("bilirubin_max")
    pt = d.get("pt_max")
    hep_score = 0
    if bili is not None and not np.isnan(bili):
        if bili >= 2.0:
            hep_score = 1
    if pt is not None and not np.isnan(pt):
        if pt >= 25:
            hep_score = max(hep_score, 1)
    score += hep_score

    return score
