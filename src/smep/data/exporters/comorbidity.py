"""Comorbidity and baseline information from MIMIC-III.

Computes per-stay:
  - elixhauser_hospital   van Walraven weighted Elixhauser comorbidity score
  - diabetes              diabetes indicator (0/1)
  - metastatic_cancer     metastatic cancer indicator (0/1)
  - tumor                 solid tumor without metastasis indicator (0/1)
  - first_service         first service line on admission
  - dbsource              data source (CareVue/MetaVision)

References
----------
- Quan H et al. Med Care 2005;43(11):1130-1139 (enhanced ICD-9-CM algorithms)
- van Walraven C et al. Med Care 2009;47(6):626-633 (Elixhauser weights)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Elixhauser ICD-9-CM mapping (Quan et al. 2005) with van Walraven weights
# -------------------------------------------------------------------------
# Each entry: (list_of_icd9_prefixes, van_walraven_weight)
# ICD-9 codes are stored without dots in MIMIC-III.

_ELIX: dict[str, tuple[list[str], int]] = {
    "chf": (
        [
            "39891",
            "40201",
            "40211",
            "40291",
            "40401",
            "40403",
            "40411",
            "40413",
            "40491",
            "40493",
            "4254",
            "4255",
            "4256",
            "4257",
            "4258",
            "4259",
            "428",
        ],
        7,
    ),
    "arrhythmia": (
        [
            "42610",
            "42611",
            "42613",
            "4267",
            "4269",
            "42610",
            "42612",
            "4270",
            "4271",
            "4272",
            "4273",
            "4274",
            "42760",
            "42761",
            "42769",
            "4279",
            "7850",
            "99601",
            "99604",
            "V450",
            "V533",
        ],
        5,
    ),
    "valve": (
        [
            "0932",
            "394",
            "395",
            "396",
            "397",
            "424",
            "7463",
            "7464",
            "7465",
            "7466",
            "V422",
            "V433",
        ],
        -1,
    ),
    "pulm_circ": (
        ["4150", "4151", "416", "4170", "4178", "4179"],
        4,
    ),
    "pvd": (
        [
            "0930",
            "4373",
            "440",
            "441",
            "4431",
            "4432",
            "4433",
            "4434",
            "4435",
            "4436",
            "4437",
            "4438",
            "4439",
            "4471",
            "5571",
            "5579",
            "V434",
        ],
        2,
    ),
    "htn_uncomp": (
        ["4011", "4019"],
        0,
    ),
    "htn_comp": (
        ["4010", "402", "403", "404", "405"],
        0,
    ),
    "paralysis": (
        [
            "3341",
            "342",
            "343",
            "3440",
            "3441",
            "3442",
            "3443",
            "3444",
            "3445",
            "3446",
            "3449",
        ],
        7,
    ),
    "neuro_other": (
        [
            "3319",
            "3320",
            "3321",
            "3334",
            "3335",
            "33392",
            "334",
            "335",
            "3362",
            "340",
            "341",
            "345",
            "3481",
            "3483",
            "7803",
            "7843",
        ],
        6,
    ),
    "chronic_pulm": (
        [
            "4168",
            "4169",
            "490",
            "491",
            "492",
            "493",
            "494",
            "495",
            "496",
            "497",
            "498",
            "499",
            "500",
            "501",
            "502",
            "503",
            "504",
            "505",
            "5064",
            "5081",
            "5088",
        ],
        3,
    ),
    "dm_uncomp": (
        ["2500", "2501", "2502", "2503"],
        0,
    ),
    "dm_comp": (
        ["2504", "2505", "2506", "2507", "2508", "2509"],
        0,
    ),
    "hypothyroid": (
        ["2409", "243", "244", "2461", "2468"],
        0,
    ),
    "renal": (
        [
            "40301",
            "40311",
            "40391",
            "40402",
            "40403",
            "40412",
            "40413",
            "40492",
            "40493",
            "585",
            "586",
            "5880",
            "V420",
            "V451",
            "V56",
        ],
        5,
    ),
    "liver": (
        [
            "07022",
            "07023",
            "07032",
            "07033",
            "07044",
            "07054",
            "0706",
            "0709",
            "4560",
            "4561",
            "4562",
            "570",
            "571",
            "5722",
            "5723",
            "5724",
            "5728",
            "5733",
            "5734",
            "5738",
            "5739",
            "V427",
        ],
        11,
    ),
    "pud": (
        ["5317", "5319", "5327", "5329", "5337", "5339", "5347", "5349"],
        0,
    ),
    "aids": (
        ["042", "043", "044"],
        0,
    ),
    "lymphoma": (
        ["200", "201", "202", "2030", "2386"],
        9,
    ),
    "mets": (
        ["196", "197", "198", "199"],
        12,
    ),
    "tumor": (
        [
            "140",
            "141",
            "142",
            "143",
            "144",
            "145",
            "146",
            "147",
            "148",
            "149",
            "150",
            "151",
            "152",
            "153",
            "154",
            "155",
            "156",
            "157",
            "158",
            "159",
            "160",
            "161",
            "162",
            "163",
            "164",
            "165",
            "170",
            "171",
            "172",
            "174",
            "175",
            "176",
            "177",
            "178",
            "179",
            "180",
            "181",
            "182",
            "183",
            "184",
            "185",
            "186",
            "187",
            "188",
            "189",
            "190",
            "191",
            "192",
            "193",
            "194",
            "195",
        ],
        4,
    ),
    "rheumatic": (
        [
            "446",
            "7010",
            "7100",
            "7101",
            "7102",
            "7103",
            "7104",
            "7108",
            "7109",
            "7112",
            "714",
            "7193",
            "720",
            "725",
            "7285",
            "72889",
            "72930",
        ],
        0,
    ),
    "coag": (
        ["286", "2871", "2873", "2874", "2875"],
        3,
    ),
    "obesity": (
        ["2780"],
        -4,
    ),
    "weight_loss": (
        ["260", "261", "262", "263", "7832"],
        6,
    ),
    "fluid_elec": (
        ["2536", "276"],
        5,
    ),
    "anemia_blood_loss": (
        ["2800"],
        -2,
    ),
    "anemia_deficiency": (
        [
            "2801",
            "2802",
            "2803",
            "2804",
            "2805",
            "2806",
            "2807",
            "2808",
            "2809",
            "281",
        ],
        -2,
    ),
    "alcohol": (
        [
            "2652",
            "2911",
            "2912",
            "2913",
            "2915",
            "2916",
            "2917",
            "2918",
            "2919",
            "3030",
            "3039",
            "3050",
            "3575",
            "4255",
            "5353",
            "5710",
            "5711",
            "5712",
            "5713",
            "980",
            "V113",
        ],
        0,
    ),
    "drug": (
        [
            "292",
            "304",
            "3052",
            "3053",
            "3054",
            "3055",
            "3056",
            "3057",
            "3058",
            "3059",
            "V6542",
        ],
        -7,
    ),
    "psychoses": (
        [
            "2938",
            "295",
            "29604",
            "29614",
            "29644",
            "29654",
            "297",
            "298",
        ],
        0,
    ),
    "depression": (
        ["2962", "2963", "2965", "3004", "309", "311"],
        -3,
    ),
}


def _match_category(code: str, prefixes: list[str]) -> bool:
    """Return True if *code* starts with any of *prefixes*."""
    return any(code.startswith(p) for p in prefixes)


def _compute_elixhauser(diagnoses: pd.DataFrame) -> pd.DataFrame:
    """Compute per-hadm_id Elixhauser flags and van Walraven score.

    Returns DataFrame with columns: hadm_id, elixhauser_hospital,
    diabetes, metastatic_cancer.
    """
    codes = diagnoses[["hadm_id", "icd9_code"]].copy()
    codes["icd9_code"] = (
        codes["icd9_code"]
        .astype(str)
        .str.strip()
        .str.replace(".", "", regex=False)
        .str.upper()
    )
    codes = codes.drop_duplicates()

    # Build per-hadm_id category flags
    hadm_ids = codes["hadm_id"].unique()
    records: list[dict[str, Any]] = []

    # Pre-compute: for each code, which categories it belongs to
    code_to_cats: dict[str, list[str]] = {}
    for _, row in codes.iterrows():
        c = row["icd9_code"]
        if c not in code_to_cats:
            cats = []
            for cat_name, (prefixes, _) in _ELIX.items():
                if _match_category(c, prefixes):
                    cats.append(cat_name)
            code_to_cats[c] = cats

    # Group codes by hadm_id
    grouped = codes.groupby("hadm_id")["icd9_code"].apply(set)

    for hadm_id, code_set in grouped.items():
        flags: set[str] = set()
        for c in code_set:
            flags.update(code_to_cats.get(c, []))

        # Hierarchy rules (Quan et al.)
        if "dm_comp" in flags:
            flags.discard("dm_uncomp")
        if "htn_comp" in flags:
            flags.discard("htn_uncomp")
        if "mets" in flags:
            flags.discard("tumor")

        # van Walraven weighted score
        score = sum(_ELIX[cat][1] for cat in flags)

        records.append(
            {
                "hadm_id": hadm_id,
                "elixhauser_hospital": score,
                "diabetes": int("dm_uncomp" in flags or "dm_comp" in flags),
                "metastatic_cancer": int("mets" in flags),
                "tumor": int("tumor" in flags),
            }
        )

    if not records:
        return pd.DataFrame(
            columns=[
                "hadm_id",
                "elixhauser_hospital",
                "diabetes",
                "metastatic_cancer",
                "tumor",
            ]
        )
    return pd.DataFrame(records)


def _extract_first_service(
    source_path: Path,
    read_csv_fn: Any,
) -> pd.DataFrame:
    """Extract the first service line per hadm_id from SERVICES.csv."""
    try:
        services = read_csv_fn(
            source_path / "SERVICES.csv",
            required=["subject_id", "hadm_id", "transfertime", "curr_service"],
        )
    except FileNotFoundError:
        logger.warning("SERVICES.csv not found – first_service will be NaN")
        return pd.DataFrame(columns=["hadm_id", "first_service"])

    services["transfertime"] = pd.to_datetime(
        services["transfertime"], errors="coerce"
    )
    services = services.sort_values("transfertime")
    first = services.drop_duplicates(subset=["hadm_id"], keep="first")
    return first[["hadm_id", "curr_service"]].rename(
        columns={"curr_service": "first_service"}
    )


def _extract_dbsource(
    source_path: Path,
    read_csv_fn: Any,
) -> pd.DataFrame:
    """Extract dbsource per icustay_id from ICUSTAYS.csv."""
    icustays = read_csv_fn(
        source_path / "ICUSTAYS.csv",
        required=["icustay_id", "dbsource"],
    )
    return icustays[["icustay_id", "dbsource"]].drop_duplicates()


# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------

COMORBIDITY_COLUMNS: list[str] = [
    "elixhauser_hospital",
    "diabetes",
    "metastatic_cancer",
    "tumor",
    "first_service",
    "dbsource",
]


def compute_comorbidity(
    source_path: Path,
    cohort_df: pd.DataFrame,
    read_csv_fn: Any,
) -> pd.DataFrame:
    """Compute comorbidity and baseline features for each ICU stay.

    Returns a DataFrame keyed on (subject_id, hadm_id, icustay_id).
    """
    keys = ["subject_id", "hadm_id", "icustay_id"]
    result = cohort_df[keys].copy()

    # --- Elixhauser + diabetes + metastatic_cancer ---
    diagnoses = read_csv_fn(
        source_path / "DIAGNOSES_ICD.csv",
        required=["subject_id", "hadm_id", "icd9_code"],
    )
    # Restrict to cohort admissions
    cohort_hadms = set(cohort_df["hadm_id"].unique())
    diagnoses = diagnoses[diagnoses["hadm_id"].isin(cohort_hadms)]

    elix_df = _compute_elixhauser(diagnoses)
    result = result.merge(elix_df, on="hadm_id", how="left")
    result["elixhauser_hospital"] = (
        result["elixhauser_hospital"].fillna(0).astype(int)
    )
    result["diabetes"] = result["diabetes"].fillna(0).astype(int)
    result["metastatic_cancer"] = (
        result["metastatic_cancer"].fillna(0).astype(int)
    )
    result["tumor"] = result["tumor"].fillna(0).astype(int)

    # --- first_service ---
    first_svc = _extract_first_service(source_path, read_csv_fn)
    result = result.merge(first_svc, on="hadm_id", how="left")

    # --- dbsource ---
    dbsource = _extract_dbsource(source_path, read_csv_fn)
    result = result.merge(dbsource, on="icustay_id", how="left")

    logger.info(
        "Comorbidity features: elixhauser range [%s, %s], "
        "diabetes=%.1f%%, mets=%.1f%%",
        result["elixhauser_hospital"].min(),
        result["elixhauser_hospital"].max(),
        result["diabetes"].mean() * 100,
        result["metastatic_cancer"].mean() * 100,
    )

    return result[keys + COMORBIDITY_COLUMNS]
