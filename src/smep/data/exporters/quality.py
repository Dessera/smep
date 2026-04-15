"""Quality report generation for base table exports."""

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_BINARY_COLUMNS = {
    "hospital_expire_flag",
    "thirtyday_expire_flag",
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
}

_PRIMARY_KEY = ["subject_id", "hadm_id", "icustay_id"]


def generate_quality_report(
    df: pd.DataFrame,
    required_columns: list[str],
) -> dict[str, Any]:
    """Run quality checks and return a structured report dict."""
    report: dict[str, Any] = {}

    # 1. Primary key uniqueness
    pk_cols = [c for c in _PRIMARY_KEY if c in df.columns]
    if pk_cols:
        n_dupes = int(df.duplicated(subset=pk_cols).sum())
        report["primary_key_duplicates"] = n_dupes
    else:
        report["primary_key_duplicates"] = -1

    # 2. Duplicate column names
    col_counts = pd.Series(df.columns).value_counts()
    dupes = col_counts[col_counts > 1]
    report["duplicate_column_names"] = (
        dupes.to_dict() if not dupes.empty else {}
    )

    # 3. Required columns presence
    present = set(df.columns)
    missing_required = sorted(set(required_columns) - present)
    report["missing_required_columns"] = missing_required

    # 4. Binary field value-domain check
    binary_issues: dict[str, list[str]] = {}
    for col in _BINARY_COLUMNS & present:
        unique_vals = set(df[col].dropna().unique())
        invalid = unique_vals - {0, 1, 0.0, 1.0}
        if invalid:
            binary_issues[col] = [str(v) for v in sorted(invalid)]
    report["binary_field_issues"] = binary_issues

    # 5. Per-column missing rates
    n_rows = len(df)
    missing: dict[str, float] = {}
    for col in df.columns:
        rate = float(df[col].isna().sum() / n_rows) if n_rows else 0.0
        missing[col] = round(rate, 6)
    report["missing_rates"] = missing

    # 6. Numeric column summary (non-placeholder)
    numeric_cols = df.select_dtypes(include="number").columns
    summary: dict[str, dict[str, float]] = {}
    for col in numeric_cols:
        s = df[col].dropna()
        if s.empty:
            continue
        summary[col] = {
            "min": round(float(s.min()), 4),
            "max": round(float(s.max()), 4),
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std()), 4),
        }
    report["numeric_summary"] = summary

    # 7. Label distribution
    for label_col in (
        "hospital_expire_flag",
        "thirtyday_expire_flag",
    ):
        if label_col in present:
            counts = df[label_col].value_counts(dropna=False)
            report[f"{label_col}_distribution"] = {
                str(k): int(v) for k, v in counts.items()
            }

    report["total_rows"] = n_rows
    report["total_columns"] = len(df.columns)

    return report
