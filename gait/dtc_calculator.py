"""
Module 7 — dtc_calculator.py
Responsibility: Compute Dual-Task Cost (DTC) for each parameter.

=== DUO-GAIT formula ===
File: src/data/dual_task_costs.py line 66

    costs = (arr_st - arr_dt) / arr_st * 100

Applied element-wise across all numeric columns of the aggregated DataFrames.
Positive DTC = performance WORSENED under dual-task (e.g. stride_lengths_avg
decreased, so ST > DT, so DTC > 0).

DUO-GAIT applies DTC to both avg and CV columns (it operates on the entire
flat array of all aggregated parameter columns).

Output:
    pd.DataFrame (one row per participant) with columns:
        sub, condition, <param>_avg_DTC, <param>_CV_DTC, <param>_SI_DTC, …
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_dtc(
    st_agg: pd.DataFrame,
    dt_agg: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute Dual-Task Cost for all numeric columns.

    Formula (DUO-GAIT dual_task_costs.py line 66):
        DTC(%) = (X_ST - X_DT) / X_ST * 100

    Parameters
    ----------
    st_agg : pd.DataFrame
        Aggregated summary for single-task condition (one row).
        Output of aggregator.aggregate() with condition='st'.
    dt_agg : pd.DataFrame
        Aggregated summary for dual-task condition (one row).
        Output of aggregator.aggregate() with condition='dt'.

    Returns
    -------
    pd.DataFrame
        One-row DataFrame with DTC(%) for every numeric column that appears
        in both inputs.  Column names are suffixed with '_DTC'.
        Also carries 'sub' and 'condition' metadata from the ST DataFrame.
    """
    if st_agg.empty or dt_agg.empty:
        return pd.DataFrame()

    # Reset index to guarantee .iloc[0] works
    st = st_agg.reset_index(drop=True).iloc[0]
    dt = dt_agg.reset_index(drop=True).iloc[0]

    result: dict = {}

    # Carry metadata
    result["sub"]       = st.get("sub", "")
    result["condition"] = st.get("condition", "st")

    # Compute DTC for all numeric parameter columns
    # DUO-GAIT operates on the entire array at once (dual_task_costs.py:63-66)
    skip_cols = {"sub", "condition", "fatigue"}
    all_cols = [c for c in st_agg.columns if c not in skip_cols]

    for col in all_cols:
        try:
            val_st = float(st[col])
            val_dt = float(dt[col])
        except (ValueError, TypeError, KeyError):
            continue

        if np.isnan(val_st) or val_st == 0:
            result[f"{col}_DTC"] = np.nan
        else:
            # Exact DUO-GAIT formula, dual_task_costs.py line 66:
            result[f"{col}_DTC"] = (val_st - val_dt) / val_st * 100.0

    return pd.DataFrame([result])


def dtc_summary_table(dtc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reformat the DTC row into a human-readable summary table (one row per
    parameter, with columns: parameter, DTC_pct, direction).

    direction: 'worsened' if DTC > 0, 'improved' if DTC < 0, 'unchanged' if 0.
    """
    dtc_cols = [c for c in dtc_df.columns if c.endswith("_DTC")]
    rows = []
    for col in dtc_cols:
        val = float(dtc_df[col].iloc[0])
        if np.isnan(val):
            direction = "n/a"
        elif val > 0:
            direction = "worsened"
        elif val < 0:
            direction = "improved"
        else:
            direction = "unchanged"
        rows.append({
            "parameter": col.replace("_DTC", ""),
            "DTC_pct":   round(val, 3),
            "direction": direction,
        })
    return pd.DataFrame(rows)
