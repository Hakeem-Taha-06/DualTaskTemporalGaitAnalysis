"""
Module 6 — aggregator.py
Responsibility: Aggregate stride-level parameters to participant-level summary
statistics using EXACTLY the same formulas as DUO-GAIT.

=== DUO-GAIT source ===
File: src/features/aggregate_gait_parameters.py

Key formulas:
    cadence = 120 / stride_times        (line 39)  — steps/min
    speed   = stride_lengths / stride_times  (line 40)
    mean    = df_param.mean()           (line 42)
    CV      = scipy.stats.variation()   (line 43)
              (= std/mean, returns a decimal fraction NOT percentage)
    SI      = |X_L - X_R| / (0.5 * (X_L + X_R))
              (lines 99-101, calculate_SI())
              NOTE: DUO-GAIT does NOT multiply SI by 100 in code,
              despite the prompt saying *100.  We follow the code.

Filtering before aggregation (lines 15-18):
    df = df[df.is_outlier != 1]
    df = df[df.turning_interval != 1]   (if column present)
    df = df[df.interrupted != 1]        (if column present)

Output schema:
    pd.DataFrame (one row) with columns:
        <param>_avg  — mean across all valid strides (both feet)
        <param>_CV   — coefficient of variation (scipy.stats.variation)
        <param>_SI   — symmetry index (left vs right means)
        <param>_avg_left,  <param>_avg_right  — per-foot means
        <param>_CV_left,   <param>_CV_right   — per-foot CV
    where <param> ∈ {stride_lengths, stride_times, swing_times, stance_times,
                     stance_ratios, cadence, speed,
                     step_time, double_support_time}

Parameters with no left/right distinction (only combined computed):
    step_time, double_support_time
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import variation


# Parameters aggregated per DUO-GAIT aggregate_gait_parameters.py:27-36
_CORE_PARAMS = [
    "stride_lengths",
    "stride_times",
    "swing_times",
    "stance_times",
    "stance_ratios",
]

# Derived parameters (computed during aggregation, lines 39-40)
_DERIVED_PARAMS = ["cadence", "speed"]

# NOT IN DUO-GAIT SOURCE — extra parameters computed for completeness
_EXTRA_PARAMS = ["step_time", "double_support_time"]

_ALL_PARAMS = _CORE_PARAMS + _DERIVED_PARAMS + _EXTRA_PARAMS


def aggregate(
    strides_df: pd.DataFrame,
    participant_id: str = "",
    condition: str = "",
) -> pd.DataFrame:
    """
    Aggregate stride-by-stride parameters to a single summary row.

    Parameters
    ----------
    strides_df : pd.DataFrame
        Output of outlier_remover.remove_outliers().  Contains one row per
        stride for both feet.
    participant_id : str
        Written into the output row for identification.
    condition : str
        'st' or 'dt' — written into the output row.

    Returns
    -------
    pd.DataFrame
        One-row summary DataFrame.
    """
    df = strides_df.copy()

    # Filter out invalid strides before computing statistics.
    # Mirrors DUO-GAIT aggregate_gait_parameters.py lines 15-18:
    #   df = df[df.is_outlier != 1]
    #   df = df[df.turning_interval != 1]
    #   df = df[df.interrupted != 1]
    if "is_outlier" in df.columns:
        df = df[df["is_outlier"] != True].reset_index(drop=True)
    if "turning_interval" in df.columns:
        df = df[df["turning_interval"] != True].reset_index(drop=True)
    if "interrupted" in df.columns:
        df = df[df["interrupted"] != True].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Add derived parameters — DUO-GAIT lines 39-40
    # ------------------------------------------------------------------
    df["cadence"] = 120.0 / df["stride_times"]   # steps/min
    df["speed"]   = df["stride_lengths"] / df["stride_times"]

    # ------------------------------------------------------------------
    # Build aggregated values for each foot and combined
    # Mirrors aggregate_parameters_from_df() and aggregate_parameters()
    # ------------------------------------------------------------------
    result: dict = {}

    params_for_si = _CORE_PARAMS + _DERIVED_PARAMS  # SI only for bilateral params

    for side in ("left", "right"):
        side_df = df[df["foot"] == side]
        for param in _ALL_PARAMS:
            if param not in side_df.columns:
                result[f"{param}_avg_{side}"] = np.nan
                result[f"{param}_CV_{side}"]  = np.nan
                continue
            vals = side_df[param].dropna().values
            if len(vals) == 0:
                result[f"{param}_avg_{side}"] = np.nan
                result[f"{param}_CV_{side}"]  = np.nan
            else:
                result[f"{param}_avg_{side}"] = float(np.mean(vals))
                result[f"{param}_CV_{side}"]  = (
                    float(variation(vals)) if len(vals) > 1 else np.nan
                )

    # Combined (both feet) means and CV — DUO-GAIT lines 42-43
    for param in _ALL_PARAMS:
        if param not in df.columns:
            result[f"{param}_avg"] = np.nan
            result[f"{param}_CV"]  = np.nan
            continue
        vals = df[param].dropna().values
        if len(vals) == 0:
            result[f"{param}_avg"] = np.nan
            result[f"{param}_CV"]  = np.nan
        else:
            result[f"{param}_avg"] = float(np.mean(vals))
            result[f"{param}_CV"]  = (
                float(variation(vals)) if len(vals) > 1 else np.nan
            )

    # Symmetry Index — DUO-GAIT calculate_SI() aggregate_gait_parameters.py:95-102
    # SI = |X_L - X_R| / (0.5 * (X_L + X_R))
    # NOTE: DUO-GAIT does NOT multiply by 100 (code line 101 has no *100)
    for param in params_for_si:
        avg_l = result.get(f"{param}_avg_left", np.nan)
        avg_r = result.get(f"{param}_avg_right", np.nan)
        if np.isnan(avg_l) or np.isnan(avg_r) or (avg_l + avg_r) == 0:
            result[f"{param}_SI"] = np.nan
        else:
            # DUO-GAIT calculate_SI, line 101:
            # SI_list = [x / (0.5 * y) for x, y in zip(diff_avg, sum_avg)]
            result[f"{param}_SI"] = abs(avg_l - avg_r) / (0.5 * (avg_l + avg_r))

    result["sub"]       = participant_id
    result["condition"] = condition

    return pd.DataFrame([result])


def aggregate_both_conditions(
    st_strides: pd.DataFrame,
    dt_strides: pd.DataFrame,
    participant_id: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper: aggregate ST and DT strides separately.

    Returns
    -------
    tuple of (st_summary, dt_summary) DataFrames.
    """
    st_agg = aggregate(st_strides, participant_id=participant_id, condition="st")
    dt_agg = aggregate(dt_strides, participant_id=participant_id, condition="dt")
    return st_agg, dt_agg
