"""
Module 5 — outlier_remover.py
Responsibility: Remove invalid strides using exactly the same criteria as DUO-GAIT.

=== DUO-GAIT sources ===
Threshold + z-score outliers:
    LFRF_parameters/pipeline/gait_parameters.py lines 119-201
    (already applied in parameter_calculator._flag_outliers)

Turning interval expansion:
    features/postprocessing.py lines 20-48
        interval_size = 2   (line 97 in mark_processed_data)
        Marks ±2 strides around each turning_step as turning_interval.
        Also marks first and last 2 strides as turning_interval (line 45-46).

Turning detection (video domain adaptation):
    DUO-GAIT uses angle_change > 0.2 (quaternion norm change per stride,
    gait_parameters.py line 128).  In the video domain, foot orientation is
    inferred from the anterior-posterior velocity of the heel:
    a sign reversal in mean heel_x velocity across a stride indicates turning.
    Threshold is set so that a stride with net AP displacement < turning_min_m
    (0.05 m by default) is flagged as a turn.
    # NOT FOUND IN DUO-GAIT SOURCE (domain translation) — ASSUMPTION: 0.05 m net AP displacement

Interrupted strides:
    features/postprocessing.py lines 51-84
    Optional: if an interruptions DataFrame is supplied, strides whose
    timestamps fall within any [start, end] interval are marked 'interrupted'.

Aggregation filtering (aggregate_gait_parameters.py lines 15-18):
    Rows with is_outlier==1 OR turning_interval==1 OR interrupted==1 are
    excluded from aggregation.

Output: same schema as parameter_calculator output, with added columns:
    turning_interval  bool  — stride is within ±interval_size of a turning step
    interrupted       bool  — stride falls in a manually-annotated interruption
    removal_reason    str   — human-readable reason ('outlier', 'turning',
                              'interrupted', 'head_tail', '') for inspection
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# Turning interval expansion — EXACT value from DUO-GAIT
# features/postprocessing.py line 97 called with interval_size=2
_TURNING_INTERVAL_SIZE = 2

# Turning detection threshold (video domain)
# NOT FOUND IN DUO-GAIT SOURCE — ASSUMPTION:
# A stride is a turning stride if net anterior-posterior heel displacement
# is less than this value (the foot moved sideways rather than forward).
_TURNING_MIN_AP_DISPLACEMENT_M = 0.05


def remove_outliers(
    strides_df: pd.DataFrame,
    traj_df: pd.DataFrame,
    interruptions_df: Optional[pd.DataFrame] = None,
    interval_size: int = _TURNING_INTERVAL_SIZE,
    turning_min_ap_m: float = _TURNING_MIN_AP_DISPLACEMENT_M,
) -> pd.DataFrame:
    """
    Apply DUO-GAIT outlier removal rules to the stride-level DataFrame.

    Parameters
    ----------
    strides_df : pd.DataFrame
        Output of parameter_calculator.calculate_parameters().
    traj_df : pd.DataFrame
        Filtered trajectory DataFrame (output of preprocessor).
    interruptions_df : pd.DataFrame or None
        Optional table with columns [start_s, end_s] marking time intervals
        where the participant was interrupted.  Mirrors DUO-GAIT
        features/postprocessing.py:mark_interrupted_strides.
    interval_size : int
        Number of strides before/after each turning stride to mark as
        turning_interval.  Default 2 — exact DUO-GAIT value.
    turning_min_ap_m : float
        Minimum net AP displacement (metres) for a stride NOT to be a turn.

    Returns
    -------
    pd.DataFrame
        Cleaned stride DataFrame with added columns:
        turning_interval, interrupted, removal_reason.
        is_outlier and turning_step columns are preserved from the input.
    """
    if strides_df.empty:
        return strides_df.copy()

    df = strides_df.copy()

    # Ensure required columns exist (may already be set by parameter_calculator)
    if "is_outlier"   not in df.columns: df["is_outlier"]   = False
    if "turning_step" not in df.columns: df["turning_step"] = False

    df["turning_interval"] = False
    df["interrupted"]      = False
    df["removal_reason"]   = ""

    # ------------------------------------------------------------------
    # 1. Detect turning strides from trajectory (video-domain analogue of
    #    angle_change > 0.2 threshold in DUO-GAIT gait_parameters.py:128)
    # ------------------------------------------------------------------
    df = _detect_turning_strides(df, traj_df, turning_min_ap_m)

    # ------------------------------------------------------------------
    # 2. Expand turning strides to turning intervals + head/tail
    #    DUO-GAIT: features/postprocessing.py lines 20-48
    # ------------------------------------------------------------------
    df = _mark_turning_interval(df, interval_size)

    # ------------------------------------------------------------------
    # 3. Mark interrupted strides
    #    DUO-GAIT: features/postprocessing.py lines 51-84
    # ------------------------------------------------------------------
    if interruptions_df is not None and not interruptions_df.empty:
        df = _mark_interrupted_strides(df, interruptions_df)

    # ------------------------------------------------------------------
    # 4. Populate removal_reason for inspection
    # ------------------------------------------------------------------
    df.loc[df["is_outlier"]       == True, "removal_reason"] += "outlier;"
    df.loc[df["turning_interval"] == True, "removal_reason"] += "turning;"
    df.loc[df["interrupted"]      == True, "removal_reason"] += "interrupted;"
    df["removal_reason"] = df["removal_reason"].str.strip(";")

    return df


# ---------------------------------------------------------------------------
# Turning stride detection
# ---------------------------------------------------------------------------

def _detect_turning_strides(
    df: pd.DataFrame,
    traj_df: pd.DataFrame,
    turning_min_ap_m: float,
) -> pd.DataFrame:
    """
    Mark strides as turning_step if the net AP heel displacement is small.
    This is the video-domain analogue of DUO-GAIT's angle_change > 0.2 check.

    # NOT FOUND IN DUO-GAIT SOURCE (domain translation)
    # ASSUMPTION: turning stride = net AP displacement < turning_min_ap_m
    """
    frame_to_row = {f: i for i, f in enumerate(traj_df["frame"].values)}

    for idx in df.index:
        side = df.at[idx, "foot"]
        # stride spans from IC[i] (timestamps → frame via fo/ic_samples) to IC[i+1]
        ic_start_sample = _nearest_frame(
            traj_df, df.at[idx, "timestamps"]
        )
        ic_end_sample = int(df.at[idx, "ic_samples"])

        x_col = f"{side}_heel_x"
        try:
            row_start = frame_to_row[ic_start_sample]
            row_end   = frame_to_row[ic_end_sample]
        except KeyError:
            continue

        ap_disp = abs(traj_df[x_col].iat[row_end] - traj_df[x_col].iat[row_start])
        if ap_disp < turning_min_ap_m:
            df.at[idx, "turning_step"] = True
            df.at[idx, "is_outlier"]   = True

    return df


def _nearest_frame(traj_df: pd.DataFrame, time_s: float) -> int:
    """Return the frame number closest to time_s."""
    idx = (traj_df["time_s"] - time_s).abs().idxmin()
    return int(traj_df["frame"].iat[idx])


# ---------------------------------------------------------------------------
# Turning interval expansion
# DUO-GAIT: features/postprocessing.py lines 20-48
# ---------------------------------------------------------------------------

def _mark_turning_interval(df: pd.DataFrame, interval_size: int) -> pd.DataFrame:
    """
    Replicate DUO-GAIT mark_turning_interval() exactly.
    Source: features/postprocessing.py lines 20-48.

    For each turning_step at index x, marks indices
    [x - interval_size, …, x + interval_size] as turning_interval.
    Also marks the first and last interval_size strides of the session
    (line 45-46 in DUO-GAIT).
    Applied per foot (DUO-GAIT processes each foot's CSV independently).
    """
    for side in ("left", "right"):
        mask    = df["foot"] == side
        sub_idx = df.index[mask].tolist()      # positional indices within df
        if not sub_idx:
            continue

        # Positions of turning steps within this foot's stride list
        ts_positions = [
            i for i, gi in enumerate(sub_idx)
            if df.at[gi, "turning_step"]
        ]

        turning_positions: set[int] = set()
        for pos in ts_positions:
            # interval around each turning step (lines 33-36)
            turning_positions.update(
                range(pos - interval_size, pos + interval_size + 1)
            )

        # Clip to valid range (lines 37-39)
        all_positions = set(range(len(sub_idx)))
        turning_positions &= all_positions

        # Head and tail strides (lines 45-46)
        head_tail = list(range(interval_size)) + list(
            range(len(sub_idx) - interval_size, len(sub_idx))
        )
        turning_positions.update(p for p in head_tail if p in all_positions)

        for pos in turning_positions:
            df.at[sub_idx[pos], "turning_interval"] = True

    return df


# ---------------------------------------------------------------------------
# Interrupted strides
# DUO-GAIT: features/postprocessing.py lines 51-84
# ---------------------------------------------------------------------------

def _mark_interrupted_strides(
    df: pd.DataFrame,
    interruptions_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Mark strides whose timestamps fall within any [start_s, end_s] interval.
    Replicates DUO-GAIT mark_interrupted_strides() (postprocessing.py:51-84).

    interruptions_df must have columns: start_s, end_s.
    (DUO-GAIT uses 'start(s)' and 'end(s)' column names — renamed here for
    Python-friendliness.)
    """
    df["interrupted"] = False
    for _, row in interruptions_df.iterrows():
        start_s = float(row["start_s"])
        end_s   = float(row["end_s"])
        mask = (df["timestamps"] >= start_s) & (df["timestamps"] <= end_s)
        df.loc[mask, "interrupted"] = True
    return df
