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

import numpy as np
import pandas as pd




def remove_outliers(
    strides_df: pd.DataFrame,
    traj_df: pd.DataFrame,
    boundaries_csv: str = "",
    speed_factor: float = 1.0,
) -> pd.DataFrame:
    """
    Post-process stride data by marking boundary strides for exclusion.

    Turning detection and head/tail trimming are disabled (turns happen
    off-camera).  The only active exclusion is **boundary stride marking**:
    the first stride after each frame entrance and the last stride before
    each frame exit are marked ``is_outlier = True``.

    Parameters
    ----------
    strides_df : pd.DataFrame
        Output of parameter_calculator.calculate_parameters().
    traj_df : pd.DataFrame
        Filtered trajectory (from preprocessor).  Not used currently but
        kept for API compatibility.
    boundaries_csv : str
        Path to a CSV with columns ``time_s, event`` where event is
        ``"enter"`` or ``"exit"``.  Empty string → no boundary exclusion.
    speed_factor : float
        Playback speed factor, used to scale the boundary margin.

    Returns
    -------
    pd.DataFrame
        Stride DataFrame with added columns: turning_interval, interrupted,
        removal_reason.  Boundary strides have ``is_outlier = True``.
    """
    if strides_df.empty:
        return strides_df.copy()

    df = strides_df.copy()

    # Add expected columns with default (no-exclusion) values
    if "is_outlier"   not in df.columns: df["is_outlier"]   = False
    if "turning_step" not in df.columns: df["turning_step"] = False
    df["turning_interval"] = False
    df["interrupted"]      = False
    df["removal_reason"]   = ""

    # Label pre-existing outliers (from parameter_calculator.flag_outliers)
    df.loc[df["is_outlier"] == True, "removal_reason"] = "threshold"

    # ------------------------------------------------------------------
    # Boundary stride exclusion
    # ------------------------------------------------------------------
    if boundaries_csv:
        df = _mark_boundary_strides(df, boundaries_csv, speed_factor=speed_factor)

    return df


# ---------------------------------------------------------------------------
# Boundary stride exclusion
# ---------------------------------------------------------------------------

def _mark_boundary_strides(
    df: pd.DataFrame,
    csv_path: str,
    margin_s: float = 1.0,
    speed_factor: float = 1.0,
) -> pd.DataFrame:
    """
    Parse an enter/exit timestamp CSV and mark boundary strides.

    CSV format::

        time_s,event
        0.00,enter
        5.20,exit
        5.80,enter
        10.50,exit

    **Exclusion logic (three rules):**

    1. For each ``enter`` event, all strides within ``[enter, enter + margin_s]``
       are marked as outliers (person still accelerating / partially in frame).

    2. For each ``exit`` event, all strides within ``[exit - margin_s, exit]``
       are marked as outliers (person decelerating / partially out of frame).

    3. All strides between an ``exit`` and the next ``enter`` (the dead zone
       where the person is fully out of frame) are marked as outliers.

    Parameters
    ----------
    margin_s : float
        Seconds of data to exclude around each boundary event.  Default 1.0.
        Scaled by speed_factor for slow-motion recordings.
    speed_factor : float
        Playback speed factor (e.g. 8.0 for 240fps→30fps slow-motion).
        The margin is multiplied by this to maintain the same real-time
        exclusion window regardless of playback speed.
    """
    from pathlib import Path

    # Scale margin for slow-motion: 1.0s video-time at 8× = 0.125s real-time,
    # so we multiply by speed_factor to keep the real-time margin constant
    effective_margin = margin_s * speed_factor

    path = Path(csv_path)
    if not path.exists() or path.stat().st_size == 0:
        return df

    try:
        boundaries = pd.read_csv(path)
    except Exception:
        return df  # silently skip unparseable files

    if "time_s" not in boundaries.columns or "event" not in boundaries.columns:
        return df

    def parse_time(t_val) -> float:
        t_str = str(t_val).strip()
        if ":" in t_str:
            parts = t_str.split(":")
            total_sec = 0.0
            for part in parts:
                total_sec = total_sec * 60 + float(part)
            return total_sec
        return float(t_str)

    # Parse all events into an ordered list
    events: list[dict] = []
    for _, row in boundaries.iterrows():
        try:
            time_s = parse_time(row["time_s"])
        except ValueError:
            continue
        event = str(row["event"]).strip().lower()
        if event in ("enter", "exit"):
            events.append({"time_s": time_s, "event": event})

    # Sort by time to ensure correct gap detection
    events.sort(key=lambda e: e["time_s"])

    # Rule 1 & 2: margin windows around each event
    for ev in events:
        t = ev["time_s"]
        if ev["event"] == "enter":
            mask = (df["timestamps"] >= t) & (df["timestamps"] <= t + effective_margin)
            df.loc[mask, "is_outlier"] = True
            df.loc[mask, "removal_reason"] = "boundary_enter"
        elif ev["event"] == "exit":
            mask = (df["timestamps"] >= t - effective_margin) & (df["timestamps"] <= t)
            df.loc[mask, "is_outlier"] = True
            df.loc[mask, "removal_reason"] = "boundary_exit"

    # Rule 3: mark all strides in dead zones (between exit → next enter)
    for i in range(len(events) - 1):
        if events[i]["event"] == "exit" and events[i + 1]["event"] == "enter":
            gap_start = events[i]["time_s"]
            gap_end   = events[i + 1]["time_s"]
            mask = (df["timestamps"] >= gap_start) & (df["timestamps"] <= gap_end)
            df.loc[mask, "is_outlier"] = True
            df.loc[mask & (df["removal_reason"] == ""), "removal_reason"] = "out_of_frame"

    return df

