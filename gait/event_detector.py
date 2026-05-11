"""
Module 3 — event_detector.py
Responsibility: Detect heel-strike (HS = IC) and toe-off (TO = FO) gait events
from filtered keypoint trajectories, replicating the logical structure of the
DUO-GAIT / Tunca event detector as closely as the different signal domain allows.

=== DUO-GAIT approach (IMU domain) ===
Source: LFRF_parameters/event_detection/imu_event_detection.py
  1. Stance phases identified as periods where gyroscope magnitude < threshold
     (gyro_threshold_stance, line 6-51).
  2. Step boundaries = transitions from swing to stance (step_begins, line 74).
  3. Within each step, a tilt signal is computed from the highest-variance gyro
     axis integrated over time (lines 88-94).
  4. IC (Initial Contact / heel-strike) and FO (Foot Off / toe-off) are found
     as peaks in ±tilt_diff within search regions bounded by tilt peaks
     (lines 99-214).
  - prominence thresholds: fo_prom_threshold=1.5, ic_prom_threshold=0.1
  - search region prominence: fo/ic_search_threshold=0.7

=== Video-domain equivalent (this module) ===
Signal analogies:
  IMU gyro magnitude ↔ foot speed derived from heel/toe displacement
  Stance (gyro < threshold) ↔ heel-Y near ground (below adaptive threshold)
  IC (heel-strike) ↔ local MINIMUM in heel-Y trajectory
      (heel is lowest when it contacts the ground; Y-up convention)
  FO (toe-off) ↔ local MINIMUM in toe-Y trajectory just before toe lifts
      (toe is at its lowest ground contact point before swing onset)

Peak detection uses scipy.signal.find_peaks with prominence thresholds tuned
to produce results equivalent to the Tunca algorithm at 120 fps.

Output schema:
    pd.DataFrame with columns:
        foot        str   — 'left' or 'right'
        event_type  str   — 'HS' (heel-strike / IC) or 'TO' (toe-off / FO)
        frame       int   — frame index in the trajectory DataFrame
        time_s      float — timestamp in seconds

This module is replaceable: any event detector returning this schema can be
dropped in without changing downstream modules.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# Peak-detection parameters (tuned for 120 fps; scale with fps if needed)
# ---------------------------------------------------------------------------

# Minimum vertical prominence (metres) for a heel-Y local minimum to count as IC.
# Equivalent to requiring a clear foot-lift above ground contact level.
# NOT FOUND IN DUO-GAIT SOURCE (domain-translated) — ASSUMPTION: 0.005 m
_IC_PROMINENCE_M = 0.005

# Minimum vertical prominence (metres) for a toe-Y local minimum to count as FO.
# Toe-Y has more noise than heel-Y, so a higher threshold is needed.
# NOT FOUND IN DUO-GAIT SOURCE (domain-translated) — ASSUMPTION: 0.008 m
_FO_PROMINENCE_M = 0.008

# Minimum seconds between successive IC events of the same foot.
# A stride shorter than 0.70 s is physiologically implausible for normal
# walking; guards against double-detections within the same gait cycle.
# NOT FOUND IN DUO-GAIT SOURCE — ASSUMPTION based on gait physiology
_MIN_IC_DISTANCE_S = 0.70  # seconds (real-time minimum between heel strikes)

# Minimum seconds between successive FO events (real-time)
_MIN_FO_DISTANCE_S = 0.70  # seconds

# ---------------------------------------------------------------------------
# AP-coordinate detector — adaptive prominence constant
# ---------------------------------------------------------------------------

# Fraction of the detrended AP signal's standard deviation used as the
# minimum peak prominence for heel-strike and toe-off detection.
# A larger value requires more pronounced peaks (fewer, more confident
# detections); a smaller value accepts subtler peaks (more detections,
# higher false-positive risk).  Because both the gait-cycle amplitude
# and pose-estimation noise scale with subject height and stride length,
# expressing the threshold as a fraction of signal SD makes it
# approximately subject-invariant without per-participant tuning.
_AP_PROMINENCE_K = 0.4


def detect_events(
    traj_df: pd.DataFrame,
    fps: float = 120.0,
    speed_factor: float = 1.0,
) -> pd.DataFrame:
    """
    Detect heel-strike (HS) and toe-off (TO) gait events from trajectory data
    using **vertical-axis (Y) minima**.

    Parameters
    ----------
    traj_df : pd.DataFrame
        Output of preprocessor.preprocess() — standardised, filtered trajectory.
        Must contain: frame, time_s, left_heel_y, right_heel_y,
                      left_toe_y, right_toe_y.
    fps : float
        Sampling rate (Hz) of the video (playback fps, NOT recording fps).
    speed_factor : float
        Slow-motion correction factor.  If the video was recorded at 240 fps
        and plays at 30 fps, speed_factor = 8.0.  A real-time 1-second stride
        appears as 8 seconds (240 frames) in the video.  The minimum-distance
        parameters are scaled by this factor so that events are not
        over-detected on slow-motion footage.

    Returns
    -------
    pd.DataFrame
        Columns: foot (str), event_type (str), frame (int), time_s (float).
        Sorted by time_s.
    """
    events: list[dict] = []

    for side in ("left", "right"):
        heel_y = traj_df[f"{side}_heel_y"].values.astype(float)
        toe_y  = traj_df[f"{side}_toe_y"].values.astype(float)
        frames = traj_df["frame"].values
        times  = traj_df["time_s"].values

        # Interpolate NaNs before peak detection (same spirit as DUO-GAIT
        # which operates on continuously sampled IMU data without NaN)
        heel_y = _interpolate_nans(heel_y)
        toe_y  = _interpolate_nans(toe_y)

        # Scale min-distance parameter to account for playback fps AND
        # slow-motion factor.  A real-time minimum of _MIN_IC_DISTANCE_S
        # translates to (min_s * speed_factor * fps) frames in the video.
        ic_dist = max(1, int(_MIN_IC_DISTANCE_S * speed_factor * fps))
        fo_dist = max(1, int(_MIN_FO_DISTANCE_S * speed_factor * fps))

        # ------------------------------------------------------------------
        # IC (heel-strike): local MINIMUM in heel_y
        # find_peaks works on maxima; invert signal to find minima
        # ------------------------------------------------------------------
        ic_idx, _ = find_peaks(
            -heel_y,
            prominence=_IC_PROMINENCE_M,
            distance=ic_dist,
        )

        for idx in ic_idx:
            events.append({
                "foot":       side,
                "event_type": "HS",
                "frame":      int(frames[idx]),
                "time_s":     float(times[idx]),
            })

        # ------------------------------------------------------------------
        # FO (toe-off): local MINIMUM in toe_y
        # ------------------------------------------------------------------
        fo_idx, _ = find_peaks(
            -toe_y,
            prominence=_FO_PROMINENCE_M,
            distance=fo_dist,
        )

        for idx in fo_idx:
            events.append({
                "foot":       side,
                "event_type": "TO",
                "frame":      int(frames[idx]),
                "time_s":     float(times[idx]),
            })

    events_df = pd.DataFrame(events, columns=["foot", "event_type", "frame", "time_s"])
    events_df.sort_values("time_s", inplace=True, ignore_index=True)
    return events_df


# ---------------------------------------------------------------------------
# AP-coordinate event detector
# ---------------------------------------------------------------------------

def detect_events_ap(
    traj_df: pd.DataFrame,
    fps: float = 120.0,
    speed_factor: float = 1.0,
    prominence_k: float | None = None,
    boundaries_csv: str = "",
) -> pd.DataFrame:
    """
    Detect heel-strike (HS) and toe-off (TO) gait events from trajectory data
    using **anterior-posterior (X) coordinate peak detection** on a
    linearly-detrended signal.

    Detrending is performed **independently on each walking segment** defined
    by enter/exit boundary pairs.  This is essential because subjects walk
    back and forth, creating a zigzag X-trajectory whose global linear trend
    is meaningless.

    Within each segment:

    1. Walking direction is auto-detected from the sign of the mean heel_x
       velocity and the signal is negated if the subject walks right-to-left.
    2. A degree-1 polynomial is subtracted so the residual oscillates around
       zero with clear peaks (heel most-forward at HS) and troughs (toe
       most-rearward at TO).
    3. Peaks are detected with adaptive prominence ``k × std(detrended)``.

    Parameters
    ----------
    traj_df : pd.DataFrame
        Output of preprocessor.preprocess() — standardised, filtered trajectory.
    fps : float
        Sampling rate (Hz) of the video (playback fps).
    speed_factor : float
        Slow-motion correction factor (same semantics as ``detect_events``).
    prominence_k : float or None
        Override for the adaptive prominence constant K.  When *None*,
        the module-level ``_AP_PROMINENCE_K`` constant is used.
    boundaries_csv : str
        Path to enter/exit CSV.  Empty string → treat entire trajectory as
        one segment.

    Returns
    -------
    pd.DataFrame
        Columns: foot (str), event_type (str), frame (int), time_s (float).
        Sorted by time_s.  Same schema as ``detect_events()``.
    """
    import logging

    k = prominence_k if prominence_k is not None else _AP_PROMINENCE_K

    # -- Parse boundaries into (enter, exit) segment pairs ----------------
    segments = _parse_boundary_segments(boundaries_csv, traj_df["time_s"].values)

    events: list[dict] = []
    frames_all = traj_df["frame"].values
    times_all  = traj_df["time_s"].values.astype(float)

    # Min-distance between successive events (same scaling as vertical detector)
    ic_dist = max(1, int(_MIN_IC_DISTANCE_S * speed_factor * fps))
    fo_dist = max(1, int(_MIN_FO_DISTANCE_S * speed_factor * fps))

    for seg_start, seg_end in segments:
        # Boolean mask for rows inside this segment
        seg_mask = (times_all >= seg_start) & (times_all <= seg_end)
        if seg_mask.sum() < 4:
            continue  # too few samples to detrend

        seg_indices = np.where(seg_mask)[0]
        seg_times  = times_all[seg_indices]
        seg_frames = frames_all[seg_indices]

        for side in ("left", "right"):
            heel_x = traj_df[f"{side}_heel_x"].values[seg_indices].astype(float)
            toe_x  = traj_df[f"{side}_toe_x"].values[seg_indices].astype(float)

            heel_x = _interpolate_nans(heel_x)
            toe_x  = _interpolate_nans(toe_x)

            # -- Auto-detect walking direction for THIS segment -----------
            mean_vel = np.diff(heel_x).mean() if len(heel_x) > 1 else 0.0
            if mean_vel < 0:
                heel_x = -heel_x
                toe_x  = -toe_x

            # -- Detrend within this segment only -------------------------
            detrended_heel = _detrend_linear(seg_times, heel_x)
            detrended_toe  = _detrend_linear(seg_times, toe_x)

            # -- Adaptive prominence --------------------------------------
            heel_std = np.std(detrended_heel)
            toe_std  = np.std(detrended_toe)
            hs_prom  = k * heel_std if heel_std > 0 else 0.01
            to_prom  = k * toe_std  if toe_std  > 0 else 0.01

            # -- HS: local MAXIMA in detrended heel_x ---------------------
            ic_idx, _ = find_peaks(
                detrended_heel,
                prominence=hs_prom,
                distance=ic_dist,
            )

            for idx in ic_idx:
                events.append({
                    "foot":       side,
                    "event_type": "HS",
                    "frame":      int(seg_frames[idx]),
                    "time_s":     float(seg_times[idx]),
                })

            # -- TO: local MINIMA in detrended toe_x ----------------------
            fo_idx, _ = find_peaks(
                -detrended_toe,
                prominence=to_prom,
                distance=fo_dist,
            )

            for idx in fo_idx:
                events.append({
                    "foot":       side,
                    "event_type": "TO",
                    "frame":      int(seg_frames[idx]),
                    "time_s":     float(seg_times[idx]),
                })

        logging.debug(
            "AP detector segment [%.2f–%.2f s]: %d events",
            seg_start, seg_end, sum(
                1 for e in events
                if seg_start <= e["time_s"] <= seg_end
            ),
        )

    events_df = pd.DataFrame(events, columns=["foot", "event_type", "frame", "time_s"])
    events_df.sort_values("time_s", inplace=True, ignore_index=True)

    if events_df.empty:
        logging.warning(
            "AP detector: zero events across %d segments", len(segments)
        )

    return events_df


def _parse_boundary_segments(
    csv_path: str,
    all_times: np.ndarray,
) -> list[tuple[float, float]]:
    """Parse a boundary CSV into (enter, exit) time pairs.

    If *csv_path* is empty or unreadable, returns a single segment spanning
    the entire trajectory.
    """
    from pathlib import Path

    # Fallback: whole trajectory is one segment
    if not csv_path:
        return [(float(all_times[0]), float(all_times[-1]))]

    path = Path(csv_path)
    if not path.exists() or path.stat().st_size == 0:
        return [(float(all_times[0]), float(all_times[-1]))]

    try:
        bdf = pd.read_csv(path)
    except Exception:
        return [(float(all_times[0]), float(all_times[-1]))]

    if "time_s" not in bdf.columns or "event" not in bdf.columns:
        return [(float(all_times[0]), float(all_times[-1]))]

    # Parse and sort events
    parsed: list[dict] = []
    for _, row in bdf.iterrows():
        try:
            t_str = str(row["time_s"]).strip()
            if ":" in t_str:
                parts = t_str.split(":")
                t = 0.0
                for part in parts:
                    t = t * 60 + float(part)
            else:
                t = float(t_str)
        except ValueError:
            continue
        ev = str(row["event"]).strip().lower()
        if ev in ("enter", "exit"):
            parsed.append({"time_s": t, "event": ev})
    parsed.sort(key=lambda e: e["time_s"])

    # Build enter-exit pairs
    segments: list[tuple[float, float]] = []
    i = 0
    while i < len(parsed):
        if parsed[i]["event"] == "enter":
            enter_t = parsed[i]["time_s"]
            # Find matching exit
            exit_t = float(all_times[-1])  # default: end of trajectory
            if i + 1 < len(parsed) and parsed[i + 1]["event"] == "exit":
                exit_t = parsed[i + 1]["time_s"]
                i += 2
            else:
                i += 1
            segments.append((enter_t, exit_t))
        else:
            i += 1  # skip orphan exit

    if not segments:
        return [(float(all_times[0]), float(all_times[-1]))]

    return segments


def events_to_gait_event_dict(events_df: pd.DataFrame) -> dict:
    """
    Convert the flat events DataFrame into the nested dict structure used
    internally by GaitParameters (mirrors DUO-GAIT's event_detector output).

    Structure (matches DUO-GAIT event_detector.py:47-51):
        {
            "stance_begin": "IC",
            "stance_end":   "FO",
            "left":  {"samples": {"IC": np.array, "FO": np.array},
                      "times":   {"IC": list,     "FO": list}},
            "right": { … }
        }

    Parameters
    ----------
    events_df : pd.DataFrame
        Output of detect_events().

    Returns
    -------
    dict
        Nested gait event dictionary compatible with parameter_calculator.py.
    """
    result: dict = {
        "stance_begin": "IC",
        "stance_end":   "FO",
    }

    for side in ("left", "right"):
        side_df = events_df[events_df["foot"] == side].copy()
        ic_df   = side_df[side_df["event_type"] == "HS"].sort_values("time_s")
        fo_df   = side_df[side_df["event_type"] == "TO"].sort_values("time_s")

        result[side] = {
            "samples": {
                "IC": ic_df["frame"].to_numpy(dtype=int),
                "FO": fo_df["frame"].to_numpy(dtype=int),
            },
            "times": {
                "IC": ic_df["time_s"].tolist(),
                "FO": fo_df["time_s"].tolist(),
            },
        }

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _interpolate_nans(signal: np.ndarray) -> np.ndarray:
    """Linear interpolation of NaN values in a 1-D array."""
    if not np.any(np.isnan(signal)):
        return signal
    idx = np.arange(len(signal))
    nan_mask = np.isnan(signal)
    if nan_mask.all():
        return np.zeros_like(signal)
    return np.interp(idx, idx[~nan_mask], signal[~nan_mask])


def _detrend_linear(times: np.ndarray, signal: np.ndarray) -> np.ndarray:
    """Remove the linear trend from *signal* sampled at *times*.

    Uses a degree-1 polynomial fit (np.polyfit) and subtracts the trend
    so that the residual oscillates around zero.  This is used by the AP
    detector to extract the periodic gait-cycle component from the
    monotonically increasing (or decreasing) position signal.
    """
    if len(signal) < 2:
        return signal.copy()
    coeffs = np.polyfit(times, signal, deg=1)
    trend = np.polyval(coeffs, times)
    return signal - trend

