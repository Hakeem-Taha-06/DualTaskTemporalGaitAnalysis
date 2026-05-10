"""
gait/video_slicer.py — Video segmentation and TRC stitching utilities.

This module provides three stateless functions for the segmented processing mode:

  1. parse_segments()   — Parse enter/exit CSV into valid time windows
  2. slice_video()      — Extract each segment from source video via FFmpeg
  3. stitch_trc_files() — Merge per-segment TRC files into a single unified TRC

All functions are pure utilities with no dependencies on runners/ or ui/.
They can be tested independently with fixture data.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Time parsing (shared with outlier_remover — same CSV format)
# ---------------------------------------------------------------------------

def _parse_time(t_val) -> float:
    """Parse a time string like '1:13.8' or '26.5' into seconds."""
    t_str = str(t_val).strip()
    if ":" in t_str:
        parts = t_str.split(":")
        total_sec = 0.0
        for part in parts:
            total_sec = total_sec * 60 + float(part)
        return total_sec
    return float(t_str)


# ---------------------------------------------------------------------------
# 1. Parse segments from enter/exit CSV
# ---------------------------------------------------------------------------

def parse_segments(
    csv_path: str | Path,
    min_duration_s: float = 10.0,
) -> list[dict]:
    """
    Parse enter/exit CSV into a list of valid segment windows.

    Parameters
    ----------
    csv_path : str or Path
        Path to the boundaries CSV with columns ``time_s`` and ``event``.
    min_duration_s : float
        Segments shorter than this (in seconds) are silently discarded.

    Returns
    -------
    list[dict]
        Each dict has keys:
        - ``index``     : int   — 0-based segment index (after filtering)
        - ``start_s``   : float — enter timestamp (playback time)
        - ``end_s``     : float — exit timestamp (playback time)
        - ``duration_s``: float — end_s - start_s
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Boundaries CSV not found: {csv_path}")

    try:
        boundaries = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Cannot parse boundaries CSV {csv_path}: {e}")

    if "time_s" not in boundaries.columns or "event" not in boundaries.columns:
        raise ValueError(
            f"Boundaries CSV must have 'time_s' and 'event' columns. "
            f"Found: {list(boundaries.columns)}"
        )

    # Parse all events into an ordered list
    events: list[dict] = []
    for _, row in boundaries.iterrows():
        try:
            time_s = _parse_time(row["time_s"])
        except (ValueError, TypeError):
            continue
        event = str(row["event"]).strip().lower()
        if event in ("enter", "exit"):
            events.append({"time_s": time_s, "event": event})

    # Sort by time
    events.sort(key=lambda e: e["time_s"])

    # Pair each 'enter' with the nearest subsequent 'exit'
    raw_segments: list[dict] = []
    i = 0
    while i < len(events):
        if events[i]["event"] == "enter":
            # Find next exit
            for j in range(i + 1, len(events)):
                if events[j]["event"] == "exit":
                    start_s = events[i]["time_s"]
                    end_s = events[j]["time_s"]
                    raw_segments.append({
                        "start_s": start_s,
                        "end_s": end_s,
                        "duration_s": end_s - start_s,
                    })
                    i = j + 1
                    break
            else:
                # No exit found after this enter — skip
                logger.warning(
                    f"Unpaired 'enter' event at {events[i]['time_s']:.1f}s — skipping"
                )
                i += 1
        else:
            # Skip leading exit events (person starts out of frame)
            i += 1

    # Filter by minimum duration
    valid_segments: list[dict] = []
    for seg in raw_segments:
        if seg["duration_s"] < min_duration_s:
            logger.warning(
                f"Discarding short segment ({seg['duration_s']:.1f}s < {min_duration_s}s): "
                f"{seg['start_s']:.1f}s – {seg['end_s']:.1f}s"
            )
            continue
        seg["index"] = len(valid_segments)
        valid_segments.append(seg)

    if not valid_segments:
        raise ValueError(
            f"No valid segments found in {csv_path} "
            f"(all segments shorter than {min_duration_s}s or no enter/exit pairs)."
        )

    logger.info(
        f"Parsed {len(valid_segments)} valid segments from {csv_path.name} "
        f"(total valid time: {sum(s['duration_s'] for s in valid_segments):.1f}s)"
    )

    return valid_segments


# ---------------------------------------------------------------------------
# 2. Slice video into segments using FFmpeg
# ---------------------------------------------------------------------------

def _find_ffmpeg() -> Path:
    """Locate the ffmpeg executable on the system PATH."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "FFmpeg not found on system PATH. "
            "Install it from https://ffmpeg.org/download.html and ensure "
            "it is available in your PATH."
        )
    return Path(ffmpeg)


def slice_video(
    video_path: str | Path,
    segments: list[dict],
    output_dir: Path,
) -> list[Path]:
    """
    Extract each segment from the source video using FFmpeg stream copy.

    Parameters
    ----------
    video_path : str or Path
        Path to the full source video.
    segments : list[dict]
        Output of :func:`parse_segments`.
    output_dir : Path
        Directory to write segment clips into (creates a ``segments/`` subdir).

    Returns
    -------
    list[Path]
        Paths to the segment video files, in segment order.

    Raises
    ------
    RuntimeError
        If FFmpeg is not found or a segment extraction fails.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    ffmpeg = _find_ffmpeg()
    seg_dir = output_dir / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)

    suffix = video_path.suffix  # preserve original container format
    segment_paths: list[Path] = []

    for seg in segments:
        idx = seg["index"]
        out_file = seg_dir / f"seg_{idx:02d}{suffix}"

        cmd = [
            str(ffmpeg),
            "-y",                           # overwrite if exists
            "-ss", f"{seg['start_s']:.3f}",  # seek to start
            "-to", f"{seg['end_s']:.3f}",    # end position
            "-i", str(video_path),           # input file
            "-c", "copy",                    # stream copy (lossless, fast)
            "-avoid_negative_ts", "make_zero",
            str(out_file),
        ]

        logger.info(
            f"Extracting segment {idx}: "
            f"{seg['start_s']:.1f}s – {seg['end_s']:.1f}s "
            f"({seg['duration_s']:.1f}s) → {out_file.name}"
        )

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg failed extracting segment {idx}:\n{result.stderr[-500:]}"
            )

        if not out_file.exists() or out_file.stat().st_size == 0:
            raise RuntimeError(
                f"FFmpeg produced empty output for segment {idx}: {out_file}"
            )

        segment_paths.append(out_file)

    logger.info(f"Sliced {len(segment_paths)} segments into {seg_dir}")
    return segment_paths


# ---------------------------------------------------------------------------
# 3. Stitch per-segment TRC files into a single unified TRC
# ---------------------------------------------------------------------------

def stitch_trc_files(
    trc_paths: list[Path],
    segments: list[dict],
    output_dir: Path,
    fallback_fps: float = 30.0,
) -> tuple[Path, float]:
    """
    Merge per-segment TRC files into a single TRC with unified timestamps.

    For each segment's TRC, timestamps are offset by the segment's original
    ``start_s`` value so that the merged file uses the original video timeline.

    .. note::
        Frame numbers in the merged TRC are renumbered sequentially (0, 1, 2, …)
        rather than preserving the original video frame numbers.  Timestamps are
        correctly mapped to the original timeline, so all time-based calculations
        remain valid.  Do not use merged frame numbers to index into original
        video frames.

    Parameters
    ----------
    trc_paths : list[Path]
        Ordered list of per-segment TRC file paths.
    segments : list[dict]
        The same segment list used for slicing (provides timestamp offsets).
    output_dir : Path
        Directory to write the merged TRC file.
    fallback_fps : float
        FPS value if TRC header cannot be parsed.

    Returns
    -------
    tuple[Path, float]
        (path_to_merged_trc, detected_fps)
    """
    if len(trc_paths) != len(segments):
        raise ValueError(
            f"Mismatch: {len(trc_paths)} TRC files but {len(segments)} segments"
        )

    merged_path = output_dir / "merged_person00.trc"

    # Read the first TRC to capture the header template
    first_trc = trc_paths[0]
    with open(first_trc, "r") as fh:
        all_lines = fh.readlines()

    # Find the data start line (after the coordinate label line with X1 Y1 Z1)
    header_lines: list[str] = []
    data_start = 0
    detected_fps = fallback_fps

    for i, line in enumerate(all_lines[:10]):
        fields = line.strip().split("\t")
        # Detect fps from numeric header line
        if i > 0:
            try:
                val = float(fields[0])
                detected_fps = val
            except (ValueError, IndexError):
                pass
        # Detect coordinate label line
        if any(f.strip() in ("X1", "Y1", "Z1") for f in fields):
            data_start = i + 1
            header_lines = all_lines[:data_start]
            break

    if data_start == 0:
        # Fallback: assume header is 5 lines
        data_start = 5
        header_lines = all_lines[:data_start]

    # Collect all data rows across segments, offsetting timestamps and frames
    all_data_rows: list[str] = []
    cumulative_frames = 0

    for trc_path, seg in zip(trc_paths, segments):
        with open(trc_path, "r") as fh:
            lines = fh.readlines()

        # Parse data rows (skip header)
        seg_data_start = _find_data_start(lines)
        time_offset = seg["start_s"]

        for line in lines[seg_data_start:]:
            stripped = line.strip()
            if not stripped:
                continue
            fields = stripped.split("\t")
            if len(fields) < 3:
                continue

            try:
                frame = int(float(fields[0]))
                time_s = float(fields[1])
            except (ValueError, IndexError):
                continue

            # Offset frame number and timestamp to original video timeline
            new_frame = frame + cumulative_frames
            new_time = time_s + time_offset

            fields[0] = str(new_frame)
            fields[1] = f"{new_time:.6f}"
            all_data_rows.append("\t".join(fields) + "\n")

        # Count frames in this segment for cumulative offset
        seg_frame_count = len(lines[seg_data_start:])
        # Filter empty lines
        seg_frame_count = sum(
            1 for line in lines[seg_data_start:] if line.strip()
        )
        cumulative_frames += seg_frame_count

    # Update NumFrames in header (line that contains the numeric fps)
    updated_header: list[str] = []
    for i, line in enumerate(header_lines):
        fields = line.strip().split("\t")
        if i > 0:
            try:
                float(fields[0])
                # This is the numeric header line — update NumFrames (3rd field)
                if len(fields) >= 3:
                    fields[2] = str(len(all_data_rows))
                    line = "\t".join(fields) + "\n"
            except (ValueError, IndexError):
                pass
        updated_header.append(line)

    # Write merged TRC
    with open(merged_path, "w") as fh:
        fh.writelines(updated_header)
        fh.writelines(all_data_rows)

    logger.info(
        f"Stitched {len(trc_paths)} TRC files → {merged_path.name} "
        f"({len(all_data_rows)} total frames, fps={detected_fps})"
    )

    return merged_path, detected_fps


def _find_data_start(lines: list[str]) -> int:
    """Find the first data row index in a TRC file."""
    for i, line in enumerate(lines[:10]):
        fields = line.strip().split("\t")
        if any(f.strip() in ("X1", "Y1", "Z1") for f in fields):
            return i + 1
    return 5  # fallback


# ---------------------------------------------------------------------------
# 4. Cleanup utility
# ---------------------------------------------------------------------------

def cleanup_segments(output_dir: Path, keep_trc: bool = False) -> None:
    """
    Remove temporary segment files after successful stitching.

    Parameters
    ----------
    output_dir : Path
        The session output directory containing the ``segments/`` subdirectory.
    keep_trc : bool
        If True, keep individual segment TRC files (for debugging).
    """
    seg_dir = output_dir / "segments"
    if not seg_dir.exists():
        return

    # Always remove segment video clips (they're just temp files)
    for vid in seg_dir.glob("seg_*.*"):
        if vid.suffix in (".mp4", ".avi", ".mov", ".mkv"):
            vid.unlink(missing_ok=True)

    if not keep_trc:
        # Remove segment subdirectories (contain individual TRCs)
        for sub in seg_dir.iterdir():
            if sub.is_dir():
                shutil.rmtree(sub, ignore_errors=True)

    # Remove segments dir if empty
    try:
        seg_dir.rmdir()
    except OSError:
        pass  # not empty — that's fine
