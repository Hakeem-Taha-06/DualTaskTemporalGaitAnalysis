"""
pipeline_runner.py
QThread-based runner that executes the gait analysis pipeline with strict
sequential stage ordering and dependency checking as specified in Addendum 2.

Stage dependency graph (Addendum 2, "Sequential Execution" section):
    1.  sports2d_st        depends_on: []
    2.  sports2d_dt        depends_on: []
    3.  load_st            depends_on: [sports2d_st]
    4.  load_dt            depends_on: [sports2d_dt]
    5.  preprocess_st      depends_on: [load_st]
    6.  preprocess_dt      depends_on: [load_dt]
    7.  detect_events_st   depends_on: [preprocess_st]
    8.  detect_events_dt   depends_on: [preprocess_dt]
    9.  calc_params_st     depends_on: [detect_events_st]
    10. calc_params_dt     depends_on: [detect_events_dt]
    11. remove_outliers_st depends_on: [calc_params_st]
    12. remove_outliers_dt depends_on: [calc_params_dt]
    13. aggregate_st       depends_on: [remove_outliers_st]
    14. aggregate_dt       depends_on: [remove_outliers_dt]
    15. dtc                depends_on: [aggregate_st, aggregate_dt]

ST and DT branches are independent; each branch is run sequentially.
Sports2D stages are skipped if a .trc file is provided directly.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

try:
    from PySide6.QtCore import QThread, Signal
except ImportError:
    from PyQt6.QtCore import QThread, pyqtSignal as Signal

# Pipeline modules
sys.path.insert(0, str(Path(__file__).parent))
import input_loader
import preprocessor
import event_detector
import parameter_calculator
import outlier_remover
import aggregator
import dtc_calculator


# ---------------------------------------------------------------------------
# Stage status enum (Addendum 2)
# ---------------------------------------------------------------------------

class StageStatus(Enum):
    PENDING  = "pending"
    RUNNING  = "running"
    COMPLETE = "complete"
    FAILED   = "failed"
    SKIPPED  = "skipped"

# Status display characters for the stage checklist UI
STATUS_ICON = {
    StageStatus.PENDING:  "·",
    StageStatus.RUNNING:  "⟳",
    StageStatus.COMPLETE: "✓",
    StageStatus.FAILED:   "✗",
    StageStatus.SKIPPED:  "—",
}


# ---------------------------------------------------------------------------
# PipelineStage dataclass (Addendum 2)
# ---------------------------------------------------------------------------

@dataclass
class PipelineStage:
    name:       str
    depends_on: list[str]
    status:     StageStatus = StageStatus.PENDING
    result:     Any         = None
    error:      Optional[str] = None


# ---------------------------------------------------------------------------
# PipelineRunner QThread
# ---------------------------------------------------------------------------

class PipelineRunner(QThread):
    """
    Executes the full gait analysis pipeline in a background thread.

    Signals
    -------
    progress(stage_name: str, status: str, percent: int)
        Emitted when a stage starts, completes, or fails.
    finished(results: dict)
        Emitted on successful completion with all stage outputs.
    error(message: str)
        Emitted on unrecoverable failure.
    """
    progress = Signal(str, str, int)   # stage_name, status, percent
    finished = Signal(dict)
    error    = Signal(str)

    def __init__(
        self,
        participant_id: str,
        st_input:  str,          # path to .trc OR video file
        dt_input:  str,
        st_is_video: bool,
        dt_is_video: bool,
        height_m:  float = 1.70,
        fps:       float = 120.0,
        output_dir: str = "",
        apply_filter: bool = False,
        interruptions_df: Optional[pd.DataFrame] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.participant_id   = participant_id
        self.st_input         = st_input
        self.dt_input         = dt_input
        self.st_is_video      = st_is_video
        self.dt_is_video      = dt_is_video
        self.height_m         = height_m
        self.fps              = fps
        self.output_dir       = Path(output_dir) if output_dir else Path(tempfile.mkdtemp())
        self.apply_filter     = apply_filter
        self.interruptions_df = interruptions_df

        self._stages = self._build_stage_graph()
        self._results: dict = {}
        self._detected_fps: dict[str, float] = {}  # per-condition fps from TRC header
        self._annotated_videos: dict[str, str] = {}  # per-condition annotated video paths

    # ------------------------------------------------------------------
    # Stage graph construction
    # ------------------------------------------------------------------

    def _build_stage_graph(self) -> dict[str, PipelineStage]:
        stages = [
            PipelineStage("sports2d_st",        []),
            PipelineStage("sports2d_dt",        []),
            PipelineStage("load_st",            ["sports2d_st"]),
            PipelineStage("load_dt",            ["sports2d_dt"]),
            PipelineStage("preprocess_st",      ["load_st"]),
            PipelineStage("preprocess_dt",      ["load_dt"]),
            PipelineStage("detect_events_st",   ["preprocess_st"]),
            PipelineStage("detect_events_dt",   ["preprocess_dt"]),
            PipelineStage("calc_params_st",     ["detect_events_st"]),
            PipelineStage("calc_params_dt",     ["detect_events_dt"]),
            PipelineStage("remove_outliers_st", ["calc_params_st"]),
            PipelineStage("remove_outliers_dt", ["calc_params_dt"]),
            PipelineStage("aggregate_st",       ["remove_outliers_st"]),
            PipelineStage("aggregate_dt",       ["remove_outliers_dt"]),
            PipelineStage("dtc",               ["aggregate_st", "aggregate_dt"]),
        ]
        return {s.name: s for s in stages}

    # ------------------------------------------------------------------
    # QThread entry
    # ------------------------------------------------------------------

    def run(self):
        try:
            # Pre-skip Sports2D stages if .trc provided directly
            if not self.st_is_video:
                self._stages["sports2d_st"].status = StageStatus.SKIPPED
            if not self.dt_is_video:
                self._stages["sports2d_dt"].status = StageStatus.SKIPPED

            # Execution order matches Addendum 2 stage list
            order = [
                "sports2d_st", "sports2d_dt",
                "load_st",     "load_dt",
                "preprocess_st", "preprocess_dt",
                "detect_events_st", "detect_events_dt",
                "calc_params_st", "calc_params_dt",
                "remove_outliers_st", "remove_outliers_dt",
                "aggregate_st", "aggregate_dt",
                "dtc",
            ]
            total = len(order)

            for i, name in enumerate(order):
                stage = self._stages[name]
                pct   = int((i / total) * 100)

                # Skip already-resolved stages
                if stage.status in (StageStatus.COMPLETE, StageStatus.SKIPPED):
                    self.progress.emit(name, stage.status.value, pct)
                    continue

                # Check dependencies
                for dep in stage.depends_on:
                    dep_stage = self._stages[dep]
                    if dep_stage.status == StageStatus.FAILED:
                        stage.status = StageStatus.SKIPPED
                        stage.error  = f"Dependency '{dep}' failed."
                        self.progress.emit(name, stage.status.value, pct)
                        break
                    if dep_stage.status not in (StageStatus.COMPLETE,
                                                StageStatus.SKIPPED):
                        stage.status = StageStatus.FAILED
                        stage.error  = f"Dependency '{dep}' not complete."
                        self.error.emit(stage.error)
                        return
                else:
                    # All deps satisfied — run stage
                    stage.status = StageStatus.RUNNING
                    self.progress.emit(name, stage.status.value, pct)
                    try:
                        result = self._execute_stage(name)
                        stage.result = result
                        self._results[name] = result
                        stage.status = StageStatus.COMPLETE
                        self.progress.emit(name, stage.status.value,
                                           int(((i + 1) / total) * 100))
                    except Exception as exc:
                        stage.status = StageStatus.FAILED
                        stage.error  = str(exc)
                        import traceback
                        tb = traceback.format_exc()
                        self.error.emit(f"Stage '{name}' failed:\n{tb}")
                        return

            self.finished.emit(self._results)

        except Exception as exc:
            import traceback
            self.error.emit(traceback.format_exc())

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    def _execute_stage(self, name: str) -> Any:
        out = self.output_dir / self.participant_id
        out.mkdir(parents=True, exist_ok=True)

        # ── Sports2D ──────────────────────────────────────────────────
        if name == "sports2d_st":
            s2d = self._run_sports2d(self.st_input, out, "st")
            self._annotated_videos["st"] = s2d.get("video", "")
            return s2d

        if name == "sports2d_dt":
            s2d = self._run_sports2d(self.dt_input, out, "dt")
            self._annotated_videos["dt"] = s2d.get("video", "")
            return s2d

        # ── Load ──────────────────────────────────────────────────────
        if name == "load_st":
            trc = self._resolve_trc("st")
            df, detected_fps = input_loader.load_trc(trc, fps=self.fps)
            self._detected_fps["st"] = detected_fps
            df.to_csv(out / "01_raw_trajectories_st.csv", index=False)
            return df

        if name == "load_dt":
            trc = self._resolve_trc("dt")
            df, detected_fps = input_loader.load_trc(trc, fps=self.fps)
            self._detected_fps["dt"] = detected_fps
            df.to_csv(out / "01_raw_trajectories_dt.csv", index=False)
            return df

        # ── Preprocess ────────────────────────────────────────────────
        if name == "preprocess_st":
            df = preprocessor.preprocess(
                self._results["load_st"], fps=self._get_fps("st"),
                apply_filter=self.apply_filter
            )
            return df

        if name == "preprocess_dt":
            df = preprocessor.preprocess(
                self._results["load_dt"], fps=self._get_fps("dt"),
                apply_filter=self.apply_filter
            )
            return df

        # ── Event detection ───────────────────────────────────────────
        if name == "detect_events_st":
            df = event_detector.detect_events(
                self._results["preprocess_st"], fps=self._get_fps("st")
            )
            df.to_csv(out / "02_events_st.csv", index=False)
            return df

        if name == "detect_events_dt":
            df = event_detector.detect_events(
                self._results["preprocess_dt"], fps=self._get_fps("dt")
            )
            df.to_csv(out / "02_events_dt.csv", index=False)
            return df

        # ── Parameter calculation ─────────────────────────────────────
        if name == "calc_params_st":
            gait_ev = event_detector.events_to_gait_event_dict(
                self._results["detect_events_st"]
            )
            self._results["gait_events_st"] = gait_ev
            df = parameter_calculator.calculate_parameters(
                gait_ev, self._results["preprocess_st"]
            )
            df.to_csv(out / "03_strides_raw_st.csv", index=False)
            return df

        if name == "calc_params_dt":
            gait_ev = event_detector.events_to_gait_event_dict(
                self._results["detect_events_dt"]
            )
            self._results["gait_events_dt"] = gait_ev
            df = parameter_calculator.calculate_parameters(
                gait_ev, self._results["preprocess_dt"]
            )
            df.to_csv(out / "03_strides_raw_dt.csv", index=False)
            return df

        # ── Outlier removal ───────────────────────────────────────────
        if name == "remove_outliers_st":
            df = outlier_remover.remove_outliers(
                self._results["calc_params_st"],
                self._results["preprocess_st"],
                interruptions_df=self.interruptions_df,
            )
            df.to_csv(out / "04_strides_cleaned_st.csv", index=False)
            return df

        if name == "remove_outliers_dt":
            df = outlier_remover.remove_outliers(
                self._results["calc_params_dt"],
                self._results["preprocess_dt"],
                interruptions_df=self.interruptions_df,
            )
            df.to_csv(out / "04_strides_cleaned_dt.csv", index=False)
            return df

        # ── Aggregation ───────────────────────────────────────────────
        if name == "aggregate_st":
            df = aggregator.aggregate(
                self._results["remove_outliers_st"],
                participant_id=self.participant_id,
                condition="st",
            )
            df.to_csv(out / "05_aggregated_st.csv", index=False)
            return df

        if name == "aggregate_dt":
            df = aggregator.aggregate(
                self._results["remove_outliers_dt"],
                participant_id=self.participant_id,
                condition="dt",
            )
            df.to_csv(out / "05_aggregated_dt.csv", index=False)
            return df

        # ── DTC ───────────────────────────────────────────────────────
        if name == "dtc":
            dtc_df  = dtc_calculator.calculate_dtc(
                self._results["aggregate_st"],
                self._results["aggregate_dt"],
            )
            dtc_sum = dtc_calculator.dtc_summary_table(dtc_df)
            dtc_df.to_csv(out / "06_dtc.csv",         index=False)
            dtc_sum.to_csv(out / "07_dtc_summary.csv", index=False)
            return {"dtc": dtc_df, "dtc_summary": dtc_sum}

        raise ValueError(f"Unknown stage: {name}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_trc(self, cond: str) -> str:
        """Return the TRC path — either directly provided or from Sports2D output."""
        if cond == "st":
            if not self.st_is_video:
                return self.st_input
            s2d = self._results.get("sports2d_st", {})
            return s2d["trc"] if isinstance(s2d, dict) else str(s2d)
        else:
            if not self.dt_is_video:
                return self.dt_input
            s2d = self._results.get("sports2d_dt", {})
            return s2d["trc"] if isinstance(s2d, dict) else str(s2d)

    def _get_fps(self, cond: str) -> float:
        """Return the fps detected from the TRC header, falling back to UI value."""
        return self._detected_fps.get(cond, self.fps)

    def _run_sports2d(self, video_path: str, out_dir: Path, cond: str) -> dict:
        """
        Run Sports2D as a subprocess on a video file.
        Returns a dict with 'trc' (path to TRC) and 'video' (path to
        annotated video, or empty string if not found).
        """
        session_dir = out_dir / f"sports2d_{cond}"
        session_dir.mkdir(parents=True, exist_ok=True)

        # Resolve the sports2d console script installed alongside this Python
        scripts_dir = Path(sys.executable).parent
        sports2d_exe = scripts_dir / "sports2d.exe"
        if not sports2d_exe.exists():
            # Fallback: try without .exe (Linux/macOS)
            sports2d_exe = scripts_dir / "sports2d"
        if not sports2d_exe.exists():
            raise FileNotFoundError(
                f"Cannot find sports2d executable in {scripts_dir}. "
                f"Install it with: pip install sports2d"
            )

        cmd = [
            str(sports2d_exe),
            "--video_input",          video_path,
            "--save_pose",            "true",
            "--to_meters",            "true",
            "--first_person_height",  str(self.height_m),
            "--result_dir",           str(session_dir),
            "--save_vid",             "true",
            "--save_img",             "false",
            # Automate: detect 1 person, auto-select largest, no UI popups
            "--nb_persons_to_detect", "1",
            "--person_ordering_method", "largest_size",
            "--show_realtime_results", "false",
            "--show_graphs",          "false",
        ]

        proc = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=3600
        )

        # Always dump full output to a log file for debugging
        log_file = session_dir / f"sports2d_{cond}_log.txt"
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"=== COMMAND ===\n{' '.join(cmd)}\n\n")
            f.write(f"=== RETURN CODE: {proc.returncode} ===\n\n")
            f.write(f"=== STDOUT ===\n{proc.stdout}\n\n")
            f.write(f"=== STDERR ===\n{proc.stderr}\n")

        # Find the generated _m_person00.trc
        trc_files = sorted(session_dir.rglob("*_m_person*.trc"))

        if proc.returncode != 0:
            if trc_files:
                # Sports2D produced output despite non-zero exit code
                # (common with OpenSim/IK warnings) — proceed with warning
                import logging
                logging.warning(
                    f"Sports2D exited with code {proc.returncode} for {cond}, "
                    f"but TRC file was produced. See log: {log_file}"
                )
            else:
                raise RuntimeError(
                    f"Sports2D failed for {cond} (exit code {proc.returncode}).\n"
                    f"Full log saved to: {log_file}\n\n"
                    f"stderr (last 1500 chars):\n{proc.stderr[-1500:]}"
                )

        if not trc_files:
            raise FileNotFoundError(
                f"No _m_personXX.trc file found in {session_dir} "
                f"after running Sports2D on {video_path}.\n"
                f"Full log saved to: {log_file}"
            )

        # Find annotated video produced by Sports2D.
        # Sports2D saves annotated videos as <name>_Sports2D.mp4 inside the
        # result subdirectory (e.g., single_task_Sports2D/single_task_Sports2D.mp4).
        vid_exts = ("*.mp4", "*.avi", "*.mov", "*.mkv")
        vid_files = []
        for ext in vid_exts:
            vid_files.extend(session_dir.rglob(ext))
        # Exclude the original input video (if it was copied into the dir)
        input_name = Path(video_path).name
        annotated = [v for v in vid_files if v.name != input_name]
        video_path_out = str(annotated[0]) if annotated else ""

        return {"trc": str(trc_files[0]), "video": video_path_out}

    # ------------------------------------------------------------------
    # Public helpers for UI (Addendum 2 checklist)
    # ------------------------------------------------------------------

    def get_stage_statuses(self) -> list[tuple[str, StageStatus]]:
        """Return [(stage_name, status), …] in execution order."""
        order = [
            "sports2d_st", "sports2d_dt",
            "load_st", "load_dt",
            "preprocess_st", "preprocess_dt",
            "detect_events_st", "detect_events_dt",
            "calc_params_st", "calc_params_dt",
            "remove_outliers_st", "remove_outliers_dt",
            "aggregate_st", "aggregate_dt",
            "dtc",
        ]
        return [(n, self._stages[n].status) for n in order]

    def get_annotated_video(self, cond: str) -> str:
        """Return path to the annotated video for 'st' or 'dt', or ''."""
        return self._annotated_videos.get(cond, "")
