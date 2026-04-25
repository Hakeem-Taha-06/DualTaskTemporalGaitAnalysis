"""
Module 8 — main.py
Responsibility: Orchestrate the full gait analysis pipeline end-to-end.

Usage:
    python main.py --config participant_config.json

Config JSON schema:
{
  "participants": [
    {
      "participant_id": "sub_01",
      "st_trc":  "/path/to/sub01_ST_m_person00.trc",
      "dt_trc":  "/path/to/sub01_DT_m_person00.trc",
      "height_m": 1.72,
      "fps": 120
    }
  ],
  "output_dir": "/path/to/results",
  "apply_filter": false,
  "interruptions_csv": ""
}

Outputs per participant (saved in output_dir/<participant_id>/):
    01_raw_trajectories_st.csv / _dt.csv
    02_events_st.csv / _dt.csv
    03_strides_raw_st.csv / _dt.csv
    04_strides_cleaned_st.csv / _dt.csv
    05_aggregated_st.csv / _dt.csv
    06_dtc.csv
    07_dtc_summary.csv
    summary_table.csv   (all participants combined)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from gait import input_loader
from gait import preprocessor
from gait import event_detector
from gait import parameter_calculator
from gait import outlier_remover
from gait import aggregator
from gait import dtc_calculator


# ---------------------------------------------------------------------------
# Pipeline runner (single participant)
# ---------------------------------------------------------------------------

def run_participant(
    participant_id: str,
    st_trc: str,
    dt_trc: str,
    output_dir: Path,
    height_m: float = 1.70,
    fps: float = 120.0,
    apply_filter: bool = False,
    interruptions_df: Optional[pd.DataFrame] = None,
    progress_callback=None,
) -> dict:
    """
    Run the full pipeline for one participant.

    Parameters
    ----------
    progress_callback : callable(stage_name, pct) or None
        Called at each stage for UI progress updates.

    Returns
    -------
    dict
        Keyed by stage name, values are the DataFrames produced at each stage.
    """
    out_dir = output_dir / participant_id
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {}

    def _emit(stage: str, pct: int, msg: str = ""):
        print(f"  [{pct:3d}%] {stage}" + (f" — {msg}" if msg else ""))
        if progress_callback:
            progress_callback(stage, pct)

    # ------------------------------------------------------------------ #
    # Stage 1-2: Load & preprocess trajectories
    # ------------------------------------------------------------------ #
    for cond, trc_path in [("st", st_trc), ("dt", dt_trc)]:
        _emit(f"Loading trajectories ({cond.upper()})", 5 if cond == "st" else 10)

        traj, detected_fps = input_loader.load_trc(trc_path, fps=fps)
        cond_fps = detected_fps  # use fps from TRC header
        traj.to_csv(out_dir / f"01_raw_trajectories_{cond}.csv", index=False)
        results[f"load_{cond}"] = traj
        results[f"fps_{cond}"] = cond_fps

        _emit(f"Preprocessing ({cond.upper()})", 15 if cond == "st" else 20,
              "no-op (Sports2D pre-filters)" if not apply_filter else "Butterworth 6 Hz")

        traj_pre = preprocessor.preprocess(traj, fps=cond_fps, apply_filter=apply_filter)
        results[f"preprocess_{cond}"] = traj_pre

    # ------------------------------------------------------------------ #
    # Stage 3: Detect gait events
    # ------------------------------------------------------------------ #
    for cond in ("st", "dt"):
        _emit(f"Detecting gait events ({cond.upper()})", 30 if cond == "st" else 35)

        cond_fps  = results[f"fps_{cond}"]
        traj_pre  = results[f"preprocess_{cond}"]
        events_df = event_detector.detect_events(traj_pre, fps=cond_fps)
        events_df.to_csv(out_dir / f"02_events_{cond}.csv", index=False)
        results[f"detect_events_{cond}"] = events_df

    # ------------------------------------------------------------------ #
    # Stage 4: Calculate gait parameters
    # ------------------------------------------------------------------ #
    for cond in ("st", "dt"):
        _emit(f"Computing gait parameters ({cond.upper()})", 45 if cond == "st" else 50)

        traj_pre  = results[f"preprocess_{cond}"]
        events_df = results[f"detect_events_{cond}"]
        gait_ev   = event_detector.events_to_gait_event_dict(events_df)
        strides   = parameter_calculator.calculate_parameters(gait_ev, traj_pre)
        strides.to_csv(out_dir / f"03_strides_raw_{cond}.csv", index=False)
        results[f"calc_params_{cond}"]   = strides
        results[f"gait_events_{cond}"]   = gait_ev

    # ------------------------------------------------------------------ #
    # Stage 5: Remove outliers
    # ------------------------------------------------------------------ #
    for cond in ("st", "dt"):
        _emit(f"Removing outliers ({cond.upper()})", 60 if cond == "st" else 65)

        traj_pre = results[f"preprocess_{cond}"]
        strides  = results[f"calc_params_{cond}"]
        cleaned  = outlier_remover.remove_outliers(
            strides, traj_pre,
            interruptions_df=interruptions_df,
        )
        cleaned.to_csv(out_dir / f"04_strides_cleaned_{cond}.csv", index=False)
        results[f"remove_outliers_{cond}"] = cleaned

    # ------------------------------------------------------------------ #
    # Stage 6: Aggregate
    # ------------------------------------------------------------------ #
    for cond in ("st", "dt"):
        _emit(f"Aggregating parameters ({cond.upper()})", 75 if cond == "st" else 80)

        cleaned = results[f"remove_outliers_{cond}"]
        agg     = aggregator.aggregate(cleaned, participant_id=participant_id,
                                       condition=cond)
        agg.to_csv(out_dir / f"05_aggregated_{cond}.csv", index=False)
        results[f"aggregate_{cond}"] = agg

    # ------------------------------------------------------------------ #
    # Stage 7: Dual-Task Cost
    # ------------------------------------------------------------------ #
    _emit("Computing Dual-Task Cost", 90)

    st_agg = results["aggregate_st"]
    dt_agg = results["aggregate_dt"]
    dtc_df = dtc_calculator.calculate_dtc(st_agg, dt_agg)
    dtc_sum = dtc_calculator.dtc_summary_table(dtc_df)
    dtc_df.to_csv(out_dir / "06_dtc.csv", index=False)
    dtc_sum.to_csv(out_dir / "07_dtc_summary.csv", index=False)
    results["dtc"]         = dtc_df
    results["dtc_summary"] = dtc_sum

    _emit("Done", 100)

    # Print summary table
    print(f"\n{'─'*60}")
    print(f"  DTC Summary — {participant_id}")
    print(f"{'─'*60}")
    print(dtc_sum.to_string(index=False))
    print()

    return results


# ---------------------------------------------------------------------------
# Batch / main entry point
# ---------------------------------------------------------------------------

def run_batch(config: dict, progress_callback=None) -> pd.DataFrame:
    """
    Run the pipeline for all participants defined in config.

    Returns
    -------
    pd.DataFrame
        Combined DTC table across all participants.
    """
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    apply_filter = config.get("apply_filter", False)

    interruptions_df: Optional[pd.DataFrame] = None
    if config.get("interruptions_csv"):
        try:
            interruptions_df = pd.read_csv(config["interruptions_csv"])
        except Exception as e:
            print(f"Warning: Could not load interruptions CSV: {e}")

    all_dtc: list[pd.DataFrame] = []

    for p in config["participants"]:
        pid      = p["participant_id"]
        st_trc   = p["st_trc"]
        dt_trc   = p["dt_trc"]
        height_m = float(p.get("height_m", 1.70))
        fps      = float(p.get("fps", 120.0))

        print(f"\n{'='*60}")
        print(f"  Processing participant: {pid}")
        print(f"{'='*60}")

        try:
            results = run_participant(
                participant_id=pid,
                st_trc=st_trc,
                dt_trc=dt_trc,
                output_dir=output_dir,
                height_m=height_m,
                fps=fps,
                apply_filter=apply_filter,
                interruptions_df=interruptions_df,
                progress_callback=progress_callback,
            )
            all_dtc.append(results["dtc"])
        except Exception as e:
            print(f"ERROR processing {pid}: {e}")
            import traceback
            traceback.print_exc()

    if all_dtc:
        combined_dtc = pd.concat(all_dtc, ignore_index=True)
        combined_dtc.to_csv(output_dir / "summary_table.csv", index=False)
        print(f"\nSummary table saved to {output_dir / 'summary_table.csv'}")
        return combined_dtc

    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(
        description="Gait Analysis Pipeline: Sports2D TRC → DUO-GAIT parameters"
    )
    parser.add_argument(
        "--config",
        help="Path to JSON config file (see main.py docstring for schema)"
    )
    parser.add_argument(
        "--batch-dir",
        help="Path to dataset directory for batch processing. "
             "Each subdirectory should contain single.mp4, dual.mp4, "
             "single.csv, dual.csv, and master.csv"
    )
    parser.add_argument(
        "--output-dir",
        default="./out",
        help="Output directory for batch results (default: ./out)"
    )
    args = parser.parse_args()

    if args.batch_dir:
        # Batch mode
        from runners.batch_runner import run_batch_cli
        batch_dir = Path(args.batch_dir)
        if not batch_dir.exists():
            print(f"Batch directory not found: {batch_dir}")
            sys.exit(1)
        run_batch_cli(batch_dir, Path(args.output_dir))

    elif args.config:
        # Single-participant / config-file mode
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Config file not found: {config_path}")
            sys.exit(1)

        with open(config_path) as f:
            config = json.load(f)

        run_batch(config)

    else:
        parser.error("Either --config or --batch-dir is required.")


if __name__ == "__main__":
    main()
