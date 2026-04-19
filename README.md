# DualTaskTemporalGaitAnalysis

A Python application for computing temporal gait parameters and Dual-Task Cost (DTC) from walking videos or pose-estimation data. The pipeline uses [Sports2D](https://github.com/davidpagnon/Sports2D) for markerless 2D pose estimation and replicates the gait parameter formulas from the [DUO-GAIT](https://github.com/2p-big/DUO-GAIT) pipeline.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage — GUI (Recommended)](#usage--gui-recommended)
- [Usage — Command Line (Batch Mode)](#usage--command-line-batch-mode)
- [Pipeline Stages](#pipeline-stages)
- [Computed Parameters](#computed-parameters)
- [Dual-Task Cost (DTC)](#dual-task-cost-dtc)
- [Output Files](#output-files)
- [Configuration Reference](#configuration-reference)
- [Input Requirements](#input-requirements)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [License](#license)

---

## Overview

This tool is designed for researchers and clinicians studying **dual-task gait analysis** — the comparison of walking performance under normal conditions (single-task) versus walking while performing a cognitive task (dual-task). The pipeline:

1. Accepts walking video files (`.mp4`, `.avi`, `.mov`) or pre-processed pose data (`.trc` files).
2. Detects 2D body keypoints using Sports2D (if video is provided).
3. Identifies gait events (heel-strikes and toe-offs) from heel and toe vertical trajectories.
4. Computes stride-by-stride temporal and spatial gait parameters.
5. Removes outlier strides and turning intervals.
6. Aggregates parameters into summary statistics (mean, CV, symmetry index).
7. Calculates Dual-Task Cost (DTC) as the percentage change between conditions.

---

## Features

- **GUI Application** — Dark-themed desktop interface with real-time progress tracking, interactive charts, and tabulated results.
- **Video Input Support** — Directly processes walking videos via Sports2D; no external pose estimation required.
- **TRC Input Support** — Also accepts pre-processed `.trc` coordinate files if Sports2D has already been run.
- **Automated Processing** — Sports2D auto-selects the largest detected person and runs without manual intervention.
- **DUO-GAIT Compatible** — Gait parameter formulas, outlier thresholds, and aggregation methods match the DUO-GAIT research pipeline.
- **Outlier Detection** — Threshold-based and z-score outlier flagging, turning stride detection, and configurable head/tail trimming.
- **Interactive Results** — Time-series plots of stride parameters (left/right foot, outliers highlighted), DTC bar charts, and a filterable raw data table.
- **Batch Processing** — Command-line mode for processing multiple participants from a JSON config file.
- **CSV Export** — All intermediate and final results saved as CSV files for further analysis.

---

## Requirements

- **Python** 3.10 or higher (tested with 3.12)
- **Operating System**: Windows 10/11 (primary), macOS and Linux should work but are untested.

### Python Dependencies

| Package | Min Version | Purpose |
|---------|-------------|---------|
| `numpy` | ≥ 1.23 | Numerical computation |
| `pandas` | ≥ 1.5 | Data manipulation |
| `scipy` | ≥ 1.9 | Signal processing, statistics |
| `PySide6` | ≥ 6.4 | GUI framework (or `PyQt6` as fallback) |
| `pyqtgraph` | ≥ 0.13 | Interactive charts (falls back to `matplotlib`) |
| `matplotlib` | ≥ 3.6 | Chart rendering (fallback) |
| `sports2d` | ≥ 0.8 | 2D pose estimation (only needed for video input) |

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Hakeem-Taha-06/DualTaskTemporalGaitAnalysis.git
cd DualTaskTemporalGaitAnalysis
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment:

- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
- **Windows (CMD):** `.venv\Scripts\activate.bat`
- **macOS/Linux:** `source .venv/bin/activate`

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Sports2D (for Video Input)

If you plan to use video files directly (rather than pre-processed `.trc` files):

```bash
pip install sports2d
```

> **Note:** Sports2D will download pose estimation models on first run (~200 MB).

---

## Quick Start

### GUI Mode (Single Participant)

```bash
python ui_main.py
```

1. Enter a participant ID.
2. Select your Single-Task (ST) and Dual-Task (DT) files (video or `.trc`).
3. Set participant height and recording FPS.
4. Choose an output directory.
5. Click **Run Analysis**.
6. View results in the tabbed panel on the right.

### CLI Mode (Batch Processing)

```bash
python main.py --config participant_config.json
```

---

## Usage — GUI (Recommended)

Launch the GUI with:

```bash
python ui_main.py
```

### Left Panel — Input & Controls

The left panel contains all input fields and pipeline controls.

#### Participant ID

A text identifier for the participant (e.g., `sub_01`). This is used to name the output subdirectory and label results.

#### Single-Task (ST) and Dual-Task (DT) Input

For each condition, you can provide:

- **`.trc` file** — A pre-processed TRC coordinate file from Sports2D. Select the `.trc` radio button and browse for the file.
- **Video file** — A raw walking video (`.mp4`, `.avi`, `.mov`, `.mkv`). Select the "Video file" radio button and browse for the file. Sports2D will run automatically to extract pose data.

> **Tip:** If you've already run Sports2D separately, use the `.trc` option to skip re-processing.

#### Recording Parameters

- **Height (m)** — Participant's height in metres (default: 1.70). Used by Sports2D to scale pixel coordinates to real-world units.
- **FPS** — Frames per second of the recording (default: 120). This is used as a **fallback** value; the pipeline will auto-detect the actual FPS from the TRC file header when available.

#### Output Directory

Where result CSV files and logs will be saved. Default: `~/gait_results`. A subdirectory named after the participant ID will be created automatically.

#### Run Analysis

Clicking **▶ Run Analysis** starts the pipeline. Progress is shown via:

- A **progress bar** at the top.
- A **stage checklist** below, showing each pipeline stage with status icons:
  - `·` Pending
  - `⟳` Running
  - `✓` Complete
  - `✗` Failed
  - `—` Skipped

### Right Panel — Results Tabs

After the pipeline completes, results are displayed across five tabs:

#### 📽 Annotated Video

Placeholder for annotated video playback. Currently displays a message; future versions may embed the Sports2D annotated output.

#### 🦶 ST Parameters

Time-series charts for each temporal parameter under the **single-task** condition. Each chart shows:

- **Left foot** strides in blue.
- **Right foot** strides in orange.
- **Outlier strides** as grey dots.

Parameters plotted: stride times, stance times, swing times, stride lengths, stance ratios, step time, double support time.

#### 🧮 DT Parameters

Identical layout to ST Parameters, but for the **dual-task** condition.

#### 📊 Dual-Task Cost

The main results tab, showing:

- **Bar chart** — DTC (%) for each parameter. Red bars indicate worsened performance; green bars indicate improved performance under dual-task.
- **Summary table** — Parameter name, DTC percentage, and direction (worsened/improved/unchanged).

#### 📋 Raw Data

A filterable, sortable table showing every stride from both conditions with all computed parameters. Features:

- **Text filter** — Type to filter rows by any column value.
- **Column sorting** — Click any column header to sort.
- **Outlier highlighting** — Outlier strides are highlighted with a red background.

---

## Usage — Command Line (Batch Mode)

For processing multiple participants without the GUI:

```bash
python main.py --config participant_config.json
```

### Config File Format

```json
{
  "output_dir": "./results",
  "apply_filter": false,
  "interruptions_csv": "",
  "participants": [
    {
      "participant_id": "sub_01",
      "st_trc": "/path/to/sub01_ST_m_person00.trc",
      "dt_trc": "/path/to/sub01_DT_m_person00.trc",
      "height_m": 1.72,
      "fps": 120
    },
    {
      "participant_id": "sub_02",
      "st_trc": "/path/to/sub02_ST_m_person00.trc",
      "dt_trc": "/path/to/sub02_DT_m_person00.trc",
      "height_m": 1.68,
      "fps": 120
    }
  ]
}
```

See [Configuration Reference](#configuration-reference) for all options.

A combined `summary_table.csv` is generated in the output directory with DTC results across all participants.

---

## Pipeline Stages

The pipeline executes the following stages sequentially:

| # | Stage | Module | Description |
|---|-------|--------|-------------|
| 1 | `sports2d_st/dt` | `pipeline_runner.py` | Run Sports2D on video to extract 2D poses (skipped if `.trc` provided) |
| 2 | `load_st/dt` | `input_loader.py` | Parse `.trc` file, extract heel/toe keypoint coordinates, detect FPS |
| 3 | `preprocess_st/dt` | `preprocessor.py` | Optional Butterworth 6 Hz low-pass filtering |
| 4 | `detect_events_st/dt` | `event_detector.py` | Detect heel-strike (IC) and toe-off (FO) events via vertical trajectory minima |
| 5 | `calc_params_st/dt` | `parameter_calculator.py` | Compute stride-by-stride temporal/spatial parameters and flag outliers |
| 6 | `remove_outliers_st/dt` | `outlier_remover.py` | Detect turning strides, mark turning intervals, mark interrupted strides |
| 7 | `aggregate_st/dt` | `aggregator.py` | Compute per-condition summary statistics (mean, CV, symmetry index) |
| 8 | `dtc` | `dtc_calculator.py` | Compute Dual-Task Cost between ST and DT aggregated values |

---

## Computed Parameters

### Stride-Level Parameters (per stride, per foot)

| Parameter | Unit | Formula | Source |
|-----------|------|---------|--------|
| `stride_times` | seconds | IC[i+1] − IC[i] | DUO-GAIT line 254 |
| `stance_times` | seconds | FO[i] − IC[i] | DUO-GAIT line 397 |
| `swing_times` | seconds | IC[i+1] − FO[i] | DUO-GAIT line 386 |
| `stride_lengths` | metres | ‖heel_pos[IC_{i+1}] − heel_pos[IC_i]‖ | DUO-GAIT line 225 |
| `stance_ratios` | — | stance_time / stride_time | DUO-GAIT line 94 |
| `step_time` | seconds | IC(this foot) → IC(contralateral foot) | *Not in DUO-GAIT* |
| `double_support_time` | seconds | IC(this foot) → FO(contralateral foot) | *Not in DUO-GAIT* |

### Aggregated Parameters (per condition)

| Statistic | Formula | Source |
|-----------|---------|--------|
| `<param>_avg` | Mean across all valid strides | DUO-GAIT line 42 |
| `<param>_CV` | Coefficient of variation (std/mean) | DUO-GAIT line 43 |
| `<param>_SI` | Symmetry Index: \|X_L − X_R\| / (0.5 × (X_L + X_R)) | DUO-GAIT line 101 |
| `cadence` | 120 / stride_time (steps/min) | DUO-GAIT line 39 |
| `speed` | stride_length / stride_time (m/s) | DUO-GAIT line 40 |

---

## Dual-Task Cost (DTC)

DTC quantifies how walking performance changes when a cognitive task is added:

```
DTC(%) = (X_ST − X_DT) / X_ST × 100
```

- **Positive DTC** → Performance **worsened** under dual-task (e.g., slower speed, shorter strides).
- **Negative DTC** → Performance **improved** under dual-task (uncommon but possible).

DTC is computed for all aggregated parameter columns (avg, CV, and SI).

---

## Output Files

All outputs are saved in `<output_dir>/<participant_id>/`:

| File | Contents |
|------|----------|
| `01_raw_trajectories_st.csv` / `_dt.csv` | Raw keypoint coordinates from TRC parsing |
| `02_events_st.csv` / `_dt.csv` | Detected heel-strike and toe-off events |
| `03_strides_raw_st.csv` / `_dt.csv` | Stride-by-stride parameters with outlier flags |
| `04_strides_cleaned_st.csv` / `_dt.csv` | Strides after turning/interruption marking |
| `05_aggregated_st.csv` / `_dt.csv` | Summary statistics per condition |
| `06_dtc.csv` | DTC values for all parameters |
| `07_dtc_summary.csv` | Human-readable DTC summary table |

When running in batch mode (CLI), an additional `summary_table.csv` is generated in the root output directory combining results across all participants.

When video input is used, a `sports2d_st/` or `sports2d_dt/` subdirectory contains the Sports2D output files and processing logs.

---

## Configuration Reference

### GUI Parameters

| Field | Default | Description |
|-------|---------|-------------|
| Participant ID | `sub_01` | Identifier for naming output files |
| Input type | `.trc file` | Toggle between TRC and video input |
| Height (m) | `1.70` | Participant height for Sports2D scaling |
| FPS | `120` | Fallback frame rate (auto-detected from TRC header) |
| Output directory | `~/gait_results` | Root directory for all outputs |

### JSON Config (CLI Only)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `output_dir` | string | `"./results"` | Root output directory |
| `apply_filter` | bool | `false` | Apply Butterworth 6 Hz filter to trajectories |
| `interruptions_csv` | string | `""` | Path to CSV with `start_s` and `end_s` columns marking interruption periods |
| `participants[].participant_id` | string | — | Participant identifier |
| `participants[].st_trc` | string | — | Path to single-task TRC file |
| `participants[].dt_trc` | string | — | Path to dual-task TRC file |
| `participants[].height_m` | float | `1.70` | Participant height in metres |
| `participants[].fps` | float | `120` | Fallback FPS (auto-detected from TRC) |

---

## Input Requirements

### Video Files

- Format: `.mp4`, `.avi`, `.mov`, `.mkv`
- The participant should be **clearly visible** and walking in a **sagittal plane** (side view).
- Only **one person** should be in frame (the pipeline auto-selects the largest detected person).
- Camera should be **stationary**.

### TRC Files

- Must be generated by Sports2D with `--to_meters true` and `--save_pose true`.
- The file must be the metres variant (filename contains `_m_person`).
- Required keypoints: `LHeel`, `RHeel`, `LBigToe`, `RBigToe`.
- Optional keypoints: `LAnkle`, `RAnkle`.

---

## Troubleshooting

### "Sports2D failed" error

- Ensure Sports2D is installed: `pip install sports2d`
- Check the log file in `<output_dir>/<participant_id>/sports2d_st/` for detailed error messages.
- Sports2D v0.8.29 is the tested version; other versions may have different CLI options.

### All strides flagged as outliers

- The pipeline auto-detects FPS from the TRC header. If FPS detection fails, the fallback value from the UI is used. Ensure the FPS field is set correctly.
- For very short walking videos (< 3 seconds), there may not be enough strides for meaningful analysis. Aim for at least 5-6 seconds of walking.

### DTC values are all NaN

- This means no valid strides survived outlier filtering in one or both conditions. Check that both ST and DT videos/TRC files show clear walking.
- Open `04_strides_cleaned_st.csv` to inspect which strides were removed and why (see `removal_reason` column).

### GUI doesn't launch

- Ensure `PySide6` is installed: `pip install PySide6`
- If `PySide6` fails, `PyQt6` is used as a fallback: `pip install PyQt6`

### Sports2D asks to click on a person

- This means the `--person_ordering_method` flag isn't being passed. Make sure you're running the latest version of the code. The pipeline now uses `--person_ordering_method largest_size` to auto-select.

---

## Project Structure

```
DualTaskTemporalGaitAnalysis/
├── ui_main.py                 # GUI application entry point
├── main.py                    # CLI batch processing entry point
├── pipeline_runner.py         # QThread-based pipeline orchestrator
├── input_loader.py            # TRC file parser
├── preprocessor.py            # Optional signal filtering
├── event_detector.py          # Heel-strike / toe-off detection
├── parameter_calculator.py    # Stride-by-stride parameter computation
├── outlier_remover.py         # Outlier and turning detection
├── aggregator.py              # Summary statistics (mean, CV, SI)
├── dtc_calculator.py          # Dual-Task Cost computation
├── participant_config.json    # Example batch config
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## License

See [LICENSE](LICENSE) for details.