<div align="center">

# 🦶 Gait Reliability Analysis

**A clinical-grade Python pipeline for validating wearable gait analysis systems**  
*Computes CV & ICC metrics · Generates publication-ready charts · Auto-builds an interactive dashboard*

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

---

## 📖 Overview

This pipeline validates a **wearable gait analysis system** by computing two standard reliability metrics on stride-level biomechanical data:

| Metric | What it measures |
|---|---|
| **CV** (Coefficient of Variation) | Within-subject consistency — how stable is each person's own gait |
| **ICC (1,1)** | Between-subject reliability — can the system distinguish individuals from each other |

Analysis is performed independently for:
- **Single Task (ST)** vs **Dual Task (DT)** walking conditions
- **Left foot** vs **Right foot**
- All 7 spatio-temporal gait parameters
- **Vertical** and/or **AP** event detection methods

---

## 🗂️ Repository Structure

```
├── gait_reliability.py     # Core pipeline: data loading, CV, ICC, and all charts
├── generate_dashboard.py   # Builds a dynamic HTML dashboard from results
├── run_all.py              # ✅ Single entry point — run this
└── README.md
```

**Generated outputs** (saved inside your dataset folder, suffixed with `_ap` for AP method):

```
<dataset_folder>/
├── reliability_results.csv            # Full CV & ICC table (Vertical)
├── reliability_results_ap.csv         # Full CV & ICC table (AP)
├── within_subject_ST_Left.png         # Per-volunteer CV — ST, Left foot
├── within_subject_ST_Left_ap.png      # Same, AP detector
├── ...                                # (all chart variants for both methods)
├── cv_comparison.png                  # Aggregated ST vs DT CV
├── icc_comparison.png                 # ICC scores with reliability thresholds
├── dashboard.html                     # 🌐 Interactive HTML dashboard (Vertical)
└── dashboard_ap.html                  # 🌐 Interactive HTML dashboard (AP)
```

---

## ⚡ Quick Start

### 1 — Install dependencies

```bash
pip install pandas numpy matplotlib
```

### 2 — Organise your dataset

The tool auto-discovers volunteer folders with either naming convention:

```
your_output_folder/
├── sub_01/                            # Pipeline default format
│   ├── 04_strides_cleaned_st.csv      # Vertical detector
│   ├── 04_strides_cleaned_dt.csv
│   ├── 04_strides_cleaned_ap_st.csv   # AP detector
│   └── 04_strides_cleaned_ap_dt.csv
├── sub_03/
│   └── ...
└── ...
```

Or bare IDs:
```
your_output_folder/
├── 01/
│   ├── 04_strides_cleaned_st.csv
│   └── 04_strides_cleaned_dt.csv
└── ...
```

> Each subfolder name is treated as a **volunteer ID**.  
> Both `sub_XX` and bare `XX` naming are auto-detected.

### 3 — Run the pipeline

```bash
# Vertical detector (default)
python run_all.py path/to/your_output_folder

# AP detector only
python run_all.py path/to/your_output_folder --method ap

# Both methods (generates separate outputs for each)
python run_all.py path/to/your_output_folder --method both
```

You can also run individual scripts directly:

```bash
# Just the analysis (no dashboard)
python gait_reliability.py path/to/your_output_folder --method ap

# Just the dashboard (after analysis has been run)
python generate_dashboard.py path/to/your_output_folder --method ap
```

The dashboard opens automatically in your browser when done. ✅

---

## 📊 Output Charts

### Within-Subject Variability
Each coloured bar = one volunteer's CV for that parameter.  
Low CV → the person's own strides are highly consistent.

### Between-Subject Distribution
Box plots show the spread across all volunteers.  
Coloured dots overlay individual volunteer values.

### Aggregated CV — ST vs DT
Grouped bar chart with error bars comparing both conditions.  
**Expected:** DT bars are noticeably taller (physiological dual-task effect).

### ICC Reliability Scores
Horizontal dashed lines mark standard reliability thresholds:

| ICC | Interpretation |
|---|---|
| < 0.20 | Poor |
| 0.20 – 0.49 | Fair |
| 0.50 – 0.74 | **Moderate** |
| ≥ 0.75 | **Good** |

---

## 📋 Input CSV Format

The `04_strides_cleaned_*.csv` files must contain:

| Column | Type | Description |
|---|---|---|
| `foot` | string | `"left"` or `"right"` |
| `stride_lengths` | float | Stride length (m) |
| `stride_times` | float | Total stride duration (s) |
| `swing_times` | float | Swing phase duration (s) |
| `stance_times` | float | Stance phase duration (s) |
| `stance_ratios` | float | Stance / stride ratio |
| `step_time` | float | Step time (s) |
| `double_support_time` | float | Double support phase (s) |
| `is_outlier` | bool | Outlier flag — filtered out automatically |
| `turning_step` | bool | Turning flag — filtered out automatically |
| `turning_interval` | bool | Turning interval flag |
| `interrupted` | bool | Interrupted stride flag |

### CSV file naming convention

| Detection Method | ST file | DT file |
|---|---|---|
| **Vertical** | `04_strides_cleaned_st.csv` | `04_strides_cleaned_dt.csv` |
| **AP** | `04_strides_cleaned_ap_st.csv` | `04_strides_cleaned_ap_dt.csv` |

---

## ➕ Adding a New Volunteer

1. Create a new subfolder (e.g. `sub_07/`) inside your output directory.
2. Add the cleaned stride CSVs for the detection method(s) you want to analyse.
3. Run `python run_all.py path/to/output --method both` — the new volunteer is detected automatically.

**Optional:** Add a colour for the new volunteer in `gait_reliability.py`:
```python
VOLUNTEER_COLORS = {
    '01': '#4361EE', '03': '#F72585', '04': '#4CC9F0',
    '05': '#F77F00', '06': '#2EC4B6', '07': '#7B2FBE',  # ← add here
}
```

---

## 🧮 Methodology

### Coefficient of Variation (CV)
Computed **per volunteer**, then averaged across subjects:

$$CV_i = \frac{\sigma_i}{\mu_i} \times 100\%$$

$$\overline{CV} = \frac{1}{n}\sum_{i=1}^{n} CV_i$$

### ICC (1,1) — One-Way Random Effects
Implemented using an **unbalanced ANOVA** formula (handles varying stride counts per volunteer):

$$ICC = \frac{\sigma^2_{between}}{\sigma^2_{between} + \sigma^2_{within}}$$

Where:
- $\sigma^2_{between}$ = variance explained by individual differences
- $\sigma^2_{within}$ = variance within each volunteer's own strides

Only **steady-state strides** are used — outliers, turning steps, and interrupted strides are filtered automatically.

---

## 📄 License

MIT License — free to use, modify, and distribute.
