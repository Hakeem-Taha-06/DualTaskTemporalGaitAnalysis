# -*- coding: utf-8 -*-
"""
gait_reliability.py  —  Full pipeline: CV, ICC, and all charts.

Standalone tool that works with the DualTaskTemporalGaitAnalysis pipeline
output format.  Supports both Vertical and AP event detection methods.

Usage
-----
  # From the command line:
  python gait_reliability.py path/to/output_folder                    # vertical (default)
  python gait_reliability.py path/to/output_folder --method ap        # AP detector
  python gait_reliability.py path/to/output_folder --method both      # side-by-side

  # Programmatic:
  from gait_reliability import run
  run(data_dir="path/to/input", output_dir="path/to/output", method="vertical")
"""

import os, sys, re, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ══════════════════════════════════════════════
#  ▶  DEFAULTS (overrideable via CLI or run())
# ══════════════════════════════════════════════
DATA_DIR = "out"

PARAMETERS = [
    'stride_lengths', 'stride_times', 'swing_times',
    'stance_times',   'stance_ratios', 'step_time', 'double_support_time'
]
PARAM_LABELS = {
    'stride_lengths':      'Stride\nLength',
    'stride_times':        'Stride\nTime',
    'swing_times':         'Swing\nTime',
    'stance_times':        'Stance\nTime',
    'stance_ratios':       'Stance\nRatio',
    'step_time':           'Step\nTime',
    'double_support_time': 'Double\nSupport',
}
PARAM_LABELS_CLEAN = {k: v.replace('\n', ' ') for k, v in PARAM_LABELS.items()}

VOLUNTEER_COLORS = {'01':'#4361EE','03':'#F72585','04':'#4CC9F0','05':'#F77F00','06':'#2EC4B6'}
TASK_COLORS      = {'ST':'#2196F3', 'DT':'#F44336'}
FOOT_COLORS      = {'Left':'#6C63FF','Right':'#FF6584'}

FILTER_COLS = ['is_outlier','turning_step','turning_interval','interrupted']

plt.rcParams.update({
    'font.family':'DejaVu Sans','axes.titlesize':14,'axes.labelsize':12,
    'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':10,'figure.dpi':150,
})


# ══════════════════════════════════════════════
#  DISCOVERY — find volunteer folders & CSVs
# ══════════════════════════════════════════════

def _discover_volunteers(data_dir: str) -> list[str]:
    """
    Auto-discover volunteer subdirectories.

    Accepts both naming conventions produced by the pipeline:
      - Bare IDs:    01, 03, 04  (original reliability format)
      - Prefixed:    sub_01, sub_03, sub_04  (pipeline default)

    Returns the list of directory names (as they exist on disk), sorted.
    """
    if not os.path.isdir(data_dir):
        print(f"[ERROR] Data directory not found: {data_dir}")
        return []

    # Pattern: bare numeric ID or sub_XX style
    vol_pattern = re.compile(r'^(sub_)?(\d{2,})$')

    volunteers = []
    for d in sorted(os.listdir(data_dir)):
        full = os.path.join(data_dir, d)
        if os.path.isdir(full) and vol_pattern.match(d):
            volunteers.append(d)

    return volunteers


def _csv_filename(method: str, task: str) -> str:
    """
    Return the expected CSV filename for a given detection method and task.

    method: 'vertical' or 'ap'
    task:   'st' or 'dt'
    """
    if method == "ap":
        return f"04_strides_cleaned_ap_{task}.csv"
    else:
        return f"04_strides_cleaned_{task}.csv"


def _method_suffix(method: str) -> str:
    """Return a file-name-safe suffix to prevent Vertical/AP output collision."""
    if method == "ap":
        return "_ap"
    return ""


# ══════════════════════════════════════════════
#  STEP 1 — LOAD & FILTER DATA
# ══════════════════════════════════════════════
def load_data(data_dir: str, method: str = "vertical"):
    volunteers = _discover_volunteers(data_dir)
    print(f"Found {len(volunteers)} volunteers: {volunteers}")
    print(f"Detection method: {method}")

    agg_data   = {(t,f): [] for t in ['st','dt'] for f in ['left','right']}
    long_rows  = []

    for vol in volunteers:
        # Extract the bare numeric ID for display (strip 'sub_' prefix)
        vol_id = re.sub(r'^sub_', '', vol)

        for task in ['st','dt']:
            fname = _csv_filename(method, task)
            fpath = os.path.join(data_dir, vol, fname)
            if not os.path.exists(fpath):
                continue
            df = pd.read_csv(fpath)
            for col in FILTER_COLS:
                if col in df.columns:
                    df = df[df[col] == False]

            for foot in ['left','right']:
                sub = df[df['foot'] == foot].copy()
                if sub.empty:
                    continue
                sub['volunteer'] = vol_id
                agg_data[(task, foot)].append(sub)

                for param in PARAMETERS:
                    if param not in sub.columns:
                        continue
                    vals = sub[param].dropna()
                    if len(vals) > 1:
                        m = vals.mean()
                        s = vals.std(ddof=1)
                        cv = (s/m*100) if m != 0 else np.nan
                        long_rows.append({'Volunteer':vol_id,'Task':task.upper(),
                                          'Foot':foot.capitalize(),'Parameter':param,
                                          'Mean':m,'SD':s,'CV':cv})
    long_df = pd.DataFrame(long_rows)
    # Return bare IDs for color lookup and display
    bare_ids = sorted(set(re.sub(r'^sub_', '', v) for v in volunteers))
    return bare_ids, agg_data, long_df

# ══════════════════════════════════════════════
#  STEP 2 — CALCULATE CV & ICC
# ══════════════════════════════════════════════
def icc11(df, subject_col, value_col):
    df = df.dropna(subset=[value_col])
    g  = df.groupby(subject_col)[value_col]
    gm, gs, gv = g.mean(), g.count(), g.var(ddof=1)
    k, N = len(gs), gs.sum()
    if k < 2: return np.nan
    grand = df[value_col].mean()
    ssb = np.sum(gs*(gm-grand)**2)
    ssw = np.sum(((gs-1)*gv).fillna(0))
    msb, msw = ssb/(k-1), ssw/(N-k)
    n0  = (N - np.sum(gs**2)/N)/(k-1)
    vb  = max((msb-msw)/n0, 0)
    vw  = msw
    return vb/(vb+vw) if (vb+vw) > 0 else np.nan

def calculate_reliability(volunteers, agg_data, data_dir: str, output_dir: str, method: str = "vertical"):
    subject_cvs = {(t,f):{p:[] for p in PARAMETERS}
                   for t in ['st','dt'] for f in ['left','right']}

    # Re-discover raw folder names for CSV lookup
    raw_dirs = _discover_volunteers(data_dir)
    vol_to_dir = {}
    for d in raw_dirs:
        bare = re.sub(r'^sub_', '', d)
        vol_to_dir[bare] = d

    for vol in volunteers:
        dir_name = vol_to_dir.get(vol, vol)
        for task in ['st','dt']:
            fname = _csv_filename(method, task)
            fpath = os.path.join(data_dir, dir_name, fname)
            if not os.path.exists(fpath): continue
            df = pd.read_csv(fpath)
            for col in FILTER_COLS:
                if col in df.columns: df = df[df[col] == False]
            for foot in ['left','right']:
                sub = df[df['foot'] == foot]
                for param in PARAMETERS:
                    if param not in sub.columns: continue
                    vals = sub[param].dropna()
                    if len(vals) > 1:
                        m = vals.mean()
                        if m != 0:
                            subject_cvs[(task,foot)][param].append(vals.std(ddof=1)/m*100)

    rows = []
    for task in ['st','dt']:
        for foot in ['left','right']:
            if not agg_data[(task,foot)]: continue
            combined = pd.concat(agg_data[(task,foot)], ignore_index=True)
            for param in PARAMETERS:
                cvs = subject_cvs[(task,foot)][param]
                icc = icc11(combined,'volunteer',param) if param in combined.columns else np.nan
                rows.append({'Task':task.upper(),'Foot':foot.capitalize(),'Parameter':param,
                             'Mean_CV (%)': np.mean(cvs) if cvs else np.nan,
                             'SD_CV (%)':   np.std(cvs,ddof=1) if len(cvs)>1 else np.nan,
                             'ICC(1,1)':    icc})

    df_out = pd.DataFrame(rows)
    suffix = _method_suffix(method)
    out_path = os.path.join(output_dir, f"reliability_results{suffix}.csv")
    df_out.to_csv(out_path, index=False)
    print("\n--- Reliability Results ---")
    print(df_out.to_string(index=False))
    print(f"\nSaved: {out_path}")
    return df_out

# ══════════════════════════════════════════════
#  STEP 3 — WITHIN-SUBJECT CHARTS
# ══════════════════════════════════════════════
def plot_within(volunteers, long_df, output_dir: str, method: str = "vertical"):
    x, total_w = np.arange(len(PARAMETERS)), 0.75
    bar_w = total_w / max(len(volunteers), 1)
    suffix = _method_suffix(method)
    method_label = "AP Detector" if method == "ap" else "Vertical Detector"

    for task in ['ST','DT']:
        for foot in ['Left','Right']:
            sub = long_df[(long_df['Task']==task)&(long_df['Foot']==foot)]
            if sub.empty: continue
            fig, ax = plt.subplots(figsize=(14,6))
            for i, vol in enumerate(volunteers):
                vd  = sub[sub['Volunteer']==vol]
                cvs = [vd[vd['Parameter']==p]['CV'].values[0]
                       if len(vd[vd['Parameter']==p]) else np.nan for p in PARAMETERS]
                ax.bar(x - total_w/2 + bar_w/2 + i*bar_w, cvs,
                       width=bar_w*0.88, color=VOLUNTEER_COLORS.get(vol,'gray'),
                       alpha=0.85, edgecolor='white', linewidth=0.5,
                       label=f'Vol {vol}', zorder=3)

            ax.set_xticks(x); ax.set_xticklabels([PARAM_LABELS[p] for p in PARAMETERS])
            ax.set_ylabel('CV (%)'); ax.grid(axis='y',linestyle='--',alpha=0.4,zorder=0)
            ax.set_title(f'Within-Subject CV — {task}, {foot} Foot  [{method_label}]', fontweight='bold', pad=12)
            ax.legend(title='Volunteer'); ax.spines[['top','right']].set_visible(False)
            plt.tight_layout()
            path = os.path.join(output_dir, f'within_subject_{task}_{foot}{suffix}.png')
            fig.savefig(path, dpi=300, bbox_inches='tight'); plt.close(fig)
            print(f"Saved: {path}")

# ══════════════════════════════════════════════
#  STEP 4 — BETWEEN-SUBJECT CHARTS
# ══════════════════════════════════════════════
def plot_between(volunteers, long_df, output_dir: str, method: str = "vertical"):
    x = np.arange(len(PARAMETERS))
    suffix = _method_suffix(method)
    method_label = "AP Detector" if method == "ap" else "Vertical Detector"

    for task in ['ST','DT']:
        for foot in ['Left','Right']:
            sub = long_df[(long_df['Task']==task)&(long_df['Foot']==foot)]
            if sub.empty: continue
            fig, ax = plt.subplots(figsize=(14,6))
            box_data = [sub[sub['Parameter']==p]['CV'].dropna().values for p in PARAMETERS]
            bp = ax.boxplot(box_data, positions=x, widths=0.45, patch_artist=True,
                            medianprops=dict(color='white',linewidth=2.5),
                            whiskerprops=dict(linewidth=1.4,color='#555'),
                            capprops=dict(linewidth=1.4,color='#555'),
                            flierprops=dict(marker='o',markersize=5,alpha=0.5))
            for patch in bp['boxes']:
                patch.set_facecolor(TASK_COLORS[task]); patch.set_alpha(0.6)
            for i, p in enumerate(PARAMETERS):
                for vol in volunteers:
                    val = sub[(sub['Parameter']==p)&(sub['Volunteer']==vol)]['CV']
                    if not val.empty and not np.isnan(val.values[0]):
                        ax.scatter(i + (np.random.rand()-0.5)*0.22, val.values[0],
                                   color=VOLUNTEER_COLORS.get(vol,'gray'), s=60,
                                   zorder=5, edgecolors='white', linewidth=0.7)
            patches = [mpatches.Patch(color=VOLUNTEER_COLORS.get(v,'gray'),label=f'Vol {v}')
                       for v in volunteers]
            ax.legend(handles=patches, title='Volunteer')
            ax.set_xticks(x); ax.set_xticklabels([PARAM_LABELS[p] for p in PARAMETERS])
            ax.set_ylabel('CV (%)'); ax.grid(axis='y',linestyle='--',alpha=0.4,zorder=0)
            ax.set_title(f'Between-Subject CV Distribution — {task}, {foot} Foot  [{method_label}]',
                         fontweight='bold', pad=12)
            ax.spines[['top','right']].set_visible(False)
            plt.tight_layout()
            path = os.path.join(output_dir, f'between_subject_{task}_{foot}{suffix}.png')
            fig.savefig(path, dpi=300, bbox_inches='tight'); plt.close(fig)
            print(f"Saved: {path}")

# ══════════════════════════════════════════════
#  STEP 5 — AGGREGATED CV & ICC CHARTS
# ══════════════════════════════════════════════
def plot_aggregated(rel_df, output_dir: str, method: str = "vertical"):
    rel_df = rel_df.copy()
    rel_df['Label'] = rel_df['Parameter'].map(PARAM_LABELS_CLEAN)
    param_order = [PARAM_LABELS_CLEAN[p] for p in PARAMETERS]
    x, w = np.arange(len(param_order)), 0.35
    suffix = _method_suffix(method)
    method_label = "AP Detector" if method == "ap" else "Vertical Detector"

    # ── CV chart ──
    fig, axes = plt.subplots(1,2,figsize=(18,7),sharey=True)
    fig.suptitle(f'Coefficient of Variation (CV): Single Task vs Dual Task  [{method_label}]',
                 fontsize=17, fontweight='bold', y=1.01)
    for ax, foot in zip(axes, ['Left','Right']):
        sub = rel_df[rel_df['Foot']==foot]
        for i, task in enumerate(['ST','DT']):
            m = sub[sub['Task']==task]
            m = m.set_index('Label').reindex(param_order)
            means, sds = m['Mean_CV (%)'].values, m['SD_CV (%)'].values
            offs = x + (i-0.5)*w
            bars = ax.bar(offs, means, width=w, color=TASK_COLORS[task],
                          alpha=0.85, edgecolor='white', linewidth=0.6,
                          label=task, zorder=3)
            ax.errorbar(offs, means, yerr=sds, fmt='none', color='#333',
                        capsize=4, linewidth=1.4, zorder=4)
            for bar, val, sd in zip(bars, means, sds):
                if not np.isnan(val):
                    ax.text(bar.get_x()+bar.get_width()/2, val+sd+0.8,
                            f'{val:.1f}%', ha='center', va='bottom', fontsize=8.5)
        ax.set_title(f'{foot} Foot', fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(param_order, rotation=40, ha='right')
        ax.set_xlabel('Parameter'); ax.grid(axis='y',linestyle='--',alpha=0.4,zorder=0)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Mean CV (%)')
    patches = [mpatches.Patch(color=TASK_COLORS[t], label=t) for t in ['ST','DT']]
    fig.legend(handles=patches, loc='upper center', ncol=2, bbox_to_anchor=(0.5,-0.02))
    plt.tight_layout()
    path = os.path.join(output_dir, f'cv_comparison{suffix}.png')
    fig.savefig(path, dpi=300, bbox_inches='tight'); plt.close(fig); print(f"Saved: {path}")

    # ── ICC chart ──
    fig, axes = plt.subplots(1,2,figsize=(18,7),sharey=True)
    fig.suptitle(f'Intraclass Correlation Coefficient (ICC 1,1) — System Reliability  [{method_label}]',
                 fontsize=17, fontweight='bold', y=1.01)
    for ax, task in zip(axes, ['ST','DT']):
        sub = rel_df[rel_df['Task']==task]
        for i, foot in enumerate(['Left','Right']):
            m = sub[sub['Foot']==foot].set_index('Label').reindex(param_order)
            vals = m['ICC(1,1)'].values
            offs = x + (i-0.5)*w
            bars = ax.bar(offs, vals, width=w, color=FOOT_COLORS[foot],
                          alpha=0.85, edgecolor='white', linewidth=0.6,
                          label=foot, zorder=3)
            for bar, val in zip(bars, vals):
                if not np.isnan(val) and val > 0.01:
                    ax.text(bar.get_x()+bar.get_width()/2, val+0.01,
                            f'{val:.2f}', ha='center', va='bottom', fontsize=8.5)
        ax.axhline(0.50, color='#FF9800', linestyle='--', linewidth=1.5, zorder=5)
        ax.axhline(0.75, color='#4CAF50', linestyle='--', linewidth=1.5, zorder=5)
        ax.text(len(param_order)-0.1, 0.51,'Moderate (0.50)',
                va='bottom', ha='right', fontsize=9, color='#E65100')
        ax.text(len(param_order)-0.1, 0.76,'Good (0.75)',
                va='bottom', ha='right', fontsize=9, color='#2E7D32')
        ax.set_title(f'{"Single Task" if task=="ST" else "Dual Task"} ({task})',
                     fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(param_order, rotation=40, ha='right')
        ax.set_xlabel('Parameter'); ax.set_ylim(0, 1.05)
        ax.grid(axis='y',linestyle='--',alpha=0.4,zorder=0)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('ICC (1,1)')
    patches = [mpatches.Patch(color=FOOT_COLORS[f], label=f'{f} Foot') for f in ['Left','Right']]
    fig.legend(handles=patches, loc='upper center', ncol=2, bbox_to_anchor=(0.5,-0.02))
    plt.tight_layout()
    path = os.path.join(output_dir, f'icc_comparison{suffix}.png')
    fig.savefig(path, dpi=300, bbox_inches='tight'); plt.close(fig); print(f"Saved: {path}")

# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════
def run(data_dir: str, output_dir: str, method: str = "vertical"):
    """
    Run the full reliability pipeline for a single detection method.

    Parameters
    ----------
    data_dir : str
        Root folder containing volunteer subdirectories to read from
        (e.g. out/sub_01/, out/sub_03/).
    output_dir : str
        Folder where all results (CSVs, charts, dashboards) are saved.
    method : str
        'vertical' or 'ap' — selects which cleaned stride CSVs to read.
    """
    method = method.lower().strip()
    if method not in ("vertical", "ap"):
        raise ValueError(f"Unknown method '{method}'. Use 'vertical' or 'ap'.")

    os.makedirs(output_dir, exist_ok=True)
    method_label = "AP Detector" if method == "ap" else "Vertical Detector"

    print(f"\n{'='*55}")
    print(f"  STEP 1/4  Loading & filtering data  [{method_label}] ...")
    print(f"  Input dir:  {os.path.abspath(data_dir)}")
    print(f"  Output dir: {os.path.abspath(output_dir)}")
    print("="*55)
    volunteers, agg_data, long_df = load_data(data_dir, method)

    if not volunteers:
        print("[ERROR] No volunteer folders found.  Check your data_dir path.")
        return None, []

    if long_df.empty:
        print(f"[SKIP] No stride data found for method '{method}' — "
              f"the cleaned CSVs for this method may not exist in the input folder.")
        return None, volunteers

    print(f"\n{'='*55}")
    print(f"  STEP 2/4  Calculating CV & ICC  [{method_label}] ...")
    print("="*55)
    rel_df = calculate_reliability(volunteers, agg_data, data_dir, output_dir, method)

    print(f"\n{'='*55}")
    print(f"  STEP 3/4  Generating within-subject charts  [{method_label}] ...")
    print("="*55)
    plot_within(volunteers, long_df, output_dir, method)

    print(f"\n{'='*55}")
    print(f"  STEP 3b   Generating between-subject charts  [{method_label}] ...")
    print("="*55)
    plot_between(volunteers, long_df, output_dir, method)

    print(f"\n{'='*55}")
    print(f"  STEP 4/4  Generating aggregated CV & ICC charts  [{method_label}] ...")
    print("="*55)
    plot_aggregated(rel_df, output_dir, method)

    # Persist volunteer list so the dashboard can read it
    import json
    vol_path = os.path.join(output_dir, "_volunteers.json")
    json_data = json.loads(open(vol_path, encoding="utf-8").read()) if os.path.exists(vol_path) else {}
    json_data["volunteers"] = volunteers
    with open(vol_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f)

    print(f"\n  [{method_label}] All done! Files saved to:", output_dir)
    return rel_df, volunteers


def main():
    parser = argparse.ArgumentParser(
        description="Gait Reliability Analysis — CV & ICC pipeline",
    )
    parser.add_argument(
        "data_dir", nargs="?", default=DATA_DIR,
        help="Path to the output folder containing volunteer subdirectories "
             "(e.g. sub_01/, sub_03/). Defaults to '%(default)s'."
    )
    parser.add_argument(
        "-o", "--output-dir", default=None,
        help="Where to save results. Defaults to <data_dir> itself."
    )
    parser.add_argument(
        "--method", choices=["vertical", "ap", "both"], default="both",
        help="Event detection method to analyse. Default: 'both'."
    )
    args = parser.parse_args()
    out = args.output_dir or args.data_dir

    if args.method == "both":
        print("\n" + "#"*55)
        print("  Running reliability for BOTH detection methods")
        print("#"*55)
        for m in ("vertical", "ap"):
            run(data_dir=args.data_dir, output_dir=out, method=m)
    else:
        run(data_dir=args.data_dir, output_dir=out, method=args.method)


if __name__ == "__main__":
    main()
