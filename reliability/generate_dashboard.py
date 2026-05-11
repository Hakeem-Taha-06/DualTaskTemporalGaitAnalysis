# -*- coding: utf-8 -*-
"""
generate_dashboard.py
Reads reliability_results.csv and reliability_results_ap.csv from the output
directory and generates a single combined dashboard.html showing both methods.
"""

import os, sys, re, json, argparse
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gait_reliability import _method_suffix

PARAM_LABELS = {
    'stride_lengths':      'Stride Length',
    'stride_times':        'Stride Time',
    'swing_times':         'Swing Time',
    'stance_times':        'Stance Time',
    'stance_ratios':       'Stance Ratio',
    'step_time':           'Step Time',
    'double_support_time': 'Double Support',
}


def _load_volunteers(output_dir: str) -> list[str]:
    """Read the volunteer list saved by gait_reliability.run()."""
    vpath = os.path.join(output_dir, "_volunteers.json")
    if os.path.exists(vpath):
        try:
            data = json.loads(open(vpath, encoding="utf-8").read())
            return data.get("volunteers", [])
        except Exception:
            pass
    return []


def _icc_badge(val):
    if pd.isna(val) or val < 0.2:
        return '<span class="badge-icc icc-poor">Poor (&lt;0.2)</span>'
    elif val < 0.5:
        return f'<span class="badge-icc icc-fair">Fair ({val:.2f})</span>'
    elif val < 0.75:
        return f'<span class="badge-icc icc-mod">Moderate ({val:.2f})</span>'
    else:
        return f'<span class="badge-icc icc-good">Good ({val:.2f})</span>'


def _build_table_rows(df):
    rows = []
    for p in PARAM_LABELS:
        label = PARAM_LABELS[p]
        st_r = df[(df['Task']=='ST') & (df['Foot']=='Right') & (df['Parameter']==p)]
        dt_r = df[(df['Task']=='DT') & (df['Foot']=='Right') & (df['Parameter']==p)]
        cv_st  = f"{st_r['Mean_CV (%)'].values[0]:.2f}%" if len(st_r) else 'N/A'
        cv_dt  = f"{dt_r['Mean_CV (%)'].values[0]:.2f}%" if len(dt_r) else 'N/A'
        icc_st = st_r['ICC(1,1)'].values[0] if len(st_r) else np.nan
        icc_dt = dt_r['ICC(1,1)'].values[0] if len(dt_r) else np.nan
        rows.append(f"""
        <tr>
          <td><strong>{label}</strong></td>
          <td>{cv_st}</td><td>{cv_dt}</td>
          <td>{_icc_badge(icc_st)}</td><td>{_icc_badge(icc_dt)}</td>
        </tr>""")
    return "\n".join(rows)


def _build_stat_cards(df, volunteers):
    st_df = df[df['Task'] == 'ST']
    if st_df['Mean_CV (%)'].dropna().empty:
        best_cv_val, best_cv_label = 'N/A', '—'
    else:
        r = st_df.loc[st_df['Mean_CV (%)'].idxmin()]
        best_cv_val = f"{r['Mean_CV (%)']:.1f}%"
        best_cv_label = f"{PARAM_LABELS.get(r['Parameter'], r['Parameter'])} – {r['Foot']}"

    if df['ICC(1,1)'].dropna().empty:
        best_icc_val, best_icc_label = 'N/A', '—'
    else:
        r = df.loc[df['ICC(1,1)'].idxmax()]
        best_icc_val = f"{r['ICC(1,1)']:.2f}"
        best_icc_label = f"{PARAM_LABELS.get(r['Parameter'], r['Parameter'])} – {r['Task']} {r['Foot']}"

    sl_st = df[(df['Task']=='ST') & (df['Parameter']=='stride_lengths')]['Mean_CV (%)'].mean()
    sl_dt = df[(df['Task']=='DT') & (df['Parameter']=='stride_lengths')]['Mean_CV (%)'].mean()
    ratio = f"{sl_dt/sl_st:.1f}x" if (not np.isnan(sl_st) and sl_st > 0) else 'N/A'

    return f"""
  <div class="stat-card blue">
    <div class="stat-label">Volunteers</div>
    <div class="stat-value">{len(volunteers)}</div>
    <div class="stat-sub">{" · ".join(volunteers)}</div>
  </div>
  <div class="stat-card green">
    <div class="stat-label">Best ST CV</div>
    <div class="stat-value">{best_cv_val}</div>
    <div class="stat-sub">{best_cv_label}</div>
  </div>
  <div class="stat-card pink">
    <div class="stat-label">Best ICC Overall</div>
    <div class="stat-value">{best_icc_val}</div>
    <div class="stat-sub">{best_icc_label}</div>
  </div>
  <div class="stat-card org">
    <div class="stat-label">DT vs ST CV (Stride Len)</div>
    <div class="stat-value">{ratio}</div>
    <div class="stat-sub">Dual-task physiological effect</div>
  </div>"""


def _method_section(method: str, df, volunteers, suffix: str) -> str:
    """Build the full HTML section for one detection method."""
    method_label = "AP Detector" if method == "ap" else "Vertical Detector"
    method_id = method  # 'vertical' or 'ap'
    stat_cards = _build_stat_cards(df, volunteers)
    table_rows = _build_table_rows(df)

    return f"""
<!-- ══════════════ {method_label} ══════════════ -->
<div class="method-block" id="method-{method_id}">

<div class="method-banner {'banner-ap' if method=='ap' else 'banner-vert'}">
  <h2>{'📐' if method=='ap' else '📏'} {method_label}</h2>
  <p>{'Anterior-Posterior coordinate peak detection' if method=='ap' else 'Vertical (Y-axis) coordinate peak detection'}</p>
</div>

<div class="stats-row">{stat_cards}</div>

<!-- Within Subject -->
<div class="section">
  <div class="section-header">
    <div class="section-dot"></div>
    <h2>Within-Subject Variability</h2>
    <span>CV per volunteer, per parameter</span>
  </div>
  <div class="tabs" id="tabs-within-{method_id}">
    <button class="tab-btn active" onclick="switchTab('within-{method_id}','ST-Left',this)">ST · Left</button>
    <button class="tab-btn"        onclick="switchTab('within-{method_id}','ST-Right',this)">ST · Right</button>
    <button class="tab-btn"        onclick="switchTab('within-{method_id}','DT-Left',this)">DT · Left</button>
    <button class="tab-btn"        onclick="switchTab('within-{method_id}','DT-Right',this)">DT · Right</button>
  </div>
  <div id="within-{method_id}-ST-Left" class="tab-content active">
    <div class="chart-panel"><div class="chart-panel-header"><span class="chart-badge badge-st">ST</span><span class="chart-badge badge-left">Left</span><h3>Within-Subject CV</h3></div><img src="within_subject_ST_Left{suffix}.png" alt="Within ST Left"></div>
  </div>
  <div id="within-{method_id}-ST-Right" class="tab-content">
    <div class="chart-panel"><div class="chart-panel-header"><span class="chart-badge badge-st">ST</span><span class="chart-badge badge-right">Right</span><h3>Within-Subject CV</h3></div><img src="within_subject_ST_Right{suffix}.png" alt="Within ST Right"></div>
  </div>
  <div id="within-{method_id}-DT-Left" class="tab-content">
    <div class="chart-panel"><div class="chart-panel-header"><span class="chart-badge badge-dt">DT</span><span class="chart-badge badge-left">Left</span><h3>Within-Subject CV</h3></div><img src="within_subject_DT_Left{suffix}.png" alt="Within DT Left"></div>
  </div>
  <div id="within-{method_id}-DT-Right" class="tab-content">
    <div class="chart-panel"><div class="chart-panel-header"><span class="chart-badge badge-dt">DT</span><span class="chart-badge badge-right">Right</span><h3>Within-Subject CV</h3></div><img src="within_subject_DT_Right{suffix}.png" alt="Within DT Right"></div>
  </div>
</div>

<!-- Between Subject -->
<div class="section">
  <div class="section-header">
    <div class="section-dot"></div>
    <h2>Between-Subject Variability</h2>
    <span>Box plot + individual data points</span>
  </div>
  <div class="tabs" id="tabs-between-{method_id}">
    <button class="tab-btn active" onclick="switchTab('between-{method_id}','ST-Left',this)">ST · Left</button>
    <button class="tab-btn"        onclick="switchTab('between-{method_id}','ST-Right',this)">ST · Right</button>
    <button class="tab-btn"        onclick="switchTab('between-{method_id}','DT-Left',this)">DT · Left</button>
    <button class="tab-btn"        onclick="switchTab('between-{method_id}','DT-Right',this)">DT · Right</button>
  </div>
  <div id="between-{method_id}-ST-Left" class="tab-content active">
    <div class="chart-panel"><div class="chart-panel-header"><span class="chart-badge badge-st">ST</span><span class="chart-badge badge-left">Left</span><h3>Between-Subject Distribution</h3></div><img src="between_subject_ST_Left{suffix}.png" alt="Between ST Left"></div>
  </div>
  <div id="between-{method_id}-ST-Right" class="tab-content">
    <div class="chart-panel"><div class="chart-panel-header"><span class="chart-badge badge-st">ST</span><span class="chart-badge badge-right">Right</span><h3>Between-Subject Distribution</h3></div><img src="between_subject_ST_Right{suffix}.png" alt="Between ST Right"></div>
  </div>
  <div id="between-{method_id}-DT-Left" class="tab-content">
    <div class="chart-panel"><div class="chart-panel-header"><span class="chart-badge badge-dt">DT</span><span class="chart-badge badge-left">Left</span><h3>Between-Subject Distribution</h3></div><img src="between_subject_DT_Left{suffix}.png" alt="Between DT Left"></div>
  </div>
  <div id="between-{method_id}-DT-Right" class="tab-content">
    <div class="chart-panel"><div class="chart-panel-header"><span class="chart-badge badge-dt">DT</span><span class="chart-badge badge-right">Right</span><h3>Between-Subject Distribution</h3></div><img src="between_subject_DT_Right{suffix}.png" alt="Between DT Right"></div>
  </div>
</div>

<!-- Aggregated -->
<div class="section">
  <div class="section-header">
    <div class="section-dot"></div>
    <h2>Aggregated CV &amp; ICC</h2>
  </div>
  <div class="chart-grid">
    <div class="chart-panel">
      <div class="chart-panel-header"><h3>Mean CV (%) — ST vs DT</h3></div>
      <img src="cv_comparison{suffix}.png" alt="CV Comparison">
    </div>
    <div class="chart-panel">
      <div class="chart-panel-header"><h3>ICC (1,1) — Reliability</h3></div>
      <img src="icc_comparison{suffix}.png" alt="ICC Comparison">
    </div>
  </div>
</div>

<!-- Table -->
<div class="section">
  <div class="section-header">
    <div class="section-dot"></div>
    <h2>Summary Table — Right Foot (ST vs DT)</h2>
  </div>
  <div class="table-wrap">
    <table>
      <thead><tr>
        <th>Parameter</th><th>Mean CV % (ST)</th><th>Mean CV % (DT)</th>
        <th>ICC (1,1) — ST</th><th>ICC (1,1) — DT</th>
      </tr></thead>
      <tbody>{table_rows}</tbody>
    </table>
  </div>
</div>

</div><!-- end method-block -->
"""


def generate(output_dir: str, data_dir_label: str = ""):
    """Generate a single combined dashboard with both Vertical and AP results."""
    source_label = data_dir_label or os.path.basename(output_dir)
    volunteers = _load_volunteers(output_dir) or ['—']

    # Load whichever result CSVs exist
    methods_html = []
    methods_found = []
    for method in ("vertical", "ap"):
        suffix = _method_suffix(method)
        csv_path = os.path.join(output_dir, f"reliability_results{suffix}.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        df['Task'] = df['Task'].str.strip().str.upper()
        df['Foot'] = df['Foot'].str.strip().str.capitalize()
        methods_html.append(_method_section(method, df, volunteers, suffix))
        methods_found.append(method)

    if not methods_found:
        print(f"[ERROR] No reliability_results*.csv found in: {output_dir}")
        return None

    methods_body = "\n<hr class='method-divider'>\n".join(methods_html)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Gait Reliability Dashboard — {source_label}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
  :root {{
    --bg:#0f1117; --surface:#1a1d2e; --surface2:#242740;
    --accent:#4361ee; --accent2:#f72585; --green:#2ec4b6;
    --orange:#f77f00; --text:#e2e8f0; --muted:#8892a4;
    --border:#2d3250; --radius:16px;
  }}
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;}}

  header{{background:linear-gradient(135deg,#0d1b4b,#1a0533);border-bottom:1px solid var(--border);padding:40px 60px 36px;display:flex;align-items:center;gap:28px;}}
  .header-icon{{width:64px;height:64px;border-radius:18px;background:linear-gradient(135deg,var(--accent),var(--accent2));display:grid;place-items:center;font-size:28px;flex-shrink:0;}}
  header h1{{font-size:28px;font-weight:800;letter-spacing:-.5px;}}
  header p{{font-size:14px;color:var(--muted);margin-top:6px;}}
  .data-source-badge{{display:inline-flex;align-items:center;gap:8px;background:rgba(46,196,182,.1);border:1px solid rgba(46,196,182,.3);border-radius:8px;padding:6px 14px;font-size:12px;color:var(--green);margin-top:10px;}}

  .method-banner{{padding:28px 60px;border-bottom:1px solid var(--border);margin-top:20px;}}
  .method-banner h2{{font-size:22px;font-weight:800;}}
  .method-banner p{{font-size:13px;color:var(--muted);margin-top:4px;}}
  .banner-vert{{background:linear-gradient(135deg,rgba(67,97,238,.08),rgba(76,201,240,.06));border-left:4px solid var(--accent);}}
  .banner-ap{{background:linear-gradient(135deg,rgba(247,37,133,.08),rgba(247,127,0,.06));border-left:4px solid var(--accent2);}}

  .method-divider{{border:none;height:2px;background:linear-gradient(90deg,transparent,var(--border),transparent);margin:48px 60px 0;}}

  .stats-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:20px;padding:28px 60px 0;}}
  .stat-card{{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:24px 28px;position:relative;overflow:hidden;}}
  .stat-card::before{{content:'';position:absolute;top:0;left:0;width:100%;height:3px;}}
  .stat-card.blue::before{{background:linear-gradient(90deg,var(--accent),#4cc9f0);}}
  .stat-card.pink::before{{background:linear-gradient(90deg,var(--accent2),#f77f00);}}
  .stat-card.green::before{{background:linear-gradient(90deg,var(--green),#4361ee);}}
  .stat-card.org::before{{background:linear-gradient(90deg,var(--orange),var(--accent2));}}
  .stat-label{{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:1px;}}
  .stat-value{{font-size:36px;font-weight:800;margin:8px 0 4px;}}
  .stat-sub{{font-size:12px;color:var(--muted);}}
  .stat-card.blue .stat-value{{color:#4cc9f0;}}
  .stat-card.pink .stat-value{{color:var(--accent2);}}
  .stat-card.green .stat-value{{color:var(--green);}}
  .stat-card.org .stat-value{{color:var(--orange);}}

  .section{{padding:36px 60px 0;}}
  .section-header{{display:flex;align-items:center;gap:14px;margin-bottom:20px;}}
  .section-dot{{width:10px;height:10px;border-radius:50%;background:linear-gradient(135deg,var(--accent),var(--accent2));flex-shrink:0;}}
  .section-header h2{{font-size:18px;font-weight:700;}}
  .section-header span{{font-size:13px;color:var(--muted);margin-left:auto;}}

  .tabs{{display:flex;gap:8px;margin-bottom:20px;flex-wrap:wrap;}}
  .tab-btn{{padding:8px 18px;border-radius:10px;border:1px solid var(--border);background:var(--surface);color:var(--muted);font-family:inherit;font-size:13px;font-weight:600;cursor:pointer;transition:all .2s;}}
  .tab-btn.active{{background:linear-gradient(135deg,var(--accent),#7b2ff7);border-color:transparent;color:#fff;}}
  .tab-btn:hover:not(.active){{border-color:var(--accent);color:var(--text);}}

  .chart-panel{{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden;transition:transform .2s,box-shadow .2s;}}
  .chart-panel:hover{{transform:translateY(-3px);box-shadow:0 12px 40px rgba(67,97,238,.15);}}
  .chart-panel-header{{padding:14px 22px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:12px;}}
  .chart-badge{{padding:4px 12px;border-radius:20px;font-size:11px;font-weight:700;letter-spacing:.5px;text-transform:uppercase;}}
  .badge-st{{background:rgba(33,150,243,.15);color:#4cc9f0;}}
  .badge-dt{{background:rgba(244,67,54,.15);color:#f48fb1;}}
  .badge-left{{background:rgba(108,99,255,.15);color:#a5b4fc;}}
  .badge-right{{background:rgba(247,37,133,.15);color:#f9a8d4;}}
  .chart-panel-header h3{{font-size:14px;font-weight:600;}}
  .chart-panel img{{width:100%;display:block;}}
  .chart-grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px;}}

  .table-wrap{{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden;}}
  table{{width:100%;border-collapse:collapse;}}
  thead{{background:var(--surface2);}}
  thead th{{padding:14px 18px;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:1px;color:var(--muted);font-weight:600;}}
  tbody tr{{border-bottom:1px solid var(--border);transition:background .15s;}}
  tbody tr:hover{{background:var(--surface2);}}
  tbody tr:last-child{{border-bottom:none;}}
  tbody td{{padding:14px 18px;font-size:13px;}}
  .badge-icc{{display:inline-block;padding:3px 10px;border-radius:6px;font-weight:700;font-size:12px;}}
  .icc-poor{{background:rgba(244,67,54,.15);color:#ef9a9a;}}
  .icc-fair{{background:rgba(255,152,0,.12);color:#ffcc80;}}
  .icc-mod{{background:rgba(255,193,7,.15);color:#ffe082;}}
  .icc-good{{background:rgba(76,175,80,.15);color:#a5d6a7;}}

  footer{{margin-top:60px;padding:28px 60px;border-top:1px solid var(--border);display:flex;justify-content:space-between;align-items:center;}}
  footer p{{font-size:12px;color:var(--muted);}}

  .tab-content{{display:none;}}
  .tab-content.active{{display:block;}}
</style>
</head>
<body>

<header>
  <div class="header-icon">&#x1F9B6;</div>
  <div>
    <h1>Gait Analysis — Reliability Dashboard</h1>
    <p>CV &amp; ICC (1,1) &nbsp;|&nbsp; ST vs DT &nbsp;|&nbsp; Left &amp; Right Foot &nbsp;|&nbsp; Vertical &amp; AP Detectors</p>
    <div class="data-source-badge">&#x1F4C1; {source_label} &nbsp;·&nbsp; {len(volunteers)} volunteers &nbsp;·&nbsp; {', '.join(m.capitalize() for m in methods_found)} methods</div>
  </div>
</header>

{methods_body}

<footer>
  <p>Biomechanics Gait Reliability Analysis &nbsp;|&nbsp; ICC (1,1) One-Way Random Effects &nbsp;|&nbsp; Auto-generated</p>
  <p>Volunteers: {' · '.join(volunteers)}</p>
</footer>

<script>
function switchTab(group, id, btn) {{
  document.querySelectorAll('[id^="' + group + '-"]').forEach(el => el.classList.remove('active'));
  btn.closest('.tabs').querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(group + '-' + id).classList.add('active');
  btn.classList.add('active');
}}
</script>
</body>
</html>"""

    out_path = os.path.join(output_dir, "dashboard.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Dashboard generated: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate reliability dashboard HTML")
    parser.add_argument("output_dir", help="Folder containing reliability results and charts.")
    parser.add_argument("--label", default="", help="Source label shown in header.")
    args = parser.parse_args()
    generate(args.output_dir, data_dir_label=args.label)


if __name__ == "__main__":
    main()
