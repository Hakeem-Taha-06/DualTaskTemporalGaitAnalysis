"""
ui_main.py — PyQt6 / PySide6 dark-themed gait analysis desktop UI.

Layout:
    Left panel (280 px fixed) — participant input + controls
    Right area — tabbed results (Annotated Video, ST Params, DT Params,
                 Dual-Task Cost, Raw Data Table)
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from PySide6.QtCore    import Qt, QThread, Signal, Slot, QTimer
    from PySide6.QtGui     import QColor, QFont, QPalette, QAction
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
        QLabel, QLineEdit, QPushButton, QFileDialog, QProgressBar,
        QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
        QSplitter, QScrollArea, QComboBox, QDoubleSpinBox, QSpinBox,
        QGroupBox, QFrame, QRadioButton, QButtonGroup, QSizePolicy,
        QMessageBox, QCheckBox, QTextEdit, QAbstractItemView,
    )
    BACKEND = "PySide6"
except ImportError:
    from PyQt6.QtCore    import Qt, QThread, pyqtSignal as Signal, pyqtSlot as Slot, QTimer
    from PyQt6.QtGui     import QColor, QFont, QPalette, QAction
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
        QLabel, QLineEdit, QPushButton, QFileDialog, QProgressBar,
        QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
        QSplitter, QScrollArea, QComboBox, QDoubleSpinBox, QSpinBox,
        QGroupBox, QFrame, QRadioButton, QButtonGroup, QSizePolicy,
        QMessageBox, QCheckBox, QTextEdit, QAbstractItemView,
    )
    BACKEND = "PyQt6"

# Attempt pyqtgraph; fall back to matplotlib if unavailable
try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False
    import matplotlib
    matplotlib.use("QtAgg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_runner import PipelineRunner, StageStatus, STATUS_ICON


# ---------------------------------------------------------------------------
# Color palette (Addendum 1 design specs)
# ---------------------------------------------------------------------------
C_BG        = "#1e1e1e"
C_SURFACE   = "#2a2a2a"
C_ACCENT    = "#4a90d9"
C_TEXT      = "#e0e0e0"
C_MUTED     = "#888888"
C_POSITIVE  = "#e05c5c"   # DTC positive (worsened) — red
C_NEGATIVE  = "#5cb85c"   # DTC negative (improved) — green
C_OUTLIER   = "#5c2a2a"   # Muted red for outlier rows in table
C_LEFT_FOOT = "#4a90d9"   # Blue
C_RIGHT_FOOT= "#f0a030"   # Orange


def _apply_dark_palette(app: QApplication):
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window,          QColor(C_BG))
    pal.setColor(QPalette.ColorRole.WindowText,      QColor(C_TEXT))
    pal.setColor(QPalette.ColorRole.Base,            QColor(C_SURFACE))
    pal.setColor(QPalette.ColorRole.AlternateBase,   QColor("#252525"))
    pal.setColor(QPalette.ColorRole.ToolTipBase,     QColor(C_BG))
    pal.setColor(QPalette.ColorRole.ToolTipText,     QColor(C_TEXT))
    pal.setColor(QPalette.ColorRole.Text,            QColor(C_TEXT))
    pal.setColor(QPalette.ColorRole.Button,          QColor(C_SURFACE))
    pal.setColor(QPalette.ColorRole.ButtonText,      QColor(C_TEXT))
    pal.setColor(QPalette.ColorRole.Highlight,       QColor(C_ACCENT))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(C_TEXT))
    pal.setColor(QPalette.ColorRole.Link,            QColor(C_ACCENT))
    app.setPalette(pal)
    app.setStyleSheet(f"""
        QGroupBox {{
            border: 1px solid #444; border-radius: 4px;
            margin-top: 1em; color: {C_MUTED}; font-size: 11px;
        }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 8px; }}
        QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {{
            background: #333; border: 1px solid #555;
            border-radius: 3px; color: {C_TEXT}; padding: 2px 4px;
        }}
        QPushButton {{
            background: {C_SURFACE}; border: 1px solid #555;
            border-radius: 4px; color: {C_TEXT}; padding: 5px 12px;
        }}
        QPushButton:hover {{ background: #3a3a3a; border-color: {C_ACCENT}; }}
        QPushButton#run_btn {{
            background: {C_ACCENT}; border: none; font-weight: bold;
            color: white; padding: 8px 16px; border-radius: 5px;
        }}
        QPushButton#run_btn:hover {{ background: #5aa0e9; }}
        QPushButton#run_btn:disabled {{ background: #444; color: {C_MUTED}; }}
        QTabWidget::pane {{ border: 1px solid #444; }}
        QTabBar::tab {{
            background: {C_SURFACE}; color: {C_MUTED};
            padding: 6px 14px; border-radius: 3px 3px 0 0;
        }}
        QTabBar::tab:selected {{ background: #363636; color: {C_TEXT}; }}
        QTableWidget {{ gridline-color: #444; }}
        QHeaderView::section {{
            background: #333; color: {C_MUTED};
            border: 1px solid #444; padding: 4px;
        }}
        QProgressBar {{
            border: 1px solid #555; border-radius: 3px;
            background: #333; text-align: center; color: {C_TEXT};
        }}
        QProgressBar::chunk {{ background: {C_ACCENT}; border-radius: 3px; }}
        QScrollBar:vertical {{ background: {C_SURFACE}; width: 10px; }}
        QScrollBar::handle:vertical {{ background: #555; border-radius: 5px; }}
        QRadioButton {{ color: {C_TEXT}; }}
        QCheckBox {{ color: {C_TEXT}; }}
        QLabel {{ color: {C_TEXT}; }}
    """)


# ---------------------------------------------------------------------------
# Left panel: Input & Controls
# ---------------------------------------------------------------------------

class InputPanel(QWidget):
    """Left panel for participant input, file browsing, and pipeline control."""

    run_requested = Signal(dict)   # emits config dict when user clicks Run

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(290)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ── Header ────────────────────────────────────────────────────
        hdr = QLabel("Gait Analysis")
        hdr.setStyleSheet(f"font-size: 15px; font-weight: bold; color: {C_ACCENT};")
        root.addWidget(hdr)

        # ── Participant ID ────────────────────────────────────────────
        grp_id = QGroupBox("Participant")
        lay_id = QVBoxLayout(grp_id)
        self.pid_edit = QLineEdit("sub_01")
        lay_id.addWidget(self.pid_edit)
        root.addWidget(grp_id)

        # ── ST input ──────────────────────────────────────────────────
        self.st_group = self._build_file_input_group(
            "Single-Task (ST)", "st"
        )
        root.addWidget(self.st_group)

        # ── DT input ──────────────────────────────────────────────────
        self.dt_group = self._build_file_input_group(
            "Dual-Task (DT)", "dt"
        )
        root.addWidget(self.dt_group)

        # ── Numeric params ────────────────────────────────────────────
        grp_num = QGroupBox("Recording Parameters")
        lay_num = QVBoxLayout(grp_num)
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Height (m):"))
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(1.0, 2.5)
        self.height_spin.setValue(1.70)
        self.height_spin.setSingleStep(0.01)
        row1.addWidget(self.height_spin)
        lay_num.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("FPS:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 500)
        self.fps_spin.setValue(120)
        row2.addWidget(self.fps_spin)
        lay_num.addLayout(row2)
        root.addWidget(grp_num)

        # ── Run button + progress ─────────────────────────────────────
        self.run_btn = QPushButton("▶  Run Analysis")
        self.run_btn.setObjectName("run_btn")
        self.run_btn.clicked.connect(self._on_run)
        root.addWidget(self.run_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        root.addWidget(self.progress_bar)

        # Stage checklist
        self.stage_log = QTextEdit()
        self.stage_log.setReadOnly(True)
        self.stage_log.setMaximumHeight(230)
        self.stage_log.setStyleSheet(
            f"background:{C_SURFACE}; color:{C_MUTED}; font-family: monospace; font-size: 11px;"
        )
        root.addWidget(self.stage_log)

        # ── Output directory ──────────────────────────────────────────
        grp_out = QGroupBox("Output")
        lay_out = QVBoxLayout(grp_out)
        row_out = QHBoxLayout()
        self.out_edit = QLineEdit(str(Path.home() / "gait_results"))
        row_out.addWidget(self.out_edit)
        btn_out = QPushButton("…")
        btn_out.setFixedWidth(30)
        btn_out.clicked.connect(self._browse_output)
        row_out.addWidget(btn_out)
        lay_out.addLayout(row_out)
        root.addWidget(grp_out)

        root.addStretch()

    def _build_file_input_group(self, label: str, cond: str) -> QGroupBox:
        grp = QGroupBox(label)
        lay = QVBoxLayout(grp)

        # Radio buttons: TRC vs Video (Addendum 2)
        btn_grp = QButtonGroup(grp)
        rdo_trc = QRadioButton(".trc file")
        rdo_vid = QRadioButton("Video file")
        rdo_trc.setChecked(True)
        btn_grp.addButton(rdo_trc, 0)
        btn_grp.addButton(rdo_vid, 1)
        row_rdo = QHBoxLayout()
        row_rdo.addWidget(rdo_trc)
        row_rdo.addWidget(rdo_vid)
        lay.addLayout(row_rdo)

        path_edit = QLineEdit()
        path_edit.setPlaceholderText("Path to file…")
        browse_btn = QPushButton("Browse")

        def browse():
            if btn_grp.checkedId() == 0:  # TRC
                filt = "TRC files (*.trc *.csv);;All files (*)"
            else:                          # Video
                filt = "Video files (*.mp4 *.avi *.mov *.mkv);;All files (*)"
            path, _ = QFileDialog.getOpenFileName(grp, f"Select {label} file", "", filt)
            if path:
                path_edit.setText(path)

        browse_btn.clicked.connect(browse)
        row_path = QHBoxLayout()
        row_path.addWidget(path_edit)
        row_path.addWidget(browse_btn)
        lay.addLayout(row_path)

        # Store references
        setattr(self, f"{cond}_rdo_grp",  btn_grp)
        setattr(self, f"{cond}_path_edit", path_edit)
        setattr(self, f"{cond}_rdo_trc",  rdo_trc)
        setattr(self, f"{cond}_rdo_vid",  rdo_vid)
        return grp

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.out_edit.setText(path)

    def _on_run(self):
        config = {
            "participant_id": self.pid_edit.text().strip() or "sub_01",
            "st_input":       getattr(self, "st_path_edit").text().strip(),
            "dt_input":       getattr(self, "dt_path_edit").text().strip(),
            "st_is_video":    getattr(self, "st_rdo_grp").checkedId() == 1,
            "dt_is_video":    getattr(self, "dt_rdo_grp").checkedId() == 1,
            "height_m":       self.height_spin.value(),
            "fps":            self.fps_spin.value(),
            "output_dir":     self.out_edit.text().strip(),
        }
        if not config["st_input"] or not config["dt_input"]:
            QMessageBox.warning(self, "Missing Input", "Please provide ST and DT file paths.")
            return
        self.run_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.stage_log.clear()
        self.run_requested.emit(config)

    def update_stage(self, stage_name: str, status: str, pct: int):
        icon = STATUS_ICON.get(StageStatus(status), "?")
        self.stage_log.append(f"{icon} {stage_name}")
        self.progress_bar.setValue(pct)

    def on_finished(self):
        self.run_btn.setEnabled(True)
        self.progress_bar.setValue(100)

    def on_error(self, msg: str):
        self.run_btn.setEnabled(True)
        QMessageBox.critical(self, "Pipeline Error", msg[:800])


# ---------------------------------------------------------------------------
# Results Tabs
# ---------------------------------------------------------------------------

class ParameterTab(QWidget):
    """Tab showing time-series stride-by-stride parameters with outliers shown."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.title = title
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._container = QWidget()
        self._container_layout = QVBoxLayout(self._container)
        self._scroll.setWidget(self._container)
        self._layout.addWidget(self._scroll)

    def load_data(self, strides_df: pd.DataFrame):
        """Render time-series charts for each parameter, one graph per param."""
        # Clear previous content
        while self._container_layout.count():
            child = self._container_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        if strides_df is None or strides_df.empty:
            self._container_layout.addWidget(QLabel("No data available."))
            return

        params = [
            "stride_times", "stance_times", "swing_times",
            "stride_lengths", "stance_ratios", "step_time",
            "double_support_time",
        ]

        for param in params:
            if param not in strides_df.columns:
                continue
            chart = self._make_chart(strides_df, param)
            if chart:
                self._container_layout.addWidget(chart)

        self._container_layout.addStretch()

    def _make_chart(self, df: pd.DataFrame, param: str) -> Optional[QWidget]:
        if HAS_PYQTGRAPH:
            return self._pg_chart(df, param)
        return self._mpl_chart(df, param)

    def _pg_chart(self, df: pd.DataFrame, param: str) -> QWidget:
        pw = pg.PlotWidget(title=param.replace("_", " ").title())
        pw.setBackground(C_SURFACE)
        pw.setFixedHeight(160)
        pw.showGrid(x=True, y=True, alpha=0.3)
        pw.setLabel("left",   param.replace("_", " "), color=C_MUTED)
        pw.setLabel("bottom", "Time (s)",               color=C_MUTED)

        colors = {"left": C_LEFT_FOOT, "right": C_RIGHT_FOOT}

        for side, col_hex in colors.items():
            sub = df[df["foot"] == side].copy()
            if sub.empty:
                continue
            valid   = sub[sub["is_outlier"] == False]
            invalid = sub[sub["is_outlier"] == True]

            col_rgb = QColor(col_hex)
            pen = pg.mkPen(color=col_rgb, width=2)
            pw.plot(valid["timestamps"].values, valid[param].values,
                    pen=pen, name=side)
            # Greyed-out outlier points
            grey = pg.ScatterPlotItem(
                x=invalid["timestamps"].values,
                y=invalid[param].values,
                brush=pg.mkBrush(QColor("#555555")),
                pen=pg.mkPen(None), size=6
            )
            pw.addItem(grey)

        pw.addLegend()
        return pw

    def _mpl_chart(self, df: pd.DataFrame, param: str) -> QWidget:
        fig, ax = plt.subplots(figsize=(7, 2.2))
        fig.patch.set_facecolor(C_SURFACE)
        ax.set_facecolor(C_BG)
        ax.tick_params(colors=C_MUTED)
        ax.spines[:].set_color(C_MUTED)
        ax.set_title(param.replace("_", " ").title(), color=C_TEXT, fontsize=10)
        ax.set_xlabel("Time (s)", color=C_MUTED, fontsize=9)
        ax.set_ylabel(param.replace("_", " "), color=C_MUTED, fontsize=9)

        colors = {"left": C_LEFT_FOOT, "right": C_RIGHT_FOOT}
        for side, col in colors.items():
            sub   = df[df["foot"] == side]
            valid = sub[sub["is_outlier"] == False]
            inv   = sub[sub["is_outlier"] == True]
            ax.plot(valid["timestamps"], valid[param], color=col,
                    linewidth=1.5, label=side)
            ax.scatter(inv["timestamps"], inv[param],
                       color="#555555", s=20, zorder=5)

        ax.legend(facecolor=C_SURFACE, labelcolor=C_TEXT, fontsize=8)
        fig.tight_layout(pad=0.5)
        canvas = FigureCanvas(fig)
        canvas.setFixedHeight(200)
        plt.close(fig)
        return canvas


class DTCTab(QWidget):
    """Tab displaying Dual-Task Cost bar chart and summary table."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        self._chart_holder = QWidget()
        self._chart_holder.setFixedHeight(320)
        lay.addWidget(self._chart_holder)

        self._table = QTableWidget()
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        lay.addWidget(self._table)

    def load_data(self, dtc_df: pd.DataFrame, dtc_summary: pd.DataFrame):
        if dtc_df is None or dtc_df.empty:
            return
        self._render_chart(dtc_summary)
        self._render_table(dtc_summary)

    def _render_chart(self, summary: pd.DataFrame):
        layout = self._chart_holder.layout()
        if layout:
            while layout.count():
                c = layout.takeAt(0)
                if c.widget():
                    c.widget().deleteLater()
        else:
            layout = QVBoxLayout(self._chart_holder)

        params  = summary["parameter"].tolist()
        values  = summary["DTC_pct"].tolist()
        colors  = [C_POSITIVE if v > 0 else (C_NEGATIVE if v < 0 else "#888888")
                   for v in values]

        if HAS_PYQTGRAPH:
            pw = pg.PlotWidget(title="Dual-Task Cost (%)")
            pw.setBackground(C_SURFACE)
            pw.showGrid(y=True, alpha=0.3)
            pw.setLabel("left", "DTC (%)", color=C_MUTED)
            x = list(range(len(params)))
            bars = pg.BarGraphItem(x=x, height=values,
                                   width=0.6, brushes=colors)
            pw.addItem(bars)
            pw.addLine(y=0, pen=pg.mkPen(C_MUTED, width=1, style=Qt.PenStyle.DashLine))
            ticks = [(i, p.replace("_", "\n")) for i, p in enumerate(params)]
            pw.getAxis("bottom").setTicks([ticks])
            layout.addWidget(pw)
        else:
            fig, ax = plt.subplots(figsize=(9, 3))
            fig.patch.set_facecolor(C_SURFACE)
            ax.set_facecolor(C_BG)
            ax.tick_params(colors=C_MUTED, labelsize=7)
            ax.spines[:].set_color(C_MUTED)
            ax.bar(params, values, color=colors)
            ax.axhline(0, color=C_MUTED, linewidth=1, linestyle="--")
            ax.set_title("Dual-Task Cost (%)", color=C_TEXT)
            ax.set_ylabel("DTC (%)", color=C_MUTED)
            ax.set_xticklabels(params, rotation=30, ha="right")
            fig.tight_layout()
            canvas = FigureCanvas(fig)
            layout.addWidget(canvas)
            plt.close(fig)

    def _render_table(self, summary: pd.DataFrame):
        self._table.clear()
        if summary.empty:
            return
        self._table.setRowCount(len(summary))
        self._table.setColumnCount(3)
        self._table.setHorizontalHeaderLabels(["Parameter", "DTC (%)", "Direction"])
        for row, r in summary.iterrows():
            self._table.setItem(row, 0, QTableWidgetItem(str(r["parameter"])))
            val_item = QTableWidgetItem(f"{r['DTC_pct']:.3f}")
            val_item.setForeground(
                QColor(C_POSITIVE) if r["DTC_pct"] > 0 else
                QColor(C_NEGATIVE) if r["DTC_pct"] < 0 else
                QColor(C_MUTED)
            )
            self._table.setItem(row, 1, val_item)
            self._table.setItem(row, 2, QTableWidgetItem(str(r["direction"])))
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)


class RawDataTab(QWidget):
    """Tab showing full stride-by-stride data with outlier highlighting, sort, and filter."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        # Filter bar
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Filter:"))
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Type to filter rows by any value…")
        self.filter_edit.textChanged.connect(self._apply_filter)
        bar.addWidget(self.filter_edit)
        lay.addLayout(bar)

        self._table = QTableWidget()
        self._table.setSortingEnabled(True)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        lay.addWidget(self._table)

        self._full_df: Optional[pd.DataFrame] = None

    def load_data(self, strides_st: pd.DataFrame, strides_dt: pd.DataFrame):
        dfs = []
        for df, cond in [(strides_st, "st"), (strides_dt, "dt")]:
            if df is not None and not df.empty:
                d = df.copy()
                d.insert(0, "condition", cond)
                dfs.append(d)
        if not dfs:
            return
        self._full_df = pd.concat(dfs, ignore_index=True)
        self._render(self._full_df)

    def _render(self, df: pd.DataFrame):
        self._table.setRowCount(0)
        if df.empty:
            return
        cols = list(df.columns)
        self._table.setColumnCount(len(cols))
        self._table.setHorizontalHeaderLabels(cols)
        self._table.setRowCount(len(df))

        outlier_col = cols.index("is_outlier") if "is_outlier" in cols else -1

        for row_i, (_, row) in enumerate(df.iterrows()):
            is_out = bool(row.get("is_outlier", False))
            for col_i, col in enumerate(cols):
                val = row[col]
                item = QTableWidgetItem(
                    f"{val:.4f}" if isinstance(val, float) else str(val)
                )
                if is_out:
                    item.setBackground(QColor(C_OUTLIER))
                self._table.setItem(row_i, col_i, item)

    def _apply_filter(self, text: str):
        if self._full_df is None:
            return
        if not text:
            self._render(self._full_df)
            return
        mask = self._full_df.apply(
            lambda col: col.astype(str).str.contains(text, case=False, na=False)
        ).any(axis=1)
        self._render(self._full_df[mask].reset_index(drop=True))


class VideoTab(QWidget):
    """Placeholder for annotated video playback (shown if video path provided)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        self._label = QLabel(
            "No video loaded.\n\n"
            "Provide a video path in the left panel and run the pipeline.\n"
            "Keypoint overlays will be shown here when available."
        )
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet(f"color: {C_MUTED}; font-size: 13px;")
        lay.addWidget(self._label)


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gait Analysis — Sports2D + DUO-GAIT")
        self.resize(1280, 820)
        self._runner: Optional[PipelineRunner] = None
        self._results: dict = {}

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Left panel
        self._input_panel = InputPanel()
        self._input_panel.run_requested.connect(self._on_run_requested)
        root.addWidget(self._input_panel)

        # Divider
        div = QFrame()
        div.setFrameShape(QFrame.Shape.VLine)
        div.setStyleSheet("color: #444;")
        root.addWidget(div)

        # Right area — results tabs
        self._tabs = QTabWidget()
        root.addWidget(self._tabs, stretch=1)

        # Participant selector (for future batch mode)
        self._sub_selector = QComboBox()
        self._sub_selector.setFixedWidth(180)

        self._video_tab    = VideoTab()
        self._st_tab       = ParameterTab("Single-Task Gait Parameters")
        self._dt_tab       = ParameterTab("Dual-Task Gait Parameters")
        self._dtc_tab      = DTCTab()
        self._raw_tab      = RawDataTab()

        self._tabs.addTab(self._video_tab, "📽  Annotated Video")
        self._tabs.addTab(self._st_tab,    "🦶 ST Parameters")
        self._tabs.addTab(self._dt_tab,    "🧮 DT Parameters")
        self._tabs.addTab(self._dtc_tab,   "📊 Dual-Task Cost")
        self._tabs.addTab(self._raw_tab,   "📋 Raw Data")

    # ------------------------------------------------------------------
    # Pipeline control
    # ------------------------------------------------------------------

    @Slot(dict)
    def _on_run_requested(self, config: dict):
        self._runner = PipelineRunner(
            participant_id = config["participant_id"],
            st_input       = config["st_input"],
            dt_input       = config["dt_input"],
            st_is_video    = config["st_is_video"],
            dt_is_video    = config["dt_is_video"],
            height_m       = config["height_m"],
            fps            = config["fps"],
            output_dir     = config["output_dir"],
        )
        self._runner.progress.connect(self._on_progress)
        self._runner.finished.connect(self._on_finished)
        self._runner.error.connect(self._on_error)
        self._runner.start()

    @Slot(str, str, int)
    def _on_progress(self, stage_name: str, status: str, pct: int):
        self._input_panel.update_stage(stage_name, status, pct)

    @Slot(dict)
    def _on_finished(self, results: dict):
        self._results = results
        self._input_panel.on_finished()

        # Populate result tabs
        try:
            self._st_tab.load_data(results.get("remove_outliers_st"))
            self._dt_tab.load_data(results.get("remove_outliers_dt"))

            dtc_payload = results.get("dtc", {})
            if isinstance(dtc_payload, dict):
                self._dtc_tab.load_data(
                    dtc_payload.get("dtc"),
                    dtc_payload.get("dtc_summary"),
                )

            self._raw_tab.load_data(
                results.get("calc_params_st"),
                results.get("calc_params_dt"),
            )

            self._tabs.setCurrentIndex(3)  # jump to DTC tab
        except Exception as e:
            QMessageBox.warning(self, "Display Error", str(e))

    @Slot(str)
    def _on_error(self, message: str):
        self._input_panel.on_error(message)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def launch():
    app = QApplication(sys.argv)
    _apply_dark_palette(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    launch()
