#!/usr/bin/env python3
"""
Optimization Monitor — Dark-mode GUI dashboard for PSO workers.

Windowed layout with scrollable plot cards, manual refresh, and a polished
dark theme. Each run gets its own card so plots never overlap.

Usage:
    python monitor.py
"""

import os
import sys
# Ensure we import config.py from THIS script's directory, not cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time
import re
from pathlib import Path

# Force Python to find config.py in the same folder as this script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# ── Qt backend (matches your existing Qt5Agg) ──────────────────────────
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QScrollArea, QFrame, QStatusBar, QToolBar,
        QSizePolicy, QGridLayout, QSpacerItem, QMessageBox
    )
    from PyQt5.QtCore import Qt, QSize
    from PyQt5.QtGui import QFont, QColor, QPalette
    QT = "PyQt5"
except ImportError:
    try:
        from PySide2.QtWidgets import (
            QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
            QPushButton, QLabel, QScrollArea, QFrame, QStatusBar, QToolBar,
            QSizePolicy, QGridLayout, QSpacerItem, QMessageBox
        )
        from PySide2.QtCore import Qt, QSize
        from PySide2.QtGui import QFont, QColor, QPalette
        QT = "PySide2"
    except ImportError:
        print("Error: PyQt5 or PySide2 required.  Install: pip install PyQt5")
        sys.exit(1)

# ── Matplotlib ────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.patheffects as pe

# ── Config (graceful fallback) ────────────────────────────────────────
try:
    from config import HIST_LOGS_DIR, MONITOR_ACTIVE_TIMEOUT, PEN_THRESHOLD
except ImportError:
    print("Cannot import")
    HIST_LOGS_DIR = "/home/lavender/Studies/Design of Wind Farms/Assignments/Assignment6/modular/history_logs"
    MONITOR_ACTIVE_TIMEOUT = 120
    PEN_THRESHOLD = 1.0

# ── Constants ─────────────────────────────────────────────────────────
HISTORY_DIR = Path(HIST_LOGS_DIR)
ACTIVE_TIMEOUT = MONITOR_ACTIVE_TIMEOUT

# Dark theme palette
DARK_BG      = "#1e1e1e"
CARD_BG      = "#2d2d2d"
AXES_BG      = "#252525"
TEXT_COLOR   = "#e0e0e0"
MUTED_TEXT   = "#888888"
ACCENT_BLUE  = "#4fc3f7"
ACCENT_RED   = "#ff5252"
ACCENT_GREEN = "#81c784"
ACCENT_ORANGE = "#ffb74d"
GRID_COLOR   = "#444444"

# ── Helpers ───────────────────────────────────────────────────────────

def _parse_hist_file(path: Path):
    if not path.exists():
        return None, False
    try:
        with open(path, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if not lines:
            return np.array([]), False
        finished = lines[-1] == "END"
        if finished:
            lines = lines[:-1]
        vals = [float(ln) for ln in lines]
        return np.array(vals), finished
    except Exception:
        return None, False


def _discover_runs() -> list[dict]:
    if not HISTORY_DIR.exists():
        return []
    pen_files = sorted(HISTORY_DIR.glob("*_penalty_hist.txt"))
    runs = []
    for pen in pen_files:
        stem = pen.name.replace("_penalty_hist.txt", "")
        aep = HISTORY_DIR / f"{stem}_AEP_hist.txt"
        if not aep.exists():
            continue
        m = re.match(r"(.+)_(\d+)$", stem)
        if m:
            t_name, n_turb = m.group(1), m.group(2)
            title = f"{t_name}  ({n_turb} turbines)"
        else:
            title = stem
        runs.append({
            "name": title,
            "stem": stem,
            "pen_file": pen,
            "aep_file": aep,
        })
    return runs


def _is_active(run: dict) -> bool:
    pen_path = run["pen_file"]
    if not pen_path.exists():
        return False
    mtime = pen_path.stat().st_mtime
    age = time.time() - mtime
    if age > ACTIVE_TIMEOUT:
        return False
    _, finished = _parse_hist_file(pen_path)
    return not finished


# ── Plot Card (one per run) ───────────────────────────────────────────

class PlotCard(QFrame):
    """A single run's plot embedded in a dark card."""

    def __init__(self, run: dict, parent=None):
        super().__init__(parent)
        self.run = run
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(f"""
            PlotCard {{
                background-color: {CARD_BG};
                border: 1px solid #3d3d3d;
                border-radius: 8px;
            }}
        """)
        self.setMinimumSize(520, 340)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        # Title
        self.title_lbl = QLabel(run["name"])
        self.title_lbl.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 13px; font-weight: bold;")
        self.title_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_lbl)

        # Matplotlib figure
        self.fig = Figure(figsize=(5.0, 3.2), dpi=100, facecolor=AXES_BG)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet(f"background-color: {AXES_BG}; border-radius: 4px;")
        layout.addWidget(self.canvas)

        # Toolbar (pan/zoom/save)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet(f"""
            QToolBar {{
                background-color: {CARD_BG};
                border: none;
                spacing: 4px;
            }}
            QToolButton {{
                color: {TEXT_COLOR};
                background-color: transparent;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 2px;
                margin: 1px;
            }}
            QToolButton:hover {{
                background-color: #3d3d3d;
            }}
        """)
        layout.addWidget(self.toolbar)

        # Status label
        self.status_lbl = QLabel("No data")
        self.status_lbl.setStyleSheet(f"color: {MUTED_TEXT}; font-size: 10px;")
        self.status_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_lbl)

        # Build axes
        self.ax_pen = self.fig.add_subplot(111)
        self.ax_pen.set_facecolor(AXES_BG)
        self.ax_aep = self.ax_pen.twinx()
        self.ax_aep.set_facecolor(AXES_BG)

        for ax in (self.ax_pen, self.ax_aep):
            ax.tick_params(colors=MUTED_TEXT, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#555555")
            ax.grid(True, alpha=0.2, color=GRID_COLOR)

        self.ax_pen.set_xlabel("Iteration", color=MUTED_TEXT, fontsize=9)
        self.ax_pen.set_ylabel("Penalty", color=ACCENT_RED, fontsize=9)
        self.ax_pen.set_yscale("log")

        self.ax_aep.set_ylabel("AEP (GWh)", color=ACCENT_BLUE, fontsize=9)

        self.fig.tight_layout(pad=2.0)

        # Threshold line (static)
        self.ax_pen.axhline(
            PEN_THRESHOLD, color=ACCENT_RED, linestyle="--",
            alpha=0.4, linewidth=1, label="threshold",
        )

        self._pen_line = None
        self._aep_line = None

    def update_data(self):
        pen, pen_done = _parse_hist_file(self.run["pen_file"])
        aep, aep_done = _parse_hist_file(self.run["aep_file"])

        done = pen_done or aep_done

        # Title styling
        if done:
            self.title_lbl.setText(f"{self.run['name']}  [DONE]")
            self.title_lbl.setStyleSheet(f"color: {ACCENT_GREEN}; font-size: 13px; font-weight: bold;")
            self.status_lbl.setText("Finished")
            self.status_lbl.setStyleSheet(f"color: {ACCENT_GREEN}; font-size: 10px;")
        else:
            self.title_lbl.setText(self.run["name"])
            self.title_lbl.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 13px; font-weight: bold;")
            if pen is not None and len(pen) > 0:
                self.status_lbl.setText(f"Iterations: {len(pen)}")
                self.status_lbl.setStyleSheet(f"color: {ACCENT_BLUE}; font-size: 10px;")
            else:
                self.status_lbl.setText("Waiting for data…")
                self.status_lbl.setStyleSheet(f"color: {MUTED_TEXT}; font-size: 10px;")

        # Clear old data lines (keep threshold)
        for line in list(self.ax_pen.get_lines()):
            if line.get_label() != "threshold":
                line.remove()
        for line in list(self.ax_aep.get_lines()):
            line.remove()

        if pen is None or aep is None or len(pen) == 0:
            self.ax_pen.text(
                0.5, 0.5, "No data yet",
                transform=self.ax_pen.transAxes,
                ha="center", va="center",
                fontsize=14, color=MUTED_TEXT, alpha=0.6,
            )
            self.ax_pen.set_xlim(0, 10)
            self.ax_pen.set_ylim(1e-12, 1.0)
            self.ax_aep.set_ylim(0, 1)
            self.canvas.draw()
            return

        iters = np.arange(len(pen))

        # Plot
        self._pen_line = self.ax_pen.plot(
            iters, pen, color=ACCENT_RED, marker="o",
            markersize=3.5, linewidth=1.5, label="Penalty",
            path_effects=[pe.withStroke(linewidth=3, foreground=AXES_BG, alpha=0.8)]
        )[0]
        self._aep_line = self.ax_aep.plot(
            iters, aep, color=ACCENT_BLUE, marker="x",
            markersize=3.5, linewidth=1.5, label="AEP",
            path_effects=[pe.withStroke(linewidth=3, foreground=AXES_BG, alpha=0.8)]
        )[0]

        # Limits
        self.ax_pen.set_xlim(0, max(10, len(pen) + 2))
        if np.any(pen > 0):
            lo = max(1e-12, np.min(pen[pen > 0]) * 0.5)
        else:
            lo = 1e-12
        hi = np.max(pen) * 2 if np.max(pen) > 0 else 1.0
        self.ax_pen.set_ylim(lo, hi)

        if len(aep) > 0:
            pad = max((np.max(aep) - np.min(aep)) * 0.1, 1e-3)
            self.ax_aep.set_ylim(np.min(aep) - pad, np.max(aep) + pad)

        # Legend
        lines = [self._pen_line, self._aep_line]
        labels = ["Penalty", "AEP"]
        self.ax_pen.legend(lines, labels, loc="upper right", fontsize=8,
                           facecolor=AXES_BG, edgecolor="#555555",
                           labelcolor=TEXT_COLOR)

        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()


# ── Main Window ───────────────────────────────────────────────────────

class MonitorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optimization Monitor")
        self.setMinimumSize(1100, 700)
        self.resize(1400, 900)

        self._apply_dark_theme()

        # Central scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"QScrollArea {{ border: none; background-color: {DARK_BG}; }}")
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Scrollbar styling
        scroll.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background-color: {DARK_BG};
            }}
            QScrollBar:vertical {{
                background-color: {DARK_BG};
                width: 12px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background-color: #555555;
                border-radius: 6px;
                min-height: 30px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: #777777;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar:horizontal {{
                background-color: {DARK_BG};
                height: 12px;
                margin: 0px;
            }}
            QScrollBar::handle:horizontal {{
                background-color: #555555;
                border-radius: 6px;
                min-width: 30px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background-color: #777777;
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0px;
            }}
        """)

        self.container = QWidget()
        self.container.setStyleSheet(f"background-color: {DARK_BG};")
        self.grid_layout = QGridLayout(self.container)
        self.grid_layout.setSpacing(12)
        self.grid_layout.setContentsMargins(16, 16, 16, 16)
        self.grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        scroll.setWidget(self.container)
        self.setCentralWidget(scroll)

        # Toolbar
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setStyleSheet(f"""
            QToolBar {{
                background-color: {CARD_BG};
                border-bottom: 1px solid #3d3d3d;
                spacing: 10px;
                padding: 6px;
            }}
        """)
        self.addToolBar(toolbar)

                # Refresh button
        self.refresh_btn = QPushButton("🔄  Refresh")
        self.refresh_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {ACCENT_BLUE};
                color: #1e1e1e;
                font-weight: bold;
                font-size: 13px;
                padding: 8px 20px;
                border-radius: 6px;
                border: none;
            }}
            QPushButton:hover {{
                background-color: #6fd5ff;
            }}
            QPushButton:pressed {{
                background-color: #3aa8d8;
            }}
        """)
        self.refresh_btn.setCursor(Qt.PointingHandCursor)
        self.refresh_btn.clicked.connect(self.do_refresh)
        toolbar.addWidget(self.refresh_btn)

        # Mode label
        self.mode_lbl = QLabel("Mode: Active runs only")
        self.mode_lbl.setStyleSheet(f"color: {MUTED_TEXT}; font-size: 12px; padding-left: 10px;")
        toolbar.addWidget(self.mode_lbl)

        # Spacer to push remaining widgets to the right
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)

        # Show-all toggle
        self.show_all_btn = QPushButton("Show All Runs")
        self.show_all_btn.setCheckable(True)
        self.show_all_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {TEXT_COLOR};
                font-size: 12px;
                padding: 6px 14px;
                border-radius: 6px;
                border: 1px solid #555555;
            }}
            QPushButton:checked {{
                background-color: {ACCENT_ORANGE};
                color: #1e1e1e;
                border: 1px solid {ACCENT_ORANGE};
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #3d3d3d;
            }}
            QPushButton:checked:hover {{
                background-color: #ffcc80;
            }}
        """)
        self.show_all_btn.clicked.connect(self.do_refresh)
        toolbar.addWidget(self.show_all_btn)

        # Status bar
        self.status = QStatusBar()
        self.status.setStyleSheet(f"""
            QStatusBar {{
                background-color: {CARD_BG};
                color: {MUTED_TEXT};
                border-top: 1px solid #3d3d3d;
                padding: 4px 12px;
                font-size: 12px;
            }}
        """)
        self.setStatusBar(self.status)

        # Data
        self.cards: dict[str, PlotCard] = {}
        self.showing_all = False

        # Initial load
        self.do_refresh()

    def _apply_dark_theme(self):
        app = QApplication.instance()
        pal = QPalette()
        pal.setColor(QPalette.Window, QColor(DARK_BG))
        pal.setColor(QPalette.WindowText, QColor(TEXT_COLOR))
        pal.setColor(QPalette.Base, QColor(DARK_BG))
        pal.setColor(QPalette.AlternateBase, QColor(CARD_BG))
        pal.setColor(QPalette.ToolTipBase, QColor(CARD_BG))
        pal.setColor(QPalette.ToolTipText, QColor(TEXT_COLOR))
        pal.setColor(QPalette.Text, QColor(TEXT_COLOR))
        pal.setColor(QPalette.Button, QColor(CARD_BG))
        pal.setColor(QPalette.ButtonText, QColor(TEXT_COLOR))
        pal.setColor(QPalette.BrightText, QColor(ACCENT_RED))
        pal.setColor(QPalette.Highlight, QColor(ACCENT_BLUE))
        pal.setColor(QPalette.HighlightedText, QColor("#1e1e1e"))
        app.setPalette(pal)

    def do_refresh(self):
        self.refresh_btn.setEnabled(False)
        self.refresh_btn.setText("⏳  Refreshing…")
        self.status.showMessage("Scanning history_logs/ …")

        all_runs = _discover_runs()

        if not all_runs:
            self._clear_cards()
            self.status.showMessage(f"No runs found in {HISTORY_DIR.resolve()}  |  Last check: {time.strftime('%H:%M:%S')}")
            self.mode_lbl.setText("Mode: —")
            self.refresh_btn.setEnabled(True)
            self.refresh_btn.setText("🔄  Refresh")
            return

        # Filter
        show_all = self.show_all_btn.isChecked()
        if show_all:
            runs = all_runs
            self.mode_lbl.setText("Mode: All runs")
            self.mode_lbl.setStyleSheet(f"color: {ACCENT_ORANGE}; font-size: 12px; padding-left: 10px;")
        else:
            active = [r for r in all_runs if _is_active(r)]
            if active:
                runs = active
                self.mode_lbl.setText("Mode: Active runs only")
                self.mode_lbl.setStyleSheet(f"color: {ACCENT_GREEN}; font-size: 12px; padding-left: 10px;")
            else:
                runs = all_runs
                self.mode_lbl.setText("Mode: All runs (nothing active)")
                self.mode_lbl.setStyleSheet(f"color: {ACCENT_ORANGE}; font-size: 12px; padding-left: 10px;")

        # Rebuild grid if run list changed
        current_stems = set(self.cards.keys())
        new_stems = {r["stem"] for r in runs}

        if current_stems != new_stems:
            self._clear_cards()
            for run in runs:
                card = PlotCard(run)
                self.cards[run["stem"]] = card

            # Layout in grid: 2 columns (or 1 if window is narrow)
            cols = 2
            for idx, run in enumerate(runs):
                r = idx // cols
                c = idx % cols
                self.grid_layout.addWidget(self.cards[run["stem"]], r, c)

            # Add stretch at bottom so cards stay at top
            self.grid_layout.setRowStretch((len(runs) // cols) + 1, 1)

        # Update data on all cards
        for run in runs:
            if run["stem"] in self.cards:
                self.cards[run["stem"]].update_data()

        finished_count = sum(1 for r in runs if _parse_hist_file(r["pen_file"])[1])
        self.status.showMessage(
            f"{len(runs)} run(s) displayed  |  {finished_count} finished  |  "
            f"Watching: {HISTORY_DIR.resolve()}  |  Last refresh: {time.strftime('%H:%M:%S')}"
        )

        self.refresh_btn.setEnabled(True)
        self.refresh_btn.setText("🔄  Refresh")

    def _clear_cards(self):
        for card in self.cards.values():
            card.deleteLater()
        self.cards.clear()
        # Clear grid
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()


# ── Entry point ───────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(" Optimization Monitor  —  Dark Mode GUI")
    print("=" * 60)
    print(f"Watching : {HISTORY_DIR.resolve()}")
    print(f"Timeout  : {ACTIVE_TIMEOUT:.0f} s (stale files hidden while active)")
    print("Close window to exit\n")

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    win = MonitorWindow()
    win.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()