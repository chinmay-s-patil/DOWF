#!/usr/bin/env python3
"""
Optimization Monitor — Live dashboard for all running PSO workers.

Watches history_logs/ for penalty and AEP text files written by each
parallel worker and displays them in a single figure with per-run subplots.

Usage:
    python monitor.py

Press Ctrl+C to quit.  The plot auto-detects new runs and refreshes every
second.  Each worker gets its own twin-axis subplot (penalty log-scale left,
AEP linear right).

Logic:
  - A run is "active" if its log file has been modified within the last
    ACTIVE_TIMEOUT seconds AND it does not contain an "END" marker.
  - While any active runs exist, only active runs are shown.
  - If ALL runs are finished (or stale), every run is shown.
  - This means when the optimization is running you see live workers only.
  - When everything is done, you see the full summary.

Add "END" as the last line of a log file to mark that run as finished.
"""

import os
import time
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HISTORY_DIR = Path("/home/lavender/Studies/Design of Wind Farms/Assignments/Assignment6/LoCE Opti/history_logs")
POLL_INTERVAL = 20.0               # seconds between refreshes
PENALTY_THRESHOLD = 1e-4
ACTIVE_TIMEOUT = 60.0             # seconds; file older than this = stale

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_hist_file(path: Path):
    """
    Read a text file with one float per line.
    Returns (values_array, finished_flag).
    finished_flag is True if the last line is exactly "END".
    """
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


def _is_active(run: dict) -> bool:
    """
    A run is active if its penalty file was modified within ACTIVE_TIMEOUT
    seconds AND it does not have an END marker.
    """
    pen_path = run["pen_file"]
    if not pen_path.exists():
        return False
    mtime = pen_path.stat().st_mtime
    age = time.time() - mtime
    if age > ACTIVE_TIMEOUT:
        return False
    _, finished = _parse_hist_file(pen_path)
    return not finished


def _discover_runs() -> list[dict]:
    """
    Scan HISTORY_DIR and return a list of run dicts:
        { "name": str, "stem": str, "pen_file": Path, "aep_file": Path }
    """
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


def _grid_shape(n: int) -> tuple[int, int]:
    """Return (rows, cols) for a neat subplot grid."""
    if n <= 1:
        return 1, 1
    if n == 2:
        return 1, 2
    if n <= 3:
        return 1, 3
    if n <= 4:
        return 2, 2
    if n <= 6:
        return 2, 3
    if n <= 8:
        return 2, 4
    if n <= 9:
        return 3, 3
    if n <= 12:
        return 3, 4
    rows = int(np.ceil(n / 4))
    return rows, 4


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print(" Optimization Monitor")
    print("=" * 60)
    print(f"Watching : {HISTORY_DIR.resolve()}")
    print(f"Refresh  : {POLL_INTERVAL:.1f} s")
    print(f"Timeout  : {ACTIVE_TIMEOUT:.0f} s (stale files hidden while active)")
    print("Ctrl+C to exit\n")

    plt.ion()
    fig = None
    axes_pen = []
    axes_aep = []
    run_cache = []
    showing_all = False   # tracks whether we are in "show all" fallback mode

    try:
        while True:
            all_runs = _discover_runs()

            if not all_runs:
                if fig is None:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.set_title("Waiting for runs to appear in history_logs/ ...")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                time.sleep(POLL_INTERVAL)
                continue

            # -----------------------------------------------------------------
            # Filter logic:
            #   If any run is active (recently updated, no END), show only those.
            #   Otherwise show every run (all finished or all stale).
            # -----------------------------------------------------------------
            active_runs = [r for r in all_runs if _is_active(r)]
            if active_runs:
                runs = active_runs
                mode_all = False
            else:
                runs = all_runs
                mode_all = True

            # If we just switched modes, force a rebuild so titles update
            if mode_all != showing_all:
                run_cache = []   # force rebuild
                showing_all = mode_all

            # -----------------------------------------------------------------
            # Rebuild figure if run list changed
            # -----------------------------------------------------------------
            if (len(runs) != len(run_cache) or
                any(r["stem"] != c["stem"] for r, c in zip(runs, run_cache))):

                if fig is not None:
                    plt.close(fig)

                rows, cols = _grid_shape(len(runs))
                fig, axes = plt.subplots(
                    rows, cols,
                    figsize=(5.5 * cols, 4.0 * rows),
                    squeeze=False,
                )
                status = "All runs (nothing active)" if mode_all else "Active runs only"
                fig.suptitle(
                    f"Live Optimization Monitor — {status}",
                    fontsize=14, fontweight="bold",
                )
                fig.tight_layout(rect=[0, 0, 1, 0.96])

                axes_pen.clear()
                axes_aep.clear()
                run_cache = runs[:]

                for idx, run in enumerate(runs):
                    r = idx // cols
                    c = idx % cols
                    ax_pen = axes[r][c]
                    ax_aep = ax_pen.twinx()

                    ax_pen.set_title(run["name"], fontsize=10, fontweight="bold")
                    ax_pen.set_xlabel("Iteration", fontsize=8)
                    ax_pen.set_ylabel("Penalty", color="tab:red", fontsize=9)
                    ax_pen.set_yscale("log")
                    ax_pen.tick_params(axis="y", labelcolor="tab:red", labelsize=7)

                    ax_pen.axhline(
                        PENALTY_THRESHOLD, color="tab:red",
                        linestyle="--", alpha=0.5, linewidth=1,
                        label="threshold",
                    )

                    ax_aep.set_ylabel("AEP (GWh)", color="tab:blue", fontsize=9)
                    ax_aep.tick_params(axis="y", labelcolor="tab:blue", labelsize=7)

                    axes_pen.append(ax_pen)
                    axes_aep.append(ax_aep)

                # Hide unused subplots
                for idx in range(len(runs), rows * cols):
                    r = idx // cols
                    c = idx % cols
                    axes[r][c].axis("off")

            # -----------------------------------------------------------------
            # Update data on existing axes
            # -----------------------------------------------------------------
            for idx, run in enumerate(runs):
                pen, pen_done = _parse_hist_file(run["pen_file"])
                aep, aep_done = _parse_hist_file(run["aep_file"])

                if pen is None or aep is None:
                    continue

                iters = np.arange(len(pen))
                ax_pen = axes_pen[idx]
                ax_aep = axes_aep[idx]

                # Remove old data lines (keep threshold by label)
                for line in list(ax_pen.get_lines()):
                    if line.get_label() != "threshold":
                        line.remove()
                for line in list(ax_aep.get_lines()):
                    line.remove()

                # Plot data if we have any
                if len(pen) > 0:
                    ax_pen.plot(
                        iters, pen, color="tab:red", marker="o",
                        markersize=3, linewidth=1.2, label="Penalty",
                    )
                    ax_aep.plot(
                        iters, aep, color="tab:blue", marker="x",
                        markersize=3, linewidth=1.2, label="AEP",
                    )

                    # Dynamic limits
                    ax_pen.set_xlim(0, max(10, len(pen) + 2))
                    lo = max(1e-12, np.min(pen[pen > 0]) * 0.5) if np.any(pen > 0) else 1e-12
                    hi = np.max(pen) * 2 if np.max(pen) > 0 else 1.0
                    ax_pen.set_ylim(lo, hi)

                    if len(aep) > 0:
                        pad = (np.max(aep) - np.min(aep)) * 0.1 + 1e-3
                        ax_aep.set_ylim(
                            np.min(aep) - pad,
                            np.max(aep) + pad,
                        )

                    # Combined legend
                    lines = []
                    labels = []
                    for line in ax_pen.get_lines():
                        if line.get_label() != "threshold":
                            lines.append(line)
                            labels.append(line.get_label())
                    for line in ax_aep.get_lines():
                        lines.append(line)
                        labels.append(line.get_label())
                    if lines:
                        ax_pen.legend(lines, labels, loc="upper right", fontsize=7)
                else:
                    ax_pen.text(
                        0.5, 0.5, "No data yet",
                        transform=ax_pen.transAxes,
                        ha="center", va="center",
                        fontsize=12, color="lightgray", alpha=0.7,
                    )
                    ax_pen.set_xlim(0, 10)
                    ax_pen.set_ylim(1e-12, 1.0)
                    ax_aep.set_ylim(0, 1)

                # Update title to show status
                done = pen_done or aep_done
                title = run["name"]
                if done:
                    title += "  [DONE]"
                ax_pen.set_title(title, fontsize=10, fontweight="bold",
                                 color="dimgray" if done else "black")

            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\nMonitor stopped by user.")
    finally:
        plt.ioff()
        if fig is not None:
            plt.show(block=True)


if __name__ == "__main__":
    main()