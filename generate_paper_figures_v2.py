"""
generate_paper_figures_v2.py — IEEE-style figures for the paper.

Produces four high-resolution (dpi=300) PNGs in ./paper_figures/:
  fig_speas_src_improvement.png  — per-sequence SPEAS / SRC bars
  fig_ablation.png               — ablation grouped bars
  fig_bdps_improvement.png       — BDPS per-sequence bars
  fig_runtime_breakdown.png      — per-stage runtime bars
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = "./paper_figures"
IEEE_COLUMN_WIDTH = 3.5  # inches (single IEEE column)

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 9
plt.rcParams["axes.linewidth"] = 0.8


# =======================================================================
# Helpers
# =======================================================================

def _despine(ax) -> None:
    """Hide top and right spines for the IEEE clean look."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig, filename: str) -> str:
    """Save figure as 300-dpi PNG in OUTPUT_DIR with tight layout."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


# =======================================================================
# Figure 1 — SPEAS and SRC per sequence (horizontal grouped bars)
# =======================================================================

def figure_1_speas_src_improvement() -> str:
    """Per-sequence SPEAS and SRC improvement, sorted by SPEAS desc."""
    data = [
        ("0001", +0.03, +0.25), ("0002", +0.93, +0.24),
        ("0005", +0.81, +1.27), ("0009", +0.89, +0.80),
        ("0011", +0.30, +0.48), ("0013", -0.00, +0.17),
        ("0014", -0.01, +0.24), ("0017", +0.09, +0.05),
        ("0018", -0.03, +0.23), ("0048", -0.01, +0.15),
        ("0051", +0.34, +0.16), ("0056", +0.32, -0.22),
        ("0057", +0.25, +0.04), ("0059", +0.62, +0.50),
        ("0060", -0.07, -0.02), ("0084", +0.72, -0.04),
        ("0091", +1.69, +0.67), ("0093", +1.72, +1.46),
        ("0095", +0.90, +0.38), ("0096", +0.73, +0.53),
        ("0104", +0.73, -0.21), ("0106", +1.21, +0.58),
        ("0113", +2.08, +0.57), ("0117", +1.17, +0.67),
    ]
    data = sorted(data, key=lambda r: r[1], reverse=True)

    seqs = [d[0] for d in data]
    speas = np.array([d[1] for d in data])
    src = np.array([d[2] for d in data])

    speas_mean = float(np.mean(speas))
    src_mean = float(np.mean(src))

    fig, ax = plt.subplots(figsize=(IEEE_COLUMN_WIDTH, 5.5))

    y = np.arange(len(seqs))
    h = 0.4
    ax.barh(y - h / 2, speas, h, color="steelblue", label="SPEAS", zorder=2)
    ax.barh(y + h / 2, src, h, color="darkorange", label="SRC", zorder=2)

    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6, zorder=1)
    ax.axvline(
        speas_mean, color="steelblue", linestyle=":", linewidth=1.0, alpha=0.85,
        label=f"SPEAS mean ({speas_mean:+.2f}%)", zorder=3,
    )
    ax.axvline(
        src_mean, color="darkorange", linestyle=":", linewidth=1.0, alpha=0.85,
        label=f"SRC mean ({src_mean:+.2f}%)", zorder=3,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(seqs, fontsize=7)
    ax.invert_yaxis()           # highest SPEAS at the top
    ax.set_xlabel("Improvement (%)")
    ax.set_ylabel("Sequence")
    ax.legend(loc="upper right", fontsize=6, framealpha=0.9)
    ax.grid(axis="x", linestyle=":", linewidth=0.4, alpha=0.4)
    _despine(ax)

    fig.tight_layout()
    return _save(fig, "fig_speas_src_improvement.png")


# =======================================================================
# Figure 2 — Ablation grouped bars (vertical)
# =======================================================================

def figure_2_ablation() -> str:
    """Vertical grouped bars: SPEAS / SRC per ablation config."""
    # Single-line labels — rotated x-ticks render multi-line labels with
    # the second lines of adjacent labels colliding, so the spec's
    # multi-line forms are joined into a single line for readability.
    configs = [
        ("Baseline",       0.00,  0.00),
        ("Flow Only",     +0.29, -0.18),
        ("No Smooth",     +0.47, +0.19),
        ("No Edge Filter", +0.63, +0.29),
        ("No Bounds",     +0.35, +0.36),
        ("α=0.25",        +0.52, +0.30),
        ("α=0.75",        -0.07, -0.20),
        ("Full (Ours)",   +0.45, +0.28),
    ]
    labels = [c[0] for c in configs]
    speas = np.array([c[1] for c in configs])
    src = np.array([c[2] for c in configs])

    x = np.arange(len(labels))
    w = 0.38

    fig, ax = plt.subplots(figsize=(IEEE_COLUMN_WIDTH, 3.0))

    speas_colors = ["steelblue"] * len(labels)
    src_colors = ["darkorange"] * len(labels)
    speas_colors[-1] = "#0b3d91"   # darker navy
    src_colors[-1] = "#a83800"     # darker orange/red

    bars_speas = ax.bar(x - w / 2, speas, w, color=speas_colors, label="SPEAS Δ%", zorder=2)
    bars_src = ax.bar(x + w / 2, src, w, color=src_colors, label="SRC Δ%", zorder=2)

    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6, zorder=1)

    # Stagger SPEAS / SRC labels vertically so adjacent in-group labels
    # at similar heights (e.g. "+0.35" / "+0.36" on No Bounds) don't pile up.
    def _label(bars, values, near_offset, far_offset):
        for bar, v in zip(bars, values):
            if v >= 0:
                y = v + near_offset
                va = "bottom"
            else:
                y = v - near_offset
                va = "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                f"{v:+.2f}",
                ha="center", va=va, fontsize=5.5,
            )
            _ = far_offset  # reserved if we ever want a second tier

    _label(bars_speas, speas, near_offset=0.04, far_offset=0.10)
    _label(bars_src, src, near_offset=0.11, far_offset=0.04)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=20, ha="right")
    ax.set_ylabel("Δ% improvement")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.4)

    # Vertical headroom for the staggered value labels.
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min - 0.10, y_max + 0.25)
    _despine(ax)

    fig.tight_layout()
    return _save(fig, "fig_ablation.png")


# =======================================================================
# Figure 3 — BDPS per-sequence bar chart
# =======================================================================

def figure_3_bdps_improvement() -> str:
    """BDPS improvement per sequence with a mean reference line."""
    seqs = ["0005", "0009", "0051", "0117"]
    bdps = np.array([+59.6, +46.8, +41.5, +46.2])
    mean_value = +48.5

    fig, ax = plt.subplots(figsize=(IEEE_COLUMN_WIDTH, 2.5))

    bars = ax.bar(seqs, bdps, color="steelblue", zorder=2)
    ax.axhline(
        mean_value, color="black", linestyle="--", linewidth=0.8,
        label=f"Mean ({mean_value:+.1f}%)", zorder=1,
    )

    # Push value labels above the mean line for sub-mean bars so the
    # dashed mean line never crosses the text. Bars above the mean keep
    # their label just above the bar top.
    for bar, v in zip(bars, bdps):
        anchor = max(v, mean_value)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            anchor + 1.2,
            f"{v:+.1f}%",
            ha="center", va="bottom", fontsize=7, zorder=4,
        )

    ax.set_xlabel("Sequence")
    ax.set_ylabel("BDPS Improvement (%)")
    ax.set_ylim(0.0, max(bdps.max(), mean_value) * 1.20)
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.4)
    _despine(ax)

    fig.tight_layout()
    return _save(fig, "fig_bdps_improvement.png")


# =======================================================================
# Figure 4 — Per-stage runtime breakdown
# =======================================================================

def figure_4_runtime_breakdown() -> str:
    """Horizontal bars of per-stage runtime, sorted descending."""
    stages = [
        ("Farneback Flow",      99.2),
        ("Event Simulation",    13.7),
        ("Temporal Smoothing",   7.0),
        ("Motion Compensation",  2.8),
        ("Event Confidence",     0.4),
    ]
    stages = sorted(stages, key=lambda s: s[1], reverse=True)
    names = [s[0] for s in stages]
    vals = np.array([s[1] for s in stages])

    # Independently-measured pipeline total (does not equal the sum of
    # stage means due to per-stage measurement variance — see paper).
    TOTAL_MS = 114.5
    FPS_HZ = 8.7

    fig, ax = plt.subplots(figsize=(IEEE_COLUMN_WIDTH, 2.5))

    y = np.arange(len(names))
    bars = ax.barh(y, vals, color="steelblue", zorder=2)

    for bar, v in zip(bars, vals):
        pct = (v / TOTAL_MS) * 100.0
        ax.text(
            v + max(vals) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.1f} ms ({pct:.1f}%)",
            va="center", fontsize=7,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Time (ms)")
    ax.set_xlim(0.0, max(vals) * 1.45)
    ax.grid(axis="x", linestyle=":", linewidth=0.4, alpha=0.4)

    # Place the total ABOVE the axes as a subtitle so it never overlaps
    # the small-bar labels (Event Confidence, 0.4 ms) at the bottom right.
    ax.set_title(
        f"Total: {TOTAL_MS:.1f} ms  ({FPS_HZ:.1f} Hz)",
        fontsize=9, pad=4,
    )

    _despine(ax)

    fig.tight_layout()
    return _save(fig, "fig_runtime_breakdown.png")


# =======================================================================
# Entry point
# =======================================================================

def main() -> None:
    # Best-effort UTF-8 stdout (cp1252 Windows consoles otherwise crash on α / Δ).
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    out = Path(OUTPUT_DIR).resolve()
    print(f"Generating IEEE figures into: {out}")

    figure_1_speas_src_improvement()
    figure_2_ablation()
    figure_3_bdps_improvement()
    figure_4_runtime_breakdown()

    print("\nAll figures saved.")


if __name__ == "__main__":
    main()
