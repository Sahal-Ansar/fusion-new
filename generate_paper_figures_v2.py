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
    """Per-sequence SPEAS / SRC improvement: event-guided vs IMU baseline.

    Four bars per sequence (EG-SPEAS, IMU-SPEAS, EG-SRC, IMU-SRC) so a
    reader of Section V-C can scan the event-guided vs IMU comparison
    for both metrics on a single row. Sorted by EG-SPEAS descending.
    Saturated colors (steelblue, darkorange) mark our event-guided
    results; lighter pair-mates (lightblue, peachpuff) mark the IMU
    baseline — the saturation gradient visually encodes "ours vs base".
    """
    # seq, EG-SPEAS%, IMU-SPEAS%, EG-SRC%, IMU-SRC%
    data = [
        ("0001", +0.03, -1.78, +0.25, -0.69),
        ("0002", +0.93, -2.34, +0.24, -1.16),
        ("0005", +0.81, -1.31, +1.27, -0.90),
        ("0009", +0.89, -1.56, +0.80, -1.33),
        ("0011", +0.30, -1.00, +0.48, -0.57),
        ("0013", -0.00, -2.56, +0.17, -1.63),
        ("0014", -0.01, -2.73, +0.24, -2.58),
        ("0017", +0.09, -0.00, +0.05, +0.00),
        ("0018", -0.03, -0.35, +0.23, -0.24),
        ("0048", -0.01, -2.16, +0.15, -1.62),
        ("0051", +0.34, -1.33, +0.16, -1.15),
        ("0056", +0.32, -2.66, -0.22, -1.97),
        ("0057", +0.25, -0.27, +0.04, -0.23),
        ("0059", +0.62, -1.17, +0.50, -0.90),
        ("0060", -0.07, +0.02, -0.02, +0.00),
        ("0084", +0.72, -1.30, -0.04, -0.89),
        ("0091", +1.69, -1.35, +0.67, -0.40),
        ("0093", +1.72, -0.89, +1.46, -0.76),
        ("0095", +0.90, -2.21, +0.38, -1.65),
        ("0096", +0.73, -1.79, +0.52, -1.31),
        ("0104", +0.73, -1.88, -0.21, -1.36),
        ("0106", +1.21, -1.29, +0.58, -0.62),
        ("0113", +2.08, -0.37, +0.57, -0.16),
        ("0117", +1.17, -1.11, +0.67, -0.68),
    ]
    data = sorted(data, key=lambda r: r[1], reverse=True)

    seqs = [d[0] for d in data]
    eg_speas = np.array([d[1] for d in data])
    imu_speas = np.array([d[2] for d in data])
    eg_src = np.array([d[3] for d in data])
    imu_src = np.array([d[4] for d in data])

    # Cross-dataset means (provided — match Table II row aggregates).
    eg_speas_mean = +0.64
    imu_speas_mean = -1.39
    eg_src_mean = +0.37
    imu_src_mean = -0.95

    fig, ax = plt.subplots(figsize=(IEEE_COLUMN_WIDTH, 8.5))

    # Group spacing 1.6 > bar block 0.88 → ~45% whitespace between groups
    # so the eye reads each sequence as a distinct block.
    group_centers = np.arange(len(seqs)) * 1.6
    bar_h = 0.22
    intra_offsets = np.array([-1.5, -0.5, +0.5, +1.5]) * bar_h

    ax.barh(group_centers + intra_offsets[0], eg_speas,  bar_h,
            color="steelblue",  linewidth=0.0, label="EG-SPEAS (ours)", zorder=2)
    ax.barh(group_centers + intra_offsets[1], imu_speas, bar_h,
            color="lightblue",  linewidth=0.0, label="IMU-SPEAS", zorder=2)
    ax.barh(group_centers + intra_offsets[2], eg_src,    bar_h,
            color="darkorange", linewidth=0.0, label="EG-SRC (ours)", zorder=2)
    ax.barh(group_centers + intra_offsets[3], imu_src,   bar_h,
            color="peachpuff",  linewidth=0.0, label="IMU-SRC", zorder=2)

    # Faint separators midway between adjacent group centers.
    for c in group_centers[:-1]:
        ax.axhline(c + 0.8, color="gray", linewidth=0.3, alpha=0.25, zorder=0)

    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6, zorder=1)
    ax.axvline(eg_speas_mean,  color="steelblue",
               linestyle=":", linewidth=1.1, alpha=0.95, zorder=3)
    ax.axvline(imu_speas_mean, color="steelblue",
               linestyle=":", linewidth=1.1, alpha=0.55, zorder=3)
    ax.axvline(eg_src_mean,    color="darkorange",
               linestyle=":", linewidth=1.1, alpha=0.95, zorder=3)
    ax.axvline(imu_src_mean,   color="darkorange",
               linestyle=":", linewidth=1.1, alpha=0.55, zorder=3)

    # Annotate the four means at the top of the plot so the dotted lines
    # are decodable without bloating the legend to 8 entries.
    ymin = group_centers.min() - 2.0
    for xv, txt, col in [
        (eg_speas_mean,  f"EG μ={eg_speas_mean:+.2f}",  "steelblue"),
        (imu_speas_mean, f"IMU μ={imu_speas_mean:+.2f}", "steelblue"),
        (eg_src_mean,    f"EG μ={eg_src_mean:+.2f}",    "darkorange"),
        (imu_src_mean,   f"IMU μ={imu_src_mean:+.2f}",  "darkorange"),
    ]:
        ax.text(xv, ymin, txt, color=col, fontsize=5.5,
                ha="center", va="bottom", rotation=90, alpha=0.9)

    ax.set_yticks(group_centers)
    ax.set_yticklabels(seqs, fontsize=7)
    ax.invert_yaxis()           # highest EG-SPEAS at the top
    ax.set_xlabel("Improvement (%)")
    ax.set_ylabel("Sequence")
    ax.legend(loc="lower right", fontsize=6, framealpha=0.9, ncol=2,
              handlelength=1.4, columnspacing=0.8, handletextpad=0.4)
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
