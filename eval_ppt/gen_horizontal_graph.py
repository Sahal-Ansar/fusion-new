"""
Re-render the per-sequence SPEAS / SRC improvement chart with sequences along
the X-axis and bars vertical, sized for a 4:3 slide.

Output: ./horizontal_graph.png
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "horizontal_graph.png"

# (seq, EG-SPEAS%, IMU-SPEAS%, EG-SRC%, IMU-SRC%) — same data as the paper.
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

seqs      = [d[0] for d in data]
eg_speas  = np.array([d[1] for d in data])
imu_speas = np.array([d[2] for d in data])
eg_src    = np.array([d[3] for d in data])
imu_src   = np.array([d[4] for d in data])

eg_speas_mean, imu_speas_mean = +0.64, -1.39
eg_src_mean,   imu_src_mean   = +0.37, -0.95

# 4:3 canvas (12 x 9 in). Generous so the 24 x 4 bars stay readable.
fig, ax = plt.subplots(figsize=(12, 9))

group_centers = np.arange(len(seqs)) * 1.6
bar_w = 0.28
intra = np.array([-1.5, -0.5, +0.5, +1.5]) * bar_w

ax.bar(group_centers + intra[0], eg_speas,  bar_w,
       color="steelblue",  linewidth=0.0, label="EG-SPEAS (ours)", zorder=2)
ax.bar(group_centers + intra[1], imu_speas, bar_w,
       color="lightblue",  linewidth=0.0, label="IMU-SPEAS",       zorder=2)
ax.bar(group_centers + intra[2], eg_src,    bar_w,
       color="darkorange", linewidth=0.0, label="EG-SRC (ours)",   zorder=2)
ax.bar(group_centers + intra[3], imu_src,   bar_w,
       color="peachpuff",  linewidth=0.0, label="IMU-SRC",         zorder=2)

# Faint vertical separators between groups.
for c in group_centers[:-1]:
    ax.axvline(c + 0.8, color="gray", linewidth=0.3, alpha=0.25, zorder=0)

# Reference lines (now horizontal)
ax.axhline(0.0, color="black", linestyle="--", linewidth=0.9, alpha=0.6, zorder=1)
ax.axhline(eg_speas_mean,  color="steelblue",  linestyle=":", linewidth=1.3, alpha=0.95, zorder=3)
ax.axhline(imu_speas_mean, color="steelblue",  linestyle=":", linewidth=1.3, alpha=0.55, zorder=3)
ax.axhline(eg_src_mean,    color="darkorange", linestyle=":", linewidth=1.3, alpha=0.95, zorder=3)
ax.axhline(imu_src_mean,   color="darkorange", linestyle=":", linewidth=1.3, alpha=0.55, zorder=3)

# Mean labels at the right edge so the dotted lines are decodable.
xmax = group_centers.max() + 1.2
for yv, txt, col in [
    (eg_speas_mean,  f"EG μ={eg_speas_mean:+.2f}",  "steelblue"),
    (imu_speas_mean, f"IMU μ={imu_speas_mean:+.2f}", "steelblue"),
    (eg_src_mean,    f"EG μ={eg_src_mean:+.2f}",    "darkorange"),
    (imu_src_mean,   f"IMU μ={imu_src_mean:+.2f}",  "darkorange"),
]:
    ax.text(xmax, yv, " " + txt, color=col, fontsize=9,
            ha="left", va="center", alpha=0.95)

ax.set_xticks(group_centers)
ax.set_xticklabels(seqs, fontsize=10, rotation=45, ha="right")
ax.set_xlabel("Sequence (sorted by EG-SPEAS, descending)", fontsize=11)
ax.set_ylabel("Improvement (%)", fontsize=11)
ax.legend(loc="lower left", fontsize=10, framealpha=0.92, ncol=2,
          handlelength=1.6, columnspacing=1.2, handletextpad=0.5)
ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.45)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Headroom on the right so the mean labels don't get clipped.
ax.set_xlim(group_centers.min() - 1.0, xmax + 2.5)

fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"wrote {OUT}")
