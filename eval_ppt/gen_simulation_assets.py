"""
Generates simulation / visualization images used in the eval PPT.

Outputs (in ./assets/):
    sweep_diagram.png       - synthesized LiDAR sweep showing temporal distortion
    flow_viz.png            - dense optical flow visualization (HSV-encoded)
    event_map_demo.png      - simulated event mask overlay on a real KITTI frame
    pipeline_panels.png     - 4-panel view: RGB / events / flow / corrected
    before_after_zoom.png   - cropped close-up of uncorrected vs corrected projection
    misalignment_arrow.png  - annotated zoom on uncorrected projection w/ displacement vectors
    first_eval_recap.png    - composite from the first-evaluation pipeline (raw outputs)

All images are 16:9-friendly with white backgrounds and limited Google palette.
Run from inside eval_ppt/.
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Wedge, Circle, Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D

# ---- Make the project root importable so we can reuse its modules ----
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import calibration as cal      # noqa: E402
import loader                  # noqa: E402
import projection as proj      # noqa: E402
import events as ev            # noqa: E402
import flow as fl              # noqa: E402

ASSETS = HERE / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

KITTI_ROOT = Path(r"c:/Users/sahaa/OneDrive/Desktop/Honors/datasets/fusion")
DEFAULT_SEQ = "2011_09_26_drive_0009_sync"

# ---- Google Material palette ----
G_BLUE   = "#1A73E8"
G_RED    = "#EA4335"
G_YELLOW = "#FBBC04"
G_GREEN  = "#34A853"
G_GREY   = "#5F6368"
G_DARK   = "#202124"
G_LIGHT  = "#F1F3F4"


def _save(fig, name, dpi=180):
    out = ASSETS / name
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out.name}")


# =========================================================================
# 1. Synthesized LiDAR sweep diagram (educational illustration)
# =========================================================================
def make_sweep_diagram():
    """Top-down diagram showing a rotating LiDAR sweeping over a scene with a
    moving car, illustrating temporal non-uniformity of the resulting scan."""
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(-7, 11)
    ax.set_ylim(-4.5, 4.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Title
    ax.text(1, 4.0, "Rotating LiDAR: one scan = many timestamps",
            fontsize=15, fontweight="bold", color=G_DARK, ha="center")

    # Ego vehicle (LiDAR mounted on top)
    ego = FancyBboxPatch((-0.9, -0.55), 1.8, 1.1, boxstyle="round,pad=0.05,rounding_size=0.18",
                         facecolor=G_BLUE, edgecolor="none")
    ax.add_patch(ego)
    ax.add_patch(Circle((0, 0), 0.18, facecolor=G_DARK, edgecolor="white", lw=1.5, zorder=5))
    ax.text(0, -1.05, "Ego vehicle", ha="center", fontsize=9, color=G_GREY)

    # Sweep wedges, colored by acquisition time (early -> late)
    sweep_cmap = plt.get_cmap("plasma")
    n_wedges = 8
    angles = np.linspace(0, 360, n_wedges + 1)
    for i in range(n_wedges):
        w = Wedge((0, 0), 3.6, angles[i], angles[i + 1],
                  width=3.4, facecolor=sweep_cmap(i / (n_wedges - 1)),
                  alpha=0.18, edgecolor="white", lw=0.8)
        ax.add_patch(w)

    # Sweep arrow indicating rotation
    sweep_arrow = mpatches.FancyArrowPatch(
        (2.8, 1.6), (1.6, 2.8),
        connectionstyle="arc3,rad=0.4", color=G_DARK, lw=1.8,
        arrowstyle="-|>,head_length=8,head_width=6")
    ax.add_patch(sweep_arrow)
    ax.text(2.9, 2.6, "10 Hz\nrotation", fontsize=9, color=G_DARK)

    # Moving car on the right that drifts during the sweep
    # Show two ghosted positions plus a "true" position
    car_y = -2.2
    car_positions = [(4.0, 0.30, "t = 0 ms"),
                     (6.0, 0.55, "t = 50 ms"),
                     (8.0, 1.00, "t = 100 ms")]
    for cx, alpha, label in car_positions:
        c = FancyBboxPatch((cx - 0.65, car_y - 0.32), 1.3, 0.64,
                           boxstyle="round,pad=0.02,rounding_size=0.14",
                           facecolor=G_RED, edgecolor="none", alpha=alpha)
        ax.add_patch(c)
        ax.text(cx, car_y - 0.80, label, ha="center", fontsize=9, color=G_GREY)
    ax.annotate("", xy=(8.7, car_y + 0.95), xytext=(3.3, car_y + 0.95),
                arrowprops=dict(arrowstyle="-|>", color=G_RED, lw=1.4))
    ax.text(6.0, car_y + 1.30, "moves ~1.67 m during one sweep",
            ha="center", fontsize=10, color=G_RED, fontweight="bold")

    # Beams sampling the moving car at three different times
    for tgt_x, tgt_y, c in [(4.0, car_y, sweep_cmap(0.05)),
                            (6.0, car_y, sweep_cmap(0.5)),
                            (8.0, car_y, sweep_cmap(0.95))]:
        ax.plot([0, tgt_x], [0, tgt_y], color=c, lw=1.4, alpha=0.85)

    # Legend strip (time gradient)
    bar_x = -6.3
    bar_y = -3.6
    bar_w = 4.0
    bar_h = 0.22
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(grad, extent=(bar_x, bar_x + bar_w, bar_y, bar_y + bar_h),
              aspect="auto", cmap="plasma", zorder=3)
    ax.text(bar_x, bar_y + 0.45, "Acquisition time within one scan",
            fontsize=9, color=G_DARK)
    ax.text(bar_x, bar_y - 0.25, "early", fontsize=8, color=G_GREY)
    ax.text(bar_x + bar_w, bar_y - 0.25, "late", fontsize=8, color=G_GREY, ha="right")

    # Annotation callout
    ax.text(-6.2, 3.0,
            "Each beam fires at a\ndifferent instant.\nDuring 100 ms the world\ndoesn't sit still.",
            fontsize=10, color=G_DARK, va="top")

    _save(fig, "sweep_diagram.png")


# =========================================================================
# Helpers for KITTI loading
# =========================================================================
def _load_pair(seq=DEFAULT_SEQ, idx=40):
    seq_dir = KITTI_ROOT / seq
    img_dir = seq_dir / "image_02" / "data"
    lid_dir = seq_dir / "velodyne_points" / "data"

    img_files = sorted(img_dir.glob("*.png"))
    lid_files = sorted(lid_dir.glob("*.bin"))

    idx = min(idx, len(img_files) - 2)

    img_t  = cv2.imread(str(img_files[idx]),     cv2.IMREAD_COLOR)
    img_t1 = cv2.imread(str(img_files[idx + 1]), cv2.IMREAD_COLOR)
    lid_t  = np.fromfile(str(lid_files[idx]), dtype=np.float32).reshape(-1, 4)[:, :3]

    Tr = cal.parse_calib_velo_to_cam(str(seq_dir / "calib_velo_to_cam.txt"))
    Rrect, Prect = cal.parse_calib_cam_to_cam(str(seq_dir / "calib_cam_to_cam.txt"), camera_id="02")

    return img_t, img_t1, lid_t, Tr, Rrect, Prect


def _flow_to_rgb(flow):
    """HSV-encoded dense flow visualization."""
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = np.uint8(ang * 180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# =========================================================================
# 2. Optical flow visualization
# =========================================================================
def make_flow_viz():
    img_t, img_t1, _, _, _, _ = _load_pair(DEFAULT_SEQ, 60)
    flow = fl.compute_rgb_flow(img_t, img_t1)
    flow_rgb = _flow_to_rgb(flow)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].imshow(cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB))
    axes[0].set_title("RGB frame  $I_t$", fontsize=13, color=G_DARK, pad=8)
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(flow_rgb, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Dense optical flow  $F(x,y)$  (hue = direction, value = magnitude)",
                      fontsize=13, color=G_DARK, pad=8)
    axes[1].axis("off")

    fig.patch.set_facecolor("white")
    fig.tight_layout()
    _save(fig, "flow_viz.png")


# =========================================================================
# 3. Event map demo (simulated DVS events overlaid on a real frame)
# =========================================================================
def make_event_demo():
    img_t, img_t1, _, _, _, _ = _load_pair(DEFAULT_SEQ, 60)
    events_arr = ev.simulate_events(img_t, img_t1, threshold=0.3)

    H, W = events_arr.shape
    overlay = cv2.cvtColor(img_t1, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay *= 0.45  # darken background
    pos = (events_arr > 0)
    neg = (events_arr < 0)
    overlay[pos] = (1.0, 1.0, 1.0)
    overlay[neg] = (0.10, 0.55, 0.95)
    overlay = np.clip(overlay, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].imshow(cv2.cvtColor(img_t1, cv2.COLOR_BGR2RGB))
    axes[0].set_title("RGB frame  $I_{t+1}$", fontsize=13, color=G_DARK, pad=8)
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("Simulated event map  (white = ON, blue = OFF)",
                      fontsize=13, color=G_DARK, pad=8)
    axes[1].axis("off")

    fig.patch.set_facecolor("white")
    fig.tight_layout()
    _save(fig, "event_map_demo.png")


# =========================================================================
# 4. 4-panel pipeline view
# =========================================================================
def make_pipeline_panels():
    img_t, img_t1, lid, Tr, Rrect, Prect = _load_pair(DEFAULT_SEQ, 40)
    uv, depth, _ = proj.project_lidar_to_image(lid, Tr, Rrect, Prect, img_t.shape)

    # Panel 1: raw projection on t
    panel1 = proj.overlay_points(img_t.copy(), uv, depth, max_points=10000, radius=1)

    # Panel 2: events
    events_arr = ev.simulate_events(img_t, img_t1, threshold=0.3)
    ev_img = ev.events_to_image(events_arr)

    # Panel 3: flow
    flow = fl.compute_rgb_flow(img_t, img_t1)
    flow_rgb = _flow_to_rgb(flow)

    # Panel 4: corrected projection on t+1
    import lidar_motion as lm
    conf = ev.event_confidence(events_arr)
    uv_corr, depth_corr = lm.move_lidar_points_weighted(uv, depth, flow, conf, conf_thresh=0.2)
    panel4 = proj.overlay_points(img_t1.copy(), uv_corr, depth_corr, max_points=10000, radius=1)

    titles = [
        "1. Uncorrected projection on $I_t$",
        "2. Simulated event mask",
        "3. Dense optical flow",
        "4. Corrected projection on $I_{t+1}$",
    ]
    images = [panel1, ev_img, flow_rgb, panel4]

    fig, axes = plt.subplots(2, 2, figsize=(13, 7.6))
    for ax, im, t in zip(axes.ravel(), images, titles):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(t, fontsize=12, color=G_DARK, pad=6)
        ax.axis("off")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    _save(fig, "pipeline_panels.png")


# =========================================================================
# 5. Before/after zoomed comparison
# =========================================================================
def make_before_after_zoom():
    img_t, img_t1, lid, Tr, Rrect, Prect = _load_pair(DEFAULT_SEQ, 40)
    uv, depth, _ = proj.project_lidar_to_image(lid, Tr, Rrect, Prect, img_t.shape)

    flow = fl.compute_rgb_flow(img_t, img_t1)
    events_arr = ev.simulate_events(img_t, img_t1, threshold=0.3)
    conf = ev.event_confidence(events_arr)
    import lidar_motion as lm
    uv_corr, depth_corr = lm.move_lidar_points_weighted(uv, depth, flow, conf, conf_thresh=0.2)

    before = proj.overlay_points(img_t1.copy(), uv,      depth,      max_points=12000, radius=2)
    after  = proj.overlay_points(img_t1.copy(), uv_corr, depth_corr, max_points=12000, radius=2)

    # crop a content-rich region around the centre-bottom
    H, W = img_t.shape[:2]
    x0, y0, w, h = int(W * 0.32), int(H * 0.42), int(W * 0.36), int(H * 0.52)
    before_c = before[y0:y0 + h, x0:x0 + w]
    after_c  = after[y0:y0 + h, x0:x0 + w]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    axes[0].imshow(cv2.cvtColor(before_c, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Before correction", fontsize=14, color=G_RED, pad=8, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(after_c, cv2.COLOR_BGR2RGB))
    axes[1].set_title("After event-guided correction", fontsize=14, color=G_GREEN,
                      pad=8, fontweight="bold")
    axes[1].axis("off")

    fig.patch.set_facecolor("white")
    fig.tight_layout()
    _save(fig, "before_after_zoom.png")


# =========================================================================
# 6. Misalignment arrows (annotated uncorrected projection close-up)
# =========================================================================
def make_misalignment_arrows():
    img_t, img_t1, lid, Tr, Rrect, Prect = _load_pair(DEFAULT_SEQ, 40)
    uv, depth, _ = proj.project_lidar_to_image(lid, Tr, Rrect, Prect, img_t.shape)
    flow = fl.compute_rgb_flow(img_t, img_t1)

    H, W = img_t.shape[:2]
    x0, y0, w, h = int(W * 0.33), int(H * 0.40), int(W * 0.34), int(H * 0.55)
    crop = cv2.cvtColor(img_t1[y0:y0 + h, x0:x0 + w], cv2.COLOR_BGR2RGB)

    # filter points inside the crop
    uvi = uv.astype(int)
    sel = (uvi[:, 0] >= x0) & (uvi[:, 0] < x0 + w) & (uvi[:, 1] >= y0) & (uvi[:, 1] < y0 + h)
    uvi = uvi[sel]
    depth_sel = depth[sel]

    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.imshow(crop)
    ax.axis("off")
    ax.set_title("Uncorrected LiDAR projection on $I_{t+1}$  +  flow displacement",
                 fontsize=13, color=G_DARK, pad=8)

    rng = np.random.default_rng(0)
    if len(uvi) > 0:
        sample = rng.choice(len(uvi), size=min(45, len(uvi)), replace=False)
        for i in sample:
            u, v = uvi[i]
            du, dv = flow[v, u]
            if abs(du) + abs(dv) < 0.5 or abs(du) + abs(dv) > 30:
                continue
            ax.plot(u - x0, v - y0, "o", markersize=3.5, color=G_RED,
                    markeredgecolor="white", markeredgewidth=0.5)
            ax.annotate("", xy=(u - x0 + du, v - y0 + dv), xytext=(u - x0, v - y0),
                        arrowprops=dict(arrowstyle="->", color=G_YELLOW, lw=1.2, alpha=0.95))

    legend_lines = [Line2D([0], [0], marker="o", color="w", markerfacecolor=G_RED,
                           markersize=8, label="Uncorrected LiDAR point"),
                    Line2D([0], [0], color=G_YELLOW, lw=2, label="Where flow says it should go")]
    ax.legend(handles=legend_lines, loc="lower right", fontsize=9, framealpha=0.92)

    fig.patch.set_facecolor("white")
    fig.tight_layout()
    _save(fig, "misalignment_arrow.png")


# =========================================================================
# 7. First-evaluation recap composite
# =========================================================================
def make_first_eval_recap():
    """A composite mimicking what the *first evaluation* delivered: the basic
    pipeline visualization (projection -> events -> flow -> corrected)."""
    img_t, img_t1, lid, Tr, Rrect, Prect = _load_pair(DEFAULT_SEQ, 40)
    uv, depth, _ = proj.project_lidar_to_image(lid, Tr, Rrect, Prect, img_t.shape)

    panel_proj = proj.overlay_points(img_t.copy(), uv, depth, max_points=8000, radius=1)
    events_arr = ev.simulate_events(img_t, img_t1, threshold=0.3)
    ev_img = ev.events_to_image(events_arr)

    flow = fl.compute_rgb_flow(img_t, img_t1)
    flow_rgb = _flow_to_rgb(flow)

    import lidar_motion as lm
    conf = ev.event_confidence(events_arr)
    uv_corr, depth_corr = lm.move_lidar_points_weighted(uv, depth, flow, conf, conf_thresh=0.2)
    panel_corr = proj.overlay_points(img_t1.copy(), uv_corr, depth_corr, max_points=8000, radius=1)

    fig = plt.figure(figsize=(13, 5.4))
    gs = fig.add_gridspec(2, 2, hspace=0.18, wspace=0.04)
    panels = [
        (panel_proj, "Stage 1  -  LiDAR projection"),
        (ev_img,     "Stage 2  -  Simulated events"),
        (flow_rgb,   "Stage 3  -  Optical flow"),
        (panel_corr, "Stage 4  -  Event-gated correction"),
    ]
    for k, (im, t) in enumerate(panels):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(t, fontsize=11, color=G_DARK, pad=4)
        ax.axis("off")
    fig.suptitle("First-evaluation pipeline (classical, CPU-only)",
                 fontsize=13.5, color=G_DARK, fontweight="bold", y=1.02)
    fig.patch.set_facecolor("white")
    _save(fig, "first_eval_recap.png")


# =========================================================================
# Run everything
# =========================================================================
def main():
    print("[gen_simulation_assets] generating images...")
    make_sweep_diagram()
    make_flow_viz()
    make_event_demo()
    make_pipeline_panels()
    make_before_after_zoom()
    make_misalignment_arrows()
    make_first_eval_recap()
    print("[gen_simulation_assets] done.")


if __name__ == "__main__":
    main()
