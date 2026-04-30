"""
Publication-quality figure generator for the paper:
  "Event-Guided Motion Compensation for LiDAR Temporal Distortion
   in Autonomous Driving"

Runs end-to-end with no user interaction. All figures are written to
<project_root>/paper_figures_final/.
"""

import os
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# Local source modules (do NOT modify these).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from loader import list_frame_files, load_image, load_lidar                      # noqa: E402
from calibration import parse_calib_velo_to_cam, parse_calib_cam_to_cam          # noqa: E402
from projection import project_lidar_to_image                                    # noqa: E402
from events import simulate_events, event_confidence                             # noqa: E402
from flow import compute_rgb_flow                                                # noqa: E402
from lidar_motion import move_lidar_points_weighted                              # noqa: E402


# ----------------------------- Paths ---------------------------------------- #

PROJECT_ROOT = r"C:\Users\sahaa\OneDrive\Desktop\Honors\fusion-revised1"
DATASETS_ROOT = r"C:\Users\sahaa\OneDrive\Desktop\Honors\datasets\fusion"

DATASET_0009 = os.path.join(DATASETS_ROOT, "2011_09_26_drive_0009_sync")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "paper_figures_final")


# --------------------------- Helpers ---------------------------------------- #

def _report_saved(path):
    """Print file path and size in KB after saving."""
    size_kb = os.path.getsize(path) / 1024.0
    print(f"  saved: {path}  ({size_kb:.1f} KB)")


def _jet_colors_shared(depth, d_min, d_max):
    """Return per-point BGR colors (uint8, Nx3) from depth via JET, using a shared range."""
    depth = np.asarray(depth, dtype=np.float32)
    if d_max - d_min < 1e-12:
        normed = np.zeros_like(depth, dtype=np.uint8)
    else:
        normed = np.clip((depth - d_min) / (d_max - d_min) * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(normed, cv2.COLORMAP_JET).reshape(-1, 3)


def _depth_colors(depth):
    """Return per-point BGR colors (uint8, Nx3) from depth via JET colormap (local range)."""
    depth = np.asarray(depth, dtype=np.float32)
    if depth.size == 0:
        return np.empty((0, 3), dtype=np.uint8)
    d_min = float(np.min(depth))
    d_max = float(np.max(depth))
    return _jet_colors_shared(depth, d_min, d_max)


def _draw_points_depth_sorted(canvas, uv, depth, radius=2):
    """Draw JET-colored points sorted far-to-near (near on top), local depth range."""
    if uv.shape[0] == 0:
        return canvas
    order = np.argsort(depth)[::-1]           # far -> near
    uv_s = uv[order]
    depth_s = depth[order]
    colors = _depth_colors(depth_s)
    pts = np.rint(uv_s).astype(np.int32)
    for (p, c) in zip(pts, colors):
        cv2.circle(canvas, (int(p[0]), int(p[1])), radius,
                   (int(c[0]), int(c[1]), int(c[2])), -1)
    return canvas


def _draw_points_shared_range(canvas, uv, depth, d_min, d_max,
                              radius=2, x_off=0, y_off=0, intensity=1.0):
    """Draw JET-colored points with a shared depth range. Optional dimming via intensity."""
    if uv.shape[0] == 0:
        return canvas
    order = np.argsort(depth)[::-1]
    uv_s = uv[order]
    depth_s = depth[order]
    colors = _jet_colors_shared(depth_s, d_min, d_max)
    if intensity != 1.0:
        colors = np.clip(colors.astype(np.float32) * float(intensity), 0, 255).astype(np.uint8)
    pts = np.rint(uv_s).astype(np.int32)
    for (p, c) in zip(pts, colors):
        cv2.circle(canvas, (int(p[0]) - x_off, int(p[1]) - y_off), radius,
                   (int(c[0]), int(c[1]), int(c[2])), -1)
    return canvas


def _load_frame(image_dir, files, idx):
    """Load an image by index from a sorted file list."""
    return load_image(os.path.join(image_dir, files[idx]))


# ------------------------- Frame selection ---------------------------------- #

def select_visualization_frame():
    """
    Search all four KITTI sequences for the frame pair whose mean
    optical flow magnitude is closest to 5.0 px.
    Returns (dataset_path, frame_idx, image_files, lidar_files).
    """
    TARGET_FLOW = 5.0

    all_datasets = [
        "2011_09_26_drive_0009_sync",
        "2011_09_26_drive_0005_sync",
        "2011_09_26_drive_0013_sync",
        "2011_09_26_drive_0017_sync",
    ]

    candidates = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]

    best_dataset = None
    best_frame_idx = 0
    best_mag = 0.0
    best_diff = float('inf')
    best_image_files = None
    best_lidar_files = None

    print("Scanning all datasets for best visualization frame...")
    for ds_name in all_datasets:
        ds_path = os.path.join(DATASETS_ROOT, ds_name)
        img_dir = os.path.join(ds_path, "image_02", "data")
        lid_dir = os.path.join(ds_path, "velodyne_points", "data")

        try:
            img_files = list_frame_files(img_dir, ".png")
            lid_files = list_frame_files(lid_dir, ".bin")
        except Exception:
            continue

        total = len(img_files)
        valid_candidates = [i for i in candidates if i + 1 < total]

        for i in valid_candidates:
            try:
                img_a = load_image(os.path.join(img_dir, img_files[i]))
                img_b = load_image(os.path.join(img_dir, img_files[i + 1]))
            except Exception:
                continue

            flow = compute_rgb_flow(img_a, img_b)
            mag = float(np.mean(np.linalg.norm(flow, axis=2)))
            diff = abs(mag - TARGET_FLOW)

            print(f"  {ds_name} frame {i:3d}: {mag:.3f} px")

            if diff < best_diff:
                best_diff = diff
                best_dataset = ds_path
                best_frame_idx = i
                best_mag = mag
                best_image_files = img_files
                best_lidar_files = lid_files

    print(f"\nSelected: {os.path.basename(best_dataset)} "
          f"frame {best_frame_idx} with mean flow {best_mag:.3f} px "
          f"(target was {TARGET_FLOW:.1f} px)")

    return best_dataset, best_frame_idx, best_image_files, best_lidar_files


# ------------------------------- Figures ------------------------------------ #

def fig1_before_correction(img_t, uv_orig, depth_orig, out_path):
    """Raw projection: depth-colored LiDAR over the RGB frame."""
    print("Generating fig1_before_correction...")
    canvas = img_t.copy()
    _draw_points_depth_sorted(canvas, uv_orig, depth_orig, radius=2)
    cv2.imwrite(out_path, canvas)
    _report_saved(out_path)


def fig2_after_correction(img_t, img_t1, uv_orig, depth_orig, uv_corr, out_path):
    """Side-by-side full-image comparison with displacement vectors on the right."""
    print("Generating fig2_after_correction...")
    H, W = img_t.shape[:2]

    # Shared depth normalization so left/right colors are comparable.
    if depth_orig.size > 0:
        d_min = float(np.min(depth_orig))
        d_max = float(np.max(depth_orig))
    else:
        d_min, d_max = 0.0, 1.0

    # --- LEFT: Before ---
    left = img_t.copy()
    _draw_points_shared_range(left, uv_orig, depth_orig, d_min, d_max, radius=2)

    cv2.putText(left, "Before Correction", (21, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(left, "Before Correction", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    # --- RIGHT: After ---
    # Use img_t1 so corrected LiDAR is overlaid on the frame whose timestamp
    # the correction predicts (roughly t + alpha * dt).
    right = img_t1.copy()

    # Displacement with 1-to-1 index correspondence (new lidar_motion returns full array).
    displacement = np.linalg.norm(uv_corr - uv_orig, axis=1)
    moved_mask = displacement > 0.5

    # Step 1: unmoved points in JET colormap at 40% intensity, radius=1.
    _draw_points_shared_range(
        right, uv_orig[~moved_mask], depth_orig[~moved_mask],
        d_min, d_max, radius=1, intensity=0.4,
    )

    # Step 2: moved points at their CORRECTED positions, full brightness, radius=2.
    _draw_points_shared_range(
        right, uv_corr[moved_mask], depth_orig[moved_mask],
        d_min, d_max, radius=2, intensity=1.0,
    )

    # Step 3: displacement vectors for up to 300 random moved points.
    moved_indices = np.where(moved_mask)[0]
    if moved_indices.size > 0:
        rng = np.random.default_rng(0)  # deterministic sampling
        n_sample = min(300, moved_indices.size)
        sample = rng.choice(moved_indices, size=n_sample, replace=False)
        for i in sample:
            orig_pt = (int(round(uv_orig[i, 0])), int(round(uv_orig[i, 1])))
            corr_pt = (int(round(uv_corr[i, 0])), int(round(uv_corr[i, 1])))
            cv2.line(right, orig_pt, corr_pt, (255, 255, 255), 1)

    cv2.putText(right, "After Correction", (21, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(right, "After Correction", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    divider = np.full((3, W, 3), 255, dtype=np.uint8)
    final_fig2 = np.vstack([left, divider, right])

    cv2.imwrite(out_path, final_fig2)
    _report_saved(out_path)


def fig3_zoom_comparison(img_t, img_t1, uv_orig, depth_orig, uv_corr, out_path):
    """Edge-aware zoom crop showing displacement vectors side by side."""
    print("Generating fig3_zoom_comparison...")

    H, W = img_t.shape[:2]

    displacement = np.linalg.norm(uv_corr - uv_orig, axis=1)
    moved_mask = displacement > 0.5

    # -------------------- Step A: choose zoom center --------------------
    moved_points_uv = uv_orig[moved_mask]
    moved_displacements = displacement[moved_mask]

    gray = cv2.cvtColor(img_t, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)

    edge_scores = np.zeros(moved_points_uv.shape[0], dtype=np.float32)
    for i in range(moved_points_uv.shape[0]):
        u_i = int(moved_points_uv[i, 0])
        v_i = int(moved_points_uv[i, 1])
        local = edges[max(0, v_i - 20):min(H, v_i + 20),
                      max(0, u_i - 20):min(W, u_i + 20)]
        edge_scores[i] = float(np.sum(local)) / 255.0

    combined_score = moved_displacements * (edge_scores + 1.0)

    # Exclude likely-sky (top 15%) and likely-road (bottom 40%) regions.
    valid_region = (moved_points_uv[:, 1] > H * 0.15) & (moved_points_uv[:, 1] < H * 0.60)

    if np.any(valid_region):
        valid_idx = np.where(valid_region)[0]
        best_local = int(np.argmax(combined_score[valid_region]))
        best = int(valid_idx[best_local])
        cx = float(moved_points_uv[best, 0])
        cy = float(moved_points_uv[best, 1])
    else:
        cx = W * 0.4
        cy = H * 0.4

    # Crop 480 x 340 centered on (cx, cy), clamped to image.
    crop_w, crop_h = 480, 340
    x1 = int(max(0, cx - crop_w // 2))
    y1 = int(max(0, cy - crop_h // 2))
    x2 = x1 + crop_w
    y2 = y1 + crop_h
    if x2 > W:
        x1 = max(0, W - crop_w)
        x2 = W
    if y2 > H:
        y1 = max(0, H - crop_h)
        y2 = H

    # Points inside the crop.
    in_crop = (
        (uv_orig[:, 0] >= x1) & (uv_orig[:, 0] < x2) &
        (uv_orig[:, 1] >= y1) & (uv_orig[:, 1] < y2)
    )
    crop_depth = depth_orig[in_crop]

    # Shared depth normalization across panels so colors mean the same thing.
    if crop_depth.size > 0:
        dmin = float(np.min(crop_depth))
        dmax = float(np.max(crop_depth))
    else:
        dmin, dmax = 0.0, 1.0

    # -------------------- Step B: left panel (Before) --------------------
    left = img_t[y1:y2, x1:x2].copy()
    _draw_points_shared_range(
        left, uv_orig[in_crop], depth_orig[in_crop],
        dmin, dmax, radius=3, x_off=x1, y_off=y1,
    )
    cv2.putText(left, "Before", (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(left, "Before", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    # -------------------- Step C: right panel (After) --------------------
    # Use img_t1 so corrected LiDAR sits on the frame its prediction targets.
    right = img_t1[y1:y2, x1:x2].copy()

    in_crop_unmoved = in_crop & ~moved_mask
    in_crop_moved = in_crop & moved_mask

    # Unmoved -> solid gray at original positions.
    pts_unmoved = np.rint(uv_orig[in_crop_unmoved]).astype(np.int32)
    for p in pts_unmoved:
        cv2.circle(right, (int(p[0]) - x1, int(p[1]) - y1), 2, (130, 130, 130), -1)

    # Moved -> JET (shared depth range) at CORRECTED positions.
    _draw_points_shared_range(
        right, uv_corr[in_crop_moved], depth_orig[in_crop_moved],
        dmin, dmax, radius=3, x_off=x1, y_off=y1,
    )

    # Displacement arrows: draw only the 60 largest-displacement movers
    # (> 3 px) with both endpoints inside the crop, to avoid clutter.
    ch, cw = right.shape[:2]
    moved_idx_crop = np.where(in_crop_moved)[0]
    disp_in_crop = displacement[moved_idx_crop]
    sort_order = np.argsort(disp_in_crop)[::-1]
    moved_idx_crop_sorted = moved_idx_crop[sort_order]
    arrow_count = 0
    for i in moved_idx_crop_sorted:
        if arrow_count >= 60:
            break
        if displacement[i] <= 3.0:
            continue
        start = (int(uv_orig[i, 0] - x1), int(uv_orig[i, 1] - y1))
        end = (int(uv_corr[i, 0] - x1), int(uv_corr[i, 1] - y1))
        if (0 <= start[0] < cw and 0 <= start[1] < ch and
                0 <= end[0] < cw and 0 <= end[1] < ch):
            cv2.arrowedLine(right, start, end, (0, 220, 255), 1,
                            tipLength=0.5)  # yellow arrows
            arrow_count += 1

    cv2.putText(right, "After (Event-Guided)", (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(right, "After (Event-Guided)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # -------------------- Step D: assemble --------------------
    divider = np.full((right.shape[0], 4, 3), 255, dtype=np.uint8)
    fig3 = np.hstack([left, divider, right])

    cv2.imwrite(out_path, fig3)
    _report_saved(out_path)


def fig4_event_map(img_t, events, out_path):
    """Binary event polarity overlay on a grayscale version of img_t."""
    print("Generating fig4_event_map...")

    # `events` is binary {-1, 0, +1} from the fixed simulate_events.
    on_mask = events > 0.5
    off_mask = events < -0.5

    density = float(np.mean(on_mask | off_mask))
    print(f"  event density (after fix): {density:.4f}")

    gray = cv2.cvtColor(img_t, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Show events as boundaries of event clusters (not solid fill) so the
    # scene stays readable even at high density. Interior of event regions
    # is darkened slightly; the 2px-wide ring around each region is drawn
    # in bright polarity colors.
    on_uint8 = on_mask.astype(np.uint8) * 255
    off_uint8 = off_mask.astype(np.uint8) * 255

    thin_kernel = np.ones((3, 3), np.uint8)
    on_eroded = cv2.erode(on_uint8, thin_kernel, iterations=2)
    off_eroded = cv2.erode(off_uint8, thin_kernel, iterations=2)

    # Boundary = original minus eroded. Safe in uint8: eroded ⊆ original.
    on_boundary = on_uint8 - on_eroded
    off_boundary = off_uint8 - off_eroded

    result = gray_bgr.copy().astype(np.float32)

    interior_mask = (on_eroded > 0) | (off_eroded > 0)
    result[interior_mask] = result[interior_mask] * 0.5

    result[on_boundary > 0] = [255, 255, 255]
    result[off_boundary > 0] = [220, 100, 0]  # BGR -> orange-blue

    result = np.clip(result, 0, 255).astype(np.uint8)

    cv2.imwrite(out_path, result)
    _report_saved(out_path)


def fig5_ablation_bar_chart(out_path):
    """Grouped bar chart: inconsistency per dataset per method (post-fix values)."""
    print("Generating fig5_ablation_bar_chart...")

    datasets = ['0009', '0005', '0013', '0017']
    no_correction_px = [16.06, 9.87, 13.14, 7.08]
    rgb_only_px      = [15.07, 9.71, 12.77, 7.08]
    event_guided_px  = [ 3.79, 3.11,  3.51, 2.83]
    self_consistency = [ 2.30, 0.72,  2.13, 1.87]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(datasets))
    bw = 0.22

    bars1 = ax.bar(x - bw, no_correction_px, bw, label='No Correction',
                   color='#d62728', alpha=0.85, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x, rgb_only_px, bw, label='RGB Flow Only',
                   color='#ff7f0e', alpha=0.85, edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + bw, event_guided_px, bw, label='Event-Guided (Ours)',
                   color='#2ca02c', alpha=0.85, edgecolor='black', linewidth=0.5)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h + 0.15,
                    f'{h:.2f}', ha='center', va='bottom',
                    fontsize=7.5, fontweight='bold')

    for i, y in enumerate(self_consistency):
        ax.hlines(y, x[i] - bw * 1.8, x[i] + bw * 1.8,
                  colors='#7f7f7f', linewidth=1.5, linestyle='--')
    ax.plot([], [], color='#7f7f7f', linewidth=1.5, linestyle='--',
            label='Image Self-Consistency (Lower Bound)')

    ax.set_ylabel('Temporal Inconsistency (px)', fontsize=12)
    ax.set_xlabel('KITTI Sequence', fontsize=12)
    ax.set_title('Ablation Study: Temporal Inconsistency Reduction by Method',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'drive_{d}_sync' for d in datasets], fontsize=10)
    ax.set_ylim(0, 20)
    ax.legend(loc='upper right', fontsize=9)
    ax.yaxis.grid(True, alpha=0.3, linestyle=':')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    _report_saved(out_path)


def fig6_results_table(out_path):
    """Clean results table (post-fix values)."""
    print("Generating fig6_results_table...")

    headers = ['Sequence', 'Original (px)', 'RGB-Only (px)', 'Ours (px)', 'Improvement (%)']
    rows = [
        ['drive_0009_sync', '16.06', '15.07', '3.79', '76.4%'],
        ['drive_0005_sync',  '9.87',  '9.71', '3.11', '68.5%'],
        ['drive_0013_sync', '13.14', '12.77', '3.51', '73.3%'],
        ['drive_0017_sync',  '7.08',  '7.08', '2.83', '60.0%'],
        ['Mean',            '11.54', '11.16', '3.31', '69.5%'],
    ]

    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.axis('off')

    table = ax.table(cellText=rows, colLabels=headers,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.7)

    n_cols = len(headers)
    n_data_rows = len(rows)
    ours_col = 3
    mean_row_idx = n_data_rows

    for c in range(n_cols):
        cell = table[(0, c)]
        cell.set_facecolor('#1f3d6e')
        cell.set_text_props(color='white', fontweight='bold')

    for r in range(1, n_data_rows + 1):
        is_mean = (r == mean_row_idx)
        for c in range(n_cols):
            cell = table[(r, c)]
            if is_mean:
                cell.set_facecolor('#dde8f5')
                cell.set_text_props(fontweight='bold')
            elif c == ours_col:
                cell.set_facecolor('#d4edda')
            elif r % 2 == 1:
                cell.set_facecolor('#f5f5f5')
            if is_mean and c == ours_col:
                cell.set_facecolor('#d4edda')
                cell.set_text_props(fontweight='bold')

    ax.text(0.5, 0.02,
            '† Improvement measured as reduction in LiDAR-image motion disagreement (px). '
            'Results after event-guided pipeline fix.',
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=8, style='italic', color='gray')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    _report_saved(out_path)


def fig7_pipeline_diagram(out_path):
    """Block diagram of the event-guided motion compensation pipeline."""
    print("Generating fig7_pipeline_diagram...")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis('off')

    def add_block(cx, cy, w, h, label, color):
        x = cx - w / 2.0
        y = cy - h / 2.0
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle="round,pad=0.15",
                             facecolor=color, edgecolor='#2c5f7a',
                             linewidth=1.5)
        ax.add_patch(box)
        ax.text(cx, cy, label, ha='center', va='center', fontsize=9)
        return (cx, cy, w, h)

    B1 = add_block(1.2, 3.5, 1.8, 0.7, "RGB Frame t\nRGB Frame t+1", '#c8e6c9')
    B2 = add_block(3.5, 3.5, 1.8, 0.7, "Optical Flow\n(Farneback)", '#aec6cf')
    B3 = add_block(1.2, 1.5, 1.8, 0.7, "LiDAR Scan t",                '#fff9c4')
    B4 = add_block(3.5, 1.5, 1.8, 0.7, "Projection\n(P·R·T)", '#aec6cf')
    B5 = add_block(1.2, 2.5, 1.8, 0.7, "Log-Intensity\nDifference",   '#f8bbd0')
    B6 = add_block(3.5, 2.5, 1.8, 0.7, "Event Confidence\nMask C(x,y)", '#f8bbd0')
    B7 = add_block(6.5, 2.5, 2.0, 0.9, "Event-Guided\nMotion\nCompensation", '#ffe082')
    B8 = add_block(9.5, 2.5, 1.8, 0.7, "Corrected\nProjection",       '#c8e6c9')

    def right_edge(B):
        cx, cy, w, _h = B
        return (cx + w / 2.0, cy)

    def left_edge(B, dy=0.0):
        cx, cy, w, _h = B
        return (cx - w / 2.0, cy + dy)

    arrow = dict(arrowstyle='->', color='#2c5f7a', lw=1.5)

    ax.annotate("", xy=left_edge(B2), xytext=right_edge(B1), arrowprops=arrow)
    ax.annotate("", xy=left_edge(B4), xytext=right_edge(B3), arrowprops=arrow)
    ax.annotate("", xy=left_edge(B6), xytext=right_edge(B5), arrowprops=arrow)
    ax.annotate("", xy=left_edge(B8), xytext=right_edge(B7), arrowprops=arrow)

    ax.annotate("", xy=left_edge(B7, dy=+0.25), xytext=right_edge(B2), arrowprops=arrow)
    ax.annotate("", xy=left_edge(B7, dy=-0.25), xytext=right_edge(B4), arrowprops=arrow)
    ax.annotate("", xy=left_edge(B7, dy=0.0),   xytext=right_edge(B6), arrowprops=arrow)

    ax.text(3.5, 1.9,
            r'$C(x,y) = \mathbf{1}[|\Delta I| > \theta]$',
            ha='center', fontsize=8, color='#555555')
    ax.text(6.5, 1.9,
            r'$(u^*,v^*) = (u+\alpha\Delta u,\, v+\alpha\Delta v)$ if $C=1$',
            ha='center', fontsize=8, color='#555555')

    ax.text(7.0, 4.7, 'Event-Guided Motion Compensation Pipeline',
            ha='center', fontsize=13, fontweight='bold')

    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    _report_saved(out_path)


# -------------------------------- Main -------------------------------------- #

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -------------------- STEP 0: Auto-select best frame across datasets ----
    best_ds_path, BEST_FRAME_IDX, image_files, lidar_files = \
        select_visualization_frame()

    # -------------------- Calibration from the selected dataset ------------
    calib_velo_path = os.path.join(best_ds_path, "calib_velo_to_cam.txt")
    calib_cam_path = os.path.join(best_ds_path, "calib_cam_to_cam.txt")

    Tr = parse_calib_velo_to_cam(calib_velo_path)
    R_rect, P_rect = parse_calib_cam_to_cam(calib_cam_path, camera_id="02")

    image_dir = os.path.join(best_ds_path, "image_02", "data")
    lidar_dir = os.path.join(best_ds_path, "velodyne_points", "data")

    # -------------------- Load frames + LiDAR for chosen index --------------
    img_t = load_image(os.path.join(image_dir, image_files[BEST_FRAME_IDX]))
    img_t1 = load_image(os.path.join(image_dir, image_files[BEST_FRAME_IDX + 1]))
    lidar_t = load_lidar(os.path.join(lidar_dir, lidar_files[BEST_FRAME_IDX]))

    # -------------------- Projection (shared across fig1-3) -----------------
    uv_orig, depth_orig, _stats = project_lidar_to_image(
        lidar_t, Tr, R_rect, P_rect, img_t.shape
    )

    # -------------------- Events + flow + confidence ------------------------
    # events are binary {-1, 0, +1} after the fixed simulate_events.
    events = simulate_events(img_t, img_t1, threshold=0.3)
    conf = event_confidence(events)

    flow = compute_rgb_flow(img_t, img_t1)

    # -------------------- Event-guided correction (computed once) ----------
    uv_corr, depth_corr = move_lidar_points_weighted(
        uv_orig, depth_orig, flow, conf, conf_thresh=0.5
    )

    # =========================== FIGURES ===================================
    fig1_before_correction(
        img_t, uv_orig, depth_orig,
        os.path.join(OUTPUT_DIR, "fig1_before_correction.png"),
    )

    fig2_after_correction(
        img_t, img_t1, uv_orig, depth_orig, uv_corr,
        os.path.join(OUTPUT_DIR, "fig2_after_correction.png"),
    )

    fig3_zoom_comparison(
        img_t, img_t1, uv_orig, depth_orig, uv_corr,
        os.path.join(OUTPUT_DIR, "fig3_zoom_comparison.png"),
    )

    fig4_event_map(
        img_t, events,
        os.path.join(OUTPUT_DIR, "fig4_event_map.png"),
    )

    fig5_ablation_bar_chart(
        os.path.join(OUTPUT_DIR, "fig5_ablation_bar_chart.png"),
    )

    fig6_results_table(
        os.path.join(OUTPUT_DIR, "fig6_results_table.png"),
    )

    fig7_pipeline_diagram(
        os.path.join(OUTPUT_DIR, "fig7_pipeline_diagram.png"),
    )

    # --------------------------- Summary -----------------------------------
    print("\n================ SUMMARY ================")
    print(f"{'Filename':<35}{'Size (KB)':>12}{'Dimensions (WxH)':>22}")
    print("-" * 69)
    total = 0
    for name in sorted(os.listdir(OUTPUT_DIR)):
        path = os.path.join(OUTPUT_DIR, name)
        if not os.path.isfile(path):
            continue
        size_kb = os.path.getsize(path) / 1024.0
        dims = ""
        if name.lower().endswith(".png"):
            im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if im is not None:
                h, w = im.shape[:2]
                dims = f"{w} x {h}"
        print(f"{name:<35}{size_kb:>12.1f}{dims:>22}")
        total += 1
    print("-" * 69)
    print(f"Total files generated: {total}")


if __name__ == "__main__":
    main()
