"""
depth_completion_eval.py — Phase 3 downstream depth completion evaluation.

Evaluates whether event-guided LiDAR temporal correction produces
better dense depth maps via classical IP-Basic completion (no
training, no GPU). Compares uncorrected vs. corrected projections
against KITTI semi-dense GT (when available) and against the
original sparse LiDAR observations (self-consistency).
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))

from calibration import parse_calib_cam_to_cam, parse_calib_velo_to_cam
from events import event_confidence, simulate_events
from flow import compute_rgb_flow, smooth_flow
from lidar_motion import move_lidar_points_weighted
from loader import list_frame_files, load_image, load_lidar
from projection import project_lidar_to_image


DEFAULT_BASE_DIR = r"C:\Users\sahaa\OneDrive\Desktop\Honors\datasets\fusion"
DEFAULT_SEQUENCES = "0009_sync,0005_sync,0051_sync,0117_sync"
DEFAULT_OUTPUT_DIR = "./depth_results"
KITTI_DATE = "2011_09_26"
KITTI_DEPTH_SCALE = 256.0  # KITTI depth PNGs encode metres * 256 in uint16


# =======================================================================
# PART A: IP-Basic Depth Completion
# =======================================================================

def rasterize_sparse_depth(
    uv: np.ndarray,
    depth: np.ndarray,
    image_shape: tuple,
    scale: float = 256.0,
) -> np.ndarray:
    """
    Rasterize projected LiDAR points to a sparse depth image (uint16).

    For pixels that receive multiple projections, the nearest depth
    (minimum value) is kept — this matches the KITTI depth completion
    convention and is the physically correct choice (closer surfaces
    occlude farther ones).

    Parameters
    ----------
    uv : (N, 2) float32
        Projected pixel coordinates (column, row order).
    depth : (N,) float32
        Per-point depth in metres.
    image_shape : tuple
        Image shape; only the first two entries (H, W) are used.
    scale : float
        Multiplier converting metres -> uint16 storage units (default
        256 to match KITTI's 16-bit PNG convention).

    Returns
    -------
    sparse_depth : (H, W) uint16
        Rasterized depth image; zeros indicate "no LiDAR observation".
    """
    if uv is None or depth is None:
        raise ValueError("uv and depth must not be None")

    uv = np.asarray(uv, dtype=np.float32)
    depth = np.asarray(depth, dtype=np.float32)

    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError(f"uv must be (N,2), got {uv.shape}")
    if depth.ndim != 1 or depth.shape[0] != uv.shape[0]:
        raise ValueError(
            f"depth must be (N,) matching uv, got {depth.shape} vs uv {uv.shape}"
        )

    h, w = int(image_shape[0]), int(image_shape[1])
    sparse = np.zeros((h, w), dtype=np.uint16)

    if uv.shape[0] == 0:
        return sparse

    # Filter to finite, in-bounds, positive-depth points.
    finite = (
        np.isfinite(uv[:, 0])
        & np.isfinite(uv[:, 1])
        & np.isfinite(depth)
        & (depth > 0.0)
    )
    if not np.any(finite):
        return sparse

    u = np.rint(uv[finite, 0]).astype(np.int32)
    v = np.rint(uv[finite, 1]).astype(np.int32)
    d = depth[finite]

    in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if not np.any(in_bounds):
        return sparse

    u = u[in_bounds]
    v = v[in_bounds]
    d = d[in_bounds]

    # Encode metres as uint16 with the requested scale, clamped into the
    # uint16 range so very-far returns do not wrap around.
    d_scaled = np.clip(np.rint(d * scale), 1.0, 65535.0).astype(np.uint16)

    # Sort by descending depth so the nearest (smallest) depth lands last
    # at each pixel — the final assignment "wins" the duplicate resolution.
    order = np.argsort(-d_scaled.astype(np.int32))
    sparse[v[order], u[order]] = d_scaled[order]
    return sparse


def _edge_aware_smooth(
    depth_dilated: np.ndarray,
    image_bgr: np.ndarray,
    sigma: float = 20.0,
    ksize: int = 15,
) -> np.ndarray:
    """
    Approximate bilateral filtering using a Sobel-edge-weighted blend
    between the dilated depth and a Gaussian-smoothed version.

    Edges (high gradient magnitude) -> trust the dilated depth (sharp).
    Smooth regions (low gradient) -> trust the Gaussian (low noise).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = np.sqrt(gx * gx + gy * gy)

    edge_weight = np.exp(-edge_mag / max(sigma, 1e-6)).astype(np.float32)
    edge_weight = np.clip(edge_weight, 0.0, 1.0)

    if ksize % 2 == 0:
        ksize += 1
    depth_gauss = cv2.GaussianBlur(depth_dilated, (ksize, ksize), 0)

    # High edge_weight -> low-gradient region -> prefer Gaussian.
    # Low edge_weight  -> high-gradient (edge) -> keep dilated depth.
    blended = edge_weight * depth_gauss + (1.0 - edge_weight) * depth_dilated
    return blended.astype(np.float32)


def ip_basic_completion(
    sparse_depth: np.ndarray,
    image_bgr: np.ndarray,
    use_bilateral: bool = True,
) -> np.ndarray:
    """
    Classical IP-Basic depth completion (Ku et al., 2018) variant.

    Pipeline (all depth math in float32 metres):
      1. Decode sparse depth to metres (uint16 / 256).
      2. Small (5x5 ellipse) dilation to fill 1-pixel gaps.
      3. Larger (15x15 ellipse) dilation for void regions.
      4. Optional edge-aware smoothing (Sobel-weighted Gaussian blend
         in place of the bilateral filter; cv2.ximgproc is not used).
      5. 5x5 median blur to reject speckle.
      6. Restore original sparse-depth values at observed pixels.
      7. Fill any remaining holes with the median of valid depths.

    Parameters
    ----------
    sparse_depth : (H, W) uint16 or float32
        Sparse depth — uint16 is treated as KITTI mm/256, float32 as metres.
    image_bgr : (H, W, 3) uint8
        RGB image used as guidance for edge-aware smoothing.
    use_bilateral : bool
        Toggle the edge-aware smoothing step.

    Returns
    -------
    dense_depth : (H, W) float32
        Completed depth in metres. Zero only if no valid input pixel
        exists at all (degenerate edge case).
    """
    if sparse_depth is None:
        raise ValueError("sparse_depth must not be None")
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must be (H,W,3)")
    if sparse_depth.shape[:2] != image_bgr.shape[:2]:
        raise ValueError(
            f"sparse_depth and image_bgr shape mismatch: "
            f"{sparse_depth.shape} vs {image_bgr.shape}"
        )

    # 1. Decode to float32 metres.
    if sparse_depth.dtype == np.uint16:
        depth = sparse_depth.astype(np.float32) / KITTI_DEPTH_SCALE
    else:
        depth = sparse_depth.astype(np.float32, copy=True)

    valid_mask = depth > 0.0
    if not np.any(valid_mask):
        # Nothing to propagate; return zeros so the caller can detect it.
        return np.zeros_like(depth, dtype=np.float32)

    original_valid_depth = depth.copy()

    # 2. Small dilation — 5x5 ellipse kernel.
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    depth_dilated = cv2.dilate(depth, small_kernel)

    # 3. Larger dilation — 15x15 ellipse kernel.
    large_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    depth_dilated = cv2.dilate(depth_dilated, large_kernel)

    # 4. Edge-aware smoothing (replaces bilateral filter).
    if use_bilateral:
        try:
            depth_smooth = _edge_aware_smooth(
                depth_dilated, image_bgr, sigma=20.0, ksize=15
            )
        except Exception as exc:
            print(f"[ip_basic] edge-aware smoothing failed ({exc}); falling back to gaussian_filter")
            depth_smooth = ndimage.gaussian_filter(depth_dilated, sigma=2.0).astype(np.float32)
    else:
        depth_smooth = depth_dilated

    # 5. Median blur — reject speckle. cv2.medianBlur on float32 supports ksize <= 5.
    depth_for_median = depth_smooth.astype(np.float32)
    depth_smooth = cv2.medianBlur(depth_for_median, 5)

    # 6. Restore original LiDAR observations exactly (preserve input accuracy).
    depth_smooth[valid_mask] = original_valid_depth[valid_mask]

    # 7. Fill any remaining holes with the median of valid depths. We
    #    treat "essentially zero" (< min_depth) as a hole — the edge-aware
    #    blend can leave tiny floating-point residuals in regions that
    #    never received LiDAR support, and those near-zeros explode the
    #    inverse-depth metrics (1/pred -> 10^15) downstream.
    remaining_holes = depth_smooth < 0.1
    if np.any(remaining_holes):
        median_depth = float(np.median(original_valid_depth[valid_mask]))
        depth_smooth[remaining_holes] = median_depth

    return depth_smooth.astype(np.float32)


def run_completion(
    uv: np.ndarray,
    depth: np.ndarray,
    image_bgr: np.ndarray,
    image_shape: tuple,
) -> np.ndarray:
    """
    Convenience wrapper: rasterize the projected points then run
    IP-Basic completion against the supplied image.
    """
    sparse = rasterize_sparse_depth(uv, depth, image_shape)
    dense = ip_basic_completion(sparse, image_bgr, use_bilateral=True)
    return dense


# =======================================================================
# PART B: Evaluation Metrics
# =======================================================================

def compute_depth_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    min_depth: float = 0.1,
    max_depth: float = 80.0,
) -> dict:
    """
    Standard KITTI depth completion metrics.

    Valid mask: gt in [min_depth, max_depth] AND pred > 0.

    Returns a dict with keys:
      rmse  (m), mae (m), irmse (1/km), imae (1/km), absrel, n_valid.
    All NaN if fewer than 10 valid pixels.
    """
    if pred is None or gt is None:
        raise ValueError("pred and gt must not be None")
    if pred.shape != gt.shape:
        raise ValueError(f"pred / gt shape mismatch: {pred.shape} vs {gt.shape}")

    pred = pred.astype(np.float32, copy=False)
    gt = gt.astype(np.float32, copy=False)

    valid = (gt > min_depth) & (gt < max_depth) & (pred > 0.0) & np.isfinite(pred) & np.isfinite(gt)
    n_valid = int(np.count_nonzero(valid))

    nan_result = {
        "rmse": float("nan"),
        "mae": float("nan"),
        "irmse": float("nan"),
        "imae": float("nan"),
        "absrel": float("nan"),
        "n_valid": n_valid,
    }
    if n_valid < 10:
        return nan_result

    p = np.clip(pred[valid], 1e-3, None)
    g = gt[valid]

    diff = p - g
    rmse = float(np.sqrt(np.mean(diff * diff)))
    mae = float(np.mean(np.abs(diff)))

    inv_diff = (1.0 / p) - (1.0 / g)
    irmse = float(np.sqrt(np.mean(inv_diff * inv_diff)) * 1000.0)
    imae = float(np.mean(np.abs(inv_diff)) * 1000.0)

    absrel = float(np.mean(np.abs(diff) / g))

    return {
        "rmse": rmse,
        "mae": mae,
        "irmse": irmse,
        "imae": imae,
        "absrel": absrel,
        "n_valid": n_valid,
    }


def load_kitti_gt_depth(gt_path: str) -> np.ndarray:
    """
    Load a KITTI ground truth depth PNG.

    KITTI stores depth as a 16-bit PNG where pixel_value / 256 is the
    depth in metres. Zero pixels are invalid / missing.
    """
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"GT depth file not found: {gt_path}")

    img = cv2.imread(gt_path, cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise ValueError(f"Failed to decode GT depth PNG: {gt_path}")
    if img.dtype != np.uint16:
        # Some tools emit 8-bit; warn loudly but proceed.
        print(f"[load_kitti_gt_depth] unexpected dtype {img.dtype} at {gt_path} — treating as uint16-equivalent")
        img = img.astype(np.uint16)

    return (img.astype(np.float32) / KITTI_DEPTH_SCALE)


def compute_boundary_depth_score(
    uv_original: np.ndarray,
    depth_original: np.ndarray,
    uv_corrected: np.ndarray,
    depth_corrected: np.ndarray,
    image_t1: np.ndarray,
) -> dict:
    """
    Boundary-Aware Depth Projection Score (BDPS).

    Measures depth-discontinuity sharpness at image boundaries by
    taking the Sobel gradient magnitude of the rasterised sparse
    projected depth at pixels within 5 px of a Canny edge in
    image_t1. Higher score = sharper depth edges = better temporal
    alignment between LiDAR and the RGB frame.

    Pipeline:
      1. Canny(image_t1, 30, 90) -> 5x5 dilation -> boundary_mask.
      2. Rasterise both projected scans to float32-metre depth maps.
      3. Sobel(Gx, Gy) of each depth map; magnitude = sqrt(Gx^2+Gy^2).
      4. Score = mean(magnitude) over (boundary_mask AND sparse > 0).
      5. improvement_pct = (after - before) / before * 100 ; positive
         means the corrected projection has sharper depth boundaries.

    Returns dict with keys:
      score_before, score_after, improvement, improvement_pct,
      n_boundary_pts_before, n_boundary_pts_after.
    Returns NaN values (with a printed warning) if either side has
    fewer than 10 boundary-mask pixels carrying valid depth.
    """
    if image_t1 is None or image_t1.ndim != 3 or image_t1.shape[2] != 3:
        raise ValueError("image_t1 must be (H,W,3) BGR uint8")

    # 1. Canny + 5x5 dilation -> boundary_mask.
    gray = cv2.cvtColor(image_t1, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 90)
    boundary_kernel = np.ones((5, 5), dtype=np.uint8)
    boundary_mask = cv2.dilate(edges, boundary_kernel, iterations=1) > 0

    # 2. Rasterise both scans to float32 metres. We reuse
    #    rasterize_sparse_depth's default scale=256 path and divide
    #    by 256; a literal scale=1.0 would round depths to integer
    #    metres via uint16 quantisation and corrupt the Sobel output.
    sparse_orig_u16 = rasterize_sparse_depth(uv_original, depth_original, image_t1.shape)
    sparse_corr_u16 = rasterize_sparse_depth(uv_corrected, depth_corrected, image_t1.shape)
    sparse_orig = sparse_orig_u16.astype(np.float32) / KITTI_DEPTH_SCALE
    sparse_corr = sparse_corr_u16.astype(np.float32) / KITTI_DEPTH_SCALE

    # 3. Sobel gradient magnitude of each sparse depth map.
    gx_o = cv2.Sobel(sparse_orig, cv2.CV_32F, 1, 0, ksize=3)
    gy_o = cv2.Sobel(sparse_orig, cv2.CV_32F, 0, 1, ksize=3)
    grad_o = np.sqrt(gx_o * gx_o + gy_o * gy_o)

    gx_c = cv2.Sobel(sparse_corr, cv2.CV_32F, 1, 0, ksize=3)
    gy_c = cv2.Sobel(sparse_corr, cv2.CV_32F, 0, 1, ksize=3)
    grad_c = np.sqrt(gx_c * gx_c + gy_c * gy_c)

    # 4. Restrict to boundary pixels carrying valid depth.
    valid_orig = boundary_mask & (sparse_orig > 0.0)
    valid_corr = boundary_mask & (sparse_corr > 0.0)
    n_before = int(np.count_nonzero(valid_orig))
    n_after = int(np.count_nonzero(valid_corr))

    if n_before < 10 or n_after < 10:
        print(f"[BDPS] WARNING: insufficient boundary points "
              f"(before={n_before}, after={n_after}) — returning NaN")
        return {
            "score_before": float("nan"),
            "score_after": float("nan"),
            "improvement": float("nan"),
            "improvement_pct": float("nan"),
            "n_boundary_pts_before": n_before,
            "n_boundary_pts_after": n_after,
        }

    score_before = float(np.mean(grad_o[valid_orig]))
    score_after = float(np.mean(grad_c[valid_corr]))
    improvement = score_after - score_before
    if score_before > 1e-9 and np.isfinite(score_before):
        improvement_pct = float(improvement / score_before * 100.0)
    else:
        improvement_pct = float("nan")

    return {
        "score_before": score_before,
        "score_after": score_after,
        "improvement": improvement,
        "improvement_pct": improvement_pct,
        "n_boundary_pts_before": n_before,
        "n_boundary_pts_after": n_after,
    }


# =======================================================================
# PART C: Main Evaluation Loop
# =======================================================================

def _safe_pct(before: float, after: float) -> float:
    """Percent improvement (positive = corrected better) for downward metrics."""
    if not np.isfinite(before) or not np.isfinite(after) or abs(before) < 1e-12:
        return float("nan")
    return float((before - after) / before * 100.0)


def _aggregate_metrics(per_frame_metrics: list) -> dict:
    """Mean over per-frame metric dicts, NaN-safe."""
    keys = ("rmse", "mae", "irmse", "imae", "absrel")
    out = {}
    for k in keys:
        vals = [m[k] for m in per_frame_metrics if m is not None and np.isfinite(m.get(k, np.nan))]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


def evaluate_sequence(
    dataset_path: str,
    gt_depth_dir,
    max_frames: int = 50,
    event_threshold: float = 0.2,
    output_dir=None,
) -> dict:
    """
    Run depth completion evaluation on a single KITTI sequence.

    Iterates consecutive frame pairs (t, t+1). The LiDAR scan at t is
    projected and corrected toward t+1 using events + flow; both the
    uncorrected and corrected projections are completed via IP-Basic
    using image_t1 as guidance. Each pair contributes one row of
    metrics.

    See the module docstring for the GT vs. self-consistency trade-off.
    """
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    sequence_name = os.path.basename(os.path.normpath(dataset_path))

    image_dir = os.path.join(dataset_path, "image_02", "data")
    lidar_dir = os.path.join(dataset_path, "velodyne_points", "data")
    velo_calib = os.path.join(dataset_path, "calib_velo_to_cam.txt")
    cam_calib = os.path.join(dataset_path, "calib_cam_to_cam.txt")

    image_files = list_frame_files(image_dir, ".png")
    try:
        lidar_files = list_frame_files(lidar_dir, ".bin")
    except FileNotFoundError:
        lidar_files = []
    if not lidar_files:
        lidar_files = list_frame_files(lidar_dir, ".txt")

    n_pairs = min(len(image_files), len(lidar_files)) - 1
    if n_pairs <= 0:
        raise RuntimeError(f"Sequence has no consecutive pairs: {dataset_path}")

    n_pairs = min(n_pairs, max_frames)

    tr_velo_to_cam = parse_calib_velo_to_cam(velo_calib)
    r_rect, p_rect = parse_calib_cam_to_cam(cam_calib, camera_id="02")

    gt_available_seq = gt_depth_dir is not None and os.path.isdir(gt_depth_dir)
    if gt_depth_dir is not None and not gt_available_seq:
        print(f"[{sequence_name}] WARNING: GT depth dir does not exist: {gt_depth_dir}")
        print(f"[{sequence_name}] Falling back to BDPS only.")

    per_frame = []
    metrics_uncorr_list = []
    metrics_corr_list = []
    bdps_list = []

    prev_flow = None
    n_evaluated = 0
    n_failed = 0
    n_gt_hits = 0

    print(f"\n[{sequence_name}] Evaluating up to {n_pairs} frame pairs "
          f"(GT available: {gt_available_seq})")

    for i in range(n_pairs):
        try:
            img_name_t = image_files[i]
            img_name_t1 = image_files[i + 1]
            lidar_name_t = lidar_files[i]

            image_t = load_image(os.path.join(image_dir, img_name_t))
            image_t1 = load_image(os.path.join(image_dir, img_name_t1))
            lidar_xyz = load_lidar(os.path.join(lidar_dir, lidar_name_t))

            uv_orig, depth_orig, _ = project_lidar_to_image(
                lidar_xyz, tr_velo_to_cam, r_rect, p_rect, image_t.shape
            )
            if uv_orig.shape[0] == 0:
                raise RuntimeError("no projected points")

            events = simulate_events(image_t, image_t1, threshold=event_threshold)
            flow_raw = compute_rgb_flow(image_t, image_t1)
            flow = smooth_flow(flow_raw, prev_flow, alpha=0.7)
            confidence = event_confidence(events)

            # Apply correction to the full set of projected points.
            # BDPS is computed on the sparse projections directly, so
            # the previous 80/20 hold-out split is no longer needed.
            uv_corr, depth_corr = move_lidar_points_weighted(
                uv_orig, depth_orig, flow, confidence
            )

            dense_uncorr = run_completion(
                uv_orig, depth_orig, image_t1, image_t1.shape
            )
            dense_corr = run_completion(
                uv_corr, depth_corr, image_t1, image_t1.shape
            )

            if i == 0:
                print(f"  [diag] uv_original.shape={uv_orig.shape}  "
                      f"depth_original.max={float(depth_orig.max()):.2f}m")
                print(f"  [diag] uv_corrected.shape={uv_corr.shape}  "
                      f"depth_corrected.max={float(depth_corr.max()):.2f}m")
                print(f"[DIAG] dense_uncorr nonzero: {int(np.count_nonzero(dense_uncorr))}")
                print(f"[DIAG] dense_corr  nonzero: {int(np.count_nonzero(dense_corr))}")

            frame_record = {
                "frame_index": i,
                "frame_name_t": os.path.splitext(img_name_t)[0],
                "frame_name_t1": os.path.splitext(img_name_t1)[0],
            }

            # GT-based metrics — keyed off image_t1 since the corrected
            # projection is aligned with frame t+1.
            if gt_available_seq:
                gt_path = os.path.join(gt_depth_dir, img_name_t1)
                if os.path.isfile(gt_path):
                    gt = load_kitti_gt_depth(gt_path)
                    if gt.shape != dense_uncorr.shape:
                        print(f"[{sequence_name}] frame {i}: GT shape {gt.shape} != "
                              f"pred {dense_uncorr.shape}; skipping GT eval")
                    else:
                        m_u = compute_depth_metrics(dense_uncorr, gt)
                        m_c = compute_depth_metrics(dense_corr, gt)
                        if m_u["n_valid"] >= 10 and m_c["n_valid"] >= 10:
                            metrics_uncorr_list.append(m_u)
                            metrics_corr_list.append(m_c)
                            frame_record["metrics_uncorrected"] = m_u
                            frame_record["metrics_corrected"] = m_c
                            n_gt_hits += 1

            bdps = compute_boundary_depth_score(
                uv_orig, depth_orig, uv_corr, depth_corr, image_t1
            )
            bdps_list.append(bdps)
            frame_record["boundary_depth_score"] = bdps

            per_frame.append(frame_record)
            prev_flow = flow
            n_evaluated += 1

            if (i + 1) % 10 == 0 or (i + 1) == n_pairs:
                m_u = frame_record.get("metrics_uncorrected")
                m_c = frame_record.get("metrics_corrected")
                if m_u is not None and m_c is not None:
                    print(f"  Frame {i + 1}/{n_pairs}: RMSE uncorr={m_u['rmse']:.3f} corr={m_c['rmse']:.3f}")
                else:
                    print(f"  Frame {i + 1}/{n_pairs}: BDPS before={bdps['score_before']:.3f} "
                          f"after={bdps['score_after']:.3f}")

        except Exception as exc:
            n_failed += 1
            print(f"  Frame {i}: FAILED ({type(exc).__name__}: {exc}) — skipping")
            continue

    # Aggregate.
    metrics_uncorr_mean = _aggregate_metrics(metrics_uncorr_list)
    metrics_corr_mean = _aggregate_metrics(metrics_corr_list)
    metrics_improvement = {
        f"{k}_pct": _safe_pct(metrics_uncorr_mean[k], metrics_corr_mean[k])
        for k in ("rmse", "mae", "irmse", "imae", "absrel")
    }

    bdps_before_vals = [b["score_before"] for b in bdps_list
                        if np.isfinite(b.get("score_before", np.nan))]
    bdps_after_vals = [b["score_after"] for b in bdps_list
                       if np.isfinite(b.get("score_after", np.nan))]
    bdps_before_mean = float(np.mean(bdps_before_vals)) if bdps_before_vals else float("nan")
    bdps_after_mean = float(np.mean(bdps_after_vals)) if bdps_after_vals else float("nan")
    if (np.isfinite(bdps_before_mean) and np.isfinite(bdps_after_mean)
            and bdps_before_mean > 1e-9):
        bdps_pct = float((bdps_after_mean - bdps_before_mean) / bdps_before_mean * 100.0)
        bdps_imp = float(bdps_after_mean - bdps_before_mean)
    else:
        bdps_pct = float("nan")
        bdps_imp = float("nan")

    result = {
        "sequence_name": sequence_name,
        "n_frames_evaluated": n_evaluated,
        "n_frames_failed": n_failed,
        "n_frames_with_gt": n_gt_hits,
        "gt_available": bool(gt_available_seq and n_gt_hits > 0),
        "metrics_uncorrected": metrics_uncorr_mean,
        "metrics_corrected": metrics_corr_mean,
        "metrics_improvement": metrics_improvement,
        "boundary_depth_score": {
            "before": bdps_before_mean,
            "after": bdps_after_mean,
            "improvement": bdps_imp,
            "improvement_pct": bdps_pct,
        },
        "per_frame": per_frame,
    }
    return result


# =======================================================================
# PART D: Results & Output
# =======================================================================

def print_depth_results(result: dict) -> None:
    """Print a formatted per-sequence results table."""
    name = result.get("sequence_name", "<unknown>")
    n = result.get("n_frames_evaluated", 0)
    print(f"\n=== Sequence: {name} ({n} frames) ===")

    if result.get("gt_available", False):
        u = result["metrics_uncorrected"]
        c = result["metrics_corrected"]
        p = result["metrics_improvement"]

        rows = [
            ("RMSE (m)         ", u["rmse"], c["rmse"], p["rmse_pct"]),
            ("MAE  (m)         ", u["mae"], c["mae"], p["mae_pct"]),
            ("iRMSE (1/km)     ", u["irmse"], c["irmse"], p["irmse_pct"]),
            ("iMAE  (1/km)     ", u["imae"], c["imae"], p["imae_pct"]),
            ("AbsRel           ", u["absrel"], c["absrel"], p["absrel_pct"]),
        ]
        print("+-----------------+------------+------------+------------+")
        print("| Metric (lower v)| Uncorr.    | Corrected  | Improv. %  |")
        print("+-----------------+------------+------------+------------+")
        for label, ub, cb, pct in rows:
            ub_s = f"{ub:>9.3f}" if np.isfinite(ub) else "      NaN"
            cb_s = f"{cb:>9.3f}" if np.isfinite(cb) else "      NaN"
            pct_s = f"{pct:+8.2f}%" if np.isfinite(pct) else "     NaN"
            print(f"| {label}|   {ub_s} |   {cb_s} |  {pct_s} |")
        print("+-----------------+------------+------------+------------+")
    else:
        print("(no GT available — BDPS only)")

    bdps = result["boundary_depth_score"]
    b_before = bdps["before"]
    b_after = bdps["after"]
    b_pct = bdps["improvement_pct"]
    print("Boundary Depth Projection Score (BDPS, higher = sharper edges):")
    if np.isfinite(b_before) and np.isfinite(b_after):
        pct_s = f"{b_pct:+.2f}%" if np.isfinite(b_pct) else "NaN"
        print(f"  Before: {b_before:.3f}  After: {b_after:.3f}  Improvement: {pct_s}")
    else:
        print("  insufficient data")


def _emit_latex_table(results: list, out_path: str) -> None:
    """Render a booktabs-style LaTeX results table."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Depth completion (IP-Basic) before vs. after event-guided correction.}")
    lines.append(r"  \label{tab:depth-completion}")
    lines.append(r"  \small")
    lines.append(r"  \begin{tabular}{lrrrrrr}")
    lines.append(r"    \toprule")
    lines.append(r"    Sequence & Frames & RMSE$\downarrow$ & RMSE$_c\downarrow$ & $\Delta$RMSE\% & BDPS$_{\text{before}}\uparrow$ & BDPS$_{\text{after}}\uparrow$ & $\Delta$BDPS\% \\")
    lines.append(r"    \midrule")
    for r in results:
        u = r.get("metrics_uncorrected", {})
        c = r.get("metrics_corrected", {})
        p = r.get("metrics_improvement", {})
        bdps = r.get("boundary_depth_score", {})

        def fmt(x, prec=3):
            return f"{x:.{prec}f}" if isinstance(x, (int, float)) and np.isfinite(x) else "--"

        seq_label = r["sequence_name"].replace("_", "\\_")
        n_frames = r["n_frames_evaluated"]
        rmse_b = fmt(u.get("rmse"))
        rmse_a = fmt(c.get("rmse"))
        rmse_pct = fmt(p.get("rmse_pct"), 2)
        bdps_b = fmt(bdps.get("before"))
        bdps_a = fmt(bdps.get("after"))
        bdps_pct = fmt(bdps.get("improvement_pct"), 2)
        lines.append(
            f"    {seq_label} & {n_frames} & {rmse_b} & {rmse_a} & {rmse_pct} & "
            f"{bdps_b} & {bdps_a} & {bdps_pct} \\\\"
        )
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    latex = "\n".join(lines)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex + "\n")
    print("\n--- LaTeX table snippet ---")
    print(latex)
    print("--- end LaTeX ---")


def save_depth_results(results: list, output_dir: str) -> None:
    """Persist the results as JSON + CSV and emit a LaTeX table."""
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "depth_completion_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))
    print(f"\nSaved JSON: {json_path}")

    csv_path = os.path.join(output_dir, "depth_completion_results.csv")
    fields = [
        "sequence", "n_frames", "gt_available",
        "rmse_before", "rmse_after", "rmse_improvement_pct",
        "mae_before", "mae_after", "mae_improvement_pct",
        "bdps_before", "bdps_after", "bdps_improvement_pct",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            u = r.get("metrics_uncorrected", {})
            c = r.get("metrics_corrected", {})
            p = r.get("metrics_improvement", {})
            bdps = r.get("boundary_depth_score", {})
            writer.writerow({
                "sequence": r["sequence_name"],
                "n_frames": r["n_frames_evaluated"],
                "gt_available": r.get("gt_available", False),
                "rmse_before": u.get("rmse", float("nan")),
                "rmse_after": c.get("rmse", float("nan")),
                "rmse_improvement_pct": p.get("rmse_pct", float("nan")),
                "mae_before": u.get("mae", float("nan")),
                "mae_after": c.get("mae", float("nan")),
                "mae_improvement_pct": p.get("mae_pct", float("nan")),
                "bdps_before": bdps.get("before", float("nan")),
                "bdps_after": bdps.get("after", float("nan")),
                "bdps_improvement_pct": bdps.get("improvement_pct", float("nan")),
            })
    print(f"Saved CSV:  {csv_path}")

    latex_path = os.path.join(output_dir, "depth_completion_table.tex")
    _emit_latex_table(results, latex_path)
    print(f"Saved LaTeX: {latex_path}")


# =======================================================================
# PART E: Main Entry Point
# =======================================================================

def _expand_sequence_name(raw: str) -> str:
    """
    Accept short ('0009_sync') or full ('2011_09_26_drive_0009_sync')
    sequence names; return the full KITTI directory name.
    """
    raw = raw.strip()
    if raw.startswith("2011_"):
        return raw
    return f"{KITTI_DATE}_drive_{raw}" if raw.endswith("_sync") else f"{KITTI_DATE}_drive_{raw}_sync"


def _build_gt_dir(gt_base_dir, seq_full_name) -> str:
    """Construct the KITTI depth completion GT folder path for a sequence."""
    return os.path.join(
        gt_base_dir, KITTI_DATE, seq_full_name,
        "proj_depth", "groundtruth", "image_02",
    )


def main():
    parser = argparse.ArgumentParser(
        description="IP-Basic depth completion eval — Phase 3 downstream task."
    )
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR,
                        help="KITTI fusion dataset base directory.")
    parser.add_argument("--sequences", default=DEFAULT_SEQUENCES,
                        help="Comma-separated sequence names (short or full) or 'all'.")
    parser.add_argument("--gt-base-dir", default=None,
                        help="KITTI depth completion GT root (optional). "
                             "If omitted, runs self-consistency only.")
    parser.add_argument("--max-frames", type=int, default=50,
                        help="Frame pairs per sequence to evaluate.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Where to save results.")
    parser.add_argument("--event-threshold", type=float, default=0.2,
                        help="Threshold for simulate_events.")
    args = parser.parse_args()

    if args.sequences.strip().lower() == "all":
        seq_list = [
            "0009_sync", "0005_sync", "0051_sync", "0117_sync",
        ]
    else:
        seq_list = [s for s in args.sequences.split(",") if s.strip()]

    seq_list = [_expand_sequence_name(s) for s in seq_list]

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("Phase 3: Depth Completion Evaluation (IP-Basic)")
    print("=" * 70)
    print(f"Base dir:       {args.base_dir}")
    print(f"GT base dir:    {args.gt_base_dir or '<none — self-consistency only>'}")
    print(f"Sequences:      {seq_list}")
    print(f"Max frames:     {args.max_frames}")
    print(f"Event threshold:{args.event_threshold}")
    print(f"Output dir:     {args.output_dir}")

    results = []
    for seq in seq_list:
        dataset_path = os.path.join(args.base_dir, seq)
        if not os.path.isdir(dataset_path):
            print(f"\n[SKIP] sequence not found on disk: {dataset_path}")
            continue

        gt_dir = None
        if args.gt_base_dir:
            candidate = _build_gt_dir(args.gt_base_dir, seq)
            gt_dir = candidate if os.path.isdir(candidate) else None
            if gt_dir is None:
                print(f"[{seq}] GT not found at {candidate} — self-consistency only.")

        try:
            result = evaluate_sequence(
                dataset_path=dataset_path,
                gt_depth_dir=gt_dir,
                max_frames=args.max_frames,
                event_threshold=args.event_threshold,
                output_dir=args.output_dir,
            )
        except Exception as exc:
            print(f"[{seq}] FAILED: {type(exc).__name__}: {exc}")
            continue

        print_depth_results(result)
        results.append(result)

    if not results:
        print("\nNo sequences produced results.")
        return

    save_depth_results(results, args.output_dir)

    # Aggregate summary across sequences.
    print("\n" + "=" * 70)
    print("AGGREGATE SUMMARY")
    print("=" * 70)

    gt_results = [r for r in results if r.get("gt_available", False)]
    if gt_results:
        rmse_b = float(np.mean([r["metrics_uncorrected"]["rmse"] for r in gt_results
                                if np.isfinite(r["metrics_uncorrected"]["rmse"])]))
        rmse_a = float(np.mean([r["metrics_corrected"]["rmse"] for r in gt_results
                                if np.isfinite(r["metrics_corrected"]["rmse"])]))
        mae_b = float(np.mean([r["metrics_uncorrected"]["mae"] for r in gt_results
                               if np.isfinite(r["metrics_uncorrected"]["mae"])]))
        mae_a = float(np.mean([r["metrics_corrected"]["mae"] for r in gt_results
                               if np.isfinite(r["metrics_corrected"]["mae"])]))
        print(f"GT-based mean RMSE:  before={rmse_b:.3f}m  after={rmse_a:.3f}m  "
              f"({_safe_pct(rmse_b, rmse_a):+.2f}%)")
        print(f"GT-based mean MAE:   before={mae_b:.3f}m  after={mae_a:.3f}m  "
              f"({_safe_pct(mae_b, mae_a):+.2f}%)")
    else:
        print("(no GT-based results available across any sequence)")

    bdps_better = sum(
        1 for r in results
        if np.isfinite(r["boundary_depth_score"].get("improvement_pct", np.nan))
        and r["boundary_depth_score"]["improvement_pct"] > 0.0
    )
    n_total = len(results)
    print()
    print("**BDPS (no GT required, higher = sharper depth boundaries):**")
    print(f"  corrected projection has sharper depth boundaries than "
          f"uncorrected in {bdps_better}/{n_total} sequences")
    bdps_b_all = [r["boundary_depth_score"]["before"] for r in results
                  if np.isfinite(r["boundary_depth_score"].get("before", np.nan))]
    bdps_a_all = [r["boundary_depth_score"]["after"] for r in results
                  if np.isfinite(r["boundary_depth_score"].get("after", np.nan))]
    if bdps_b_all and bdps_a_all:
        bdps_b_mean = float(np.mean(bdps_b_all))
        bdps_a_mean = float(np.mean(bdps_a_all))
        if bdps_b_mean > 1e-9:
            pct = (bdps_a_mean - bdps_b_mean) / bdps_b_mean * 100.0
            print(f"  mean BDPS: before={bdps_b_mean:.3f}  after={bdps_a_mean:.3f}  "
                  f"({pct:+.2f}%)")
        else:
            print(f"  mean BDPS: before={bdps_b_mean:.3f}  after={bdps_a_mean:.3f}")


if __name__ == "__main__":
    main()
