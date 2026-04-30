"""
Independent edge-based metrics for LiDAR-camera projection quality.

The Edge Alignment Score (EAS) measures how well LiDAR depth
discontinuities coincide with RGB intensity edges. Because it consumes
only the projected LiDAR points (uv, depth) and the RGB image, it
contains no dependency on optical flow and is therefore independent
of the correction signal used by the pipeline.

Higher score => better alignment. The score is dimensionless and lies
in [0, 1] (in practice well below 1 because both maps are sparse).
"""

from typing import Dict, List, Tuple

import cv2
import numpy as np


def compute_depth_edge_map(
    uv: np.ndarray,
    depth: np.ndarray,
    image_shape: Tuple[int, ...],
    sigma: float = 1.0,
) -> np.ndarray:
    """Rasterize projected LiDAR depth into an image and extract its edges.

    The function builds a sparse depth image at the integer pixel
    locations of the projected points (keeping the nearest depth on
    collisions), fills small holes with a Gaussian blur, and returns
    the normalized magnitude of the Sobel gradient. Pixels that
    received no LiDAR sample produce no gradient response.

    Args:
        uv: (N, 2) float array of projected pixel coordinates (u, v),
            in pixels. Non-finite rows are dropped.
        depth: (N,) float array of camera-frame Z depths in meters,
            corresponding row-wise to ``uv``. Non-finite values are
            dropped.
        image_shape: shape tuple of the RGB image (H, W) or (H, W, C).
            Only the first two entries are used.
        sigma: Gaussian blur sigma in pixels. The kernel size is
            derived as 2 * ceil(3 * sigma) + 1. Must be > 0.

    Returns:
        depth_edge_map: (H, W) float32 array in [0, 1]. Zero where the
        gradient magnitude is zero (including pixels with no nearby
        LiDAR support).
    """
    assert uv.ndim == 2 and uv.shape[1] == 2, (
        f"uv must be (N, 2), got {uv.shape}"
    )
    assert depth.ndim == 1, f"depth must be (N,), got {depth.shape}"
    assert uv.shape[0] == depth.shape[0], (
        f"uv and depth length mismatch: {uv.shape[0]} vs {depth.shape[0]}"
    )
    assert len(image_shape) >= 2, (
        f"image_shape must have at least 2 dims, got {image_shape}"
    )
    assert sigma > 0.0, f"sigma must be > 0, got {sigma}"

    h, w = int(image_shape[0]), int(image_shape[1])
    depth_image = np.zeros((h, w), dtype=np.float32)

    if uv.shape[0] == 0:
        print("[EAS] Warning: empty LiDAR projection — depth edge map is zero.")
        return depth_image

    u = np.asarray(uv[:, 0], dtype=np.float64)
    v = np.asarray(uv[:, 1], dtype=np.float64)
    d = np.asarray(depth, dtype=np.float64)

    finite = np.isfinite(u) & np.isfinite(v) & np.isfinite(d)
    if not np.all(finite):
        n_bad = int(np.size(finite) - np.count_nonzero(finite))
        print(
            f"[EAS] Warning: dropping {n_bad} non-finite point(s) "
            "before depth rasterization."
        )
    u = u[finite]
    v = v[finite]
    d = d[finite]

    if u.size == 0:
        print("[EAS] Warning: no finite points to rasterize.")
        return depth_image

    u_idx = np.rint(u).astype(np.int64)
    v_idx = np.rint(v).astype(np.int64)
    in_bounds = (u_idx >= 0) & (u_idx < w) & (v_idx >= 0) & (v_idx < h)
    if not np.any(in_bounds):
        print("[EAS] Warning: all projected points fall outside the image.")
        return depth_image

    u_idx = u_idx[in_bounds]
    v_idx = v_idx[in_bounds]
    d = d[in_bounds]

    # Keep the nearest (minimum) depth at each pixel. Sort by depth
    # descending so later writes (smaller depths) win.
    order = np.argsort(-d, kind="stable")
    flat_idx = v_idx[order] * w + u_idx[order]
    depth_image.reshape(-1)[flat_idx] = d[order].astype(np.float32)

    ksize = 2 * int(np.ceil(3.0 * sigma)) + 1
    blurred = cv2.GaussianBlur(
        depth_image, (ksize, ksize), sigmaX=float(sigma), sigmaY=float(sigma)
    )

    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)

    max_val = float(grad_mag.max())
    if max_val <= 0.0:
        print(
            "[EAS] Warning: depth gradient magnitude is identically zero."
        )
        return np.zeros_like(grad_mag, dtype=np.float32)

    return (grad_mag / max_val).astype(np.float32)


def compute_rgb_edge_map(
    image_bgr: np.ndarray,
    low_thresh: int = 30,
    high_thresh: int = 90,
) -> np.ndarray:
    """Detect dilated, normalized RGB edges using Canny.

    The image is converted to grayscale, blurred with a 5x5 Gaussian
    to suppress noise, edges are extracted with Canny using the
    supplied thresholds, and the binary edge image is dilated with a
    3x3 kernel to provide a one-pixel matching tolerance.

    Args:
        image_bgr: (H, W, 3) uint8 BGR image as returned by cv2.imread.
        low_thresh: lower Canny hysteresis threshold (0-255).
        high_thresh: upper Canny hysteresis threshold (0-255).

    Returns:
        rgb_edge_map: (H, W) float32 array, values in {0.0, 1.0}.
    """
    assert image_bgr.ndim == 3 and image_bgr.shape[2] == 3, (
        f"image_bgr must be (H, W, 3), got {image_bgr.shape}"
    )
    assert image_bgr.dtype == np.uint8, (
        f"image_bgr must be uint8, got {image_bgr.dtype}"
    )
    assert 0 <= low_thresh <= 255, f"low_thresh out of range: {low_thresh}"
    assert 0 <= high_thresh <= 255, f"high_thresh out of range: {high_thresh}"
    assert low_thresh < high_thresh, (
        f"low_thresh ({low_thresh}) must be < high_thresh ({high_thresh})"
    )

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, low_thresh, high_thresh)
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    rgb_edge_map = (dilated > 0).astype(np.float32)
    if not np.any(rgb_edge_map):
        print(
            "[EAS] Warning: Canny produced no edges — check thresholds "
            f"(low={low_thresh}, high={high_thresh})."
        )
    return rgb_edge_map


def edge_alignment_score(
    uv: np.ndarray,
    depth: np.ndarray,
    image_bgr: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute the Edge Alignment Score for one projected frame.

    EAS = mean over all pixels of (depth_edge_map * rgb_edge_map).
    Higher is better. The score is independent of optical flow.

    Args:
        uv: (N, 2) float array of projected pixel coordinates.
        depth: (N,) float array of depths in meters.
        image_bgr: (H, W, 3) uint8 BGR image.

    Returns:
        score: scalar float in [0, 1] — mean of the Hadamard product
            of the two edge maps.
        depth_edge_map: (H, W) float32, normalized depth-gradient map.
        rgb_edge_map: (H, W) float32, dilated Canny edge map.
    """
    assert image_bgr.ndim == 3, f"image_bgr must be 3D, got {image_bgr.shape}"

    depth_edge_map = compute_depth_edge_map(uv, depth, image_bgr.shape)
    rgb_edge_map = compute_rgb_edge_map(image_bgr)

    assert depth_edge_map.shape == rgb_edge_map.shape, (
        f"edge map shape mismatch: depth={depth_edge_map.shape}, "
        f"rgb={rgb_edge_map.shape}"
    )

    hadamard = depth_edge_map * rgb_edge_map
    score = float(hadamard.mean())
    print(f"EAS: {score:.6f}")
    return score, depth_edge_map, rgb_edge_map


def compare_eas(
    uv_before: np.ndarray,
    depth_before: np.ndarray,
    uv_after: np.ndarray,
    depth_after: np.ndarray,
    image_bgr: np.ndarray,
) -> Dict[str, object]:
    """Compute EAS for an uncorrected and a corrected projection.

    The improvement is reported both in absolute units and as a
    percentage relative to the uncorrected score. When the uncorrected
    score is zero, ``improvement_pct`` is set to NaN.

    Args:
        uv_before: (N, 2) projected points before correction.
        depth_before: (N,) depths corresponding to ``uv_before``.
        uv_after: (M, 2) projected points after correction.
        depth_after: (M,) depths corresponding to ``uv_after``.
        image_bgr: (H, W, 3) uint8 reference image. The same image is
            used for both before and after, so the RGB edge map is
            identical for both terms; only the depth edge map differs.

    Returns:
        Dict with keys:
            - score_before: float
            - score_after:  float
            - improvement:  float, score_after - score_before
            - improvement_pct: float, percent change vs score_before
            - depth_edge_before: (H, W) float32
            - depth_edge_after:  (H, W) float32
            - rgb_edge_map:      (H, W) float32 (computed once)
    """
    rgb_edge_map = compute_rgb_edge_map(image_bgr)

    depth_edge_before = compute_depth_edge_map(
        uv_before, depth_before, image_bgr.shape
    )
    depth_edge_after = compute_depth_edge_map(
        uv_after, depth_after, image_bgr.shape
    )

    score_before = float((depth_edge_before * rgb_edge_map).mean())
    score_after = float((depth_edge_after * rgb_edge_map).mean())
    print(f"EAS: {score_before:.6f}")
    print(f"EAS: {score_after:.6f}")

    improvement = score_after - score_before
    if score_before > 0.0:
        improvement_pct = (improvement / score_before) * 100.0
    else:
        print(
            "[EAS] Warning: score_before is zero — improvement_pct undefined."
        )
        improvement_pct = float("nan")

    print(f"EAS Before: {score_before:.6f}")
    print(f"EAS After:  {score_after:.6f}")
    print(
        f"EAS Improvement: {improvement:+.6f} ({improvement_pct:+.2f}%)"
    )

    return {
        "score_before": score_before,
        "score_after": score_after,
        "improvement": improvement,
        "improvement_pct": improvement_pct,
        "depth_edge_before": depth_edge_before,
        "depth_edge_after": depth_edge_after,
        "rgb_edge_map": rgb_edge_map,
    }


def accumulate_eas_results(
    results_list: List[Dict[str, object]],
) -> Dict[str, float]:
    """Aggregate per-frame EAS results into mean and standard deviation.

    Frames where ``improvement_pct`` is NaN (typically because the
    uncorrected score was zero) are excluded from the percentage
    statistics but still contribute to the absolute-score statistics.

    Args:
        results_list: list of dicts produced by ``compare_eas``.

    Returns:
        Summary dict with keys:
            n_frames, n_pct_frames,
            score_before_mean, score_before_std,
            score_after_mean, score_after_std,
            improvement_pct_mean, improvement_pct_std.
    """
    assert isinstance(results_list, list), "results_list must be a list"

    if not results_list:
        print("[EAS] Warning: empty results_list — nothing to aggregate.")
        return {
            "n_frames": 0,
            "n_pct_frames": 0,
            "score_before_mean": float("nan"),
            "score_before_std": float("nan"),
            "score_after_mean": float("nan"),
            "score_after_std": float("nan"),
            "improvement_pct_mean": float("nan"),
            "improvement_pct_std": float("nan"),
        }

    score_before = np.array(
        [r["score_before"] for r in results_list], dtype=np.float64
    )
    score_after = np.array(
        [r["score_after"] for r in results_list], dtype=np.float64
    )
    improvement_pct = np.array(
        [r["improvement_pct"] for r in results_list], dtype=np.float64
    )
    pct_finite = np.isfinite(improvement_pct)

    summary = {
        "n_frames": int(len(results_list)),
        "n_pct_frames": int(np.count_nonzero(pct_finite)),
        "score_before_mean": float(np.mean(score_before)),
        "score_before_std": float(np.std(score_before)),
        "score_after_mean": float(np.mean(score_after)),
        "score_after_std": float(np.std(score_after)),
        "improvement_pct_mean": (
            float(np.mean(improvement_pct[pct_finite]))
            if np.any(pct_finite)
            else float("nan")
        ),
        "improvement_pct_std": (
            float(np.std(improvement_pct[pct_finite]))
            if np.any(pct_finite)
            else float("nan")
        ),
    }

    print("=" * 56)
    print("EAS SUMMARY")
    print("-" * 56)
    print(f"Frames                : {summary['n_frames']}")
    print(
        "Score Before          : "
        f"{summary['score_before_mean']:.6f} "
        f"+/- {summary['score_before_std']:.6f}"
    )
    print(
        "Score After           : "
        f"{summary['score_after_mean']:.6f} "
        f"+/- {summary['score_after_std']:.6f}"
    )
    print(
        "Improvement (%)       : "
        f"{summary['improvement_pct_mean']:+.2f} "
        f"+/- {summary['improvement_pct_std']:.2f} "
        f"(n={summary['n_pct_frames']})"
    )
    print("=" * 56)

    return summary
