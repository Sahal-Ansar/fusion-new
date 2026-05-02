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


def sparse_point_eas(
    uv: np.ndarray,
    depth: np.ndarray,
    image_bgr: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """Sparse-Point EAS: mean RGB edge strength sampled at LiDAR points.

    Unlike ``edge_alignment_score``, which averages the depth-edge x
    RGB-edge product over every pixel of the image, this metric samples
    the RGB edge map only at the projected LiDAR point locations. With
    ~3% LiDAR coverage on KITTI, the dense formulation is dominated by
    the 97% of pixels that contain no LiDAR signal, so artifacts from
    sparse depth rasterization (e.g. point displacement smearing edges
    across blank pixels) systematically drag the score down. Sampling
    only at the points sidesteps that confound entirely.

    Args:
        uv: (N, 2) float32 projected pixel coordinates (u, v).
        depth: (N,) float32 — kept for API consistency; not used.
        image_bgr: (H, W, 3) uint8 BGR reference image.

    Returns:
        score: scalar float in [0, 1] — mean RGB edge strength at the
            sampled point locations.
        edge_strengths: (N,) float32 — per-point sampled edge values.
    """
    rgb_edge_map = compute_rgb_edge_map(image_bgr)
    h, w = rgb_edge_map.shape[:2]

    u_idx = np.clip(np.rint(uv[:, 0]).astype(np.int64), 0, w - 1)
    v_idx = np.clip(np.rint(uv[:, 1]).astype(np.int64), 0, h - 1)
    edge_strengths = rgb_edge_map[v_idx, u_idx]

    score = float(np.mean(edge_strengths))
    print(f"SPEAS: {score:.6f} ({len(uv)} points sampled)")
    _ = depth
    return score, edge_strengths


def compare_speas(
    uv_before: np.ndarray,
    depth_before: np.ndarray,
    uv_after: np.ndarray,
    depth_after: np.ndarray,
    image_bgr: np.ndarray,
) -> Dict[str, object]:
    """Compare Sparse-Point EAS before and after correction.

    Args:
        uv_before: (N, 2) uncorrected projected coordinates.
        depth_before: (N,) depths corresponding to ``uv_before``.
        uv_after: (M, 2) corrected projected coordinates.
        depth_after: (M,) depths corresponding to ``uv_after``.
        image_bgr: (H, W, 3) uint8 reference image used for both terms.

    Returns:
        Dict with keys ``score_before``, ``score_after``,
        ``improvement``, ``improvement_pct``.
    """
    score_before, _ = sparse_point_eas(uv_before, depth_before, image_bgr)
    score_after, _ = sparse_point_eas(uv_after, depth_after, image_bgr)

    improvement = score_after - score_before
    if score_before > 0.0:
        improvement_pct = (improvement / score_before) * 100.0
    else:
        improvement_pct = 0.0

    print(f"SPEAS Before: {score_before:.6f}")
    print(f"SPEAS After:  {score_after:.6f}")
    print(
        f"SPEAS Improvement: {improvement:+.6f} ({improvement_pct:+.2f}%)"
    )

    return {
        "score_before": score_before,
        "score_after": score_after,
        "improvement": improvement,
        "improvement_pct": improvement_pct,
    }


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


# =====================================================================
#                    STEREO REPROJECTION CONSISTENCY
# ---------------------------------------------------------------------
# A second flow-independent metric. After temporal correction the 3-D
# positions of the LiDAR points have not changed, only their image
# coordinates. So if the corrected 2-D positions are more accurate, the
# expected stereo reprojection (left -> right via known stereo geometry)
# should land more often on RGB edges in the right image.
# =====================================================================


# Physically plausible depth window for KITTI passenger-car LiDAR.
# Points outside this range are excluded from disparity computation.
_STEREO_DEPTH_MIN_M = 0.1
_STEREO_DEPTH_MAX_M = 80.0


def compute_stereo_reprojection(
    uv_left: np.ndarray,
    depth: np.ndarray,
    p_rect_left: np.ndarray,
    p_rect_right: np.ndarray,
    image_shape: Tuple[int, ...],
) -> Tuple[np.ndarray, np.ndarray]:
    """Project left-camera LiDAR points into the right rectified image.

    For rectified KITTI stereo the right-image position of a 3-D point
    that projects to ``(u_L, v_L)`` with depth ``Z`` is given by
    horizontal disparity::

        disparity = f * B / Z
        u_R       = u_L - disparity
        v_R       = v_L      # rows are aligned in rectified stereo

    where ``f = p_rect_left[0, 0]`` and ``B = |p_rect_right[0, 3] /
    p_rect_right[0, 0]|`` is the stereo baseline in metres.

    Args:
        uv_left: (N, 2) float32, left-image pixel coordinates (u, v).
        depth: (N,) float32, camera-Z depth in metres, row-aligned with
            ``uv_left``.
        p_rect_left: (3, 4) float64 left-camera projection matrix.
        p_rect_right: (3, 4) float64 right-camera projection matrix.
        image_shape: (H, W, ...) shape of the right image (or any
            shape sharing the same width/height after rectification).

    Returns:
        uv_right: (M, 2) float32, valid right-image pixel coordinates
            for the subset of input points that produced an in-bounds
            disparity (M <= N).
        valid_mask: (N,) bool, ``True`` where the corresponding input
            point produced a valid right-image reprojection.
    """
    assert uv_left.ndim == 2 and uv_left.shape[1] == 2, (
        f"uv_left must be (N, 2), got {uv_left.shape}"
    )
    assert depth.ndim == 1, f"depth must be (N,), got {depth.shape}"
    assert uv_left.shape[0] == depth.shape[0], (
        f"uv_left and depth length mismatch: "
        f"{uv_left.shape[0]} vs {depth.shape[0]}"
    )
    assert np.asarray(p_rect_left).shape == (3, 4), (
        f"p_rect_left must be (3, 4), got {np.asarray(p_rect_left).shape}"
    )
    assert np.asarray(p_rect_right).shape == (3, 4), (
        f"p_rect_right must be (3, 4), got {np.asarray(p_rect_right).shape}"
    )
    assert len(image_shape) >= 2, (
        f"image_shape must have at least 2 dims, got {image_shape}"
    )

    n_input = int(uv_left.shape[0])
    valid_mask = np.zeros((n_input,), dtype=bool)
    if n_input == 0:
        print("[Stereo] Warning: empty input — returning zero reprojections.")
        return np.empty((0, 2), dtype=np.float32), valid_mask

    p_left = np.asarray(p_rect_left, dtype=np.float64)
    p_right = np.asarray(p_rect_right, dtype=np.float64)

    f = float(p_left[0, 0])
    p_right_fx = float(p_right[0, 0])
    if not np.isfinite(f) or f <= 0.0:
        print(f"[Stereo] Warning: invalid focal length f={f}; aborting.")
        return np.empty((0, 2), dtype=np.float32), valid_mask
    if not np.isfinite(p_right_fx) or abs(p_right_fx) < 1e-9:
        print(
            "[Stereo] Warning: invalid right focal length "
            f"p_rect_right[0,0]={p_right_fx}; aborting."
        )
        return np.empty((0, 2), dtype=np.float32), valid_mask

    baseline_m = abs(float(p_right[0, 3]) / p_right_fx)
    if not np.isfinite(baseline_m) or baseline_m <= 0.0:
        print(
            f"[Stereo] Warning: invalid baseline B={baseline_m} m; aborting."
        )
        return np.empty((0, 2), dtype=np.float32), valid_mask

    h, w = int(image_shape[0]), int(image_shape[1])
    u_left = np.asarray(uv_left[:, 0], dtype=np.float64)
    v_left = np.asarray(uv_left[:, 1], dtype=np.float64)
    z = np.asarray(depth, dtype=np.float64)

    finite = (
        np.isfinite(u_left) & np.isfinite(v_left) & np.isfinite(z)
    )
    depth_ok = (z >= _STEREO_DEPTH_MIN_M) & (z <= _STEREO_DEPTH_MAX_M)
    safe = finite & depth_ok

    if not np.any(safe):
        print(
            "[Stereo] Warning: no points within physically plausible "
            f"depth range [{_STEREO_DEPTH_MIN_M}, {_STEREO_DEPTH_MAX_M}] m."
        )
        return np.empty((0, 2), dtype=np.float32), valid_mask

    disparity = np.zeros_like(z)
    disparity[safe] = (f * baseline_m) / z[safe]

    # Disparities exceeding image width are unphysical (would push the
    # right-image point off-screen by more than the entire frame).
    disparity_ok = safe & (disparity < float(w)) & (disparity >= 0.0)

    u_right = u_left - disparity
    v_right = v_left

    in_bounds = (
        disparity_ok
        & (u_right >= 0.0)
        & (u_right < float(w))
        & (v_right >= 0.0)
        & (v_right < float(h))
    )

    valid_mask = in_bounds
    uv_right = np.column_stack(
        (u_right[in_bounds], v_right[in_bounds])
    ).astype(np.float32)

    n_dropped = int(n_input - int(np.count_nonzero(valid_mask)))
    if n_dropped > 0:
        print(
            f"[Stereo] Dropped {n_dropped}/{n_input} points "
            "(depth out of range, invalid disparity, or off the right image)."
        )

    return uv_right, valid_mask


def stereo_consistency_score(
    uv_left: np.ndarray,
    depth: np.ndarray,
    image_right_bgr: np.ndarray,
    p_rect_left: np.ndarray,
    p_rect_right: np.ndarray,
) -> Tuple[float, np.ndarray, int]:
    """Score how often left->right reprojections land on right-image edges.

    Each LiDAR point projected to the left image is mapped into the
    right rectified image using known stereo geometry (no optical flow
    involved). The right RGB image's dilated Canny edge map is sampled
    at each reprojected location with nearest-neighbour rounding. The
    mean of those samples is the score; higher means the reprojected
    points coincide more often with intensity edges in the right view.

    Args:
        uv_left: (N, 2) float32 left-image pixel coordinates.
        depth: (N,) float32 camera-Z depths in metres, row-aligned.
        image_right_bgr: (H, W, 3) uint8 BGR right-camera image.
        p_rect_left: (3, 4) left projection matrix.
        p_rect_right: (3, 4) right projection matrix.

    Returns:
        score: scalar float in [0, 1] — mean right-image edge strength
            sampled at the valid reprojected points. Returns 0.0 when
            no points reproject into the right image.
        uv_right: (M, 2) float32 right-image coordinates of the M
            valid reprojections.
        n_valid: int, number of valid reprojected points sampled.
    """
    assert image_right_bgr.ndim == 3 and image_right_bgr.shape[2] == 3, (
        f"image_right_bgr must be (H, W, 3), got {image_right_bgr.shape}"
    )

    uv_right, valid_mask = compute_stereo_reprojection(
        uv_left, depth, p_rect_left, p_rect_right, image_right_bgr.shape
    )

    n_valid = int(uv_right.shape[0])
    if n_valid == 0:
        print("Stereo Consistency Score: 0.000000 (0 valid points)")
        return 0.0, uv_right, n_valid

    rgb_edge_map = compute_rgb_edge_map(image_right_bgr)
    h, w = rgb_edge_map.shape[:2]

    u_idx = np.clip(np.rint(uv_right[:, 0]).astype(np.int64), 0, w - 1)
    v_idx = np.clip(np.rint(uv_right[:, 1]).astype(np.int64), 0, h - 1)
    samples = rgb_edge_map[v_idx, u_idx]

    score = float(samples.mean())
    print(
        f"Stereo Consistency Score: {score:.6f} ({n_valid} valid points)"
    )
    _ = valid_mask  # kept for future callers that need per-input membership
    return score, uv_right, n_valid


def compare_stereo_consistency(
    uv_before: np.ndarray,
    depth_before: np.ndarray,
    uv_after: np.ndarray,
    depth_after: np.ndarray,
    image_right_bgr: np.ndarray,
    p_rect_left: np.ndarray,
    p_rect_right: np.ndarray,
) -> Dict[str, object]:
    """Compare stereo-consistency scores before and after correction.

    Args:
        uv_before: (N, 2) uncorrected left-image coordinates.
        depth_before: (N,) depths corresponding to ``uv_before``.
        uv_after: (M, 2) corrected left-image coordinates.
        depth_after: (M,) depths corresponding to ``uv_after``.
        image_right_bgr: (H, W, 3) uint8 BGR right-camera image used
            as the edge reference for both terms.
        p_rect_left: (3, 4) left projection matrix.
        p_rect_right: (3, 4) right projection matrix.

    Returns:
        Dict with keys:
            - score_before: float
            - score_after:  float
            - improvement:  float, score_after - score_before
            - improvement_pct: float, percent change vs score_before
              (NaN if score_before == 0)
            - uv_right_before: (M_b, 2) float32
            - uv_right_after:  (M_a, 2) float32
            - n_valid_before:  int
            - n_valid_after:   int
    """
    score_before, uv_right_before, n_valid_before = stereo_consistency_score(
        uv_before, depth_before, image_right_bgr, p_rect_left, p_rect_right
    )
    score_after, uv_right_after, n_valid_after = stereo_consistency_score(
        uv_after, depth_after, image_right_bgr, p_rect_left, p_rect_right
    )

    improvement = score_after - score_before
    if score_before > 0.0:
        improvement_pct = (improvement / score_before) * 100.0
    else:
        print(
            "[Stereo] Warning: score_before is zero — "
            "improvement_pct undefined."
        )
        improvement_pct = float("nan")

    print(f"Stereo Before: {score_before:.6f}")
    print(f"Stereo After:  {score_after:.6f}")
    print(
        f"Stereo Improvement: {improvement:+.6f} "
        f"({improvement_pct:+.2f}%)"
    )

    return {
        "score_before": score_before,
        "score_after": score_after,
        "improvement": improvement,
        "improvement_pct": improvement_pct,
        "uv_right_before": uv_right_before,
        "uv_right_after": uv_right_after,
        "n_valid_before": n_valid_before,
        "n_valid_after": n_valid_after,
    }


# =====================================================================
#                  DEPTH GRADIENT CORRELATION (DGC)
# ---------------------------------------------------------------------
# A third flow-independent metric. Densifies the projected LiDAR depth
# into a dense image, computes gradient orientations on both the depth
# image and the RGB image, and measures their circular correlation
# (mean cosine of orientation difference) over pixels with strong
# gradients on both modalities. Higher = LiDAR depth edges share
# orientation with RGB intensity edges, independent of optical flow.
# =====================================================================


def compute_dense_depth_image(
    uv: np.ndarray,
    depth: np.ndarray,
    image_shape: Tuple[int, ...],
    fill_radius: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rasterize projected LiDAR points to a dense depth image.

    For each pixel that received one or more LiDAR projections, the
    smallest (nearest) depth is retained. The float32 depth image is
    then dilated with a square structuring element of side
    ``2 * fill_radius + 1`` to bridge small holes in the sparse
    coverage; ``valid_mask`` is recomputed from the dilated image.

    Args:
        uv: (N, 2) float32 projected pixel coordinates (u, v).
        depth: (N,) float32 camera-frame Z depths in metres,
            row-aligned with ``uv``. Non-finite or non-positive depths
            are dropped.
        image_shape: shape tuple of the target image; only the first
            two entries (H, W) are used.
        fill_radius: half-side of the square dilation kernel in pixels
            (must be >= 0). 0 disables dilation.

    Returns:
        depth_image: (H, W) float32 — depth in metres, 0.0 where no
            LiDAR coverage even after dilation.
        valid_mask:  (H, W) bool  — True where ``depth_image > 0``.
    """
    assert uv.ndim == 2 and uv.shape[1] == 2, (
        f"uv must be (N, 2), got {uv.shape}"
    )
    assert depth.ndim == 1 and depth.shape[0] == uv.shape[0], (
        f"depth must be (N,) matching uv; "
        f"got depth {depth.shape}, uv {uv.shape}"
    )
    assert len(image_shape) >= 2, (
        f"image_shape must have at least 2 dims, got {image_shape}"
    )
    assert int(fill_radius) >= 0, (
        f"fill_radius must be >= 0, got {fill_radius}"
    )

    h, w = int(image_shape[0]), int(image_shape[1])
    depth_image = np.zeros((h, w), dtype=np.float32)

    if uv.shape[0] == 0:
        return depth_image, np.zeros((h, w), dtype=bool)

    u = np.asarray(uv[:, 0], dtype=np.float64)
    v = np.asarray(uv[:, 1], dtype=np.float64)
    d = np.asarray(depth, dtype=np.float64)

    finite = (
        np.isfinite(u) & np.isfinite(v) & np.isfinite(d) & (d > 0.0)
    )
    u = u[finite]
    v = v[finite]
    d = d[finite]
    if u.size == 0:
        return depth_image, np.zeros((h, w), dtype=bool)

    u_idx = np.clip(np.rint(u).astype(np.int64), 0, w - 1)
    v_idx = np.clip(np.rint(v).astype(np.int64), 0, h - 1)

    # Sort by depth descending so the LAST write at each pixel is the
    # smallest (nearest) depth — i.e. min-on-collision without a loop.
    order = np.argsort(-d, kind="stable")
    flat_idx = v_idx[order] * w + u_idx[order]
    depth_image.reshape(-1)[flat_idx] = d[order].astype(np.float32)

    fr = int(fill_radius)
    if fr > 0:
        ksize = 2 * fr + 1
        kernel = np.ones((ksize, ksize), dtype=np.uint8)
        depth_image = cv2.dilate(depth_image, kernel, iterations=1)

    valid_mask = depth_image > 0.0
    return depth_image.astype(np.float32), valid_mask


def compute_gradient_orientation(
    image: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-pixel gradient orientation and magnitude via Sobel.

    Args:
        image: (H, W) array, float32 or uint8. Non-2D inputs raise
            AssertionError.

    Returns:
        orientation: (H, W) float32, orientation angle in radians on
            [-pi, pi]. NaN/inf cells are replaced with 0.0.
        magnitude:   (H, W) float32, gradient magnitude. NaN/inf cells
            are replaced with 0.0.
    """
    assert image.ndim == 2, f"image must be 2D, got shape {image.shape}"

    img32 = image.astype(np.float32) if image.dtype != np.float32 else image
    gx = cv2.Sobel(img32, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img32, cv2.CV_32F, 0, 1, ksize=3)

    orientation = np.arctan2(gy, gx).astype(np.float32)
    magnitude = np.sqrt(gx * gx + gy * gy).astype(np.float32)

    orientation = np.where(
        np.isfinite(orientation), orientation, 0.0
    ).astype(np.float32)
    magnitude = np.where(
        np.isfinite(magnitude), magnitude, 0.0
    ).astype(np.float32)
    return orientation, magnitude


def depth_gradient_correlation(
    uv: np.ndarray,
    depth: np.ndarray,
    image_bgr: np.ndarray,
    min_depth_mag: float = 0.5,
    min_rgb_mag: float = 5.0,
):
    """Circular correlation of LiDAR-depth and RGB gradient orientations.

    Densifies the projected LiDAR depth, takes Sobel gradients on both
    the depth image and the RGB grayscale image, restricts to pixels
    where both modalities show meaningful gradient magnitude, and
    returns the mean cosine of the orientation difference.
    Interpretation:

        +1.0  perfect alignment (parallel gradients)
         0.0  random / decorrelated
        -1.0  anti-aligned

    Args:
        uv: (N, 2) projected pixel coordinates.
        depth: (N,) camera-Z depths in metres, row-aligned.
        image_bgr: (H, W, 3) uint8 BGR image.
        min_depth_mag: minimum depth-gradient magnitude (m / px) to
            include a pixel in the evaluation mask.
        min_rgb_mag: minimum RGB-gradient magnitude (intensity / px)
            to include a pixel in the evaluation mask.

    Returns:
        correlation: float in [-1, 1], or None if fewer than 100
            evaluation pixels were found.
        n_pixels: int — number of pixels in the evaluation mask.
        eval_mask: (H, W) bool, or None if correlation is None.
    """
    assert image_bgr.ndim == 3 and image_bgr.shape[2] == 3, (
        f"image_bgr must be (H, W, 3), got {image_bgr.shape}"
    )

    depth_image, valid_mask = compute_dense_depth_image(
        uv, depth, image_bgr.shape
    )
    depth_orient, depth_mag = compute_gradient_orientation(depth_image)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    rgb_orient, rgb_mag = compute_gradient_orientation(gray)

    eval_mask = (
        valid_mask
        & (depth_mag > float(min_depth_mag))
        & (rgb_mag > float(min_rgb_mag))
    )
    n_pixels = int(np.sum(eval_mask))

    if n_pixels < 100:
        print(
            f"[DGC] Warning: only {n_pixels} evaluation pixels "
            "(< 100); returning None."
        )
        return None, n_pixels, None

    delta = depth_orient[eval_mask] - rgb_orient[eval_mask]
    delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
    correlation = float(np.mean(np.cos(delta)))

    print(f"DGC: {correlation:.6f} ({n_pixels} evaluation pixels)")
    return correlation, n_pixels, eval_mask


def compare_dgc(
    uv_before: np.ndarray,
    depth_before: np.ndarray,
    uv_after: np.ndarray,
    depth_after: np.ndarray,
    image_bgr: np.ndarray,
) -> Dict[str, object]:
    """Compare depth gradient correlation before and after correction.

    The improvement percentage is reported relative to the gap from
    perfect alignment (1.0):

        improvement_pct = (gap_before - gap_after) / gap_before * 100
        gap_before = 1.0 - corr_before
        gap_after  = 1.0 - corr_after

    so a positive value means the corrected projection closed the
    distance to perfect orientation alignment. ``improvement_pct``
    is set to None when ``gap_before == 0`` (already perfect) or
    when either correlation is None (insufficient pixels). The
    function never raises — any exception is caught and the
    invalid-result dict is returned.

    Returns:
        Dict with keys ``corr_before``, ``corr_after``, ``improvement``,
        ``improvement_pct``, ``n_pixels_before``, ``n_pixels_after``,
        ``valid``. When ``valid`` is False all numeric fields are None.
    """
    invalid = {
        "corr_before": None,
        "corr_after": None,
        "improvement": None,
        "improvement_pct": None,
        "n_pixels_before": None,
        "n_pixels_after": None,
        "valid": False,
    }
    try:
        corr_before, n_before, _ = depth_gradient_correlation(
            uv_before, depth_before, image_bgr
        )
        corr_after, n_after, _ = depth_gradient_correlation(
            uv_after, depth_after, image_bgr
        )

        if corr_before is None or corr_after is None:
            print(
                "[DGC] Warning: insufficient evaluation pixels — "
                "comparison invalid."
            )
            return invalid

        improvement = corr_after - corr_before
        gap_before = 1.0 - corr_before
        gap_after = 1.0 - corr_after
        if gap_before > 0.0:
            improvement_pct = (gap_before - gap_after) / gap_before * 100.0
        else:
            improvement_pct = float("nan")

        print(f"DGC Before: {corr_before:.6f}")
        print(f"DGC After:  {corr_after:.6f}")
        if np.isfinite(improvement_pct):
            print(
                f"DGC Improvement: {improvement:+.6f} "
                f"({improvement_pct:+.2f}%)"
            )
        else:
            print(
                f"DGC Improvement: {improvement:+.6f} "
                "(improvement_pct undefined: gap_before == 0)"
            )

        return {
            "corr_before": float(corr_before),
            "corr_after": float(corr_after),
            "improvement": float(improvement),
            "improvement_pct": (
                float(improvement_pct)
                if np.isfinite(improvement_pct)
                else None
            ),
            "n_pixels_before": int(n_before),
            "n_pixels_after": int(n_after),
            "valid": True,
        }
    except Exception as exc:
        print(f"[DGC] Warning: compare_dgc failed: {exc}")
        return invalid
