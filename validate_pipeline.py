import os
import sys
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np

from calibration import parse_calib_cam_to_cam, parse_calib_velo_to_cam
from events import event_confidence, simulate_events
from flow import compute_rgb_flow, smooth_flow
from lidar_motion import move_lidar_points_weighted
from loader import load_image, load_lidar
from main import process_frame


PROJECT_ROOT = r"C:\Users\sahaa\OneDrive\Desktop\Honors\fusion-revised1"
DEBUG_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "validation_outputs")
TEST_LOG_DIR = os.path.join(PROJECT_ROOT, "test_logs")
DATASETS = [
    r"C:\Users\sahaa\OneDrive\Desktop\Honors\datasets\fusion\2011_09_26_drive_0009_sync",
    r"C:\Users\sahaa\OneDrive\Desktop\Honors\datasets\fusion\2011_09_26_drive_0005_sync",
    r"C:\Users\sahaa\OneDrive\Desktop\Honors\datasets\fusion\2011_09_26_drive_0013_sync",
    r"C:\Users\sahaa\OneDrive\Desktop\Honors\datasets\fusion\2011_09_26_drive_0017_sync",
]
DATASET_PATH = DATASETS[0]

EDGE_MEAN_PASS_PX = 5.0
EDGE_MEAN_FAIL_PX = 10.0
EDGE_MEDIAN_PASS_PX = 3.0
SURVIVAL_PASS_RATIO = 0.30
VERTICAL_SPREAD_PASS_PX = 4.0
VERTICAL_SPREAD_FAIL_PX = 8.0
TEMPORAL_RESIDUAL_THRESHOLD_PX = 2.0
REPROJECTION_PASS_RMSE = 1e-6
REPROJECTION_FAIL_RMSE = 1e-3
EDGE_GRADIENT_THRESHOLD = 30.0
TEMPORAL_STATS_MIN_FRAMES = 6
TEMPORAL_STATS_MIN_MOTION_PX = 1.0
TEMPORAL_STATS_MATCH_MAX_DIST_PX = 25.0
TEMPORAL_STATS_OUTLIER_QUANTILE = 95.0
EDGE_VALIDATION_FRAMES = 5
TEMPORAL_VALIDATION_FRAMES = max(EDGE_VALIDATION_FRAMES, int(TEMPORAL_STATS_MIN_FRAMES))


@dataclass
class FrameProjection:
    name: str
    image: np.ndarray
    lidar_xyz: np.ndarray
    lidar_h: np.ndarray
    rect_cam: np.ndarray
    projected: np.ndarray
    uv: np.ndarray
    uv_int: np.ndarray
    depth: np.ndarray
    stats: dict
    valid_mask_input: np.ndarray
    in_frame_mask_cam: np.ndarray


@dataclass
class MotionFramePair:
    original_t: FrameProjection
    original_t1: FrameProjection
    rgb_only_uv: np.ndarray
    rgb_only_uv_int: np.ndarray
    rgb_only_depth: np.ndarray
    corrected_uv: np.ndarray
    corrected_uv_int: np.ndarray
    corrected_depth: np.ndarray
    corrected_image: np.ndarray
    events: np.ndarray
    confidence: np.ndarray
    flow: np.ndarray


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _list_first_n_pairs(count):
    image_dir = os.path.join(DATASET_PATH, "image_02", "data")
    lidar_dir = os.path.join(DATASET_PATH, "velodyne_points", "data")
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
    lidar_files = sorted([f for f in os.listdir(lidar_dir) if f.endswith(".bin")])
    if not lidar_files:
        lidar_files = sorted([f for f in os.listdir(lidar_dir) if f.endswith(".txt")])
    if len(image_files) < count or len(lidar_files) < count:
        raise RuntimeError("Insufficient frames for validation")
    return [(image_files[i], lidar_files[i]) for i in range(count)]


def _project_with_full_trace(image_name, lidar_name, tr_velo_to_cam, r_rect, p_rect):
    image_dir = os.path.join(DATASET_PATH, "image_02", "data")
    lidar_dir = os.path.join(DATASET_PATH, "velodyne_points", "data")
    image_path = os.path.join(image_dir, image_name)
    lidar_path = os.path.join(lidar_dir, lidar_name)
    image = load_image(image_path)
    lidar_xyz = load_lidar(lidar_path)

    lidar_h = np.hstack((lidar_xyz, np.ones((lidar_xyz.shape[0], 1), dtype=np.float64))).T
    rectified_tf = r_rect @ tr_velo_to_cam
    proj = p_rect @ rectified_tf
    rect_cam = rectified_tf @ lidar_h
    projected = proj @ lidar_h

    valid_cam = (rect_cam[2, :] > 0.0) & np.isfinite(rect_cam).all(axis=0)
    valid_proj = valid_cam & (projected[2, :] != 0.0) & np.isfinite(projected).all(axis=0)

    u_full = np.full(lidar_xyz.shape[0], np.nan, dtype=np.float64)
    v_full = np.full(lidar_xyz.shape[0], np.nan, dtype=np.float64)
    u_full[valid_proj] = projected[0, valid_proj] / projected[2, valid_proj]
    v_full[valid_proj] = projected[1, valid_proj] / projected[2, valid_proj]

    h, w = image.shape[:2]
    in_frame = valid_proj.copy()
    in_frame[valid_proj] &= (
        (u_full[valid_proj] >= 0.0)
        & (u_full[valid_proj] < w)
        & (v_full[valid_proj] >= 0.0)
        & (v_full[valid_proj] < h)
    )

    uv = np.column_stack((u_full[in_frame], v_full[in_frame])).astype(np.float32)
    depth = rect_cam[2, in_frame].astype(np.float32)
    uv_int = np.clip(
        np.rint(uv).astype(np.int32),
        [0, 0],
        [w - 1, h - 1],
    )

    stats = {
        "input_points": int(lidar_xyz.shape[0]),
        "after_cam_filter": int(np.count_nonzero(valid_cam)),
        "after_proj_filter": int(np.count_nonzero(valid_proj)),
        "in_frame": int(np.count_nonzero(in_frame)),
    }

    return FrameProjection(
        name=os.path.splitext(image_name)[0],
        image=image,
        lidar_xyz=lidar_xyz,
        lidar_h=lidar_h,
        rect_cam=rect_cam,
        projected=projected,
        uv=uv,
        uv_int=uv_int,
        depth=depth,
        stats=stats,
        valid_mask_input=valid_proj,
        in_frame_mask_cam=in_frame[valid_proj],
    )


def _distance_transform_from_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)
    edges = cv2.Canny(blurred, 50, 150)
    strong_edges = np.where(
        (edges > 0) & (grad_mag >= EDGE_GRADIENT_THRESHOLD),
        255,
        0,
    ).astype(np.uint8)
    dist = cv2.distanceTransform(255 - strong_edges, cv2.DIST_L2, 3)
    return strong_edges, dist, grad_mag


def _detect_vertical_structures(frame, min_points=12):
    edges, _, _ = _distance_transform_from_edges(frame.image)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=40,
        minLineLength=40,
        maxLineGap=10,
    )
    if lines is None or frame.uv.shape[0] == 0:
        return []

    candidates = []
    pts = frame.uv.astype(np.float64)
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = [float(v) for v in line]
        dx = x2 - x1
        dy = y2 - y1
        if abs(dy) < 30.0:
            continue
        if abs(dx / dy) > 0.12:
            continue

        x_mid = 0.5 * (x1 + x2)
        y_min = min(y1, y2)
        y_max = max(y1, y2)
        mask = (
            (np.abs(pts[:, 0] - x_mid) <= 6.0)
            & (pts[:, 1] >= y_min - 5.0)
            & (pts[:, 1] <= y_max + 5.0)
            & (frame.depth < 40.0)
        )
        if np.count_nonzero(mask) < min_points:
            continue

        selected = pts[mask]
        x_residual = selected[:, 0] - np.median(selected[:, 0])
        vertical_span = np.max(selected[:, 1]) - np.min(selected[:, 1])
        spread = float(np.std(x_residual))
        tilt = float(np.polyfit(selected[:, 1], selected[:, 0], 1)[0]) if selected.shape[0] >= 2 else np.nan
        score = float(np.count_nonzero(mask)) / (spread + 1e-6)
        candidates.append(
            {
                "x_mid": x_mid,
                "y_min": y_min,
                "y_max": y_max,
                "count": int(np.count_nonzero(mask)),
                "spread_px": spread,
                "tilt_dxdy": tilt,
                "vertical_span_px": float(vertical_span),
                "mask": mask,
                "score": score,
            }
        )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates


def _temporal_profile(candidate, frame, bins=5):
    selected = frame.uv[candidate["mask"]].astype(np.float64)
    y = selected[:, 1]
    x = selected[:, 0]
    if selected.shape[0] < bins * 2:
        return None

    order = np.argsort(y)
    y = y[order]
    x = x[order]
    x_center = float(np.median(x))
    y_edges = np.linspace(y.min(), y.max(), bins + 1)
    residuals = []
    y_centers = []
    for idx in range(bins):
        if idx == bins - 1:
            mask = (y >= y_edges[idx]) & (y <= y_edges[idx + 1])
        else:
            mask = (y >= y_edges[idx]) & (y < y_edges[idx + 1])
        if np.count_nonzero(mask) < 2:
            residuals.append(np.nan)
            y_centers.append(0.5 * (y_edges[idx] + y_edges[idx + 1]))
            continue
        residuals.append(float(np.median(x[mask]) - x_center))
        y_centers.append(float(np.median(y[mask])))
    residuals = np.asarray(residuals, dtype=np.float64)
    y_centers = np.asarray(y_centers, dtype=np.float64)
    return x_center, y_centers, residuals


def _select_tracked_candidates(frames):
    per_frame = [_detect_vertical_structures(frame) for frame in frames]
    if any(len(candidates) == 0 for candidates in per_frame):
        return [None] * len(frames)

    image_w = frames[0].image.shape[1]
    filtered = []
    for candidates in per_frame:
        central = [
            c for c in candidates
            if 80.0 <= c["x_mid"] <= image_w - 80.0 and c["vertical_span_px"] >= 30.0
        ]
        filtered.append(central if central else candidates[:8])

    best_combo = None
    best_score = -np.inf
    for cand0 in filtered[0][:8]:
        for cand1 in filtered[1][:8]:
            for cand2 in filtered[2][:8]:
                xs = np.array([cand0["x_mid"], cand1["x_mid"], cand2["x_mid"]], dtype=np.float64)
                if np.max(xs) - np.min(xs) > 80.0:
                    continue
                counts = cand0["count"] + cand1["count"] + cand2["count"]
                spreads = cand0["spread_px"] + cand1["spread_px"] + cand2["spread_px"]
                centrality = np.mean(np.abs(xs - image_w * 0.5))
                score = counts - 8.0 * spreads - 0.25 * np.var(xs) - 0.02 * centrality
                if score > best_score:
                    best_score = score
                    best_combo = [cand0, cand1, cand2]

    if best_combo is not None:
        return best_combo
    return [candidates[0] if candidates else None for candidates in per_frame]


def _save_debug_modes(tr_velo_to_cam, r_rect, p_rect, frame_pairs):
    os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)
    image_dir = os.path.join(DATASET_PATH, "image_02", "data")
    lidar_dir = os.path.join(DATASET_PATH, "velodyne_points", "data")
    first_image, first_lidar = frame_pairs[0]
    for mode in (1, 2, 3):
        vis, _, _, _, _, _, _ = process_frame(
            os.path.join(image_dir, first_image),
            os.path.join(lidar_dir, first_lidar),
            tr_velo_to_cam,
            r_rect,
            p_rect,
            max_points=12000,
            debug_mode=mode,
        )
        out_path = os.path.join(DEBUG_OUTPUT_DIR, f"debug_mode_{mode}_{os.path.splitext(first_image)[0]}.png")
        cv2.imwrite(out_path, vis)
        print(f"debug_mode_{mode}_output: {out_path}")


def _filter_uv_depth_to_image(uv, depth, image_shape):
    uv = np.asarray(uv, dtype=np.float32)
    depth = np.asarray(depth, dtype=np.float32)
    if uv.size == 0 or depth.size == 0:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0, 2), dtype=np.int32),
        )

    h, w = image_shape[:2]
    valid = (
        np.isfinite(uv[:, 0])
        & np.isfinite(uv[:, 1])
        & np.isfinite(depth)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < w)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < h)
    )

    uv = uv[valid]
    depth = depth[valid]
    if uv.shape[0] == 0:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0, 2), dtype=np.int32),
        )

    uv_int = np.clip(
        np.rint(uv).astype(np.int32),
        [0, 0],
        [w - 1, h - 1],
    )
    return uv, depth, uv_int


def _build_motion_frame_pair(frame_t, frame_t1, prev_flow=None, use_smoothing=False):
    events = simulate_events(frame_t.image, frame_t1.image)
    flow_raw = compute_rgb_flow(frame_t.image, frame_t1.image)
    if use_smoothing:
        flow = smooth_flow(flow_raw, prev_flow, alpha=0.7)
    else:
        flow = np.nan_to_num(flow_raw.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    confidence = event_confidence(events)

    rgb_only_uv, rgb_only_depth = _move_points_with_rgb_flow(
        frame_t.uv,
        frame_t.depth,
        flow,
    )
    rgb_only_uv, rgb_only_depth, rgb_only_uv_int = _filter_uv_depth_to_image(
        rgb_only_uv,
        rgb_only_depth,
        frame_t1.image.shape,
    )

    corrected_uv, corrected_depth = move_lidar_points_weighted(
        frame_t.uv,
        frame_t.depth,
        flow,
        confidence,
    )
    corrected_uv, corrected_depth, corrected_uv_int = _filter_uv_depth_to_image(
        corrected_uv,
        corrected_depth,
        frame_t1.image.shape,
    )

    return MotionFramePair(
        original_t=frame_t,
        original_t1=frame_t1,
        rgb_only_uv=rgb_only_uv,
        rgb_only_uv_int=rgb_only_uv_int,
        rgb_only_depth=rgb_only_depth,
        corrected_uv=corrected_uv,
        corrected_uv_int=corrected_uv_int,
        corrected_depth=corrected_depth,
        corrected_image=frame_t1.image,
        events=events,
        confidence=confidence,
        flow=flow,
    )


def _build_motion_frame_pairs(frames, use_smoothing=False):
    motion_pairs = []
    prev_flow = None
    for idx in range(len(frames) - 1):
        motion_pair = _build_motion_frame_pair(
            frames[idx],
            frames[idx + 1],
            prev_flow=prev_flow,
            use_smoothing=use_smoothing,
        )
        motion_pairs.append(motion_pair)
        if use_smoothing:
            prev_flow = motion_pair.flow
    return motion_pairs


def _sample_flow_at_points(flow, points):
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    map_x = points[:, 0].astype(np.float32).reshape(-1, 1)
    map_y = points[:, 1].astype(np.float32).reshape(-1, 1)
    sampled = cv2.remap(
        flow,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return sampled.reshape(-1, 2)


def _move_points_with_rgb_flow(uv, depth, flow):
    uv = np.asarray(uv, dtype=np.float32)
    depth = np.asarray(depth, dtype=np.float32)
    flow = np.asarray(flow, dtype=np.float32)

    if uv.size == 0 or depth.size == 0 or flow.ndim != 3 or flow.shape[2] != 2:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

    h, w = flow.shape[:2]
    u = uv[:, 0]
    v = uv[:, 1]
    u_idx = np.rint(u).astype(np.int32)
    v_idx = np.rint(v).astype(np.int32)

    valid = (
        np.isfinite(u)
        & np.isfinite(v)
        & np.isfinite(depth)
        & (u_idx >= 0)
        & (u_idx < w)
        & (v_idx >= 0)
        & (v_idx < h)
    )

    if not np.any(valid):
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

    uv_valid = uv[valid]
    depth_valid = depth[valid]
    sampled_flow = flow[v_idx[valid], u_idx[valid], :]
    uv_rgb = uv_valid.copy()
    uv_rgb[:, 0] += sampled_flow[:, 0]
    uv_rgb[:, 1] += sampled_flow[:, 1]
    uv_rgb = np.nan_to_num(uv_rgb.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return uv_rgb, depth_valid


def _flann_knn_1nn(train_points, query_points):
    if train_points.shape[0] == 0 or query_points.shape[0] == 0:
        return (
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
        )

    index = cv2.flann_Index(train_points.astype(np.float32), dict(algorithm=1, trees=4))
    indices, dists = index.knnSearch(
        query_points.astype(np.float32),
        1,
        params={},
    )
    return indices[:, 0].astype(np.int32), np.sqrt(np.maximum(dists[:, 0], 0.0)).astype(np.float32)


def _filter_top_quantile(values, quantile):
    if values.size == 0:
        return values
    cutoff = np.percentile(values, quantile)
    return values[values <= cutoff]


def _compute_edge_alignment(uv_int, image):
    _, dist, grad_mag = _distance_transform_from_edges(image)
    if uv_int.size == 0:
        return {
            "mean_px": float("inf"),
            "median_px": float("inf"),
            "keep_ratio": 0.0,
        }

    strong_points = grad_mag[uv_int[:, 1], uv_int[:, 0]] >= EDGE_GRADIENT_THRESHOLD
    keep_ratio = float(np.mean(strong_points))
    if not np.any(strong_points):
        dists = np.empty((0,), dtype=np.float32)
    else:
        dists = dist[uv_int[strong_points, 1], uv_int[strong_points, 0]]

    return {
        "mean_px": float(np.mean(dists)) if dists.size else float("inf"),
        "median_px": float(np.median(dists)) if dists.size else float("inf"),
        "keep_ratio": keep_ratio,
    }


def compute_edge_alignment_error(image, uv_points, max_dist=20.0):
    """
    Compute edge alignment error: mean distance from each projected LiDAR
    point to the nearest detected image edge.

    This is an independent metric — it uses only the image edges and the
    projected point positions, with no dependency on optical flow.

    A lower value means LiDAR points are better aligned with scene boundaries.

    Args:
        image: BGR image (H, W, 3)
        uv_points: (N, 2) float32 projected LiDAR point coordinates
        max_dist: cap distance at this value to limit outlier influence

    Returns:
        mean_error: float, mean distance in pixels (lower = better alignment)
        median_error: float, median distance in pixels
        pct_within_2px: float, fraction of points within 2px of an edge
    """
    if uv_points.shape[0] == 0:
        return float('nan'), float('nan'), float('nan')

    # Detect edges using Canny on grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Mild blur before Canny to reduce noise sensitivity
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    edges = cv2.Canny(blurred, 50, 150)

    # Distance transform: each non-edge pixel gets its distance to the
    # nearest edge pixel. distanceTransform treats zero pixels as the source,
    # so we invert the edge image before calling it.
    edge_inv = np.where(edges > 0, 0, 255).astype(np.uint8)
    dist_map = cv2.distanceTransform(edge_inv, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    # Sample distance at each projected LiDAR point.
    H, W = image.shape[:2]
    u_idx = np.clip(np.rint(uv_points[:, 0]).astype(np.int32), 0, W - 1)
    v_idx = np.clip(np.rint(uv_points[:, 1]).astype(np.int32), 0, H - 1)

    distances = dist_map[v_idx, u_idx]
    distances = np.minimum(distances, max_dist)  # cap outliers

    mean_error = float(np.mean(distances))
    median_error = float(np.median(distances))
    pct_within_2px = float(np.mean(distances <= 2.0))

    return mean_error, median_error, pct_within_2px


def temporal_statistics_test(
    tr_velo_to_cam,
    r_rect,
    p_rect,
    n_frames=TEMPORAL_STATS_MIN_FRAMES,
    frames=None,
    use_smoothing=False,
):
    print("\n--- TEST 10: STATISTICAL TEMPORAL INCONSISTENCY ---")
    if frames is None:
        frame_count = max(5, int(n_frames))
        frame_pairs = _list_first_n_pairs(frame_count)
        frames = [
            _project_with_full_trace(image_name, lidar_name, tr_velo_to_cam, r_rect, p_rect)
            for image_name, lidar_name in frame_pairs
        ]
    motion_pairs = _build_motion_frame_pairs(frames, use_smoothing=use_smoothing)

    original_pair_errors = []
    rgb_only_pair_errors = []
    corrected_pair_errors = []
    baseline_errors = []

    for idx in range(len(frames) - 1):
        frame_t = frames[idx]
        frame_t1 = frames[idx + 1]
        motion_pair = motion_pairs[idx]

        flow_fwd = motion_pair.flow
        flow_bwd = compute_rgb_flow(frame_t1.image, frame_t.image)
        confidence = motion_pair.confidence

        nn_forward_idx, nn_forward_dist = _flann_knn_1nn(frame_t1.uv, frame_t.uv)
        nn_reverse_idx, _ = _flann_knn_1nn(frame_t.uv, frame_t1.uv)

        if frame_t.uv.shape[0] == 0 or frame_t1.uv.shape[0] == 0 or nn_forward_idx.size == 0:
            print(f"{frame_t.name}->{frame_t1.name}: insufficient_points_for_temporal_statistics")
            continue

        query_idx = np.arange(frame_t.uv.shape[0], dtype=np.int32)
        mutual = nn_reverse_idx[nn_forward_idx] == query_idx
        distance_ok = nn_forward_dist <= TEMPORAL_STATS_MATCH_MAX_DIST_PX
        selected_mask = mutual & distance_ok

        matched_uv_t = frame_t.uv[selected_mask]
        matched_uv_t1 = frame_t1.uv[nn_forward_idx[selected_mask]]
        matched_depth_t = frame_t.depth[selected_mask]

        if matched_uv_t.shape[0] == 0:
            print(f"{frame_t.name}->{frame_t1.name}: no_mutual_matches_after_distance_filter")
            continue

        lidar_disp = matched_uv_t1 - matched_uv_t
        img_disp = _sample_flow_at_points(flow_fwd, matched_uv_t)

        lidar_motion = np.linalg.norm(lidar_disp, axis=1)
        img_motion = np.linalg.norm(img_disp, axis=1)
        motion_mask = (lidar_motion >= TEMPORAL_STATS_MIN_MOTION_PX) & (img_motion >= TEMPORAL_STATS_MIN_MOTION_PX)

        matched_uv_t = matched_uv_t[motion_mask]
        lidar_disp = lidar_disp[motion_mask]
        img_disp = img_disp[motion_mask]
        matched_uv_t1 = matched_uv_t1[motion_mask]
        matched_depth_t = matched_depth_t[motion_mask]

        if matched_uv_t.shape[0] == 0:
            print(f"{frame_t.name}->{frame_t1.name}: no_matches_after_motion_filter")
            continue

        original_temporal_error = np.linalg.norm(lidar_disp - img_disp, axis=1)

        h_t1, w_t1 = frame_t1.image.shape[:2]
        u_idx = np.rint(matched_uv_t[:, 0]).astype(np.int32)
        v_idx = np.rint(matched_uv_t[:, 1]).astype(np.int32)
        source_valid = (
            (u_idx >= 0)
            & (u_idx < w_t1)
            & (v_idx >= 0)
            & (v_idx < h_t1)
            & np.isfinite(matched_uv_t[:, 0])
            & np.isfinite(matched_uv_t[:, 1])
            & np.isfinite(matched_depth_t)
        )

        matched_uv_t_corr = matched_uv_t[source_valid]
        matched_uv_t1_corr = matched_uv_t1[source_valid]
        matched_depth_t_corr = matched_depth_t[source_valid]

        rgb_only_uv_raw, rgb_only_depth_raw = _move_points_with_rgb_flow(
            matched_uv_t_corr,
            matched_depth_t_corr,
            flow_fwd,
        )
        rgb_only_valid = (
            np.isfinite(rgb_only_uv_raw[:, 0])
            & np.isfinite(rgb_only_uv_raw[:, 1])
            & (rgb_only_uv_raw[:, 0] >= 0.0)
            & (rgb_only_uv_raw[:, 0] < w_t1)
            & (rgb_only_uv_raw[:, 1] >= 0.0)
            & (rgb_only_uv_raw[:, 1] < h_t1)
        )
        rgb_only_uv = rgb_only_uv_raw[rgb_only_valid]
        matched_uv_t1_rgb_only = matched_uv_t1_corr[rgb_only_valid]

        if rgb_only_uv.shape[0] == 0:
            rgb_only_temporal_error = np.empty((0,), dtype=np.float32)
        else:
            rgb_only_temporal_error = np.linalg.norm(
                matched_uv_t1_rgb_only - rgb_only_uv,
                axis=1,
            )

        corrected_uv_raw, corrected_depth_raw = move_lidar_points_weighted(
            matched_uv_t_corr,
            matched_depth_t_corr,
            flow_fwd,
            confidence,
        )
        corrected_valid = (
            np.isfinite(corrected_uv_raw[:, 0])
            & np.isfinite(corrected_uv_raw[:, 1])
            & (corrected_uv_raw[:, 0] >= 0.0)
            & (corrected_uv_raw[:, 0] < w_t1)
            & (corrected_uv_raw[:, 1] >= 0.0)
            & (corrected_uv_raw[:, 1] < h_t1)
        )
        corrected_uv = corrected_uv_raw[corrected_valid]
        matched_uv_t1_corrected = matched_uv_t1_corr[corrected_valid]

        if corrected_uv.shape[0] == 0:
            corrected_temporal_error = np.empty((0,), dtype=np.float32)
        else:
            corrected_temporal_error = np.linalg.norm(
                matched_uv_t1_corrected - corrected_uv,
                axis=1,
            )

        original_temporal_error = _filter_top_quantile(
            original_temporal_error,
            TEMPORAL_STATS_OUTLIER_QUANTILE,
        )
        rgb_only_temporal_error = _filter_top_quantile(
            rgb_only_temporal_error,
            TEMPORAL_STATS_OUTLIER_QUANTILE,
        )
        corrected_temporal_error = _filter_top_quantile(
            corrected_temporal_error,
            TEMPORAL_STATS_OUTLIER_QUANTILE,
        )

        flow_at_uv = _sample_flow_at_points(flow_fwd, matched_uv_t)
        warped_uv = matched_uv_t + flow_at_uv
        in_bounds = (
            (warped_uv[:, 0] >= 0.0)
            & (warped_uv[:, 0] < frame_t.image.shape[1])
            & (warped_uv[:, 1] >= 0.0)
            & (warped_uv[:, 1] < frame_t.image.shape[0])
        )
        flow_at_uv = flow_at_uv[in_bounds]
        warped_uv = warped_uv[in_bounds]
        back_flow = _sample_flow_at_points(flow_bwd, warped_uv)
        self_consistency_error = np.linalg.norm(flow_at_uv + back_flow, axis=1)
        self_consistency_error = self_consistency_error[
            np.linalg.norm(flow_at_uv, axis=1) >= TEMPORAL_STATS_MIN_MOTION_PX
        ]
        self_consistency_error = _filter_top_quantile(self_consistency_error, TEMPORAL_STATS_OUTLIER_QUANTILE)

        original_pair_errors.append(original_temporal_error)
        rgb_only_pair_errors.append(rgb_only_temporal_error)
        corrected_pair_errors.append(corrected_temporal_error)
        baseline_errors.append(self_consistency_error)

        original_pair_mean = float(np.mean(original_temporal_error)) if original_temporal_error.size else float("inf")
        original_pair_median = float(np.median(original_temporal_error)) if original_temporal_error.size else float("inf")
        rgb_only_pair_mean = float(np.mean(rgb_only_temporal_error)) if rgb_only_temporal_error.size else float("inf")
        rgb_only_pair_median = float(np.median(rgb_only_temporal_error)) if rgb_only_temporal_error.size else float("inf")
        corrected_pair_mean = float(np.mean(corrected_temporal_error)) if corrected_temporal_error.size else float("inf")
        corrected_pair_median = float(np.median(corrected_temporal_error)) if corrected_temporal_error.size else float("inf")
        print(
            f"{frame_t.name}->{frame_t1.name}: "
            f"original_mean_error_px={original_pair_mean:.6f} "
            f"original_median_error_px={original_pair_median:.6f} "
            f"rgb_only_mean_error_px={rgb_only_pair_mean:.6f} "
            f"rgb_only_median_error_px={rgb_only_pair_median:.6f} "
            f"corrected_mean_error_px={corrected_pair_mean:.6f} "
            f"corrected_median_error_px={corrected_pair_median:.6f} "
            f"matches={original_temporal_error.size}"
        )

    global_original_temporal = (
        np.concatenate(original_pair_errors)
        if original_pair_errors
        else np.empty((0,), dtype=np.float32)
    )
    global_rgb_only_temporal = (
        np.concatenate(rgb_only_pair_errors)
        if rgb_only_pair_errors
        else np.empty((0,), dtype=np.float32)
    )
    global_corrected_temporal = (
        np.concatenate(corrected_pair_errors)
        if corrected_pair_errors
        else np.empty((0,), dtype=np.float32)
    )
    global_baseline = np.concatenate(baseline_errors) if baseline_errors else np.empty((0,), dtype=np.float32)

    original_temporal_mean = float(np.mean(global_original_temporal)) if global_original_temporal.size else float("inf")
    original_temporal_median = float(np.median(global_original_temporal)) if global_original_temporal.size else float("inf")
    rgb_only_temporal_mean = float(np.mean(global_rgb_only_temporal)) if global_rgb_only_temporal.size else float("inf")
    rgb_only_temporal_median = float(np.median(global_rgb_only_temporal)) if global_rgb_only_temporal.size else float("inf")
    corrected_temporal_mean = float(np.mean(global_corrected_temporal)) if global_corrected_temporal.size else float("inf")
    corrected_temporal_median = float(np.median(global_corrected_temporal)) if global_corrected_temporal.size else float("inf")
    baseline_mean = float(np.mean(global_baseline)) if global_baseline.size else float("inf")
    baseline_median = float(np.median(global_baseline)) if global_baseline.size else float("inf")

    temporal_statistical = (
        global_original_temporal.size > 0
        and global_baseline.size > 0
        and original_temporal_mean > baseline_mean * 2.0
        and original_temporal_median > baseline_median * 2.0
    )

    print(f"Original temporal inconsistency mean error: {original_temporal_mean:.6f} px")
    print(f"Original temporal inconsistency median error: {original_temporal_median:.6f} px")
    print(f"RGB-only temporal inconsistency mean error: {rgb_only_temporal_mean:.6f} px")
    print(f"RGB-only temporal inconsistency median error: {rgb_only_temporal_median:.6f} px")
    print(f"Corrected temporal inconsistency mean error: {corrected_temporal_mean:.6f} px")
    print(f"Corrected temporal inconsistency median error: {corrected_temporal_median:.6f} px")
    print(f"Image self-consistency error: {baseline_mean:.6f} px")
    print("Temporal distortion (statistical): " + ("YES" if temporal_statistical else "NO"))

    return {
        "original_temporal_mean_px": original_temporal_mean,
        "original_temporal_median_px": original_temporal_median,
        "rgb_only_temporal_mean_px": rgb_only_temporal_mean,
        "rgb_only_temporal_median_px": rgb_only_temporal_median,
        "corrected_temporal_mean_px": corrected_temporal_mean,
        "corrected_temporal_median_px": corrected_temporal_median,
        "baseline_mean_px": baseline_mean,
        "baseline_median_px": baseline_median,
        "detected": temporal_statistical,
    }


def _evaluate_mode_metrics(traced_frames, temporal_frames, tr_velo_to_cam, r_rect, p_rect, use_smoothing):
    edge_frames = traced_frames[:5]
    motion_pairs = _build_motion_frame_pairs(edge_frames, use_smoothing=use_smoothing)

    original_edge_medians = []
    rgb_only_edge_medians = []
    corrected_edge_medians = []

    for frame in edge_frames:
        edge_metrics = _compute_edge_alignment(frame.uv_int, frame.image)
        original_edge_medians.append(edge_metrics["median_px"])

    for motion_pair in motion_pairs:
        rgb_only_metrics = _compute_edge_alignment(motion_pair.rgb_only_uv_int, motion_pair.corrected_image)
        corrected_metrics = _compute_edge_alignment(motion_pair.corrected_uv_int, motion_pair.corrected_image)
        rgb_only_edge_medians.append(rgb_only_metrics["median_px"])
        corrected_edge_medians.append(corrected_metrics["median_px"])

    # ---- Edge alignment to Canny edges (independent metric) ----
    # For each motion pair, compare uv_orig and uv_corr against edges in
    # image_t (the frame used for projection).
    ea_before_mean = []
    ea_before_median = []
    ea_before_pct2 = []
    ea_after_mean = []
    ea_after_median = []
    ea_after_pct2 = []

    for motion_pair in motion_pairs:
        image_t = motion_pair.original_t.image
        image_t1 = motion_pair.original_t1.image
        uv_orig = np.asarray(motion_pair.original_t.uv, dtype=np.float32)
        uv_corr = np.asarray(motion_pair.corrected_uv, dtype=np.float32)

        if uv_orig.size > 0:
            vo = np.isfinite(uv_orig[:, 0]) & np.isfinite(uv_orig[:, 1])
            uv_orig_valid = uv_orig[vo]
        else:
            uv_orig_valid = uv_orig

        if uv_corr.size > 0:
            vc = np.isfinite(uv_corr[:, 0]) & np.isfinite(uv_corr[:, 1])
            uv_corr_valid = uv_corr[vc]
        else:
            uv_corr_valid = uv_corr

        # Before: uncorrected projection vs frame t edges (source frame).
        # After:  corrected projection vs frame t+1 edges (target frame).
        b_mean, b_med, b_pct = compute_edge_alignment_error(image_t, uv_orig_valid)
        a_mean, a_med, a_pct = compute_edge_alignment_error(image_t1, uv_corr_valid)

        if np.isfinite(b_mean):
            ea_before_mean.append(b_mean)
            ea_before_median.append(b_med)
            ea_before_pct2.append(b_pct)
        if np.isfinite(a_mean):
            ea_after_mean.append(a_mean)
            ea_after_median.append(a_med)
            ea_after_pct2.append(a_pct)

    edge_align_before_mean = float(np.mean(ea_before_mean)) if ea_before_mean else float("nan")
    edge_align_before_median = float(np.mean(ea_before_median)) if ea_before_median else float("nan")
    edge_align_before_pct2 = float(np.mean(ea_before_pct2)) if ea_before_pct2 else float("nan")
    edge_align_after_mean = float(np.mean(ea_after_mean)) if ea_after_mean else float("nan")
    edge_align_after_median = float(np.mean(ea_after_median)) if ea_after_median else float("nan")
    edge_align_after_pct2 = float(np.mean(ea_after_pct2)) if ea_after_pct2 else float("nan")

    temporal_stats = temporal_statistics_test(
        tr_velo_to_cam,
        r_rect,
        p_rect,
        n_frames=len(temporal_frames),
        frames=temporal_frames,
        use_smoothing=use_smoothing,
    )

    distortion_ratio = (
        temporal_stats["original_temporal_mean_px"] / temporal_stats["baseline_mean_px"]
        if np.isfinite(temporal_stats["original_temporal_mean_px"])
        and np.isfinite(temporal_stats["baseline_mean_px"])
        and temporal_stats["baseline_mean_px"] > 0.0
        else float("inf")
    )

    return {
        "original_temporal_mean_px": temporal_stats["original_temporal_mean_px"],
        "original_temporal_median_px": temporal_stats["original_temporal_median_px"],
        "rgb_only_temporal_mean_px": temporal_stats["rgb_only_temporal_mean_px"],
        "rgb_only_temporal_median_px": temporal_stats["rgb_only_temporal_median_px"],
        "corrected_temporal_mean_px": temporal_stats["corrected_temporal_mean_px"],
        "corrected_temporal_median_px": temporal_stats["corrected_temporal_median_px"],
        "original_edge_median_px": float(np.median(original_edge_medians)) if original_edge_medians else float("inf"),
        "rgb_only_edge_median_px": float(np.median(rgb_only_edge_medians)) if rgb_only_edge_medians else float("inf"),
        "corrected_edge_median_px": float(np.median(corrected_edge_medians)) if corrected_edge_medians else float("inf"),
        "edge_align_before_mean_px": edge_align_before_mean,
        "edge_align_before_median_px": edge_align_before_median,
        "edge_align_before_within_2px": edge_align_before_pct2,
        "edge_align_after_mean_px": edge_align_after_mean,
        "edge_align_after_median_px": edge_align_after_median,
        "edge_align_after_within_2px": edge_align_after_pct2,
        "distortion_ratio": distortion_ratio,
        "detected": temporal_stats["detected"],
    }


def _compute_improvement(original_error, corrected_error):
    improvement_px = original_error - corrected_error
    improvement_percent = (
        (improvement_px / original_error) * 100.0
        if np.isfinite(original_error) and np.isfinite(corrected_error) and original_error != 0.0
        else 0.0
    )
    return improvement_px, improvement_percent


def _format_px(value):
    return f"{value:.6f} px"


def _format_percent(value):
    return f"{value:.2f} %"


def run_validation_on_dataset(dataset_path):
    global DATASET_PATH

    DATASET_PATH = dataset_path

    tr_velo_to_cam = parse_calib_velo_to_cam(os.path.join(DATASET_PATH, "calib_velo_to_cam.txt"))
    r_rect, p_rect = parse_calib_cam_to_cam(os.path.join(DATASET_PATH, "calib_cam_to_cam.txt"), camera_id="02")

    frame_pairs = _list_first_n_pairs(EDGE_VALIDATION_FRAMES)
    traced_frames = [
        _project_with_full_trace(image_name, lidar_name, tr_velo_to_cam, r_rect, p_rect)
        for image_name, lidar_name in frame_pairs
    ]

    temporal_frame_pairs = _list_first_n_pairs(TEMPORAL_VALIDATION_FRAMES)
    temporal_traced_frames = [
        _project_with_full_trace(image_name, lidar_name, tr_velo_to_cam, r_rect, p_rect)
        for image_name, lidar_name in temporal_frame_pairs
    ]

    no_smoothing_metrics = _evaluate_mode_metrics(
        traced_frames,
        temporal_traced_frames,
        tr_velo_to_cam,
        r_rect,
        p_rect,
        use_smoothing=False,
    )
    smoothing_metrics = _evaluate_mode_metrics(
        traced_frames,
        temporal_traced_frames,
        tr_velo_to_cam,
        r_rect,
        p_rect,
        use_smoothing=True,
    )

    no_smoothing_improvement_px, no_smoothing_improvement_percent = _compute_improvement(
        no_smoothing_metrics["original_temporal_mean_px"],
        no_smoothing_metrics["corrected_temporal_mean_px"],
    )
    smoothing_improvement_px, smoothing_improvement_percent = _compute_improvement(
        smoothing_metrics["original_temporal_mean_px"],
        smoothing_metrics["corrected_temporal_mean_px"],
    )

    # Independent edge-alignment metric (no flow dependency). Report the
    # no_smoothing mode as the primary, since that is the headline pipeline.
    ea_before_mean = no_smoothing_metrics["edge_align_before_mean_px"]
    ea_after_mean = no_smoothing_metrics["edge_align_after_mean_px"]
    if (
        np.isfinite(ea_before_mean)
        and np.isfinite(ea_after_mean)
        and ea_before_mean > 0.0
    ):
        ea_improvement_percent = (ea_before_mean - ea_after_mean) / ea_before_mean * 100.0
    else:
        ea_improvement_percent = float("nan")

    return {
        "dataset_path": dataset_path,
        "dataset_name": os.path.basename(dataset_path),
        "no_smoothing": {
            "original_error": no_smoothing_metrics["original_temporal_mean_px"],
            "rgb_only_error": no_smoothing_metrics["rgb_only_temporal_mean_px"],
            "corrected_error": no_smoothing_metrics["corrected_temporal_mean_px"],
            "improvement_px": no_smoothing_improvement_px,
            "improvement_percent": no_smoothing_improvement_percent,
        },
        "with_smoothing": {
            "original_error": smoothing_metrics["original_temporal_mean_px"],
            "rgb_only_error": smoothing_metrics["rgb_only_temporal_mean_px"],
            "corrected_error": smoothing_metrics["corrected_temporal_mean_px"],
            "improvement_px": smoothing_improvement_px,
            "improvement_percent": smoothing_improvement_percent,
        },
        "smoothing_effect": {
            "original_change": no_smoothing_metrics["original_temporal_mean_px"] - smoothing_metrics["original_temporal_mean_px"],
            "rgb_only_change": no_smoothing_metrics["rgb_only_temporal_mean_px"] - smoothing_metrics["rgb_only_temporal_mean_px"],
            "corrected_change": no_smoothing_metrics["corrected_temporal_mean_px"] - smoothing_metrics["corrected_temporal_mean_px"],
        },
        "edge_alignment": {
            "before_mean": ea_before_mean,
            "before_median": no_smoothing_metrics["edge_align_before_median_px"],
            "before_within_2px": no_smoothing_metrics["edge_align_before_within_2px"],
            "after_mean": ea_after_mean,
            "after_median": no_smoothing_metrics["edge_align_after_median_px"],
            "after_within_2px": no_smoothing_metrics["edge_align_after_within_2px"],
            "improvement_percent": ea_improvement_percent,
        },
    }


def _print_dataset_results(result):
    no_smoothing = result["no_smoothing"]
    with_smoothing = result["with_smoothing"]
    smoothing_effect = result["smoothing_effect"]

    print("==============================")
    print(f"DATASET: {result['dataset_name']}")
    print("=======================")
    print("")
    print("--- NO SMOOTHING ---")
    print(f"Original: {_format_px(no_smoothing['original_error'])}")
    print(f"RGB-only: {_format_px(no_smoothing['rgb_only_error'])}")
    print(f"Corrected: {_format_px(no_smoothing['corrected_error'])}")
    print(
        f"Improvement: {_format_px(no_smoothing['improvement_px'])} "
        f"({_format_percent(no_smoothing['improvement_percent'])})"
    )
    print("")
    print("--- WITH SMOOTHING ---")
    print(f"Original: {_format_px(with_smoothing['original_error'])}")
    print(f"RGB-only: {_format_px(with_smoothing['rgb_only_error'])}")
    print(f"Corrected: {_format_px(with_smoothing['corrected_error'])}")
    print(
        f"Improvement: {_format_px(with_smoothing['improvement_px'])} "
        f"({_format_percent(with_smoothing['improvement_percent'])})"
    )
    print("")
    print("--- SMOOTHING EFFECT ---")
    print(f"Original change: {_format_px(smoothing_effect['original_change'])}")
    print(f"RGB-only change: {_format_px(smoothing_effect['rgb_only_change'])}")
    print(f"Corrected change: {_format_px(smoothing_effect['corrected_change'])}")
    print("")
    edge_align = result["edge_alignment"]
    print("=== Edge Alignment Error (independent metric) ===")
    print(
        f"Before correction (vs frame t edges):   mean={edge_align['before_mean']:.3f} px, "
        f"median={edge_align['before_median']:.3f} px, "
        f"within_2px={edge_align['before_within_2px']:.1%}"
    )
    print(
        f"After correction  (vs frame t+1 edges): mean={edge_align['after_mean']:.3f} px, "
        f"median={edge_align['after_median']:.3f} px, "
        f"within_2px={edge_align['after_within_2px']:.1%}"
    )
    improvement = edge_align['improvement_percent']
    if np.isfinite(improvement):
        print(f"Improvement: {improvement:.1f}% reduction in mean edge alignment error")
    else:
        print("Improvement: n/a")
    print("")


def _mean_of(values):
    return float(np.mean(values)) if values else float("nan")


def run_all_datasets():
    no_smoothing_original_errors = []
    no_smoothing_corrected_errors = []
    no_smoothing_improvements = []
    smoothing_original_errors = []
    smoothing_corrected_errors = []
    smoothing_improvements = []
    dataset_results = []

    for dataset_path in DATASETS:
        result = run_validation_on_dataset(dataset_path)
        dataset_results.append(result)
        _print_dataset_results(result)

        no_smoothing_original_errors.append(result["no_smoothing"]["original_error"])
        no_smoothing_corrected_errors.append(result["no_smoothing"]["corrected_error"])
        no_smoothing_improvements.append(result["no_smoothing"]["improvement_percent"])

        smoothing_original_errors.append(result["with_smoothing"]["original_error"])
        smoothing_corrected_errors.append(result["with_smoothing"]["corrected_error"])
        smoothing_improvements.append(result["with_smoothing"]["improvement_percent"])

    best_dataset = max(dataset_results, key=lambda item: item["no_smoothing"]["improvement_percent"])
    worst_dataset = min(dataset_results, key=lambda item: item["no_smoothing"]["improvement_percent"])

    print("==============================")
    print("FINAL AGGREGATED RESULTS")
    print("========================")
    print("")
    print("NO SMOOTHING:")
    print(f"Avg Original: {_format_px(_mean_of(no_smoothing_original_errors))}")
    print(f"Avg Corrected: {_format_px(_mean_of(no_smoothing_corrected_errors))}")
    print(f"Avg Improvement: {_format_percent(_mean_of(no_smoothing_improvements))}")
    print("")
    print("WITH SMOOTHING:")
    print(f"Avg Original: {_format_px(_mean_of(smoothing_original_errors))}")
    print(f"Avg Corrected: {_format_px(_mean_of(smoothing_corrected_errors))}")
    print(f"Avg Improvement: {_format_percent(_mean_of(smoothing_improvements))}")
    print("")
    print("---")
    print("")
    print(f"Best dataset improvement: {_format_percent(best_dataset['no_smoothing']['improvement_percent'])}")
    print(f"Worst dataset improvement: {_format_percent(worst_dataset['no_smoothing']['improvement_percent'])}")

    # -------- Cross-dataset edge alignment summary --------
    print("")
    print("EDGE ALIGNMENT ERROR — PER DATASET (independent metric):")
    header = f"{'Dataset':<22}{'Before (px)':>14}{'After (px)':>14}{'Improvement':>14}"
    print(header)
    print("-" * len(header))
    edge_before_vals = []
    edge_after_vals = []
    edge_impr_vals = []
    for r in dataset_results:
        ea = r["edge_alignment"]
        before_s = f"{ea['before_mean']:.3f}" if np.isfinite(ea['before_mean']) else "n/a"
        after_s = f"{ea['after_mean']:.3f}" if np.isfinite(ea['after_mean']) else "n/a"
        if np.isfinite(ea['improvement_percent']):
            impr_s = f"{ea['improvement_percent']:+.2f} %"
            edge_impr_vals.append(ea['improvement_percent'])
        else:
            impr_s = "n/a"
        if np.isfinite(ea['before_mean']):
            edge_before_vals.append(ea['before_mean'])
        if np.isfinite(ea['after_mean']):
            edge_after_vals.append(ea['after_mean'])
        print(f"{r['dataset_name']:<22}{before_s:>14}{after_s:>14}{impr_s:>14}")
    print("-" * len(header))
    mean_before = f"{_mean_of(edge_before_vals):.3f}" if edge_before_vals else "n/a"
    mean_after = f"{_mean_of(edge_after_vals):.3f}" if edge_after_vals else "n/a"
    mean_impr = f"{_mean_of(edge_impr_vals):+.2f} %" if edge_impr_vals else "n/a"
    print(f"{'Mean':<22}{mean_before:>14}{mean_after:>14}{mean_impr:>14}")


if __name__ == "__main__":
    os.makedirs(TEST_LOG_DIR, exist_ok=True)
    run_timestamp = datetime.now()
    log_filename = f"validation_{run_timestamp.strftime('%Y-%m-%d_%H-%M-%S')}.txt"
    log_path = os.path.join(TEST_LOG_DIR, log_filename)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    original_stdout = sys.stdout
    with open(log_path, "w", encoding="utf-8") as log_file:
        sys.stdout = TeeStream(original_stdout, log_file)
        try:
            run_all_datasets()
        finally:
            sys.stdout.flush()
            sys.stdout = original_stdout
