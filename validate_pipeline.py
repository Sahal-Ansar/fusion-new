import os
from dataclasses import dataclass

import cv2
import numpy as np

from calibration import parse_calib_cam_to_cam, parse_calib_velo_to_cam
from loader import load_image, load_lidar
from main import process_frame


PROJECT_ROOT = r"C:\Users\sahaa\OneDrive\Desktop\Honors\fusion-revised1"
DATASET_PATH = r"C:\Users\sahaa\OneDrive\Desktop\Honors\datasets\fusion\2011_09_26_drive_0009_extract"
DEBUG_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "validation_outputs")
IMAGE_DIR = os.path.join(DATASET_PATH, "image_02", "data")
LIDAR_DIR = os.path.join(DATASET_PATH, "velodyne_points", "data")

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


def _list_first_n_pairs(count):
    image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(".png")])
    lidar_files = sorted([f for f in os.listdir(LIDAR_DIR) if f.endswith(".bin")])
    if not lidar_files:
        lidar_files = sorted([f for f in os.listdir(LIDAR_DIR) if f.endswith(".txt")])
    if len(image_files) < count or len(lidar_files) < count:
        raise RuntimeError("Insufficient frames for validation")
    return [(image_files[i], lidar_files[i]) for i in range(count)]


def _project_with_full_trace(image_name, lidar_name, tr_velo_to_cam, r_rect, p_rect):
    image_path = os.path.join(IMAGE_DIR, image_name)
    lidar_path = os.path.join(LIDAR_DIR, lidar_name)
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
    first_image, first_lidar = frame_pairs[0]
    for mode in (1, 2, 3):
        vis, _, _, _ = process_frame(
            os.path.join(IMAGE_DIR, first_image),
            os.path.join(LIDAR_DIR, first_lidar),
            tr_velo_to_cam,
            r_rect,
            p_rect,
            max_points=12000,
            debug_mode=mode,
        )
        out_path = os.path.join(DEBUG_OUTPUT_DIR, f"debug_mode_{mode}_{os.path.splitext(first_image)[0]}.png")
        cv2.imwrite(out_path, vis)
        print(f"debug_mode_{mode}_output: {out_path}")


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


def temporal_statistics_test(tr_velo_to_cam, r_rect, p_rect, n_frames=TEMPORAL_STATS_MIN_FRAMES):
    print("\n--- TEST 10: STATISTICAL TEMPORAL INCONSISTENCY ---")
    frame_count = max(5, int(n_frames))
    frame_pairs = _list_first_n_pairs(frame_count)
    frames = [
        _project_with_full_trace(image_name, lidar_name, tr_velo_to_cam, r_rect, p_rect)
        for image_name, lidar_name in frame_pairs
    ]

    pair_errors = []
    baseline_errors = []

    for idx in range(len(frames) - 1):
        frame_t = frames[idx]
        frame_t1 = frames[idx + 1]

        gray_t = cv2.cvtColor(frame_t.image, cv2.COLOR_BGR2GRAY)
        gray_t1 = cv2.cvtColor(frame_t1.image, cv2.COLOR_BGR2GRAY)

        flow_fwd = cv2.calcOpticalFlowFarneback(
            gray_t,
            gray_t1,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=21,
            iterations=5,
            poly_n=7,
            poly_sigma=1.5,
            flags=0,
        )
        flow_bwd = cv2.calcOpticalFlowFarneback(
            gray_t1,
            gray_t,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=21,
            iterations=5,
            poly_n=7,
            poly_sigma=1.5,
            flags=0,
        )

        nn_forward_idx, nn_forward_dist = _flann_knn_1nn(frame_t1.uv, frame_t.uv)
        nn_reverse_idx, _ = _flann_knn_1nn(frame_t.uv, frame_t1.uv)

        if frame_t.uv.shape[0] == 0 or frame_t1.uv.shape[0] == 0 or nn_forward_idx.size == 0:
            print(f"{frame_t.name}->{frame_t1.name}: insufficient_points_for_temporal_statistics")
            continue

        query_idx = np.arange(frame_t.uv.shape[0], dtype=np.int32)
        mutual = nn_reverse_idx[nn_forward_idx] == query_idx
        distance_ok = nn_forward_dist <= TEMPORAL_STATS_MATCH_MAX_DIST_PX

        matched_uv_t = frame_t.uv[mutual & distance_ok]
        matched_uv_t1 = frame_t1.uv[nn_forward_idx[mutual & distance_ok]]

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

        if matched_uv_t.shape[0] == 0:
            print(f"{frame_t.name}->{frame_t1.name}: no_matches_after_motion_filter")
            continue

        temporal_error = np.linalg.norm(lidar_disp - img_disp, axis=1)
        temporal_error = _filter_top_quantile(temporal_error, TEMPORAL_STATS_OUTLIER_QUANTILE)

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

        pair_errors.append(temporal_error)
        baseline_errors.append(self_consistency_error)

        pair_mean = float(np.mean(temporal_error)) if temporal_error.size else float("inf")
        pair_median = float(np.median(temporal_error)) if temporal_error.size else float("inf")
        pair_std = float(np.std(temporal_error)) if temporal_error.size else float("inf")
        print(
            f"{frame_t.name}->{frame_t1.name}: "
            f"mean_error_px={pair_mean:.6f} "
            f"median_error_px={pair_median:.6f} "
            f"std_error_px={pair_std:.6f} "
            f"matches={temporal_error.size}"
        )

    global_temporal = np.concatenate(pair_errors) if pair_errors else np.empty((0,), dtype=np.float32)
    global_baseline = np.concatenate(baseline_errors) if baseline_errors else np.empty((0,), dtype=np.float32)

    temporal_mean = float(np.mean(global_temporal)) if global_temporal.size else float("inf")
    temporal_median = float(np.median(global_temporal)) if global_temporal.size else float("inf")
    baseline_mean = float(np.mean(global_baseline)) if global_baseline.size else float("inf")
    baseline_median = float(np.median(global_baseline)) if global_baseline.size else float("inf")

    temporal_statistical = (
        global_temporal.size > 0
        and global_baseline.size > 0
        and temporal_mean > baseline_mean * 2.0
        and temporal_median > baseline_median * 2.0
    )

    print(f"Temporal inconsistency mean error: {temporal_mean:.6f} px")
    print(f"Temporal inconsistency median error: {temporal_median:.6f} px")
    print(f"Image self-consistency error: {baseline_mean:.6f} px")
    print("Temporal distortion (statistical): " + ("YES" if temporal_statistical else "NO"))

    return {
        "temporal_mean_px": temporal_mean,
        "temporal_median_px": temporal_median,
        "baseline_mean_px": baseline_mean,
        "baseline_median_px": baseline_median,
        "detected": temporal_statistical,
    }


def run_tests():
    tr_velo_to_cam = parse_calib_velo_to_cam(os.path.join(DATASET_PATH, "calib_velo_to_cam.txt"))
    r_rect, p_rect = parse_calib_cam_to_cam(os.path.join(DATASET_PATH, "calib_cam_to_cam.txt"), camera_id="02")

    print("--- TEST 1: MATRIX VALIDATION ---")
    print("Tr_velo_to_cam:")
    print(tr_velo_to_cam)
    print("R_rect:")
    print(r_rect)
    print("P_rect:")
    print(p_rect)

    image_sample = load_image(os.path.join(IMAGE_DIR, sorted(os.listdir(IMAGE_DIR))[0]))
    image_h, image_w = image_sample.shape[:2]

    tr_last_row_ok = np.allclose(tr_velo_to_cam[3], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64))
    tr_translation = tr_velo_to_cam[:3, 3]
    tr_translation_small = np.all(np.abs(tr_translation) < 2.0)
    rect_identity_delta = float(np.max(np.abs(r_rect[:3, :3] - np.eye(3, dtype=np.float64))))
    fx = float(p_rect[0, 0])
    fy = float(p_rect[1, 1])
    cx = float(p_rect[0, 2])
    cy = float(p_rect[1, 2])
    focal_ok = 700.0 <= fx <= 1200.0 and 700.0 <= fy <= 1200.0
    center_ok = 0.0 <= cx <= image_w and 0.0 <= cy <= image_h

    print(f"Tr_last_row_ok: {tr_last_row_ok}")
    print(f"Tr_translation_m: [{tr_translation[0]:.6f}, {tr_translation[1]:.6f}, {tr_translation[2]:.6f}]")
    print(f"Tr_translation_small: {tr_translation_small}")
    print(f"R_rect_identity_max_abs_diff: {rect_identity_delta:.9f}")
    print(f"P_rect_fx_fy: [{fx:.6f}, {fy:.6f}]")
    print(f"P_rect_cx_cy: [{cx:.6f}, {cy:.6f}]")
    print(f"Image_size_wh: [{image_w}, {image_h}]")

    matrix_pass = all([tr_last_row_ok, tr_translation_small, focal_ok, center_ok, rect_identity_delta < 0.05])

    print("\n--- TEST 2: POINT DISTRIBUTION CHECK ---")
    frame_pairs = _list_first_n_pairs(5)
    traced_frames = []
    distribution_pass = True
    for image_name, lidar_name in frame_pairs:
        frame = _project_with_full_trace(image_name, lidar_name, tr_velo_to_cam, r_rect, p_rect)
        traced_frames.append(frame)
        survive_ratio = frame.stats["after_proj_filter"] / max(frame.stats["input_points"], 1)
        print(
            f"{frame.name}: "
            f"input_points={frame.stats['input_points']} "
            f"after_cam_filter={frame.stats['after_cam_filter']} "
            f"after_proj_filter={frame.stats['after_proj_filter']} "
            f"in_frame={frame.stats['in_frame']} "
            f"survival_ratio={survive_ratio:.6f}"
        )
        if survive_ratio < SURVIVAL_PASS_RATIO:
            distribution_pass = False

    print("\n--- TEST 3: EDGE ALIGNMENT ERROR (NUMERICAL) ---")
    edge_means = []
    edge_medians = []
    edge_kept = []
    for frame in traced_frames:
        _, dist, grad_mag = _distance_transform_from_edges(frame.image)
        if frame.uv_int.size == 0:
            dists = np.array([], dtype=np.float32)
            keep_ratio = 0.0
        else:
            strong_points = grad_mag[frame.uv_int[:, 1], frame.uv_int[:, 0]] >= EDGE_GRADIENT_THRESHOLD
            keep_ratio = float(np.mean(strong_points))
            dists = dist[frame.uv_int[strong_points, 1], frame.uv_int[strong_points, 0]] if np.any(strong_points) else np.array([], dtype=np.float32)
        mean_dist = float(np.mean(dists)) if dists.size else float("inf")
        median_dist = float(np.median(dists)) if dists.size else float("inf")
        edge_means.append(mean_dist)
        edge_medians.append(median_dist)
        edge_kept.append(keep_ratio)
        print(
            f"{frame.name}: median_distance_px={median_dist:.6f} "
            f"mean_distance_px={mean_dist:.6f} strong_edge_point_ratio={keep_ratio:.6f}"
        )

    edge_mean_global = float(np.mean(edge_means))
    edge_median_global = float(np.median(edge_medians))
    edge_keep_global = float(np.mean(edge_kept))
    print(f"edge_strong_point_ratio_global={edge_keep_global:.6f}")
    print(f"edge_mean_global_px={edge_mean_global:.6f}")
    print(f"edge_median_global_px={edge_median_global:.6f}")
    edge_pass = (edge_median_global < EDGE_MEDIAN_PASS_PX) and (edge_mean_global < EDGE_MEAN_FAIL_PX)
    edge_fail = edge_mean_global > EDGE_MEAN_FAIL_PX

    print("\n--- TEST 4: STATIC STRUCTURE CHECK ---")
    static_spreads = []
    static_tilts = []
    static_pass = True
    static_candidates_found = 0
    best_candidates = _select_tracked_candidates(traced_frames[:3])
    for idx, frame in enumerate(traced_frames[:3]):
        candidates = _detect_vertical_structures(frame)
        if not candidates:
            static_pass = False
            print(f"{frame.name}: no_vertical_structure_detected")
            continue
        best = best_candidates[idx]
        if best is None:
            best = candidates[0]
        static_candidates_found += 1
        static_spreads.append(best["spread_px"])
        static_tilts.append(abs(best["tilt_dxdy"]))
        print(
            f"{frame.name}: "
            f"vertical_points={best['count']} "
            f"spread_px={best['spread_px']:.6f} "
            f"tilt_dxdy={best['tilt_dxdy']:.6f} "
            f"vertical_span_px={best['vertical_span_px']:.6f} "
            f"x_mid={best['x_mid']:.6f}"
        )
        if best["spread_px"] > VERTICAL_SPREAD_PASS_PX:
            static_pass = False

    if static_spreads:
        print(f"static_spread_mean_px={float(np.mean(static_spreads)):.6f}")
        print(f"static_tilt_mean_abs={float(np.mean(static_tilts)):.6f}")
    else:
        print("static_spread_mean_px=inf")
        print("static_tilt_mean_abs=inf")

    static_fail = bool(static_spreads) and float(np.mean(static_spreads)) > VERTICAL_SPREAD_FAIL_PX

    print("\n--- TEST 5: TEMPORAL DISTORTION CHECK ---")
    temporal_detected = False
    temporal_reports = []
    temporal_frames = traced_frames[:3]
    for frame, candidate in zip(temporal_frames, best_candidates):
        if candidate is None:
            temporal_reports.append((frame.name, None))
            print(f"{frame.name}: temporal_profile=unavailable")
            continue
        profile = _temporal_profile(candidate, frame)
        temporal_reports.append((frame.name, profile))
        if profile is None:
            print(f"{frame.name}: temporal_profile=insufficient_points")
            continue
        x_center, y_centers, residuals = profile
        finite = np.isfinite(residuals)
        valid_residuals = residuals[finite]
        if valid_residuals.size >= 3:
            has_negative = np.min(valid_residuals) < -TEMPORAL_RESIDUAL_THRESHOLD_PX
            has_positive = np.max(valid_residuals) > TEMPORAL_RESIDUAL_THRESHOLD_PX
            zero_crossings = np.count_nonzero(np.diff(np.signbit(valid_residuals)))
            if has_negative and has_positive and zero_crossings >= 1:
                temporal_detected = True
        residual_text = ", ".join("nan" if not np.isfinite(v) else f"{v:.6f}" for v in residuals)
        y_text = ", ".join(f"{v:.6f}" for v in y_centers)
        print(f"{frame.name}: x_center_px={x_center:.6f} y_bins=[{y_text}] residuals_px=[{residual_text}]")

    print("\n--- TEST 6: REPROJECTION CONSISTENCY ---")
    reprojection_rmses = []
    total_transform_inv = np.linalg.inv(r_rect @ tr_velo_to_cam)
    tx = float(p_rect[0, 3])
    ty = float(p_rect[1, 3])
    tz = float(p_rect[2, 3])

    for frame in traced_frames[:3]:
        rect_cam_valid = frame.rect_cam[:, frame.valid_mask_input][:, frame.in_frame_mask_cam]
        projected_valid = frame.projected[:, frame.valid_mask_input][:, frame.in_frame_mask_cam]
        if rect_cam_valid.shape[1] == 0:
            reprojection_rmses.append(float("inf"))
            print(f"{frame.name}: reprojection_rmse_m=inf")
            continue

        Z = rect_cam_valid[2, :]
        denom = Z + tz
        u = projected_valid[0, :] / projected_valid[2, :]
        v = projected_valid[1, :] / projected_valid[2, :]
        X = (u * denom - cx * Z - tx) / fx
        Y = (v * denom - cy * Z - ty) / fy
        rect_reconstructed = np.vstack((X, Y, Z, np.ones_like(Z)))
        lidar_reconstructed = total_transform_inv @ rect_reconstructed
        lidar_original = frame.lidar_h[:, frame.valid_mask_input][:, frame.in_frame_mask_cam]
        diff = lidar_reconstructed[:3, :] - lidar_original[:3, :]
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        reprojection_rmses.append(rmse)
        print(f"{frame.name}: reprojection_rmse_m={rmse:.12e}")

    reprojection_rmse_global = float(np.mean(reprojection_rmses))
    print(f"reprojection_rmse_global_m={reprojection_rmse_global:.12e}")
    reprojection_pass = reprojection_rmse_global < REPROJECTION_PASS_RMSE
    reprojection_fail = reprojection_rmse_global > REPROJECTION_FAIL_RMSE

    print("\n--- TEST 7: CAMERA CONSISTENCY ---")
    image_02_exists = os.path.isdir(IMAGE_DIR)
    p_rect_02_matches = np.allclose(
        p_rect,
        parse_calib_cam_to_cam(os.path.join(DATASET_PATH, "calib_cam_to_cam.txt"), camera_id="02")[1],
    )
    p_rect_03_differs = not np.allclose(
        p_rect,
        parse_calib_cam_to_cam(os.path.join(DATASET_PATH, "calib_cam_to_cam.txt"), camera_id="03")[1],
    )
    print(f"image_02_exists: {image_02_exists}")
    print(f"P_rect_02_used: {p_rect_02_matches}")
    print(f"P_rect_03_mismatch_confirmed: {p_rect_03_differs}")
    camera_consistency_pass = image_02_exists and p_rect_02_matches and p_rect_03_differs

    print("\n--- TEST 8: VISUAL MODES ---")
    _save_debug_modes(tr_velo_to_cam, r_rect, p_rect, frame_pairs)
    print("visual_mode_1: near_points_only_lt_30m")
    print("visual_mode_2: far_points_only_ge_30m")
    print("visual_mode_3: edge_aligned_points_only")

    print("\n--- TEST 9: DISTORTION vs ERROR CLASSIFICATION ---")
    physically_correct = edge_pass and reprojection_pass and static_pass and temporal_detected
    bug_detected = not physically_correct
    if physically_correct:
        print("classification: PHYSICALLY_CORRECT_LIDAR_TEMPORAL_DISTORTION")
    else:
        print("classification: BUG")
        print(
            "classification_inputs: "
            f"edge_pass={edge_pass} "
            f"reprojection_pass={reprojection_pass} "
            f"static_pass={static_pass} "
            f"temporal_detected={temporal_detected}"
        )

    projection_correctness = matrix_pass and distribution_pass and edge_pass and not reprojection_fail
    calibration_correctness = matrix_pass and camera_consistency_pass and not static_fail and edge_pass
    ready_for_research = physically_correct and projection_correctness and calibration_correctness
    temporal_stats_result = temporal_statistics_test(tr_velo_to_cam, r_rect, p_rect)

    print("\nProjection correctness: " + ("PASS" if projection_correctness else "FAIL"))
    print("Calibration correctness: " + ("PASS" if calibration_correctness else "FAIL"))
    print("Temporal distortion detected: " + ("YES" if temporal_detected else "NO"))
    print(
        "Temporal distortion detected (statistical): "
        + ("YES" if temporal_stats_result["detected"] else "NO")
    )
    print("Bug detected: " + ("YES" if bug_detected else "NO"))
    print("Ready for research stage: " + ("YES" if ready_for_research else "NO"))


if __name__ == "__main__":
    run_tests()
