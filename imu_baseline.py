"""
IMU-based LiDAR de-skewing baseline for KITTI raw sequences.

Uses ego-velocity from oxts IMU/GPS data to approximate per-point
temporal offsets within a LiDAR sweep and correct projected image
coordinates accordingly. This serves as a physics-informed baseline
that does not use optical flow, for comparison against the
event-guided method.
"""

import os
from typing import Dict, Tuple

import numpy as np


_OXTS_COLUMN_NAMES = (
    "lat", "lon", "alt",
    "roll", "pitch", "yaw",
    "vn", "ve",
    "vf", "vl", "vu",
)


def has_oxts(dataset_path: str) -> bool:
    """Return True iff ``<dataset_path>/oxts/data`` contains a .txt file.

    Safe to call on any string — never raises. Used by validation
    pipelines as a precondition for invoking the IMU baseline.

    Args:
        dataset_path: KITTI raw-sequence root directory.

    Returns:
        bool: True if at least one ``.txt`` file is present in
        ``oxts/data`` under ``dataset_path``; False otherwise
        (including when the directory is missing or unreadable).
    """
    try:
        oxts_data_dir = os.path.join(str(dataset_path), "oxts", "data")
        if not os.path.isdir(oxts_data_dir):
            return False
        for entry in os.listdir(oxts_data_dir):
            if entry.endswith(".txt"):
                return True
        return False
    except OSError:
        return False


def load_oxts(oxts_dir: str, frame_idx: int) -> Dict[str, float]:
    """Load a single KITTI oxts record by frame index.

    KITTI oxts ``.txt`` files contain one row of space-separated
    floats. The first 11 column indices used here are::

        0: lat   1: lon   2: alt
        3: roll  4: pitch 5: yaw   (radians)
        6: vn    7: ve    (north/east velocity, m/s)
        8: vf    9: vl    10: vu   (forward/leftward/upward, m/s)

    Args:
        oxts_dir: directory containing the per-frame ``.txt`` files
            (typically ``<dataset>/oxts/data``).
        frame_idx: zero-padded frame index used to construct the
            filename ``f"{frame_idx:010d}.txt"``.

    Returns:
        Dict[str, float] with keys lat, lon, alt, roll, pitch, yaw,
        vn, ve, vf, vl, vu.

    Raises:
        FileNotFoundError: if the oxts file is missing.
        ValueError: if the file has fewer than 11 columns.
    """
    filename = f"{int(frame_idx):010d}.txt"
    path = os.path.join(str(oxts_dir), filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"oxts file not found: {path}")

    row = np.loadtxt(path, dtype=np.float64)
    if row.ndim == 0:
        raise ValueError(f"oxts file has no values: {path}")
    if row.ndim > 1:
        if row.shape[0] == 1:
            row = row[0]
        else:
            raise ValueError(
                f"oxts file {path} has unexpected shape {row.shape}"
            )
    if row.size < 11:
        raise ValueError(
            f"oxts file {path} has {row.size} columns, expected at least 11"
        )

    return {name: float(row[i]) for i, name in enumerate(_OXTS_COLUMN_NAMES)}


def estimate_ego_displacement_px(
    oxts_t: Dict[str, float],
    oxts_t1: Dict[str, float],
    p_rect: np.ndarray,
    dt: float = 0.1,
) -> Tuple[float, float]:
    """Estimate expected pixel displacement due to ego-motion between frames.

    Uses a single reference 3-D point at 10 m ahead of the camera and
    re-projects it after a small ego-displacement built from the mean
    of the per-frame forward/leftward velocities. Convention used (per
    KITTI camera frame):

        forward world motion -> +Z camera
        leftward world motion -> -X camera

    so the displaced reference point is::

        P0 = [0,        0,    10.0,            1]
        P1 = [-vl*dt,   0,    10.0 + vf*dt,    1]

    Args:
        oxts_t:  dict from ``load_oxts`` for the source frame.
        oxts_t1: dict from ``load_oxts`` for the target frame.
        p_rect:  (3, 4) projection matrix (KITTI ``P_rect_*``).
        dt:      time gap between the two frames in seconds.

    Returns:
        (du, dv): float pixel displacement of the reference point.
        Returns (0.0, 0.0) if either projected Z is non-positive.

    Raises:
        ValueError: if ``p_rect`` is not (3, 4).
    """
    p_rect = np.asarray(p_rect, dtype=np.float64)
    if p_rect.shape != (3, 4):
        raise ValueError(f"p_rect must be (3, 4), got {p_rect.shape}")

    vf = 0.5 * (float(oxts_t["vf"]) + float(oxts_t1["vf"]))
    vl = 0.5 * (float(oxts_t["vl"]) + float(oxts_t1["vl"]))
    dt_f = float(dt)

    p0 = np.array([0.0, 0.0, 10.0, 1.0], dtype=np.float64)
    p1 = np.array([-vl * dt_f, 0.0, 10.0 + vf * dt_f, 1.0], dtype=np.float64)

    proj0 = p_rect @ p0
    proj1 = p_rect @ p1

    if proj0[2] <= 0.0 or proj1[2] <= 0.0:
        return 0.0, 0.0

    u0 = proj0[0] / proj0[2]
    v0 = proj0[1] / proj0[2]
    u1 = proj1[0] / proj1[2]
    v1 = proj1[1] / proj1[2]
    return float(u1 - u0), float(v1 - v0)


def imu_deskew_projection(
    uv: np.ndarray,
    depth: np.ndarray,
    lidar_xyz: np.ndarray,
    oxts_t: Dict[str, float],
    oxts_t1: Dict[str, float],
    p_rect: np.ndarray,
    image_shape: Tuple[int, ...],
    dt: float = 0.1,
) -> np.ndarray:
    """Apply IMU-derived per-point de-skew to projected LiDAR coordinates.

    Each LiDAR point is assigned a per-point temporal ratio in [0, 1]
    derived from its azimuth angle (sweep timing proxy):

        az = arctan2(y, x) in [-pi, pi]
        temporal_ratio = (az + pi) / (2 * pi)

    The ego-induced reference displacement (du, dv) at 10 m ahead is
    then scaled by ``temporal_ratio * ALPHA`` (with ALPHA = 0.5, the
    same mean-offset convention used by the event-guided pipeline)
    and added to each point's pixel coordinates.

    Args:
        uv:         (N, 2) float32 left-image projected coordinates.
        depth:      (N,) float32 depths in metres, row-aligned with uv.
        lidar_xyz:  (N, 3) float64 ORIGINAL LiDAR-frame 3-D points,
                    same index order as ``uv``.
        oxts_t:     ego-state dict for the source frame.
        oxts_t1:    ego-state dict for the target frame.
        p_rect:     (3, 4) left projection matrix.
        image_shape: shape tuple of the image (only H, W are used).
        dt:         frame interval in seconds.

    Returns:
        uv_imu: (N, 2) float32, IMU-de-skewed pixel coordinates,
        clamped to the image bounds. If ``len(lidar_xyz) != len(uv)``
        a warning is printed and the input ``uv`` is returned unchanged.
    """
    uv = np.asarray(uv, dtype=np.float32)
    depth = np.asarray(depth, dtype=np.float32)
    lidar_xyz = np.asarray(lidar_xyz, dtype=np.float64)

    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError(f"uv must be (N, 2), got {uv.shape}")
    if depth.ndim != 1 or depth.shape[0] != uv.shape[0]:
        raise ValueError(
            f"depth must be (N,) matching uv length; "
            f"got depth {depth.shape}, uv {uv.shape}"
        )
    if len(image_shape) < 2:
        raise ValueError(
            f"image_shape must have at least 2 dims, got {image_shape}"
        )

    if lidar_xyz.shape[0] != uv.shape[0]:
        print(
            f"[IMU] Warning: lidar_xyz length {lidar_xyz.shape[0]} "
            f"!= uv length {uv.shape[0]}; returning uv unchanged."
        )
        return uv.copy()
    if lidar_xyz.ndim != 2 or lidar_xyz.shape[1] != 3:
        print(
            "[IMU] Warning: lidar_xyz must be (N, 3), got "
            f"{lidar_xyz.shape}; returning uv unchanged."
        )
        return uv.copy()

    h, w = int(image_shape[0]), int(image_shape[1])

    az = np.arctan2(lidar_xyz[:, 1], lidar_xyz[:, 0])
    temporal_ratio = ((az + np.pi) / (2.0 * np.pi)).astype(np.float32)

    du, dv = estimate_ego_displacement_px(oxts_t, oxts_t1, p_rect, dt)
    alpha = 0.5

    u_new = uv[:, 0] + temporal_ratio * (np.float32(du) * np.float32(alpha))
    v_new = uv[:, 1] + temporal_ratio * (np.float32(dv) * np.float32(alpha))

    u_new = np.clip(u_new, 0.0, float(w - 1))
    v_new = np.clip(v_new, 0.0, float(h - 1))

    return np.stack([u_new, v_new], axis=1).astype(np.float32)


def compare_imu_correction(
    uv_original: np.ndarray,
    uv_imu: np.ndarray,
    uv_event_guided: np.ndarray,
    depth: np.ndarray,
    image_bgr: np.ndarray,
) -> Dict[str, object]:
    """Three-way EAS comparison: original vs IMU baseline vs event-guided.

    Computes the Edge Alignment Score (EAS) for each of the three
    projections against the same RGB image and reports IMU and
    event-guided improvements as percentages relative to the original
    EAS. Imports ``metrics.edge_alignment_score`` lazily to avoid a
    circular import. Never raises — any exception returns a
    None-valued result dict.

    Args:
        uv_original:     (N, 2) uncorrected left-image coordinates.
        uv_imu:          (N, 2) IMU-baseline left-image coordinates.
        uv_event_guided: (N, 2) event-guided left-image coordinates.
        depth:           (N,) depths in metres shared by all three uv
            sets (typically the original projection's depths, since
            none of the corrections change physical depth).
        image_bgr:       (H, W, 3) uint8 BGR reference image.

    Returns:
        Dict with keys ``original_eas``, ``imu_eas``,
        ``event_guided_eas``, ``imu_improvement_pct``,
        ``event_guided_improvement_pct``. All values are None on
        failure or if the original score is non-positive.
    """
    null_result = {
        "original_eas": None,
        "imu_eas": None,
        "event_guided_eas": None,
        "imu_improvement_pct": None,
        "event_guided_improvement_pct": None,
    }
    try:
        from metrics import edge_alignment_score  # local import: avoid cycle

        score_original, _, _ = edge_alignment_score(
            uv_original, depth, image_bgr
        )
        score_imu, _, _ = edge_alignment_score(uv_imu, depth, image_bgr)
        score_event, _, _ = edge_alignment_score(
            uv_event_guided, depth, image_bgr
        )

        if score_original > 0.0:
            imu_pct = (score_imu - score_original) / score_original * 100.0
            event_pct = (
                (score_event - score_original) / score_original * 100.0
            )
        else:
            imu_pct = float("nan")
            event_pct = float("nan")

        print("IMU BASELINE COMPARISON")
        print(f"Original EAS:      {score_original:.6f}")
        if np.isfinite(imu_pct):
            print(f"IMU Baseline EAS:  {score_imu:.6f}  ({imu_pct:+.2f}%)")
        else:
            print(f"IMU Baseline EAS:  {score_imu:.6f}  (n/a)")
        if np.isfinite(event_pct):
            print(
                f"Event-Guided EAS:  {score_event:.6f}  ({event_pct:+.2f}%)"
            )
        else:
            print(f"Event-Guided EAS:  {score_event:.6f}  (n/a)")

        return {
            "original_eas": float(score_original),
            "imu_eas": float(score_imu),
            "event_guided_eas": float(score_event),
            "imu_improvement_pct": (
                float(imu_pct) if np.isfinite(imu_pct) else None
            ),
            "event_guided_improvement_pct": (
                float(event_pct) if np.isfinite(event_pct) else None
            ),
        }
    except Exception as exc:
        print(f"[IMU] Warning: compare_imu_correction failed: {exc}")
        return null_result
