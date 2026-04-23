import cv2
import numpy as np


def project_lidar_to_image(lidar_xyz, tr_velo_to_cam, r_rect, p_rect, image_shape):
    """
    Project LiDAR XYZ points into the image plane.

    Returns:
    - uv: (M, 2) float32 image coordinates
    - depth: (M,) float32 depth values in camera coordinates
    - stats: dict for debugging counts
    """
    stats = {
        "input_points": int(lidar_xyz.shape[0]),
        "after_cam_filter": 0,
        "after_proj_filter": 0,
        "in_frame": 0,
    }

    if lidar_xyz.size == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32), stats

    xyz = np.asarray(lidar_xyz, dtype=np.float64)
    tr_velo_to_cam = np.asarray(tr_velo_to_cam, dtype=np.float64)
    r_rect = np.asarray(r_rect, dtype=np.float64)
    p_rect = np.asarray(p_rect, dtype=np.float64)

    if tr_velo_to_cam.shape != (4, 4):
        raise ValueError(f"tr_velo_to_cam must be (4,4), got {tr_velo_to_cam.shape}")
    if r_rect.shape != (4, 4):
        raise ValueError(f"r_rect must be (4,4), got {r_rect.shape}")
    if p_rect.shape != (3, 4):
        raise ValueError(f"p_rect must be (3,4), got {p_rect.shape}")
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"lidar_xyz must be (N,3), got {xyz.shape}")

    # LiDAR homogeneous: (N,3) -> (N,4) -> (4,N)
    ones = np.ones((xyz.shape[0], 1), dtype=np.float64)
    lidar_h = np.hstack((xyz, ones)).T

    rectified_tf = r_rect @ tr_velo_to_cam
    proj = p_rect @ rectified_tf

    # KITTI projection: img = P_rect @ R_rect @ Tr_velo_to_cam @ lidar_h
    cam = rectified_tf @ lidar_h

    # Remove invalid/behind-camera points before projection
    valid_cam = (cam[2, :] > 0.0) & np.isfinite(cam).all(axis=0)
    stats["after_cam_filter"] = int(np.count_nonzero(valid_cam))

    if stats["after_cam_filter"] == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32), stats

    cam = cam[:, valid_cam]
    lidar_h = lidar_h[:, valid_cam]

    # Project with the single composed matrix: (3,4) @ (4,N) -> (3,N)
    img = proj @ lidar_h

    # Remove invalid projected points and z==0 before normalization
    valid_img = (img[2, :] != 0.0) & np.isfinite(img).all(axis=0)
    img = img[:, valid_img]
    cam = cam[:, valid_img]
    stats["after_proj_filter"] = int(img.shape[1])

    if img.shape[1] == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32), stats

    # Normalize using projected depth z = img[2]
    z = img[2, :]
    u = img[0, :] / z
    v = img[1, :] / z

    finite_uv = np.isfinite(u) & np.isfinite(v)
    h, w = image_shape[:2]
    in_frame = finite_uv & (u >= 0.0) & (u < w) & (v >= 0.0) & (v < h)
    stats["in_frame"] = int(np.count_nonzero(in_frame))

    uv = np.column_stack((u[in_frame], v[in_frame])).astype(np.float32)
    depth = cam[2, in_frame].astype(np.float32)
    return uv, depth, stats


def overlay_points(image_bgr, uv, depth, max_points=12000, radius=1):
    """Draw depth-colored projected points on an image."""
    canvas = image_bgr.copy()
    if uv.shape[0] == 0:
        return canvas

    valid = (
        np.isfinite(uv[:, 0])
        & np.isfinite(uv[:, 1])
        & np.isfinite(depth)
    )
    uv = uv[valid]
    depth = depth[valid]
    if uv.shape[0] == 0:
        return canvas

    if uv.shape[0] > max_points:
        idx = np.linspace(0, uv.shape[0] - 1, num=max_points, dtype=np.int32)
        uv = uv[idx]
        depth = depth[idx]

    d_min = float(np.min(depth))
    d_max = float(np.max(depth))

    if d_max - d_min < 1e-12:
        depth_norm = np.zeros_like(depth, dtype=np.uint8)
    else:
        depth_norm = ((depth - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)

    colors = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET).reshape(-1, 3)

    pts = np.rint(uv).astype(np.int32)
    for p, c in zip(pts, colors):
        cv2.circle(canvas, (int(p[0]), int(p[1])), radius, (int(c[0]), int(c[1]), int(c[2])), -1)

    return canvas
