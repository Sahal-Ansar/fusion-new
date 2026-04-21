import numpy as np


def move_lidar_points_weighted(uv, depth, flow, conf, conf_thresh=0.2):
    """Move LiDAR points using RGB flow but only where events exist."""

    uv = np.asarray(uv, dtype=np.float32)
    depth = np.asarray(depth, dtype=np.float32)
    flow = np.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)
    conf = np.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0)

    if uv.size == 0 or flow.ndim != 3:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

    u = uv[:, 0]
    v = uv[:, 1]

    u_idx = np.rint(u).astype(np.int32)
    v_idx = np.rint(v).astype(np.int32)

    h, w = flow.shape[:2]

    valid = (
        (u_idx >= 0) & (u_idx < w) &
        (v_idx >= 0) & (v_idx < h)
    )

    if not np.any(valid):
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

    u = u[valid]
    v = v[valid]
    depth = depth[valid]
    u_idx = u_idx[valid]
    v_idx = v_idx[valid]

    dx = flow[v_idx, u_idx, 0]
    dy = flow[v_idx, u_idx, 1]

    c = conf[v_idx, u_idx]

    strong = c > conf_thresh

    u_new = u.copy()
    v_new = v.copy()

    u_new[strong] += dx[strong]
    v_new[strong] += dy[strong]

    uv_new = np.stack([u_new, v_new], axis=1)

    return uv_new, depth