import numpy as np


def move_lidar_points_weighted(uv, depth, flow, conf, conf_thresh=0.5):
    """Move LiDAR points using RGB flow only in high-confidence, low-noise regions."""

    uv = np.asarray(uv, dtype=np.float32)
    depth = np.asarray(depth, dtype=np.float32)
    flow = np.nan_to_num(np.asarray(flow, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    conf = np.nan_to_num(np.asarray(conf, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    if uv.size == 0 or flow.ndim != 3:
        print("Total points:", len(uv))
        print("Valid moved points:", 0)
        print("Movement ratio:", 0.0)
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

    u = uv[:, 0]
    v = uv[:, 1]

    u_idx = np.rint(u).astype(np.int32)
    v_idx = np.rint(v).astype(np.int32)

    h, w = flow.shape[:2]

    in_bounds = (
        (u_idx >= 0) & (u_idx < w) &
        (v_idx >= 0) & (v_idx < h)
    )

    if not np.any(in_bounds):
        print("Total points:", len(uv))
        print("Valid moved points:", 0)
        print("Movement ratio:", 0.0)
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

    u = u[in_bounds]
    v = v[in_bounds]
    depth = depth[in_bounds]
    u_idx = u_idx[in_bounds]
    v_idx = v_idx[in_bounds]

    dx = flow[v_idx, u_idx, 0]
    dy = flow[v_idx, u_idx, 1]
    c = conf[v_idx, u_idx]

    flow_mag = np.linalg.norm(flow, axis=2)
    sampled_flow_mag = flow_mag[v_idx, u_idx]
    scale = 0.5

    valid_mask = (c > 0.5) & (sampled_flow_mag < 5.0)

    print("Total points:", len(uv))
    print("Valid moved points:", np.sum(valid_mask))
    print("Movement ratio:", np.sum(valid_mask) / len(uv))

    u_new = u.copy()
    v_new = v.copy()

    u_new[valid_mask] += scale * dx[valid_mask]
    v_new[valid_mask] += scale * dy[valid_mask]

    moved_in_bounds = (
        np.isfinite(u_new) & np.isfinite(v_new) &
        (u_new >= 0.0) & (u_new < w) &
        (v_new >= 0.0) & (v_new < h)
    )

    uv_new = np.stack([u_new[moved_in_bounds], v_new[moved_in_bounds]], axis=1)
    depth_new = depth[moved_in_bounds]

    return uv_new, depth_new
