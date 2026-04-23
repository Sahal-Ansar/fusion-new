# Fixed: 2026-04-24 — see pipeline_fixes.md for reasoning
import numpy as np


def move_lidar_points_weighted(uv, depth, flow, conf, conf_thresh=0.5):
    """
    Apply event-guided motion compensation to projected LiDAR points.

    For each projected LiDAR point, motion correction is applied if and
    only if:
      1. The point lies within a high-confidence event region (C > conf_thresh)
      2. The local optical flow magnitude is meaningful (> 0.5 px) indicating
         real motion, not sensor noise
      3. The local optical flow magnitude is plausible (< 30.0 px) filtering
         optical flow estimation failures at occlusion boundaries

    The correction displacement is scaled by alpha=0.5, which approximates
    the mean temporal offset ratio for a 10 Hz LiDAR (mean scan delay = 50ms,
    frame interval = 100ms, so mean ratio = 0.5).

    Returns:
      uv_new: (N, 2) float32 — corrected positions for ALL input points.
              Points that were not moved retain their original position.
              This preserves 1-to-1 index correspondence with input uv.
      depth_new: (N,) float32 — depths corresponding to uv_new.
    """
    uv = np.asarray(uv, dtype=np.float32)
    depth = np.asarray(depth, dtype=np.float32)
    flow = np.nan_to_num(
        np.asarray(flow, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    conf = np.nan_to_num(
        np.asarray(conf, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )

    # Return copies of original positions if no valid input
    if uv.size == 0 or flow.ndim != 3 or flow.shape[2] != 2:
        return uv.copy(), depth.copy()

    h, w = flow.shape[:2]

    u = uv[:, 0]
    v = uv[:, 1]

    u_idx = np.rint(u).astype(np.int32)
    v_idx = np.rint(v).astype(np.int32)

    in_bounds = (
        (u_idx >= 0) & (u_idx < w) &
        (v_idx >= 0) & (v_idx < h) &
        np.isfinite(u) & np.isfinite(v)
    )

    # Start with copies of original positions (unmoved points stay in place)
    u_new = u.copy()
    v_new = v.copy()

    if not np.any(in_bounds):
        print(f"Total points: {len(uv)}")
        print(f"Valid moved points: 0")
        print(f"Movement ratio: 0.0000")
        return np.stack([u_new, v_new], axis=1), depth.copy()

    u_b = u[in_bounds]
    v_b = v[in_bounds]
    u_idx_b = u_idx[in_bounds]
    v_idx_b = v_idx[in_bounds]

    dx = flow[v_idx_b, u_idx_b, 0]
    dy = flow[v_idx_b, u_idx_b, 1]
    c = conf[v_idx_b, u_idx_b]

    flow_mag = np.sqrt(dx ** 2 + dy ** 2)

    # Gate: must be in a confident event region AND have meaningful,
    # plausible flow. Cap raised to 30.0 px to allow fast ego-motion.
    FLOW_MIN = 0.5   # px — below this is noise/stationary
    FLOW_MAX = 30.0  # px — above this is likely flow estimation failure
    ALPHA = 0.5      # temporal offset ratio (mean delay / frame interval)

    valid_mask = (c > conf_thresh) & (flow_mag > FLOW_MIN) & (flow_mag < FLOW_MAX)

    n_moved = int(np.sum(valid_mask))
    print(f"Total points: {len(uv)}")
    print(f"Valid moved points: {n_moved}")
    print(f"Movement ratio: {n_moved / len(uv):.4f}")

    # Apply scaled correction only to valid points
    original_indices = np.flatnonzero(in_bounds)
    valid_global = original_indices[valid_mask]

    u_new[valid_global] = u_b[valid_mask] + ALPHA * dx[valid_mask]
    v_new[valid_global] = v_b[valid_mask] + ALPHA * dy[valid_mask]

    # Clamp corrected positions to image bounds (prevents points going off-screen)
    u_new = np.clip(u_new, 0.0, w - 1.0)
    v_new = np.clip(v_new, 0.0, h - 1.0)

    uv_new = np.stack([u_new, v_new], axis=1).astype(np.float32)
    return uv_new, depth.copy()
