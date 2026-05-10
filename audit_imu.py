"""Numerical audit of the IMU baseline against uncorrected and event-guided
projections, on a fast and a slow KITTI sequence.

Run from the repo root: ``python audit_imu.py`` (no CLI arguments).

For each of two sequences and 5 evenly-spaced frame pairs per sequence,
reports: mean ego forward velocity (vf), |du| / |dv| from
``estimate_ego_displacement_px``, mean per-point pixel shift
``||uv_imu - uv_uncorrected||``, mean per-point pixel shift
``||uv_event - uv_uncorrected||``, and the fraction of points whose IMU
correction exceeds 0.5 px.
"""

import os
import sys

import numpy as np

from calibration import parse_calib_cam_to_cam, parse_calib_velo_to_cam
from events import event_confidence, simulate_events
from flow import compute_rgb_flow
from imu_baseline import (
    estimate_ego_displacement_px,
    has_oxts,
    imu_deskew_projection,
    load_oxts,
)
from lidar_motion import move_lidar_points_weighted
from loader import load_image, load_lidar


DATASET_ROOT = r"C:\Users\sahaa\OneDrive\Desktop\Honors\datasets\fusion"
SEQUENCES = [
    ("fast (highway)", "2011_09_26_drive_0009_sync"),
    ("slow (urban)",   "2011_09_26_drive_0017_sync"),
]
N_PAIRS = 5
DT_DEFAULT = 0.1


def project_frame(image_path, lidar_path, tr_velo_to_cam, r_rect, p_rect):
    """Replicates validate_pipeline._project_with_full_trace, returning
    only what this audit needs: image, in-frame uv (N,2), depth (N,),
    and the LiDAR points aligned 1:1 with uv."""
    image = load_image(image_path)
    lidar_xyz = load_lidar(lidar_path)

    lidar_h = np.hstack(
        (lidar_xyz, np.ones((lidar_xyz.shape[0], 1), dtype=np.float64))
    ).T
    rectified_tf = r_rect @ tr_velo_to_cam
    proj = p_rect @ rectified_tf
    rect_cam = rectified_tf @ lidar_h
    projected = proj @ lidar_h

    valid_cam = (rect_cam[2, :] > 0.0) & np.isfinite(rect_cam).all(axis=0)
    valid_proj = (
        valid_cam
        & (projected[2, :] != 0.0)
        & np.isfinite(projected).all(axis=0)
    )

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

    uv = np.column_stack(
        (u_full[in_frame], v_full[in_frame])
    ).astype(np.float32)
    depth = rect_cam[2, in_frame].astype(np.float32)
    lidar_aligned = lidar_xyz[in_frame]
    return image, uv, depth, lidar_aligned


def list_pairs(image_dir, lidar_dir, n_pairs):
    image_files = sorted(
        f for f in os.listdir(image_dir) if f.endswith(".png")
    )
    lidar_files = sorted(
        f for f in os.listdir(lidar_dir) if f.endswith(".bin")
    )
    if not lidar_files:
        lidar_files = sorted(
            f for f in os.listdir(lidar_dir) if f.endswith(".txt")
        )
    n = min(len(image_files), len(lidar_files))
    if n < 2:
        return []
    last = n - 2
    if n_pairs >= last + 1:
        idxs = list(range(last + 1))
    else:
        idxs = sorted({int(round(x)) for x in np.linspace(0, last, n_pairs)})
    pairs = []
    for i in idxs:
        pairs.append(
            (image_files[i], lidar_files[i],
             image_files[i + 1], lidar_files[i + 1],
             i, i + 1)
        )
    return pairs


def read_dt_seconds(sequence_dir):
    """Return the median consecutive-frame dt from velodyne timestamps,
    or None if the file is unreadable. Used as a sanity check on the
    hardcoded dt=0.1s."""
    ts_path = os.path.join(
        sequence_dir, "velodyne_points", "timestamps.txt"
    )
    if not os.path.isfile(ts_path):
        return None
    try:
        from datetime import datetime
        ts = []
        with open(ts_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # KITTI timestamp format: "YYYY-MM-DD HH:MM:SS.fffffffff"
                # Trim to microseconds for fromisoformat.
                if "." in line:
                    head, frac = line.split(".", 1)
                    frac = frac[:6]
                    line = f"{head}.{frac}"
                ts.append(datetime.fromisoformat(line).timestamp())
        if len(ts) < 2:
            return None
        return float(np.median(np.diff(ts)))
    except Exception:
        return None


def audit_sequence(label, sequence_dir):
    print()
    print("=" * 104)
    print(f"[{label}] {os.path.basename(sequence_dir)}")
    print("=" * 104)

    if not os.path.isdir(sequence_dir):
        print(f"  ! sequence directory missing: {sequence_dir}")
        return
    if not has_oxts(sequence_dir):
        print(f"  ! oxts unavailable: {sequence_dir}")
        return

    tr_velo_to_cam = parse_calib_velo_to_cam(
        os.path.join(sequence_dir, "calib_velo_to_cam.txt")
    )
    r_rect, p_rect = parse_calib_cam_to_cam(
        os.path.join(sequence_dir, "calib_cam_to_cam.txt"),
        camera_id="02",
    )

    image_dir = os.path.join(sequence_dir, "image_02", "data")
    lidar_dir = os.path.join(sequence_dir, "velodyne_points", "data")
    oxts_dir = os.path.join(sequence_dir, "oxts", "data")

    pairs = list_pairs(image_dir, lidar_dir, N_PAIRS)
    if not pairs:
        print("  ! not enough frames")
        return

    dt_observed = read_dt_seconds(sequence_dir)
    if dt_observed is not None:
        print(
            f"  velodyne dt (median of consecutive frames): "
            f"{dt_observed:.4f} s   (hardcoded: {DT_DEFAULT:.4f} s)"
        )

    header = (
        f"{'idx_t':>5} {'idx_t1':>6} {'mean_vf':>8} "
        f"{'|du|_px':>8} {'|dv|_px':>8} "
        f"{'imu_shift':>10} {'eg_shift':>10} "
        f"{'frac>0.5px':>11} {'N':>7}"
    )
    print(header)
    print("-" * len(header))

    sums = {
        "vf": [], "du": [], "dv": [],
        "imu": [], "eg": [], "frac": [],
    }

    for img_t, lid_t, img_t1, _lid_t1, idx_t, idx_t1 in pairs:
        try:
            image_t, uv_t, depth_t, lidar_aligned = project_frame(
                os.path.join(image_dir, img_t),
                os.path.join(lidar_dir, lid_t),
                tr_velo_to_cam, r_rect, p_rect,
            )
            image_t1 = load_image(os.path.join(image_dir, img_t1))
            oxts_t = load_oxts(oxts_dir, idx_t)
            oxts_t1 = load_oxts(oxts_dir, idx_t1)
        except (FileNotFoundError, ValueError) as exc:
            print(f"  ! pair {idx_t}->{idx_t1} skipped: {exc}")
            continue

        if uv_t.shape[0] == 0:
            print(f"  ! pair {idx_t}->{idx_t1} skipped: no in-frame points")
            continue

        du, dv = estimate_ego_displacement_px(
            oxts_t, oxts_t1, p_rect, dt=DT_DEFAULT
        )

        uv_imu = imu_deskew_projection(
            uv=uv_t,
            depth=depth_t,
            lidar_xyz=lidar_aligned,
            oxts_t=oxts_t,
            oxts_t1=oxts_t1,
            p_rect=p_rect,
            image_shape=image_t.shape,
            dt=DT_DEFAULT,
        )

        events = simulate_events(image_t, image_t1)
        flow_raw = compute_rgb_flow(image_t, image_t1)
        flow = np.nan_to_num(
            flow_raw.astype(np.float32),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        confidence = event_confidence(events)
        uv_eg, _ = move_lidar_points_weighted(
            uv_t, depth_t, flow, confidence
        )

        diff_imu = np.linalg.norm(uv_imu - uv_t, axis=1)
        diff_eg = np.linalg.norm(uv_eg - uv_t, axis=1)
        mean_vf = 0.5 * (float(oxts_t["vf"]) + float(oxts_t1["vf"]))
        m_imu = float(np.mean(diff_imu)) if diff_imu.size else 0.0
        m_eg = float(np.mean(diff_eg)) if diff_eg.size else 0.0
        frac_imu = (
            float(np.mean(diff_imu > 0.5)) if diff_imu.size else 0.0
        )

        sums["vf"].append(mean_vf)
        sums["du"].append(abs(float(du)))
        sums["dv"].append(abs(float(dv)))
        sums["imu"].append(m_imu)
        sums["eg"].append(m_eg)
        sums["frac"].append(frac_imu)

        print(
            f"{idx_t:5d} {idx_t1:6d} {mean_vf:8.3f} "
            f"{abs(du):8.4f} {abs(dv):8.4f} "
            f"{m_imu:10.4f} {m_eg:10.4f} "
            f"{frac_imu:11.4f} {uv_t.shape[0]:7d}"
        )

    if sums["imu"]:
        print("-" * len(header))
        n = len(sums["imu"])
        print(
            f"  mean over {n} pairs: "
            f"vf={np.mean(sums['vf']):.3f}  "
            f"|du|={np.mean(sums['du']):.4f}  "
            f"|dv|={np.mean(sums['dv']):.4f}  "
            f"imu_shift={np.mean(sums['imu']):.4f}  "
            f"eg_shift={np.mean(sums['eg']):.4f}  "
            f"frac>0.5px={np.mean(sums['frac']):.4f}"
        )


def main():
    for label, name in SEQUENCES:
        audit_sequence(label, os.path.join(DATASET_ROOT, name))


if __name__ == "__main__":
    sys.exit(main())
