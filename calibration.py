import os

import numpy as np


def _parse_calibration_file(calib_path):
    """Parse KITTI calibration text file into dict[str, np.ndarray(float64)]."""
    if not os.path.isfile(calib_path):
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")

    parsed = {}
    with open(calib_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, values = line.split(":", 1)
            values = values.strip()
            if not values:
                parsed[key] = np.array([], dtype=np.float64)
                continue
            tokens = values.split()
            float_values = []
            all_float = True
            for token in tokens:
                try:
                    float_values.append(float(token))
                except ValueError:
                    all_float = False
                    break
            if not all_float:
                continue
            parsed[key] = np.array(float_values, dtype=np.float64)
    return parsed


def parse_calib_velo_to_cam(calib_path):
    """Build Tr_velo_to_cam (4x4) from calib_velo_to_cam.txt (R and T)."""
    data = _parse_calibration_file(calib_path)

    if "R" not in data or "T" not in data:
        raise KeyError(f"Expected keys 'R' and 'T' in {calib_path}")

    if data["R"].size != 9:
        raise ValueError(f"Key 'R' must have 9 values in {calib_path}")
    if data["T"].size != 3:
        raise ValueError(f"Key 'T' must have 3 values in {calib_path}")

    R = data["R"].reshape((3, 3), order="C")
    T = data["T"].reshape((3, 1), order="C")

    tr_velo_to_cam = np.eye(4, dtype=np.float64)
    tr_velo_to_cam[:3, :3] = R
    tr_velo_to_cam[:3, 3] = T[:, 0]

    t = tr_velo_to_cam[:3, 3]
    print("Tr_velo_to_cam (4x4):")
    print(tr_velo_to_cam)
    print(f"Tr translation (m): tx={t[0]:.6f}, ty={t[1]:.6f}, tz={t[2]:.6f}")

    return tr_velo_to_cam


def parse_calib_cam_to_cam(calib_path, camera_id="02"):
    """
    Parse cam calibration and return:
    - R_rect (4x4) from R_rect_00
    - P_rect (3x4) from P_rect_{camera_id}
    """
    data = _parse_calibration_file(calib_path)

    camera_id = str(camera_id).zfill(2)
    rect_key = "R_rect_00"
    proj_key = f"P_rect_{camera_id}"

    if rect_key not in data:
        raise KeyError(f"Missing key '{rect_key}' in {calib_path}")
    if proj_key not in data:
        raise KeyError(f"Missing key '{proj_key}' in {calib_path}")

    if data[rect_key].size != 9:
        raise ValueError(f"Key '{rect_key}' must have 9 values in {calib_path}")
    if data[proj_key].size != 12:
        raise ValueError(f"Key '{proj_key}' must have 12 values in {calib_path}")

    R_rect_3x3 = data[rect_key].reshape((3, 3), order="C")
    P_rect = data[proj_key].reshape((3, 4), order="C")

    R_rect = np.eye(4, dtype=np.float64)
    R_rect[:3, :3] = R_rect_3x3

    print(f"{proj_key} (3x4):")
    print(P_rect)
    print("R_rect (4x4):")
    print(R_rect)

    fx = P_rect[0, 0]
    fy = P_rect[1, 1]
    if not (700.0 <= fx <= 1200.0 and 700.0 <= fy <= 1200.0):
        print(f"Warning: unexpected focal values in {proj_key}: fx={fx:.6f}, fy={fy:.6f}")

    rect_delta = np.max(np.abs(R_rect[:3, :3] - np.eye(3, dtype=np.float64)))
    print(f"R_rect max deviation from identity: {rect_delta:.6e}")

    return R_rect, P_rect
