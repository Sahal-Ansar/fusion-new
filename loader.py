import os
from datetime import datetime

import cv2
import numpy as np


def list_frame_files(directory, suffix):
    """Return a sorted list of files with the given suffix in a directory."""
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    return sorted([f for f in os.listdir(directory) if f.endswith(suffix)])


def load_image(image_path):
    """Load image with validation and clear errors."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to decode image: {image_path}")
    return image


def load_lidar(lidar_path):
    """
    Load LiDAR and return XYZ in float64.

    Primary format is KITTI .bin (N,4 float32: x,y,z,reflectance).
    For robustness, .txt is also supported when present in derived datasets.
    """
    if not os.path.isfile(lidar_path):
        raise FileNotFoundError(f"LiDAR file not found: {lidar_path}")

    ext = os.path.splitext(lidar_path)[1].lower()

    if ext == ".bin":
        raw = np.fromfile(lidar_path, dtype=np.float32)
        if raw.size == 0:
            return np.empty((0, 3), dtype=np.float64)
        if raw.size % 4 != 0:
            raise ValueError(
                f"Corrupted .bin file (float count not divisible by 4): {lidar_path}"
            )
        points = raw.reshape(-1, 4)
        return points[:, :3].astype(np.float64, copy=False)

    if ext == ".txt":
        points = np.loadtxt(lidar_path, dtype=np.float64)
        if points.size == 0:
            return np.empty((0, 3), dtype=np.float64)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if points.shape[1] < 3:
            raise ValueError(f"LiDAR .txt must have at least 3 columns: {lidar_path}")
        return points[:, :3]

    raise ValueError(f"Unsupported LiDAR extension '{ext}' for file: {lidar_path}")


def parse_timestamps(timestamp_path):
    """Parse KITTI timestamp file into seconds since epoch (float64)."""
    if not os.path.isfile(timestamp_path):
        raise FileNotFoundError(f"Timestamp file not found: {timestamp_path}")

    values = []
    with open(timestamp_path, "r", encoding="utf-8") as f:
        for line in f:
            token = line.strip()
            if not token:
                continue
            if "." in token:
                base, frac = token.split(".", 1)
                frac = (frac + "000000")[:6]
                token = f"{base}.{frac}"
            dt = datetime.strptime(token, "%Y-%m-%d %H:%M:%S.%f")
            values.append(dt.timestamp())
    return np.array(values, dtype=np.float64)