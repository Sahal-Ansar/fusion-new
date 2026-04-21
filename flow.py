import cv2
import numpy as np


def _zero_flow(shape):
    h, w = shape[:2]
    return np.zeros((h, w, 2), dtype=np.float32)


def _to_gray(image):
    image = np.asarray(image)
    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Expected grayscale or BGR image, got shape {image.shape}")


def compute_flow(events1, events2):
    """Compute dense optical flow between two event maps."""
    prev = np.where(np.nan_to_num(events1) != 0, 255, 0).astype(np.uint8)
    next_ = np.where(np.nan_to_num(events2) != 0, 255, 0).astype(np.uint8)

    if prev.ndim != 2 or next_.ndim != 2 or prev.shape != next_.shape:
        raise ValueError(
            f"Event maps must be 2D and shape-matched, got {prev.shape} and {next_.shape}"
        )

    if prev.size == 0 or (not np.any(prev) and not np.any(next_)):
        return _zero_flow(prev.shape)

    flow = cv2.calcOpticalFlowFarneback(
        prev,
        next_,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    return np.nan_to_num(flow.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def compute_rgb_flow(img1, img2):
    """Compute dense optical flow from two grayscale/BGR frames."""
    gray1 = _to_gray(img1)
    gray2 = _to_gray(img2)

    if gray1.shape != gray2.shape:
        raise ValueError(f"Input images must have the same shape, got {gray1.shape} and {gray2.shape}")

    if gray1.size == 0:
        return _zero_flow(gray1.shape)

    flow = cv2.calcOpticalFlowFarneback(
        gray1,
        gray2,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=21,
        iterations=5,
        poly_n=7,
        poly_sigma=1.5,
        flags=0,
    )
    return np.nan_to_num(flow.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
