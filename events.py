import cv2
import numpy as np


def simulate_events(img1, img2, threshold=0.2):
    """Simulate a simple event map from two RGB/BGR frames."""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)

    gray1 = np.maximum(gray1, 1.0)
    gray2 = np.maximum(gray2, 1.0)

    diff = np.log(gray2) - np.log(gray1)
    diff = np.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)

    events = np.zeros_like(diff, dtype=np.float32)
    events[diff > threshold] = 1.0
    events[diff < -threshold] = -1.0
    return events


def events_to_image(events):
    """Render event polarities as a color image."""
    event_map = np.nan_to_num(events, nan=0.0, posinf=0.0, neginf=0.0)
    vis = np.zeros((event_map.shape[0], event_map.shape[1], 3), dtype=np.uint8)
    vis[event_map > 0] = (255, 255, 255)
    vis[event_map < 0] = (0, 0, 255)
    return vis

def event_confidence(events):
    """Convert event map to confidence mask (0–1)."""
    conf = np.abs(events)
    if conf.max() > 0:
        conf = conf / conf.max()
    return conf.astype(np.float32)