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
    events = cv2.GaussianBlur(events.astype(np.float32), (5, 5), 0)
    return events


def events_to_image(events):
    """Render event polarities as a color image."""
    event_map = np.nan_to_num(events, nan=0.0, posinf=0.0, neginf=0.0)
    vis = np.zeros((event_map.shape[0], event_map.shape[1], 3), dtype=np.uint8)
    vis[event_map > 0] = (255, 255, 255)
    vis[event_map < 0] = (0, 0, 255)
    return vis


def event_confidence(events):
    """Convert event map to a sparse binary confidence mask."""
    conf = (np.abs(events) > 0.1).astype(np.float32)
    print("Event density:", np.mean(conf))
    return conf
