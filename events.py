# Fixed: 2026-04-24 — see pipeline_fixes.md for reasoning
import cv2
import numpy as np


def simulate_events(img1, img2, threshold=0.3):
    """
    Simulate event camera output from two consecutive RGB frames.

    Uses log-intensity difference (mimicking DVS hardware response).
    Events fire where brightness change exceeds threshold.
    Morphological dilation slightly expands event regions for
    confidence masking without causing density explosion.

    threshold=0.3 corresponds to ~30% relative brightness change,
    which filters camera noise while capturing real motion edges.
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)

    gray1 = np.maximum(gray1, 1.0)
    gray2 = np.maximum(gray2, 1.0)

    diff = np.log(gray2) - np.log(gray1)
    diff = np.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)

    events = np.zeros_like(diff, dtype=np.float32)
    events[diff > threshold] = 1.0
    events[diff < -threshold] = -1.0

    # Morphological dilation: expand each event pixel to 8-neighbors.
    # This is physically motivated: real DVS pixels have slight spatial
    # coupling and our simulated events are pixel-exact, so mild dilation
    # compensates for sub-pixel misalignment in LiDAR projection.
    kernel = np.ones((3, 3), np.uint8)
    on_events = (events == 1.0).astype(np.uint8)
    off_events = (events == -1.0).astype(np.uint8)
    on_dilated = cv2.dilate(on_events, kernel, iterations=1)
    off_dilated = cv2.dilate(off_events, kernel, iterations=1)

    events_out = np.zeros_like(events, dtype=np.float32)
    events_out[on_dilated > 0] = 1.0
    events_out[off_dilated > 0] = -1.0
    # Where both on and off overlap after dilation, set to zero (ambiguous)
    events_out[(on_dilated > 0) & (off_dilated > 0)] = 0.0

    return events_out


def events_to_image(events):
    """Render event polarities as a color image."""
    event_map = np.nan_to_num(events, nan=0.0, posinf=0.0, neginf=0.0)
    vis = np.zeros((event_map.shape[0], event_map.shape[1], 3), dtype=np.uint8)
    vis[event_map > 0] = (255, 255, 255)
    vis[event_map < 0] = (0, 0, 255)
    return vis


def event_confidence(events):
    """
    Convert event map to a sparse binary confidence mask.

    Threshold at 0.5 since events are binary {-1, 0, +1} after simulate_events.
    Only pixels directly at event locations (and their dilated neighbors) pass.
    Target density: 5% to 20% of pixels.
    """
    conf = (np.abs(events) > 0.5).astype(np.float32)
    density = float(np.mean(conf))
    print(f"Event density: {density:.4f}  (target: 0.05 to 0.20)")
    return conf
