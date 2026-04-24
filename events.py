# Fixed: 2026-04-24 — see pipeline_fixes.md for reasoning
import cv2
import numpy as np


def simulate_events(img1, img2, threshold=0.3):
    """
    Simulate event camera output from two consecutive RGB frames.

    Combines temporal contrast (log-intensity difference) with spatial
    contrast filtering (Canny edges) to approximate real DVS behavior.
    Real event cameras fire only where moving spatial edges cross pixels,
    not across entire moving uniform surfaces.

    Returns binary event map with values in {-1.0, 0.0, 1.0}.
    Target density: 5-20% of pixels active.
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)

    gray1 = np.maximum(gray1, 1.0)
    gray2 = np.maximum(gray2, 1.0)

    # Temporal contrast: log-intensity difference (DVS hardware response)
    diff = np.log(gray2) - np.log(gray1)
    diff = np.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)

    # Threshold temporal contrast to get candidate events
    candidates = np.zeros_like(diff, dtype=np.float32)
    candidates[diff > threshold] = 1.0
    candidates[diff < -threshold] = -1.0

    # Spatial contrast filter: detect edges in BOTH frames
    # Real DVS pixels fire only when a spatial edge moves across them.
    # We approximate this by keeping only candidate events that are
    # spatially near edges in either the source or destination frame.
    edges1 = cv2.Canny(gray1.astype(np.uint8), 30, 90)
    edges2 = cv2.Canny(gray2.astype(np.uint8), 30, 90)

    # Dilate edges to give a small neighborhood (3px radius = 7x7 kernel)
    edge_kernel = np.ones((7, 7), np.uint8)
    edges1_dilated = cv2.dilate(edges1, edge_kernel, iterations=1)
    edges2_dilated = cv2.dilate(edges2, edge_kernel, iterations=1)

    # Combined edge mask: near an edge in either frame
    edge_mask = ((edges1_dilated > 0) | (edges2_dilated > 0)).astype(np.float32)

    # Keep only events that are spatially near edges
    events = candidates * edge_mask

    # Small morphological dilation to slightly expand surviving events
    # (compensates for sub-pixel LiDAR projection misalignment)
    kernel = np.ones((3, 3), np.uint8)
    on_events = (events == 1.0).astype(np.uint8)
    off_events = (events == -1.0).astype(np.uint8)
    on_dilated = cv2.dilate(on_events, kernel, iterations=1)
    off_dilated = cv2.dilate(off_events, kernel, iterations=1)

    events_out = np.zeros_like(events, dtype=np.float32)
    events_out[on_dilated > 0] = 1.0
    events_out[off_dilated > 0] = -1.0
    # Resolve overlap: ambiguous pixels set to zero
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
