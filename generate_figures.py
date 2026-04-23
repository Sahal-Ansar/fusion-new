import os

import cv2
import numpy as np

from loader import list_frame_files, load_image, load_lidar
from calibration import parse_calib_velo_to_cam, parse_calib_cam_to_cam
from projection import project_lidar_to_image, overlay_points
from events import events_to_image, simulate_events
from flow import compute_rgb_flow
from lidar_motion import move_lidar_points_weighted


# ==========================
# CONFIG
# ==========================
DATASET_PATH = r"C:\Users\sahaa\OneDrive\Desktop\Honors\datasets\fusion\2011_09_26_drive_0009_sync"
FRAME_INDEX = 60
OUTPUT_DIR = "paper_figures_2"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================
# LOAD FILE LISTS
# ==========================
image_dir = os.path.join(DATASET_PATH, "image_02", "data")
lidar_dir = os.path.join(DATASET_PATH, "velodyne_points", "data")

image_files = sorted(list_frame_files(image_dir, ".png"))
lidar_files = sorted(list_frame_files(lidar_dir, ".bin"))

assert len(image_files) > FRAME_INDEX + 1, "Not enough frames"
assert len(lidar_files) > FRAME_INDEX + 1, "Not enough lidar frames"


# ==========================
# LOAD CALIBRATION
# ==========================
calib_velo = os.path.join(DATASET_PATH, "calib_velo_to_cam.txt")
calib_cam = os.path.join(DATASET_PATH, "calib_cam_to_cam.txt")

Tr = parse_calib_velo_to_cam(calib_velo)
R_rect, P_rect = parse_calib_cam_to_cam(calib_cam, camera_id="02")


# ==========================
# LOAD FRAMES
# ==========================
img_t = load_image(os.path.join(image_dir, image_files[FRAME_INDEX]))
img_t1 = load_image(os.path.join(image_dir, image_files[FRAME_INDEX + 1]))
lidar_t = load_lidar(os.path.join(lidar_dir, lidar_files[FRAME_INDEX]))


# ==========================
# 1. ORIGINAL PROJECTION (BEFORE)
# ==========================
uv, depth, _ = project_lidar_to_image(
    lidar_t,
    Tr,
    R_rect,
    P_rect,
    img_t.shape,
)


# ==========================
# 2. EVENT + FLOW
# ==========================
events = simulate_events(img_t, img_t1, threshold=0.3)
events = cv2.GaussianBlur(events.astype(np.float32), (5, 5), 0)
conf = (np.abs(events) > 0.15).astype(np.float32)
print("Event density:", np.mean(conf))

flow = compute_rgb_flow(img_t, img_t1)


# ==========================
# 3. MOTION CORRECTION
# ==========================
uv_moved, depth_moved = move_lidar_points_weighted(
    uv,
    depth,
    flow,
    conf,
    conf_thresh=0.5,
)


# ==========================
# 4. FULL-FRAME FIGURES
# ==========================
before_base = img_t.copy()
after_base = img_t.copy()

before = overlay_points(before_base, uv, depth)
after = overlay_points(after_base, uv_moved, depth_moved)

cv2.putText(before, "Before", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.putText(after, "After", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

comparison = np.hstack((before, after))
blend = cv2.addWeighted(before, 0.5, after, 0.5, 0)


# ==========================
# 5. ZOOMED FIGURES
# ==========================
x1, y1 = 400, 150
x2, y2 = 900, 450

zoom_before = before[y1:y2, x1:x2]
zoom_after = after[y1:y2, x1:x2]
zoom_comparison = np.hstack((zoom_before, zoom_after))


# ==========================
# 6. MOVEMENT HEATMAP
# ==========================
heatmap = np.zeros_like(img_t)
common_count = min(len(uv), len(uv_moved))

if common_count > 0:
    movement = np.linalg.norm(uv_moved[:common_count] - uv[:common_count], axis=1)
    movement_norm = (movement / (movement.max() + 1e-6) * 255).astype(np.uint8)

    for i in range(common_count):
        if not np.isfinite(uv[i, 0]) or not np.isfinite(uv[i, 1]):
            continue
        px = int(uv[i, 0])
        py = int(uv[i, 1])
        if px < 0 or px >= img_t.shape[1] or py < 0 or py >= img_t.shape[0]:
            continue
        color = cv2.applyColorMap(
            np.array([[movement_norm[i]]], dtype=np.uint8),
            cv2.COLORMAP_JET,
        )[0][0]
        cv2.circle(
            heatmap,
            (px, py),
            1,
            (int(color[0]), int(color[1]), int(color[2])),
            -1,
        )


# ==========================
# 7. EVENT VISUAL
# ==========================
event_img = events_to_image(events)


# ==========================
# 8. SAVE OUTPUTS
# ==========================
cv2.imwrite(os.path.join(OUTPUT_DIR, "before.png"), before)
cv2.imwrite(os.path.join(OUTPUT_DIR, "after.png"), after)
cv2.imwrite(os.path.join(OUTPUT_DIR, "comparison.png"), comparison)
cv2.imwrite(os.path.join(OUTPUT_DIR, "zoom_before.png"), zoom_before)
cv2.imwrite(os.path.join(OUTPUT_DIR, "zoom_after.png"), zoom_after)
cv2.imwrite(os.path.join(OUTPUT_DIR, "zoom_comparison.png"), zoom_comparison)
cv2.imwrite(os.path.join(OUTPUT_DIR, "movement_heatmap.png"), heatmap)
cv2.imwrite(os.path.join(OUTPUT_DIR, "blend.png"), blend)
cv2.imwrite(os.path.join(OUTPUT_DIR, "events.png"), event_img)

print("Saved figures to:", OUTPUT_DIR)
