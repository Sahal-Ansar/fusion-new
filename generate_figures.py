import os
import cv2
import numpy as np

from loader import list_frame_files, load_image, load_lidar
from calibration import parse_calib_velo_to_cam, parse_calib_cam_to_cam
from projection import project_lidar_to_image, overlay_points
from events import simulate_events, events_to_image, event_confidence
from flow import compute_rgb_flow, smooth_flow
from lidar_motion import move_lidar_points_weighted


# ==========================
# CONFIG
# ==========================
DATASET_PATH = r"C:\Users\sahaa\OneDrive\Desktop\Honors\datasets\fusion\2011_09_26_drive_0009_sync"
FRAME_INDEX = 30   # change this to pick a good frame
OUTPUT_DIR = "paper_figures"

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

h, w = img_t.shape[:2]


# ==========================
# 1. ORIGINAL PROJECTION (BEFORE)
# ==========================
uv, depth, _ = project_lidar_to_image(
    lidar_t, Tr, R_rect, P_rect, img_t.shape
)

before = overlay_points(img_t, uv, depth)


# ==========================
# 2. EVENT + FLOW
# ==========================
events = simulate_events(img_t, img_t1)
conf = event_confidence(events)

flow = compute_rgb_flow(img_t, img_t1)


# ==========================
# 3. MOTION CORRECTION
# ==========================
uv_moved, depth_moved = move_lidar_points_weighted(
    uv, depth, flow, conf, conf_thresh=0.2
)


# ==========================
# 4. AFTER PROJECTION (on t+1)
# ==========================
after = overlay_points(img_t, uv_moved, depth_moved)

cv2.putText(before, "Before", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.putText(after, "After", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


# ==========================
# 5. EVENT VISUAL (optional)
# ==========================
event_img = events_to_image(events)


# ==========================
# 6. COMBINE (SIDE BY SIDE)
# ==========================
def resize_to_same(a, b):
    h = min(a.shape[0], b.shape[0])
    a = cv2.resize(a, (int(a.shape[1] * h / a.shape[0]), h))
    b = cv2.resize(b, (int(b.shape[1] * h / b.shape[0]), h))
    return a, b

before_r, after_r = resize_to_same(before, after)

comparison = np.hstack((before_r, after_r))
blend = cv2.addWeighted(before, 0.5, after, 0.5, 0)


# ==========================
# 7. SAVE OUTPUTS
# ==========================
cv2.imwrite(os.path.join(OUTPUT_DIR, "before.png"), before)
cv2.imwrite(os.path.join(OUTPUT_DIR, "after.png"), after)
cv2.imwrite(os.path.join(OUTPUT_DIR, "comparison.png"), comparison)
cv2.imwrite(os.path.join(OUTPUT_DIR, "blend.png"), blend)
cv2.imwrite(os.path.join(OUTPUT_DIR, "events.png"), event_img)

print("Saved figures to:", OUTPUT_DIR)
