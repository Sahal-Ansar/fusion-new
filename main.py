import argparse
import os

import cv2
import numpy as np

from calibration import parse_calib_cam_to_cam, parse_calib_velo_to_cam
from events import events_to_image, simulate_events, event_confidence
from flow import compute_rgb_flow, smooth_flow
from lidar_motion import move_lidar_points_weighted
from loader import list_frame_files, load_image, load_lidar
from projection import overlay_points, project_lidar_to_image


DEFAULT_DATASET_PATH = (
    r"C:\Users\sahaa\OneDrive\Desktop\Honors\datasets\fusion\2011_09_26_drive_0009_extract"
)

DISPLAY_WINDOW_NAME = "Original Projection (t)"
EVENT_WINDOW_NAME = "Event Image"
MOTION_WINDOW_NAME = "Corrected Projection (t+1)"
COMPARE_WINDOW_NAME = "Corrected Projection (t)"


# ---------------- FRAME PROCESSING ---------------- #

def process_frame(
    image_path,
    lidar_path,
    tr_velo_to_cam,
    r_rect,
    p_rect,
    max_points,
    debug_mode=0,
):
    image = load_image(image_path)
    lidar_xyz = load_lidar(lidar_path)

    uv, depth, stats = project_lidar_to_image(
        lidar_xyz,
        tr_velo_to_cam,
        r_rect,
        p_rect,
        image.shape,
    )

    if debug_mode == 1:
        mask = depth < 30.0
        uv = uv[mask]
        depth = depth[mask]

    elif debug_mode == 2:
        mask = depth >= 30.0
        uv = uv[mask]
        depth = depth[mask]

    elif debug_mode == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        dist_transform = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)

        uv_int = np.clip(
            np.rint(uv).astype(np.int32),
            [0, 0],
            [image.shape[1] - 1, image.shape[0] - 1],
        )

        mask = dist_transform[uv_int[:, 1], uv_int[:, 0]] <= 2.0
        uv = uv[mask]
        depth = depth[mask]

    out = overlay_points(image, uv, depth, max_points=max_points, radius=1)

    return out, image, uv, depth, lidar_xyz.shape[0], uv.shape[0], stats


def process_motion_frame(
    image_path_t,
    image_path_t1,
    lidar_path,
    tr_velo_to_cam,
    r_rect,
    p_rect,
    max_points,
    prev_flow=None,
    debug_mode=0,
    event_threshold=0.2,
):
    # --- ORIGINAL FRAME ---
    original_vis, image_t, uv, depth, n_lidar_points, n_valid_points, stats = process_frame(
        image_path_t,
        lidar_path,
        tr_velo_to_cam,
        r_rect,
        p_rect,
        max_points=max_points,
        debug_mode=debug_mode,
    )

    # --- NEXT FRAME ---
    image_t1 = load_image(image_path_t1)

    # --- EVENTS ---
    events = simulate_events(image_t, image_t1, threshold=event_threshold)
    event_vis = events_to_image(events)

    # --- RGB FLOW ---
    flow_raw = compute_rgb_flow(image_t, image_t1)
    flow = smooth_flow(flow_raw, prev_flow, alpha=0.7)

    # --- EVENT CONFIDENCE ---
    conf = event_confidence(events)

    # --- WEIGHTED LIDAR MOTION ---
    uv_new, depth_new = move_lidar_points_weighted(
        uv, depth, flow, conf
    )

    # --- VISUALIZATION ---
    moved_vis_t1 = overlay_points(image_t1, uv_new, depth_new, max_points=max_points, radius=1)
    moved_vis_t = overlay_points(image_t, uv_new, depth_new, max_points=max_points, radius=1)

    # --- DEBUG ---
    active_mask = events != 0
    flow_mag = np.linalg.norm(flow, axis=2)

    debug = {
        "num_events": int(np.count_nonzero(active_mask)),
        "active_percent": float(active_mask.mean() * 100.0),
        "mean_flow_magnitude": float(flow_mag.mean()),
        "valid_lidar_after_motion": int(uv_new.shape[0]),
        "n_lidar_points": int(n_lidar_points),
        "n_valid_points": int(n_valid_points),
        "stats": stats,
    }

    return {
        "original_vis": original_vis,
        "event_vis": event_vis,
        "moved_vis_t1": moved_vis_t1,
        "moved_vis_t": moved_vis_t,
        "flow": flow,
        "debug": debug,
    }


# ---------------- MAIN PIPELINE ---------------- #

def run_pipeline(args):
    dataset_path = args.dataset_path
    image_dir = os.path.join(dataset_path, "image_02", "data")
    lidar_dir = os.path.join(dataset_path, "velodyne_points", "data")

    tr_velo_to_cam = parse_calib_velo_to_cam(
        os.path.join(dataset_path, "calib_velo_to_cam.txt")
    )

    r_rect, p_rect = parse_calib_cam_to_cam(
        os.path.join(dataset_path, "calib_cam_to_cam.txt"),
        camera_id="02",
    )

    image_files = list_frame_files(image_dir, ".png")
    lidar_files = list_frame_files(lidar_dir, ".bin")

    pairs = list(zip(image_files, lidar_files))

    idx = 0
    prev_flow = None

    cv2.namedWindow(DISPLAY_WINDOW_NAME)
    cv2.namedWindow(EVENT_WINDOW_NAME)
    cv2.namedWindow(MOTION_WINDOW_NAME)
    cv2.namedWindow(COMPARE_WINDOW_NAME)

    while 0 <= idx < len(pairs) - 1:
        image_name, lidar_name = pairs[idx]
        next_image_name, _ = pairs[idx + 1]

        image_path = os.path.join(image_dir, image_name)
        next_image_path = os.path.join(image_dir, next_image_name)
        lidar_path = os.path.join(lidar_dir, lidar_name)

        result = process_motion_frame(
            image_path,
            next_image_path,
            lidar_path,
            tr_velo_to_cam,
            r_rect,
            p_rect,
            max_points=args.max_draw_points,
            prev_flow=prev_flow,
            debug_mode=args.debug_mode,
            event_threshold=args.event_threshold,
        )
        prev_flow = result["flow"]

        debug = result["debug"]

        print(
            f"Frame {image_name} | Events: {debug['num_events']} | "
            f"Active: {debug['active_percent']:.2f}% | "
            f"Flow: {debug['mean_flow_magnitude']:.3f} | "
            f"Moved points: {debug['valid_lidar_after_motion']}"
        )

        cv2.imshow(DISPLAY_WINDOW_NAME, result["original_vis"])
        cv2.imshow(EVENT_WINDOW_NAME, result["event_vis"])
        cv2.imshow(MOTION_WINDOW_NAME, result["moved_vis_t1"])
        cv2.imshow(COMPARE_WINDOW_NAME, result["moved_vis_t"])

        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("n"):
            idx += 1
        elif key == ord("p"):
            idx = max(0, idx - 1)

    cv2.destroyAllWindows()


# ---------------- ARGUMENTS ---------------- #

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--max-draw-points", type=int, default=12000)
    parser.add_argument("--event-threshold", type=float, default=0.2)

    parser.add_argument(
        "--debug-mode",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
    )

    args = parser.parse_args()

    run_pipeline(args)


if __name__ == "__main__":
    main()
