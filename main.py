import argparse
import os

import cv2
import numpy as np

from calibration import parse_calib_cam_to_cam, parse_calib_velo_to_cam
from loader import list_frame_files, load_image, load_lidar, parse_timestamps
from projection import overlay_points, project_lidar_to_image


DEFAULT_DATASET_PATH = (
    r"C:\Users\sahaa\OneDrive\Desktop\Honors\datasets\fusion\2011_09_26_drive_0009_extract"
)

DISPLAY_WINDOW_NAME = "KITTI LiDAR Projection"
CONTROL_WINDOW_NAME = "Playback Controls"
PAUSE_BUTTON = (10, 10, 120, 60)
PLAY_BUTTON = (140, 10, 250, 60)
ZOOM_TRACKBAR_NAME = "Zoom x100"
ZOOM_MIN = 0.5
ZOOM_MAX = 3.0
ZOOM_DEFAULT = 1.0
ZOOM_STEP = 0.1
ZOOM_TRACKBAR_MIN = int(ZOOM_MIN * 100)
ZOOM_TRACKBAR_MAX = int(ZOOM_MAX * 100)


def _draw_controls(paused):
    panel = np.full((70, 260, 3), 35, dtype=np.uint8)

    pause_color = (40, 40, 180) if paused else (90, 90, 90)
    play_color = (40, 160, 40) if not paused else (90, 90, 90)

    cv2.rectangle(panel, (PAUSE_BUTTON[0], PAUSE_BUTTON[1]), (PAUSE_BUTTON[2], PAUSE_BUTTON[3]), pause_color, -1)
    cv2.rectangle(panel, (PLAY_BUTTON[0], PLAY_BUTTON[1]), (PLAY_BUTTON[2], PLAY_BUTTON[3]), play_color, -1)

    cv2.putText(panel, "Pause", (32, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, "Play", (176, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def _control_click_handler(event, x, y, flags, state):
    del flags
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if PAUSE_BUTTON[0] <= x <= PAUSE_BUTTON[2] and PAUSE_BUTTON[1] <= y <= PAUSE_BUTTON[3]:
        state["paused"] = True
    elif PLAY_BUTTON[0] <= x <= PLAY_BUTTON[2] and PLAY_BUTTON[1] <= y <= PLAY_BUTTON[3]:
        state["paused"] = False


def _slider_to_zoom(value):
    value = int(np.clip(value, ZOOM_TRACKBAR_MIN, ZOOM_TRACKBAR_MAX))
    return value / 100.0


def _zoom_to_slider(zoom):
    return int(np.clip(round(zoom * 100.0), ZOOM_TRACKBAR_MIN, ZOOM_TRACKBAR_MAX))


def _set_zoom(view_state, zoom):
    view_state["zoom"] = float(np.clip(zoom, ZOOM_MIN, ZOOM_MAX))


def _reset_view(view_state):
    _set_zoom(view_state, ZOOM_DEFAULT)


def _zoom_trackbar_handler(value, view_state):
    _set_zoom(view_state, _slider_to_zoom(value))


def _render_image_only_zoom(image, uv, depth, zoom, max_points):
    scaled_w = max(1, int(round(image.shape[1] * zoom)))
    scaled_h = max(1, int(round(image.shape[0] * zoom)))
    scaled_image = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
    display = overlay_points(scaled_image, uv, depth, max_points=max_points, radius=1)
    cv2.putText(
        display,
        f"Image zoom: {zoom:.2f}x",
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        display,
        "Image-only zoom diagnostic",
        (20, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return display


def _build_pairs_by_filename(image_dir, lidar_dir):
    image_files = list_frame_files(image_dir, ".png")
    lidar_bin = list_frame_files(lidar_dir, ".bin")
    lidar_txt = list_frame_files(lidar_dir, ".txt")

    lidar_all = lidar_bin + lidar_txt
    lidar_map = {os.path.splitext(name)[0]: name for name in lidar_all}

    pairs = []
    skipped = 0
    for image_name in image_files:
        frame_id = os.path.splitext(image_name)[0]
        if frame_id not in lidar_map:
            skipped += 1
            continue
        pairs.append((frame_id, image_name, lidar_map[frame_id]))
    return pairs, len(image_files), len(lidar_all), skipped


def _build_pairs_by_timestamps(dataset_path, image_dir, lidar_dir):
    image_ts_path = os.path.join(dataset_path, "image_02", "timestamps.txt")
    lidar_ts_path = os.path.join(dataset_path, "velodyne_points", "timestamps.txt")

    if not (os.path.isfile(image_ts_path) and os.path.isfile(lidar_ts_path)):
        return None

    image_files = list_frame_files(image_dir, ".png")
    lidar_bin = list_frame_files(lidar_dir, ".bin")
    lidar_txt = list_frame_files(lidar_dir, ".txt")
    lidar_files = lidar_bin if lidar_bin else lidar_txt

    if not image_files or not lidar_files:
        return None

    image_ts = parse_timestamps(image_ts_path)
    lidar_ts = parse_timestamps(lidar_ts_path)

    n_img = min(len(image_files), len(image_ts))
    n_lidar = min(len(lidar_files), len(lidar_ts))
    if n_img == 0 or n_lidar == 0:
        return None

    image_files = image_files[:n_img]
    image_ts = image_ts[:n_img]
    lidar_files = lidar_files[:n_lidar]
    lidar_ts = lidar_ts[:n_lidar]

    idx = np.searchsorted(lidar_ts, image_ts)
    idx = np.clip(idx, 0, len(lidar_ts) - 1)

    left_idx = np.maximum(idx - 1, 0)
    right_idx = idx
    left_delta = np.abs(image_ts - lidar_ts[left_idx])
    right_delta = np.abs(image_ts - lidar_ts[right_idx])
    best_idx = np.where(left_delta <= right_delta, left_idx, right_idx)

    pairs = []
    for i in range(n_img):
        frame_id = os.path.splitext(image_files[i])[0]
        pairs.append((frame_id, image_files[i], lidar_files[best_idx[i]]))

    return pairs, len(image_files), len(lidar_files), 0


def process_frame(image_path, lidar_path, tr_velo_to_cam, r_rect, p_rect, max_points, debug_mode=0):
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
        uv_int = np.clip(np.rint(uv).astype(np.int32), [0, 0], [image.shape[1]-1, image.shape[0]-1])
        mask = dist_transform[uv_int[:, 1], uv_int[:, 0]] <= 2.0
        uv = uv[mask]
        depth = depth[mask]

    out = overlay_points(image, uv, depth, max_points=max_points, radius=1)
    return out, image, uv, depth, lidar_xyz.shape[0], uv.shape[0], stats


def run_pipeline(args):
    dataset_path = args.dataset_path
    image_dir = os.path.join(dataset_path, "image_02", "data")
    lidar_dir = os.path.join(dataset_path, "velodyne_points", "data")

    if not os.path.isdir(image_dir) or not os.path.isdir(lidar_dir):
        raise FileNotFoundError("Missing image_02/data or velodyne_points/data directory")

    tr_velo_to_cam = parse_calib_velo_to_cam(os.path.join(dataset_path, "calib_velo_to_cam.txt"))
    r_rect, p_rect = parse_calib_cam_to_cam(
        os.path.join(dataset_path, "calib_cam_to_cam.txt"), camera_id="02"
    )

    print("Calibration loaded")
    print(f"Tr_velo_to_cam shape: {tr_velo_to_cam.shape}")
    print(f"R_rect shape: {r_rect.shape}")
    print(f"P_rect_02 shape: {p_rect.shape}")

    pairs_info = None
    if args.sync_mode == "timestamp":
        pairs_info = _build_pairs_by_timestamps(dataset_path, image_dir, lidar_dir)
        if pairs_info is None:
            print("Timestamp sync unavailable. Falling back to filename sync.")

    if pairs_info is None:
        pairs_info = _build_pairs_by_filename(image_dir, lidar_dir)

    pairs, n_images, n_lidar, n_skipped_sync = pairs_info
    print(f"Images found: {n_images}")
    print(f"LiDAR frames found: {n_lidar}")
    print(f"Pairs synchronized: {len(pairs)}")
    print(f"Frames skipped during sync: {n_skipped_sync}")

    if len(pairs) == 0:
        raise RuntimeError("No synchronized frame pairs available to process")

    start = max(args.start, 0)
    end = min(args.end, len(pairs) - 1) if args.end >= 0 else len(pairs) - 1
    if start > end:
        raise ValueError(f"Invalid range: start={start}, end={end}")

    selected = pairs[start : end + 1 : args.step]
    print(f"Processing {len(selected)} frame(s) from index {start} to {end} step {args.step}")

    if args.save_output and not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    processed = 0
    skipped = 0

    control_state = {"paused": False}
    view_state = {
        "zoom": ZOOM_DEFAULT,
    }
    if not args.no_display:
        cv2.namedWindow(DISPLAY_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.namedWindow(CONTROL_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(CONTROL_WINDOW_NAME, _control_click_handler, control_state)
        cv2.createTrackbar(
            ZOOM_TRACKBAR_NAME,
            CONTROL_WINDOW_NAME,
            _zoom_to_slider(ZOOM_DEFAULT),
            ZOOM_TRACKBAR_MAX,
            lambda value: _zoom_trackbar_handler(value, view_state),
        )
        cv2.setTrackbarMin(ZOOM_TRACKBAR_NAME, CONTROL_WINDOW_NAME, ZOOM_TRACKBAR_MIN)
        cv2.imshow(CONTROL_WINDOW_NAME, _draw_controls(control_state["paused"]))

    try:
        for i, (frame_id, image_name, lidar_name) in enumerate(selected):
            image_path = os.path.join(image_dir, image_name)
            lidar_path = os.path.join(lidar_dir, lidar_name)
            try:
                vis, image, uv, depth, n_lidar_points, n_valid_points, stats = process_frame(
                    image_path,
                    lidar_path,
                    tr_velo_to_cam,
                    r_rect,
                    p_rect,
                    max_points=args.max_draw_points,
                    debug_mode=getattr(args, 'debug_mode', 0)
                )

                print(
                    f"Frame {frame_id}: lidar={n_lidar_points}, "
                    f"cam_valid={stats['after_cam_filter']}, "
                    f"proj_valid={stats['after_proj_filter']}, "
                    f"in_frame={n_valid_points}"
                )

                if args.save_output:
                    out_path = os.path.join(args.save_dir, f"{frame_id}.png")
                    ok = cv2.imwrite(out_path, vis)
                    if not ok:
                        print(f"Warning: failed to save {out_path}")

                if not args.no_display:
                    display_vis = _render_image_only_zoom(
                        image,
                        uv,
                        depth,
                        view_state["zoom"],
                        args.max_draw_points,
                    )
                    cv2.imshow(DISPLAY_WINDOW_NAME, display_vis)
                    quit_requested = False

                    while True:
                        cv2.imshow(CONTROL_WINDOW_NAME, _draw_controls(control_state["paused"]))
                        display_vis = _render_image_only_zoom(
                            image,
                            uv,
                            depth,
                            view_state["zoom"],
                            args.max_draw_points,
                        )
                        cv2.imshow(DISPLAY_WINDOW_NAME, display_vis)
                        wait_ms = 50 if control_state["paused"] else args.wait_ms
                        key = cv2.waitKey(wait_ms) & 0xFF

                        if key == ord("q"):
                            print("User requested exit (q)")
                            quit_requested = True
                            break

                        if key == ord("s"):
                            save_path = os.path.join(args.save_dir, f"{frame_id}_manual.png")
                            cv2.imwrite(save_path, vis)
                            print(f"Saved snapshot: {save_path}")

                        if key in (ord("+"), ord("=")):
                            _set_zoom(view_state, view_state["zoom"] + ZOOM_STEP)
                            cv2.setTrackbarPos(
                                ZOOM_TRACKBAR_NAME,
                                CONTROL_WINDOW_NAME,
                                _zoom_to_slider(view_state["zoom"]),
                            )

                        if key == ord("-"):
                            _set_zoom(view_state, view_state["zoom"] - ZOOM_STEP)
                            cv2.setTrackbarPos(
                                ZOOM_TRACKBAR_NAME,
                                CONTROL_WINDOW_NAME,
                                _zoom_to_slider(view_state["zoom"]),
                            )

                        if key == ord("0"):
                            _reset_view(view_state)
                            cv2.setTrackbarPos(
                                ZOOM_TRACKBAR_NAME,
                                CONTROL_WINDOW_NAME,
                                _zoom_to_slider(view_state["zoom"]),
                            )

                        if control_state["paused"]:
                            continue

                        break

                    if quit_requested:
                        break

                processed += 1
            except Exception as frame_error:
                skipped += 1
                print(f"Frame {frame_id}: skipped due to error: {frame_error}")
    finally:
        cv2.destroyAllWindows()

    print(f"Done. Processed={processed}, Skipped={skipped}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Project KITTI LiDAR onto camera images")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--sync-mode", choices=["filename", "timestamp"], default="filename")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--max-draw-points", type=int, default=12000)
    parser.add_argument("--wait-ms", type=int, default=30)
    parser.add_argument("--save-output", action="store_true")
    parser.add_argument("--save-dir", default="outputs")
    parser.add_argument("--no-display", action="store_true")
    return parser


def main():
    parser = build_arg_parser()
    parser.add_argument(
        "--debug-mode",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="0: all, 1: near (<30m), 2: far (>=30m), 3: edge-aligned points"
    )
    args = parser.parse_args()
    if args.step <= 0:
        raise ValueError("--step must be > 0")
    run_pipeline(args)


if __name__ == "__main__":
    main()
