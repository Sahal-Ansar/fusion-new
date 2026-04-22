# Project Context: LiDAR-RGB-Event Fusion Pipeline

## 1. Project Overview

This repository implements a classical sensor-fusion pipeline for projecting KITTI LiDAR points into the image plane, estimating image motion between consecutive RGB frames, simulating event-style activity from brightness changes, and applying a confidence-gated motion correction to projected LiDAR points.

At a high level, the project has two major execution modes:

1. Interactive pipeline mode in `main.py`
   This lets you step through frame pairs visually and inspect:
   - original LiDAR projection on frame `t`
   - simulated event map from `RGB_t` and `RGB_t+1`
   - motion-corrected LiDAR projection on frame `t+1`
   - corrected LiDAR projection drawn back on frame `t` for comparison

2. Validation and ablation mode in `validate_pipeline.py`
   This runs numerical evaluation across multiple KITTI sequences and compares:
   - original projection behavior
   - RGB-only motion warping
   - corrected fusion pipeline
   - smoothing vs no smoothing

The code is deliberately classical and explainable. It does not depend on deep learning frameworks. Core dependencies are NumPy and OpenCV.

## 2. Current Repository Structure

Current root: `C:\Users\sahaa\OneDrive\Desktop\Honors\fusion-revised1`

```text
fusion-revised1/
|-- calibration.py
|-- context.md
|-- events.py
|-- flow.py
|-- howto.txt
|-- lidar_motion.py
|-- loader.py
|-- main.py
|-- projection.py
|-- validate_pipeline.py
|-- validation_outputs/
|   |-- debug_mode_1_*.png
|   |-- debug_mode_2_*.png
|   `-- debug_mode_3_*.png
|-- test_logs/
|   `-- validation_*.txt
|-- __pycache__/
|-- .venv/
`-- .git/
```

## 3. File-by-File Responsibilities

### `loader.py`
Purpose:
- filesystem-safe loading of images and LiDAR frames
- simple file listing for datasets
- timestamp parsing for KITTI-style timestamp files

Key functions:
- `list_frame_files(directory, suffix)`
- `load_image(image_path)`
- `load_lidar(lidar_path)`
- `parse_timestamps(timestamp_path)`

Technical details:
- images are loaded with `cv2.imread(..., cv2.IMREAD_COLOR)`
- LiDAR `.bin` files are assumed to be KITTI format `(x, y, z, reflectance)` as `float32`
- only XYZ is used downstream; reflectance is ignored
- `.txt` LiDAR fallback is supported for derived or exported datasets
- `parse_timestamps()` normalizes fractional second precision to microseconds before parsing

Error handling:
- raises explicit exceptions for missing files/directories
- validates `.bin` file length is divisible by 4
- handles empty LiDAR files by returning an empty `(0, 3)` array

### `calibration.py`
Purpose:
- parse KITTI calibration text files
- construct the matrices needed to map LiDAR coordinates into the camera image plane

Key functions:
- `_parse_calibration_file(calib_path)`
- `parse_calib_velo_to_cam(calib_path)`
- `parse_calib_cam_to_cam(calib_path, camera_id="02")`

Technical details:
- `_parse_calibration_file()` reads `key: value...` lines and keeps only numeric entries
- non-numeric metadata lines are ignored
- `parse_calib_velo_to_cam()` builds `Tr_velo_to_cam` as a `4x4` homogeneous transform from:
  - `R` as `3x3`
  - `T` as `3x1`
- `parse_calib_cam_to_cam()` builds:
  - `R_rect` as `4x4` from `R_rect_00`
  - `P_rect` as `3x4` from `P_rect_02` by default

Important project assumption:
- camera `02` is the active image stream used throughout the main and validation pipelines

### `projection.py`
Purpose:
- perform geometric projection of LiDAR points into the image plane
- render projected points as a depth-colored overlay

Key functions:
- `project_lidar_to_image(lidar_xyz, tr_velo_to_cam, r_rect, p_rect, image_shape)`
- `overlay_points(image_bgr, uv, depth, max_points=12000, radius=1)`

Projection implementation:
1. convert LiDAR XYZ to homogeneous coordinates
2. compose `rectified_tf = R_rect @ Tr_velo_to_cam`
3. compose projection matrix `proj = P_rect @ rectified_tf`
4. transform LiDAR points into rectified camera coordinates
5. filter points behind the camera using positive camera-space Z
6. project into image coordinates
7. normalize with projected Z
8. filter non-finite and out-of-frame points

Returned values:
- `uv`: projected image coordinates `(M, 2)`
- `depth`: camera-space depth `(M,)`
- `stats`: counts used for debugging and validation

Visualization details:
- depth is normalized and mapped with `cv2.COLORMAP_JET`
- if point count exceeds `max_points`, points are uniformly subsampled for drawing

### `events.py`
Purpose:
- simulate event-camera-like activity from two RGB frames
- generate an event confidence mask used to gate LiDAR motion updates

Key functions:
- `simulate_events(img1, img2, threshold=0.2)`
- `events_to_image(events)`
- `event_confidence(events)`

Technical details:
- converts RGB/BGR frames to grayscale
- applies a log-intensity difference: `log(I_t+1) - log(I_t)`
- positive changes become `+1`
- negative changes become `-1`
- small changes within the threshold are zero

Interpretation:
- event simulation is not a true event camera model
- it is a simple surrogate for motion/activity emphasis between frames

Confidence generation:
- `event_confidence()` uses absolute event polarity magnitude
- result is normalized into `[0, 1]`
- in the current implementation, because events are binary `{-1, 0, 1}`, the confidence is effectively:
  - `1` where an event exists
  - `0` otherwise

### `flow.py`
Purpose:
- compute dense optical flow from either event maps or RGB frames
- optionally smooth flow over time

Key functions:
- `_zero_flow(shape)`
- `_to_gray(image)`
- `compute_flow(events1, events2)`
- `compute_rgb_flow(img1, img2)`
- `smooth_flow(flow, prev_flow, alpha=0.7)`

Technical details:
- both flow methods use OpenCV Farneback optical flow
- `compute_flow()` converts sparse event maps into binary images first
- `compute_rgb_flow()` works directly on grayscale image frames
- all flow outputs are converted to `float32`
- NaN/Inf values are zeroed out using `np.nan_to_num`

Current practical usage:
- `main.py` and `validate_pipeline.py` primarily use `compute_rgb_flow()`
- `compute_flow()` exists as an event-based option but is not the main path in current evaluation

Temporal smoothing:
- `smooth_flow()` applies exponential moving average smoothing
- formula: `alpha * current + (1 - alpha) * previous`
- if shape mismatch occurs, smoothing is skipped and current flow is returned

### `lidar_motion.py`
Purpose:
- move projected LiDAR points according to image-space motion
- gate those updates using event-derived confidence

Key function:
- `move_lidar_points_weighted(uv, depth, flow, conf, conf_thresh=0.2)`

Implementation details:
- projected LiDAR points are sampled into image indices using rounded pixel coordinates
- RGB flow vectors `(dx, dy)` are sampled at those pixels
- confidence is sampled from the event confidence map
- only points whose confidence exceeds `conf_thresh` are moved
- points with weak or absent event evidence stay at their original image coordinates

Important behavior:
- this function moves only the 2D projected positions
- it does not alter 3D LiDAR geometry or recalibrate the sensors
- depth is preserved for visualization and evaluation

This is the core fusion idea of the repository:
- RGB flow provides motion direction/magnitude
- events decide where that flow should be trusted enough to update projected LiDAR points

### `main.py`
Purpose:
- user-facing interactive visualizer and demo runner

Key functions:
- `process_frame(...)`
- `process_motion_frame(...)`
- `run_pipeline(args)`
- `main()`

What `process_frame()` does:
- loads one image and one LiDAR frame
- projects LiDAR into the image
- optionally filters by debug mode
- overlays the points for visualization

Debug modes in `process_frame()`:
- `0`: no extra filtering
- `1`: near points only, `depth < 30m`
- `2`: far points only, `depth >= 30m`
- `3`: edge-aligned points only using a distance transform from Canny edges

What `process_motion_frame()` does:
- computes original projection at frame `t`
- loads frame `t+1`
- simulates events from `t` and `t+1`
- computes RGB optical flow from `t` to `t+1`
- smooths flow with the previous frame pair’s flow
- builds event confidence
- moves LiDAR points using `move_lidar_points_weighted()`
- renders corrected projections on both frame `t+1` and frame `t`

Displayed windows:
- `Original Projection (t)`
- `Event Image`
- `Corrected Projection (t+1)`
- `Corrected Projection (t)`

Interactive controls:
- `n`: next frame pair
- `p`: previous frame pair
- `q`: quit

Dataset assumptions:
- images come from `image_02/data`
- LiDAR comes from `velodyne_points/data`
- image and LiDAR files are zipped by sorted filename order

Default dataset:
- `2011_09_26_drive_0009_extract`

### `validate_pipeline.py`
Purpose:
- numerical validation harness for projection correctness and temporal consistency
- ablation runner comparing multiple processing modes
- multi-dataset aggregation script for publication-style logs

This is the largest and most research-oriented file in the repository.

Main responsibilities:
- create traceable projected frame objects
- build frame-to-frame motion pairs
- compute edge alignment metrics
- measure temporal inconsistency statistically
- compare:
  - original projections
  - RGB-only warped projections
  - corrected fusion projections
  - smoothing vs no smoothing
- run the above across multiple KITTI datasets

Hardcoded validation datasets:
- `2011_09_26_drive_0009_sync`
- `2011_09_26_drive_0005_sync`
- `2011_09_26_drive_0013_sync`
- `2011_09_26_drive_0017_sync`

Key data classes:
- `FrameProjection`
  Stores a full traced projection for one frame, including raw image, LiDAR, homogeneous coordinates, rectified camera coordinates, projected coordinates, filtered UV points, depth, and statistics.
- `MotionFramePair`
  Stores paired frame data for temporal evaluation, including RGB-only warped points, corrected points, corrected image, events, confidence, and flow.
- `TeeStream`
  Writes validation output to both console and log file.

Important helper functions:
- `_project_with_full_trace(...)`
  Projection with retained intermediate arrays for validation and reprojection checks.
- `_distance_transform_from_edges(image)`
  Builds edge maps and distance transforms used by alignment metrics.
- `_detect_vertical_structures(frame, min_points=12)`
  Finds candidate vertical image structures with Hough lines and associated LiDAR support.
- `_temporal_profile(candidate, frame, bins=5)`
  Measures consistency of a vertical structure across y-bins.
- `_select_tracked_candidates(frames)`
  Chooses stable candidates across multiple frames.
- `_build_motion_frame_pair(...)`
  Builds original, RGB-only, and corrected warped outputs for a frame pair.
- `_build_motion_frame_pairs(frames, use_smoothing=False)`
  Produces a sequence of motion pairs, optionally carrying previous flow for EMA smoothing.
- `_sample_flow_at_points(flow, points)`
  Bilinear-samples dense flow at arbitrary UV locations.
- `_move_points_with_rgb_flow(uv, depth, flow)`
  Baseline ablation that moves projected LiDAR with RGB flow only, without event gating.
- `_compute_edge_alignment(uv_int, image)`
  Measures projected point alignment to strong image edges.
- `temporal_statistics_test(...)`
  Core statistical metric for temporal inconsistency.
- `_evaluate_mode_metrics(...)`
  Produces summarized metrics for one mode.
- `run_validation_on_dataset(dataset_path)`
  Runs the full numerical validation flow on one dataset and returns structured results.
- `run_all_datasets()`
  Runs the same validation across all hardcoded datasets and prints aggregated results.

Validation metrics used:
- edge alignment error
- temporal inconsistency mean/median error
- baseline image self-consistency error
- RGB-only vs corrected improvement
- smoothing effect per mode

Research framing:
- original = uncorrected temporal mismatch between projected LiDAR and observed image motion
- RGB-only = full optical-flow warping baseline
- corrected = event-gated fusion output
- smoothing = temporal stabilization of dense flow before applying correction

Current output behavior:
- per-dataset logs with:
  - no smoothing block
  - with smoothing block
  - smoothing effect block
- final aggregated averages
- best and worst dataset improvement
- log files written into `test_logs/`

### `howto.txt`
Purpose:
- lightweight run notes and setup reminders

Important note:
- `howto.txt` is partially outdated relative to current `main.py`
- it mentions CLI flags such as `--start`, `--end`, `--no-display`, `--sync-mode`, and `--save-output`
- those flags do not exist in the current `main.py`

Treat `howto.txt` as historical notes, not authoritative runtime documentation.

## 4. End-to-End Data Flow

### Interactive runtime path
1. load one RGB frame and one LiDAR scan
2. parse KITTI calibration
3. project LiDAR into the image plane
4. load next RGB frame
5. simulate event activity from the RGB pair
6. compute dense RGB optical flow
7. optionally smooth flow with previous flow
8. derive event confidence mask
9. move projected LiDAR points only where event confidence is strong enough
10. render original and corrected overlays for visual inspection

### Validation path
1. load a fixed number of frame pairs from each dataset
2. project LiDAR with full trace retention
3. build frame-to-frame motion pairs
4. compute original, RGB-only, and corrected metrics
5. repeat with and without flow smoothing
6. print per-dataset results
7. aggregate results across multiple datasets

## 5. Core Technical Design Choices

### 2D correction instead of 3D correction
The project corrects projected LiDAR points in image space, not in 3D world space. This keeps the implementation simple and directly aligned with image-level consistency metrics.

### Event-gated motion
The fusion step does not blindly apply RGB optical flow everywhere. Instead, it uses event activity as a confidence gate. This is intended to reduce overcorrection in static or low-confidence regions.

### RGB-only baseline
The repository includes an RGB-only ablation path. That is important because it isolates the value of event gating from the value of flow itself.

### Smoothed vs unsmoothed flow
Flow smoothing is treated as a separate ablation. The validation harness explicitly compares both modes.

### Classical geometry preserved
Projection and calibration are treated as fixed and validated components. Motion correction operates after projection rather than altering calibration math.

## 6. Important Mathematical Components

### KITTI projection chain
The core geometry follows:

`image = P_rect @ R_rect @ Tr_velo_to_cam @ lidar_h`

Where:
- `Tr_velo_to_cam` maps LiDAR coordinates into camera coordinates
- `R_rect` rectifies the camera frame
- `P_rect` projects rectified camera coordinates to the image plane
- `lidar_h` is the homogeneous LiDAR point matrix

### Event simulation
The event surrogate is based on:

`diff = log(I_t+1) - log(I_t)`

Then:
- `diff > threshold` -> positive event
- `diff < -threshold` -> negative event
- otherwise -> no event

### Flow smoothing
Exponential moving average:

`flow_smooth = alpha * flow_current + (1 - alpha) * flow_previous`

### Event-gated LiDAR motion
For each projected point `(u, v)`:
- sample RGB flow `(dx, dy)`
- sample event confidence `c`
- if `c > conf_thresh`, move point to `(u + dx, v + dy)`
- otherwise, keep `(u, v)`

## 7. Outputs Generated by the Project

### Runtime outputs
- on-screen OpenCV windows from `main.py`
- console debug summaries per frame

### Validation outputs
- `validation_outputs/debug_mode_*.png`
- `test_logs/validation_*.txt`

### Documentation
- this file, `context.md`

## 8. External Data and Environment Assumptions

Expected dataset layout per sequence:

```text
<dataset_root>/
|-- calib_cam_to_cam.txt
|-- calib_velo_to_cam.txt
|-- image_02/
|   `-- data/
|       `-- *.png
`-- velodyne_points/
    `-- data/
        `-- *.bin
```

Common datasets referenced in this project:
- extracted sequence for interactive use:
  - `2011_09_26_drive_0009_extract`
- synchronized sequences for validation:
  - `2011_09_26_drive_0009_sync`
  - `2011_09_26_drive_0005_sync`
  - `2011_09_26_drive_0013_sync`
  - `2011_09_26_drive_0017_sync`

Dependencies:
- Python
- NumPy
- OpenCV (`opencv-python`)

## 9. Current Limitations and Caveats

- event simulation is synthetic, not sourced from a real event camera
- LiDAR-image pairing in `main.py` is filename-order based rather than timestamp synchronized
- the correction is purely 2D and does not reconstruct corrected 3D point positions
- no explicit occlusion handling is implemented
- no learned confidence model is used; event confidence is currently binary-like
- `howto.txt` does not fully match the current CLI
- `validate_pipeline.py` is comprehensive but large; maintenance cost is concentrated there

## 10. Quick Mental Model

If you need a short way to think about the repo:

- `loader.py` gets the data in
- `calibration.py` builds the camera/LiDAR transforms
- `projection.py` maps LiDAR into the image
- `events.py` marks where brightness changed
- `flow.py` estimates how image content moved
- `lidar_motion.py` uses event confidence to decide where projected LiDAR points should move
- `main.py` visualizes the pipeline
- `validate_pipeline.py` measures whether the correction is actually helping

## 11. Most Important Entry Points

For interactive usage:
- `main.py -> main() -> run_pipeline() -> process_motion_frame()`

For evaluation:
- `validate_pipeline.py -> run_all_datasets()`
- `validate_pipeline.py -> run_validation_on_dataset(dataset_path)`

For core fusion behavior:
- `events.py -> simulate_events()`
- `flow.py -> compute_rgb_flow()`
- `flow.py -> smooth_flow()`
- `lidar_motion.py -> move_lidar_points_weighted()`

## 12. Recommended Maintenance Priorities

If this repo is extended further, the highest-value follow-ups would be:

1. bring `howto.txt` in sync with the actual CLI
2. add a small `README.md` for setup and usage
3. split `validate_pipeline.py` into smaller modules
4. optionally add timestamp-based pairing to `main.py`
5. make event confidence richer than binary polarity magnitude
