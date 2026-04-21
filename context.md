# LiDAR to Camera Projection Pipeline - Context

## Overview
This project contains a production-quality Python pipeline for projecting KITTI LiDAR points onto camera images.

It includes:
- robust frame loading
- robust calibration parsing
- vectorized 3D-to-2D projection
- filtering of invalid projected points
- depth-colored overlay rendering
- interactive display and optional saving

## Project Root
`C:\Users\sahaa\OneDrive\Desktop\Honors\fusion-revised1`

## Dataset Path
`C:\Users\sahaa\OneDrive\Desktop\Honors\datasets\fusion\2011_09_26_drive_0009_extract`

## Current File Structure
```
fusion-revised1/
├── loader.py
├── calibration.py
├── projection.py
├── main.py
├── howto.txt
└── context.md
```

## Module Responsibilities

### loader.py
Functions:
- `list_frame_files(directory, suffix)`
- `load_image(image_path)`
- `load_lidar(lidar_path)`
- `parse_timestamps(timestamp_path)`

Behavior:
- loads images with `cv2.imread(..., cv2.IMREAD_COLOR)` and validation
- loads LiDAR from `.bin` using `numpy.fromfile(..., float32)`, reshapes to `(-1, 4)`, uses only XYZ
- supports `.txt` LiDAR fallback for derived datasets
- handles empty/corrupted files with clear exceptions
- parses KITTI timestamps (including long fractional precision)

### calibration.py
Functions:
- `_parse_calibration_file(calib_path)`
- `parse_calib_velo_to_cam(calib_path)`
- `parse_calib_cam_to_cam(calib_path, camera_id="02")`

Behavior:
- parses calibration key/value lines into float arrays
- safely ignores non-numeric metadata lines (for example `calib_time`)
- builds:
   - `Tr_velo_to_cam` as 4x4 from `R` and `T`
   - `R_rect` as 4x4 from `R_rect_00`
   - `P_rect` as 3x4 from `P_rect_02`
- validates presence and shape of required keys
- uses `float64` throughout

### projection.py
Functions:
- `project_lidar_to_image(lidar_xyz, tr_velo_to_cam, r_rect, p_rect, image_shape)`
- `overlay_points(image_bgr, uv, depth, max_points=12000, radius=1)`

Projection pipeline:
1. convert XYZ to homogeneous coordinates
2. transform: `cam = R_rect @ Tr_velo_to_cam @ lidar_h`
3. remove points behind camera and non-finite values
4. project with `P_rect`
5. normalize by projected Z
6. clip to image bounds

Returns:
- projected points `uv`
- depth values
- debug `stats` (`input_points`, `after_cam_filter`, `after_proj_filter`, `in_frame`)

Visualization:
- depth-normalized coloring with `cv2.COLORMAP_JET`
- point cap to avoid drawing overload
- `cv2.circle` rendering

### main.py
Entry points:
- `run_pipeline(args)`
- `main()` with CLI argument parser

Sync strategies:
- filename sync (`--sync-mode filename`, default)
- timestamp nearest-neighbor sync (`--sync-mode timestamp`, optional bonus)

Frame loop behavior:
- loads synchronized image/LiDAR
- projects points
- prints debug counts per frame
- displays visualization unless `--no-display`
- supports keyboard controls:
   - `q` to quit
   - `s` to save snapshot

## Edge Cases Handled
- missing image/LiDAR/calibration/timestamp files
- empty LiDAR frames
- malformed `.bin` LiDAR (size not divisible by 4)
- invalid or missing calibration keys
- non-numeric calibration metadata
- division-by-zero during projection via Z filtering
- NaN/inf filtering
- out-of-frame point filtering
- mismatched image/LiDAR counts
- per-frame exception isolation (skip bad frame, continue)

## Debug Output
At startup:
- `Tr_velo_to_cam` shape
- `R_rect` shape
- `P_rect_02` shape
- image/LiDAR counts and synchronized pair count

Per frame:
- LiDAR input count
- camera-valid count
- projection-valid count
- in-frame projected count

Final summary:
- processed/skipped frame totals

## Performance Notes
- vectorized NumPy projection math (no per-point projection loops)
- `float64` for geometric stability
- bounded draw count for visualization responsiveness
- no deep learning or external dependencies beyond NumPy/OpenCV

## Run Examples
Display mode:
`python main.py`

Headless quick test:
`python main.py --start 0 --end 5 --no-display --sync-mode filename`

Save overlaid output frames:
`python main.py --save-output --save-dir outputs`