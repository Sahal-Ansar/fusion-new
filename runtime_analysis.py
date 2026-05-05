"""
runtime_analysis.py — Phase 4 runtime / complexity analysis.

Measures wall-clock time of every pipeline stage on a single KITTI
frame pair and produces:
  - Table 1: per-stage breakdown (mean ± std ms over N repeats)
  - Table 2: full-pipeline comparison (No Correction / RGB-Only / Ours)
  - LaTeX booktabs version of both tables for the paper

Standalone — no modifications to existing files. CPU-only timings.
"""

import argparse
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

try:
    from calibration import parse_calib_cam_to_cam, parse_calib_velo_to_cam
    from events import event_confidence, simulate_events
    from flow import compute_rgb_flow, smooth_flow
    from lidar_motion import move_lidar_points_weighted
    from loader import list_frame_files, load_image, load_lidar
    from projection import project_lidar_to_image
except ImportError as exc:
    print(f"[runtime_analysis] FATAL: missing required module ({exc}).")
    sys.exit(1)


DEFAULT_DATASET = (
    r"C:\Users\sahaa\OneDrive\Desktop\Honors\datasets\fusion"
    r"\2011_09_26_drive_0009_sync"
)
DEFAULT_OUTPUT_DIR = "./runtime_results"
DEFAULT_N_REPEATS = 20
DEFAULT_FRAME_IDX = 50

# Order matters for the printed/saved tables.
STAGE_NAMES = (
    "Farneback Flow",
    "Temporal Smoothing",
    "Event Simulation",
    "Event Confidence",
    "Motion Compensation",
    "TOTAL Pipeline",
)


# =======================================================================
# Helpers
# =======================================================================

@contextlib.contextmanager
def _silent():
    """Suppress stdout from verbose internal prints during timing."""
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        yield


def _time_callable(fn, n_repeats: int, warmup: int = 3) -> Dict[str, float]:
    """Time a zero-arg callable, returning {mean_ms, std_ms, min_ms, max_ms}.

    Suppresses stdout from the callable so verbose internal prints do
    not contaminate the measurement (StringIO writes are fast and
    consistent across calls, so they do not bias relative timings).
    """
    if n_repeats <= 0:
        raise ValueError(f"n_repeats must be > 0, got {n_repeats}")

    with _silent():
        for _ in range(max(0, warmup)):
            fn()

        samples_ms: List[float] = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            fn()
            t1 = time.perf_counter()
            samples_ms.append((t1 - t0) * 1000.0)

    arr = np.asarray(samples_ms, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std(ddof=0)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


# =======================================================================
# Per-stage timing
# =======================================================================

def time_pipeline_stages(
    image_t: np.ndarray,
    image_t1: np.ndarray,
    uv: np.ndarray,
    depth: np.ndarray,
    n_repeats: int = DEFAULT_N_REPEATS,
) -> Dict[str, Dict[str, float]]:
    """Time each pipeline stage independently on the supplied frame pair.

    Each stage is warmed up 3 times and then measured n_repeats times
    with time.perf_counter(). The TOTAL Pipeline entry runs every stage
    sequentially in one timed block (so it captures both the per-stage
    work and any pass-through overhead).
    """
    # Pre-compute the inputs each downstream stage needs. These are not
    # part of the timed work — they are the snapshots fed into each
    # stage so per-stage timing is decoupled from upstream variance.
    flow_snapshot = compute_rgb_flow(image_t, image_t1)
    events_snapshot = simulate_events(image_t, image_t1, threshold=0.3)
    with _silent():
        conf_snapshot = event_confidence(events_snapshot)

    timings: Dict[str, Dict[str, float]] = {}

    # Stage: Farneback flow
    timings["Farneback Flow"] = _time_callable(
        lambda: compute_rgb_flow(image_t, image_t1),
        n_repeats=n_repeats,
    )

    # Stage: Temporal smoothing (EMA against the same snapshot each time)
    prev_flow = flow_snapshot.copy()
    timings["Temporal Smoothing"] = _time_callable(
        lambda: smooth_flow(flow_snapshot, prev_flow, alpha=0.7),
        n_repeats=n_repeats,
    )

    # Stage: Event simulation
    timings["Event Simulation"] = _time_callable(
        lambda: simulate_events(image_t, image_t1, threshold=0.3),
        n_repeats=n_repeats,
    )

    # Stage: Event confidence
    timings["Event Confidence"] = _time_callable(
        lambda: event_confidence(events_snapshot),
        n_repeats=n_repeats,
    )

    # Stage: Motion compensation
    timings["Motion Compensation"] = _time_callable(
        lambda: move_lidar_points_weighted(uv, depth, flow_snapshot, conf_snapshot),
        n_repeats=n_repeats,
    )

    # Stage: TOTAL Pipeline (all stages chained, prev_flow=None on first run)
    def _full_pipeline():
        flow_raw = compute_rgb_flow(image_t, image_t1)
        flow_s = smooth_flow(flow_raw, None, alpha=0.7)
        events = simulate_events(image_t, image_t1, threshold=0.3)
        conf = event_confidence(events)
        move_lidar_points_weighted(uv, depth, flow_s, conf)

    timings["TOTAL Pipeline"] = _time_callable(_full_pipeline, n_repeats=n_repeats)

    return timings


# =======================================================================
# Method comparison
# =======================================================================

def time_comparison_methods(
    image_t: np.ndarray,
    image_t1: np.ndarray,
    uv: np.ndarray,
    depth: np.ndarray,
    n_repeats: int = DEFAULT_N_REPEATS,
) -> Dict[str, Dict[str, float]]:
    """Time three end-to-end correction strategies for the headline table."""
    h, w = image_t1.shape[:2]
    ones_conf = np.ones((h, w), dtype=np.float32)

    methods: Dict[str, Dict[str, float]] = {}

    # No Correction: identity over uv. We measure np.copy(uv) as a
    # proxy for "do nothing" so the row carries a real (tiny) number
    # rather than literal zero, which would be misleading.
    methods["No Correction"] = _time_callable(
        lambda: np.copy(uv),
        n_repeats=n_repeats,
    )

    # RGB-Only Flow: Farneback + motion compensation with all-ones conf.
    def _rgb_only():
        flow = compute_rgb_flow(image_t, image_t1)
        move_lidar_points_weighted(uv, depth, flow, ones_conf)

    methods["RGB-Only Flow"] = _time_callable(_rgb_only, n_repeats=n_repeats)

    # Ours (Event-Guided): full pipeline.
    def _ours():
        flow_raw = compute_rgb_flow(image_t, image_t1)
        flow_s = smooth_flow(flow_raw, None, alpha=0.7)
        events = simulate_events(image_t, image_t1, threshold=0.3)
        conf = event_confidence(events)
        move_lidar_points_weighted(uv, depth, flow_s, conf)

    methods["Ours (Event-Guided)"] = _time_callable(_ours, n_repeats=n_repeats)

    return methods


# =======================================================================
# Output
# =======================================================================

def print_runtime_table(
    stage_times: Dict[str, Dict[str, float]],
    method_times: Dict[str, Dict[str, float]],
) -> None:
    """Print Table 1 (per-stage) and Table 2 (method comparison) to stdout."""
    print()
    print("Table 1 — Per-stage breakdown")
    print("-" * 60)
    print(f"{'Stage':<24}{'Mean (ms)':>14}{'Std (ms)':>14}")
    print("-" * 60)
    for name in STAGE_NAMES:
        if name not in stage_times:
            continue
        if name == "TOTAL Pipeline":
            print("-" * 60)
        s = stage_times[name]
        print(f"{name:<24}{s['mean_ms']:>14.3f}{s['std_ms']:>14.3f}")
    print("-" * 60)

    print()
    print("Table 2 — Method comparison")
    print("-" * 60)
    print(f"{'Method':<24}{'Total (ms)':>14}{'FPS':>14}")
    print("-" * 60)
    for name in ("No Correction", "RGB-Only Flow", "Ours (Event-Guided)"):
        if name not in method_times:
            continue
        m = method_times[name]
        mean_ms = m["mean_ms"]
        if mean_ms > 1e-6:
            fps = 1000.0 / mean_ms
            fps_s = f"{fps:>14.1f}"
        else:
            fps_s = f"{'inf':>14}"
        print(f"{name:<24}{mean_ms:>14.3f}{fps_s}")
    print("-" * 60)
    print("Note: FPS = 1000 / mean_ms (single-frame processing rate, "
          "not steady-state real-time throughput).")


def generate_runtime_latex(
    stage_times: Dict[str, Dict[str, float]],
    method_times: Dict[str, Dict[str, float]],
    output_path: str,
) -> str:
    """IEEE-booktabs LaTeX runtime tables; also returned as a string."""
    def _fmt(x: float, prec: int = 2) -> str:
        if not isinstance(x, (int, float)) or not np.isfinite(x):
            return "--"
        return f"{x:.{prec}f}"

    lines: List[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Per-stage runtime breakdown on a single frame "
                 r"pair (1242$\times$375 image, $\sim$15k LiDAR points). "
                 r"All timings on CPU only (no GPU required). Mean $\pm$ std "
                 r"over 20 repeated measurements.}")
    lines.append(r"  \label{tab:runtime}")
    lines.append(r"  \small")

    # Subtable A: per-stage breakdown
    lines.append(r"  \begin{minipage}{0.48\linewidth}")
    lines.append(r"    \centering")
    lines.append(r"    \subcaption{Per-stage breakdown.}")
    lines.append(r"    \begin{tabular}{lr}")
    lines.append(r"      \toprule")
    lines.append(r"      Stage & Time (ms) \\")
    lines.append(r"      \midrule")
    for name in STAGE_NAMES:
        if name not in stage_times:
            continue
        s = stage_times[name]
        time_str = f"{_fmt(s['mean_ms'])} $\\pm$ {_fmt(s['std_ms'])}"
        if name == "TOTAL Pipeline":
            lines.append(r"      \midrule")
            lines.append(f"      \\textbf{{{name}}} & \\textbf{{{time_str}}} \\\\")
        else:
            lines.append(f"      {name} & {time_str} \\\\")
    lines.append(r"      \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"  \end{minipage}")
    lines.append(r"  \hfill")

    # Subtable B: method comparison
    lines.append(r"  \begin{minipage}{0.48\linewidth}")
    lines.append(r"    \centering")
    lines.append(r"    \subcaption{Method comparison (CPU only).}")
    lines.append(r"    \begin{tabular}{lrr}")
    lines.append(r"      \toprule")
    lines.append(r"      Method & Total (ms) & FPS \\")
    lines.append(r"      \midrule")
    for name in ("No Correction", "RGB-Only Flow", "Ours (Event-Guided)"):
        if name not in method_times:
            continue
        m = method_times[name]
        mean_ms = m["mean_ms"]
        fps_str = "$\\infty$" if mean_ms <= 1e-6 else _fmt(1000.0 / mean_ms, 1)
        method_label = name.replace("&", r"\&").replace("_", r"\_")
        if name.startswith("Ours"):
            lines.append(f"      \\textbf{{{method_label}}} & "
                         f"\\textbf{{{_fmt(mean_ms)}}} & \\textbf{{{fps_str}}} \\\\")
        else:
            lines.append(f"      {method_label} & {_fmt(mean_ms)} & {fps_str} \\\\")
    lines.append(r"      \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"  \end{minipage}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex + "\n")
    print(f"[runtime] saved LaTeX: {output_path}")
    return latex


# =======================================================================
# CLI
# =======================================================================

def _load_frame_pair(
    dataset_path: str,
    frame_idx: int,
) -> Dict[str, np.ndarray]:
    """Load image_t / image_t1 / uv / depth for one frame pair."""
    image_dir = os.path.join(dataset_path, "image_02", "data")
    lidar_dir = os.path.join(dataset_path, "velodyne_points", "data")
    velo_calib = os.path.join(dataset_path, "calib_velo_to_cam.txt")
    cam_calib = os.path.join(dataset_path, "calib_cam_to_cam.txt")

    image_files = list_frame_files(image_dir, ".png")
    try:
        lidar_files = list_frame_files(lidar_dir, ".bin")
    except FileNotFoundError:
        lidar_files = []
    if not lidar_files:
        lidar_files = list_frame_files(lidar_dir, ".txt")

    n_pairs = max(0, min(len(image_files), len(lidar_files)) - 1)
    if frame_idx < 0 or frame_idx >= n_pairs:
        raise ValueError(
            f"--frame-idx {frame_idx} out of range; sequence has {n_pairs} pairs."
        )

    image_t = load_image(os.path.join(image_dir, image_files[frame_idx]))
    image_t1 = load_image(os.path.join(image_dir, image_files[frame_idx + 1]))
    lidar_xyz = load_lidar(os.path.join(lidar_dir, lidar_files[frame_idx]))

    with _silent():
        tr_velo_to_cam = parse_calib_velo_to_cam(velo_calib)
        r_rect, p_rect_left = parse_calib_cam_to_cam(cam_calib, camera_id="02")
        uv, depth, _ = project_lidar_to_image(
            lidar_xyz, tr_velo_to_cam, r_rect, p_rect_left, image_t.shape
        )

    if uv.shape[0] == 0:
        raise RuntimeError(f"frame {frame_idx} produced zero projected points.")

    return {
        "image_t": image_t,
        "image_t1": image_t1,
        "uv": uv,
        "depth": depth,
    }


def main() -> None:
    # Windows consoles default to cp1252 and crash on non-ASCII (e.g. ±).
    # Best-effort UTF-8 reconfigure; harmless if already UTF-8 or unsupported.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="Phase 4 runtime / complexity analysis."
    )
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET,
                        help="Full path to one KITTI sequence directory.")
    parser.add_argument("--n-repeats", type=int, default=DEFAULT_N_REPEATS,
                        help="Per-stage timing repeats (after warm-up).")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Where to save JSON + LaTeX results.")
    parser.add_argument("--frame-idx", type=int, default=DEFAULT_FRAME_IDX,
                        help="Index of the frame pair to time.")
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_path):
        print(f"[runtime] dataset path not found: {args.dataset_path}")
        sys.exit(2)

    print("=" * 70)
    print("Phase 4 — Runtime / Complexity Analysis (CPU only)")
    print("=" * 70)
    print(f"Dataset:       {args.dataset_path}")
    print(f"Frame index:   {args.frame_idx}")
    print(f"Repeats:       {args.n_repeats}")
    print(f"Output dir:    {args.output_dir}")

    frame = _load_frame_pair(args.dataset_path, args.frame_idx)
    h, w = frame["image_t"].shape[:2]
    n_pts = int(frame["uv"].shape[0])
    print(f"Frame loaded:  {w}x{h} image, {n_pts} projected LiDAR points")

    print("\n[runtime] timing per-stage breakdown ...")
    stage_times = time_pipeline_stages(
        frame["image_t"], frame["image_t1"], frame["uv"], frame["depth"],
        n_repeats=args.n_repeats,
    )

    print("[runtime] timing method comparison ...")
    method_times = time_comparison_methods(
        frame["image_t"], frame["image_t1"], frame["uv"], frame["depth"],
        n_repeats=args.n_repeats,
    )

    print_runtime_table(stage_times, method_times)

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "runtime_results.json")
    payload = {
        "dataset_path": args.dataset_path,
        "frame_idx": args.frame_idx,
        "image_shape": [int(h), int(w)],
        "n_lidar_points": n_pts,
        "n_repeats": int(args.n_repeats),
        "stage_times_ms": stage_times,
        "method_times_ms": method_times,
        "notes": "CPU-only timings using time.perf_counter(); 3 warm-up "
                 "iterations per stage before measurement.",
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            payload, f, indent=2,
            default=lambda o: float(o) if isinstance(o, np.floating) else str(o),
        )
    print(f"\n[runtime] saved JSON: {json_path}")

    latex_path = os.path.join(args.output_dir, "runtime_table.tex")
    latex = generate_runtime_latex(stage_times, method_times, latex_path)
    print("\n--- LaTeX snippet ---")
    print(latex)
    print("--- end LaTeX ---")


if __name__ == "__main__":
    main()
