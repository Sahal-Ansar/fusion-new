"""
ablation_study.py — Phase 4 ablation study.

Evaluates eight configurations of the event-guided LiDAR temporal
correction pipeline on a single KITTI sequence and produces an ASCII
table + LaTeX booktabs table for the paper.

Each row of the ablation removes or modifies exactly one component
of the full method so the contribution of every design choice can be
read directly off the SPEAS / Stereo improvement columns.

Standalone — no modifications to existing files.
"""

import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

try:
    from calibration import parse_calib_cam_to_cam, parse_calib_velo_to_cam
    from events import event_confidence, simulate_events
    from flow import compute_rgb_flow, smooth_flow
    from lidar_motion import move_lidar_points_weighted  # noqa: F401 — kept for parity with the full pipeline
    from loader import list_frame_files, load_image, load_lidar
    from metrics import sparse_point_eas, stereo_consistency_score
    from projection import project_lidar_to_image
except ImportError as exc:
    print(f"[ablation_study] FATAL: missing required module ({exc}).")
    print("Make sure ablation_study.py lives next to calibration.py, events.py, "
          "flow.py, lidar_motion.py, loader.py, metrics.py, projection.py.")
    sys.exit(1)


# =======================================================================
# Defaults
# =======================================================================

DEFAULT_DATASET = (
    r"C:\Users\sahaa\OneDrive\Desktop\Honors\datasets\fusion"
    r"\2011_09_26_drive_0009_sync"
)
DEFAULT_OUTPUT_DIR = "./ablation_results"
DEFAULT_N_FRAMES = 100


# =======================================================================
# Ablation configurations
# =======================================================================

CONFIG_FULL = {
    "use_flow": True,
    "use_smoothing": True,
    "use_events": True,
    "use_edge_filter": True,
    "use_conf_gating": True,
    "use_flow_bounds": True,
    "alpha": 0.5,
    "name": "Full Method (Ours)",
}

CONFIG_NO_CORRECTION = {
    "use_flow": False,
    "use_smoothing": False,
    "use_events": False,
    "use_edge_filter": False,
    "use_conf_gating": False,
    "use_flow_bounds": False,
    "alpha": 0.5,
    "name": "No Correction (Baseline)",
}

CONFIG_FLOW_ONLY = {
    "use_flow": True,
    "use_smoothing": False,
    "use_events": False,
    "use_edge_filter": False,
    "use_conf_gating": False,
    "use_flow_bounds": False,
    "alpha": 0.5,
    "name": "RGB Flow Only (no gating)",
}

CONFIG_NO_SMOOTHING = {
    "use_flow": True,
    "use_smoothing": False,
    "use_events": True,
    "use_edge_filter": True,
    "use_conf_gating": True,
    "use_flow_bounds": True,
    "alpha": 0.5,
    "name": "No Temporal Smoothing",
}

CONFIG_NO_EDGE_FILTER = {
    "use_flow": True,
    "use_smoothing": True,
    "use_events": True,
    "use_edge_filter": False,
    "use_conf_gating": True,
    "use_flow_bounds": True,
    "alpha": 0.5,
    "name": "No Edge Spatial Filter",
}

CONFIG_NO_BOUNDS = {
    "use_flow": True,
    "use_smoothing": True,
    "use_events": True,
    "use_edge_filter": True,
    "use_conf_gating": True,
    "use_flow_bounds": False,
    "alpha": 0.5,
    "name": "No Flow Magnitude Bounds",
}

CONFIG_ALPHA_025 = {
    "use_flow": True,
    "use_smoothing": True,
    "use_events": True,
    "use_edge_filter": True,
    "use_conf_gating": True,
    "use_flow_bounds": True,
    "alpha": 0.25,
    "name": "Alpha=0.25 (quarter-frame)",
}

CONFIG_ALPHA_075 = {
    "use_flow": True,
    "use_smoothing": True,
    "use_events": True,
    "use_edge_filter": True,
    "use_conf_gating": True,
    "use_flow_bounds": True,
    "alpha": 0.75,
    "name": "Alpha=0.75 (three-quarter-frame)",
}

ALL_CONFIGS: List[dict] = [
    CONFIG_NO_CORRECTION,
    CONFIG_FLOW_ONLY,
    CONFIG_NO_SMOOTHING,
    CONFIG_NO_EDGE_FILTER,
    CONFIG_NO_BOUNDS,
    CONFIG_ALPHA_025,
    CONFIG_ALPHA_075,
    CONFIG_FULL,
]

BASELINE_NAMES = {CONFIG_NO_CORRECTION["name"], CONFIG_FLOW_ONLY["name"]}
ALPHA_NAMES = {CONFIG_ALPHA_025["name"], CONFIG_ALPHA_075["name"]}


# =======================================================================
# Helpers
# =======================================================================

@contextlib.contextmanager
def _silent():
    """Suppress stdout from verbose internal prints (event_confidence,
    move_lidar_points_weighted, sparse_point_eas, stereo_consistency_score)."""
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        yield


def _simulate_events_no_edge_filter(
    image_t: np.ndarray,
    image_t1: np.ndarray,
    threshold: float = 0.3,
) -> np.ndarray:
    """Log-intensity threshold events without the Canny spatial filter.

    Mirrors the events.simulate_events pipeline up to the Canny step,
    skipping the spatial-edge gate so we can ablate that component.
    """
    gray1 = cv2.cvtColor(image_t, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(image_t1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray1 = np.maximum(gray1, 1.0)
    gray2 = np.maximum(gray2, 1.0)

    diff = np.log(gray2) - np.log(gray1)
    diff = np.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)

    raw = np.zeros_like(diff, dtype=np.float32)
    raw[diff > threshold] = 1.0
    raw[diff < -threshold] = -1.0

    kernel = np.ones((3, 3), np.uint8)
    on_d = cv2.dilate((raw == 1.0).astype(np.uint8), kernel)
    off_d = cv2.dilate((raw == -1.0).astype(np.uint8), kernel)

    out = np.zeros_like(raw)
    out[on_d > 0] = 1.0
    out[off_d > 0] = -1.0
    out[(on_d > 0) & (off_d > 0)] = 0.0
    return out


def _move_points_ablation(
    uv: np.ndarray,
    depth: np.ndarray,
    flow: np.ndarray,
    conf: np.ndarray,
    alpha: float,
    use_bounds: bool,
    conf_thresh: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Inline motion-correction with toggleable alpha and bounds.

    The math mirrors lidar_motion.py:move_lidar_points_weighted exactly
    when alpha=0.5 and use_bounds=True. We re-implement (instead of
    using the imported function with a flow-scaling trick) because that
    function hardcodes ALPHA=0.5, FLOW_MIN=0.5, FLOW_MAX=30.0 as local
    constants. That hardcoding makes use_flow_bounds=False a silent
    no-op (the internal bounds always fire) and couples alpha to the
    effective bounds (since they are checked on the scaled flow). The
    inline version separates the two knobs cleanly.
    """
    uv = np.asarray(uv, dtype=np.float32)
    depth = np.asarray(depth, dtype=np.float32)
    flow = np.nan_to_num(np.asarray(flow, dtype=np.float32),
                         nan=0.0, posinf=0.0, neginf=0.0)
    conf = np.nan_to_num(np.asarray(conf, dtype=np.float32),
                         nan=0.0, posinf=0.0, neginf=0.0)

    if uv.size == 0 or flow.ndim != 3 or flow.shape[2] != 2:
        return uv.copy(), depth.copy()

    h, w = flow.shape[:2]
    u = uv[:, 0]
    v = uv[:, 1]

    u_idx = np.rint(u).astype(np.int32)
    v_idx = np.rint(v).astype(np.int32)
    in_bounds = (
        (u_idx >= 0) & (u_idx < w)
        & (v_idx >= 0) & (v_idx < h)
        & np.isfinite(u) & np.isfinite(v)
    )

    u_new = u.copy()
    v_new = v.copy()

    if not np.any(in_bounds):
        return np.stack([u_new, v_new], axis=1), depth.copy()

    u_b = u[in_bounds]
    v_b = v[in_bounds]
    u_idx_b = u_idx[in_bounds]
    v_idx_b = v_idx[in_bounds]

    dx = flow[v_idx_b, u_idx_b, 0]
    dy = flow[v_idx_b, u_idx_b, 1]
    c = conf[v_idx_b, u_idx_b]

    flow_mag = np.sqrt(dx ** 2 + dy ** 2)

    if use_bounds:
        valid_mask = (c > conf_thresh) & (flow_mag > 0.5) & (flow_mag < 30.0)
    else:
        valid_mask = c > conf_thresh

    original_indices = np.flatnonzero(in_bounds)
    valid_global = original_indices[valid_mask]

    u_new[valid_global] = u_b[valid_mask] + alpha * dx[valid_mask]
    v_new[valid_global] = v_b[valid_mask] + alpha * dy[valid_mask]

    u_new = np.clip(u_new, 0.0, w - 1.0)
    v_new = np.clip(v_new, 0.0, h - 1.0)

    return np.stack([u_new, v_new], axis=1).astype(np.float32), depth.copy()


# =======================================================================
# Core: per-frame correction under one config
# =======================================================================

def apply_correction_config(
    uv_orig: np.ndarray,
    depth_orig: np.ndarray,
    image_t: np.ndarray,
    image_t1: np.ndarray,
    config: dict,
    prev_flow: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Apply one ablation config to a single frame pair.

    Returns
    -------
    (uv_corrected, depth_corrected, flow_for_next_frame)
        flow_for_next_frame is the unsmoothed Farneback flow that the
        caller threads into prev_flow on the next call (so use_smoothing
        means "EMA against the previous frame's raw flow").
    """
    if not config["use_flow"]:
        return uv_orig.copy(), depth_orig.copy(), None

    flow_raw = compute_rgb_flow(image_t, image_t1)
    if config["use_smoothing"] and prev_flow is not None:
        flow = smooth_flow(flow_raw, prev_flow, alpha=0.7)
    else:
        flow = flow_raw

    if config["use_conf_gating"]:
        if config["use_edge_filter"]:
            events = simulate_events(image_t, image_t1, threshold=0.3)
        else:
            events = _simulate_events_no_edge_filter(image_t, image_t1, threshold=0.3)
        conf = event_confidence(events)
    else:
        h, w = flow.shape[:2]
        conf = np.ones((h, w), dtype=np.float32)

    uv_corr, depth_corr = _move_points_ablation(
        uv_orig, depth_orig, flow, conf,
        alpha=float(config["alpha"]),
        use_bounds=bool(config["use_flow_bounds"]),
        conf_thresh=0.5,
    )
    return uv_corr, depth_corr, flow_raw


# =======================================================================
# Sequence-level evaluation
# =======================================================================

def _safe_pct(before: float, after: float) -> float:
    """Higher-is-better percent improvement: (after - before) / before * 100."""
    if not np.isfinite(before) or not np.isfinite(after) or abs(before) < 1e-12:
        return float("nan")
    return float((after - before) / before * 100.0)


def evaluate_config(
    config: dict,
    dataset_path: str,
    p_rect_right: Optional[np.ndarray],
    n_frames: int = DEFAULT_N_FRAMES,
) -> dict:
    """Evaluate one ablation config on n_frames consecutive frame pairs.

    Computes SPEAS and (when stereo geometry is provided and the right
    image is available) Stereo Consistency before and after correction.
    Per-frame failures are caught and skipped.
    """
    image_dir = os.path.join(dataset_path, "image_02", "data")
    right_image_dir = os.path.join(dataset_path, "image_03", "data")
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
    n_pairs = min(n_pairs, n_frames)
    if n_pairs <= 0:
        raise RuntimeError(f"No consecutive frame pairs in {dataset_path}")

    with _silent():
        tr_velo_to_cam = parse_calib_velo_to_cam(velo_calib)
        r_rect, p_rect_left = parse_calib_cam_to_cam(cam_calib, camera_id="02")

    have_stereo = p_rect_right is not None and os.path.isdir(right_image_dir)
    print(f"  Config: {config['name']}  (stereo: {have_stereo})")

    speas_before_list, speas_after_list = [], []
    stereo_before_list, stereo_after_list = [], []
    n_evaluated = 0
    n_failed = 0

    prev_flow: Optional[np.ndarray] = None

    for i in range(n_pairs):
        try:
            img_t_path = os.path.join(image_dir, image_files[i])
            img_t1_path = os.path.join(image_dir, image_files[i + 1])
            lidar_t_path = os.path.join(lidar_dir, lidar_files[i])

            image_t = load_image(img_t_path)
            image_t1 = load_image(img_t1_path)
            lidar_xyz = load_lidar(lidar_t_path)

            with _silent():
                uv_orig, depth_orig, _ = project_lidar_to_image(
                    lidar_xyz, tr_velo_to_cam, r_rect, p_rect_left, image_t.shape
                )
                if uv_orig.shape[0] == 0:
                    raise RuntimeError("no projected points")

                uv_corr, depth_corr, flow_for_next = apply_correction_config(
                    uv_orig, depth_orig, image_t, image_t1, config, prev_flow
                )

                speas_before, _ = sparse_point_eas(uv_orig, depth_orig, image_t1)
                speas_after, _ = sparse_point_eas(uv_corr, depth_corr, image_t1)

                stereo_before = float("nan")
                stereo_after = float("nan")
                if have_stereo:
                    right_path = os.path.join(right_image_dir, image_files[i + 1])
                    if os.path.isfile(right_path):
                        image_right = load_image(right_path)
                        sb, _, _ = stereo_consistency_score(
                            uv_orig, depth_orig, image_right, p_rect_left, p_rect_right
                        )
                        sa, _, _ = stereo_consistency_score(
                            uv_corr, depth_corr, image_right, p_rect_left, p_rect_right
                        )
                        stereo_before = float(sb)
                        stereo_after = float(sa)

            speas_before_list.append(float(speas_before))
            speas_after_list.append(float(speas_after))
            if np.isfinite(stereo_before) and np.isfinite(stereo_after):
                stereo_before_list.append(stereo_before)
                stereo_after_list.append(stereo_after)

            prev_flow = flow_for_next
            n_evaluated += 1

            if (i + 1) % 20 == 0 or (i + 1) == n_pairs:
                speas_pct = _safe_pct(
                    float(np.mean(speas_before_list)) if speas_before_list else float("nan"),
                    float(np.mean(speas_after_list)) if speas_after_list else float("nan"),
                )
                print(f"    [{config['name']}] frame {i + 1}/{n_pairs}  "
                      f"running SPEAS pct={speas_pct:+.2f}%")

        except Exception as exc:
            n_failed += 1
            if n_failed <= 3:
                print(f"    frame {i}: FAILED ({type(exc).__name__}: {exc})")
            continue

    def _mean(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    speas_before_mean = _mean(speas_before_list)
    speas_after_mean = _mean(speas_after_list)
    stereo_before_mean = _mean(stereo_before_list)
    stereo_after_mean = _mean(stereo_after_list)

    return {
        "name": config["name"],
        "config": {k: v for k, v in config.items() if k != "name"},
        "n_frames_evaluated": n_evaluated,
        "n_frames_failed": n_failed,
        "speas_before": speas_before_mean,
        "speas_after": speas_after_mean,
        "speas_improvement_pct": _safe_pct(speas_before_mean, speas_after_mean),
        "stereo_before": stereo_before_mean,
        "stereo_after": stereo_after_mean,
        "stereo_improvement_pct": _safe_pct(stereo_before_mean, stereo_after_mean),
        "stereo_available": bool(stereo_before_list),
    }


def run_ablation(
    dataset_path: str,
    configs: List[dict],
    n_frames: int = DEFAULT_N_FRAMES,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> List[dict]:
    """Run all configs on dataset_path and return per-config result dicts."""
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    cam_calib = os.path.join(dataset_path, "calib_cam_to_cam.txt")
    p_rect_right: Optional[np.ndarray] = None
    try:
        with _silent():
            _, p_rect_right = parse_calib_cam_to_cam(cam_calib, camera_id="03")
    except Exception as exc:
        print(f"[ablation] right-camera calibration unavailable ({exc}); "
              f"stereo metrics will be NaN.")
        p_rect_right = None

    results: List[dict] = []
    for cfg in configs:
        try:
            result = evaluate_config(
                cfg, dataset_path, p_rect_right, n_frames=n_frames
            )
        except Exception as exc:
            print(f"[ablation] config '{cfg['name']}' FAILED: {exc}")
            continue
        results.append(result)

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "ablation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            results, f, indent=2,
            default=lambda o: float(o) if isinstance(o, np.floating) else str(o),
        )
    print(f"\n[ablation] saved JSON: {json_path}")
    return results


# =======================================================================
# Output
# =======================================================================

def print_ablation_table(results: List[dict]) -> None:
    """ASCII comparison table sorted by SPEAS improvement (descending)."""
    if not results:
        print("[ablation] no results to print.")
        return

    sorted_results = sorted(
        results,
        key=lambda r: (
            r.get("speas_improvement_pct", float("-inf"))
            if np.isfinite(r.get("speas_improvement_pct", float("-inf")))
            else float("-inf")
        ),
        reverse=True,
    )

    full_name = CONFIG_FULL["name"]
    non_full = [r for r in sorted_results if r["name"] != full_name]
    best_non_full_name = non_full[0]["name"] if non_full else None

    name_w = max(len(r["name"]) for r in sorted_results)
    name_w = max(name_w, len("Configuration"))
    annot_w = len(" <- BEST ABLATION")

    sep_top = "+" + "-" * (name_w + 2) + "+" + "+".join(["-" * 11] * 4) + "+" + "-" * (annot_w + 2) + "+"
    header = (f"| {'Configuration':<{name_w}} "
              f"| SPEAS Bef.| SPEAS Aft.| SPEAS %   | Stereo %  "
              f"| {'':<{annot_w}} |")
    print(sep_top)
    print(header)
    print(sep_top)
    for r in sorted_results:
        speas_b = r.get("speas_before", float("nan"))
        speas_a = r.get("speas_after", float("nan"))
        speas_p = r.get("speas_improvement_pct", float("nan"))
        stereo_p = r.get("stereo_improvement_pct", float("nan"))

        annot = ""
        if r["name"] == full_name:
            annot = " <- OURS"
        elif r["name"] == best_non_full_name:
            annot = " <- BEST ABLATION"

        sb = f"{speas_b:>9.4f}" if np.isfinite(speas_b) else "      NaN"
        sa = f"{speas_a:>9.4f}" if np.isfinite(speas_a) else "      NaN"
        sp = f"{speas_p:>+8.2f}%" if np.isfinite(speas_p) else "      NaN"
        tp = f"{stereo_p:>+8.2f}%" if np.isfinite(stereo_p) else "      NaN"

        print(f"| {r['name']:<{name_w}} | {sb} | {sa} | {sp} | {tp} | {annot:<{annot_w}} |")
    print(sep_top)


def generate_ablation_latex(
    results: List[dict],
    output_path: str,
) -> str:
    """IEEE-booktabs LaTeX ablation table; also returned as a string."""
    full_name = CONFIG_FULL["name"]
    full_speas_pct = float("nan")
    for r in results:
        if r["name"] == full_name:
            full_speas_pct = r.get("speas_improvement_pct", float("nan"))
            break

    def _fmt(x: float, prec: int = 2, sign: bool = True) -> str:
        if not isinstance(x, (int, float)) or not np.isfinite(x):
            return "--"
        return f"{x:+.{prec}f}" if sign else f"{x:.{prec}f}"

    best_speas_pct = float("-inf")
    best_stereo_pct = float("-inf")
    for r in results:
        if np.isfinite(r.get("speas_improvement_pct", float("nan"))):
            best_speas_pct = max(best_speas_pct, r["speas_improvement_pct"])
        if np.isfinite(r.get("stereo_improvement_pct", float("nan"))):
            best_stereo_pct = max(best_stereo_pct, r["stereo_improvement_pct"])

    baseline_rows, alpha_rows, full_row, other_rows = [], [], [], []
    for r in results:
        if r["name"] in BASELINE_NAMES:
            baseline_rows.append(r)
        elif r["name"] in ALPHA_NAMES:
            alpha_rows.append(r)
        elif r["name"] == full_name:
            full_row.append(r)
        else:
            other_rows.append(r)

    ordered = baseline_rows + other_rows + alpha_rows + full_row

    lines: List[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Ablation study on drive\_0009\_sync (100 frames). "
                 r"Each row removes or modifies one component of the full "
                 r"pipeline. SPEAS $\Delta$\% and Stereo $\Delta$\% are mean "
                 r"improvements over uncorrected projection. Best results in "
                 r"\textbf{bold}.}")
    lines.append(r"  \label{tab:ablation}")
    lines.append(r"  \small")
    lines.append(r"  \begin{tabular}{lrrr}")
    lines.append(r"    \toprule")
    lines.append(r"    Configuration & SPEAS $\Delta$\% & Stereo $\Delta$\% & vs Full (SPEAS) \\")
    lines.append(r"    \midrule")

    def _emit(rows: List[dict]) -> None:
        for r in rows:
            name_tex = r["name"].replace("_", r"\_").replace("%", r"\%")
            speas_pct = r.get("speas_improvement_pct", float("nan"))
            stereo_pct = r.get("stereo_improvement_pct", float("nan"))

            speas_str = _fmt(speas_pct)
            if np.isfinite(speas_pct) and abs(speas_pct - best_speas_pct) < 1e-9:
                speas_str = r"\textbf{" + speas_str + r"}"
            stereo_str = _fmt(stereo_pct)
            if np.isfinite(stereo_pct) and abs(stereo_pct - best_stereo_pct) < 1e-9:
                stereo_str = r"\textbf{" + stereo_str + r"}"

            if r["name"] == full_name:
                vs_full = r"\textbf{0.00}"
                name_tex = r"\textbf{" + name_tex + r"}"
            elif np.isfinite(speas_pct) and np.isfinite(full_speas_pct):
                vs_full = _fmt(speas_pct - full_speas_pct)
            else:
                vs_full = "--"

            lines.append(f"    {name_tex} & {speas_str} & {stereo_str} & {vs_full} \\\\")

    _emit(baseline_rows)
    if baseline_rows and (other_rows or alpha_rows or full_row):
        lines.append(r"    \midrule")
    _emit(other_rows)
    if other_rows and (alpha_rows or full_row):
        lines.append(r"    \midrule")
    _emit(alpha_rows)
    if alpha_rows and full_row:
        lines.append(r"    \midrule")
    _emit(full_row)

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    latex = "\n".join(lines)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex + "\n")
    print(f"[ablation] saved LaTeX: {output_path}")
    return latex


# =======================================================================
# CLI
# =======================================================================

def main() -> None:
    # Windows consoles default to cp1252 and crash on non-ASCII (e.g. ±).
    # Best-effort UTF-8 reconfigure; harmless if already UTF-8 or unsupported.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="Phase 4 ablation study for event-guided LiDAR temporal correction."
    )
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET,
                        help="Full path to one KITTI sequence directory.")
    parser.add_argument("--n-frames", type=int, default=DEFAULT_N_FRAMES,
                        help="Frame pairs per config to evaluate.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Where to save JSON + LaTeX results.")
    parser.add_argument("--right-image-dir", default=None,
                        help="Optional explicit path to image_03/data; "
                             "auto-detected from --dataset-path if omitted.")
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 4 — Ablation Study")
    print("=" * 70)
    print(f"Dataset: {args.dataset_path}")
    print(f"Frames per config: {args.n_frames}")
    print(f"Output dir: {args.output_dir}")
    print(f"Configs: {len(ALL_CONFIGS)}")

    if args.right_image_dir is not None:
        # Override only if the user explicitly passes one — evaluate_config
        # auto-detects from the dataset path otherwise. We surface mismatches
        # loudly rather than silently swap paths.
        expected = os.path.join(args.dataset_path, "image_03", "data")
        if os.path.normpath(args.right_image_dir) != os.path.normpath(expected):
            print(f"[ablation] WARNING: --right-image-dir ({args.right_image_dir}) "
                  f"does not match expected {expected}; using the auto-detected one.")

    results = run_ablation(
        dataset_path=args.dataset_path,
        configs=ALL_CONFIGS,
        n_frames=args.n_frames,
        output_dir=args.output_dir,
    )

    if not results:
        print("[ablation] no results produced — aborting.")
        return

    print()
    print_ablation_table(results)
    latex_path = os.path.join(args.output_dir, "ablation_table.tex")
    print()
    latex = generate_ablation_latex(results, latex_path)
    print("\n--- LaTeX snippet ---")
    print(latex)
    print("--- end LaTeX ---")


if __name__ == "__main__":
    main()
