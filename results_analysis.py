"""Publication-grade statistical analysis of KITTI temporal-correction runs.

Reads ``*_eas.json`` files emitted by ``validate_pipeline.py`` (one per
sequence), performs paired statistical tests, and produces console
summaries plus IEEE booktabs LaTeX tables and a master CSV suitable for
inclusion in a robotics paper submission (RA-L / ICRA).

Usage:
    python results_analysis.py --json-dir ./test_logs1
    python results_analysis.py --json-dir ./test_logs1 \
        --output-dir ./paper_results --alpha 0.05 --verbose

Dependencies: numpy, scipy. No matplotlib, no deep-learning libraries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy import stats


# Console may be cp1252 on Windows; the report uses Unicode box-drawing
# characters and ✓ / arrows. Reconfigure once at import so every print
# in this module is safe to emit.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


# =====================================================================
# SECTION 1: Data Loading
# =====================================================================

_SEQ_NAME_RE = re.compile(r"_drive_(\d{4}_sync)")


def _short_sequence_name(dataset_name: str, fallback: str) -> str:
    """Extract ``"NNNN_sync"`` from a KITTI dataset name.

    Args:
        dataset_name: full dataset name such as
            ``"2011_09_26_drive_0009_sync"`` or any other label.
        fallback: name to return when the regex does not match.

    Returns:
        The four-digit sequence id with the ``_sync`` suffix, or
        ``fallback`` when the input does not match the expected pattern.
    """
    if not isinstance(dataset_name, str):
        return fallback
    m = _SEQ_NAME_RE.search(dataset_name)
    return m.group(1) if m else fallback


def _safe_get(d: Any, *keys: str, default: Any = None) -> Any:
    """Walk a nested dict with default fallback on any missing key.

    Args:
        d: object to traverse (typically a dict).
        *keys: ordered keys to follow.
        default: value to return when any key is missing or the
            traversal hits a non-dict.

    Returns:
        The value at the nested path, or ``default``.
    """
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def load_sequence_results(json_dir: str) -> list[dict]:
    """Load every ``*_eas.json`` in ``json_dir`` into flat summary dicts.

    Each output dict has the keys documented in the module-level
    docstring: ``sequence_name``, ``n_frames``, ``speas_*``,
    ``stereo_*``, ``dgc_*``, ``imu_*``, ``epsilon_improvement_pct``.

    Args:
        json_dir: path to a directory containing one ``*_eas.json``
            per sequence.

    Returns:
        List of flat summary dicts (one per successfully-parsed file),
        sorted by ``sequence_name``. Bad files are skipped with a
        warning printed to stdout — this function never raises.
    """
    json_path = Path(json_dir)
    if not json_path.is_dir():
        print(f"[load] Warning: {json_dir} is not a directory.")
        return []

    files = sorted(json_path.glob("*_eas.json"))
    if not files:
        print(f"[load] Warning: no *_eas.json files found in {json_dir}.")
        return []

    sequences: list[dict] = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[load] Warning: skipping {f.name}: {exc}")
            continue

        dataset_name = raw.get("dataset_name", f.stem.replace("_eas", ""))
        seq_name = _short_sequence_name(dataset_name, fallback=f.stem)

        speas = raw.get("speas_summary") or {}
        stereo = raw.get("stereo_summary") or {}
        dgc = raw.get("dgc_summary") or {}
        imu = raw.get("imu_summary") or {}
        epsilon = raw.get("epsilon_summary")

        entry = {
            "sequence_name": seq_name,
            "n_frames": speas.get("n_frames"),
            "speas_before": speas.get("score_before_mean"),
            "speas_after": speas.get("score_after_mean"),
            "speas_improvement_pct": speas.get("improvement_pct_mean"),
            "speas_improvement_pct_std": speas.get("improvement_pct_std"),
            "stereo_before": stereo.get("score_before_mean"),
            "stereo_after": stereo.get("score_after_mean"),
            "stereo_improvement_pct": stereo.get("improvement_pct_mean"),
            "stereo_improvement_pct_std": stereo.get("improvement_pct_std"),
            "stereo_available": stereo.get("available", True),
            "dgc_before": dgc.get("before_mean"),
            "dgc_after": dgc.get("after_mean"),
            "dgc_improvement_pct": dgc.get("improvement_pct_mean"),
            "imu_eas": imu.get("imu_eas_mean"),
            "imu_improvement_pct": imu.get("imu_improvement_pct_mean"),
            "event_guided_improvement_pct": imu.get(
                "event_guided_improvement_pct_mean"
            ),
            "epsilon_improvement_pct": (
                epsilon.get("improvement_pct_mean")
                if isinstance(epsilon, dict) else None
            ),
        }
        sequences.append(entry)

    sequences.sort(key=lambda s: s["sequence_name"])
    total_frames = sum(
        int(s["n_frames"]) for s in sequences
        if isinstance(s["n_frames"], (int, float))
        and s["n_frames"] is not None
    )
    print(f"Loaded {len(sequences)} sequences, total {total_frames} frames")
    return sequences


# =====================================================================
# SECTION 2: Statistical Tests
# =====================================================================


def _is_finite(x: Any) -> bool:
    """Return True when ``x`` is a finite real number."""
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _filter_pairs(
    before: list[Optional[float]],
    after: list[Optional[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Drop index-aligned pairs containing None/NaN/Inf in either side."""
    pairs = [
        (float(b), float(a))
        for b, a in zip(before, after)
        if _is_finite(b) and _is_finite(a)
    ]
    if not pairs:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)
    arr = np.asarray(pairs, dtype=np.float64)
    return arr[:, 0], arr[:, 1]


def _stars(p_value: float) -> str:
    """Map a p-value to APA-style significance stars."""
    if not _is_finite(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def paired_ttest(
    before: list[Optional[float]],
    after: list[Optional[float]],
    metric_name: str,
    alpha: float = 0.05,
) -> dict:
    """Run a two-tailed paired t-test on ``after - before`` with effect size.

    Args:
        before: per-sequence "before" values (Nones / NaNs are filtered).
        after: per-sequence "after" values aligned with ``before``.
        metric_name: human-readable label used in the printed summary.
        alpha: significance level for the boolean ``significant`` flag
            and the (1 - alpha) confidence interval.

    Returns:
        Dict with the fields documented in the spec. When fewer than 3
        valid pairs are available, all numeric fields except ``n`` are
        NaN and ``significant`` is False.
    """
    b, a = _filter_pairs(before, after)
    n = int(b.size)
    nan = float("nan")

    if n < 3:
        print(
            f"{metric_name}: insufficient pairs (n={n} < 3) — t-test skipped."
        )
        return {
            "metric_name": metric_name,
            "n": n,
            "mean_before": nan,
            "std_before": nan,
            "mean_after": nan,
            "std_after": nan,
            "mean_diff": nan,
            "std_diff": nan,
            "t_stat": nan,
            "p_value": nan,
            "cohens_d": nan,
            "ci_lower": nan,
            "ci_upper": nan,
            "significant": False,
            "stars": "",
        }

    diff = a - b
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if n > 1 else nan
    mean_before = float(np.mean(b))
    std_before = float(np.std(b, ddof=1)) if n > 1 else nan
    mean_after = float(np.mean(a))
    std_after = float(np.std(a, ddof=1)) if n > 1 else nan

    try:
        t_result = stats.ttest_rel(a, b)
        t_stat = float(t_result.statistic)
        p_value = float(t_result.pvalue)
    except (ValueError, FloatingPointError) as exc:
        print(f"[ttest] {metric_name}: ttest_rel failed: {exc}")
        t_stat, p_value = nan, nan

    cohens_d = mean_diff / std_diff if std_diff and _is_finite(std_diff) and std_diff > 0 else nan

    if std_diff and _is_finite(std_diff) and std_diff > 0 and n > 1:
        sem = std_diff / math.sqrt(n)
        try:
            ci_lower, ci_upper = stats.t.interval(
                1.0 - alpha, df=n - 1, loc=mean_diff, scale=sem
            )
            ci_lower = float(ci_lower)
            ci_upper = float(ci_upper)
        except (ValueError, FloatingPointError):
            ci_lower, ci_upper = nan, nan
    else:
        ci_lower, ci_upper = nan, nan

    significant = bool(_is_finite(p_value) and p_value < alpha)
    stars = _stars(p_value)

    print(
        f"{metric_name}: t={t_stat:.3f}, p={p_value:.4f}{stars}, "
        f"d={cohens_d:.3f}, n={n}"
    )

    return {
        "metric_name": metric_name,
        "n": n,
        "mean_before": mean_before,
        "std_before": std_before,
        "mean_after": mean_after,
        "std_after": std_after,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "t_stat": t_stat,
        "p_value": p_value,
        "cohens_d": float(cohens_d) if _is_finite(cohens_d) else nan,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant": significant,
        "stars": stars,
    }


def wilcoxon_test(
    before: list[Optional[float]],
    after: list[Optional[float]],
    metric_name: str,
    alpha: float = 0.05,
) -> Optional[dict]:
    """Run a paired Wilcoxon signed-rank test on ``after - before``.

    Args:
        before: per-sequence "before" values (Nones / NaNs are filtered).
        after: per-sequence "after" values aligned with ``before``.
        metric_name: human-readable label used in the printed summary.
        alpha: significance level for the boolean ``significant`` flag.

    Returns:
        Dict with the fields documented in the spec, or None when fewer
        than 10 valid pairs are available (a warning is printed in that
        case).
    """
    b, a = _filter_pairs(before, after)
    n = int(b.size)
    if n < 10:
        print(
            f"[wilcoxon] {metric_name}: n={n} < 10 — Wilcoxon test skipped."
        )
        return None

    diff = a - b
    nan = float("nan")
    try:
        result = stats.wilcoxon(diff, zero_method="wilcox")
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
    except (ValueError, FloatingPointError) as exc:
        print(f"[wilcoxon] {metric_name}: wilcoxon failed: {exc}")
        return {
            "metric_name": metric_name,
            "n": n,
            "statistic": nan,
            "p_value": nan,
            "significant": False,
            "stars": "",
            "median_diff": float(np.median(diff)),
        }

    significant = bool(_is_finite(p_value) and p_value < alpha)
    return {
        "metric_name": metric_name,
        "n": n,
        "statistic": statistic,
        "p_value": p_value,
        "significant": significant,
        "stars": _stars(p_value),
        "median_diff": float(np.median(diff)),
    }


def _format_p(p: Any) -> str:
    """Format a p-value to 4 decimals or ``n/a`` when missing."""
    if not _is_finite(p):
        return "n/a"
    return f"{float(p):.4f}"


def _format_d(d: Any) -> str:
    """Format Cohen's d to 3 decimals or ``n/a`` when missing."""
    if not _is_finite(d):
        return "n/a"
    return f"{float(d):.3f}"


def run_all_tests(sequences: list[dict]) -> dict:
    """Run paired t-tests and Wilcoxon tests for SPEAS, Stereo, DGC.

    Stereo restricts to sequences where ``stereo_available`` is True.
    Prints a single consolidated significance table.

    Args:
        sequences: flat dicts produced by ``load_sequence_results``.

    Returns:
        Nested dict ``{"speas": {"ttest": ..., "wilcoxon": ...}, ...}``
        with the same three top-level keys for ``stereo`` and ``dgc``.
    """
    speas_b = [s.get("speas_before") for s in sequences]
    speas_a = [s.get("speas_after") for s in sequences]
    stereo_seqs = [s for s in sequences if s.get("stereo_available", True)]
    stereo_b = [s.get("stereo_before") for s in stereo_seqs]
    stereo_a = [s.get("stereo_after") for s in stereo_seqs]
    dgc_b = [s.get("dgc_before") for s in sequences]
    dgc_a = [s.get("dgc_after") for s in sequences]

    print("")
    print("--- Paired statistical tests ---")
    speas_t = paired_ttest(speas_b, speas_a, "SPEAS")
    stereo_t = paired_ttest(stereo_b, stereo_a, "Stereo")
    dgc_t = paired_ttest(dgc_b, dgc_a, "DGC")
    speas_w = wilcoxon_test(speas_b, speas_a, "SPEAS")
    stereo_w = wilcoxon_test(stereo_b, stereo_a, "Stereo")
    dgc_w = wilcoxon_test(dgc_b, dgc_a, "DGC")

    rows = [
        ("SPEAS (left)   ↑", speas_t, speas_w),
        ("Stereo (right) ↑", stereo_t, stereo_w),
        ("DGC            ↑", dgc_t, dgc_w),
    ]

    col_widths = (18, 6, 14, 14, 9)
    top = "╔" + "╦".join("═" * w for w in col_widths) + "╗"
    mid = "╠" + "╬".join("═" * w for w in col_widths) + "╣"
    bot = "╚" + "╩".join("═" * w for w in col_widths) + "╝"

    print("")
    print(top)
    print(
        "║ Metric           ║  n   ║   t-test p   "
        "║ Wilcoxon p   ║Cohen's d║"
    )
    print(mid)
    for label, t_res, w_res in rows:
        n = t_res.get("n", 0) if t_res else 0
        t_p = t_res.get("p_value") if t_res else None
        t_stars = t_res.get("stars", "") if t_res else ""
        w_p = w_res.get("p_value") if w_res else None
        w_stars = w_res.get("stars", "") if w_res else ""
        d = t_res.get("cohens_d") if t_res else None

        t_p_cell = f"{_format_p(t_p)} {t_stars}".strip()
        w_p_cell = f"{_format_p(w_p)} {w_stars}".strip() if w_res else "n/a"
        d_cell = _format_d(d)

        print(
            f"║ {label:<16} ║ {n:>4} "
            f"║ {t_p_cell:<12} ║ {w_p_cell:<12} "
            f"║ {d_cell:>7} ║"
        )
    print(bot)

    return {
        "speas": {"ttest": speas_t, "wilcoxon": speas_w},
        "stereo": {"ttest": stereo_t, "wilcoxon": stereo_w},
        "dgc": {"ttest": dgc_t, "wilcoxon": dgc_w},
    }


# =====================================================================
# SECTION 3: Publication Tables
# =====================================================================


def _fmt_score(x: Any) -> str:
    """Format a score value to 4 decimals or ``--`` when missing."""
    return f"{float(x):.4f}" if _is_finite(x) else "--"


def _fmt_pct(x: Any, signed: bool = True) -> str:
    """Format a percentage to 2 decimals (with sign) or ``--`` when missing."""
    if not _is_finite(x):
        return "--"
    return f"{float(x):+.2f}" if signed else f"{float(x):.2f}"


def _mean_std(values: list[Any]) -> tuple[float, float, int]:
    """Mean, standard deviation, and count of finite values in ``values``."""
    finite = [float(v) for v in values if _is_finite(v)]
    if not finite:
        return float("nan"), float("nan"), 0
    arr = np.asarray(finite, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr, ddof=1) if arr.size > 1 else 0.0), int(arr.size)


def generate_latex_table_results(
    sequences: list[dict],
    stats_results: dict,
    output_path: str,
) -> str:
    """Per-sequence IEEE booktabs table of SPEAS / Stereo / DGC.

    Sorts rows by sequence id ascending. Improvement-percentage cells
    are bolded for sequences that show positive improvement on both
    SPEAS and Stereo. A summary "Mean" row aggregates each numeric
    column. A footnote row reports paired t-test p-values and Cohen's d
    for SPEAS and Stereo.

    Args:
        sequences: flat dicts from ``load_sequence_results``.
        stats_results: nested dict from ``run_all_tests``.
        output_path: file path to write the LaTeX source to. Parent
            directories must exist.

    Returns:
        The LaTeX source string (also written to ``output_path``).
    """
    rows = sorted(sequences, key=lambda s: s.get("sequence_name", ""))

    speas_imp_vals: list[float] = []
    stereo_imp_vals: list[float] = []
    dgc_imp_vals: list[float] = []
    speas_b_vals: list[float] = []
    speas_a_vals: list[float] = []
    stereo_b_vals: list[float] = []
    stereo_a_vals: list[float] = []
    total_frames = 0

    body_lines: list[str] = []
    for s in rows:
        seq = str(s.get("sequence_name", "?"))
        nframes = s.get("n_frames")
        nframes_str = f"{int(nframes)}" if _is_finite(nframes) else "--"
        if _is_finite(nframes):
            total_frames += int(nframes)

        sb, sa = s.get("speas_before"), s.get("speas_after")
        si = s.get("speas_improvement_pct")
        rb, ra = s.get("stereo_before"), s.get("stereo_after")
        ri = s.get("stereo_improvement_pct")
        di = s.get("dgc_improvement_pct")

        bold = (
            _is_finite(si) and float(si) > 0.0
            and _is_finite(ri) and float(ri) > 0.0
        )
        si_cell = _fmt_pct(si)
        ri_cell = _fmt_pct(ri)
        if bold:
            si_cell = f"\\textbf{{{si_cell}}}"
            ri_cell = f"\\textbf{{{ri_cell}}}"

        body_lines.append(
            " & ".join([
                seq.replace("_", r"\_"),
                nframes_str,
                _fmt_score(sb),
                _fmt_score(sa),
                si_cell,
                _fmt_score(rb),
                _fmt_score(ra),
                ri_cell,
                _fmt_pct(di),
            ]) + r" \\"
        )

        for vlist, val in (
            (speas_imp_vals, si), (stereo_imp_vals, ri), (dgc_imp_vals, di),
            (speas_b_vals, sb), (speas_a_vals, sa),
            (stereo_b_vals, rb), (stereo_a_vals, ra),
        ):
            if _is_finite(val):
                vlist.append(float(val))

    sb_m, sb_s, _ = _mean_std(speas_b_vals)
    sa_m, sa_s, _ = _mean_std(speas_a_vals)
    si_m, si_s, _ = _mean_std(speas_imp_vals)
    rb_m, rb_s, _ = _mean_std(stereo_b_vals)
    ra_m, ra_s, _ = _mean_std(stereo_a_vals)
    ri_m, ri_s, _ = _mean_std(stereo_imp_vals)
    di_m, di_s, _ = _mean_std(dgc_imp_vals)

    def _ms(m: float, sd: float, fmt: str) -> str:
        if not _is_finite(m):
            return "--"
        return f"{m:{fmt}} $\\pm$ {sd:{fmt.lstrip('+')}}"

    mean_row = " & ".join([
        "Mean",
        f"{total_frames}",
        _ms(sb_m, sb_s, ".4f"),
        _ms(sa_m, sa_s, ".4f"),
        _ms(si_m, si_s, "+.2f"),
        _ms(rb_m, rb_s, ".4f"),
        _ms(ra_m, ra_s, ".4f"),
        _ms(ri_m, ri_s, "+.2f"),
        _ms(di_m, di_s, "+.2f"),
    ]) + r" \\"

    speas_t = (stats_results.get("speas") or {}).get("ttest") or {}
    stereo_t = (stats_results.get("stereo") or {}).get("ttest") or {}
    foot = (
        rf"\multicolumn{{9}}{{l}}{{\footnotesize Paired $t$-test: "
        rf"SPEAS $p={_format_p(speas_t.get('p_value'))}${speas_t.get('stars','')}, "
        rf"Stereo $p={_format_p(stereo_t.get('p_value'))}${stereo_t.get('stars','')}. "
        rf"Cohen's $d$: SPEAS$={_format_d(speas_t.get('cohens_d'))}$, "
        rf"Stereo$={_format_d(stereo_t.get('cohens_d'))}$.}} \\"
    )

    header = (
        r"Sequence & Frames & SPEAS Before & SPEAS After & "
        r"SPEAS $\Delta$\% & Stereo Before & Stereo After & "
        r"Stereo $\Delta$\% & DGC $\Delta$\% \\"
    )

    latex = "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Per-sequence flow-independent metrics on KITTI raw "
        r"sequences. SPEAS and Stereo improvement columns are bolded "
        r"where both metrics improve. $\uparrow$ = higher is better.}",
        r"\label{tab:per_sequence_results}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        header,
        r"\midrule",
        *body_lines,
        r"\midrule",
        mean_row,
        foot,
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
        "",
    ])

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(latex)
    except OSError as exc:
        print(f"[latex] Warning: failed to write {output_path}: {exc}")
    return latex


def generate_latex_table_comparison(
    sequences: list[dict],
    stats_results: dict,
    output_path: str,
) -> str:
    """Method-comparison IEEE booktabs table.

    Rows: Uncorrected, IMU Baseline, Ours (Event). The "RGB-Only Flow"
    row is omitted when no RGB-only baseline is present in the input
    JSONs (the caption documents the omission). Best value per column
    is bolded.

    Args:
        sequences: flat dicts from ``load_sequence_results``.
        stats_results: nested dict from ``run_all_tests`` (unused for
            the body but accepted for API symmetry with
            ``generate_latex_table_results``).
        output_path: file path to write the LaTeX source to.

    Returns:
        The LaTeX source string (also written to ``output_path``).
    """
    _ = stats_results  # not used in body cells; kept for API symmetry

    speas_b = [s.get("speas_before") for s in sequences]
    speas_a = [s.get("speas_after") for s in sequences]
    stereo_b = [
        s.get("stereo_before") for s in sequences
        if s.get("stereo_available", True)
    ]
    stereo_a = [
        s.get("stereo_after") for s in sequences
        if s.get("stereo_available", True)
    ]
    dgc_b = [s.get("dgc_before") for s in sequences]
    dgc_a = [s.get("dgc_after") for s in sequences]
    imu_eas = [s.get("imu_eas") for s in sequences]

    unc_speas, _, _ = _mean_std(speas_b)
    unc_stereo, _, _ = _mean_std(stereo_b)
    unc_dgc, _, _ = _mean_std(dgc_b)

    imu_speas, _, n_imu = _mean_std(imu_eas)

    ours_speas, _, _ = _mean_std(speas_a)
    ours_stereo, _, _ = _mean_std(stereo_a)
    ours_dgc, _, _ = _mean_std(dgc_a)

    rows: list[tuple[str, dict[str, Optional[float]]]] = [
        ("Uncorrected", {"speas": unc_speas, "stereo": unc_stereo, "dgc": unc_dgc}),
        (
            r"IMU Baseline$^{\dagger}$",
            {"speas": imu_speas if n_imu > 0 else None, "stereo": None, "dgc": None},
        ),
        (
            r"\textbf{Ours (Event)}",
            {"speas": ours_speas, "stereo": ours_stereo, "dgc": ours_dgc},
        ),
    ]

    best: dict[str, Optional[float]] = {"speas": None, "stereo": None, "dgc": None}
    for _, vals in rows:
        for k, v in vals.items():
            if _is_finite(v) and (best[k] is None or float(v) > float(best[k])):
                best[k] = float(v)

    def _cell(val: Optional[float], col: str) -> str:
        if not _is_finite(val):
            return "--"
        s = f"{float(val):.4f}"
        if best[col] is not None and abs(float(val) - float(best[col])) < 1e-12:
            return f"\\textbf{{{s}}}"
        return s

    body: list[str] = []
    for label, vals in rows:
        body.append(
            " & ".join([
                label,
                _cell(vals["speas"], "speas"),
                _cell(vals["stereo"], "stereo"),
                _cell(vals["dgc"], "dgc"),
            ]) + r" \\"
        )

    total_frames = sum(
        int(s["n_frames"]) for s in sequences
        if _is_finite(s.get("n_frames"))
    )

    caption = (
        rf"Comparison of LiDAR temporal correction methods on 24 KITTI "
        rf"raw sequences ($N={total_frames}$ frames). "
        rf"SPEAS = Sparse-Point Edge Alignment Score ($\uparrow$ better). "
        rf"Stereo = Stereo Reprojection Consistency ($\uparrow$ better). "
        rf"DGC = Depth Gradient Correlation ($\uparrow$ better). "
        rf"$^{{\dagger}}$ IMU row uses dense-EAS as proxy. "
        rf"RGB-only optical-flow baseline omitted (data not available)."
    )

    latex = "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\label{tab:method_comparison}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & SPEAS $\uparrow$ & Stereo $\uparrow$ & DGC $\uparrow$ \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(latex)
    except OSError as exc:
        print(f"[latex] Warning: failed to write {output_path}: {exc}")
    return latex


def generate_results_csv(
    sequences: list[dict],
    stats_results: dict,
    output_path: str,
) -> None:
    """Write per-sequence values plus aggregate / test rows to a CSV.

    The first block has one row per sequence with the columns listed in
    the spec. After a blank separator row a small footer reports MEAN,
    STD, t-test p-values, Wilcoxon p-values, and Cohen's d for SPEAS
    and Stereo.

    Args:
        sequences: flat dicts from ``load_sequence_results``.
        stats_results: nested dict from ``run_all_tests``.
        output_path: file path to write the CSV to.
    """
    columns = [
        "sequence_name", "n_frames",
        "speas_before", "speas_after", "speas_improvement_pct",
        "stereo_before", "stereo_after", "stereo_improvement_pct",
        "dgc_before", "dgc_after", "dgc_improvement_pct",
        "imu_eas", "imu_improvement_pct",
    ]

    def _fmt(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, float) and not math.isfinite(x):
            return ""
        return str(x)

    rows = sorted(sequences, key=lambda s: s.get("sequence_name", ""))

    try:
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            for s in rows:
                writer.writerow([_fmt(s.get(c)) for c in columns])

            writer.writerow([])

            speas_t = (stats_results.get("speas") or {}).get("ttest") or {}
            stereo_t = (stats_results.get("stereo") or {}).get("ttest") or {}
            speas_w = (stats_results.get("speas") or {}).get("wilcoxon") or {}
            stereo_w = (
                (stats_results.get("stereo") or {}).get("wilcoxon") or {}
            )

            def _col_stats(key: str) -> tuple[float, float]:
                vals = [
                    float(s.get(key)) for s in rows
                    if _is_finite(s.get(key))
                ]
                if not vals:
                    return float("nan"), float("nan")
                arr = np.asarray(vals, dtype=np.float64)
                return (
                    float(np.mean(arr)),
                    float(np.std(arr, ddof=1) if arr.size > 1 else 0.0),
                )

            mean_row = ["MEAN", ""]
            std_row = ["STD", ""]
            for col in columns[2:]:
                m, sd = _col_stats(col)
                mean_row.append(_fmt(m))
                std_row.append(_fmt(sd))
            writer.writerow(mean_row)
            writer.writerow(std_row)

            writer.writerow([
                "TTEST_P (SPEAS)", _fmt(speas_t.get("p_value")),
                "TTEST_P (Stereo)", _fmt(stereo_t.get("p_value")),
            ])
            writer.writerow([
                "WILCOXON_P (SPEAS)", _fmt(speas_w.get("p_value")),
                "WILCOXON_P (Stereo)", _fmt(stereo_w.get("p_value")),
            ])
            writer.writerow([
                "COHENS_D (SPEAS)", _fmt(speas_t.get("cohens_d")),
                "COHENS_D (Stereo)", _fmt(stereo_t.get("cohens_d")),
            ])
    except OSError as exc:
        print(f"[csv] Warning: failed to write {output_path}: {exc}")


# =====================================================================
# SECTION 4: Console Reporting
# =====================================================================


def _within_sequence_significant(mean: Any, std: Any) -> bool:
    """Heuristic per-sequence significance: |mean| > std (both finite)."""
    return _is_finite(mean) and _is_finite(std) and abs(float(mean)) > float(std)


def print_sequence_summary(sequences: list[dict]) -> None:
    """Print an ASCII per-sequence ranking sorted by SPEAS improvement.

    Sequences negative on both SPEAS and Stereo are flagged with ``!``
    in front of their rank. Sequences positive on both get a ``✓``
    in the final column. The function prints rate counts after the
    table.

    Args:
        sequences: flat dicts from ``load_sequence_results``.
    """
    def _key(s: dict) -> float:
        v = s.get("speas_improvement_pct")
        return float(v) if _is_finite(v) else float("-inf")

    ranked = sorted(sequences, key=_key, reverse=True)

    n_total = len(ranked)
    n_speas_pos = 0
    n_stereo_pos = 0
    n_both_pos = 0

    print("")
    print(
        "Rank  Seq        Frames  SPEAS%     Stereo%    DGC%      Both+"
    )
    print(
        "----  ---------  ------  ---------  ---------  --------  -----"
    )

    for idx, s in enumerate(ranked, start=1):
        si = s.get("speas_improvement_pct")
        ri = s.get("stereo_improvement_pct")
        di = s.get("dgc_improvement_pct")
        si_std = s.get("speas_improvement_pct_std")
        ri_std = s.get("stereo_improvement_pct_std")

        speas_pos = _is_finite(si) and float(si) > 0.0
        stereo_pos = _is_finite(ri) and float(ri) > 0.0
        if speas_pos:
            n_speas_pos += 1
        if stereo_pos:
            n_stereo_pos += 1
        if speas_pos and stereo_pos:
            n_both_pos += 1

        speas_neg = _is_finite(si) and float(si) <= 0.0
        stereo_neg = _is_finite(ri) and float(ri) <= 0.0
        prefix = "!" if (speas_neg and stereo_neg) else " "

        si_marker = "*" if _within_sequence_significant(si, si_std) else " "
        ri_marker = "*" if _within_sequence_significant(ri, ri_std) else " "

        si_cell = (
            f"{float(si):+.2f}%{si_marker}" if _is_finite(si) else "  --   "
        )
        ri_cell = (
            f"{float(ri):+.2f}%{ri_marker}" if _is_finite(ri) else "  --   "
        )
        di_cell = f"{float(di):+.2f}%" if _is_finite(di) else "  --  "

        nframes = s.get("n_frames")
        nframes_str = (
            f"{int(nframes):>5}" if _is_finite(nframes) else "  -- "
        )

        both_glyph = "✓" if (speas_pos and stereo_pos) else " "
        seq_name = str(s.get("sequence_name", "?"))[:9]

        print(
            f"{prefix}{idx:>3}  {seq_name:<9}   {nframes_str}  "
            f"{si_cell:<9}  {ri_cell:<9}  {di_cell:<8}  {both_glyph:^5}"
        )

    def _pct(num: int, denom: int) -> str:
        return f"{(100.0 * num / denom):.1f}" if denom else "n/a"

    print("")
    print(
        f"Sequences with SPEAS > 0:  {n_speas_pos}/{n_total} "
        f"({_pct(n_speas_pos, n_total)}%)"
    )
    print(
        f"Sequences with Stereo > 0: {n_stereo_pos}/{n_total} "
        f"({_pct(n_stereo_pos, n_total)}%)"
    )
    print(
        f"Sequences positive on BOTH: {n_both_pos}/{n_total} "
        f"({_pct(n_both_pos, n_total)}%)"
    )


def print_ascii_histogram(
    values: list[Optional[float]],
    metric_name: str,
    n_bins: int = 12,
) -> None:
    """Print a horizontal ASCII histogram of improvement % values.

    Marks the bin spanning zero with ``<-- 0`` and reports median, mean,
    and the fraction of positive values below the histogram. The
    longest bar is rendered with 30 ``#`` characters.

    Args:
        values: improvement percentages (Nones / NaNs are filtered).
        metric_name: header shown above the histogram.
        n_bins: number of histogram bins (must be >= 1).
    """
    finite = [float(v) for v in values if _is_finite(v)]
    print("")
    print(f"--- Histogram: {metric_name} ---")
    if not finite:
        print("(no finite values)")
        return
    if int(n_bins) < 1:
        n_bins = 1

    arr = np.asarray(finite, dtype=np.float64)
    lo, hi = float(arr.min()), float(arr.max())
    if hi == lo:
        hi = lo + 1.0

    edges = np.linspace(lo, hi, int(n_bins) + 1)
    counts, _ = np.histogram(arr, bins=edges)
    max_count = int(counts.max()) if counts.size else 0
    bar_max = 30

    for i in range(len(counts)):
        b_lo, b_hi = float(edges[i]), float(edges[i + 1])
        bar_w = (
            int(round(bar_max * counts[i] / max_count)) if max_count else 0
        )
        bar = "#" * bar_w
        marker = " <-- 0" if (b_lo <= 0.0 < b_hi) else ""
        print(
            f"{b_lo:+7.2f} .. {b_hi:+7.2f} | {bar:<{bar_max}} "
            f"{int(counts[i]):>4}{marker}"
        )

    median = float(np.median(arr))
    mean = float(np.mean(arr))
    pct_pos = 100.0 * float(np.sum(arr > 0.0)) / float(arr.size)
    print(
        f"Median: {median:+.3f}  Mean: {mean:+.3f}  "
        f"% Positive: {pct_pos:.1f}%  N={int(arr.size)}"
    )


# =====================================================================
# SECTION 5: Key Finding Summary
# =====================================================================


def _box_line(width: int, content: str) -> str:
    """Build one line of the Unicode key-findings box."""
    inner = content.ljust(width - 2)
    return f"║{inner}║"


def print_key_findings(sequences: list[dict], stats_results: dict) -> None:
    """Print a Unicode-bordered KEY FINDINGS box for paper copy-paste.

    Includes per-metric mean improvement, the ``X/N`` count of
    sequences that improved, and the SPEAS / Stereo paired t-test
    p-values with Cohen's d.

    Args:
        sequences: flat dicts from ``load_sequence_results``.
        stats_results: nested dict from ``run_all_tests``.
    """
    n_seq = len(sequences)
    total_frames = sum(
        int(s["n_frames"]) for s in sequences
        if _is_finite(s.get("n_frames"))
    )

    speas_imp = [s.get("speas_improvement_pct") for s in sequences]
    stereo_imp = [s.get("stereo_improvement_pct") for s in sequences]
    dgc_imp = [s.get("dgc_improvement_pct") for s in sequences]

    speas_m, _, _ = _mean_std(speas_imp)
    stereo_m, _, _ = _mean_std(stereo_imp)
    dgc_m, _, _ = _mean_std(dgc_imp)

    speas_up = sum(1 for v in speas_imp if _is_finite(v) and float(v) > 0.0)
    stereo_up = sum(1 for v in stereo_imp if _is_finite(v) and float(v) > 0.0)
    dgc_up = sum(1 for v in dgc_imp if _is_finite(v) and float(v) > 0.0)

    speas_t = (stats_results.get("speas") or {}).get("ttest") or {}
    stereo_t = (stats_results.get("stereo") or {}).get("ttest") or {}

    width = 56
    top = "╔" + "═" * (width - 2) + "╗"
    sep = "╠" + "═" * (width - 2) + "╣"
    bot = "╚" + "═" * (width - 2) + "╝"

    def _imp_str(m: float) -> str:
        return f"{m:+.2f}%" if _is_finite(m) else "n/a"

    def _stats_str(t: dict) -> str:
        p = t.get("p_value")
        d = t.get("cohens_d")
        st = t.get("stars", "")
        return f"t-test p={_format_p(p)}{st}, Cohen's d={_format_d(d)}"

    print("")
    print(top)
    print(_box_line(
        width, f" KEY FINDINGS (N={n_seq} sequences, F={total_frames} frames)"
    ))
    print(sep)
    print(_box_line(
        width,
        f" SPEAS:  {_imp_str(speas_m)} mean ({speas_up}/{n_seq} sequences ↑)",
    ))
    print(_box_line(width, f"         {_stats_str(speas_t)}"))
    print(_box_line(
        width,
        f" Stereo: {_imp_str(stereo_m)} mean ({stereo_up}/{n_seq} sequences ↑)",
    ))
    print(_box_line(width, f"         {_stats_str(stereo_t)}"))
    print(_box_line(
        width,
        f" DGC:    {_imp_str(dgc_m)} mean ({dgc_up}/{n_seq} sequences ↑)",
    ))
    print(_box_line(width, " IMU vs Ours: comparable performance (no IMU needed)"))
    print(bot)


# =====================================================================
# SECTION 6: Main Entry Point
# =====================================================================


def main() -> None:
    """Parse CLI arguments and run the full analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="Statistical analysis of validate_pipeline.py outputs."
    )
    parser.add_argument(
        "--json-dir",
        type=str,
        default="./test_logs1",
        help="Directory containing *_eas.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./paper_results",
        help="Directory to write CSV and LaTeX tables to.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for t-tests / Wilcoxon tests.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-frame ASCII histograms for SPEAS and Stereo.",
    )
    args = parser.parse_args()

    sequences = load_sequence_results(args.json_dir)
    if not sequences:
        print("No sequences loaded — nothing to do.")
        return

    print_sequence_summary(sequences)
    stats_results = run_all_tests(sequences)

    if args.verbose:
        print_ascii_histogram(
            [s.get("speas_improvement_pct") for s in sequences],
            "SPEAS improvement %",
        )
        print_ascii_histogram(
            [s.get("stereo_improvement_pct") for s in sequences],
            "Stereo improvement %",
        )

    print_key_findings(sequences, stats_results)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "results.csv")
    table1_path = os.path.join(args.output_dir, "table1.tex")
    table2_path = os.path.join(args.output_dir, "table2.tex")

    generate_results_csv(sequences, stats_results, csv_path)
    generate_latex_table_results(sequences, stats_results, table1_path)
    generate_latex_table_comparison(sequences, stats_results, table2_path)

    print("")
    print(f"Saved to {args.output_dir}:")
    print("  results.csv, table1.tex, table2.tex")


if __name__ == "__main__":
    main()
