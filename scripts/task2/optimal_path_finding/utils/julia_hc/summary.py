from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass
class JuliaSummary:
    success_rate: float
    total_steps_mean: float
    total_steps_std: float
    wall_time_mean: float
    wall_time_std: float
    total_rejected_steps_mean: float
    total_rejected_steps_std: float
    return_code_counts_top: list[tuple[str, int]]
    # Optional extras (may be unavailable depending on caller / result dict contents)
    total_accepted_steps_mean: float | None = None
    total_accepted_steps_std: float | None = None
    total_newton_iters_mean: float | None = None
    total_newton_iters_std: float | None = None
    return_code_non_success_total: int | None = None
    return_code_total: int | None = None
    min_n_success_per_segment: int | None = None
    min_n_success_segment: int | None = None


def _aggregate_return_codes_and_min_success(meas: list[dict]) -> tuple[Counter[str], int | None, int | None]:
    """
    Aggregate return codes across all runs/segments/paths, and find the minimum observed
    `n_success` over segments (and which segment it occurred at).
    """

    rc_counter: Counter[str] = Counter()
    min_n_success: int | None = None
    min_n_success_seg: int | None = None
    for r in meas:
        for seg_detail in r.get("detail", []) or []:
            for rc in seg_detail.get("return_codes", []) or []:
                rc_counter[str(rc)] += 1
            ns = seg_detail.get("n_success", None)
            if ns is not None:
                ns_int = int(ns)
                if min_n_success is None or ns_int < min_n_success:
                    min_n_success = ns_int
                    min_n_success_seg = int(seg_detail.get("segment", -1))
    return rc_counter, min_n_success, min_n_success_seg


def _normalize_success_count(rc_counter: Counter[str]) -> tuple[int, int, int]:
    """
    Normalize success key for summary (handles both ':success' and 'success').

    Returns:
        (success_cnt, total_cnt, non_success_cnt)
    """

    total_cnt = int(sum(rc_counter.values())) if rc_counter else 0
    success_cnt = int(rc_counter.get(":success", 0) + rc_counter.get("success", 0)) if rc_counter else 0
    non_success = int(total_cnt - success_cnt) if total_cnt else 0
    return success_cnt, total_cnt, non_success


def summarize_julia_runs(meas: list[dict], *, topk: int = 8) -> JuliaSummary:
    """
    Summarize repeated Julia tracking runs produced by the Julia HC tracker functions.

    Args:
        meas: list of result dicts (each run).
        topk: number of most common return_code values to store.
    """

    ok = np.array([bool(r.get("ok", False)) for r in meas], dtype=np.float64)
    steps = np.array([float(r.get("total_steps", 0)) for r in meas], dtype=np.float64)
    acc = np.array([float(r.get("total_accepted_steps", 0)) for r in meas], dtype=np.float64)
    rej = np.array([float(r.get("total_rejected_steps", 0)) for r in meas], dtype=np.float64)
    times = np.array([float(r.get("wall_time_sec", 0.0)) for r in meas], dtype=np.float64)

    # Optional metric: Newton iteration totals (currently only available for some callers, e.g. Bezier).
    newton_list = [float(r["total_newton_iters"]) for r in meas if "total_newton_iters" in r]
    newton = np.array(newton_list, dtype=np.float64) if newton_list else None

    rc_counter, min_n_success, min_n_success_seg = _aggregate_return_codes_and_min_success(meas)
    top = rc_counter.most_common(int(topk)) if rc_counter else []

    _success_cnt, total_cnt, non_success = _normalize_success_count(rc_counter)

    return JuliaSummary(
        success_rate=float(ok.mean() if ok.size else 0.0),
        total_steps_mean=float(steps.mean() if steps.size else 0.0),
        total_steps_std=float(steps.std(ddof=0) if steps.size else 0.0),
        total_accepted_steps_mean=float(acc.mean() if acc.size else 0.0),
        total_accepted_steps_std=float(acc.std(ddof=0) if acc.size else 0.0),
        total_newton_iters_mean=None if newton is None else float(newton.mean() if newton.size else 0.0),
        total_newton_iters_std=None if newton is None else float(newton.std(ddof=0) if newton.size else 0.0),
        wall_time_mean=float(times.mean() if times.size else 0.0),
        wall_time_std=float(times.std(ddof=0) if times.size else 0.0),
        total_rejected_steps_mean=float(rej.mean() if rej.size else 0.0),
        total_rejected_steps_std=float(rej.std(ddof=0) if rej.size else 0.0),
        return_code_counts_top=[(str(k), int(v)) for (k, v) in top],
        return_code_non_success_total=int(non_success),
        return_code_total=int(total_cnt),
        min_n_success_per_segment=None if min_n_success is None else int(min_n_success),
        min_n_success_segment=None if min_n_success_seg is None else int(min_n_success_seg),
    )


def print_julia_run_report(name: str, meas: list[dict], *, topk: int = 12) -> None:
    """
    Print a detailed report for repeated Julia tracking runs.

    This mirrors the high-signal console summary previously implemented inline in `julia_hc_compare.py`.
    """

    if not meas:
        print(f"{name}: (no measured runs)")
        return

    ok = np.array([bool(r.get("ok", False)) for r in meas], dtype=np.int32)
    steps = np.array([int(r.get("total_steps", 0)) for r in meas], dtype=np.float64)
    acc = np.array([int(r.get("total_accepted_steps", 0)) for r in meas], dtype=np.float64)
    rej = np.array([int(r.get("total_rejected_steps", 0)) for r in meas], dtype=np.float64)
    times = np.array([float(r.get("wall_time_sec", 0.0)) for r in meas], dtype=np.float64)
    newton_list = [float(r["total_newton_iters"]) for r in meas if "total_newton_iters" in r]
    newton = np.array(newton_list, dtype=np.float64) if newton_list else None

    print(f"{name}: success_rate={ok.mean()*100:.1f}% (n={len(meas)})")
    print(
        "  total_steps:          "
        f"mean={steps.mean():.2f}, std={steps.std(ddof=0):.2f}, min={steps.min():.0f}, max={steps.max():.0f}"
    )
    print(
        "  total_accepted_steps: "
        f"mean={acc.mean():.2f}, std={acc.std(ddof=0):.2f}, min={acc.min():.0f}, max={acc.max():.0f}"
    )
    print(
        "  total_rejected_steps: "
        f"mean={rej.mean():.2f}, std={rej.std(ddof=0):.2f}, min={rej.min():.0f}, max={rej.max():.0f}"
    )
    print(
        "  runtime[s]:           "
        f"mean={times.mean():.6f}, std={times.std(ddof=0):.6f}, min={times.min():.6f}, max={times.max():.6f}"
    )
    if newton is not None and newton.size:
        print(
            "  total_newton_iters:   "
            f"mean={newton.mean():.2f}, std={newton.std(ddof=0):.2f}, min={newton.min():.0f}, max={newton.max():.0f}"
        )

    rc_counter, min_n_success, min_n_success_seg = _aggregate_return_codes_and_min_success(meas)

    if rc_counter:
        top = rc_counter.most_common(int(topk))
        print("  return_code counts (top):")
        for code, cnt in top:
            print(f"    {code}: {cnt}")
        _success_cnt, total_cnt, non_success = _normalize_success_count(rc_counter)
        print(f"  return_code non-success total: {non_success} / {total_cnt}")

    if min_n_success is not None:
        print(f"  min n_success per segment observed: {min_n_success} (at segment {min_n_success_seg})")


