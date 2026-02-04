from __future__ import annotations

import numpy as np

from .tracker import (
    JuliaTrackerOptions,
    track_bezier_curve_julia,
    track_piecewise_linear_julia,
    track_total_degree_julia,
)


def _benchmark(
    jl,
    *,
    warmup_runs: int,
    runs: int,
    warmup_call,
    measured_call,
) -> list[dict]:
    """Benchmark helper: run warmups (not measured) then measured runs, returning measured results only."""

    if jl is None:
        return []
    for _ in range(int(warmup_runs)):
        warmup_call()
    meas: list[dict] = []
    for _ in range(int(runs)):
        meas.append(measured_call())
    return meas


def benchmark_piecewise_linear_julia(
    jl,
    P_np: np.ndarray,
    *,
    warmup_runs: int,
    runs: int,
    warmup_tracker_opts: JuliaTrackerOptions,
    measured_tracker_opts: JuliaTrackerOptions,
    warmup_tracker_opts_per_segment: list[JuliaTrackerOptions] | None = None,
    measured_tracker_opts_per_segment: list[JuliaTrackerOptions] | None = None,
    start_solutions: list[list[complex]] | None = None,
) -> list[dict]:
    """Benchmark Julia tracking for a piecewise-linear coefficient path."""
    return _benchmark(
        jl,
        warmup_runs=warmup_runs,
        runs=runs,
        warmup_call=lambda: track_piecewise_linear_julia(
            jl,
            P_np,
            tracker_opts=warmup_tracker_opts,
            tracker_opts_per_segment=warmup_tracker_opts_per_segment,
            start_solutions=start_solutions,
        ),
        measured_call=lambda: track_piecewise_linear_julia(
            jl,
            P_np,
            tracker_opts=measured_tracker_opts,
            tracker_opts_per_segment=measured_tracker_opts_per_segment,
            start_solutions=start_solutions,
        ),
    )


def benchmark_bezier_curve_julia(
    jl,
    P_ctrl_np: np.ndarray,
    *,
    warmup_runs: int,
    runs: int,
    warmup_tracker_opts: JuliaTrackerOptions,
    measured_tracker_opts: JuliaTrackerOptions,
    start_solutions: list[list[complex]] | None = None,
) -> list[dict]:
    """Benchmark Julia tracking for a Bezier coefficient curve (single solve per run)."""
    return _benchmark(
        jl,
        warmup_runs=warmup_runs,
        runs=runs,
        warmup_call=lambda: track_bezier_curve_julia(
            jl, P_ctrl_np, tracker_opts=warmup_tracker_opts, start_solutions=start_solutions
        ),
        measured_call=lambda: track_bezier_curve_julia(
            jl, P_ctrl_np, tracker_opts=measured_tracker_opts, start_solutions=start_solutions
        ),
    )


def benchmark_total_degree_julia(
    jl,
    p_target_ri_np: np.ndarray,
    *,
    warmup_runs: int,
    runs: int,
    warmup_tracker_opts: JuliaTrackerOptions,
    measured_tracker_opts: JuliaTrackerOptions,
) -> list[dict]:
    """Benchmark Julia total-degree baseline solve for a target polynomial."""
    return _benchmark(
        jl,
        warmup_runs=warmup_runs,
        runs=runs,
        warmup_call=lambda: track_total_degree_julia(jl, p_target_ri_np, tracker_opts=warmup_tracker_opts),
        measured_call=lambda: track_total_degree_julia(jl, p_target_ri_np, tracker_opts=measured_tracker_opts),
    )


