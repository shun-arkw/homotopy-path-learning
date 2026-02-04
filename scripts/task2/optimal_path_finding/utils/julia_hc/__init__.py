from __future__ import annotations

"""
Julia (juliacall + HomotopyContinuation.jl) helpers for task2/optimal_path_finding.

Important:
  - Do NOT import torch in this package.
  - Call `_maybe_import_julia(...)` before importing torch in entry scripts, to avoid
    potential segfaults (mirrors the existing pattern in task1/task2 scripts).
"""

from .importing import _maybe_import_julia
from .tracker import JuliaTrackerOptions, track_bezier_curve_julia, track_piecewise_linear_julia, track_total_degree_julia
from .summary import JuliaSummary, summarize_julia_runs
from .adaptive import adaptive_julia_tracker_opts_per_segment
from .parsing import parse_int_list
from .runner import benchmark_bezier_curve_julia, benchmark_piecewise_linear_julia, benchmark_total_degree_julia

__all__ = [
    "_maybe_import_julia",
    "JuliaTrackerOptions",
    "track_piecewise_linear_julia",
    "track_bezier_curve_julia",
    "track_total_degree_julia",
    "JuliaSummary",
    "summarize_julia_runs",
    "parse_int_list",
    "adaptive_julia_tracker_opts_per_segment",
    "benchmark_piecewise_linear_julia",
    "benchmark_bezier_curve_julia",
    "benchmark_total_degree_julia",
]


