"""Numerical and performance comparison helpers for Phase 7."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class ComparisonSummary:
    """Summary of baseline-vs-optimized equivalence checks."""

    compared_items: int
    differing_items: int
    path_success_differences: int = 0
    tracking_step_differences: int = 0
    failure_code_differences: int = 0
    max_endpoint_abs_difference: float = 0.0
    max_residual_abs_difference: float = 0.0
    nonfinite_values: int = 0

    def to_dict(self) -> dict[str, int | float]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)


def _row_key(row: dict[str, Any]) -> tuple[str, int, int, str]:
    return (
        str(row.get("method", "")),
        int(row.get("target_seed", 0)),
        int(row.get("instance_id", 0)),
        str(row.get("experiment_id", "")),
    )


def compare_evaluation_rows(
    baseline_rows: Iterable[dict[str, Any]],
    optimized_rows: Iterable[dict[str, Any]],
    *,
    ignore_fields: tuple[str, ...] = ("elapsed_seconds",),
) -> ComparisonSummary:
    """Compare evaluation CSV rows while ignoring wall-clock timing fields."""

    baseline = list(baseline_rows)
    optimized = list(optimized_rows)
    if len(baseline) != len(optimized):
        return ComparisonSummary(
            compared_items=min(len(baseline), len(optimized)),
            differing_items=abs(len(baseline) - len(optimized)),
        )
    ignored = set(ignore_fields)
    differing = 0
    for left, right in zip(sorted(baseline, key=_row_key), sorted(optimized, key=_row_key)):
        keys = (set(left) | set(right)) - ignored
        if any(str(left.get(key, "")) != str(right.get(key, "")) for key in keys):
            differing += 1
    return ComparisonSummary(compared_items=len(baseline), differing_items=differing)


def _complex_matrix(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] != 2:
        raise ValueError("complex arrays must be encoded with trailing real/imag dimension.")
    return array[..., 0] + 1j * array[..., 1]


def compare_tracking_records(
    baseline_records: list[dict[str, Any]],
    optimized_records: list[dict[str, Any]],
    *,
    endpoint_atol: float = 1e-10,
    residual_atol: float = 1e-10,
) -> ComparisonSummary:
    """Compare fixed-seed detailed tracking records."""

    if len(baseline_records) != len(optimized_records):
        return ComparisonSummary(
            compared_items=min(len(baseline_records), len(optimized_records)),
            differing_items=abs(len(baseline_records) - len(optimized_records)),
        )

    differing = 0
    path_success_diffs = 0
    step_diffs = 0
    failure_diffs = 0
    max_endpoint = 0.0
    max_residual = 0.0
    nonfinite = 0
    for left, right in zip(baseline_records, optimized_records):
        left_success = list(left["path_success"])
        right_success = list(right["path_success"])
        if left_success != right_success:
            path_success_diffs += 1
            differing += 1
        for key in (
            "success",
            "n_paths",
            "n_success",
            "n_failed",
            "accepted_steps",
            "rejected_steps",
            "per_path_accepted_steps",
            "per_path_rejected_steps",
        ):
            if left[key] != right[key]:
                step_diffs += 1
                differing += 1
                break
        if list(left["failure_codes"]) != list(right["failure_codes"]):
            failure_diffs += 1
            differing += 1

        left_endpoints = _complex_matrix(left["endpoints"])
        right_endpoints = _complex_matrix(right["endpoints"])
        endpoint_delta = np.abs(left_endpoints - right_endpoints)
        if endpoint_delta.size:
            max_endpoint = max(max_endpoint, float(np.max(endpoint_delta)))
        residual_delta = np.abs(
            np.asarray(left["residual_norms"], dtype=np.float64)
            - np.asarray(right["residual_norms"], dtype=np.float64)
        )
        if residual_delta.size:
            max_residual = max(max_residual, float(np.max(residual_delta)))
        if not np.all(np.isfinite(endpoint_delta)) or not np.all(np.isfinite(residual_delta)):
            nonfinite += 1
            differing += 1
        if np.any(endpoint_delta > endpoint_atol) or np.any(residual_delta > residual_atol):
            differing += 1

    return ComparisonSummary(
        compared_items=len(baseline_records),
        differing_items=differing,
        path_success_differences=path_success_diffs,
        tracking_step_differences=step_diffs,
        failure_code_differences=failure_diffs,
        max_endpoint_abs_difference=max_endpoint,
        max_residual_abs_difference=max_residual,
        nonfinite_values=nonfinite,
    )


def compare_performance_stats(
    baseline: dict[str, Any],
    optimized: dict[str, Any],
    *,
    field: str = "median_seconds",
) -> dict[str, dict[str, float]]:
    """Return speedup summaries for matching timing statistic dictionaries."""

    names = sorted(set(baseline) & set(optimized))
    output: dict[str, dict[str, float]] = {}
    for name in names:
        base = float(baseline[name][field])
        opt = float(optimized[name][field])
        speedup = base / opt if opt > 0.0 else 0.0
        reduction = (base - opt) / base if base > 0.0 else 0.0
        output[name] = {
            "baseline_seconds": base,
            "optimized_seconds": opt,
            "speedup": float(speedup),
            "time_reduction_fraction": float(reduction),
        }
    return output


__all__ = [
    "ComparisonSummary",
    "compare_evaluation_rows",
    "compare_performance_stats",
    "compare_tracking_records",
]
