from __future__ import annotations

from homotopy_path_learning.performance.comparison import (
    compare_evaluation_rows,
    compare_performance_stats,
    compare_tracking_records,
)


def test_evaluation_row_comparison_ignores_elapsed_and_preserves_order() -> None:
    baseline = [
        {"method": "Linear", "target_seed": 1, "instance_id": 0, "mean_cost": 4, "elapsed_seconds": 1.0},
        {
            "method": "LearnedBezier",
            "target_seed": 1,
            "instance_id": 0,
            "mean_cost": 3,
            "elapsed_seconds": 2.0,
        },
    ]
    optimized = list(reversed([dict(row, elapsed_seconds=9.0) for row in baseline]))

    summary = compare_evaluation_rows(baseline, optimized)

    assert summary.compared_items == 2
    assert summary.differing_items == 0


def test_evaluation_row_comparison_reports_differences() -> None:
    summary = compare_evaluation_rows(
        [{"method": "Linear", "target_seed": 1, "instance_id": 0, "mean_cost": 4}],
        [{"method": "Linear", "target_seed": 1, "instance_id": 0, "mean_cost": 5}],
    )

    assert summary.differing_items == 1


def test_performance_stats_comparison_uses_baseline_over_optimized() -> None:
    result = compare_performance_stats(
        {"track": {"median_seconds": 2.0}},
        {"track": {"median_seconds": 1.0}},
    )

    assert result["track"]["speedup"] == 2.0
    assert result["track"]["time_reduction_fraction"] == 0.5


def test_tracking_record_comparison_reports_numerical_equivalence() -> None:
    record = {
        "success": True,
        "n_paths": 1,
        "n_success": 1,
        "n_failed": 0,
        "accepted_steps": 2,
        "rejected_steps": 0,
        "per_path_accepted_steps": [2],
        "per_path_rejected_steps": [0],
        "path_success": [True],
        "endpoints": [[[1.0, 0.0], [0.0, 1.0]]],
        "residual_norms": [1e-12],
        "failure_codes": [""],
    }

    summary = compare_tracking_records([record], [dict(record)])

    assert summary.compared_items == 1
    assert summary.differing_items == 0
    assert summary.max_endpoint_abs_difference == 0.0
    assert summary.max_residual_abs_difference == 0.0
