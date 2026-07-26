"""Evaluation CSV and aggregate metrics for Phase 6."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


EVALUATION_FIELDS = [
    "experiment_id",
    "run_id",
    "instance_id",
    "target_seed",
    "method",
    "problem_success",
    "n_paths",
    "n_success",
    "n_failed",
    "mean_cost",
    "accepted_steps",
    "rejected_steps",
    "cost_improvement_vs_linear",
    "reward_equivalent",
    "max_residual_norm",
    "failure_codes",
    "action_l2_norm",
    "elapsed_seconds",
]


SUMMARY_FIELDS = [
    "method",
    "n_instances",
    "problem_success_rate",
    "path_success_rate",
    "mean_cost",
    "median_cost",
    "std_cost_ddof0",
    "min_cost",
    "max_cost",
    "mean_accepted_steps",
    "mean_rejected_steps",
    "mean_improvement_vs_linear",
    "positive_improvement_rate",
    "maximum_residual_norm",
    "total_elapsed_seconds",
]


def _finite(value: Any, *, name: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return number


def write_rows(path: str | Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    """Write rows to CSV with fixed column ordering."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_rows(path: str | Path) -> list[dict[str, str]]:
    """Read a CSV file and reject empty or malformed rows."""

    input_path = Path(path)
    with input_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV is missing a header.")
        rows = list(reader)
    if not rows:
        raise ValueError("CSV must contain at least one data row.")
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate evaluation rows by method.

    Standard deviation is the population standard deviation with ``ddof=0``.
    """

    if not rows:
        raise ValueError("rows must not be empty.")
    methods = sorted({str(row["method"]) for row in rows})
    summaries: list[dict[str, Any]] = []
    for method in methods:
        method_rows = [row for row in rows if str(row["method"]) == method]
        n_instances = len(method_rows)
        costs = np.array([_finite(row["mean_cost"], name="mean_cost") for row in method_rows])
        n_success = np.array([_finite(row["n_success"], name="n_success") for row in method_rows])
        n_paths = np.array([_finite(row["n_paths"], name="n_paths") for row in method_rows])
        improvements = np.array(
            [
                _finite(row["cost_improvement_vs_linear"], name="cost_improvement_vs_linear")
                for row in method_rows
            ]
        )
        residuals = np.array(
            [_finite(row["max_residual_norm"], name="max_residual_norm") for row in method_rows]
        )
        summary = {
            "method": method,
            "n_instances": n_instances,
            "problem_success_rate": float(
                np.mean([float(row["problem_success"]) for row in method_rows])
            ),
            "path_success_rate": float(np.sum(n_success) / np.sum(n_paths)),
            "mean_cost": float(np.mean(costs)),
            "median_cost": float(np.median(costs)),
            "std_cost_ddof0": float(np.std(costs, ddof=0)),
            "min_cost": float(np.min(costs)),
            "max_cost": float(np.max(costs)),
            "mean_accepted_steps": float(
                np.mean([_finite(row["accepted_steps"], name="accepted_steps") for row in method_rows])
            ),
            "mean_rejected_steps": float(
                np.mean([_finite(row["rejected_steps"], name="rejected_steps") for row in method_rows])
            ),
            "mean_improvement_vs_linear": float(np.mean(improvements)),
            "positive_improvement_rate": float(np.mean(improvements > 0.0)),
            "maximum_residual_norm": float(np.max(residuals)),
            "total_elapsed_seconds": float(
                np.sum([_finite(row["elapsed_seconds"], name="elapsed_seconds") for row in method_rows])
            ),
        }
        for key, value in summary.items():
            if key != "method" and not np.isfinite(float(value)):
                raise ValueError(f"summary field {key} must be finite.")
        summaries.append(summary)
    return summaries


def write_summary_csv(path: str | Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate rows and write benchmark summary CSV."""

    summaries = summarize_rows(rows)
    write_rows(path, summaries, fieldnames=SUMMARY_FIELDS)
    return summaries


__all__ = [
    "EVALUATION_FIELDS",
    "SUMMARY_FIELDS",
    "read_rows",
    "summarize_rows",
    "write_rows",
    "write_summary_csv",
]
