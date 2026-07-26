"""Phase 7 performance configuration and report writers."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from homotopy_path_learning.config import ROOT_KEYS, Phase6Config, config_to_dict, parse_config_dict
from homotopy_path_learning.performance.timing import TimingStats


@dataclass(frozen=True)
class PerformanceSettings:
    warmup_runs: int
    measure_runs: int
    trials: int
    fixed_problem_count: int
    repeated_target_seed: int
    evaluation_seed: int
    action_seed: int


@dataclass(frozen=True)
class PerformanceConfig:
    phase6: Phase6Config
    performance: PerformanceSettings


def _mapping(value: Any, *, name: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping.")
    actual = set(value)
    missing = keys - actual
    extra = actual - keys
    if missing:
        raise ValueError(f"{name} is missing required keys: {sorted(missing)}.")
    if extra:
        raise ValueError(f"{name} contains unknown keys: {sorted(extra)}.")
    return dict(value)


def _int(value: Any, *, name: str, positive: bool = False, nonnegative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative, got {value}.")
    return int(value)


def parse_performance_config_dict(data: dict[str, Any]) -> PerformanceConfig:
    """Parse a Phase 7 profile YAML mapping with strict keys."""

    root = _mapping(data, name="profile config", keys=set(ROOT_KEYS) | {"performance"})
    perf_raw = _mapping(
        root["performance"],
        name="performance",
        keys={
            "warmup_runs",
            "measure_runs",
            "trials",
            "fixed_problem_count",
            "repeated_target_seed",
            "evaluation_seed",
            "action_seed",
        },
    )
    phase6_data = {key: root[key] for key in ROOT_KEYS}
    return PerformanceConfig(
        phase6=parse_config_dict(phase6_data),
        performance=PerformanceSettings(
            warmup_runs=_int(perf_raw["warmup_runs"], name="performance.warmup_runs", nonnegative=True),
            measure_runs=_int(perf_raw["measure_runs"], name="performance.measure_runs", positive=True),
            trials=_int(perf_raw["trials"], name="performance.trials", positive=True),
            fixed_problem_count=_int(
                perf_raw["fixed_problem_count"],
                name="performance.fixed_problem_count",
                positive=True,
            ),
            repeated_target_seed=_int(
                perf_raw["repeated_target_seed"],
                name="performance.repeated_target_seed",
                nonnegative=True,
            ),
            evaluation_seed=_int(
                perf_raw["evaluation_seed"],
                name="performance.evaluation_seed",
                nonnegative=True,
            ),
            action_seed=_int(perf_raw["action_seed"], name="performance.action_seed", nonnegative=True),
        ),
    )


def load_performance_config(path: str | Path) -> PerformanceConfig:
    """Load a strict Phase 7 profile YAML file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("profile config must contain a YAML mapping.")
    return parse_performance_config_dict(data)


def _finite(value: object, *, name: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def write_performance_csv(path: str | Path, stats: list[TimingStats]) -> None:
    """Write timing stats to CSV with fixed column order."""

    fieldnames = list(TimingStats.__dataclass_fields__)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in stats:
            row = item.to_dict()
            for key, value in row.items():
                if key != "name":
                    _finite(value, name=key)
            writer.writerow(row)


def write_summary_markdown(
    path: str | Path,
    *,
    payload: dict[str, Any],
) -> None:
    """Generate a Markdown summary from the JSON performance payload."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    measurements = payload["measurements"]
    lines = [
        f"# {payload['experiment_id']} Phase 7 Performance",
        "",
        f"- Run ID: `{payload['run_id']}`",
        f"- Mode: `{payload['mode']}`",
        f"- Git commit: `{payload['metadata']['git_commit']}`",
        f"- Git dirty: `{payload['metadata']['git_dirty']}`",
        f"- Docker container: `{payload['metadata']['docker_container']}`",
        f"- Julia threads: `{payload['metadata']['julia_num_threads']}`",
        f"- Warmup runs: {payload['settings']['warmup_runs']}",
        f"- Measure runs: {payload['settings']['measure_runs']}",
        f"- Trials: {payload['settings']['trials']}",
        "",
        "## Timing",
        "",
        "| metric | median s | p95 s | throughput/s |",
        "|---|---:|---:|---:|",
    ]
    for name in sorted(measurements):
        stat = measurements[name]
        lines.append(
            "| "
            f"{name} | {float(stat['median_seconds']):.9f} | "
            f"{float(stat['p95_seconds']):.9f} | "
            f"{float(stat['throughput_per_second']):.3f} |"
        )
    comparison = payload.get("performance_comparison", {})
    if comparison:
        lines.extend(
            [
                "",
                "## Speedup",
                "",
                "Speedup is defined as `baseline_time / optimized_time` using median seconds.",
                "",
                "| metric | speedup | time reduction |",
                "|---|---:|---:|",
            ]
        )
        for name in sorted(comparison):
            row = comparison[name]
            lines.append(
                "| "
                f"{name} | {float(row['speedup']):.6f} | "
                f"{100.0 * float(row['time_reduction_fraction']):.3f}% |"
            )
    lines.extend(
        [
            "",
            "## Equivalence",
            "",
            f"- Detailed tracking record differences: {payload['equivalence']['tracking']['differing_items']}",
            f"- Evaluation row differences excluding elapsed time: {payload['equivalence']['evaluation_rows']['differing_items']}",
            f"- Path success differences: {payload['equivalence']['tracking']['path_success_differences']}",
            f"- Step-count differences: {payload['equivalence']['tracking']['tracking_step_differences']}",
            f"- Failure-code differences: {payload['equivalence']['tracking']['failure_code_differences']}",
            f"- Max endpoint absolute difference: {payload['equivalence']['tracking']['max_endpoint_abs_difference']}",
            f"- Max residual absolute difference: {payload['equivalence']['tracking']['max_residual_abs_difference']}",
            "",
            "## Optimization Decisions",
            "",
        ]
    )
    for decision in payload["optimization_decisions"]:
        lines.append(f"- {decision}")
    lines.extend(
        [
            "",
            "This is a small fixed-support smoke performance run. It is not a "
            "claim about large multivariate systems.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_performance_outputs(
    output_dir: str | Path,
    *,
    config: PerformanceConfig,
    payload: dict[str, Any],
    stats: list[TimingStats],
) -> None:
    """Write config, JSON, CSV, equivalence JSON, profile text, and summary."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    config_payload = config_to_dict(config.phase6)
    config_payload["performance"] = asdict(config.performance)
    with (directory / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config_payload, handle, sort_keys=False)
    (directory / "profiling").mkdir(parents=True, exist_ok=True)
    (directory / "performance.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_performance_csv(directory / "performance.csv", stats)
    (directory / "equivalence.json").write_text(
        json.dumps(payload["equivalence"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (directory / "profiling" / "python.txt").write_text(
        "\n".join(payload["profiling"]["python"]),
        encoding="utf-8",
    )
    (directory / "profiling" / "julia.txt").write_text(
        "\n".join(payload["profiling"]["julia"]),
        encoding="utf-8",
    )
    write_summary_markdown(directory / "summary.md", payload=payload)


__all__ = [
    "PerformanceConfig",
    "PerformanceSettings",
    "load_performance_config",
    "parse_performance_config_dict",
    "write_performance_csv",
    "write_performance_outputs",
    "write_summary_markdown",
]
