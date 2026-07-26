from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.phase6]


def run_command(args: list[str], *, env: dict[str, str]) -> None:
    subprocess.run(args, check=True, env=env)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def assert_numeric_columns_are_finite(rows: list[dict[str, str]], columns: list[str]) -> None:
    for row in rows:
        for column in columns:
            assert np.isfinite(float(row[column]))


def test_phase6_cli_smoke_pipeline(tmp_path) -> None:
    env = os.environ.copy()
    env["PYTHON_JULIACALL_EXE"] = env.get("PYTHON_JULIACALL_EXE", "/usr/local/bin/julia")
    env["PYTHON_JULIACALL_PROJECT"] = env.get("PYTHON_JULIACALL_PROJECT", str(Path.cwd() / "julia"))
    run_id = "pytest-phase6-smoke"
    output_root = tmp_path / "outputs"
    run_dir = output_root / "EXP-0004" / run_id
    checkpoint = run_dir / "checkpoints" / "last.pt"

    run_command(
        [
            sys.executable,
            "experiments/multivariate_pham/train.py",
            "--config",
            "experiments/multivariate_pham/configs/exp-0004-smoke.yaml",
            "--run-id",
            run_id,
            "--device",
            "cpu",
            "--total-timesteps",
            "64",
            "--output-root",
            str(output_root),
            "--allow-dirty",
        ],
        env=env,
    )
    run_command(
        [
            sys.executable,
            "experiments/multivariate_pham/evaluate.py",
            "--run-dir",
            str(run_dir),
            "--checkpoint",
            str(checkpoint),
            "--num-instances",
            "2",
            "--device",
            "cpu",
        ],
        env=env,
    )
    run_command(
        [
            sys.executable,
            "experiments/multivariate_pham/benchmark.py",
            "--run-dir",
            str(run_dir),
            "--checkpoint",
            str(checkpoint),
            "--num-instances",
            "2",
            "--device",
            "cpu",
        ],
        env=env,
    )
    run_command(
        [
            sys.executable,
            "experiments/multivariate_pham/analyze.py",
            "--run-dir",
            str(run_dir),
        ],
        env=env,
    )

    assert checkpoint.is_file()
    assert (run_dir / "checkpoints" / "best.pt").is_file()
    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "metadata.json").is_file()
    assert (run_dir / "latent_basis.npy").is_file()
    assert (run_dir / "train_metrics.csv").is_file()
    assert (run_dir / "evaluation_seeds.json").is_file()
    assert (run_dir / "evaluation" / "evaluation.csv").is_file()
    assert (run_dir / "benchmark" / "benchmark.csv").is_file()
    assert (run_dir / "benchmark" / "benchmark_summary.csv").is_file()
    assert (run_dir / "analysis" / "summary.md").is_file()

    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "Completed"
    assert metadata["run_id"] == run_id

    train_rows = read_csv_rows(run_dir / "train_metrics.csv")
    assert len(train_rows) == 1
    assert_numeric_columns_are_finite(
        train_rows,
        [
            "rollout_reward_mean",
            "rollout_reward_min",
            "rollout_reward_max",
            "rollout_linear_cost_mean",
            "rollout_bezier_cost_mean",
            "policy_loss",
            "value_loss",
            "approx_kl",
            "clip_fraction",
        ],
    )

    eval_rows = read_csv_rows(run_dir / "evaluation" / "evaluation.csv")
    assert len(eval_rows) == 4
    assert {row["method"] for row in eval_rows} == {"Linear", "LearnedBezier"}
    assert_numeric_columns_are_finite(eval_rows, ["mean_cost", "reward_equivalent"])

    benchmark_rows = read_csv_rows(run_dir / "benchmark" / "benchmark.csv")
    assert len(benchmark_rows) == 6
    assert {row["method"] for row in benchmark_rows} == {
        "Linear",
        "RandomBezier",
        "LearnedBezier",
    }
    by_seed: dict[str, set[str]] = {}
    for row in benchmark_rows:
        by_seed.setdefault(row["target_seed"], set()).add(row["method"])
    assert all(methods == {"Linear", "RandomBezier", "LearnedBezier"} for methods in by_seed.values())
    assert_numeric_columns_are_finite(
        benchmark_rows,
        ["mean_cost", "cost_improvement_vs_linear", "max_residual_norm"],
    )

    summary_rows = read_csv_rows(run_dir / "benchmark" / "benchmark_summary.csv")
    assert len(summary_rows) == 3
    assert_numeric_columns_are_finite(
        summary_rows,
        ["problem_success_rate", "path_success_rate", "mean_cost", "median_cost"],
    )
    assert "smoke test" in (run_dir / "analysis" / "summary.md").read_text(encoding="utf-8")
