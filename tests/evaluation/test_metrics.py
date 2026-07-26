from __future__ import annotations

import csv

import pytest

from homotopy_path_learning.config import load_config
from homotopy_path_learning.evaluation.analyzer import generate_summary_markdown
from homotopy_path_learning.evaluation.datasets import fixed_evaluation_seeds
from homotopy_path_learning.evaluation.metrics import (
    EVALUATION_FIELDS,
    SUMMARY_FIELDS,
    read_rows,
    summarize_rows,
    write_rows,
    write_summary_csv,
)


def sample_rows():
    return [
        {
            "experiment_id": "EXP",
            "run_id": "run",
            "instance_id": 0,
            "target_seed": 10,
            "method": "Linear",
            "problem_success": 1,
            "n_paths": 4,
            "n_success": 4,
            "n_failed": 0,
            "mean_cost": 10.0,
            "accepted_steps": 40,
            "rejected_steps": 0,
            "cost_improvement_vs_linear": 0.0,
            "reward_equivalent": 0.0,
            "max_residual_norm": 1e-12,
            "failure_codes": "",
            "action_l2_norm": 0.0,
            "elapsed_seconds": 0.1,
        },
        {
            "experiment_id": "EXP",
            "run_id": "run",
            "instance_id": 0,
            "target_seed": 10,
            "method": "LearnedBezier",
            "problem_success": 1,
            "n_paths": 4,
            "n_success": 4,
            "n_failed": 0,
            "mean_cost": 8.0,
            "accepted_steps": 32,
            "rejected_steps": 0,
            "cost_improvement_vs_linear": 2.0,
            "reward_equivalent": 2.0,
            "max_residual_norm": 1e-12,
            "failure_codes": "",
            "action_l2_norm": 0.5,
            "elapsed_seconds": 0.2,
        },
    ]


def test_fixed_evaluation_seed_rule() -> None:
    assert fixed_evaluation_seeds(seed=100, num_instances=4) == [100, 101, 102, 103]


def test_summary_metrics_match_manual_values() -> None:
    summaries = summarize_rows(sample_rows())
    learned = next(row for row in summaries if row["method"] == "LearnedBezier")

    assert learned["problem_success_rate"] == 1.0
    assert learned["path_success_rate"] == 1.0
    assert learned["mean_cost"] == 8.0
    assert learned["median_cost"] == 8.0
    assert learned["std_cost_ddof0"] == 0.0
    assert learned["mean_improvement_vs_linear"] == 2.0
    assert learned["positive_improvement_rate"] == 1.0


def test_csv_field_order_and_markdown_generation(tmp_path) -> None:
    rows = sample_rows()
    csv_path = tmp_path / "evaluation.csv"
    summary_path = tmp_path / "summary.csv"
    markdown_path = tmp_path / "summary.md"

    write_rows(csv_path, rows, fieldnames=EVALUATION_FIELDS)
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        assert next(reader) == EVALUATION_FIELDS
    loaded = read_rows(csv_path)
    summaries = write_summary_csv(summary_path, loaded)
    with summary_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        assert next(reader) == SUMMARY_FIELDS
    assert summaries

    generate_summary_markdown(
        config=load_config("experiments/multivariate_pham/configs/exp-0004-smoke.yaml"),
        run_id="run",
        benchmark_csv=csv_path,
        checkpoint="last.pt",
        git_commit="abc",
        output_path=markdown_path,
    )
    text = markdown_path.read_text(encoding="utf-8")
    assert "Linear" in text
    assert "LearnedBezier" in text
    assert "smoke test" in text


def test_invalid_csv_is_rejected(tmp_path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("method,mean_cost\n", encoding="utf-8")

    with pytest.raises(ValueError, match="data row"):
        read_rows(path)
