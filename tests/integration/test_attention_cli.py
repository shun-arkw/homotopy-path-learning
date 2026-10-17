from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from homotopy_path_learning.rl.checkpoint import load_checkpoint


pytestmark = [pytest.mark.integration, pytest.mark.slow]

CONFIG = "experiments/multivariate_pham/configs/exp-0004-smoke.yaml"
SMALL = ["--attention-d-model", "16", "--attention-heads", "2", "--attention-layers", "2", "--attention-ff-dim", "32"]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_cli(script: str, *arguments: str) -> None:
    env = os.environ.copy()
    env["PYTHON_JULIACALL_EXE"] = env.get("PYTHON_JULIACALL_EXE", "/usr/local/bin/julia")
    env["PYTHON_JULIACALL_PROJECT"] = env.get("PYTHON_JULIACALL_PROJECT", str(Path.cwd() / "julia"))
    subprocess.run([sys.executable, f"experiments/multivariate_pham/{script}", *arguments], check=True, env=env)


def oracle(output_root: Path, run_id: str, *extra: str) -> Path:
    run_cli(
        "oracle.py", "--config", CONFIG, "--experiment-id", "EXP-TEST", "--run-id", run_id,
        "--latent-dim", "full", "--budget", "6", "--n-random", "3",
        "--output-root", str(output_root), "--allow-dirty", *extra,
    )
    return output_root / "EXP-TEST" / run_id


def test_attention_bc_finetune_evaluate_timing_compare(tmp_path) -> None:
    root = tmp_path / "outputs"
    labels = oracle(root, "labels", "--evaluation-seed", "900000", "--num-instances", "4")
    ref = oracle(root, "oracle", "--num-instances", "2")

    run_cli(
        "behavior_cloning.py", "--config", CONFIG, "--label-run-dir", str(labels), "--latent-dim", "full",
        "--actor-type", "attention", *SMALL, "--augment-symmetry", "--epochs", "2", "--val-fraction", "0.25",
        "--num-instances", "2", "--experiment-id", "EXP-TEST", "--run-id", "bc", "--oracle-run-dir", str(ref),
        "--output-root", str(root), "--allow-dirty",
    )
    bc = root / "EXP-TEST" / "bc"
    checkpoint = load_checkpoint(bc / "checkpoints" / "last.pt")
    assert checkpoint["actor_type"] == "attention"
    assert checkpoint["attention_kwargs"]["d_model"] == 16
    assert checkpoint["ppo_config"]["actor_type"] == "attention"
    metadata = json.loads((bc / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["behavior_cloning"]["actor_type"] == "attention"

    run_cli(
        "finetune.py", "--config", CONFIG, "--latent-dim", "full", "--init-run-dir", str(bc),
        "--iterations", "2", "--batch-problems", "2", "--pairs", "1", "--val-instances", "2",
        "--val-interval", "1", "--num-instances", "2", "--experiment-id", "EXP-TEST", "--run-id", "ft",
        "--oracle-run-dir", str(ref), "--output-root", str(root), "--allow-dirty",
        "--bc-weight", "1", "--label-run-dir", str(labels), "--augment-symmetry", "--bc-batch-size", "4",
    )
    ft = root / "EXP-TEST" / "ft"
    assert load_checkpoint(ft / "checkpoints" / "last.pt")["actor_type"] == "attention"
    assert len(read_csv_rows(ft / "val_metrics.csv")) == 3

    run_cli("evaluate.py", "--run-dir", str(ft), "--checkpoint", str(ft / "checkpoints" / "last.pt"), "--num-instances", "2")
    assert len(read_csv_rows(ft / "evaluation" / "evaluation.csv")) == 4

    # --output-subdir: default unchanged, custom subdir elsewhere, never overwritten.
    default_csv = ft / "evaluation" / "evaluation.csv"
    before = default_csv.read_bytes()
    extra = ["--run-dir", str(ft), "--checkpoint", str(ft / "checkpoints" / "last.pt"), "--num-instances", "3",
             "--evaluation-seed", "900000", "--output-subdir", "evaluation_test"]
    run_cli("evaluate.py", *extra)
    assert default_csv.read_bytes() == before
    assert len(read_csv_rows(ft / "evaluation_test" / "evaluation.csv")) == 6
    meta = json.loads((ft / "metadata.json").read_text(encoding="utf-8"))
    assert meta["extra_evaluations"][0]["subdir"] == "evaluation_test"
    assert meta["extra_evaluations"][0]["num_instances"] == 3
    assert "evaluation_test/evaluation.csv" in meta["output_files"]
    with pytest.raises(subprocess.CalledProcessError):
        run_cli("evaluate.py", *extra)
    run_cli("evaluate.py", "--run-dir", str(ft), "--checkpoint", str(ft / "checkpoints" / "last.pt"), "--num-instances", "2")
    assert len(read_csv_rows(default_csv)) == 4  # default subdir may be rewritten

    timing = tmp_path / "timing"
    run_cli(
        "timing.py", "--config", CONFIG, "--latent-dim", "full", "--run-dir", str(bc), "--run-dir", str(ft),
        "--num-instances", "2", "--repeats", "2", "--warmup", "1", "--output-dir", str(timing),
    )
    rows = read_csv_rows(timing / "timing.csv")
    assert len(rows) == 2 * 3
    summary = {r["method"]: r for r in read_csv_rows(timing / "timing_summary.csv")}
    assert set(summary) == {"Linear", "bc", "ft"}
    assert float(summary["Linear"]["ratio_to_linear"]) == pytest.approx(1.0)
    assert float(summary["Linear"]["delta_j_percent"]) == 0.0
    assert (timing / "metadata.json").is_file()

    compare = tmp_path / "compare"
    run_cli(
        "compare_runs.py", "--run-dir", str(bc), "--run-dir", str(ft), "--group", f"both={bc},{ft}",
        "--oracle-run-dir", str(ref), "--pair", "bc,ft", "--pair", "bc,both", "--bootstrap", "50",
        "--output-dir", str(compare),
    )
    run_cli(
        "compare_runs.py", "--run-dir", str(ft), "--evaluation-subdir", "evaluation_test",
        "--output-dir", str(tmp_path / "compare_sub"),
    )
    assert float(read_csv_rows(tmp_path / "compare_sub" / "comparison_summary.csv")[-1]["n_problems"]) == 3
    names = [r["name"] for r in read_csv_rows(compare / "comparison_summary.csv")]
    assert names == ["Linear", "Oracle", "bc", "ft", "both"]
    pairs = read_csv_rows(compare / "comparison_pairs.csv")
    assert len(pairs) == 6
    assert all(float(r["ci_low"]) <= float(r["ci_high"]) for r in pairs if r["ci_low"] != "nan")
    assert np.isfinite(float(pairs[0]["diff"]))


def test_attention_zero_epoch_bc_then_finetune(tmp_path) -> None:
    root = tmp_path / "outputs"
    labels = oracle(root, "labels", "--evaluation-seed", "900000", "--num-instances", "4")

    run_cli(
        "behavior_cloning.py", "--config", CONFIG, "--label-run-dir", str(labels), "--latent-dim", "full",
        "--actor-type", "attention", *SMALL, "--epochs", "0", "--val-fraction", "0.25",
        "--num-instances", "2", "--experiment-id", "EXP-TEST", "--run-id", "bc0", "--output-root", str(root),
        "--allow-dirty",
    )
    bc = root / "EXP-TEST" / "bc0"
    assert load_checkpoint(bc / "checkpoints" / "last.pt")["actor_type"] == "attention"
    assert read_csv_rows(bc / "train_metrics.csv") == []  # header only
    metadata = json.loads((bc / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["behavior_cloning"]["epochs"] == 0
    assert metadata["behavior_cloning"]["best_epoch"] == 0
    assert len(read_csv_rows(bc / "evaluation" / "evaluation.csv")) == 4

    run_cli(
        "finetune.py", "--config", CONFIG, "--latent-dim", "full", "--init-run-dir", str(bc),
        "--iterations", "2", "--batch-problems", "2", "--pairs", "1", "--val-instances", "2",
        "--val-interval", "1", "--num-instances", "2", "--experiment-id", "EXP-TEST", "--run-id", "ft0",
        "--output-root", str(root), "--allow-dirty",
    )
    ft = root / "EXP-TEST" / "ft0"
    assert load_checkpoint(ft / "checkpoints" / "last.pt")["actor_type"] == "attention"
    assert len(read_csv_rows(ft / "val_metrics.csv")) == 3


def test_finetune_from_a_run_with_other_tracker_settings_needs_the_flag(tmp_path) -> None:
    import yaml

    root = tmp_path / "outputs"
    labels = oracle(root, "labels", "--evaluation-seed", "900000", "--num-instances", "4")
    run_cli(
        "behavior_cloning.py", "--config", CONFIG, "--label-run-dir", str(labels), "--latent-dim", "full",
        "--actor-type", "attention", *SMALL, "--epochs", "1", "--val-fraction", "0.25", "--num-instances", "1",
        "--experiment-id", "EXP-TEST", "--run-id", "bc", "--output-root", str(root), "--allow-dirty",
    )
    data = yaml.safe_load(Path(CONFIG).read_text(encoding="utf-8"))
    data["tracker"].update({"beta_omega_p": 0.8, "max_step_size": 1.0})
    other = tmp_path / "other-tracker.yaml"
    other.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    common = [
        "finetune.py", "--config", str(other), "--latent-dim", "full", "--init-run-dir", str(root / "EXP-TEST" / "bc"),
        "--iterations", "1", "--batch-problems", "2", "--pairs", "1", "--val-instances", "2", "--val-interval", "1",
        "--num-instances", "1", "--experiment-id", "EXP-TEST", "--output-root", str(root), "--allow-dirty",
    ]
    with pytest.raises(subprocess.CalledProcessError):
        run_cli(*common[:1], *common[1:], "--run-id", "ft-refused")
    run_cli(*common[:1], *common[1:], "--run-id", "ft", "--allow-init-tracker-difference")
    ft = root / "EXP-TEST" / "ft"
    assert len(read_csv_rows(ft / "val_metrics.csv")) == 2
    saved = yaml.safe_load((ft / "config.yaml").read_text(encoding="utf-8"))
    assert saved["tracker"]["beta_omega_p"] == 0.8
