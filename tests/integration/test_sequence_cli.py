from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

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


def test_sequence_bc_finetune_evaluate_timing(tmp_path) -> None:
    root = tmp_path / "outputs"
    labels = oracle(root, "labels", "--evaluation-seed", "900000", "--num-instances", "4")
    ref = oracle(root, "oracle", "--num-instances", "2")

    run_cli(
        "behavior_cloning.py", "--config", CONFIG, "--label-run-dir", str(labels), "--latent-dim", "full",
        "--actor-type", "sequence", *SMALL, "--augment-symmetry", "--epochs", "2", "--val-fraction", "0.25",
        "--num-instances", "2", "--experiment-id", "EXP-TEST", "--run-id", "bc", "--oracle-run-dir", str(ref),
        "--output-root", str(root), "--allow-dirty",
    )
    bc = root / "EXP-TEST" / "bc"
    checkpoint = load_checkpoint(bc / "checkpoints" / "last.pt")
    assert checkpoint["actor_type"] == "sequence"
    assert checkpoint["attention_kwargs"]["d_model"] == 16
    assert checkpoint["ppo_config"]["actor_type"] == "sequence"
    metadata = json.loads((bc / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["behavior_cloning"]["actor_type"] == "sequence"
    assert metadata["behavior_cloning"]["attention"]["d_model"] == 16

    run_cli(
        "finetune.py", "--config", CONFIG, "--latent-dim", "full", "--init-run-dir", str(bc),
        "--iterations", "2", "--batch-problems", "2", "--pairs", "1", "--val-instances", "2",
        "--val-interval", "1", "--num-instances", "2", "--experiment-id", "EXP-TEST", "--run-id", "ft",
        "--oracle-run-dir", str(ref), "--output-root", str(root), "--allow-dirty",
        "--bc-weight", "1", "--label-run-dir", str(labels), "--augment-symmetry", "--bc-batch-size", "4",
    )
    ft = root / "EXP-TEST" / "ft"
    assert load_checkpoint(ft / "checkpoints" / "last.pt")["actor_type"] == "sequence"
    assert len(read_csv_rows(ft / "val_metrics.csv")) == 3

    run_cli("evaluate.py", "--run-dir", str(ft), "--checkpoint", str(ft / "checkpoints" / "last.pt"), "--num-instances", "2")
    assert len(read_csv_rows(ft / "evaluation" / "evaluation.csv")) == 4

    timing = tmp_path / "timing"
    run_cli(
        "timing.py", "--config", CONFIG, "--latent-dim", "full", "--run-dir", str(bc), "--run-dir", str(ft),
        "--num-instances", "2", "--repeats", "2", "--warmup", "1", "--output-dir", str(timing),
    )
    summary = {r["method"] for r in read_csv_rows(timing / "timing_summary.csv")}
    assert summary == {"Linear", "bc", "ft"}
