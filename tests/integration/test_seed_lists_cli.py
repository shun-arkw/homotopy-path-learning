from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.integration, pytest.mark.slow]

CONFIG = "experiments/multivariate_pham/configs/exp-0004-smoke.yaml"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def cli_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHON_JULIACALL_EXE"] = env.get("PYTHON_JULIACALL_EXE", "/usr/local/bin/julia")
    env["PYTHON_JULIACALL_PROJECT"] = env.get("PYTHON_JULIACALL_PROJECT", str(Path.cwd() / "julia"))
    return env


def run_cli(script: str, *arguments: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, f"experiments/multivariate_pham/{script}", *arguments],
        check=check,
        env=cli_env(),
        capture_output=not check,
        text=True,
    )


def oracle(output_root: Path, run_id: str, *extra: str) -> Path:
    run_cli(
        "oracle.py", "--config", CONFIG, "--experiment-id", "EXP-TEST", "--run-id", run_id,
        "--budget", "4", "--n-random", "2", "--output-root", str(output_root), "--allow-dirty", *extra,
    )
    return output_root / "EXP-TEST" / run_id


@pytest.fixture(scope="module")
def bc_run(tmp_path_factory) -> tuple[Path, Path]:
    output_root = tmp_path_factory.mktemp("seedlists") / "outputs"
    labels = oracle(output_root, "labels", "--evaluation-seed", "900000", "--num-instances", "4")
    run_cli(
        "behavior_cloning.py", "--config", CONFIG, "--label-run-dir", str(labels),
        "--hidden-sizes", "16,16", "--epochs", "2", "--val-fraction", "0.25", "--num-instances", "2",
        "--experiment-id", "EXP-TEST", "--run-id", "bc-init", "--output-root", str(output_root),
        "--allow-dirty",
    )
    return output_root, output_root / "EXP-TEST" / "bc-init"


def test_target_seeds_file_oracle_and_evaluate(bc_run, tmp_path) -> None:
    output_root, init_dir = bc_run
    seeds_file = tmp_path / "seeds.txt"
    seeds_file.write_text("# test\n900002\n\n900000\n900003\n", encoding="utf-8")

    contiguous = oracle(output_root, "or-contig", "--evaluation-seed", "900000", "--num-instances", "4")
    listed = oracle(output_root, "or-list", "--target-seeds-file", str(seeds_file))
    base = {r["target_seed"]: r for r in read_csv_rows(contiguous / "oracle.csv")}
    rows = read_csv_rows(listed / "oracle.csv")
    assert [int(r["instance_id"]) for r in rows] == [0, 1, 2]
    assert [int(r["target_seed"]) for r in rows] == [900002, 900000, 900003]
    for row in rows:
        for key in ("linear_cost", "zero_action_cost", "oracle_cost", "oracle_accepted_steps"):
            assert row[key] == base[row["target_seed"]][key]
    metadata = json.loads((listed / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["oracle"]["target_seeds_file"]["path"] == str(seeds_file)
    assert len(metadata["oracle"]["target_seeds_file"]["sha256"]) == 64

    ckpt = init_dir / "checkpoints" / "last.pt"
    for name, extra in (("ev-contig", ["--evaluation-seed", "900000", "--num-instances", "4"]),
                        ("ev-list", ["--target-seeds-file", str(seeds_file)])):
        run_cli("evaluate.py", "--run-dir", str(init_dir), "--checkpoint", str(ckpt),
                "--output-subdir", name, *extra)
    ev_base = {(r["target_seed"], r["method"]): r for r in read_csv_rows(init_dir / "ev-contig" / "evaluation.csv")}
    ev_rows = read_csv_rows(init_dir / "ev-list" / "evaluation.csv")
    assert [int(r["target_seed"]) for r in ev_rows if r["method"] == "Linear"] == [900002, 900000, 900003]
    assert [int(r["instance_id"]) for r in ev_rows if r["method"] == "Linear"] == [0, 1, 2]
    for row in ev_rows:
        assert row["mean_cost"] == ev_base[(row["target_seed"], row["method"])]["mean_cost"]
    meta = json.loads((init_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["extra_evaluations"][-1]["target_seeds_file"]["path"] == str(seeds_file)

    for args in (["--evaluation-seed", "1"], ["--num-instances", "2"]):
        failed = run_cli("evaluate.py", "--run-dir", str(init_dir), "--checkpoint", str(ckpt),
                         "--output-subdir", "bad", "--target-seeds-file", str(seeds_file), *args, check=False)
        assert failed.returncode != 0
    dup = tmp_path / "dup.txt"
    dup.write_text("1\n1\n", encoding="utf-8")
    assert run_cli("oracle.py", "--config", CONFIG, "--experiment-id", "EXP-TEST", "--run-id", "dup",
                   "--target-seeds-file", str(dup), "--output-root", str(output_root), "--allow-dirty",
                   check=False).returncode != 0


def finetune(output_root: Path, init_dir: Path, run_id: str, *extra: str) -> Path:
    run_cli(
        "finetune.py", "--config", CONFIG, "--init-run-dir", str(init_dir),
        "--iterations", "3", "--batch-problems", "2", "--pairs", "1", "--val-instances", "3",
        "--val-interval", "1", "--num-instances", "2", "--experiment-id", "EXP-TEST", "--run-id", run_id,
        "--output-root", str(output_root), "--allow-dirty", *extra,
    )
    return output_root / "EXP-TEST" / run_id


def assert_same_run(a: Path, b: Path) -> None:
    assert (a / "val_metrics.csv").read_text() == (b / "val_metrics.csv").read_text()
    drop = lambda rows: [{k: v for k, v in r.items() if k != "elapsed"} for r in rows]
    assert drop(read_csv_rows(a / "finetune_metrics.csv")) == drop(read_csv_rows(b / "finetune_metrics.csv"))
    ma = json.loads((a / "metadata.json").read_text())["finetune"]
    mb = json.loads((b / "metadata.json").read_text())["finetune"]
    assert ma["best_iteration"] == mb["best_iteration"]
    sa = torch.load(a / "checkpoints" / "last.pt", map_location="cpu")["model_state_dict"]
    sb = torch.load(b / "checkpoints" / "last.pt", map_location="cpu")["model_state_dict"]
    assert sa.keys() == sb.keys()
    for key in sa:
        assert torch.equal(sa[key], sb[key]), key


def test_finetune_seed_files_and_workers(bc_run, tmp_path) -> None:
    output_root, init_dir = bc_run
    default = finetune(output_root, init_dir, "ft-default")

    train_file = tmp_path / "train.txt"
    train_file.write_text("".join(f"{6000000 + i}\n" for i in range(6)), encoding="utf-8")
    val_file = tmp_path / "val.txt"
    val_file.write_text("".join(f"{5000000 + i}\n" for i in range(5)), encoding="utf-8")  # longer than needed
    listed = finetune(output_root, init_dir, "ft-list", "--train-seeds-file", str(train_file),
                      "--val-seeds-file", str(val_file))
    assert_same_run(default, listed)
    meta = json.loads((listed / "metadata.json").read_text())["finetune"]
    assert meta["train_seeds_file"]["path"] == str(train_file) and len(meta["train_seeds_file"]["sha256"]) == 64
    assert meta["val_seeds_file"]["path"] == str(val_file)
    assert meta["workers"] == 1

    parallel = finetune(output_root, init_dir, "ft-w2", "--workers", "2")
    assert_same_run(default, parallel)
    assert json.loads((parallel / "metadata.json").read_text())["finetune"]["workers"] == 2

    short = tmp_path / "short.txt"
    short.write_text("1\n2\n3\n", encoding="utf-8")
    for flag in ("--train-seeds-file", "--val-seeds-file"):
        failed = run_cli(
            "finetune.py", "--config", CONFIG, "--init-run-dir", str(init_dir), "--iterations", "3",
            "--batch-problems", "2", "--val-instances", "4", "--experiment-id", "EXP-TEST",
            "--run-id", "short" + flag[2:5], "--output-root", str(output_root), "--allow-dirty",
            flag, str(short), check=False,
        )
        # train needs 6 seeds, val needs 4 seeds: both too short
        assert failed.returncode != 0
    assert not (output_root / "EXP-TEST" / "shorttra").exists()


def test_screen_linear_and_select_hard_seeds(tmp_path) -> None:
    output_root = tmp_path / "outputs"
    for run_id, start in (("scr-a", 700000), ("scr-b", 700006)):
        run_cli("screen_linear.py", "--config", CONFIG, "--experiment-id", "EXP-TEST", "--run-id", run_id,
                "--start-seed", str(start), "--num-candidates", "6", "--output-root", str(output_root),
                "--allow-dirty")
    run_a, run_b = (output_root / "EXP-TEST" / n for n in ("scr-a", "scr-b"))
    rows = read_csv_rows(run_a / "linear_costs.csv")
    assert list(rows[0]) == ["target_seed", "linear_cost", "linear_success", "accepted_steps", "rejected_steps"]
    assert [int(r["target_seed"]) for r in rows] == list(range(700000, 700006))
    assert (run_a / "config.yaml").is_file()
    meta = json.loads((run_a / "metadata.json").read_text())
    assert meta["screen_linear"]["num_candidates"] == 6

    # The screened linear cost equals the oracle's linear cost for the same seed.
    orc = oracle(output_root, "orc", "--evaluation-seed", "700000", "--num-instances", "2")
    for srow, orow in zip(rows, read_csv_rows(orc / "oracle.csv")):
        assert srow["linear_cost"] == orow["linear_cost"]
        assert srow["linear_success"] == orow["linear_success"]

    out = tmp_path / "selected"
    run_cli("select_hard_seeds.py", "--screen-run-dir", str(run_b), "--screen-run-dir", str(run_a),
            "--calibration", "4", "--quantile", "0.0", "--split", "first:3", "--split", "second:2",
            "--output-dir", str(out))
    first = [int(x) for x in (out / "first.txt").read_text().split()]
    second = [int(x) for x in (out / "second.txt").read_text().split()]
    assert first == [700004, 700005, 700006] and second == [700007, 700008]  # quantile 0 keeps all successes
    selection = json.loads((out / "selection.json").read_text())
    assert selection["calibration"] == {"count": 4, "seed_min": 700000, "seed_max": 700003}
    assert selection["splits"]["second"] == {"count": 2, "seed_min": 700007, "seed_max": 700008}
    assert selection["threshold"] == min(float(r["linear_cost"]) for r in rows[:4])

    again = run_cli("select_hard_seeds.py", "--screen-run-dir", str(run_a), "--calibration", "2",
                    "--quantile", "0.5", "--split", "x:1", "--output-dir", str(out), check=False)
    assert again.returncode != 0
    overlap = run_cli("select_hard_seeds.py", "--screen-run-dir", str(run_a), "--screen-run-dir", str(run_a),
                      "--calibration", "2", "--quantile", "0.5", "--split", "x:1",
                      "--output-dir", str(tmp_path / "o2"), check=False)
    assert overlap.returncode != 0 and not (tmp_path / "o2").exists()
    short = run_cli("select_hard_seeds.py", "--screen-run-dir", str(run_a), "--calibration", "2",
                    "--quantile", "0.5", "--split", "x:100", "--output-dir", str(tmp_path / "o3"), check=False)
    assert short.returncode != 0
