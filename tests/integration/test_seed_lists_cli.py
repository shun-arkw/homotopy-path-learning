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


def drop_elapsed(path: Path):
    return [{k: v for k, v in r.items() if k != "elapsed"} for r in read_csv_rows(path)]


def test_finetune_train_pool_and_workers(bc_run) -> None:
    output_root, init_dir = bc_run
    default = finetune(output_root, init_dir, "ft-nopool")
    assert not (default / "pool_metrics.csv").exists()
    default_meta = json.loads((default / "metadata.json").read_text())["finetune"]
    assert default_meta["train_pool_size"] is None and default_meta["train_pool_seeds"] is None
    assert list(read_csv_rows(default / "finetune_metrics.csv")[0]) == [
        "iteration", "mean_cost_ratio", "mean_grad_norm", "surrogate_loss", "bc_loss", "elapsed",
    ]

    extra = ("--train-pool-size", "4", "--batch-problems", "4")
    single = finetune(output_root, init_dir, "ft-pool1", *extra)
    double = finetune(output_root, init_dir, "ft-pool2", *extra, "--workers", "2")
    assert_same_run(single, double)
    assert (single / "pool_metrics.csv").read_text() == (double / "pool_metrics.csv").read_text()
    rows = read_csv_rows(single / "pool_metrics.csv")
    assert list(rows[0]) == ["iteration", "evaluations_per_target", "pool_delta_j"]
    assert [int(r["iteration"]) for r in rows] == [1, 2, 3]
    assert [int(r["evaluations_per_target"]) for r in rows] == [2, 4, 6]  # 2 * pairs(=1) * iteration
    assert all(np.isfinite(float(r["pool_delta_j"])) for r in rows)
    meta = json.loads((single / "metadata.json").read_text())["finetune"]
    assert meta["train_pool_size"] == 4 and meta["train_pool_seeds"] == [6000000, 6000003]

    # pool smaller than the batch (repeats) and a seed-list file shorter than the batch
    small = finetune(output_root, init_dir, "ft-pool-small", "--train-pool-size", "2", "--batch-problems", "4")
    assert len(read_csv_rows(small / "pool_metrics.csv")) == 3
    short = Path(output_root) / "short-seeds.txt"
    short.write_text("1\n2\n", encoding="utf-8")
    failed = run_cli(
        "finetune.py", "--config", CONFIG, "--init-run-dir", str(init_dir), "--iterations", "3",
        "--train-pool-size", "3", "--train-seeds-file", str(short), "--experiment-id", "EXP-TEST",
        "--run-id", "pool-short", "--output-root", str(output_root), "--allow-dirty", check=False,
    )
    assert failed.returncode != 0 and not (output_root / "EXP-TEST" / "pool-short").exists()
    # a pool smaller than iterations * batch is accepted with a short seed file
    listed = finetune(output_root, init_dir, "ft-pool-file", "--train-pool-size", "2", "--train-seeds-file", str(short))
    assert json.loads((listed / "metadata.json").read_text())["finetune"]["train_pool_seeds"] == [1, 2]
    zero = run_cli(
        "finetune.py", "--config", CONFIG, "--init-run-dir", str(init_dir), "--train-pool-size", "0",
        "--experiment-id", "EXP-TEST", "--run-id", "pool-zero", "--output-root", str(output_root),
        "--allow-dirty", check=False,
    )
    assert zero.returncode != 0


def test_finetune_sigma_schedule_and_workers(bc_run) -> None:
    output_root, init_dir = bc_run
    default = finetune(output_root, init_dir, "ft-const", "--sigma", "0.1")
    assert "sigma" not in read_csv_rows(default / "finetune_metrics.csv")[0]
    meta = json.loads((default / "metadata.json").read_text())["finetune"]
    assert meta["sigma_end"] is None and meta["sigma_schedule"] == "constant"

    one = finetune(output_root, init_dir, "ft-decay1", "--sigma", "0.1", "--sigma-end", "0.04")
    two = finetune(output_root, init_dir, "ft-decay2", "--sigma", "0.1", "--sigma-end", "0.04", "--workers", "2")
    assert_same_run(one, two)
    sigmas = [float(r["sigma"]) for r in read_csv_rows(one / "finetune_metrics.csv")]
    assert sigmas == pytest.approx([0.1, 0.07, 0.04])
    meta = json.loads((one / "metadata.json").read_text())["finetune"]
    assert meta["sigma_end"] == 0.04 and meta["sigma_schedule"] == "linear" and meta["sigma"] == 0.1
    # sigma_end == sigma reproduces the constant schedule exactly (apart from the extra column)
    same = finetune(output_root, init_dir, "ft-flat", "--sigma", "0.1", "--sigma-end", "0.1")
    strip = lambda rows: [{k: v for k, v in r.items() if k not in ("elapsed", "sigma")} for r in rows]
    assert strip(read_csv_rows(same / "finetune_metrics.csv")) == strip(read_csv_rows(default / "finetune_metrics.csv"))
    for bad in ("0.2", "0", "-0.1"):
        failed = run_cli(
            "finetune.py", "--config", CONFIG, "--init-run-dir", str(init_dir), "--sigma", "0.1",
            "--sigma-end", bad, "--experiment-id", "EXP-TEST", "--run-id", "sigma-bad",
            "--output-root", str(output_root), "--allow-dirty", check=False,
        )
        assert failed.returncode != 0 and not (output_root / "EXP-TEST" / "sigma-bad").exists()


def bc_run_cli(output_root: Path, run_id: str, *extra: str) -> Path:
    run_cli(
        "behavior_cloning.py", "--config", CONFIG, "--hidden-sizes", "16,16", "--epochs", "3",
        "--val-fraction", "0.25", "--num-instances", "2", "--experiment-id", "EXP-TEST", "--run-id", run_id,
        "--output-root", str(output_root), "--allow-dirty", *extra,
    )
    return output_root / "EXP-TEST" / run_id


def test_nearest_label_bc_and_centroid_labels_cli(bc_run) -> None:
    output_root, _ = bc_run
    labels = output_root / "EXP-TEST" / "labels"
    labels2 = oracle(output_root, "labels-s1", "--evaluation-seed", "900000", "--num-instances", "4", "--search-seed", "1")

    plain = bc_run_cli(output_root, "bc-plain", "--label-run-dir", str(labels))
    assert list(read_csv_rows(plain / "train_metrics.csv")[0]) == ["epoch", "train_loss", "val_loss"]
    meta = json.loads((plain / "metadata.json").read_text())["behavior_cloning"]
    assert meta["label_groups"] == [] and meta["n_labels_per_target"] == 1 and meta["label_reduction"] == "mean"

    # a K=1 group is the same computation as --label-run-dir
    group1 = bc_run_cli(output_root, "bc-group1", "--label-group", str(labels))
    assert (plain / "train_metrics.csv").read_text() == (group1 / "train_metrics.csv").read_text()
    sa = torch.load(plain / "checkpoints" / "last.pt", map_location="cpu")["model_state_dict"]
    sb = torch.load(group1 / "checkpoints" / "last.pt", map_location="cpu")["model_state_dict"]
    assert all(torch.equal(sa[k], sb[k]) for k in sa)

    nearest = bc_run_cli(
        output_root, "bc-min", "--label-group", f"{labels},{labels2}", "--label-reduction", "min",
    )
    rows = read_csv_rows(nearest / "train_metrics.csv")
    assert list(rows[0]) == ["epoch", "train_loss", "val_loss", "switch_fraction"]
    assert all(0.0 <= float(r["switch_fraction"]) <= 1.0 for r in rows)
    meta = json.loads((nearest / "metadata.json").read_text())["behavior_cloning"]
    assert meta["label_groups"] == [[str(labels), str(labels2)]]
    assert meta["n_labels_per_target"] == 2 and meta["label_reduction"] == "min"
    assert meta["n_train"] + meta["n_val"] == 4  # split by target, not by (target, label)

    for args in (
        ["--label-group", f"{labels},{labels2}"],  # K=2 with the default 'mean'
        ["--label-group", f"{labels},{labels2}", "--label-group", str(labels), "--label-reduction", "min"],  # K differs
        ["--label-group", f"{labels},{labels2}", "--label-run-dir", str(labels), "--label-reduction", "min"],
    ):
        failed = run_cli(
            "behavior_cloning.py", "--config", CONFIG, "--epochs", "1", "--experiment-id", "EXP-TEST",
            "--run-id", "bc-bad", "--output-root", str(output_root), "--allow-dirty", *args, check=False,
        )
        assert failed.returncode != 0 and not (output_root / "EXP-TEST" / "bc-bad").exists()

    centroid = output_root / "centroid"
    run_cli("centroid_labels.py", "--out", str(centroid), "--first", "900000", "--num", "4",
            "--run-dir", str(labels), "--run-dir", str(labels2))
    expected = (np.load(labels / "oracle_actions.npy") + np.load(labels2 / "oracle_actions.npy")) / 2
    np.testing.assert_allclose(np.load(centroid / "oracle_actions.npy"), expected)
    cmeta = json.loads((centroid / "metadata.json").read_text())
    assert cmeta["n_runs"] == 2 and cmeta["target_seeds"] == [900000, 900003]
    bc_centroid = bc_run_cli(output_root, "bc-centroid", "--label-run-dir", str(centroid))
    assert json.loads((bc_centroid / "metadata.json").read_text())["behavior_cloning"]["n_labels"] == 4
