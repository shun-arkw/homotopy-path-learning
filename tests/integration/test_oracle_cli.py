from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


pytestmark = [pytest.mark.integration, pytest.mark.slow]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_oracle_cli_smoke(tmp_path) -> None:
    env = os.environ.copy()
    env["PYTHON_JULIACALL_EXE"] = env.get("PYTHON_JULIACALL_EXE", "/usr/local/bin/julia")
    env["PYTHON_JULIACALL_PROJECT"] = env.get("PYTHON_JULIACALL_PROJECT", str(Path.cwd() / "julia"))
    output_root = tmp_path / "outputs"
    run_dir = output_root / "EXP-TEST" / "oracle-smoke"

    subprocess.run(
        [
            sys.executable,
            "experiments/multivariate_pham/oracle.py",
            "--config",
            "experiments/multivariate_pham/configs/exp-0004-smoke.yaml",
            "--experiment-id",
            "EXP-TEST",
            "--run-id",
            "oracle-smoke",
            "--latent-dim",
            "full",
            "--budget",
            "6",
            "--n-random",
            "3",
            "--num-instances",
            "2",
            "--output-root",
            str(output_root),
            "--allow-dirty",
        ],
        check=True,
        env=env,
    )

    rows = read_csv_rows(run_dir / "oracle.csv")
    assert len(rows) == 2
    for row in rows:
        assert int(row["latent_dim"]) == 12
        assert int(row["action_dim"]) == 24
        assert int(row["n_evaluations"]) == 6
        # The zero action is the linear path, so it must reproduce the linear cost.
        assert float(row["zero_action_cost"]) == float(row["linear_cost"])
        assert float(row["oracle_cost"]) <= float(row["linear_cost"])
    summary = read_csv_rows(run_dir / "oracle_summary.csv")
    assert [row["method"] for row in summary] == ["Linear", "Oracle"]
    assert np.load(run_dir / "oracle_actions.npy").shape == (2, 24)
    assert np.load(run_dir / "oracle_cost_traces.npy").shape == (2, 6)
    assert np.load(run_dir / "latent_basis.npy").shape == (12, 12)
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["experiment_id"] == "EXP-TEST"
    assert metadata["oracle"]["latent_dim"] == 12
    assert "oracle.csv" in metadata["output_files"]


CONFIG = "experiments/multivariate_pham/configs/exp-0004-smoke.yaml"


def cli_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHON_JULIACALL_EXE"] = env.get("PYTHON_JULIACALL_EXE", "/usr/local/bin/julia")
    env["PYTHON_JULIACALL_PROJECT"] = env.get("PYTHON_JULIACALL_PROJECT", str(Path.cwd() / "julia"))
    return env


def run_cli(script: str, *arguments: str) -> None:
    subprocess.run(
        [sys.executable, f"experiments/multivariate_pham/{script}", *arguments],
        check=True,
        env=cli_env(),
    )


def run_oracle_cli(output_root: Path, run_id: str, *extra: str) -> Path:
    run_cli(
        "oracle.py",
        "--config",
        CONFIG,
        "--experiment-id",
        "EXP-TEST",
        "--run-id",
        run_id,
        "--budget",
        "6",
        "--n-random",
        "3",
        "--output-root",
        str(output_root),
        "--allow-dirty",
        *extra,
    )
    return output_root / "EXP-TEST" / run_id


def test_oracle_evaluation_seed_saves_observations(tmp_path) -> None:
    run_dir = run_oracle_cli(
        tmp_path / "outputs", "labels", "--evaluation-seed", "900000", "--num-instances", "4"
    )

    rows = read_csv_rows(run_dir / "oracle.csv")
    assert [int(row["target_seed"]) for row in rows] == [900000, 900001, 900002, 900003]
    assert np.load(run_dir / "oracle_observations.npy").shape == (4, 12)
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["oracle"]["evaluation_seed"] == 900000


def test_behavior_cloning_constant_and_modes_cli(tmp_path) -> None:
    output_root = tmp_path / "outputs"
    labels = run_oracle_cli(output_root, "labels", "--evaluation-seed", "900000", "--num-instances", "4")
    oracle_a = run_oracle_cli(output_root, "oracle-a", "--num-instances", "2")
    oracle_b = run_oracle_cli(output_root, "oracle-b", "--num-instances", "2", "--search-seed", "1")

    run_cli(
        "behavior_cloning.py",
        "--config",
        CONFIG,
        "--label-run-dir",
        str(labels),
        "--hidden-sizes",
        "16,16",
        "--epochs",
        "3",
        "--val-fraction",
        "0.25",
        "--num-instances",
        "2",
        "--experiment-id",
        "EXP-TEST",
        "--run-id",
        "bc",
        "--oracle-run-dir",
        str(oracle_a),
        "--output-root",
        str(output_root),
        "--allow-dirty",
    )
    bc_dir = output_root / "EXP-TEST" / "bc"
    assert (bc_dir / "checkpoints" / "last.pt").is_file()
    history = read_csv_rows(bc_dir / "train_metrics.csv")
    assert [int(row["epoch"]) for row in history] == [1, 2, 3]
    assert [row["method"] for row in read_csv_rows(bc_dir / "bc_summary.csv")] == [
        "Linear",
        "Oracle",
        "BehaviorCloning",
    ]
    metadata = json.loads((bc_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["behavior_cloning"]["n_train"] + metadata["behavior_cloning"]["n_val"] == 4
    assert "evaluation/evaluation.csv" in metadata["output_files"]

    run_cli(
        "oracle.py",
        "--config",
        CONFIG,
        "--mode",
        "constant",
        "--train-seed",
        "900000",
        "--num-train",
        "3",
        "--budget",
        "5",
        "--n-random",
        "2",
        "--num-instances",
        "2",
        "--oracle-run-dir",
        str(oracle_a),
        "--experiment-id",
        "EXP-TEST",
        "--run-id",
        "const",
        "--output-root",
        str(output_root),
        "--allow-dirty",
    )
    const_dir = output_root / "EXP-TEST" / "const"
    assert np.load(const_dir / "constant_action.npy").shape == (8,)
    assert np.load(const_dir / "constant_costs.npy").shape == (5,)
    assert [row["method"] for row in read_csv_rows(const_dir / "constant_summary.csv")] == [
        "Linear",
        "Oracle",
        "ConstantBezier",
    ]
    constant_metadata = json.loads((const_dir / "metadata.json").read_text(encoding="utf-8"))
    assert constant_metadata["constant"]["num_train"] == 3

    modes_dir = tmp_path / "modes"
    run_cli(
        "oracle_modes.py",
        "--oracle-run-dir",
        str(oracle_a),
        "--oracle-run-dir",
        str(oracle_b),
        "--num-instances",
        "2",
        "--tolerance",
        "0.5",
        "--output-dir",
        str(modes_dir),
    )
    modes = read_csv_rows(modes_dir / "modes.csv")
    assert len(modes) == 2
    assert all(int(row["n_runs"]) == 2 for row in modes)
    assert (modes_dir / "modes_summary.csv").is_file()
    assert (modes_dir / "metadata.json").is_file()


def test_full_latent_symmetry_bc_and_symmetry_check_cli(tmp_path) -> None:
    output_root = tmp_path / "outputs"
    labels = run_oracle_cli(
        output_root, "labels-full", "--latent-dim", "full", "--evaluation-seed", "900000", "--num-instances", "4"
    )
    assert np.load(labels / "oracle_actions.npy").shape == (4, 24)
    oracle_full = run_oracle_cli(output_root, "oracle-full", "--latent-dim", "full", "--num-instances", "2")

    run_cli(
        "behavior_cloning.py",
        "--config",
        CONFIG,
        "--label-run-dir",
        str(labels),
        "--latent-dim",
        "full",
        "--augment-symmetry",
        "--hidden-sizes",
        "16,16",
        "--epochs",
        "2",
        "--val-fraction",
        "0.25",
        "--num-instances",
        "2",
        "--experiment-id",
        "EXP-TEST",
        "--run-id",
        "bc-sym",
        "--oracle-run-dir",
        str(oracle_full),
        "--output-root",
        str(output_root),
        "--allow-dirty",
    )
    metadata = json.loads((output_root / "EXP-TEST" / "bc-sym" / "metadata.json").read_text(encoding="utf-8"))
    bc = metadata["behavior_cloning"]
    assert bc["augment_symmetry"] is True
    assert bc["n_symmetries"] == 16
    assert 0.0 <= bc["clipped_label_fraction"] <= 1.0

    check_dir = tmp_path / "symcheck"
    run_cli(
        "symmetry_check.py",
        "--config",
        CONFIG,
        "--num-instances",
        "2",
        "--random-actions",
        "1",
        "--output-dir",
        str(check_dir),
        "--allow-dirty",
    )
    rows = read_csv_rows(check_dir / "symmetry_check.csv")
    assert len(rows) == 2 * 2 * 15
    summary = {row["element_type"]: row for row in read_csv_rows(check_dir / "symmetry_summary.csv")}
    assert int(summary["overall"]["n_pairs"]) == len(rows)
    assert (check_dir / "metadata.json").is_file()


def test_finetune_cli(tmp_path) -> None:
    output_root = tmp_path / "outputs"
    labels = run_oracle_cli(
        output_root, "labels-ft", "--latent-dim", "full", "--evaluation-seed", "900000", "--num-instances", "4"
    )
    oracle_full = run_oracle_cli(output_root, "oracle-ft", "--latent-dim", "full", "--num-instances", "2")
    run_cli(
        "behavior_cloning.py",
        "--config",
        CONFIG,
        "--label-run-dir",
        str(labels),
        "--latent-dim",
        "full",
        "--hidden-sizes",
        "16,16",
        "--epochs",
        "2",
        "--val-fraction",
        "0.25",
        "--num-instances",
        "2",
        "--experiment-id",
        "EXP-TEST",
        "--run-id",
        "bc-init",
        "--output-root",
        str(output_root),
        "--allow-dirty",
    )
    init_dir = output_root / "EXP-TEST" / "bc-init"
    for run_id, extra in (
        ("ft-plain", []),
        ("ft-bc", ["--bc-weight", "1", "--label-run-dir", str(labels), "--augment-symmetry", "--bc-batch-size", "4"]),
    ):
        run_cli(
            "finetune.py",
            "--config",
            CONFIG,
            "--latent-dim",
            "full",
            "--init-run-dir",
            str(init_dir),
            "--iterations",
            "3",
            "--batch-problems",
            "2",
            "--pairs",
            "1",
            "--val-instances",
            "2",
            "--val-interval",
            "1",
            "--num-instances",
            "2",
            "--experiment-id",
            "EXP-TEST",
            "--run-id",
            run_id,
            "--oracle-run-dir",
            str(oracle_full),
            "--output-root",
            str(output_root),
            "--allow-dirty",
            *extra,
        )
        run_dir = output_root / "EXP-TEST" / run_id
        val = read_csv_rows(run_dir / "val_metrics.csv")
        assert [int(row["iteration"]) for row in val] == [0, 1, 2, 3]
        assert len(read_csv_rows(run_dir / "finetune_metrics.csv")) == 3
        assert (run_dir / "checkpoints" / "last.pt").is_file()
        assert [row["method"] for row in read_csv_rows(run_dir / "finetune_summary.csv")] == [
            "Linear",
            "Oracle",
            "FineTuned",
        ]
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["finetune"]["best_val_delta_j"] <= metadata["finetune"]["initial_val_delta_j"]
        assert metadata["finetune"]["bc_weight"] == (1.0 if extra else 0.0)


def test_oracle_cli_init_checkpoint(tmp_path) -> None:
    import gymnasium as gym
    import torch

    from homotopy_path_learning.config import load_config
    from homotopy_path_learning.rl.checkpoint import save_checkpoint
    from homotopy_path_learning.rl.networks import ActorCritic
    from homotopy_path_learning.systems import build_pham_spec

    config = load_config(CONFIG)
    space = gym.spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
    agent = ActorCritic(
        observation_shape=(12,),
        action_space=space,
        hidden_sizes=tuple(config.ppo.hidden_sizes),
        actor_logstd_init=config.ppo.actor_logstd_init,
    )
    checkpoint_path = tmp_path / "model.pt"
    save_checkpoint(
        checkpoint_path,
        agent=agent,
        optimizer=torch.optim.Adam(agent.parameters()),
        config=config,
        system_spec=build_pham_spec(config.system.degrees, config.system.free_exponents),
        global_step=1,
        update=1,
        observation_shape=(12,),
        action_space=space,
        experiment_id=config.experiment.id,
        run_id="run",
        git_commit="abc",
    )

    run_dir = run_oracle_cli(
        tmp_path / "outputs", "init", "--num-instances", "2", "--init-checkpoint", str(checkpoint_path)
    )

    observations = np.load(run_dir / "oracle_observations.npy")
    traces = np.load(run_dir / "oracle_cost_traces.npy")
    assert traces.shape == (2, 6)
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["oracle"]["init_checkpoint"] == str(checkpoint_path)
    assert len(metadata["oracle"]["init_checkpoint_sha256"]) == 64
    assert metadata["oracle"]["initial_sigma"] == 0.3
    assert metadata["oracle"]["initial_action_rule"] is not None
    for row, trace in zip(read_csv_rows(run_dir / "oracle.csv"), traces):
        assert float(row["oracle_cost"]) <= min(trace[0], trace[1])
    with torch.no_grad():
        policy_actions = agent.deterministic_action(torch.as_tensor(observations)).numpy()
    assert policy_actions.shape == (2, 8)

    plain = run_oracle_cli(tmp_path / "outputs", "plain", "--num-instances", "2")
    plain_metadata = json.loads((plain / "metadata.json").read_text(encoding="utf-8"))
    assert plain_metadata["oracle"]["init_checkpoint"] is None
    assert plain_metadata["oracle"]["init_checkpoint_sha256"] is None
    assert plain_metadata["oracle"]["initial_action_rule"] is None
