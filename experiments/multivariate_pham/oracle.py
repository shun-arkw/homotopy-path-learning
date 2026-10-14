from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
from pathlib import Path

from _bootstrap import preload_julia

preload_julia()

import numpy as np
import torch

from homotopy_path_learning.config import build_env, load_config, save_config
from homotopy_path_learning.evaluation.metrics import read_rows, write_rows
from homotopy_path_learning.evaluation.oracle import (
    ORACLE_FIELDS,
    ORACLE_SUMMARY_FIELDS,
    search_constant_action,
    learned_costs_from_evaluation,
    resolve_latent_dim,
    run_oracle,
    summarize_oracle,
    with_latent_dim,
)
from homotopy_path_learning.evaluation.evaluator import (
    _action_from_agent,
    evaluate_policy_rows,
    load_policy_for_config,
)
from homotopy_path_learning.evaluation.seeds import read_seed_list, seed_file_record
from homotopy_path_learning.evaluation.metrics import EVALUATION_FIELDS
from homotopy_path_learning.rl.train_utils import (
    collect_metadata,
    refresh_metadata_output_files,
    require_clean_git,
    write_metadata,
)


class ConstantAgent:
    """Policy that returns the same action for every observation."""

    def __init__(self, action: np.ndarray) -> None:
        self.action = torch.as_tensor(np.asarray(action, dtype=np.float32))

    def deterministic_action(self, observation: torch.Tensor) -> torch.Tensor:
        """Return the action repeated to shape ``(batch, action_dim)`` (float32)."""

        batch = 1 if observation.ndim == 1 else int(observation.shape[0])
        return self.action.to(observation.device).unsqueeze(0).repeat(batch, 1)


def run_constant_mode(args: argparse.Namespace, config, run_dir: Path, start_time: str) -> None:
    """Search one constant action on training targets and evaluate it."""

    search = search_constant_action(
        config,
        train_seed=args.train_seed,
        num_train=args.num_train,
        budget=args.budget,
        n_random=args.n_random,
        search_seed=args.search_seed,
    )
    np.save(run_dir / "constant_action.npy", search.action)
    np.save(run_dir / "constant_costs.npy", search.costs)
    rows = evaluate_policy_rows(
        config=config,
        run_id=args.run_id,
        agent=ConstantAgent(search.action),
        device="cpu",
        methods=("Linear", "LearnedBezier"),
        num_instances=args.num_instances,
        evaluation_seed=args.evaluation_seed,
    )
    for row in rows:
        if row["method"] == "LearnedBezier":
            row["method"] = "ConstantBezier"
    write_rows(run_dir / "evaluation" / "evaluation.csv", rows, fieldnames=EVALUATION_FIELDS)

    summaries = []
    if args.oracle_run_dir is not None:
        oracle_rows = read_rows(Path(args.oracle_run_dir) / "oracle.csv")
        learned = learned_costs_from_evaluation(rows, oracle_rows, method="ConstantBezier")
        summaries = summarize_oracle(oracle_rows, {"ConstantBezier": learned}, label_format="{label}")
        write_rows(run_dir / "constant_summary.csv", summaries, fieldnames=ORACLE_SUMMARY_FIELDS)

    env = build_env(config)
    try:
        metadata = collect_metadata(
            config=config,
            run_id=args.run_id,
            run_dir=run_dir,
            status="Completed",
            start_time_utc=start_time,
            end_time_utc=datetime.now(timezone.utc).isoformat(),
            config_path=args.config,
            env=env,
        )
    finally:
        env.close()
    metadata["constant"] = {
        "train_seed": args.train_seed,
        "num_train": args.num_train,
        "budget": args.budget,
        "n_random": args.n_random,
        "search_seed": args.search_seed,
        "train_mean_cost": search.train_mean_cost,
        "train_linear_mean_cost": search.train_linear_mean_cost,
        "oracle_run_dir": args.oracle_run_dir,
    }
    write_metadata(run_dir / "metadata.json", metadata)
    refresh_metadata_output_files(run_dir)

    print(f"train: linear_mean={search.train_linear_mean_cost:.4f} constant_mean={search.train_mean_cost:.4f}")
    for summary in summaries:
        print(
            f"{summary['method']}: mean_cost={summary['mean_cost']:.4f} "
            f"delta_j={summary['delta_j_percent']:+.2f}% positive_rate={summary['positive_rate']:.2f} "
            f"captured={summary['captured_fraction']}"
        )
    print(f"RUN_DIR={run_dir}")


def sha256_file(path: str | Path) -> str:
    """Return the hex SHA-256 digest of the file at ``path``."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


INITIAL_ACTION_RULE = "deterministic policy action (tanh of actor mean), float32 cast to float64"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Directly optimize the latent action per evaluation instance (oracle baseline)."
    )
    parser.add_argument("--config", required=True, help="Config of the PPO run to compare against.")
    parser.add_argument("--experiment-id", required=True, help="Experiment ID of the oracle run.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--latent-dim", default=None, help="Integer or 'full'; default keeps the config.")
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--n-random", type=int, default=100)
    parser.add_argument("--search-seed", type=int, default=0)
    parser.add_argument("--num-instances", type=int, default=None)
    parser.add_argument("--evaluation-seed", type=int, default=None)
    parser.add_argument(
        "--target-seeds-file",
        default=None,
        help="Per-instance mode: seed-list file (one seed per line); excludes --evaluation-seed and --num-instances.",
    )
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help=(
            "Per-instance mode: policy checkpoint whose deterministic action is evaluated as the second "
            "search point (after the zero action); requires --n-random >= 2. Its cost is "
            "oracle_cost_traces.npy[:, 1]; output file formats are unchanged."
        ),
    )
    parser.add_argument(
        "--initial-sigma",
        type=float,
        default=0.3,
        help="Initial evolution-strategy step size relative to the action limit (default 0.3).",
    )
    parser.add_argument("--mode", choices=["per-instance", "constant"], default="per-instance")
    parser.add_argument("--train-seed", type=int, default=None, help="Constant mode: base seed of training targets.")
    parser.add_argument("--num-train", type=int, default=200, help="Constant mode: number of training targets.")
    parser.add_argument("--oracle-run-dir", default=None, help="Constant mode: per-instance oracle run to compare to.")
    parser.add_argument("--compare-run-dir", action="append", default=[])
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    if args.mode == "constant" and args.train_seed is None:
        parser.error("--train-seed is required in constant mode.")
    if args.init_checkpoint is not None and args.mode != "per-instance":
        parser.error("--init-checkpoint is only supported in per-instance mode.")
    target_seeds = None
    if args.target_seeds_file is not None:
        if args.mode != "per-instance":
            parser.error("--target-seeds-file is only supported in per-instance mode.")
        if args.evaluation_seed is not None or args.num_instances is not None:
            parser.error("--target-seeds-file cannot be combined with --evaluation-seed or --num-instances.")
        target_seeds = read_seed_list(args.target_seeds_file)
    require_clean_git(allow_dirty=args.allow_dirty)
    source_config = load_config(args.config)
    latent_dim = resolve_latent_dim(source_config, args.latent_dim)
    config = with_latent_dim(source_config, latent_dim)
    config = replace(
        config,
        experiment=replace(config.experiment, id=args.experiment_id, name=args.run_id),
        output=replace(config.output, root=args.output_root or config.output.root),
    )
    run_dir = Path(config.output.root) / config.experiment.id / args.run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    save_config(config, run_dir / "config.yaml")
    start_time = datetime.now(timezone.utc).isoformat()

    if args.mode == "constant":
        run_constant_mode(args, config, run_dir, start_time)
        return

    initial_action_fn = None
    if args.init_checkpoint is not None:
        agent = load_policy_for_config(checkpoint_path=args.init_checkpoint, config=config, device="cpu")

        def initial_action_fn(observation: np.ndarray) -> np.ndarray:
            return _action_from_agent(agent, observation, device="cpu").astype(np.float64)

    result = run_oracle(
        config,
        budget=args.budget,
        n_random=args.n_random,
        search_seed=args.search_seed,
        num_instances=args.num_instances,
        evaluation_seed=args.evaluation_seed,
        target_seeds=target_seeds,
        initial_action_fn=initial_action_fn,
        initial_sigma=args.initial_sigma,
    )
    write_rows(run_dir / "oracle.csv", result.rows, fieldnames=ORACLE_FIELDS)
    np.save(run_dir / "oracle_actions.npy", result.best_actions)
    np.save(run_dir / "oracle_cost_traces.npy", result.cost_traces)
    np.save(run_dir / "latent_basis.npy", result.basis)
    np.save(run_dir / "oracle_observations.npy", result.observations)

    learned = {}
    for compare_dir in args.compare_run_dir:
        compare_path = Path(compare_dir)
        evaluation_rows = read_rows(compare_path / "evaluation" / "evaluation.csv")
        learned[compare_path.name] = learned_costs_from_evaluation(evaluation_rows, result.rows)
    summaries = summarize_oracle(result.rows, learned)
    write_rows(run_dir / "oracle_summary.csv", summaries, fieldnames=ORACLE_SUMMARY_FIELDS)

    env = build_env(config)
    try:
        metadata = collect_metadata(
            config=config,
            run_id=args.run_id,
            run_dir=run_dir,
            status="Completed",
            start_time_utc=start_time,
            end_time_utc=datetime.now(timezone.utc).isoformat(),
            config_path=args.config,
            env=env,
        )
    finally:
        env.close()
    metadata["oracle"] = {
        "source_config": args.config,
        "latent_dim": latent_dim,
        "budget": args.budget,
        "n_random": args.n_random,
        "search_seed": args.search_seed,
        "evaluation_seed": args.evaluation_seed,
        "search_rng_rule": "default_rng([search_seed, target_seed])",
        "compare_run_dirs": list(args.compare_run_dir),
        "init_checkpoint": args.init_checkpoint,
        "init_checkpoint_sha256": None if args.init_checkpoint is None else sha256_file(args.init_checkpoint),
        "initial_sigma": args.initial_sigma,
        "initial_action_rule": INITIAL_ACTION_RULE if args.init_checkpoint is not None else None,
    }
    if args.target_seeds_file is not None:
        metadata["oracle"]["target_seeds_file"] = seed_file_record(args.target_seeds_file)
    write_metadata(run_dir / "metadata.json", metadata)
    refresh_metadata_output_files(run_dir)

    for summary in summaries:
        print(
            f"{summary['method']}: mean_cost={summary['mean_cost']:.4f} "
            f"delta_j={summary['delta_j_percent']:+.2f}% positive_rate={summary['positive_rate']:.2f} "
            f"captured={summary['captured_fraction']}"
        )
    print(f"RUN_DIR={run_dir}")


if __name__ == "__main__":
    main()
