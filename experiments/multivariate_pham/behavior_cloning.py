from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import preload_julia

preload_julia()

import numpy as np

from homotopy_path_learning.config import build_env, build_system_spec, load_config, save_config
from homotopy_path_learning.evaluation.evaluator import evaluate_checkpoint
from homotopy_path_learning.evaluation.metrics import read_rows, write_rows
from homotopy_path_learning.evaluation.oracle import (
    ORACLE_SUMMARY_FIELDS,
    learned_costs_from_evaluation,
    summarize_oracle,
)
from homotopy_path_learning.rl.behavior_cloning import train_behavior_cloning
from homotopy_path_learning.rl.checkpoint import save_checkpoint
from homotopy_path_learning.rl.networks import ActorCritic
from homotopy_path_learning.rl.train_utils import (
    collect_metadata,
    git_value,
    refresh_metadata_output_files,
    require_clean_git,
    seed_everything,
    write_metadata,
)


def _compatibility_key(config) -> dict:
    return {
        "path.bezier_degree": config.path.bezier_degree,
        "path.latent_dim": config.path.latent_dim,
        "path.basis_seed": config.path.basis_seed,
        "path.action_limit": config.path.action_limit,
        "system.degrees": config.system.degrees,
        "system.free_exponents": config.system.free_exponents,
        "sampler.type": config.sampler.type,
        "sampler.bound": config.sampler.bound,
        "tracker": config.tracker,
    }


def load_labels(base_config, label_dirs: list[str]):
    """Load and concatenate (observations, actions, target_seeds) of oracle runs."""

    base_key = _compatibility_key(base_config)
    observations, actions, seeds = [], [], []
    for label_dir in label_dirs:
        directory = Path(label_dir)
        for name in ("config.yaml", "oracle_observations.npy", "oracle_actions.npy", "oracle.csv"):
            if not (directory / name).is_file():
                raise FileNotFoundError(f"label run {directory} is missing {name}.")
        key = _compatibility_key(load_config(directory / "config.yaml"))
        for name, value in base_key.items():
            if key[name] != value:
                raise ValueError(f"label run {directory} differs from the base config in {name}.")
        obs = np.load(directory / "oracle_observations.npy")
        act = np.load(directory / "oracle_actions.npy")
        rows = read_rows(directory / "oracle.csv")
        if not (obs.shape[0] == act.shape[0] == len(rows)):
            raise ValueError(
                f"label run {directory} has inconsistent lengths: observations={obs.shape[0]}, "
                f"actions={act.shape[0]}, rows={len(rows)}."
            )
        observations.append(obs)
        actions.append(act)
        seeds.extend(int(row["target_seed"]) for row in rows)
    return np.concatenate(observations), np.concatenate(actions), seeds


def main() -> None:
    parser = argparse.ArgumentParser(description="Behavior-clone oracle actions into a policy network.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--label-run-dir", action="append", required=True)
    parser.add_argument("--hidden-sizes", default="64,64")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--num-train", type=int, default=None)
    parser.add_argument("--num-instances", type=int, default=None, help="Evaluation instances.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--oracle-run-dir", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    require_clean_git(allow_dirty=args.allow_dirty)
    base = load_config(args.config)
    hidden_sizes = tuple(int(v) for v in args.hidden_sizes.split(",") if v.strip())
    config = replace(
        base,
        experiment=replace(base.experiment, id=args.experiment_id, name=args.run_id, seed=args.seed),
        ppo=replace(base.ppo, hidden_sizes=hidden_sizes),
        output=replace(base.output, root=args.output_root or base.output.root),
    )
    run_dir = Path(config.output.root) / config.experiment.id / args.run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")

    observations, actions, target_seeds = load_labels(base, args.label_run_dir)
    n_labels = int(observations.shape[0])
    if args.num_train is not None:
        observations, actions, target_seeds = (
            observations[: args.num_train],
            actions[: args.num_train],
            target_seeds[: args.num_train],
        )

    run_dir.mkdir(parents=True)
    start_time = datetime.now(timezone.utc).isoformat()
    seed_everything(args.seed)
    env = build_env(config)
    try:
        for index in range(min(3, len(target_seeds))):
            observation, _ = env.reset(seed=target_seeds[index])
            if not np.array_equal(np.asarray(observation, dtype=np.float32), observations[index]):
                raise ValueError(
                    f"stored observation of label {index} (target_seed={target_seeds[index]}) "
                    "does not match the environment observation."
                )
        agent = ActorCritic(
            observation_shape=tuple(env.observation_space.shape),
            action_space=env.action_space,
            hidden_sizes=config.ppo.hidden_sizes,
            activation=config.ppo.activation,
            actor_logstd_init=config.ppo.actor_logstd_init,
            device=args.device,
        )
        result, optimizer = train_behavior_cloning(
            agent,
            observations,
            actions,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            val_fraction=args.val_fraction,
            seed=args.seed,
            device=args.device,
        )
        write_rows(
            run_dir / "train_metrics.csv",
            [dict(item, epoch=int(item["epoch"])) for item in result.history],
            fieldnames=["epoch", "train_loss", "val_loss"],
        )
        save_checkpoint(
            run_dir / "checkpoints" / "last.pt",
            agent=agent,
            optimizer=optimizer,
            config=config,
            system_spec=build_system_spec(config),
            global_step=0,
            update=0,
            observation_shape=tuple(env.observation_space.shape),
            action_space=env.action_space,
            experiment_id=config.experiment.id,
            run_id=args.run_id,
            git_commit=git_value(["rev-parse", "HEAD"]),
        )
        save_config(config, run_dir / "config.yaml")
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
    metadata["behavior_cloning"] = {
        "label_run_dirs": list(args.label_run_dir),
        "n_labels": n_labels,
        "n_train": result.n_train,
        "n_val": result.n_val,
        "hidden_sizes": list(hidden_sizes),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "best_epoch": result.best_epoch,
        "best_val_loss": result.best_val_loss,
    }
    write_metadata(run_dir / "metadata.json", metadata)

    eval_rows = evaluate_checkpoint(
        config=config,
        run_id=args.run_id,
        checkpoint_path=run_dir / "checkpoints" / "last.pt",
        output_path=run_dir / "evaluation" / "evaluation.csv",
        num_instances=args.num_instances,
        device=args.device,
    )
    summaries = []
    if args.oracle_run_dir is not None:
        oracle_rows = read_rows(Path(args.oracle_run_dir) / "oracle.csv")
        learned = learned_costs_from_evaluation(eval_rows, oracle_rows)
        summaries = summarize_oracle(oracle_rows, {"BehaviorCloning": learned}, label_format="{label}")
        write_rows(run_dir / "bc_summary.csv", summaries, fieldnames=ORACLE_SUMMARY_FIELDS)
    refresh_metadata_output_files(run_dir)

    print(
        f"best_epoch={result.best_epoch} best_val_loss={result.best_val_loss:.6f} "
        f"n_train={result.n_train} n_val={result.n_val}"
    )
    for summary in summaries:
        print(
            f"{summary['method']}: mean_cost={summary['mean_cost']:.4f} "
            f"delta_j={summary['delta_j_percent']:+.2f}% positive_rate={summary['positive_rate']:.2f} "
            f"captured={summary['captured_fraction']}"
        )
    print(f"RUN_DIR={run_dir}")


if __name__ == "__main__":
    main()
