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
    resolve_latent_dim,
    summarize_oracle,
    with_latent_dim,
)
from homotopy_path_learning.systems.symmetry import enumerate_symmetries, free_real_matrix, latent_action_matrix
from homotopy_path_learning.rl.behavior_cloning import train_behavior_cloning
from homotopy_path_learning.rl.checkpoint import save_checkpoint
from homotopy_path_learning.rl.train_utils import (
    build_agent,
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


def load_label_group(base_config, dirs: list[str]):
    """Load a group of ``K = len(dirs)`` oracle runs over the same targets.

    Every run is checked like :func:`load_labels` (compatible config, files,
    lengths). The target-seed sets of all runs must coincide (the row order of
    the first run is used; other runs are re-ordered by seed), the observations
    must be exactly equal and the ``linear_cost`` values must be equal.
    Returns ``(observations, actions, seeds)`` with shapes ``(N, obs_dim)``
    (dtype of the first run), ``(N, K, action_dim)`` (float64; candidate ``k``
    comes from ``dirs[k]``) and a list of ``N`` ints. Raises ``ValueError`` on
    any mismatch (including duplicate seeds within a run).
    """

    if len(dirs) == 0:
        raise ValueError("a label group needs at least one run directory.")
    reference = None
    actions = []
    for label_dir in dirs:
        obs, act, seeds = load_labels(base_config, [label_dir])
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"label run {label_dir} contains duplicate target seeds.")
        linear = np.array([float(row["linear_cost"]) for row in read_rows(Path(label_dir) / "oracle.csv")])
        if reference is None:
            reference = (obs, seeds, linear, label_dir)
            actions.append(act)
            continue
        ref_obs, ref_seeds, ref_linear, ref_dir = reference
        if set(seeds) != set(ref_seeds):
            raise ValueError(f"label run {label_dir} has different target seeds than {ref_dir}.")
        position = {seed: index for index, seed in enumerate(seeds)}
        order = np.array([position[seed] for seed in ref_seeds])
        if not np.array_equal(obs[order], ref_obs):
            raise ValueError(f"label run {label_dir} has observations that differ from {ref_dir}.")
        if not np.array_equal(linear[order], ref_linear):
            raise ValueError(f"label run {label_dir} has linear costs that differ from {ref_dir}.")
        actions.append(act[order])
    return reference[0], np.stack(actions, axis=1).astype(np.float64), list(reference[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Behavior-clone oracle actions into a policy network.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--label-run-dir", action="append", default=[])
    parser.add_argument(
        "--label-group",
        action="append",
        default=[],
        help="Comma-separated oracle runs over the same targets (K candidate labels per target); repeatable. "
        "All groups must have the same K.",
    )
    parser.add_argument(
        "--label-reduction",
        choices=("mean", "min"),
        default="mean",
        help="Loss over the K label candidates: 'mean' needs K=1; 'min' uses the nearest candidate.",
    )
    parser.add_argument("--hidden-sizes", default="64,64")
    parser.add_argument("--actor-type", choices=("mlp", "attention", "sequence"), default="mlp")
    parser.add_argument("--attention-d-model", type=int, default=128)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--attention-layers", type=int, default=4)
    parser.add_argument("--attention-ff-dim", type=int, default=256)
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Training epochs (>= 0). 0 = no training: save the freshly initialized policy; "
        "train_metrics.csv then contains only the header and best_epoch=0.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--latent-dim", default=None, help="Integer or 'full'; default keeps the config.")
    parser.add_argument("--augment-symmetry", action="store_true", help="Augment with the symmetry group (needs full latent space).")
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

    label_groups = [[part.strip() for part in group.split(",") if part.strip()] for group in args.label_group]
    if not args.label_run_dir and not label_groups:
        parser.error("one of --label-run-dir or --label-group is required.")
    if any(len(group) == 0 for group in label_groups):
        parser.error("--label-group must list at least one run directory.")
    if len({len(group) for group in label_groups}) > 1:
        parser.error("all --label-group entries must have the same number of runs K.")
    n_labels_per_target = len(label_groups[0]) if label_groups else 1
    if args.label_run_dir and n_labels_per_target != 1:
        parser.error("--label-run-dir (K=1) cannot be combined with --label-group of K>1.")
    if n_labels_per_target > 1 and args.label_reduction == "mean":
        parser.error("--label-group with K>1 requires --label-reduction min.")
    require_clean_git(allow_dirty=args.allow_dirty)
    base = load_config(args.config)
    base = with_latent_dim(base, resolve_latent_dim(base, args.latent_dim))
    hidden_sizes = tuple(int(v) for v in args.hidden_sizes.split(",") if v.strip())
    config = replace(
        base,
        experiment=replace(base.experiment, id=args.experiment_id, name=args.run_id, seed=args.seed),
        ppo=replace(
            base.ppo,
            hidden_sizes=hidden_sizes,
            actor_type=args.actor_type,
            attention_d_model=args.attention_d_model,
            attention_heads=args.attention_heads,
            attention_layers=args.attention_layers,
            attention_ff_dim=args.attention_ff_dim,
        ),
        output=replace(base.output, root=args.output_root or base.output.root),
    )
    run_dir = Path(config.output.root) / config.experiment.id / args.run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")

    if not label_groups:
        observations, actions, target_seeds = load_labels(base, args.label_run_dir)
    else:
        parts = []
        if args.label_run_dir:
            legacy_obs, legacy_act, legacy_seeds = load_labels(base, args.label_run_dir)
            parts.append((legacy_obs, legacy_act[:, None, :], legacy_seeds))
        parts += [load_label_group(base, group) for group in label_groups]
        observations = np.concatenate([part[0] for part in parts])
        actions = np.concatenate([part[1] for part in parts])
        target_seeds = [seed for part in parts for seed in part[2]]
        if n_labels_per_target == 1:
            actions = actions[:, 0, :]
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
        agent = build_agent(config, env, device=args.device)
        for label_dir in list(args.label_run_dir) + [d for group in label_groups for d in group]:
            basis_path = Path(label_dir) / "latent_basis.npy"
            if basis_path.is_file() and not np.allclose(np.load(basis_path), env.parameterization.basis, atol=1e-12):
                raise ValueError(f"latent basis of label run {label_dir} differs from the environment basis.")
        augmentations = None
        if args.augment_symmetry:
            spec = build_system_spec(config)
            basis = env.parameterization.basis
            augmentations = [
                (
                    free_real_matrix(spec, g).astype(np.float32),
                    latent_action_matrix(spec, g, basis, config.path.bezier_degree).astype(np.float32),
                )
                for g in enumerate_symmetries(spec)
            ]
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
            augmentations=augmentations,
            label_reduction=args.label_reduction,
        )
        write_rows(
            run_dir / "train_metrics.csv",
            [dict(item, epoch=int(item["epoch"])) for item in result.history],
            fieldnames=["epoch", "train_loss", "val_loss"] + (["switch_fraction"] if args.label_reduction == "min" else []),
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
        "label_groups": label_groups,
        "n_labels_per_target": n_labels_per_target,
        "label_reduction": args.label_reduction,
        "n_labels": n_labels,
        "n_train": result.n_train,
        "n_val": result.n_val,
        "hidden_sizes": list(hidden_sizes),
        "actor_type": config.ppo.actor_type,
        "attention": config.ppo.attention_kwargs if config.ppo.actor_type in ("attention", "sequence") else None,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "best_epoch": result.best_epoch,
        "best_val_loss": result.best_val_loss,
        "latent_dim": config.path.latent_dim,
        "augment_symmetry": bool(args.augment_symmetry),
        "n_symmetries": result.n_augmentations,
        "clipped_label_fraction": result.clipped_label_fraction,
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
