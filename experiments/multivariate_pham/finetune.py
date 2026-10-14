from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import time

from _bootstrap import preload_julia

preload_julia()

import numpy as np
import torch

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
from homotopy_path_learning.rl.behavior_cloning import predicted_actions
from homotopy_path_learning.rl.checkpoint import (
    build_agent_from_checkpoint,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint,
)
from homotopy_path_learning.evaluation.seeds import read_seed_list, seed_file_record, train_problem_seed
from homotopy_path_learning.rl.cost_finetune import (
    antithetic_action_gradients,
    antithetic_candidates,
    antithetic_gradients_from_costs,
    finetune_step,
    sigma_at,
)
from homotopy_path_learning.rl.parallel import EnvPool
from homotopy_path_learning.rl.train_utils import (
    collect_metadata,
    git_value,
    refresh_metadata_output_files,
    require_clean_git,
    seed_everything,
    write_metadata,
)
from homotopy_path_learning.systems.symmetry import enumerate_symmetries, free_real_matrix, latent_action_matrix


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


def _attention_settings(checkpoint: dict) -> dict:
    kwargs = checkpoint.get("attention_kwargs") or {}
    names = {"d_model": "attention_d_model", "n_heads": "attention_heads", "n_layers": "attention_layers", "ff_dim": "attention_ff_dim"}
    return {names[key]: int(value) for key, value in kwargs.items() if key in names}


def _require_compatible(base, directory: Path) -> None:
    key = _compatibility_key(load_config(directory / "config.yaml"))
    for name, value in _compatibility_key(base).items():
        if key[name] != value:
            raise ValueError(f"run {directory} differs from the base config in {name}.")


def load_labels(base, label_dirs: list[str]):
    observations, actions = [], []
    for label_dir in label_dirs:
        directory = Path(label_dir)
        for name in ("config.yaml", "oracle_observations.npy", "oracle_actions.npy"):
            if not (directory / name).is_file():
                raise FileNotFoundError(f"label run {directory} is missing {name}.")
        _require_compatible(base, directory)
        obs = np.load(directory / "oracle_observations.npy")
        act = np.load(directory / "oracle_actions.npy")
        if obs.shape[0] != act.shape[0]:
            raise ValueError(f"label run {directory} has inconsistent lengths.")
        observations.append(obs)
        actions.append(act)
    return np.concatenate(observations).astype(np.float32), np.concatenate(actions).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a BC policy by minimizing the tracking cost.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--latent-dim", default=None, help="Integer or 'full'; default keeps the config.")
    parser.add_argument("--init-run-dir", required=True, help="Behavior-cloning run providing checkpoints/last.pt.")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--batch-problems", type=int, default=32)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=0.1, help="Perturbation scale (sigma_start when --sigma-end is given).")
    parser.add_argument(
        "--sigma-end",
        type=float,
        default=None,
        help="Final perturbation scale of a linear decay from --sigma (0 < sigma_end <= sigma). Default: constant sigma.",
    )
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--bc-weight", type=float, default=0.0)
    parser.add_argument("--label-run-dir", action="append", default=[])
    parser.add_argument("--augment-symmetry", action="store_true", help="Symmetry augmentation of the BC term only.")
    parser.add_argument("--bc-batch-size", type=int, default=256)
    parser.add_argument("--train-seed", type=int, default=6000000)
    parser.add_argument("--val-seed", type=int, default=5000000)
    parser.add_argument(
        "--train-seeds-file",
        default=None,
        help="Seed-list file; problem b of iteration k (1-based) uses element (k-1)*batch_problems+b. Replaces --train-seed.",
    )
    parser.add_argument(
        "--train-pool-size",
        type=int,
        default=None,
        help="Use a fixed pool of N training targets (seeds train_seed..train_seed+N-1, or the first N of "
        "--train-seeds-file) cyclically; also writes pool_metrics.csv. Default: a new target every time.",
    )
    parser.add_argument(
        "--val-seeds-file",
        default=None,
        help="Seed-list file; the first --val-instances entries are the validation seeds. Replaces --val-seed.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Worker processes for tracking (1 = sequential).")
    parser.add_argument("--val-instances", type=int, default=200)
    parser.add_argument("--val-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--oracle-run-dir", default=None)
    parser.add_argument("--num-instances", type=int, default=None, help="Evaluation instances.")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    if args.bc_weight > 0 and not args.label_run_dir:
        parser.error("--label-run-dir is required when --bc-weight > 0.")
    if min(args.iterations, args.batch_problems, args.pairs, args.val_instances, args.val_interval, args.bc_batch_size) <= 0:
        parser.error("iteration, batch, pair, validation and interval counts must be positive.")
    if args.workers < 1:
        parser.error("--workers must be at least 1.")
    if args.train_pool_size is not None and args.train_pool_size < 1:
        parser.error("--train-pool-size must be at least 1.")
    if args.sigma_end is not None and not 0.0 < args.sigma_end <= args.sigma:
        parser.error("--sigma-end must satisfy 0 < sigma_end <= sigma.")
    train_seeds = val_seeds = None
    if args.train_seeds_file is not None:
        train_seeds = read_seed_list(args.train_seeds_file)
        needed = args.iterations * args.batch_problems if args.train_pool_size is None else args.train_pool_size
        if len(train_seeds) < needed:
            what = "iterations*batch_problems" if args.train_pool_size is None else "--train-pool-size"
            parser.error(f"--train-seeds-file has {len(train_seeds)} seeds; {what} = {needed} are needed.")
    if args.val_seeds_file is not None:
        val_seeds = read_seed_list(args.val_seeds_file)
        if len(val_seeds) < args.val_instances:
            parser.error(f"--val-seeds-file has {len(val_seeds)} seeds; --val-instances = {args.val_instances} are needed.")
        val_seeds = val_seeds[: args.val_instances]

    def problem_seed(iteration: int, b: int) -> int:
        return train_problem_seed(
            iteration,
            b,
            batch_problems=args.batch_problems,
            train_seed=args.train_seed,
            train_seeds=train_seeds,
            pool_size=args.train_pool_size,
        )

    def val_problem_seed(i: int) -> int:
        return val_seeds[i] if val_seeds is not None else args.val_seed + i
    require_clean_git(allow_dirty=args.allow_dirty)
    base = load_config(args.config)
    base = with_latent_dim(base, resolve_latent_dim(base, args.latent_dim))
    init_dir = Path(args.init_run_dir)
    _require_compatible(base, init_dir)
    checkpoint = load_checkpoint(init_dir / "checkpoints" / "last.pt", map_location=args.device)
    ppo_ckpt = checkpoint["ppo_config"]
    config = replace(
        base,
        experiment=replace(base.experiment, id=args.experiment_id, name=args.run_id, seed=args.seed),
        ppo=replace(
            base.ppo,
            hidden_sizes=tuple(int(v) for v in ppo_ckpt["hidden_sizes"]),
            activation=str(ppo_ckpt["activation"]),
            actor_logstd_init=float(ppo_ckpt["actor_logstd_init"]),
            # Actor type and attention size are inherited from the init checkpoint
            # (checkpoints without these keys are MLP).
            actor_type=str(checkpoint.get("actor_type", ppo_ckpt.get("actor_type", "mlp"))),
            **_attention_settings(checkpoint),
        ),
        output=replace(base.output, root=args.output_root or base.output.root),
    )
    run_dir = Path(config.output.root) / config.experiment.id / args.run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    labels = None
    if args.bc_weight > 0:
        labels = load_labels(base, args.label_run_dir)
    run_dir.mkdir(parents=True)
    start_time = datetime.now(timezone.utc).isoformat()
    seed_everything(args.seed)
    rng = np.random.default_rng([args.seed, 17])
    generator = torch.Generator().manual_seed(int(args.seed))

    env = build_env(config)
    pool = EnvPool(config, args.workers) if args.workers > 1 else None
    try:
        spec = build_system_spec(config)
        validate_checkpoint(
            checkpoint,
            config=config,
            system_spec=spec,
            observation_shape=tuple(env.observation_space.shape),
            action_space=env.action_space,
        )
        agent = build_agent_from_checkpoint(checkpoint, action_space=env.action_space, device=args.device)
        agent.train()
        optimizer = torch.optim.Adam(agent.actor_mean.parameters(), lr=args.learning_rate)
        low = agent.distribution.low.detach().cpu().numpy().astype(np.float64)
        high = agent.distribution.high.detach().cpu().numpy().astype(np.float64)

        augmentations = None
        if args.augment_symmetry:
            basis = env.parameterization.basis
            augmentations = [
                (
                    free_real_matrix(spec, g).astype(np.float32),
                    latent_action_matrix(spec, g, basis, config.path.bezier_degree).astype(np.float32),
                )
                for g in enumerate_symmetries(spec)
            ]

        def policy_actions(observations: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                return predicted_actions(agent, torch.as_tensor(observations, dtype=torch.float32, device=args.device)).cpu().numpy().astype(np.float64)

        val_contexts, val_obs = [], []
        if pool is not None:
            val_obs, val_contexts = pool.reset_many([val_problem_seed(i) for i in range(args.val_instances)])
        else:
            for i in range(args.val_instances):
                observation, _ = env.reset(seed=val_problem_seed(i))
                val_contexts.append(env.current_episode_context())
                val_obs.append(np.asarray(observation, dtype=np.float32))
        val_obs_array = np.stack(val_obs)
        val_linear = np.array([float(c.linear_cost.mean) for c in val_contexts])

        def evaluate_policy_on(obs_array: np.ndarray, contexts_: list, linear: np.ndarray) -> tuple[float, float]:
            actions = policy_actions(obs_array)
            if pool is not None:
                costs = np.array(pool.mean_costs(contexts_, list(actions)))
            else:
                costs = np.array(
                    [float(env.evaluate_action_against_context(c, a).tracking_cost.mean) for c, a in zip(contexts_, actions)]
                )
            return (float(costs.mean()) - float(linear.mean())) / float(linear.mean()) * 100.0, float(np.mean(costs < linear))

        def validate() -> tuple[float, float]:
            return evaluate_policy_on(val_obs_array, val_contexts, val_linear)

        # Fixed training pool (diagnostic only; never used for model selection).
        pool_contexts, pool_obs_array, pool_linear = [], None, None
        pool_rows = []
        if args.train_pool_size is not None:
            pool_seeds = [
                train_problem_seed(
                    1, index, batch_problems=args.train_pool_size, train_seed=args.train_seed, train_seeds=train_seeds
                )
                for index in range(args.train_pool_size)
            ]
            pool_obs = []
            if pool is not None:
                pool_obs, pool_contexts = pool.reset_many(pool_seeds)
            else:
                for seed_ in pool_seeds:
                    observation, _ = env.reset(seed=seed_)
                    pool_contexts.append(env.current_episode_context())
                    pool_obs.append(np.asarray(observation, dtype=np.float32))
            pool_obs_array = np.stack(pool_obs)
            pool_linear = np.array([float(c.linear_cost.mean) for c in pool_contexts])

        val_rows = []
        initial_delta, positive = validate()
        val_rows.append({"iteration": 0, "val_delta_j": initial_delta, "val_positive_rate": positive})
        best_delta, best_iteration = initial_delta, 0
        best_state = deepcopy(agent.actor_mean.state_dict())
        metrics_rows = []
        timer = time.time()
        for iteration in range(1, args.iterations + 1):
            contexts, observations = [], []
            if pool is not None:
                observations, contexts = pool.reset_many(
                    [problem_seed(iteration, b) for b in range(args.batch_problems)]
                )
            else:
                for b in range(args.batch_problems):
                    observation, _ = env.reset(seed=problem_seed(iteration, b))
                    contexts.append(env.current_episode_context())
                    observations.append(np.asarray(observation, dtype=np.float32))
            observations = np.stack(observations)
            means = policy_actions(observations)
            sigma = sigma_at(iteration, iterations=args.iterations, sigma_start=args.sigma, sigma_end=args.sigma_end)

            if pool is not None:
                # Candidates are drawn first (same RNG order as the sequential
                # estimator), then all (context, action) pairs are tracked.
                eps, plus, minus = antithetic_candidates(
                    means, sigma=sigma, n_pairs=args.pairs, rng=rng, low=low, high=high
                )
                items_contexts, items_actions = [], []
                for b in range(args.batch_problems):
                    for p in range(args.pairs):
                        items_contexts += [contexts[b], contexts[b]]
                        items_actions += [plus[b, p], minus[b, p]]
                raw = np.array(pool.mean_costs(items_contexts, items_actions)).reshape(
                    args.batch_problems, args.pairs, 2
                )
                linear = np.array([float(c.linear_cost.mean) for c in contexts])[:, None]
                gradients, info = antithetic_gradients_from_costs(
                    eps, raw[:, :, 0] / linear, raw[:, :, 1] / linear, sigma=sigma
                )
            else:

                def cost_fn(b: int, action: np.ndarray) -> float:
                    return float(env.evaluate_action_against_context(contexts[b], action).tracking_cost.mean) / float(
                        contexts[b].linear_cost.mean
                    )

                gradients, info = antithetic_action_gradients(
                    cost_fn, means, sigma=sigma, n_pairs=args.pairs, rng=rng, low=low, high=high
                )
            bc_kwargs = {}
            if labels is not None:
                pick = rng.integers(0, labels[0].shape[0], size=args.bc_batch_size)
                bc_kwargs = dict(
                    bc_observations=labels[0][pick],
                    bc_actions=labels[1][pick],
                    bc_weight=args.bc_weight,
                    augmentations=augmentations,
                    generator=generator,
                )
            losses = finetune_step(agent, observations, gradients, optimizer, **bc_kwargs)
            metrics_rows.append(
                {
                    "iteration": iteration,
                    "mean_cost_ratio": info["mean_cost_ratio"],
                    "mean_grad_norm": info["mean_grad_norm"],
                    "surrogate_loss": losses["surrogate_loss"],
                    "bc_loss": losses["bc_loss"],
                    "elapsed": time.time() - timer,
                }
            )
            if args.sigma_end is not None:
                metrics_rows[-1]["sigma"] = sigma
            if pool_obs_array is not None and (
                iteration <= 50 or iteration % args.val_interval == 0 or iteration == args.iterations
            ):
                pool_delta, _ = evaluate_policy_on(pool_obs_array, pool_contexts, pool_linear)
                pool_rows.append(
                    {
                        "iteration": iteration,
                        "evaluations_per_target": 2 * args.pairs * iteration,
                        "pool_delta_j": pool_delta,
                    }
                )
            if iteration % args.val_interval == 0 or iteration == args.iterations:
                delta, positive = validate()
                val_rows.append({"iteration": iteration, "val_delta_j": delta, "val_positive_rate": positive})
                if delta < best_delta:
                    best_delta, best_iteration = delta, iteration
                    best_state = deepcopy(agent.actor_mean.state_dict())
        agent.actor_mean.load_state_dict(best_state)
        agent.eval()

        write_rows(
            run_dir / "finetune_metrics.csv",
            metrics_rows,
            fieldnames=["iteration", "mean_cost_ratio", "mean_grad_norm", "surrogate_loss", "bc_loss", "elapsed"]
            + (["sigma"] if args.sigma_end is not None else []),
        )
        if args.train_pool_size is not None:
            write_rows(
                run_dir / "pool_metrics.csv",
                pool_rows,
                fieldnames=["iteration", "evaluations_per_target", "pool_delta_j"],
            )
        write_rows(run_dir / "val_metrics.csv", val_rows, fieldnames=["iteration", "val_delta_j", "val_positive_rate"])
        save_checkpoint(
            run_dir / "checkpoints" / "last.pt",
            agent=agent,
            optimizer=optimizer,
            config=config,
            system_spec=spec,
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
        if pool is not None:
            pool.close()
        env.close()
    metadata["finetune"] = {
        "init_run_dir": args.init_run_dir,
        "actor_type": config.ppo.actor_type,
        "latent_dim": config.path.latent_dim,
        "iterations": args.iterations,
        "batch_problems": args.batch_problems,
        "pairs": args.pairs,
        "sigma": args.sigma,
        "sigma_end": args.sigma_end,
        "sigma_schedule": "constant" if args.sigma_end is None else "linear",
        "train_pool_size": args.train_pool_size,
        "train_pool_seeds": None
        if args.train_pool_size is None
        else [
            train_problem_seed(1, 0, batch_problems=args.train_pool_size, train_seed=args.train_seed, train_seeds=train_seeds),
            train_problem_seed(
                1, args.train_pool_size - 1, batch_problems=args.train_pool_size, train_seed=args.train_seed, train_seeds=train_seeds
            ),
        ],
        "learning_rate": args.learning_rate,
        "bc_weight": args.bc_weight,
        "label_run_dirs": list(args.label_run_dir),
        "augment_symmetry": bool(args.augment_symmetry),
        "bc_batch_size": args.bc_batch_size,
        "train_seed": args.train_seed,
        "val_seed": args.val_seed,
        "val_instances": args.val_instances,
        "val_interval": args.val_interval,
        "seed": args.seed,
        "workers": args.workers,
        "best_iteration": best_iteration,
        "best_val_delta_j": best_delta,
        "initial_val_delta_j": initial_delta,
    }
    if args.train_seeds_file is not None:
        metadata["finetune"]["train_seeds_file"] = seed_file_record(args.train_seeds_file)
    if args.val_seeds_file is not None:
        metadata["finetune"]["val_seeds_file"] = seed_file_record(args.val_seeds_file)
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
        summaries = summarize_oracle(oracle_rows, {"FineTuned": learned}, label_format="{label}")
        write_rows(run_dir / "finetune_summary.csv", summaries, fieldnames=ORACLE_SUMMARY_FIELDS)
    refresh_metadata_output_files(run_dir)

    print(f"val_delta_j: initial={initial_delta:+.2f}% best={best_delta:+.2f}% (iteration {best_iteration})")
    for summary in summaries:
        print(
            f"{summary['method']}: mean_cost={summary['mean_cost']:.4f} "
            f"delta_j={summary['delta_j_percent']:+.2f}% positive_rate={summary['positive_rate']:.2f} "
            f"captured={summary['captured_fraction']}"
        )
    print(f"RUN_DIR={run_dir}")


if __name__ == "__main__":
    main()
