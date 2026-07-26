"""Shared training utilities for Phase 6 experiment CLIs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time
from typing import Any

import gymnasium
import numpy as np
import torch

from homotopy_path_learning.config import (
    Phase6Config,
    build_env,
    build_system_spec,
    build_tracker_config,
    save_config,
    with_overrides,
)
from homotopy_path_learning.rl.checkpoint import save_checkpoint, system_spec_to_dict
from homotopy_path_learning.rl.networks import ActorCritic
from homotopy_path_learning.rl.ppo import PPOMetrics, ppo_update
from homotopy_path_learning.rl.rollout import RolloutBatch, collect_rollout


TRAIN_METRIC_FIELDS = [
    "update",
    "global_step",
    "elapsed_seconds",
    "learning_rate",
    "rollout_reward_mean",
    "rollout_reward_std",
    "rollout_reward_min",
    "rollout_reward_max",
    "rollout_linear_cost_mean",
    "rollout_bezier_cost_mean",
    "rollout_cost_improvement_mean",
    "rollout_linear_problem_success_rate",
    "rollout_bezier_problem_success_rate",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "explained_variance",
    "gradient_norm",
    "actor_logstd_mean",
]


@dataclass(frozen=True)
class TrainingRunResult:
    run_dir: Path
    run_id: str
    global_step: int
    updates: int
    elapsed_seconds: float
    best_mean_cost: float
    final_metrics: dict[str, float]


def seed_everything(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for CPU smoke reproducibility."""

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = bool(deterministic_torch)
    torch.backends.cudnn.benchmark = False


def git_value(args: list[str], *, default: str = "unknown") -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return default
    return completed.stdout.strip()


def git_dirty() -> bool:
    return git_value(["status", "--short"], default="dirty") != ""


def require_clean_git(*, allow_dirty: bool) -> None:
    if git_dirty() and not allow_dirty:
        raise RuntimeError("Git worktree is dirty; pass --allow-dirty for smoke/development runs.")


def _run_text(command: list[str], *, default: str = "unknown") -> str:
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except Exception:
        return default
    return completed.stdout.strip() or default


def _julia_dependency_version(name: str) -> str:
    code = (
        "using Pkg; deps=Pkg.dependencies(); "
        f"for (_, dep) in deps; dep.name == \"{name}\" && println(dep.version); end"
    )
    return _run_text(["julia", "--startup-file=no", "--project=julia", "-e", code])


def collect_metadata(
    *,
    config: Phase6Config,
    run_id: str,
    run_dir: Path,
    status: str,
    start_time_utc: str,
    end_time_utc: str | None,
    config_path: str,
    env,
) -> dict[str, Any]:
    """Collect JSON-serializable run metadata."""

    return {
        "experiment_id": config.experiment.id,
        "experiment_name": config.experiment.name,
        "run_id": run_id,
        "utc_start_time": start_time_utc,
        "utc_end_time": end_time_utc,
        "status": status,
        "git_branch": git_value(["branch", "--show-current"]),
        "git_commit": git_value(["rev-parse", "HEAD"]),
        "git_dirty": git_dirty(),
        "docker_container": "homotopy-continuation",
        "python_version": platform.python_version(),
        "python_executable": os.sys.executable,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "gymnasium_version": gymnasium.__version__,
        "juliacall_version": _package_version("juliacall"),
        "julia_version": _run_text(["julia", "--version"]),
        "julia_executable": _run_text(["bash", "-lc", "command -v julia"]),
        "julia_project": os.environ.get("PYTHON_JULIACALL_PROJECT", str(Path.cwd() / "julia")),
        "homotopycontinuation_version": _julia_dependency_version("HomotopyContinuation"),
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "cuda_available": torch.cuda.is_available(),
        "device": config.experiment.device,
        "seeds": {
            "experiment": config.experiment.seed,
            "basis": config.path.basis_seed,
            "evaluation": config.evaluation.seed,
            "random_action": config.evaluation.random_action_seed,
        },
        "observation_shape": tuple(int(v) for v in env.observation_space.shape),
        "action_shape": tuple(int(v) for v in env.action_space.shape),
        "action_low": np.array(env.action_space.low, dtype=np.float32).tolist(),
        "action_high": np.array(env.action_space.high, dtype=np.float32).tolist(),
        "path_count": int(np.prod(config.system.degrees)),
        "support": [[list(exp) for exp in block] for block in build_system_spec(config).supports or ()],
        "coefficient_ordering": "equation blocks; leading coefficient first in each block",
        "latent_basis_shape": tuple(int(v) for v in env.parameterization.basis.shape),
        "latent_basis_file": "latent_basis.npy",
        "config_path": config_path,
        "evaluation_seed_rule": "evaluation.seed + instance_index",
        "evaluation_num_instances": config.evaluation.num_instances,
        "output_files": _output_files(run_dir),
    }


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _output_files(run_dir: Path) -> list[str]:
    if not run_dir.exists():
        return []
    return sorted(str(path.relative_to(run_dir)) for path in run_dir.rglob("*") if path.is_file())


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def refresh_metadata_output_files(run_dir: str | Path) -> None:
    """Refresh the ``output_files`` field in an existing run metadata file."""

    directory = Path(run_dir)
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["output_files"] = _output_files(directory)
    write_metadata(metadata_path, metadata)


def write_evaluation_seeds(path: Path, *, seed: int, num_instances: int) -> list[int]:
    seeds = [int(seed) + index for index in range(int(num_instances))]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "seed": int(seed),
                "num_instances": int(num_instances),
                "rule": "seed + instance_index",
                "seeds": seeds,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    return seeds


class CSVLogger:
    """Append finite training metrics to a CSV with a fixed header."""

    def __init__(self, path: Path, fieldnames: list[str]) -> None:
        self.path = path
        self.fieldnames = list(fieldnames)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writeheader()

    def write(self, row: dict[str, Any]) -> None:
        output: dict[str, Any] = {}
        for key in self.fieldnames:
            value = row[key]
            if isinstance(value, (int, np.integer)):
                output[key] = int(value)
            else:
                numeric = float(value)
                if not np.isfinite(numeric):
                    raise ValueError(f"training metric {key} must be finite, got {value!r}.")
                output[key] = numeric
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writerow(output)


def rollout_metric_summary(batch: RolloutBatch) -> dict[str, float]:
    """Summarize Bezier environment metrics recorded in rollout infos."""

    def mean_key(key: str) -> float:
        values = [float(metrics[key]) for metrics in batch.env_metrics if key in metrics]
        return float(np.mean(values)) if values else 0.0

    rewards = batch.rewards.detach().cpu().numpy()
    return {
        "rollout_reward_mean": float(np.mean(rewards)),
        "rollout_reward_std": float(np.std(rewards, ddof=0)),
        "rollout_reward_min": float(np.min(rewards)),
        "rollout_reward_max": float(np.max(rewards)),
        "rollout_linear_cost_mean": mean_key("linear_cost"),
        "rollout_bezier_cost_mean": mean_key("bezier_cost"),
        "rollout_cost_improvement_mean": mean_key("cost_improvement"),
        "rollout_linear_problem_success_rate": mean_key("linear_success"),
        "rollout_bezier_problem_success_rate": mean_key("bezier_success"),
    }


def reinitialize_env_backend(env, config: Phase6Config) -> None:
    """Restore Julia global state for an existing environment backend.

    Phase 4 uses process-global Julia state. Creating and closing a temporary
    evaluation backend clears that state, so the long-lived training environment
    reinitializes its backend before continuing collection.
    """

    env.backend.initialize(
        build_system_spec(config),
        config.path.bezier_degree,
        build_tracker_config(config),
    )
    if config.environment.warmup_backend:
        env.backend.warmup()


def run_training(
    *,
    config: Phase6Config,
    run_id: str,
    config_path: str,
    total_timesteps: int | None = None,
    output_root: str | None = None,
    device: str | None = None,
    allow_dirty: bool = False,
) -> TrainingRunResult:
    """Run a single-env PPO smoke training loop and write Phase 6 artifacts."""

    require_clean_git(allow_dirty=allow_dirty)
    run_id = str(run_id)
    if run_id.strip() == "":
        raise ValueError("run_id must not be empty.")
    if output_root is not None:
        config = with_overrides(config, output_root=output_root)
    if device is not None:
        config = with_overrides(config, device=device)
    if total_timesteps is not None:
        config = with_overrides(config, total_timesteps=total_timesteps)
    seed_everything(config.experiment.seed)
    device_obj = torch.device(config.experiment.device)
    run_dir = Path(config.output.root) / config.experiment.id / run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "evaluation").mkdir()
    (run_dir / "benchmark").mkdir()
    (run_dir / "analysis").mkdir()

    start_time = time.time()
    start_time_utc = datetime.now(timezone.utc).isoformat()
    env = None
    metadata_path = run_dir / "metadata.json"
    try:
        save_config(config, run_dir / "config.yaml")
        env = build_env(config)
        env.action_space.seed(config.experiment.seed)
        np.save(run_dir / "latent_basis.npy", env.parameterization.basis)
        write_evaluation_seeds(
            run_dir / "evaluation_seeds.json",
            seed=config.evaluation.seed,
            num_instances=config.evaluation.num_instances,
        )
        metadata = collect_metadata(
            config=config,
            run_id=run_id,
            run_dir=run_dir,
            status="Running",
            start_time_utc=start_time_utc,
            end_time_utc=None,
            config_path=config_path,
            env=env,
        )
        write_metadata(metadata_path, metadata)

        agent = ActorCritic(
            observation_shape=tuple(env.observation_space.shape),
            action_space=env.action_space,
            hidden_sizes=config.ppo.hidden_sizes,
            activation=config.ppo.activation,
            actor_logstd_init=config.ppo.actor_logstd_init,
            device=device_obj,
        )
        optimizer = torch.optim.Adam(agent.parameters(), lr=config.ppo.learning_rate, eps=1e-5)
        logger = CSVLogger(run_dir / "train_metrics.csv", TRAIN_METRIC_FIELDS)
        observation, _ = env.reset(seed=config.experiment.seed)
        best_mean_cost = float("inf")
        final_metrics: dict[str, float] = {}
        global_step = 0
        system_spec = build_system_spec(config)
        git_commit = git_value(["rev-parse", "HEAD"])

        for update in range(1, config.ppo.num_updates + 1):
            batch = collect_rollout(
                env,
                agent,
                rollout_steps=config.ppo.rollout_steps,
                device=device_obj,
                initial_observation=observation,
            )
            observation = batch.next_observation
            global_step += config.ppo.rollout_steps
            metrics: PPOMetrics = ppo_update(
                agent=agent,
                optimizer=optimizer,
                batch=batch,
                settings=config.ppo,
                update_index=update,
                total_updates=config.ppo.num_updates,
            )
            rollout_summary = rollout_metric_summary(batch)
            row = {
                "update": update,
                "global_step": global_step,
                "elapsed_seconds": time.time() - start_time,
                "learning_rate": metrics.learning_rate,
                **rollout_summary,
                "policy_loss": metrics.policy_loss,
                "value_loss": metrics.value_loss,
                "entropy": metrics.entropy,
                "approx_kl": metrics.approx_kl,
                "clip_fraction": metrics.clip_fraction,
                "explained_variance": metrics.explained_variance,
                "gradient_norm": metrics.gradient_norm,
                "actor_logstd_mean": float(agent.actor_logstd.detach().mean().cpu()),
            }
            logger.write(row)
            final_metrics = {key: float(value) for key, value in row.items() if key not in {"update", "global_step"}}

            save_step = global_step % config.ppo.checkpoint_interval == 0
            if save_step:
                save_checkpoint(
                    checkpoints_dir / f"step_{global_step:08d}.pt",
                    agent=agent,
                    optimizer=optimizer,
                    config=config,
                    system_spec=system_spec,
                    global_step=global_step,
                    update=update,
                    observation_shape=tuple(env.observation_space.shape),
                    action_space=env.action_space,
                    experiment_id=config.experiment.id,
                    run_id=run_id,
                    git_commit=git_commit,
                )
            evaluation_due = update % config.ppo.evaluation_interval_updates == 0
            if config.output.save_best_checkpoint and evaluation_due:
                from homotopy_path_learning.evaluation.evaluator import evaluate_policy_rows

                eval_rows = evaluate_policy_rows(
                    config=config,
                    run_id=run_id,
                    agent=agent,
                    device=device_obj,
                    methods=("LearnedBezier",),
                    num_instances=config.evaluation.num_instances,
                    evaluation_seed=config.evaluation.seed,
                )
                eval_mean_cost = float(np.mean([float(row["mean_cost"]) for row in eval_rows]))
                reinitialize_env_backend(env, config)
            else:
                eval_mean_cost = rollout_summary["rollout_bezier_cost_mean"]
            if config.output.save_best_checkpoint and eval_mean_cost < best_mean_cost:
                best_mean_cost = eval_mean_cost
                save_checkpoint(
                    checkpoints_dir / "best.pt",
                    agent=agent,
                    optimizer=optimizer,
                    config=config,
                    system_spec=system_spec,
                    global_step=global_step,
                    update=update,
                    observation_shape=tuple(env.observation_space.shape),
                    action_space=env.action_space,
                    experiment_id=config.experiment.id,
                    run_id=run_id,
                    git_commit=git_commit,
                )

        if config.output.save_last_checkpoint:
            save_checkpoint(
                checkpoints_dir / "last.pt",
                agent=agent,
                optimizer=optimizer,
                config=config,
                system_spec=system_spec,
                global_step=global_step,
                update=config.ppo.num_updates,
                observation_shape=tuple(env.observation_space.shape),
                action_space=env.action_space,
                experiment_id=config.experiment.id,
                run_id=run_id,
                git_commit=git_commit,
            )
        if config.output.save_best_checkpoint and not (checkpoints_dir / "best.pt").exists():
            save_checkpoint(
                checkpoints_dir / "best.pt",
                agent=agent,
                optimizer=optimizer,
                config=config,
                system_spec=system_spec,
                global_step=global_step,
                update=config.ppo.num_updates,
                observation_shape=tuple(env.observation_space.shape),
                action_space=env.action_space,
                experiment_id=config.experiment.id,
                run_id=run_id,
                git_commit=git_commit,
            )

        end_time_utc = datetime.now(timezone.utc).isoformat()
        metadata = collect_metadata(
            config=config,
            run_id=run_id,
            run_dir=run_dir,
            status="Completed",
            start_time_utc=start_time_utc,
            end_time_utc=end_time_utc,
            config_path=config_path,
            env=env,
        )
        write_metadata(metadata_path, metadata)
        return TrainingRunResult(
            run_dir=run_dir,
            run_id=run_id,
            global_step=global_step,
            updates=config.ppo.num_updates,
            elapsed_seconds=time.time() - start_time,
            best_mean_cost=best_mean_cost,
            final_metrics=final_metrics,
        )
    except Exception:
        if env is not None:
            failed_metadata = collect_metadata(
                config=config,
                run_id=run_id,
                run_dir=run_dir,
                status="Failed",
                start_time_utc=start_time_utc,
                end_time_utc=datetime.now(timezone.utc).isoformat(),
                config_path=config_path,
                env=env,
            )
            write_metadata(metadata_path, failed_metadata)
        raise
    finally:
        if env is not None:
            env.close()


__all__ = [
    "CSVLogger",
    "TRAIN_METRIC_FIELDS",
    "TrainingRunResult",
    "collect_metadata",
    "git_dirty",
    "git_value",
    "require_clean_git",
    "refresh_metadata_output_files",
    "rollout_metric_summary",
    "run_training",
    "seed_everything",
    "reinitialize_env_backend",
    "write_evaluation_seeds",
]
