"""Fixed-seed evaluation for Phase 6 experiments."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Iterable

import numpy as np
import torch

from homotopy_path_learning.config import Phase6Config, build_env, build_system_spec
from homotopy_path_learning.evaluation.datasets import fixed_evaluation_seeds
from homotopy_path_learning.evaluation.metrics import EVALUATION_FIELDS, write_rows, write_summary_csv
from homotopy_path_learning.rl.checkpoint import (
    build_agent_from_checkpoint,
    load_checkpoint,
    validate_checkpoint,
)


def _action_from_agent(agent, observation: np.ndarray, *, device: torch.device | str) -> np.ndarray:
    obs = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        action = agent.deterministic_action(obs)
    return action.squeeze(0).detach().cpu().numpy().astype(np.float32)


def _row(
    *,
    config: Phase6Config,
    run_id: str,
    instance_id: int,
    target_seed: int,
    method: str,
    success: bool,
    n_paths: int,
    n_success: int,
    n_failed: int,
    mean_cost: float,
    accepted_steps: int,
    rejected_steps: int,
    improvement: float,
    reward: float,
    residual_norms: np.ndarray,
    failure_codes: Iterable[str],
    action: np.ndarray,
    elapsed_seconds: float,
) -> dict[str, object]:
    return {
        "experiment_id": config.experiment.id,
        "run_id": run_id,
        "instance_id": int(instance_id),
        "target_seed": int(target_seed),
        "method": method,
        "problem_success": int(bool(success)),
        "n_paths": int(n_paths),
        "n_success": int(n_success),
        "n_failed": int(n_failed),
        "mean_cost": float(mean_cost),
        "accepted_steps": int(accepted_steps),
        "rejected_steps": int(rejected_steps),
        "cost_improvement_vs_linear": float(improvement),
        "reward_equivalent": float(reward),
        "max_residual_norm": float(np.max(np.asarray(residual_norms, dtype=np.float64))),
        "failure_codes": "|".join(str(code) for code in failure_codes),
        "action_l2_norm": float(np.linalg.norm(np.asarray(action, dtype=np.float64))),
        "elapsed_seconds": float(elapsed_seconds),
    }


def evaluate_policy_rows(
    *,
    config: Phase6Config,
    run_id: str,
    agent,
    device: torch.device | str,
    methods: tuple[str, ...],
    num_instances: int | None = None,
    evaluation_seed: int | None = None,
    random_action_seed: int | None = None,
) -> list[dict[str, object]]:
    """Evaluate Linear, RandomBezier, and/or LearnedBezier on fixed target seeds."""

    seeds = fixed_evaluation_seeds(
        seed=config.evaluation.seed if evaluation_seed is None else evaluation_seed,
        num_instances=config.evaluation.num_instances if num_instances is None else num_instances,
    )
    random_rng = np.random.default_rng(
        config.evaluation.random_action_seed if random_action_seed is None else random_action_seed
    )
    env = build_env(config)
    rows: list[dict[str, object]] = []
    try:
        for instance_id, target_seed in enumerate(seeds):
            observation, reset_info = env.reset(seed=target_seed)
            context = env.current_episode_context()
            linear_target = np.array(context.target_coefficients, dtype=np.complex128, copy=True)
            linear_cost = float(context.linear_cost.mean)
            if "Linear" in methods:
                rows.append(
                    _row(
                        config=config,
                        run_id=run_id,
                        instance_id=instance_id,
                        target_seed=target_seed,
                        method="Linear",
                        success=bool(context.linear_result.success),
                        n_paths=int(context.linear_result.n_paths),
                        n_success=int(context.linear_result.n_success),
                        n_failed=int(context.linear_result.n_failed),
                        mean_cost=linear_cost,
                        accepted_steps=int(context.linear_result.accepted_steps),
                        rejected_steps=int(context.linear_result.rejected_steps),
                        improvement=0.0,
                        reward=0.0,
                        residual_norms=context.linear_result.residual_norms,
                        failure_codes=context.linear_result.failure_codes,
                        action=np.zeros(env.action_space.shape, dtype=np.float32),
                        elapsed_seconds=0.0,
                    )
                )

            if "LearnedBezier" in methods:
                if not np.array_equal(linear_target, context.target_coefficients):
                    raise RuntimeError("LearnedBezier target coefficients differ from Linear target.")
                start = time.time()
                action = _action_from_agent(agent, observation, device=device)
                evaluation = env.evaluate_action_against_context(context, action)
                reward = evaluation.reward
                info = evaluation.info
                rows.append(
                    _row(
                        config=config,
                        run_id=run_id,
                        instance_id=instance_id,
                        target_seed=target_seed,
                        method="LearnedBezier",
                        success=bool(info["bezier_success"]),
                        n_paths=int(info["bezier_n_success"]) + int(info["bezier_n_failed"]),
                        n_success=int(info["bezier_n_success"]),
                        n_failed=int(info["bezier_n_failed"]),
                        mean_cost=float(info["bezier_cost"]),
                        accepted_steps=int(info["bezier_accepted_steps"]),
                        rejected_steps=int(info["bezier_rejected_steps"]),
                        improvement=float(info["cost_improvement"]),
                        reward=float(reward),
                        residual_norms=info["bezier_residual_norms"],
                        failure_codes=info["bezier_failure_codes"],
                        action=action,
                        elapsed_seconds=time.time() - start,
                    )
                )

            if "RandomBezier" in methods:
                for random_index in range(config.evaluation.random_actions_per_instance):
                    if not np.array_equal(linear_target, context.target_coefficients):
                        raise RuntimeError("RandomBezier target coefficients differ from Linear target.")
                    start = time.time()
                    action = random_rng.uniform(
                        low=env.action_space.low,
                        high=env.action_space.high,
                    ).astype(np.float32)
                    evaluation = env.evaluate_action_against_context(context, action)
                    reward = evaluation.reward
                    info = evaluation.info
                    rows.append(
                        _row(
                            config=config,
                            run_id=run_id,
                            instance_id=instance_id * config.evaluation.random_actions_per_instance + random_index,
                            target_seed=target_seed,
                            method="RandomBezier",
                            success=bool(info["bezier_success"]),
                            n_paths=int(info["bezier_n_success"]) + int(info["bezier_n_failed"]),
                            n_success=int(info["bezier_n_success"]),
                            n_failed=int(info["bezier_n_failed"]),
                            mean_cost=float(info["bezier_cost"]),
                            accepted_steps=int(info["bezier_accepted_steps"]),
                            rejected_steps=int(info["bezier_rejected_steps"]),
                            improvement=float(info["cost_improvement"]),
                            reward=float(reward),
                            residual_norms=info["bezier_residual_norms"],
                            failure_codes=info["bezier_failure_codes"],
                            action=action,
                            elapsed_seconds=time.time() - start,
                        )
                    )
    finally:
        env.close()
    return rows


def load_policy_for_config(
    *,
    checkpoint_path: str | Path,
    config: Phase6Config,
    device: torch.device | str,
):
    """Load and validate a checkpoint policy for ``config``."""

    env = build_env(config)
    try:
        checkpoint = load_checkpoint(checkpoint_path, map_location=device)
        validate_checkpoint(
            checkpoint,
            config=config,
            system_spec=build_system_spec(config),
            observation_shape=tuple(env.observation_space.shape),
            action_space=env.action_space,
        )
        return build_agent_from_checkpoint(checkpoint, action_space=env.action_space, device=device)
    finally:
        env.close()


def evaluate_checkpoint(
    *,
    config: Phase6Config,
    run_id: str,
    checkpoint_path: str | Path,
    output_path: str | Path,
    num_instances: int | None = None,
    evaluation_seed: int | None = None,
    device: torch.device | str = "cpu",
) -> list[dict[str, object]]:
    """Evaluate Linear and LearnedBezier and write evaluation CSV."""

    agent = load_policy_for_config(checkpoint_path=checkpoint_path, config=config, device=device)
    rows = evaluate_policy_rows(
        config=config,
        run_id=run_id,
        agent=agent,
        device=device,
        methods=("Linear", "LearnedBezier"),
        num_instances=num_instances,
        evaluation_seed=evaluation_seed,
    )
    write_rows(output_path, rows, fieldnames=EVALUATION_FIELDS)
    return rows


def benchmark_checkpoint(
    *,
    config: Phase6Config,
    run_id: str,
    checkpoint_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path,
    num_instances: int | None = None,
    evaluation_seed: int | None = None,
    device: torch.device | str = "cpu",
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Evaluate Linear, RandomBezier, and LearnedBezier and write CSV outputs."""

    agent = load_policy_for_config(checkpoint_path=checkpoint_path, config=config, device=device)
    rows = evaluate_policy_rows(
        config=config,
        run_id=run_id,
        agent=agent,
        device=device,
        methods=("Linear", "RandomBezier", "LearnedBezier"),
        num_instances=num_instances,
        evaluation_seed=evaluation_seed,
    )
    write_rows(output_path, rows, fieldnames=EVALUATION_FIELDS)
    summaries = write_summary_csv(summary_path, rows)
    return rows, summaries


__all__ = [
    "benchmark_checkpoint",
    "evaluate_checkpoint",
    "evaluate_policy_rows",
    "load_policy_for_config",
]
