from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

from _bootstrap import preload_julia

preload_julia()

import numpy as np
import torch

from homotopy_path_learning.backends.julia import (
    tracking_result_from_julia,
    validate_control_points_for_backend,
)
from homotopy_path_learning.config import build_env, build_system_spec
from homotopy_path_learning.envs.costs import TrackingCost
from homotopy_path_learning.evaluation.evaluator import evaluate_policy_rows
from homotopy_path_learning.performance import (
    TimingStats,
    compare_evaluation_rows,
    compare_performance_stats,
    load_performance_config,
    measure_repeated,
    timed_call,
    write_performance_outputs,
)
from homotopy_path_learning.performance.comparison import compare_tracking_records
from homotopy_path_learning.rl.networks import ActorCritic
from homotopy_path_learning.rl.train_utils import git_dirty, git_value


class ZeroAgent:
    """Deterministic zero-action policy used only by the performance profiler."""

    def __init__(self, action_dim: int) -> None:
        self.action_dim = int(action_dim)

    def deterministic_action(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (observation.shape[0], self.action_dim),
            dtype=torch.float32,
            device=observation.device,
        )


def _complex_array_payload(values: np.ndarray) -> list[Any]:
    array = np.asarray(values, dtype=np.complex128)
    stacked = np.stack([array.real, array.imag], axis=-1)
    return stacked.tolist()


def _tracking_record(
    *,
    target_seed: int,
    method: str,
    result,
    cost: TrackingCost,
    reward: float,
    action: np.ndarray,
) -> dict[str, Any]:
    return {
        "target_seed": int(target_seed),
        "method": method,
        "success": bool(result.success),
        "n_paths": int(result.n_paths),
        "n_success": int(result.n_success),
        "n_failed": int(result.n_failed),
        "accepted_steps": int(result.accepted_steps),
        "rejected_steps": int(result.rejected_steps),
        "per_path_accepted_steps": [
            int(value) for value in np.asarray(result.per_path_accepted_steps).reshape(-1)
        ],
        "per_path_rejected_steps": [
            int(value) for value in np.asarray(result.per_path_rejected_steps).reshape(-1)
        ],
        "path_success": [bool(value) for value in np.asarray(result.path_success).reshape(-1)],
        "endpoints": _complex_array_payload(np.asarray(result.endpoints, dtype=np.complex128)),
        "residual_norms": [
            float(value) for value in np.asarray(result.residual_norms, dtype=np.float64).reshape(-1)
        ],
        "failure_codes": [str(value) for value in result.failure_codes],
        "mean_cost": float(cost.mean),
        "reward": float(reward),
        "action": [float(value) for value in np.asarray(action, dtype=np.float64).reshape(-1)],
    }


def _phase6_without_warmup(config):
    return replace(
        config,
        environment=replace(config.environment, warmup_backend=False),
    )


def _make_random_action(rng: np.random.Generator, env) -> np.ndarray:
    return rng.uniform(low=env.action_space.low, high=env.action_space.high).astype(np.float32)


def _collect_detailed_records(config, *, num_instances: int, seed: int, action_seed: int):
    env = build_env(config)
    rng = np.random.default_rng(action_seed)
    records: list[dict[str, Any]] = []
    try:
        for offset in range(num_instances):
            target_seed = int(seed) + offset
            env.reset(seed=target_seed)
            context = env.current_episode_context()
            zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
            records.append(
                _tracking_record(
                    target_seed=target_seed,
                    method="Linear",
                    result=context.linear_result,
                    cost=context.linear_cost,
                    reward=0.0,
                    action=zero_action,
                )
            )
            learned = env.evaluate_action_against_context(context, zero_action)
            records.append(
                _tracking_record(
                    target_seed=target_seed,
                    method="LearnedBezier",
                    result=learned.tracking_result,
                    cost=learned.tracking_cost,
                    reward=learned.reward,
                    action=zero_action,
                )
            )
            random_action = _make_random_action(rng, env)
            random_eval = env.evaluate_action_against_context(context, random_action)
            records.append(
                _tracking_record(
                    target_seed=target_seed,
                    method="RandomBezier",
                    result=random_eval.tracking_result,
                    cost=random_eval.tracking_cost,
                    reward=random_eval.reward,
                    action=random_action,
                )
            )
    finally:
        env.close()
    return records


def _measure_profile(config, performance) -> tuple[list[TimingStats], dict[str, list[dict[str, Any]]]]:
    stats: list[TimingStats] = []
    profile_config = _phase6_without_warmup(config)

    env, build_ns = timed_call(lambda: build_env(profile_config))
    try:
        stats.append(
            TimingStats(
                name="backend_initialize_cold",
                count=1,
                total_seconds=build_ns / 1_000_000_000.0,
                mean_seconds=build_ns / 1_000_000_000.0,
                median_seconds=build_ns / 1_000_000_000.0,
                std_seconds=0.0,
                min_seconds=build_ns / 1_000_000_000.0,
                max_seconds=build_ns / 1_000_000_000.0,
                p50_seconds=build_ns / 1_000_000_000.0,
                p95_seconds=build_ns / 1_000_000_000.0,
                throughput_per_second=1_000_000_000.0 / build_ns if build_ns > 0 else 0.0,
                problem_seconds=build_ns / 1_000_000_000.0,
                path_seconds=build_ns / 1_000_000_000.0 / int(np.prod(config.system.degrees)),
            )
        )

        warmup_stats, _ = measure_repeated(
            "backend_warmup",
            lambda: env.backend.warmup(),
            warmup=0,
            repeat=max(1, min(performance.measure_runs, 5)),
            trials=performance.trials,
            paths_per_sample=int(np.prod(config.system.degrees)),
        )
        stats.append(warmup_stats)

        observation, _ = env.reset(seed=performance.repeated_target_seed)
        context = env.current_episode_context()
        control_points = context.linear_control_points
        validation_stats, _ = measure_repeated(
            "python_input_validation",
            lambda: validate_control_points_for_backend(
                control_points,
                system_spec=build_system_spec(config),
                bezier_degree=config.path.bezier_degree,
            ),
            warmup=performance.warmup_runs,
            repeat=performance.measure_runs,
            trials=performance.trials,
        )
        stats.append(validation_stats)

        sample_payload = {
            "success": context.linear_result.success,
            "n_paths": context.linear_result.n_paths,
            "n_success": context.linear_result.n_success,
            "n_failed": context.linear_result.n_failed,
            "accepted_steps": context.linear_result.accepted_steps,
            "rejected_steps": context.linear_result.rejected_steps,
            "per_path_accepted_steps": context.linear_result.per_path_accepted_steps,
            "per_path_rejected_steps": context.linear_result.per_path_rejected_steps,
            "path_success": context.linear_result.path_success,
            "endpoints": context.linear_result.endpoints,
            "residual_norms": context.linear_result.residual_norms,
            "failure_codes": context.linear_result.failure_codes,
        }
        conversion_stats, _ = measure_repeated(
            "python_result_conversion",
            lambda: tracking_result_from_julia(sample_payload, n_vars=env.system_spec.n_vars),
            warmup=performance.warmup_runs,
            repeat=performance.measure_runs,
            trials=performance.trials,
        )
        stats.append(conversion_stats)

        track_stats, _ = measure_repeated(
            "track_same_control_points",
            lambda: env.backend.track(control_points),
            warmup=performance.warmup_runs,
            repeat=performance.measure_runs,
            trials=performance.trials,
            paths_per_sample=int(np.prod(config.system.degrees)),
        )
        stats.append(track_stats)

        reset_counter = {"value": 0}

        def reset_once():
            seed = performance.evaluation_seed + reset_counter["value"]
            reset_counter["value"] += 1
            return env.reset(seed=seed)

        reset_stats, _ = measure_repeated(
            "env_reset",
            reset_once,
            warmup=performance.warmup_runs,
            repeat=performance.measure_runs,
            trials=performance.trials,
            paths_per_sample=int(np.prod(config.system.degrees)),
        )
        stats.append(reset_stats)

        step_counter = {"value": 0}
        step_rng = np.random.default_rng(performance.action_seed)

        def step_once():
            seed = performance.evaluation_seed + 10_000 + step_counter["value"]
            step_counter["value"] += 1
            env.reset(seed=seed)
            action = _make_random_action(step_rng, env)
            return env.step(action)

        step_stats, _ = measure_repeated(
            "env_step",
            step_once,
            warmup=performance.warmup_runs,
            repeat=performance.measure_runs,
            trials=performance.trials,
            paths_per_sample=int(np.prod(config.system.degrees)),
        )
        stats.append(step_stats)

        agent = ActorCritic(
            observation_shape=tuple(int(value) for value in env.observation_space.shape),
            action_space=env.action_space,
            hidden_sizes=config.ppo.hidden_sizes,
            activation=config.ppo.activation,
            actor_logstd_init=config.ppo.actor_logstd_init,
        )
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        policy_stats, _ = measure_repeated(
            "ppo_policy_forward",
            lambda: agent.deterministic_action(obs_tensor),
            warmup=performance.warmup_runs,
            repeat=performance.measure_runs,
            trials=performance.trials,
        )
        stats.append(policy_stats)
    finally:
        env.close()

    zero_agent = ZeroAgent(action_dim=config.path.latent_dim * (config.path.bezier_degree - 1))
    rows_by_name: dict[str, list[dict[str, Any]]] = {}

    def fixed_evaluation():
        return evaluate_policy_rows(
            config=config,
            run_id="phase7-profile",
            agent=zero_agent,
            device="cpu",
            methods=("Linear", "LearnedBezier"),
            num_instances=performance.fixed_problem_count,
            evaluation_seed=performance.evaluation_seed,
            random_action_seed=performance.action_seed,
        )

    eval_rows, eval_ns = timed_call(fixed_evaluation)
    rows_by_name["evaluation_rows"] = eval_rows
    stats.append(
        TimingStats(
            name="fixed_evaluation",
            count=1,
            total_seconds=eval_ns / 1_000_000_000.0,
            mean_seconds=eval_ns / 1_000_000_000.0,
            median_seconds=eval_ns / 1_000_000_000.0,
            std_seconds=0.0,
            min_seconds=eval_ns / 1_000_000_000.0,
            max_seconds=eval_ns / 1_000_000_000.0,
            p50_seconds=eval_ns / 1_000_000_000.0,
            p95_seconds=eval_ns / 1_000_000_000.0,
            throughput_per_second=performance.fixed_problem_count / (eval_ns / 1_000_000_000.0),
            problem_seconds=(eval_ns / 1_000_000_000.0) / performance.fixed_problem_count,
            path_seconds=(eval_ns / 1_000_000_000.0)
            / (performance.fixed_problem_count * int(np.prod(config.system.degrees))),
        )
    )

    def fixed_benchmark():
        return evaluate_policy_rows(
            config=config,
            run_id="phase7-profile",
            agent=zero_agent,
            device="cpu",
            methods=("Linear", "RandomBezier", "LearnedBezier"),
            num_instances=performance.fixed_problem_count,
            evaluation_seed=performance.evaluation_seed,
            random_action_seed=performance.action_seed,
        )

    benchmark_rows, benchmark_ns = timed_call(fixed_benchmark)
    rows_by_name["benchmark_rows"] = benchmark_rows
    stats.append(
        TimingStats(
            name="benchmark",
            count=1,
            total_seconds=benchmark_ns / 1_000_000_000.0,
            mean_seconds=benchmark_ns / 1_000_000_000.0,
            median_seconds=benchmark_ns / 1_000_000_000.0,
            std_seconds=0.0,
            min_seconds=benchmark_ns / 1_000_000_000.0,
            max_seconds=benchmark_ns / 1_000_000_000.0,
            p50_seconds=benchmark_ns / 1_000_000_000.0,
            p95_seconds=benchmark_ns / 1_000_000_000.0,
            throughput_per_second=performance.fixed_problem_count / (benchmark_ns / 1_000_000_000.0),
            problem_seconds=(benchmark_ns / 1_000_000_000.0) / performance.fixed_problem_count,
            path_seconds=(benchmark_ns / 1_000_000_000.0)
            / (performance.fixed_problem_count * int(np.prod(config.system.degrees))),
        )
    )
    detailed_records = _collect_detailed_records(
        config,
        num_instances=performance.fixed_problem_count,
        seed=performance.evaluation_seed,
        action_seed=performance.action_seed,
    )
    rows_by_name["tracking_records"] = detailed_records
    return stats, rows_by_name


def _metadata(config, *, run_id: str) -> dict[str, Any]:
    def run_text(command: list[str]) -> str:
        try:
            return subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except Exception:
            return "unknown"

    return {
        "git_branch": git_value(["branch", "--show-current"]),
        "git_commit": git_value(["rev-parse", "HEAD"]),
        "git_dirty": git_dirty(),
        "docker_container": "homotopy-continuation",
        "docker_image": os.environ.get("HPL_DOCKER_IMAGE", "torch-2.3.0-sage-julia-sysimage"),
        "cpu": platform.processor() or platform.machine(),
        "python_version": platform.python_version(),
        "python_executable": os.sys.executable,
        "julia_version": run_text(["julia", "--version"]),
        "julia_executable": run_text(["bash", "-lc", "command -v julia"]),
        "julia_num_threads": os.environ.get("JULIA_NUM_THREADS", "<unset>"),
        "juliacall_version": importlib.metadata.version("juliacall"),
        "homotopycontinuation_version": run_text(
            [
                "julia",
                "--startup-file=no",
                "--project=julia",
                "-e",
                (
                    "using Pkg; "
                    "for (_, dep) in Pkg.dependencies(); "
                    "dep.name == \"HomotopyContinuation\" && println(dep.version); end"
                ),
            ]
        ),
        "run_id": run_id,
        "experiment_id": config.experiment.id,
    }


def _load_baseline(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile Phase 7 multivariate Pham performance.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("baseline", "optimized"), required=True)
    parser.add_argument("--baseline", default=None)
    args = parser.parse_args()

    started = time.time()
    profile_config = load_performance_config(args.config)
    config = profile_config.phase6
    output_dir = Path(args.output_dir)
    run_id = output_dir.name

    stats, rows = _measure_profile(config, profile_config.performance)
    stats_by_name = {item.name: item.to_dict() for item in stats}
    baseline = _load_baseline(args.baseline)
    if baseline is None:
        tracking_equivalence = compare_tracking_records(
            rows["tracking_records"],
            rows["tracking_records"],
        )
        evaluation_equivalence = compare_evaluation_rows(
            rows["benchmark_rows"],
            rows["benchmark_rows"],
            ignore_fields=("elapsed_seconds", "run_id"),
        )
    else:
        tracking_equivalence = compare_tracking_records(
            baseline["equivalence_records"]["tracking_records"],
            rows["tracking_records"],
        )
        evaluation_equivalence = compare_evaluation_rows(
            baseline["equivalence_records"]["benchmark_rows"],
            rows["benchmark_rows"],
            ignore_fields=("elapsed_seconds", "run_id"),
        )
    performance_comparison = {}
    if baseline is not None:
        performance_comparison = compare_performance_stats(
            baseline["measurements"],
            stats_by_name,
        )

    decisions = [
        "Evaluation duplicate linear tracking is the Phase 7 optimization candidate measured by this CLI.",
        "Julia kernel, Tracker reuse, batch API, and threaded mode are not enabled unless separately adopted.",
    ]
    payload = {
        "experiment_id": config.experiment.id,
        "run_id": run_id,
        "mode": args.mode,
        "metadata": _metadata(config, run_id=run_id),
        "settings": {
            "warmup_runs": profile_config.performance.warmup_runs,
            "measure_runs": profile_config.performance.measure_runs,
            "trials": profile_config.performance.trials,
            "fixed_problem_count": profile_config.performance.fixed_problem_count,
            "evaluation_seed_rule": "evaluation_seed + instance_index",
            "evaluation_seed": profile_config.performance.evaluation_seed,
            "action_seed": profile_config.performance.action_seed,
        },
        "measurements": stats_by_name,
        "performance_comparison": performance_comparison,
        "equivalence": {
            "tracking": tracking_equivalence.to_dict(),
            "evaluation_rows": evaluation_equivalence.to_dict(),
        },
        "equivalence_records": rows,
        "profiling": {
            "python": [
                "Python profiling is explicit to this CLI and not part of normal training.",
                "Timing uses time.perf_counter_ns().",
                f"Total profile elapsed seconds: {time.time() - started:.6f}",
            ],
            "julia": [
                "Julia runtime is invoked through juliacall with the repository julia project.",
                "Julia kernel-specific optimizations were not enabled for this run.",
                "Boundary timings include Julia set_control_points! and all-path tracking.",
            ],
        },
        "optimization_decisions": decisions,
    }
    write_performance_outputs(
        output_dir,
        config=profile_config,
        payload=payload,
        stats=stats,
    )
    print(f"PERFORMANCE_JSON={output_dir / 'performance.json'}")
    print(f"PERFORMANCE_CSV={output_dir / 'performance.csv'}")
    print(f"EQUIVALENCE_JSON={output_dir / 'equivalence.json'}")
    print(f"SUMMARY_MD={output_dir / 'summary.md'}")
    print(f"MEASUREMENT_COUNT={len(stats)}")


if __name__ == "__main__":
    main()
