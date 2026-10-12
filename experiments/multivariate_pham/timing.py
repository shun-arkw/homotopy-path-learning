"""Inference-time comparison of trained policies and the linear path.

For every evaluation problem and every method the wall-clock time is measured
with ``time.perf_counter`` and the median over ``--repeats`` repetitions is
reported:

* ``Linear``: build the linear Bezier control points and run the backend track.
* each ``--run-dir``: build the observation, run the policy forward pass (batch
  size 1, ``torch.no_grad``), build the control points and run the backend track.

Run it as a single process on an otherwise idle machine, with
``OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1`` (and no other
experiment running), otherwise the timings are not comparable.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from _bootstrap import preload_julia

preload_julia()

import numpy as np
import torch

from homotopy_path_learning.config import build_env, build_system_spec, load_config
from homotopy_path_learning.envs.costs import tracking_cost
from homotopy_path_learning.evaluation.datasets import fixed_evaluation_seeds
from homotopy_path_learning.evaluation.metrics import write_rows
from homotopy_path_learning.evaluation.oracle import resolve_latent_dim, with_latent_dim
from homotopy_path_learning.paths.parameterization import (
    complex_coefficients_to_real,
    linear_interpolation_control_points,
)
from homotopy_path_learning.rl.checkpoint import build_agent_from_checkpoint, load_checkpoint, validate_checkpoint
from homotopy_path_learning.rl.train_utils import git_value

TIMING_FIELDS = [
    "instance_id", "target_seed", "method", "repeats",
    "median_seconds", "min_seconds", "median_forward_seconds", "mean_cost",
]
SUMMARY_FIELDS = [
    "method", "n_problems", "mean_seconds", "median_seconds", "mean_forward_seconds",
    "ratio_to_linear", "mean_cost", "delta_j_percent", "positive_rate",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--latent-dim", default=None, help="Integer or 'full'; default keeps the config.")
    parser.add_argument("--run-dir", action="append", default=[], help="Trained run with checkpoints/last.pt (repeatable).")
    parser.add_argument("--num-instances", type=int, default=None)
    parser.add_argument("--evaluation-seed", type=int, default=None)
    parser.add_argument("--device", default="cpu", help="Policy device (tracking always runs on the CPU).")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if args.repeats <= 0 or args.warmup < 0:
        parser.error("--repeats must be positive and --warmup nonnegative.")
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    names = [Path(d).name for d in args.run_dir]
    if len(set(names)) != len(names) or "Linear" in names:
        parser.error("run directories need distinct basenames different from 'Linear'.")

    base = load_config(args.config)
    config = with_latent_dim(base, resolve_latent_dim(base, args.latent_dim))
    seeds = fixed_evaluation_seeds(
        seed=config.evaluation.seed if args.evaluation_seed is None else args.evaluation_seed,
        num_instances=config.evaluation.num_instances if args.num_instances is None else args.num_instances,
    )
    torch.set_grad_enabled(False)
    env = build_env(config)
    rows: list[dict] = []
    try:
        spec = build_system_spec(config)
        agents = {}
        for name, directory in zip(names, args.run_dir):
            checkpoint = load_checkpoint(Path(directory) / "checkpoints" / "last.pt", map_location=args.device)
            validate_checkpoint(
                checkpoint, config=config, system_spec=spec,
                observation_shape=tuple(env.observation_space.shape), action_space=env.action_space,
            )
            agents[name] = build_agent_from_checkpoint(checkpoint, action_space=env.action_space, device=args.device)

        start = env.start_coefficients
        free = env.free_indices
        backend = env.backend
        degree = env.bezier_degree

        def cost_of(result) -> float:
            return float(tracking_cost(result, reject_weight=env.reject_weight, failure_penalty=env.failure_penalty).mean)

        def run_linear(target: np.ndarray) -> tuple[float, float, float]:
            begin = time.perf_counter()
            points = linear_interpolation_control_points(start, target, degree)
            result = backend.track(points)
            elapsed = time.perf_counter() - begin
            return elapsed, 0.0, cost_of(result)

        def run_policy(agent, target: np.ndarray) -> tuple[float, float, float]:
            begin = time.perf_counter()
            observation = complex_coefficients_to_real(target[free]).astype(np.float32)
            action = agent.deterministic_action(torch.from_numpy(observation).unsqueeze(0).to(args.device))
            action = action.squeeze(0).cpu().numpy().astype(np.float64)
            if args.device != "cpu":
                torch.cuda.synchronize()
            forward = time.perf_counter() - begin
            points = env.parameterization.build_control_points(start, target, action)
            result = backend.track(points)
            elapsed = time.perf_counter() - begin
            return elapsed, forward, cost_of(result)

        methods = [("Linear", run_linear)] + [(n, (lambda t, a=agents[n]: run_policy(a, t))) for n in names]
        targets = []
        for seed in seeds:
            env.reset(seed=seed)
            targets.append(np.array(env.current_episode_context().target_coefficients, copy=True))
        for _, fn in methods:
            for _ in range(args.warmup):
                fn(targets[0])
        for instance_id, (seed, target) in enumerate(zip(seeds, targets)):
            for name, fn in methods:
                samples = [fn(target) for _ in range(args.repeats)]
                total = np.array([s[0] for s in samples])
                forward = np.array([s[1] for s in samples])
                rows.append(
                    {
                        "instance_id": instance_id, "target_seed": seed, "method": name,
                        "repeats": args.repeats, "median_seconds": float(np.median(total)),
                        "min_seconds": float(total.min()), "median_forward_seconds": float(np.median(forward)),
                        "mean_cost": samples[-1][2],
                    }
                )
    finally:
        env.close()

    linear_rows = [r for r in rows if r["method"] == "Linear"]
    linear_time = float(np.mean([r["median_seconds"] for r in linear_rows]))
    linear_cost = np.array([r["mean_cost"] for r in linear_rows])
    summary = []
    for name in ["Linear"] + names:
        selected = [r for r in rows if r["method"] == name]
        seconds = np.array([r["median_seconds"] for r in selected])
        cost = np.array([r["mean_cost"] for r in selected])
        summary.append(
            {
                "method": name, "n_problems": len(selected),
                "mean_seconds": float(seconds.mean()), "median_seconds": float(np.median(seconds)),
                "mean_forward_seconds": float(np.mean([r["median_forward_seconds"] for r in selected])),
                "ratio_to_linear": float(seconds.mean()) / linear_time,
                "mean_cost": float(cost.mean()),
                "delta_j_percent": (float(cost.mean()) - float(linear_cost.mean())) / float(linear_cost.mean()) * 100.0,
                "positive_rate": float(np.mean(cost < linear_cost)),
            }
        )
    output_dir.mkdir(parents=True)
    write_rows(output_dir / "timing.csv", rows, fieldnames=TIMING_FIELDS)
    write_rows(output_dir / "timing_summary.csv", summary, fieldnames=SUMMARY_FIELDS)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "utc_time": datetime.now(timezone.utc).isoformat(),
                "git_commit": git_value(["rev-parse", "HEAD"]),
                "config": args.config,
                "run_dirs": list(args.run_dir),
                "target_seeds": [seeds[0], seeds[-1]],
                "num_instances": len(seeds),
                "repeats": args.repeats,
                "warmup": args.warmup,
                "device": args.device,
                "torch_num_threads": torch.get_num_threads(),
                "thread_env": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "JULIA_NUM_THREADS", "OMP_NUM_THREADS")},
                "cpu_count": os.cpu_count(),
                "load_average_at_end": list(os.getloadavg()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for row in summary:
        print(
            f"{row['method']}: mean={row['mean_seconds'] * 1e3:.2f} ms ratio={row['ratio_to_linear']:.3f} "
            f"dJ={row['delta_j_percent']:+.2f}% positive={row['positive_rate']:.2f}"
        )
    print(f"OUTPUT_DIR={output_dir}")


if __name__ == "__main__":
    main()
