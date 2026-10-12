from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from _bootstrap import preload_julia

preload_julia()

import numpy as np

from homotopy_path_learning.config import build_env, build_system_spec, load_config
from homotopy_path_learning.envs.costs import tracking_cost
from homotopy_path_learning.evaluation.datasets import fixed_evaluation_seeds
from homotopy_path_learning.evaluation.metrics import write_rows
from homotopy_path_learning.rl.train_utils import git_dirty, git_value, require_clean_git
from homotopy_path_learning.systems.symmetry import enumerate_symmetries, transform_coefficients

FIELDS = [
    "instance_id",
    "target_seed",
    "action_kind",
    "element",
    "element_type",
    "J_orig",
    "J_trans",
    "accepted_orig",
    "rejected_orig",
    "accepted_trans",
    "rejected_trans",
    "success_orig",
    "success_trans",
    "abs_diff",
    "rel_diff",
]
SUMMARY_FIELDS = [
    "element_type",
    "n_pairs",
    "fraction_exactly_equal",
    "fraction_rel_diff_le_1pct",
    "max_rel_diff",
    "mean_abs_diff",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check numerically that tracking cost is invariant under the Pham symmetry group."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-instances", type=int, default=None)
    parser.add_argument("--random-actions", type=int, default=2)
    parser.add_argument("--action-seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    require_clean_git(allow_dirty=args.allow_dirty)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    config = load_config(args.config)
    spec = build_system_spec(config)
    elements = [g for g in enumerate_symmetries(spec) if not g.is_identity]
    seeds = fixed_evaluation_seeds(
        seed=config.evaluation.seed,
        num_instances=config.evaluation.num_instances if args.num_instances is None else args.num_instances,
    )
    start_time = datetime.now(timezone.utc).isoformat()
    timer = time.time()
    env = build_env(config)
    rows = []
    try:
        action_dim = int(env.action_space.shape[0])
        limit = config.path.action_limit

        def track(points: np.ndarray):
            result = env.backend.track(points)
            cost = tracking_cost(
                result, reject_weight=env.reject_weight, failure_penalty=env.failure_penalty
            ).mean
            return result, cost

        for instance_id, seed in enumerate(seeds):
            env.reset(seed=seed)
            context = env.current_episode_context()
            target = np.array(context.target_coefficients, dtype=np.complex128, copy=True)
            rng = np.random.default_rng([int(args.action_seed), int(seed)])
            actions = [("zero", np.zeros(action_dim))] + [
                (f"random{i}", rng.uniform(-limit, limit, size=action_dim)) for i in range(args.random_actions)
            ]
            for kind, action in actions:
                points = env.parameterization.build_control_points(env.start_coefficients, target, action)
                result0, cost0 = track(points)
                for g in elements:
                    result1, cost1 = track(transform_coefficients(spec, g, points))
                    diff = abs(cost1 - cost0)
                    rows.append(
                        {
                            "instance_id": instance_id,
                            "target_seed": int(seed),
                            "action_kind": kind,
                            "element": g.label,
                            "element_type": g.kind,
                            "J_orig": cost0,
                            "J_trans": cost1,
                            "accepted_orig": int(result0.accepted_steps),
                            "rejected_orig": int(result0.rejected_steps),
                            "accepted_trans": int(result1.accepted_steps),
                            "rejected_trans": int(result1.rejected_steps),
                            "success_orig": int(bool(result0.success)),
                            "success_trans": int(bool(result1.success)),
                            "abs_diff": diff,
                            "rel_diff": diff / cost0 if cost0 > 0 else float("nan"),
                        }
                    )
    finally:
        env.close()

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["element_type"]].append(row)
        groups["overall"].append(row)
    summary = []
    for name in [*sorted(k for k in groups if k != "overall"), "overall"]:
        items = groups[name]
        rel = np.array([r["rel_diff"] for r in items], dtype=np.float64)
        summary.append(
            {
                "element_type": name,
                "n_pairs": len(items),
                "fraction_exactly_equal": float(np.mean([r["abs_diff"] == 0.0 for r in items])),
                "fraction_rel_diff_le_1pct": float(np.mean(rel <= 0.01)),
                "max_rel_diff": float(np.nanmax(rel)),
                "mean_abs_diff": float(np.mean([r["abs_diff"] for r in items])),
            }
        )
    output_dir.mkdir(parents=True)
    write_rows(output_dir / "symmetry_check.csv", rows, fieldnames=FIELDS)
    write_rows(output_dir / "symmetry_summary.csv", summary, fieldnames=SUMMARY_FIELDS)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "config": args.config,
                "num_instances": len(seeds),
                "random_actions": args.random_actions,
                "action_seed": args.action_seed,
                "n_symmetry_elements": len(elements) + 1,
                "git_commit": git_value(["rev-parse", "HEAD"]),
                "git_dirty": git_dirty(),
                "start_time_utc": start_time,
                "end_time_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.time() - timer,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    for item in summary:
        print(
            f"{item['element_type']}: n={item['n_pairs']} exact={item['fraction_exactly_equal']:.3f} "
            f"within1%={item['fraction_rel_diff_le_1pct']:.3f} max_rel={item['max_rel_diff']:.4f}"
        )
    print(f"OUTPUT_DIR={output_dir}")


if __name__ == "__main__":
    main()
