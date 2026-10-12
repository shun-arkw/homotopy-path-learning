from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from _bootstrap import preload_julia

preload_julia()

import numpy as np

from homotopy_path_learning.config import build_env, load_config
from homotopy_path_learning.evaluation.metrics import read_rows, write_rows
from homotopy_path_learning.evaluation.oracle import near_best_action_stats
from homotopy_path_learning.rl.train_utils import git_value

MODE_FIELDS = [
    "target_seed",
    "linear_cost",
    "n_runs",
    "n_good",
    "best_cost_spread_percent",
    "mean_pairwise_cosine",
    "mean_pairwise_distance",
    "max_midpoint_excess_percent",
    "mean_midpoint_excess_percent",
]
SUMMARY_FIELDS = ["metric", "value"]


def _nanmean(values) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure multimodality of per-instance oracle actions across search seeds."
    )
    parser.add_argument("--oracle-run-dir", action="append", required=True)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--num-instances", type=int, default=None)
    parser.add_argument("--no-midpoint", action="store_true")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if len(args.oracle_run_dir) < 2:
        parser.error("at least two --oracle-run-dir are required.")
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")

    runs = []
    for directory in args.oracle_run_dir:
        path = Path(directory)
        runs.append((read_rows(path / "oracle.csv"), np.load(path / "oracle_actions.npy")))
    seeds = [int(row["target_seed"]) for row in runs[0][0]]
    for directory, (rows, actions) in zip(args.oracle_run_dir, runs):
        if [int(row["target_seed"]) for row in rows] != seeds:
            raise ValueError(f"{directory} has different target seeds than {args.oracle_run_dir[0]}.")
        if actions.shape[0] != len(rows):
            raise ValueError(f"{directory}: oracle_actions.npy and oracle.csv lengths differ.")
        for key in ("latent_dim", "action_dim"):
            if rows[0][key] != runs[0][0][0][key]:
                raise ValueError(f"{directory} has a different {key}.")
    config = load_config(Path(args.oracle_run_dir[0]) / "config.yaml")
    count = len(seeds) if args.num_instances is None else min(int(args.num_instances), len(seeds))

    env = None if args.no_midpoint else build_env(config)
    out_rows = []
    try:
        for i in range(count):
            costs = np.array([float(rows[i]["oracle_cost"]) for rows, _ in runs])
            actions = np.stack([acts[i] for _, acts in runs])
            stats = near_best_action_stats(costs, actions, tolerance=args.tolerance)
            linear = float(runs[0][0][i]["linear_cost"])
            max_excess = mean_excess = float("nan")
            if env is not None:
                env.reset(seed=seeds[i])
                context = env.current_episode_context()
                good = np.flatnonzero(costs <= costs.min() * (1.0 + args.tolerance))
                excesses = []
                for a in range(len(good)):
                    for b in range(a + 1, len(good)):
                        mid = (actions[good[a]] + actions[good[b]]) / 2.0
                        j_mid = float(env.evaluate_action_against_context(context, mid).tracking_cost.mean)
                        excesses.append(
                            (j_mid - max(costs[good[a]], costs[good[b]])) / linear * 100.0
                        )
                if excesses:
                    max_excess, mean_excess = float(np.max(excesses)), float(np.mean(excesses))
            out_rows.append(
                {
                    "target_seed": seeds[i],
                    "linear_cost": linear,
                    **stats,
                    "max_midpoint_excess_percent": max_excess,
                    "mean_midpoint_excess_percent": mean_excess,
                }
            )
    finally:
        if env is not None:
            env.close()

    output_dir.mkdir(parents=True)
    write_rows(output_dir / "modes.csv", out_rows, fieldnames=MODE_FIELDS)
    max_excess_all = np.array([r["max_midpoint_excess_percent"] for r in out_rows], dtype=np.float64)
    summary = [
        {"metric": "n_instances", "value": len(out_rows)},
        {"metric": "mean_best_cost_spread_percent", "value": _nanmean([r["best_cost_spread_percent"] for r in out_rows])},
        {"metric": "mean_n_good", "value": _nanmean([r["n_good"] for r in out_rows])},
        {"metric": "mean_pairwise_cosine", "value": _nanmean([r["mean_pairwise_cosine"] for r in out_rows])},
        {"metric": "mean_pairwise_distance", "value": _nanmean([r["mean_pairwise_distance"] for r in out_rows])},
        {"metric": "mean_max_midpoint_excess_percent", "value": _nanmean(max_excess_all)},
        {"metric": "mean_mean_midpoint_excess_percent", "value": _nanmean([r["mean_midpoint_excess_percent"] for r in out_rows])},
        {"metric": "fraction_n_good_ge_2", "value": float(np.mean([r["n_good"] >= 2 for r in out_rows]))},
        {
            "metric": "fraction_max_midpoint_excess_gt_2",
            "value": float(np.mean(np.nan_to_num(max_excess_all, nan=-np.inf) > 2.0)),
        },
    ]
    write_rows(output_dir / "modes_summary.csv", summary, fieldnames=SUMMARY_FIELDS)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "oracle_run_dirs": list(args.oracle_run_dir),
                "tolerance": args.tolerance,
                "num_instances": len(out_rows),
                "midpoint_test": not args.no_midpoint,
                "git_commit": git_value(["rev-parse", "HEAD"]),
                "time_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    for item in summary:
        print(f"{item['metric']}={item['value']}")
    print(f"OUTPUT_DIR={output_dir}")


if __name__ == "__main__":
    main()
