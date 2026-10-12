"""Compare policy runs on a common problem set (paired bootstrap over problems).

Each ``--run-dir`` must contain ``evaluation/evaluation.csv`` (Linear and
LearnedBezier rows) over the same target seeds. ``--group NAME=DIR1,DIR2,...``
averages the per-problem policy costs over its member runs (e.g. training
seeds) before computing metrics. ``--pair A,B`` reports ``B - A`` for runs or
groups, with 95% percentile bootstrap intervals over problems.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from _bootstrap import preload_julia

preload_julia()

import numpy as np

from homotopy_path_learning.evaluation.compare import (
    METRICS,
    average_group,
    metrics,
    paired_bootstrap_difference,
)
from homotopy_path_learning.evaluation.metrics import read_rows, write_rows
from homotopy_path_learning.rl.train_utils import git_value

SUMMARY_FIELDS = ["name", "kind", "members", "n_problems", "mean_cost", "delta_j_percent", "positive_rate", "captured_fraction"]
PAIR_FIELDS = ["a", "b", "metric", "value_a", "value_b", "diff", "ci_low", "ci_high", "n_problems", "n_boot"]
LINEAR_ATOL = 1.0e-9


def load_run(directory: Path, subdir: str = "evaluation") -> tuple[list[int], np.ndarray, np.ndarray]:
    """Return ``(target_seeds, linear_costs, policy_costs)`` sorted by target seed."""

    path = directory / subdir / "evaluation.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist.")
    linear: dict[int, float] = {}
    policy: dict[int, float] = {}
    for row in read_rows(path):
        seed = int(row["target_seed"])
        if row["method"] == "Linear":
            linear[seed] = float(row["mean_cost"])
        elif row["method"] == "LearnedBezier":
            policy[seed] = float(row["mean_cost"])
    seeds = sorted(linear)
    if not seeds or sorted(policy) != seeds:
        raise ValueError(f"{path} must contain Linear and LearnedBezier rows for the same seeds.")
    return seeds, np.array([linear[s] for s in seeds]), np.array([policy[s] for s in seeds])


def load_oracle(directory: Path) -> tuple[list[int], np.ndarray, np.ndarray]:
    rows = read_rows(directory / "oracle.csv")
    by_seed = {int(r["target_seed"]): (float(r["linear_cost"]), float(r["oracle_cost"])) for r in rows}
    seeds = sorted(by_seed)
    return seeds, np.array([by_seed[s][0] for s in seeds]), np.array([by_seed[s][1] for s in seeds])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", action="append", default=[], help="Run with evaluation/evaluation.csv (repeatable).")
    parser.add_argument("--group", action="append", default=[], help="NAME=DIR1,DIR2,... (repeatable).")
    parser.add_argument("--oracle-run-dir", default=None, help="Oracle run with oracle.csv on the same seeds.")
    parser.add_argument("--evaluation-subdir", default="evaluation", help="Read run-dir/<NAME>/evaluation.csv.")
    parser.add_argument("--pair", action="append", default=[], help="A,B: report B - A (run or group names).")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if args.bootstrap <= 0:
        parser.error("--bootstrap must be positive.")

    groups: dict[str, list[str]] = {}
    for item in args.group:
        name, _, members = item.partition("=")
        dirs = [m for m in members.split(",") if m]
        if not name or not dirs:
            parser.error(f"--group must look like NAME=DIR1,DIR2,...; got {item!r}.")
        if name in groups:
            parser.error(f"duplicate group name {name!r}.")
        groups[name] = dirs
    run_dirs = list(dict.fromkeys(args.run_dir + [d for dirs in groups.values() for d in dirs]))
    if not run_dirs:
        parser.error("give at least one --run-dir or --group.")
    names = {Path(d).name: d for d in run_dirs}
    if len(names) != len(run_dirs):
        parser.error("run directories must have distinct basenames (they are used as names).")
    if set(names) & set(groups):
        parser.error("group names must differ from run names.")

    seeds: list[int] | None = None
    linear: np.ndarray | None = None
    costs: dict[str, np.ndarray] = {}
    for name, directory in names.items():
        run_seeds, run_linear, run_policy = load_run(Path(directory), args.evaluation_subdir)
        if seeds is None:
            seeds, linear = run_seeds, run_linear
        else:
            if run_seeds != seeds:
                raise ValueError(f"run {directory} was evaluated on different target seeds.")
            if not np.allclose(run_linear, linear, rtol=0.0, atol=LINEAR_ATOL):
                raise ValueError(f"Linear costs of run {directory} differ from the other runs.")
        costs[name] = run_policy
    assert seeds is not None and linear is not None

    oracle: np.ndarray | None = None
    if args.oracle_run_dir is not None:
        oracle_seeds, oracle_linear, oracle = load_oracle(Path(args.oracle_run_dir))
        if oracle_seeds != seeds:
            raise ValueError("oracle run was evaluated on different target seeds.")
        if not np.allclose(oracle_linear, linear, rtol=0.0, atol=LINEAR_ATOL):
            raise ValueError("Linear costs of the oracle run differ from the evaluation runs.")

    entities: dict[str, tuple[str, list[str], np.ndarray]] = {}
    for name in names:
        entities[name] = ("run", [names[name]], costs[name])
    for name, dirs in groups.items():
        entities[name] = ("group", dirs, average_group([costs[Path(d).name] for d in dirs]))

    summary_rows = []

    def add_summary(name: str, kind: str, members: list[str], values: np.ndarray) -> None:
        m = metrics(values, linear, oracle)
        summary_rows.append(
            {
                "name": name,
                "kind": kind,
                "members": ";".join(members),
                "n_problems": len(seeds),
                "mean_cost": float(values.mean()),
                "delta_j_percent": m["delta_j_percent"],
                "positive_rate": m["positive_rate"],
                "captured_fraction": "" if oracle is None else m["captured_fraction"],
            }
        )

    add_summary("Linear", "baseline", [], linear)
    if oracle is not None:
        add_summary("Oracle", "baseline", [], oracle)
    for name, (kind, members, values) in entities.items():
        add_summary(name, kind, members, values)

    pair_rows = []
    rng = np.random.default_rng(args.seed)
    for item in args.pair:
        parts = item.split(",")
        if len(parts) != 2 or any(p not in entities for p in parts):
            parser.error(f"--pair must be 'A,B' with known run/group names; got {item!r}.")
        a, b = parts
        result = paired_bootstrap_difference(
            entities[a][2], entities[b][2], linear, oracle, n_boot=args.bootstrap, rng=rng
        )
        for metric in METRICS:
            if metric == "captured_fraction" and oracle is None:
                continue
            r = result[metric]
            pair_rows.append(
                {
                    "a": a, "b": b, "metric": metric, "value_a": r["a"], "value_b": r["b"],
                    "diff": r["diff"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                    "n_problems": len(seeds), "n_boot": args.bootstrap,
                }
            )

    output_dir.mkdir(parents=True)
    write_rows(output_dir / "comparison_summary.csv", summary_rows, fieldnames=SUMMARY_FIELDS)
    write_rows(output_dir / "comparison_pairs.csv", pair_rows, fieldnames=PAIR_FIELDS)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "utc_time": datetime.now(timezone.utc).isoformat(),
                "git_commit": git_value(["rev-parse", "HEAD"]),
                "run_dirs": run_dirs,
                "groups": groups,
                "evaluation_subdir": args.evaluation_subdir,
                "oracle_run_dir": args.oracle_run_dir,
                "pairs": args.pair,
                "bootstrap": args.bootstrap,
                "seed": args.seed,
                "n_problems": len(seeds),
                "target_seed_range": [seeds[0], seeds[-1]],
                "difference_convention": "B - A; delta_j in percentage points",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for row in summary_rows:
        print(
            f"{row['name']}: dJ={row['delta_j_percent']:+.2f}% positive={row['positive_rate']:.3f} "
            f"captured={row['captured_fraction']}"
        )
    for row in pair_rows:
        print(f"{row['b']} - {row['a']} [{row['metric']}]: {row['diff']:+.4f} CI [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]")
    print(f"OUTPUT_DIR={output_dir}")


if __name__ == "__main__":
    main()
