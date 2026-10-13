from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import preload_julia

preload_julia()

from homotopy_path_learning.config import build_env, load_config, save_config
from homotopy_path_learning.evaluation.metrics import write_rows
from homotopy_path_learning.rl.parallel import EnvPool
from homotopy_path_learning.rl.train_utils import (
    collect_metadata,
    refresh_metadata_output_files,
    require_clean_git,
    write_metadata,
)

SCREEN_FIELDS = ["target_seed", "linear_cost", "linear_success", "accepted_steps", "rejected_steps"]
CHUNK = 256


def screening_row(seed: int, context) -> dict[str, object]:
    """One ``linear_costs.csv`` row from an episode context (mean cost, step totals over paths)."""

    return {
        "target_seed": int(seed),
        "linear_cost": float(context.linear_cost.mean),
        "linear_success": int(bool(context.linear_result.success)),
        "accepted_steps": int(context.linear_result.accepted_steps),
        "rejected_steps": int(context.linear_result.rejected_steps),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track only the linear (zero-action) path for consecutive target seeds."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--start-seed", type=int, required=True)
    parser.add_argument("--num-candidates", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1, help="Worker processes (1 = sequential).")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if args.start_seed < 0:
        parser.error("--start-seed must be nonnegative.")
    if args.num_candidates <= 0 or args.workers < 1:
        parser.error("--num-candidates and --workers must be positive.")

    require_clean_git(allow_dirty=args.allow_dirty)
    config = load_config(args.config)
    config = replace(
        config,
        experiment=replace(config.experiment, id=args.experiment_id, name=args.run_id),
        output=replace(config.output, root=args.output_root or config.output.root),
    )
    run_dir = Path(config.output.root) / config.experiment.id / args.run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    save_config(config, run_dir / "config.yaml")
    start_time = datetime.now(timezone.utc).isoformat()

    seeds = [args.start_seed + i for i in range(args.num_candidates)]
    rows: list[dict[str, object]] = []
    env = build_env(config)
    pool = EnvPool(config, args.workers) if args.workers > 1 else None
    try:
        if pool is not None:
            for begin in range(0, len(seeds), CHUNK):
                chunk = seeds[begin : begin + CHUNK]
                _, contexts = pool.reset_many(chunk)
                rows.extend(screening_row(seed, context) for seed, context in zip(chunk, contexts))
        else:
            for seed in seeds:
                env.reset(seed=seed)
                rows.append(screening_row(seed, env.current_episode_context()))
        write_rows(run_dir / "linear_costs.csv", rows, fieldnames=SCREEN_FIELDS)
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
    metadata["screen_linear"] = {
        "source_config": args.config,
        "start_seed": args.start_seed,
        "num_candidates": args.num_candidates,
        "workers": args.workers,
    }
    write_metadata(run_dir / "metadata.json", metadata)
    refresh_metadata_output_files(run_dir)
    print(f"candidates={len(rows)} linear_success={sum(int(r['linear_success']) for r in rows)}")
    print(f"RUN_DIR={run_dir}")


if __name__ == "__main__":
    main()
