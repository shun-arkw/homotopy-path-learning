from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import preload_julia

preload_julia()

import numpy as np

from homotopy_path_learning.config import build_env, load_config, save_config
from homotopy_path_learning.evaluation.metrics import read_rows, write_rows
from homotopy_path_learning.evaluation.oracle import (
    ORACLE_FIELDS,
    ORACLE_SUMMARY_FIELDS,
    learned_costs_from_evaluation,
    resolve_latent_dim,
    run_oracle,
    summarize_oracle,
    with_latent_dim,
)
from homotopy_path_learning.rl.train_utils import (
    collect_metadata,
    refresh_metadata_output_files,
    require_clean_git,
    write_metadata,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Directly optimize the latent action per evaluation instance (oracle baseline)."
    )
    parser.add_argument("--config", required=True, help="Config of the PPO run to compare against.")
    parser.add_argument("--experiment-id", required=True, help="Experiment ID of the oracle run.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--latent-dim", default=None, help="Integer or 'full'; default keeps the config.")
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--n-random", type=int, default=100)
    parser.add_argument("--search-seed", type=int, default=0)
    parser.add_argument("--num-instances", type=int, default=None)
    parser.add_argument("--compare-run-dir", action="append", default=[])
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    require_clean_git(allow_dirty=args.allow_dirty)
    source_config = load_config(args.config)
    latent_dim = resolve_latent_dim(source_config, args.latent_dim)
    config = with_latent_dim(source_config, latent_dim)
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

    result = run_oracle(
        config,
        budget=args.budget,
        n_random=args.n_random,
        search_seed=args.search_seed,
        num_instances=args.num_instances,
    )
    write_rows(run_dir / "oracle.csv", result.rows, fieldnames=ORACLE_FIELDS)
    np.save(run_dir / "oracle_actions.npy", result.best_actions)
    np.save(run_dir / "oracle_cost_traces.npy", result.cost_traces)
    np.save(run_dir / "latent_basis.npy", result.basis)

    learned = {}
    for compare_dir in args.compare_run_dir:
        compare_path = Path(compare_dir)
        evaluation_rows = read_rows(compare_path / "evaluation" / "evaluation.csv")
        learned[compare_path.name] = learned_costs_from_evaluation(evaluation_rows, result.rows)
    summaries = summarize_oracle(result.rows, learned)
    write_rows(run_dir / "oracle_summary.csv", summaries, fieldnames=ORACLE_SUMMARY_FIELDS)

    env = build_env(config)
    try:
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
    metadata["oracle"] = {
        "source_config": args.config,
        "latent_dim": latent_dim,
        "budget": args.budget,
        "n_random": args.n_random,
        "search_seed": args.search_seed,
        "search_rng_rule": "default_rng([search_seed, target_seed])",
        "compare_run_dirs": list(args.compare_run_dir),
    }
    write_metadata(run_dir / "metadata.json", metadata)
    refresh_metadata_output_files(run_dir)

    for summary in summaries:
        print(
            f"{summary['method']}: mean_cost={summary['mean_cost']:.4f} "
            f"delta_j={summary['delta_j_percent']:+.2f}% positive_rate={summary['positive_rate']:.2f} "
            f"captured={summary['captured_fraction']}"
        )
    print(f"RUN_DIR={run_dir}")


if __name__ == "__main__":
    main()
