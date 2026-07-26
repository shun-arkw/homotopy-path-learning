from __future__ import annotations

import argparse

from _bootstrap import preload_julia

preload_julia()

from homotopy_path_learning.config import load_config
from homotopy_path_learning.rl.train_utils import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on BezierPhamEnv.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    result = run_training(
        config=config,
        run_id=args.run_id,
        config_path=args.config,
        total_timesteps=args.total_timesteps,
        output_root=args.output_root,
        device=args.device,
        allow_dirty=args.allow_dirty,
    )
    print(f"RUN_DIR={result.run_dir}")
    print(f"GLOBAL_STEP={result.global_step}")
    print(f"UPDATES={result.updates}")
    print(f"ELAPSED_SECONDS={result.elapsed_seconds:.6f}")


if __name__ == "__main__":
    main()
