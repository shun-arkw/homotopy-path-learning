from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import preload_julia

preload_julia()

from homotopy_path_learning.config import load_config
from homotopy_path_learning.evaluation import evaluate_checkpoint
from homotopy_path_learning.rl.train_utils import refresh_metadata_output_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Phase 6 checkpoint.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-instances", type=int, default=None)
    parser.add_argument("--evaluation-seed", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config = load_config(run_dir / "config.yaml")
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    output_path = run_dir / "evaluation" / "evaluation.csv"
    rows = evaluate_checkpoint(
        config=config,
        run_id=str(metadata["run_id"]),
        checkpoint_path=args.checkpoint,
        output_path=output_path,
        num_instances=args.num_instances,
        evaluation_seed=args.evaluation_seed,
        device=args.device,
    )
    refresh_metadata_output_files(run_dir)
    print(f"EVALUATION_CSV={output_path}")
    print(f"EVALUATION_ROWS={len(rows)}")


if __name__ == "__main__":
    main()
