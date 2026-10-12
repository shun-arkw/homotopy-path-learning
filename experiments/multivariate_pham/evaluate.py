from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import preload_julia

preload_julia()

from homotopy_path_learning.config import load_config
from homotopy_path_learning.evaluation import evaluate_checkpoint
from homotopy_path_learning.rl.train_utils import refresh_metadata_output_files, write_metadata

DEFAULT_SUBDIR = "evaluation"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Phase 6 checkpoint.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-instances", type=int, default=None)
    parser.add_argument("--evaluation-seed", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-subdir",
        default=DEFAULT_SUBDIR,
        help="Write run-dir/<NAME>/evaluation.csv. A non-default NAME is never overwritten.",
    )
    args = parser.parse_args()
    if Path(args.output_subdir).name != args.output_subdir or args.output_subdir in ("", ".", ".."):
        parser.error("--output-subdir must be a plain directory name.")

    run_dir = Path(args.run_dir)
    config = load_config(run_dir / "config.yaml")
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    output_path = run_dir / args.output_subdir / "evaluation.csv"
    if args.output_subdir != DEFAULT_SUBDIR and output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing evaluation: {output_path}")
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
    if args.output_subdir != DEFAULT_SUBDIR:
        metadata_path = run_dir / "metadata.json"
        updated = json.loads(metadata_path.read_text(encoding="utf-8"))
        updated.setdefault("extra_evaluations", []).append(
            {
                "subdir": args.output_subdir,
                "evaluation_seed": config.evaluation.seed if args.evaluation_seed is None else args.evaluation_seed,
                "num_instances": len(rows) // 2,  # Linear + LearnedBezier row per instance
                "checkpoint": str(args.checkpoint),
            }
        )
        write_metadata(metadata_path, updated)
    print(f"EVALUATION_CSV={output_path}")
    print(f"EVALUATION_ROWS={len(rows)}")


if __name__ == "__main__":
    main()
