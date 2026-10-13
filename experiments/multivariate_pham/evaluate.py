from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import preload_julia

preload_julia()

from homotopy_path_learning.config import load_config
from homotopy_path_learning.evaluation import evaluate_checkpoint
from homotopy_path_learning.evaluation.seeds import read_seed_list, seed_file_record
from homotopy_path_learning.rl.train_utils import refresh_metadata_output_files, write_metadata

DEFAULT_SUBDIR = "evaluation"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Phase 6 checkpoint.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-instances", type=int, default=None)
    parser.add_argument("--evaluation-seed", type=int, default=None)
    parser.add_argument(
        "--target-seeds-file",
        default=None,
        help="Seed-list file (one seed per line); excludes --evaluation-seed and --num-instances.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-subdir",
        default=DEFAULT_SUBDIR,
        help="Write run-dir/<NAME>/evaluation.csv. A non-default NAME is never overwritten.",
    )
    args = parser.parse_args()
    if Path(args.output_subdir).name != args.output_subdir or args.output_subdir in ("", ".", ".."):
        parser.error("--output-subdir must be a plain directory name.")

    target_seeds = None
    if args.target_seeds_file is not None:
        if args.evaluation_seed is not None or args.num_instances is not None:
            parser.error("--target-seeds-file cannot be combined with --evaluation-seed or --num-instances.")
        target_seeds = read_seed_list(args.target_seeds_file)

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
        target_seeds=target_seeds,
    )
    refresh_metadata_output_files(run_dir)
    if args.output_subdir != DEFAULT_SUBDIR:
        metadata_path = run_dir / "metadata.json"
        updated = json.loads(metadata_path.read_text(encoding="utf-8"))
        record = {
            "subdir": args.output_subdir,
            "evaluation_seed": config.evaluation.seed if args.evaluation_seed is None else args.evaluation_seed,
            "num_instances": len(rows) // 2,  # Linear + LearnedBezier row per instance
            "checkpoint": str(args.checkpoint),
        }
        if args.target_seeds_file is not None:
            record["target_seeds_file"] = seed_file_record(args.target_seeds_file)
        updated.setdefault("extra_evaluations", []).append(record)
        write_metadata(metadata_path, updated)
    elif args.target_seeds_file is not None:
        metadata_path = run_dir / "metadata.json"
        updated = json.loads(metadata_path.read_text(encoding="utf-8"))
        updated["evaluation_target_seeds_file"] = seed_file_record(args.target_seeds_file)
        write_metadata(metadata_path, updated)
    print(f"EVALUATION_CSV={output_path}")
    print(f"EVALUATION_ROWS={len(rows)}")


if __name__ == "__main__":
    main()
