from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import preload_julia

preload_julia()

from homotopy_path_learning.config import load_config
from homotopy_path_learning.evaluation.analyzer import generate_summary_markdown
from homotopy_path_learning.rl.train_utils import refresh_metadata_output_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Markdown analysis from benchmark CSV.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config = load_config(run_dir / "config.yaml")
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    checkpoint = args.checkpoint or str(run_dir / "checkpoints" / "last.pt")
    output_path = generate_summary_markdown(
        config=config,
        run_id=str(metadata["run_id"]),
        benchmark_csv=run_dir / "benchmark" / "benchmark.csv",
        checkpoint=checkpoint,
        git_commit=str(metadata["git_commit"]),
        output_path=run_dir / "analysis" / "summary.md",
    )
    refresh_metadata_output_files(run_dir)
    print(f"SUMMARY_MD={output_path}")


if __name__ == "__main__":
    main()
