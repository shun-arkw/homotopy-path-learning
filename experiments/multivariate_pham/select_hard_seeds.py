from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import preload_julia

preload_julia()

from homotopy_path_learning.evaluation.metrics import read_rows
from homotopy_path_learning.evaluation.selection import parse_split, select_hard_seeds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select hard target seeds (large linear-path cost) from screening runs."
    )
    parser.add_argument("--screen-run-dir", action="append", required=True, help="screen_linear.py run; repeatable.")
    parser.add_argument("--calibration", type=int, required=True, help="Number of lowest seeds used for the threshold.")
    parser.add_argument("--quantile", type=float, required=True)
    parser.add_argument("--split", action="append", required=True, help="name:count; repeatable, assigned in order.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    try:
        splits = [parse_split(text) for text in args.split]
    except ValueError as error:
        parser.error(str(error))
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")

    candidates = []
    owner: dict[int, str] = {}
    for run_dir in args.screen_run_dir:
        for row in read_rows(Path(run_dir) / "linear_costs.csv"):
            seed = int(row["target_seed"])
            if seed in owner:
                raise ValueError(f"seed {seed} appears in both {owner[seed]} and {run_dir}.")
            owner[seed] = run_dir
            candidates.append(row)

    selection = select_hard_seeds(
        candidates, calibration=args.calibration, quantile=args.quantile, splits=splits
    )
    output_dir.mkdir(parents=True)
    split_info = {}
    for name, seeds in selection.splits.items():
        (output_dir / f"{name}.txt").write_text("".join(f"{seed}\n" for seed in seeds), encoding="utf-8")
        split_info[name] = {"count": len(seeds), "seed_min": seeds[0], "seed_max": seeds[-1]}
    record = {
        "threshold": selection.threshold,
        "quantile": args.quantile,
        "calibration": {
            "count": selection.calibration_count,
            "seed_min": selection.calibration_seed_range[0],
            "seed_max": selection.calibration_seed_range[1],
        },
        "n_candidates": selection.n_candidates,
        "n_eligible": selection.n_eligible,
        "splits": split_info,
        "screen_run_dirs": list(args.screen_run_dir),
    }
    (output_dir / "selection.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"threshold={selection.threshold:.6g} eligible={selection.n_eligible}")
    print(f"OUTPUT_DIR={output_dir}")


if __name__ == "__main__":
    main()
