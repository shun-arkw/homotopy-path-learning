"""Fixed evaluation seed helpers."""

from __future__ import annotations

import json
from pathlib import Path


def fixed_evaluation_seeds(*, seed: int, num_instances: int) -> list[int]:
    """Return deterministic target seeds ``seed + instance_index``."""

    if int(num_instances) <= 0:
        raise ValueError("num_instances must be positive.")
    return [int(seed) + index for index in range(int(num_instances))]


def save_evaluation_seeds(path: str | Path, *, seed: int, num_instances: int) -> list[int]:
    """Write fixed evaluation seeds to JSON and return them."""

    seeds = fixed_evaluation_seeds(seed=seed, num_instances=num_instances)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "seed": int(seed),
                "num_instances": int(num_instances),
                "rule": "seed + instance_index",
                "seeds": seeds,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    return seeds


__all__ = ["fixed_evaluation_seeds", "save_evaluation_seeds"]
