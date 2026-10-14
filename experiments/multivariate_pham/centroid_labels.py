"""Centroid labels: average the oracle actions of several search runs over the same targets.

For every target seed in ``[FIRST, FIRST + NUM)`` the actions of all input oracle runs are averaged
(arithmetic mean) and saved as a label run readable by ``behavior_cloning.py --label-run-dir``:
``config.yaml``, ``oracle_observations.npy`` (``(N, obs_dim)``, float32), ``oracle_actions.npy``
(``(N, A)``, float64), ``oracle.csv`` and ``metadata.json``. Input runs are never modified.

``oracle.csv`` keeps, per target, the row of the best run (lowest ``oracle_cost``; ties go to the
earlier ``--run-dir``). ``oracle_cost`` and ``delta_j_percent`` are therefore the best run's values and
NOT the cost of the centroid (unknown); ``best_action_l2_norm`` and ``best_action_max_abs`` are replaced
by the values of the centroid.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from homotopy_path_learning.config import load_config


def _compatibility_key(config) -> dict:
    return {
        "path.bezier_degree": config.path.bezier_degree,
        "path.latent_dim": config.path.latent_dim,
        "path.basis_seed": config.path.basis_seed,
        "path.action_limit": config.path.action_limit,
        "system.degrees": config.system.degrees,
        "system.free_exponents": config.system.free_exponents,
        "sampler.type": config.sampler.type,
        "sampler.bound": config.sampler.bound,
        "tracker": config.tracker,
    }


def centroid_labels(out: str | Path, first: int, num: int, run_dirs: list[str]) -> dict:
    """Write the centroid label run ``out`` and return its metadata dict.

    ``run_dirs`` are oracle runs (at least 2) containing ``config.yaml``, ``oracle.csv``,
    ``oracle_observations.npy`` (``(M, obs_dim)``) and ``oracle_actions.npy`` (``(M, A)``).
    Raises ``ValueError`` if the configs are incompatible, a run lacks one of the seeds
    ``first .. first+num-1`` or has it twice, observations or linear costs of a seed differ between
    runs, the latent bases differ, or a centroid leaves the box ``[-1, 1]``.
    """

    if len(run_dirs) < 2:
        raise ValueError("at least two --run-dir are required.")
    if num < 1 or first < 0:
        raise ValueError("--first must be >= 0 and --num >= 1.")
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {out}")
    seeds = list(range(first, first + num))
    runs = [Path(d) for d in run_dirs]
    base_key = _compatibility_key(load_config(runs[0] / "config.yaml"))
    basis0 = None
    header = None
    per_run = []  # per run: (rows by seed, obs by seed row, actions)
    for run in runs:
        for name in ("config.yaml", "oracle.csv", "oracle_observations.npy", "oracle_actions.npy"):
            if not (run / name).is_file():
                raise FileNotFoundError(f"run {run} is missing {name}.")
        key = _compatibility_key(load_config(run / "config.yaml"))
        for name, value in base_key.items():
            if key[name] != value:
                raise ValueError(f"run {run} differs from {runs[0]} in {name}.")
        basis_path = run / "latent_basis.npy"
        if basis_path.is_file():
            basis = np.load(basis_path)
            if basis0 is None:
                basis0 = (basis, run)
            elif basis.shape != basis0[0].shape or not np.allclose(basis, basis0[0], atol=1.0e-12):
                raise ValueError(f"latent basis of run {run} differs from that of {basis0[1]}.")
        with (run / "oracle.csv").open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
        header = header or fieldnames
        if fieldnames != header:
            raise ValueError(f"run {run} has different oracle.csv columns than {runs[0]}.")
        obs = np.load(run / "oracle_observations.npy")
        act = np.load(run / "oracle_actions.npy")
        if not (obs.shape[0] == act.shape[0] == len(rows)):
            raise ValueError(
                f"run {run} has inconsistent lengths: observations={obs.shape[0]}, actions={act.shape[0]}, rows={len(rows)}."
            )
        index: dict[int, int] = {}
        for i, row in enumerate(rows):
            seed = int(row["target_seed"])
            if first <= seed < first + num:
                if seed in index:
                    raise ValueError(f"run {run} has target seed {seed} more than once.")
                index[seed] = i
        missing = [s for s in seeds if s not in index]
        if missing:
            raise ValueError(f"run {run} is missing target seeds, e.g. {missing[:5]}.")
        per_run.append((rows, obs, act, index))

    observations = []
    labels = np.zeros((num, len(runs), per_run[0][2].shape[1]), dtype=np.float64)
    out_rows = []
    best_costs = []
    linear_costs = []
    for n, seed in enumerate(seeds):
        ref_rows, ref_obs, _, ref_index = per_run[0]
        ref_row = ref_rows[ref_index[seed]]
        best_k, best_cost = 0, float(ref_row["oracle_cost"])
        for k, (rows, obs, act, index) in enumerate(per_run):
            i = index[seed]
            if not np.array_equal(obs[i], ref_obs[ref_index[seed]]):
                raise ValueError(f"observation mismatch for seed {seed} between {runs[0]} and {runs[k]}.")
            if float(rows[i]["linear_cost"]) != float(ref_row["linear_cost"]):
                raise ValueError(f"linear cost mismatch for seed {seed} between {runs[0]} and {runs[k]}.")
            if act.shape[1] != labels.shape[2]:
                raise ValueError(f"run {runs[k]} has a different action dimension.")
            labels[n, k] = act[i]
            cost = float(rows[i]["oracle_cost"])
            if k > 0 and cost < best_cost:
                best_k, best_cost = k, cost
        observations.append(ref_obs[ref_index[seed]])
        best_rows, _, _, best_index = per_run[best_k]
        row = dict(best_rows[best_index[seed]])
        centroid = labels[n].mean(axis=0)
        if np.any(np.abs(centroid) > 1.0 + 1.0e-6):
            raise ValueError(f"centroid of seed {seed} leaves the box [-1, 1].")
        row["best_action_l2_norm"] = float(np.linalg.norm(centroid))
        row["best_action_max_abs"] = float(np.max(np.abs(centroid)))
        out_rows.append(row)
        best_costs.append(best_cost)
        linear_costs.append(float(ref_row["linear_cost"]))
    centroids = labels.mean(axis=1)

    k_runs = len(runs)
    diffs = labels[:, :, None, :] - labels[:, None, :, :]
    distances = np.linalg.norm(diffs, axis=3)
    upper = np.triu_indices(k_runs, k=1)
    mean_pairwise = float(distances[:, upper[0], upper[1]].mean())
    mean_linear = float(np.mean(linear_costs))
    mean_best = float(np.mean(best_costs))
    metadata = {
        "centroid_of": [str(r) for r in runs],
        "n_runs": k_runs,
        "target_seeds": [first, first + num - 1],
        "mean_linear_cost": mean_linear,
        "mean_best_oracle_cost": mean_best,
        "best_delta_j_percent": (mean_best / mean_linear - 1.0) * 100.0,
        "mean_pairwise_label_distance": mean_pairwise,
        "mean_label_norm": float(np.linalg.norm(centroids, axis=1).mean()),
        "oracle_cost_is_best_run": True,
    }

    out.mkdir(parents=True)
    with (out / "oracle.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(out_rows)
    np.save(out / "oracle_observations.npy", np.stack(observations).astype(np.float32))
    np.save(out / "oracle_actions.npy", centroids)
    shutil.copy(runs[0] / "config.yaml", out / "config.yaml")
    if basis0 is not None:
        np.save(out / "latent_basis.npy", basis0[0])
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Average oracle actions of several runs into a centroid label run.")
    parser.add_argument("--out", required=True, help="Output label-run directory (must not exist).")
    parser.add_argument("--first", type=int, required=True, help="First target seed.")
    parser.add_argument("--num", type=int, required=True, help="Number of consecutive target seeds.")
    parser.add_argument("--run-dir", action="append", required=True, help="Input oracle run; repeat (>= 2).")
    args = parser.parse_args()
    metadata = centroid_labels(args.out, args.first, args.num, args.run_dir)
    print(
        f"centroid_labels: n={args.num} K={metadata['n_runs']} linear={metadata['mean_linear_cost']:.3f} "
        f"best={metadata['mean_best_oracle_cost']:.3f} dJ(best)={metadata['best_delta_j_percent']:.2f}% "
        f"pairwise_distance={metadata['mean_pairwise_label_distance']:.4f}"
    )
    print(f"OUT_DIR={args.out}")


if __name__ == "__main__":
    main()
