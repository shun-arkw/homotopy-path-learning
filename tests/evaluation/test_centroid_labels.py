from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from homotopy_path_learning.evaluation.oracle import ORACLE_FIELDS

CONFIG = Path("experiments/multivariate_pham/configs/exp-0004-smoke.yaml")
SCRIPTS = Path("experiments/multivariate_pham")
OBS_DIM, ACTION_DIM = 12, 8


def load_script(name: str):
    """Import an experiment script without starting Julia (its ``_bootstrap`` is replaced by a no-op)."""

    stub = types.ModuleType("_bootstrap")
    stub.preload_julia = lambda: None
    saved = sys.modules.get("_bootstrap")
    sys.modules["_bootstrap"] = stub
    try:
        spec = importlib.util.spec_from_file_location(f"_script_{name}", SCRIPTS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if saved is None:
            sys.modules.pop("_bootstrap", None)
        else:
            sys.modules["_bootstrap"] = saved
    return module


def make_run(root: Path, name: str, seeds, *, actions=None, costs=None, obs=None, linear=None) -> Path:
    """Write a synthetic oracle run with rows in the order of ``seeds``."""

    run = root / name
    run.mkdir(parents=True)
    shutil.copy(CONFIG, run / "config.yaml")
    n = len(seeds)
    rng = np.random.default_rng(sum(map(ord, name)))
    obs = np.arange(n * OBS_DIM, dtype=np.float32).reshape(n, OBS_DIM) / 100.0 if obs is None else obs
    actions = rng.uniform(-1, 1, size=(n, ACTION_DIM)) if actions is None else actions
    costs = rng.uniform(1.0, 2.0, size=n) if costs is None else costs
    linear = np.full(n, 3.0) if linear is None else linear
    rows = []
    for i, seed in enumerate(seeds):
        row = {key: "0" for key in ORACLE_FIELDS}
        row.update(
            instance_id=str(i), target_seed=str(seed), linear_cost=repr(float(linear[i])),
            oracle_cost=repr(float(costs[i])), delta_j_percent="-10.0",
            best_action_l2_norm="9.0", best_action_max_abs="9.0",
        )
        rows.append(row)
    with (run / "oracle.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ORACLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    np.save(run / "oracle_observations.npy", obs)
    np.save(run / "oracle_actions.npy", np.asarray(actions, dtype=np.float64))
    return run


def read_csv(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_centroid_is_mean_and_output_is_a_label_run(tmp_path):
    module = load_script("centroid_labels")
    seeds = [10, 11, 12]
    a = make_run(tmp_path, "a", seeds, costs=[1.0, 2.0, 1.5])
    b = make_run(tmp_path, "b", seeds, costs=[1.5, 1.0, 1.5])
    out = tmp_path / "centroid"
    meta = module.centroid_labels(out, 10, 3, [str(a), str(b)])

    act_a, act_b = np.load(a / "oracle_actions.npy"), np.load(b / "oracle_actions.npy")
    centroid = np.load(out / "oracle_actions.npy")
    assert centroid.dtype == np.float64 and centroid.shape == (3, ACTION_DIM)
    np.testing.assert_allclose(centroid, (act_a + act_b) / 2)
    observations = np.load(out / "oracle_observations.npy")
    assert observations.dtype == np.float32
    np.testing.assert_array_equal(observations, np.load(a / "oracle_observations.npy"))
    rows = read_csv(out / "oracle.csv")
    assert [int(r["target_seed"]) for r in rows] == seeds
    assert [r["oracle_cost"] for r in rows] == ["1.0", "1.0", "1.5"]  # best run per seed, ties to the first
    np.testing.assert_allclose([float(r["best_action_l2_norm"]) for r in rows], np.linalg.norm(centroid, axis=1))
    np.testing.assert_allclose([float(r["best_action_max_abs"]) for r in rows], np.abs(centroid).max(axis=1))
    assert (out / "config.yaml").is_file()
    stored = json.loads((out / "metadata.json").read_text())
    assert stored == meta
    assert meta["oracle_cost_is_best_run"] is True and meta["n_runs"] == 2
    assert meta["target_seeds"] == [10, 12]
    assert meta["mean_best_oracle_cost"] == pytest.approx(np.mean([1.0, 1.0, 1.5]))
    assert meta["mean_pairwise_label_distance"] == pytest.approx(np.linalg.norm(act_a - act_b, axis=1).mean())
    assert meta["mean_label_norm"] == pytest.approx(np.linalg.norm(centroid, axis=1).mean())

    # readable by the behavior-cloning label loader (which also checks compatibility)
    bc = load_script("behavior_cloning")
    from homotopy_path_learning.config import load_config

    obs, act, loaded_seeds = bc.load_labels(load_config(CONFIG), [str(out)])
    assert loaded_seeds == seeds and act.shape == (3, ACTION_DIM)
    # inputs untouched
    np.testing.assert_array_equal(np.load(a / "oracle_actions.npy"), act_a)


def test_centroid_rejects_inconsistent_inputs(tmp_path):
    module = load_script("centroid_labels")
    seeds = [10, 11]
    a = make_run(tmp_path, "a", seeds)
    out = lambda name: tmp_path / name
    b_obs = np.load(a / "oracle_observations.npy").copy()
    b_obs[1, 0] += 1.0
    cases = {
        "obs": make_run(tmp_path, "bo", seeds, obs=b_obs),
        "linear": make_run(tmp_path, "bl", seeds, linear=[3.0, 4.0]),
        "missing": make_run(tmp_path, "bm", [10, 12]),
        "dup": make_run(tmp_path, "bd", [10, 10, 11]),
    }
    for name, run in cases.items():
        with pytest.raises(ValueError):
            module.centroid_labels(out("o-" + name), 10, 2, [str(a), str(run)])
        assert not out("o-" + name).exists()
    with pytest.raises(ValueError):
        module.centroid_labels(out("single"), 10, 2, [str(a)])
    good = make_run(tmp_path, "g", [11, 10])  # default observations follow the row index, so seed 10 differs
    with pytest.raises(ValueError):
        module.centroid_labels(out("o-order"), 10, 2, [str(a), str(good)])
    other = make_run(tmp_path, "cfg", seeds)
    text = (other / "config.yaml").read_text()
    assert "bezier_degree: 3" in text
    (other / "config.yaml").write_text(text.replace("bezier_degree: 3", "bezier_degree: 2"))
    with pytest.raises(ValueError):
        module.centroid_labels(out("o-cfg"), 10, 2, [str(a), str(other)])
    with pytest.raises(FileExistsError):
        module.centroid_labels(tmp_path / "a", 10, 2, [str(a), str(a)])


def test_load_label_group_validation_and_reordering(tmp_path):
    bc = load_script("behavior_cloning")
    from homotopy_path_learning.config import load_config

    base = load_config(CONFIG)
    seeds = [10, 11, 12]
    obs = np.arange(3 * OBS_DIM, dtype=np.float32).reshape(3, OBS_DIM)
    a = make_run(tmp_path, "a", seeds, obs=obs)
    b = make_run(tmp_path, "b", seeds, obs=obs)
    got_obs, got_act, got_seeds = bc.load_label_group(base, [str(a), str(b)])
    assert got_seeds == seeds and got_act.shape == (3, 2, ACTION_DIM)
    np.testing.assert_array_equal(got_act[:, 0], np.load(a / "oracle_actions.npy"))
    np.testing.assert_array_equal(got_act[:, 1], np.load(b / "oracle_actions.npy"))
    np.testing.assert_array_equal(got_obs, obs)

    # same targets in a different row order are re-sorted by seed
    perm = [2, 0, 1]
    shuffled = make_run(
        tmp_path, "shuffled", [seeds[i] for i in perm], obs=obs[perm], actions=np.load(b / "oracle_actions.npy")[perm]
    )
    _, act2, seeds2 = bc.load_label_group(base, [str(a), str(shuffled)])
    assert seeds2 == seeds
    np.testing.assert_array_equal(act2[:, 1], np.load(b / "oracle_actions.npy"))

    bad_obs = obs.copy()
    bad_obs[0, 0] += 1
    for run in (
        make_run(tmp_path, "bad-obs", seeds, obs=bad_obs),
        make_run(tmp_path, "bad-seed", [10, 11, 13], obs=obs),
        make_run(tmp_path, "bad-linear", seeds, obs=obs, linear=[3.0, 3.0, 5.0]),
        make_run(tmp_path, "bad-len", [10, 11], obs=obs[:2]),
    ):
        with pytest.raises(ValueError):
            bc.load_label_group(base, [str(a), str(run)])
    with pytest.raises(ValueError):
        bc.load_label_group(base, [])
