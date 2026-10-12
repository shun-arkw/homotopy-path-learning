from __future__ import annotations

import numpy as np
import pytest

from homotopy_path_learning.evaluation.compare import (
    average_group,
    captured_fraction,
    delta_j_percent,
    metrics,
    paired_bootstrap_difference,
    positive_rate,
)


LINEAR = np.array([10.0, 20.0, 30.0, 40.0])
ORACLE = np.array([6.0, 12.0, 18.0, 24.0])


def test_point_metrics() -> None:
    cost = np.array([9.0, 20.0, 27.0, 40.0])
    assert delta_j_percent(cost, LINEAR) == pytest.approx(-4.0)
    assert positive_rate(cost, LINEAR) == pytest.approx(0.5)
    assert captured_fraction(cost, LINEAR, ORACLE) == pytest.approx(4.0 / 40.0)
    assert np.isnan(captured_fraction(cost, LINEAR, LINEAR))
    assert np.isnan(metrics(cost, LINEAR)["captured_fraction"])
    with pytest.raises(ValueError, match="same length"):
        delta_j_percent(cost[:3], LINEAR)


def test_average_group() -> None:
    np.testing.assert_allclose(average_group([[1.0, 3.0], [3.0, 5.0]]), [2.0, 4.0])
    with pytest.raises(ValueError):
        average_group([])


def test_paired_bootstrap_identical_methods_has_zero_difference() -> None:
    result = paired_bootstrap_difference(
        LINEAR * 0.9, LINEAR * 0.9, LINEAR, ORACLE, n_boot=200, rng=np.random.default_rng(0)
    )
    for name, item in result.items():
        assert item["diff"] == 0.0
        assert item["ci_low"] == 0.0 and item["ci_high"] == 0.0


def test_paired_bootstrap_detects_consistent_improvement_and_is_reproducible() -> None:
    rng = np.random.default_rng(5)
    linear = rng.uniform(50.0, 150.0, size=300)
    oracle = linear * 0.5
    a = linear * 0.95
    b = linear * 0.90  # B is uniformly better by 5 percentage points
    r1 = paired_bootstrap_difference(a, b, linear, oracle, n_boot=500, rng=np.random.default_rng(1))
    r2 = paired_bootstrap_difference(a, b, linear, oracle, n_boot=500, rng=np.random.default_rng(1))
    assert r1 == r2
    d = r1["delta_j_percent"]
    assert d["diff"] == pytest.approx(-5.0)
    assert d["ci_low"] < -5.0 + 1e-9 and d["ci_high"] > -5.0 - 1e-9
    assert d["ci_high"] < 0.0
    assert r1["captured_fraction"]["diff"] == pytest.approx(0.1)
    assert r1["positive_rate"]["diff"] == 0.0


def test_paired_bootstrap_matches_manual_resampling() -> None:
    a, b = LINEAR * 0.9, LINEAR * np.array([0.8, 0.9, 1.0, 1.1])
    rng = np.random.default_rng(3)
    result = paired_bootstrap_difference(a, b, LINEAR, None, n_boot=50, rng=rng)
    idx = np.random.default_rng(3).integers(0, 4, size=(50, 4))
    manual = [delta_j_percent(b[i], LINEAR[i]) - delta_j_percent(a[i], LINEAR[i]) for i in idx]
    low, high = np.percentile(manual, [2.5, 97.5])
    assert result["delta_j_percent"]["ci_low"] == pytest.approx(low)
    assert result["delta_j_percent"]["ci_high"] == pytest.approx(high)
    assert np.isnan(result["captured_fraction"]["ci_low"])


def test_bootstrap_input_validation() -> None:
    with pytest.raises(TypeError):
        paired_bootstrap_difference(LINEAR, LINEAR, LINEAR, n_boot=5, rng=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        paired_bootstrap_difference(LINEAR, LINEAR, LINEAR, n_boot=0, rng=np.random.default_rng(0))
