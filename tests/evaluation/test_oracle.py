from __future__ import annotations

import numpy as np
import pytest

from homotopy_path_learning.evaluation.oracle import search_best_action


def quadratic(center: np.ndarray):
    def objective(action: np.ndarray) -> float:
        return float(np.sum((action - center) ** 2))

    return objective


def test_search_approaches_interior_minimum():
    center = np.array([0.4, -0.3, 0.2, -0.1])
    result = search_best_action(
        quadratic(center),
        dim=4,
        action_limit=1.0,
        rng=np.random.default_rng(0),
        budget=400,
        n_random=50,
    )

    assert result.n_evaluations == 400
    assert result.costs.shape == (400,)
    assert result.best_action.shape == (4,)
    assert result.best_action.dtype == np.float64
    assert result.zero_cost == pytest.approx(float(np.sum(center**2)))
    assert result.costs[0] == result.zero_cost
    assert result.best_cost == pytest.approx(float(np.min(result.costs)))
    assert result.best_cost < 1.0e-3


def test_search_keeps_zero_action_when_it_is_optimal():
    result = search_best_action(
        quadratic(np.zeros(3)),
        dim=3,
        action_limit=1.0,
        rng=np.random.default_rng(1),
        budget=50,
        n_random=10,
    )

    assert result.best_cost == 0.0
    np.testing.assert_array_equal(result.best_action, np.zeros(3))


def test_search_never_worse_than_zero_action_and_respects_box():
    seen: list[np.ndarray] = []
    center = np.array([3.0, -3.0])

    def objective(action: np.ndarray) -> float:
        seen.append(action.copy())
        return quadratic(center)(action)

    result = search_best_action(
        objective,
        dim=2,
        action_limit=0.5,
        rng=np.random.default_rng(2),
        budget=200,
        n_random=20,
    )

    assert result.best_cost <= result.zero_cost
    assert all(np.all(np.abs(action) <= 0.5) for action in seen)
    np.testing.assert_allclose(result.best_action, [0.5, -0.5], atol=1.0e-2)


def test_search_is_reproducible_for_same_seed():
    center = np.array([0.2, 0.7, -0.5])
    first = search_best_action(
        quadratic(center), dim=3, action_limit=1.0, rng=np.random.default_rng(5), budget=60, n_random=10
    )
    second = search_best_action(
        quadratic(center), dim=3, action_limit=1.0, rng=np.random.default_rng(5), budget=60, n_random=10
    )

    np.testing.assert_array_equal(first.costs, second.costs)
    np.testing.assert_array_equal(first.best_action, second.best_action)


def test_search_with_budget_one_evaluates_only_zero_action():
    result = search_best_action(
        quadratic(np.ones(2)), dim=2, action_limit=1.0, rng=np.random.default_rng(0), budget=1, n_random=1
    )

    assert result.n_evaluations == 1
    assert result.best_cost == result.zero_cost == 2.0


def test_objective_cannot_mutate_incumbent():
    def objective(action: np.ndarray) -> float:
        cost = float(np.sum((action - 0.5) ** 2))
        action[:] = 99.0
        return cost

    result = search_best_action(
        objective, dim=2, action_limit=1.0, rng=np.random.default_rng(3), budget=40, n_random=5
    )

    assert np.all(np.abs(result.best_action) <= 1.0)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"dim": 0}, ValueError),
        ({"dim": 2.0}, TypeError),
        ({"budget": 0}, ValueError),
        ({"n_random": 11}, ValueError),
        ({"action_limit": 0.0}, ValueError),
        ({"action_limit": float("inf")}, ValueError),
        ({"initial_sigma": -1.0}, ValueError),
        ({"min_sigma": 1.0}, ValueError),
        ({"rng": 0}, TypeError),
    ],
)
def test_search_rejects_invalid_arguments(kwargs, error):
    arguments = {
        "dim": 2,
        "action_limit": 1.0,
        "rng": np.random.default_rng(0),
        "budget": 10,
        "n_random": 5,
    }
    arguments.update(kwargs)
    with pytest.raises(error):
        search_best_action(quadratic(np.zeros(2)), **arguments)


def test_search_rejects_non_finite_objective():
    with pytest.raises(ValueError, match="non-finite"):
        search_best_action(
            lambda action: float("nan"),
            dim=2,
            action_limit=1.0,
            rng=np.random.default_rng(0),
            budget=5,
            n_random=2,
        )


from homotopy_path_learning.backends.types import TrackerConfig, TrackingResult
from homotopy_path_learning.config import load_config
from homotopy_path_learning.evaluation.oracle import (
    ORACLE_FIELDS,
    learned_costs_from_evaluation,
    resolve_latent_dim,
    run_oracle,
    summarize_oracle,
    with_latent_dim,
)


SMOKE_CONFIG = "experiments/multivariate_pham/configs/exp-0004-smoke.yaml"


class DeviationBackend:
    """Fake tracker whose step count depends on the first control-point deviation.

    Each of the 4 paths takes ``100 + round(100 * |D - 0.1|)`` accepted steps,
    where ``D`` is the real part of coefficient 1 of control point 1 minus its
    value on the linear path. The zero action therefore costs 110 and the
    optimum (``D = 0.1``) costs 100.
    """

    def __init__(self) -> None:
        self.track_calls = 0

    def initialize(self, system_spec, bezier_degree: int, tracker_config: TrackerConfig) -> None:
        del system_spec, tracker_config
        self.bezier_degree = bezier_degree

    def warmup(self) -> TrackingResult:
        return self.track(np.zeros((self.bezier_degree + 1, 8), dtype=np.complex128))

    def track(self, control_points: np.ndarray) -> TrackingResult:
        self.track_calls += 1
        degree = control_points.shape[0] - 1
        linear = control_points[0] + (control_points[-1] - control_points[0]) / degree
        deviation = float((control_points[1, 1] - linear[1]).real)
        steps = 100 + int(round(100.0 * abs(deviation - 0.1)))
        accepted = np.full(4, steps, dtype=np.int64)
        success = np.ones(4, dtype=np.bool_)
        return TrackingResult(
            success=True,
            n_paths=4,
            n_success=4,
            n_failed=0,
            accepted_steps=int(np.sum(accepted)),
            rejected_steps=0,
            per_path_accepted_steps=accepted,
            per_path_rejected_steps=np.zeros(4, dtype=np.int64),
            path_success=success,
            endpoints=np.ones((4, 2), dtype=np.complex128),
            residual_norms=np.full(4, 1e-12, dtype=np.float64),
            failure_codes=("", "", "", ""),
        )

    def close(self) -> None:
        return None


def test_run_oracle_improves_over_linear_with_fake_backend():
    config = load_config(SMOKE_CONFIG)
    backend = DeviationBackend()

    result = run_oracle(config, budget=60, n_random=20, search_seed=0, num_instances=3, backend=backend)

    assert len(result.rows) == 3
    assert set(result.rows[0]) == set(ORACLE_FIELDS)
    assert [row["target_seed"] for row in result.rows] == [10000, 10001, 10002]
    assert result.best_actions.shape == (3, 8)
    assert result.cost_traces.shape == (3, 60)
    assert result.basis.shape == (12, 4)
    for row in result.rows:
        assert row["linear_cost"] == 110.0
        assert row["zero_action_cost"] == row["linear_cost"]
        assert row["oracle_cost"] < row["linear_cost"]
        assert row["delta_j_percent"] < 0.0
        assert row["best_action_max_abs"] <= config.path.action_limit
    # one warmup, then per instance: linear trace + search budget + re-evaluation of the best action
    assert backend.track_calls == 1 + 3 * (1 + 60 + 1)


def test_run_oracle_is_reproducible():
    config = load_config(SMOKE_CONFIG)
    first = run_oracle(config, budget=30, n_random=10, search_seed=4, num_instances=2, backend=DeviationBackend())
    second = run_oracle(config, budget=30, n_random=10, search_seed=4, num_instances=2, backend=DeviationBackend())

    np.testing.assert_array_equal(first.best_actions, second.best_actions)
    np.testing.assert_array_equal(first.cost_traces, second.cost_traces)


def test_resolve_latent_dim():
    config = load_config(SMOKE_CONFIG)

    assert resolve_latent_dim(config, None) == 4
    assert resolve_latent_dim(config, "full") == 12
    assert resolve_latent_dim(config, "8") == 8
    assert resolve_latent_dim(config, 2) == 2
    with pytest.raises(ValueError, match="exceeds"):
        resolve_latent_dim(config, 13)
    with pytest.raises(ValueError):
        resolve_latent_dim(config, "wide")
    with pytest.raises(ValueError):
        resolve_latent_dim(config, 0)

    full = with_latent_dim(config, 12)
    assert full.path.latent_dim == 12
    assert config.path.latent_dim == 4


def oracle_row(seed: int, linear: float, oracle: float) -> dict[str, object]:
    return {
        "target_seed": seed,
        "linear_cost": linear,
        "linear_success": 1,
        "oracle_cost": oracle,
        "oracle_success": 1,
    }


def evaluation_rows(entries: list[tuple[int, float, float]]) -> list[dict[str, str]]:
    rows = []
    for seed, linear, learned in entries:
        rows.append({"target_seed": str(seed), "method": "Linear", "mean_cost": str(linear), "problem_success": "1"})
        rows.append(
            {"target_seed": str(seed), "method": "LearnedBezier", "mean_cost": str(learned), "problem_success": "1"}
        )
    return rows


def test_summarize_oracle_with_learned_costs():
    rows = [oracle_row(0, 100.0, 80.0), oracle_row(1, 100.0, 100.0)]
    learned = learned_costs_from_evaluation(evaluation_rows([(1, 100.0, 105.0), (0, 100.0, 85.0)]), rows)

    assert learned == ([85.0, 105.0], [1, 1])

    summary = {item["method"]: item for item in summarize_oracle(rows, {"ppo": learned})}
    assert summary["Linear"]["delta_j_percent"] == 0.0
    assert summary["Linear"]["captured_fraction"] == 0.0
    assert summary["Oracle"]["mean_cost"] == 90.0
    assert summary["Oracle"]["delta_j_percent"] == pytest.approx(-10.0)
    assert summary["Oracle"]["positive_rate"] == 0.5
    assert summary["Oracle"]["captured_fraction"] == 1.0
    assert summary["LearnedBezier[ppo]"]["delta_j_percent"] == pytest.approx(-5.0)
    assert summary["LearnedBezier[ppo]"]["captured_fraction"] == pytest.approx(0.5)
    assert summary["LearnedBezier[ppo]"]["problem_success_rate"] == 1.0


def test_summarize_oracle_without_headroom_leaves_captured_fraction_empty():
    summary = summarize_oracle([oracle_row(0, 50.0, 50.0)])

    assert summary[1]["method"] == "Oracle"
    assert summary[1]["captured_fraction"] == ""


def test_learned_costs_reject_mismatched_instances():
    rows = [oracle_row(0, 100.0, 90.0)]
    with pytest.raises(ValueError, match="target_seed=0"):
        learned_costs_from_evaluation(evaluation_rows([(1, 100.0, 95.0)]), rows)
    with pytest.raises(ValueError, match="Linear cost mismatch"):
        learned_costs_from_evaluation(evaluation_rows([(0, 101.0, 95.0)]), rows)
