from __future__ import annotations

import numpy as np
import pytest

from homotopy_path_learning.envs.costs import reward_from_costs, tracking_cost

from .conftest import make_tracking_result


def test_tracking_cost_all_success_uses_accepted_and_rejected_steps() -> None:
    result = make_tracking_result(accepted=(1, 2, 3), rejected=(0, 1, 2), success=(True,) * 3)

    cost = tracking_cost(result, reject_weight=2.0, failure_penalty=100.0)

    np.testing.assert_array_equal(cost.per_path, np.array([1.0, 4.0, 7.0]))
    assert cost.mean == 4.0


def test_tracking_cost_reject_weight_changes_successful_path_costs() -> None:
    result = make_tracking_result(accepted=(10, 10), rejected=(1, 3), success=(True, True))

    cost = tracking_cost(result, reject_weight=0.5, failure_penalty=100.0)

    np.testing.assert_array_equal(cost.per_path, np.array([10.5, 11.5]))
    assert cost.mean == 11.0


def test_tracking_cost_applies_failure_penalty_only_to_failed_paths() -> None:
    result = make_tracking_result(
        accepted=(1, 2, 3),
        rejected=(1, 1, 1),
        success=(True, False, True),
    )

    cost = tracking_cost(result, reject_weight=2.0, failure_penalty=50.0)

    np.testing.assert_array_equal(cost.per_path, np.array([3.0, 50.0, 5.0]))
    assert cost.mean == pytest.approx(58.0 / 3.0)


def test_tracking_cost_all_failed_mean_is_failure_penalty() -> None:
    result = make_tracking_result(accepted=(1, 2), rejected=(1, 2), success=(False, False))

    cost = tracking_cost(result, reject_weight=1.0, failure_penalty=25.0)

    np.testing.assert_array_equal(cost.per_path, np.array([25.0, 25.0]))
    assert cost.mean == 25.0


def test_tracking_cost_rejects_invalid_reject_weight() -> None:
    result = make_tracking_result()

    with pytest.raises(ValueError, match="reject_weight"):
        tracking_cost(result, reject_weight=-1.0, failure_penalty=100.0)

    with pytest.raises(ValueError, match="finite"):
        tracking_cost(result, reject_weight=np.inf, failure_penalty=100.0)


def test_tracking_cost_rejects_invalid_failure_penalty() -> None:
    result = make_tracking_result()

    with pytest.raises(ValueError, match="failure_penalty"):
        tracking_cost(result, reject_weight=1.0, failure_penalty=0.0)

    with pytest.raises(ValueError, match="finite"):
        tracking_cost(result, reject_weight=1.0, failure_penalty=np.nan)


def test_tracking_cost_does_not_mutate_input_arrays() -> None:
    result = make_tracking_result()
    accepted_before = result.per_path_accepted_steps.copy()
    rejected_before = result.per_path_rejected_steps.copy()
    success_before = result.path_success.copy()

    cost = tracking_cost(result, reject_weight=1.0, failure_penalty=100.0)
    cost.per_path[0] = 999.0

    np.testing.assert_array_equal(result.per_path_accepted_steps, accepted_before)
    np.testing.assert_array_equal(result.per_path_rejected_steps, rejected_before)
    np.testing.assert_array_equal(result.path_success, success_before)


def test_reward_from_costs_uses_linear_minus_bezier_cost() -> None:
    assert reward_from_costs(linear_cost=10.0, bezier_cost=8.0, reward_scale=2.0) == 4.0
    assert reward_from_costs(linear_cost=10.0, bezier_cost=12.0, reward_scale=1.0) == -2.0
    assert reward_from_costs(linear_cost=10.0, bezier_cost=10.0, reward_scale=1.0) == 0.0

    with pytest.raises(ValueError, match="reward_scale"):
        reward_from_costs(linear_cost=1.0, bezier_cost=1.0, reward_scale=0.0)
