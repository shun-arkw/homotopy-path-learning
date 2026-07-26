from __future__ import annotations

import json

import numpy as np
import pytest

from homotopy_path_learning.backends.types import TrackerConfig
from homotopy_path_learning.envs import BezierPhamEnv
from homotopy_path_learning.paths import linear_interpolation_control_points
from homotopy_path_learning.systems import ComplexUniformSampler, build_pham_spec, start_system_coefficients


pytestmark = pytest.mark.integration


SUPPORT_2X2 = (
    ((1, 0), (0, 1), (0, 0)),
    ((1, 0), (0, 1), (0, 0)),
)


def make_spec():
    return build_pham_spec((2, 2), SUPPORT_2X2)


def make_env(*, basis_seed: int = 0) -> BezierPhamEnv:
    return BezierPhamEnv(
        system_spec=make_spec(),
        bezier_degree=3,
        latent_dim=4,
        coefficient_sampler=ComplexUniformSampler(bound=1.0),
        tracker_config=TrackerConfig(),
        reject_weight=1.0,
        failure_penalty=100_000.0,
        reward_scale=1.0,
        action_limit=1.0,
        basis_seed=basis_seed,
    )


def assert_required_step_shapes(info: dict[str, object]) -> None:
    assert np.asarray(info["bezier_residual_norms"]).shape == (4,)
    assert np.asarray(info["linear_per_path_costs"]).shape == (4,)
    assert np.asarray(info["bezier_per_path_costs"]).shape == (4,)
    assert len(info["bezier_failure_codes"]) == 4


def test_julia_backed_env_reset_and_zero_action_step() -> None:
    spec = make_spec()
    env = make_env()
    try:
        observation, reset_info = env.reset(seed=123)
        assert observation.shape == (12,)
        assert observation.dtype == np.float32
        assert env.observation_space.contains(observation)
        assert reset_info["linear_n_paths"] == 4
        assert reset_info["linear_success"] is True

        target = reset_info["target_coefficients"]
        expected_linear = linear_interpolation_control_points(
            start_system_coefficients(spec),
            target,
            3,
        )
        np.testing.assert_array_equal(reset_info["linear_control_points"], expected_linear)

        zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
        next_observation, reward, terminated, truncated, info = env.step(zero_action)

        np.testing.assert_array_equal(next_observation, observation)
        np.testing.assert_array_equal(info["control_points"], expected_linear)
        assert info["linear_cost"] == pytest.approx(info["bezier_cost"])
        assert reward == pytest.approx(0.0, abs=1e-12)
        assert info["reward"] == pytest.approx(0.0, abs=1e-12)
        assert info["bezier_success"] is True
        assert info["bezier_n_success"] == 4
        assert info["bezier_n_failed"] == 0
        assert np.max(info["bezier_residual_norms"]) < 1e-8
        assert terminated is True
        assert truncated is False
        assert_required_step_shapes(info)
    finally:
        env.close()
    assert env.closed


def test_julia_backed_env_reproduces_same_seed_action_and_basis() -> None:
    action = np.linspace(-0.1, 0.1, 8, dtype=np.float32)
    env_a = make_env(basis_seed=0)
    env_b = make_env(basis_seed=0)
    try:
        obs_a, reset_a = env_a.reset(seed=123)
        step_a = env_a.step(action)
        obs_b, reset_b = env_b.reset(seed=123)
        step_b = env_b.step(action)

        np.testing.assert_array_equal(obs_a, obs_b)
        np.testing.assert_array_equal(reset_a["target_coefficients"], reset_b["target_coefficients"])
        assert step_a[1] == pytest.approx(step_b[1], abs=1e-12)
        assert step_a[2] is step_b[2] is True
        assert step_a[3] is step_b[3] is False
        info_a = step_a[4]
        info_b = step_b[4]
        np.testing.assert_array_equal(info_a["control_points"], info_b["control_points"])
        np.testing.assert_allclose(info_a["bezier_residual_norms"], info_b["bezier_residual_norms"])
        np.testing.assert_allclose(info_a["bezier_per_path_costs"], info_b["bezier_per_path_costs"])
        np.testing.assert_allclose(info_a["linear_per_path_costs"], info_b["linear_per_path_costs"])
        assert info_a["linear_cost"] == pytest.approx(info_b["linear_cost"])
        assert info_a["bezier_cost"] == pytest.approx(info_b["bezier_cost"])
        assert info_a["bezier_failure_codes"] == info_b["bezier_failure_codes"]
    finally:
        env_a.close()
        env_b.close()


def test_julia_backed_env_nonzero_action_completes() -> None:
    env = make_env()
    try:
        observation, _ = env.reset(seed=124)
        action = np.linspace(-0.2, 0.2, env.action_space.shape[0], dtype=np.float32)
        next_observation, reward, terminated, truncated, info = env.step(action)

        np.testing.assert_array_equal(next_observation, observation)
        assert isinstance(reward, float)
        assert np.isfinite(reward)
        assert terminated is True
        assert truncated is False
        assert info["control_points"].shape == (4, 8)
        assert info["control_points"].dtype == np.complex128
        assert_required_step_shapes(info)
    finally:
        env.close()


def test_julia_backed_env_close_rejects_future_use() -> None:
    env = make_env()
    env.close()
    assert env.closed

    with pytest.raises(RuntimeError, match="closed"):
        env.reset(seed=123)
    with pytest.raises(RuntimeError, match="closed"):
        env.step(np.zeros(8, dtype=np.float32))


@pytest.mark.slow
def test_julia_backed_env_random_action_100_episode_smoke() -> None:
    env = make_env()
    env.action_space.seed(2025)
    episodes = 100
    summary = {
        "episodes": episodes,
        "linear_all_success": 0,
        "bezier_all_success": 0,
        "linear_path_success": 0,
        "bezier_path_success": 0,
        "nonfinite_rewards": 0,
        "exceptions": 0,
    }
    linear_costs: list[float] = []
    bezier_costs: list[float] = []
    rewards: list[float] = []
    try:
        for episode_index in range(episodes):
            try:
                observation, reset_info = env.reset(seed=10_000 + episode_index)
                action = env.action_space.sample()
                assert env.observation_space.contains(observation)
                assert env.action_space.contains(action)
                next_observation, reward, terminated, truncated, info = env.step(action)
            except Exception:
                summary["exceptions"] += 1
                raise

            assert env.observation_space.contains(next_observation)
            assert np.isfinite(reward)
            assert terminated is True
            assert truncated is False
            assert np.isfinite(info["linear_cost"])
            assert np.isfinite(info["bezier_cost"])
            assert_required_step_shapes(info)

            linear_success = bool(reset_info["linear_success"])
            bezier_success = bool(info["bezier_success"])
            summary["linear_all_success"] += int(linear_success)
            summary["bezier_all_success"] += int(bezier_success)
            summary["linear_path_success"] += int(reset_info["linear_n_success"])
            summary["bezier_path_success"] += int(info["bezier_n_success"])
            summary["nonfinite_rewards"] += int(not np.isfinite(reward))
            linear_costs.append(float(info["linear_cost"]))
            bezier_costs.append(float(info["bezier_cost"]))
            rewards.append(float(reward))
    finally:
        env.close()

    reward_array = np.asarray(rewards, dtype=np.float64)
    summary.update(
        {
            "mean_linear_cost": float(np.mean(linear_costs)),
            "mean_bezier_cost": float(np.mean(bezier_costs)),
            "mean_reward": float(np.mean(reward_array)),
            "min_reward": float(np.min(reward_array)),
            "max_reward": float(np.max(reward_array)),
        }
    )
    assert summary["exceptions"] == 0
    assert summary["nonfinite_rewards"] == 0
    print("PHASE5_SMOKE_SUMMARY", json.dumps(summary, sort_keys=True))
