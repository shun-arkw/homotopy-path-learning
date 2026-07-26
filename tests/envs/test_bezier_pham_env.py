from __future__ import annotations

import numpy as np
import pytest

from homotopy_path_learning.envs import BezierPhamEnv
from homotopy_path_learning.paths import linear_interpolation_control_points
from homotopy_path_learning.systems import start_system_coefficients

from .conftest import FakeBackend, FixedSampler, make_tracking_result


def make_env(smoke_spec, target_coefficients=None, backend=None, **kwargs) -> BezierPhamEnv:
    sampler = FixedSampler(target_coefficients) if target_coefficients is not None else None
    bezier_degree = kwargs.pop("bezier_degree", 3)
    latent_dim = kwargs.pop("latent_dim", 4)
    return BezierPhamEnv(
        system_spec=smoke_spec,
        bezier_degree=bezier_degree,
        latent_dim=latent_dim,
        coefficient_sampler=sampler,
        backend=backend if backend is not None else FakeBackend(),
        **kwargs,
    )


def test_environment_spaces_and_injected_backend(smoke_spec) -> None:
    backend = FakeBackend()
    env = make_env(smoke_spec, backend=backend)

    assert env.action_space.shape == (env.parameterization.action_dim,)
    assert env.action_space.shape == (8,)
    assert env.action_space.dtype == np.float32
    assert env.observation_space.shape == (2 * env.parameterization.n_free_coeffs,)
    assert env.observation_space.shape == (12,)
    assert env.observation_space.dtype == np.float32
    assert env.backend is backend
    assert not env.owns_backend
    assert backend.initialize_calls == 1


def test_warmup_backend_runs_only_when_requested(smoke_spec) -> None:
    cold_backend = FakeBackend()
    make_env(smoke_spec, backend=cold_backend, warmup_backend=False)
    assert cold_backend.warmup_calls == 0

    warm_backend = FakeBackend()
    make_env(smoke_spec, backend=warm_backend, warmup_backend=True)
    assert warm_backend.warmup_calls == 1


def test_parameterization_basis_is_fixed_by_basis_seed(smoke_spec) -> None:
    env_a = make_env(smoke_spec, basis_seed=7)
    env_b = make_env(smoke_spec, basis_seed=7)
    env_c = make_env(smoke_spec, basis_seed=8)

    np.testing.assert_array_equal(env_a.parameterization.basis, env_b.parameterization.basis)
    assert not np.array_equal(env_a.parameterization.basis, env_c.parameterization.basis)


def test_constructor_rejects_invalid_values(smoke_spec) -> None:
    with pytest.raises(ValueError, match="bezier_degree"):
        make_env(smoke_spec, bezier_degree=1)
    with pytest.raises(ValueError, match="latent_dim"):
        make_env(smoke_spec, latent_dim=0)
    with pytest.raises(ValueError, match="reject_weight"):
        make_env(smoke_spec, reject_weight=-1.0)
    with pytest.raises(ValueError, match="failure_penalty"):
        make_env(smoke_spec, failure_penalty=0.0)
    with pytest.raises(ValueError, match="reward_scale"):
        make_env(smoke_spec, reward_scale=0.0)
    with pytest.raises(ValueError, match="action_limit"):
        make_env(smoke_spec, action_limit=0.0)
    with pytest.raises(TypeError, match="basis_seed"):
        make_env(smoke_spec, basis_seed=True)
    with pytest.raises(TypeError, match="warmup_backend"):
        make_env(smoke_spec, warmup_backend=1)


def test_reset_returns_free_target_coefficients_in_re_then_im_order(
    smoke_spec,
    target_coefficients,
) -> None:
    env = make_env(smoke_spec, target_coefficients=target_coefficients)

    observation, info = env.reset(seed=123)

    expected = np.array(
        [0.25, -0.5, 0.1, -0.3, 0.6, -0.7, 0.5, 0.75, -0.2, 0.4, -0.1, 0.2],
        dtype=np.float32,
    )
    assert observation.shape == env.observation_space.shape
    assert observation.dtype == np.float32
    np.testing.assert_array_equal(observation, expected)
    assert 1.0 not in observation[:6]
    np.testing.assert_array_equal(info["target_coefficients"], target_coefficients)


def test_reset_seed_reproduces_default_sampler_target_and_observation(smoke_spec) -> None:
    env = make_env(smoke_spec)

    first_observation, first_info = env.reset(seed=123)
    second_observation, second_info = env.reset(seed=123)
    third_observation, third_info = env.reset(seed=124)

    np.testing.assert_array_equal(first_observation, second_observation)
    np.testing.assert_array_equal(
        first_info["target_coefficients"],
        second_info["target_coefficients"],
    )
    assert not np.array_equal(first_info["target_coefficients"], third_info["target_coefficients"])
    assert not np.array_equal(first_observation, third_observation)


def test_reset_tracks_linear_baseline_once_and_caches_it(smoke_spec, target_coefficients) -> None:
    backend = FakeBackend()
    env = make_env(smoke_spec, target_coefficients=target_coefficients, backend=backend)

    _, info = env.reset(seed=1)

    start = start_system_coefficients(smoke_spec)
    expected = linear_interpolation_control_points(start, target_coefficients, 3)
    assert backend.track_calls == 1
    np.testing.assert_array_equal(backend.tracked_control_points[0], expected)
    np.testing.assert_array_equal(info["linear_control_points"], expected)
    np.testing.assert_array_equal(env.linear_control_points, expected)
    assert env.linear_cost == info["linear_cost"]


def test_reset_info_arrays_are_independent(smoke_spec, target_coefficients) -> None:
    env = make_env(smoke_spec, target_coefficients=target_coefficients)
    _, info = env.reset(seed=1)

    info["target_coefficients"][1] = 99.0 + 0.0j
    info["linear_control_points"][0, 1] = 99.0 + 0.0j
    info["linear_per_path_costs"][0] = 99.0

    np.testing.assert_array_equal(env.target_coefficients, target_coefficients)
    assert env.linear_control_points[0, 1] != 99.0 + 0.0j
    assert env.linear_cost == 6.0


def test_episode_context_arrays_are_independent(smoke_spec, target_coefficients) -> None:
    env = make_env(smoke_spec, target_coefficients=target_coefficients)
    env.reset(seed=1)

    context = env.current_episode_context()
    context.target_coefficients[1] = 99.0 + 0.0j
    context.linear_control_points[0, 1] = 99.0 + 0.0j
    context.linear_cost.per_path[0] = 99.0

    np.testing.assert_array_equal(env.target_coefficients, target_coefficients)
    assert env.linear_control_points[0, 1] != 99.0 + 0.0j
    assert env.linear_cost == 6.0


def test_evaluate_action_against_context_does_not_recompute_linear_baseline(
    smoke_spec,
    target_coefficients,
) -> None:
    backend = FakeBackend(
        [
            make_tracking_result(accepted=(10, 10, 10, 10), rejected=(0, 0, 0, 0)),
            make_tracking_result(accepted=(5, 5, 5, 5), rejected=(0, 0, 0, 0)),
            make_tracking_result(accepted=(6, 6, 6, 6), rejected=(0, 0, 0, 0)),
        ]
    )
    env = make_env(smoke_spec, target_coefficients=target_coefficients, backend=backend)
    env.reset(seed=1)
    context = env.current_episode_context()

    first = env.evaluate_action_against_context(
        context,
        np.zeros(env.parameterization.action_dim, dtype=np.float32),
    )
    second = env.evaluate_action_against_context(
        context,
        np.full(env.parameterization.action_dim, 0.25, dtype=np.float32),
    )

    assert backend.track_calls == 3
    assert first.info["linear_cost"] == 10.0
    assert second.info["linear_cost"] == 10.0
    assert first.info["bezier_cost"] == 5.0
    assert second.info["bezier_cost"] == 6.0


def test_step_with_valid_action_terminates_and_uses_cached_linear_cost(
    smoke_spec,
    target_coefficients,
) -> None:
    backend = FakeBackend(
        results=[
            make_tracking_result(accepted=(10, 10, 10, 10), rejected=(0, 0, 0, 0)),
            make_tracking_result(accepted=(5, 5, 5, 5), rejected=(0, 0, 0, 0)),
        ]
    )
    env = make_env(smoke_spec, target_coefficients=target_coefficients, backend=backend)
    observation, _ = env.reset(seed=1)
    action = np.linspace(-0.25, 0.25, env.parameterization.action_dim, dtype=np.float32)
    action_before = action.copy()

    next_observation, reward, terminated, truncated, info = env.step(action)

    np.testing.assert_array_equal(action, action_before)
    np.testing.assert_array_equal(next_observation, observation)
    assert next_observation.dtype == np.float32
    assert isinstance(reward, float)
    assert np.isfinite(reward)
    assert terminated is True
    assert truncated is False
    assert backend.track_calls == 2
    assert reward > 0.0
    assert info["linear_cost"] == 10.0
    assert info["bezier_cost"] == 5.0
    assert info["cost_improvement"] == 5.0
    assert info["reward"] == reward


def test_step_control_points_reflect_action_and_keep_invariants(
    smoke_spec,
    target_coefficients,
) -> None:
    backend = FakeBackend()
    env = make_env(smoke_spec, target_coefficients=target_coefficients, backend=backend)
    env.reset(seed=1)
    action = np.linspace(-0.5, 0.5, env.parameterization.action_dim, dtype=np.float64)

    _, _, _, _, info = env.step(action)

    expected = env.parameterization.build_control_points(
        start_system_coefficients(smoke_spec),
        target_coefficients,
        action,
    )
    np.testing.assert_array_equal(info["control_points"], expected)
    np.testing.assert_array_equal(backend.tracked_control_points[-1], expected)
    np.testing.assert_array_equal(
        info["control_points"][:, smoke_spec.leading_indices],
        np.ones((4, 2), dtype=np.complex128),
    )
    np.testing.assert_array_equal(info["control_points"][0], start_system_coefficients(smoke_spec))
    np.testing.assert_array_equal(info["control_points"][-1], target_coefficients)


def test_reward_signs_follow_cost_improvement(smoke_spec, target_coefficients) -> None:
    improved = FakeBackend(
        [
            make_tracking_result(accepted=(10, 10), rejected=(0, 0)),
            make_tracking_result(accepted=(5, 5), rejected=(0, 0)),
        ]
    )
    worsened = FakeBackend(
        [
            make_tracking_result(accepted=(10, 10), rejected=(0, 0)),
            make_tracking_result(accepted=(15, 15), rejected=(0, 0)),
        ]
    )
    equal = FakeBackend(
        [
            make_tracking_result(accepted=(10, 10), rejected=(0, 0)),
            make_tracking_result(accepted=(10, 10), rejected=(0, 0)),
        ]
    )

    env = make_env(smoke_spec, target_coefficients=target_coefficients, backend=improved)
    env.reset(seed=1)
    assert env.step(np.zeros(env.parameterization.action_dim))[1] > 0.0

    env = make_env(smoke_spec, target_coefficients=target_coefficients, backend=worsened)
    env.reset(seed=1)
    assert env.step(np.zeros(env.parameterization.action_dim))[1] < 0.0

    env = make_env(smoke_spec, target_coefficients=target_coefficients, backend=equal)
    env.reset(seed=1)
    assert env.step(np.zeros(env.parameterization.action_dim))[1] == 0.0


def test_partial_tracking_failure_returns_finite_reward(smoke_spec, target_coefficients) -> None:
    backend = FakeBackend(
        [
            make_tracking_result(accepted=(10, 10, 10, 10), rejected=(0, 0, 0, 0)),
            make_tracking_result(
                accepted=(5, 5, 5, 5),
                rejected=(0, 0, 0, 0),
                success=(True, False, True, False),
            ),
        ]
    )
    env = make_env(smoke_spec, target_coefficients=target_coefficients, backend=backend)
    env.reset(seed=1)

    _, reward, _, _, info = env.step(np.zeros(env.parameterization.action_dim))

    assert np.isfinite(reward)
    assert info["bezier_success"] is False
    assert info["bezier_n_failed"] == 2
    np.testing.assert_array_equal(
        info["bezier_per_path_costs"],
        np.array([5.0, 100_000.0, 5.0, 100_000.0]),
    )


def test_backend_program_exception_is_not_converted_to_tracking_failure(
    smoke_spec,
    target_coefficients,
) -> None:
    backend = FakeBackend(raise_on_track_call=2)
    env = make_env(smoke_spec, target_coefficients=target_coefficients, backend=backend)
    env.reset(seed=1)

    with pytest.raises(RuntimeError, match="backend failure"):
        env.step(np.zeros(env.parameterization.action_dim))


def test_step_info_arrays_are_independent(smoke_spec, target_coefficients) -> None:
    backend = FakeBackend()
    env = make_env(smoke_spec, target_coefficients=target_coefficients, backend=backend)
    env.reset(seed=1)
    _, _, _, _, info = env.step(np.zeros(env.parameterization.action_dim))
    recorded = backend.tracked_control_points[-1].copy()

    info["target_coefficients"][1] = 99.0 + 0.0j
    info["control_points"][0, 1] = 99.0 + 0.0j
    info["bezier_per_path_costs"][0] = 99.0
    info["bezier_residual_norms"][0] = 99.0

    np.testing.assert_array_equal(env.target_coefficients, target_coefficients)
    np.testing.assert_array_equal(backend.tracked_control_points[-1], recorded)


def test_invalid_step_states_and_actions(smoke_spec, target_coefficients) -> None:
    env = make_env(smoke_spec, target_coefficients=target_coefficients)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(env.parameterization.action_dim))

    env.reset(seed=1)
    with pytest.raises(ValueError, match="shape"):
        env.step(np.zeros((2, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="NaN|Inf"):
        bad = np.zeros(env.parameterization.action_dim, dtype=np.float32)
        bad[0] = np.nan
        env.step(bad)
    with pytest.raises(ValueError, match="NaN|Inf"):
        bad = np.zeros(env.parameterization.action_dim, dtype=np.float32)
        bad[0] = np.inf
        env.step(bad)
    with pytest.raises(ValueError, match="bounds"):
        bad = np.zeros(env.parameterization.action_dim, dtype=np.float32)
        bad[0] = env.action_limit + 0.25
        env.step(bad)

    action = np.zeros(env.parameterization.action_dim, dtype=np.float32)
    env.step(action)
    with pytest.raises(RuntimeError, match="one-step"):
        env.step(action)


def test_close_state_and_backend_ownership(smoke_spec, target_coefficients, monkeypatch) -> None:
    injected = FakeBackend()
    env = make_env(smoke_spec, target_coefficients=target_coefficients, backend=injected)
    env.close()
    env.close()
    assert injected.close_calls == 0
    with pytest.raises(RuntimeError, match="closed"):
        env.reset(seed=1)
    with pytest.raises(RuntimeError, match="closed"):
        env.step(np.zeros(env.parameterization.action_dim))

    created: list[FakeBackend] = []

    class OwnedFakeBackend(FakeBackend):
        def __init__(self) -> None:
            super().__init__()
            created.append(self)

    monkeypatch.setattr("homotopy_path_learning.envs.bezier_pham.JuliaTrackerBackend", OwnedFakeBackend)
    owned_env = BezierPhamEnv(system_spec=smoke_spec, coefficient_sampler=FixedSampler(target_coefficients))
    owned_env.close()
    owned_env.close()
    assert created[0].close_calls == 1


def test_consecutive_resets_start_new_episodes_without_reinitializing_backend(
    smoke_spec,
    target_coefficients,
) -> None:
    backend = FakeBackend()
    env = make_env(smoke_spec, target_coefficients=target_coefficients, backend=backend)

    env.reset(seed=1)
    env.reset(seed=2)

    assert backend.initialize_calls == 1
    assert backend.track_calls == 2
