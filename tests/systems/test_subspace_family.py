"""Gaussian subspace target families (ADR-0008): sampler, real basis, config, environment, checkpoints."""

from __future__ import annotations

from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
import torch

from homotopy_path_learning.backends.types import TrackingResult
from homotopy_path_learning.config import (
    ConfigError,
    build_env,
    build_sampler,
    build_system_spec,
    config_to_dict,
    load_config,
    parse_config_dict,
    require_symmetry_augmentation_allowed,
    save_config,
    target_distribution_is_symmetric,
)
from homotopy_path_learning.evaluation.oracle import with_latent_dim
from homotopy_path_learning.paths.parameterization import (
    complex_coefficients_to_real,
    linear_interpolation_control_points,
    realify_complex_basis,
)
from homotopy_path_learning.rl.checkpoint import (
    CheckpointError,
    build_agent_from_checkpoint,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint,
)
from homotopy_path_learning.rl.networks import ActorCritic
from homotopy_path_learning.systems import GaussianSubspaceSampler, make_complex_subspace_basis

BASE = "experiments/multivariate_pham/configs/exp-0007-n3-d222-seed0.yaml"  # n3-d222, 12 free coefficients


def family_config(k: int = 4, seed: int = 7, basis: str = "family", latent_dim: int | None = None):
    data = config_to_dict(load_config(BASE))
    data["sampler"].update(type="gaussian_subspace", subspace_dim=k, family_seed=seed)
    data["path"].update(basis=basis, latent_dim=2 * k if latent_dim is None else latent_dim)
    data["environment"]["warmup_backend"] = False
    return parse_config_dict(data)


class FakeBackend:
    def initialize(self, system_spec, bezier_degree, tracker_config) -> None:
        self.n_paths = int(np.prod(system_spec.degrees))
        self.n_vars = system_spec.n_vars
        self.tracked: list[np.ndarray] = []

    def warmup(self):
        return self.track(None)

    def track(self, control_points) -> TrackingResult:
        if control_points is not None:
            self.tracked.append(np.array(control_points, copy=True))
        n = self.n_paths
        return TrackingResult(
            success=True,
            n_paths=n,
            n_success=n,
            n_failed=0,
            accepted_steps=5 * n,
            rejected_steps=n,
            per_path_accepted_steps=np.full(n, 5, dtype=np.int64),
            per_path_rejected_steps=np.ones(n, dtype=np.int64),
            path_success=np.ones(n, dtype=np.bool_),
            endpoints=np.ones((n, self.n_vars), dtype=np.complex128),
            residual_norms=np.full(n, 1e-12),
            failure_codes=("",) * n,
        )

    def close(self) -> None:
        pass


def residual_outside(w: np.ndarray, vectors: np.ndarray) -> float:
    """Largest norm of the component of ``vectors`` (columns) outside span_C(W)."""

    return float(np.abs(vectors - w @ (w.conj().T @ vectors)).max())


# --- basis and sampler ---------------------------------------------------------------------------


def test_complex_subspace_basis_is_orthonormal_and_deterministic() -> None:
    w = make_complex_subspace_basis(np.random.default_rng(3), n_free=12, subspace_dim=5)
    assert w.shape == (12, 5) and w.dtype == np.complex128
    np.testing.assert_allclose(w.conj().T @ w, np.eye(5), atol=1e-12)
    again = make_complex_subspace_basis(np.random.default_rng(3), n_free=12, subspace_dim=5)
    assert np.array_equal(w, again)
    with pytest.raises(ValueError, match="exceed"):
        make_complex_subspace_basis(np.random.default_rng(0), n_free=4, subspace_dim=5)


def test_sampler_targets_lie_in_the_family_and_keep_leading_coefficients() -> None:
    config = family_config(k=4)
    spec = build_system_spec(config)
    sampler = build_sampler(config)
    assert isinstance(sampler, GaussianSubspaceSampler) and sampler.subspace_dim == 4
    free = spec.free_coefficient_mask
    targets = np.stack([sampler.sample(np.random.default_rng(s), spec) for s in range(50)])
    assert targets.shape == (50, spec.n_coeffs)
    assert np.array_equal(targets[:, spec.leading_indices], np.ones((50, spec.leading_indices.size)))
    assert residual_outside(sampler.basis, targets[:, free].T) < 1e-12
    same = sampler.sample(np.random.default_rng(11), spec)
    assert np.array_equal(same, sampler.sample(np.random.default_rng(11), spec))
    # The same family seed gives the same W in every rebuilt sampler (EnvPool workers).
    assert np.array_equal(build_sampler(config).basis, sampler.basis)


def test_full_dimensional_family_has_independent_complex_gaussian_coefficients() -> None:
    config = family_config(k=12)
    spec = build_system_spec(config)
    sampler = build_sampler(config)
    rng = np.random.default_rng(0)
    c = np.stack([sampler.sample(rng, spec)[spec.free_coefficient_mask] for _ in range(40000)])
    covariance = c.T @ c.conj() / c.shape[0]
    pseudo = c.T @ c / c.shape[0]
    np.testing.assert_allclose(covariance, (2.0 / 3.0) * np.eye(12), atol=0.03)
    np.testing.assert_allclose(pseudo, 0.0, atol=0.03)
    assert target_distribution_is_symmetric(config)
    assert not target_distribution_is_symmetric(family_config(k=4))


def test_mean_second_moment_matches_the_uniform_sampler() -> None:
    config = family_config(k=4)
    spec = build_system_spec(config)
    rng = np.random.default_rng(1)
    means = []
    for seed in range(40):  # average over W as well
        sampler = GaussianSubspaceSampler.from_spec(spec, subspace_dim=4, family_seed=seed)
        c = np.stack([sampler.sample(rng, spec)[spec.free_coefficient_mask] for _ in range(500)])
        means.append(np.mean(np.abs(c) ** 2))
    assert abs(np.mean(means) - 2.0 / 3.0) < 0.03


# --- real form of the basis --------------------------------------------------------------------


def test_realified_basis_is_orthonormal_and_matches_the_complex_map() -> None:
    w = make_complex_subspace_basis(np.random.default_rng(5), n_free=6, subspace_dim=3)
    u = realify_complex_basis(w)
    assert u.shape == (12, 6) and u.dtype == np.float64
    np.testing.assert_allclose(u.T @ u, np.eye(6), atol=1e-12)
    z_c = np.array([0.3 - 0.1j, -0.7 + 0.2j, 0.05 + 0.9j])
    z = np.concatenate([z_c.real, z_c.imag])
    np.testing.assert_allclose(u @ z, complex_coefficients_to_real(w @ z_c), atol=1e-14)


# --- config ------------------------------------------------------------------------------------


def test_family_config_round_trip_and_defaults(tmp_path) -> None:
    config = family_config(k=4)
    path = tmp_path / "config.yaml"
    save_config(config, path)
    assert load_config(path) == config
    uniform = load_config(BASE)
    assert uniform.sampler.subspace_dim is None and uniform.sampler.family_seed is None
    assert uniform.path.basis == "random"
    save_config(uniform, path)
    assert load_config(path) == uniform  # nulls written by save_config are read back


@pytest.mark.parametrize(
    ("sampler", "path", "message"),
    [
        ({"subspace_dim": 4}, {}, "only allowed"),
        ({"type": "gaussian_subspace", "subspace_dim": 4}, {}, "requires"),
        ({"type": "gaussian_subspace", "subspace_dim": 13, "family_seed": 0}, {}, r"\[1, 12\]"),
        ({}, {"basis": "family", "latent_dim": 8}, "requires sampler.type"),
        ({"type": "gaussian_subspace", "subspace_dim": 4, "family_seed": 0}, {"basis": "family", "latent_dim": 6}, "2 \\* sampler"),
        ({}, {"basis": "other"}, "path.basis"),
        ({}, {"latent_dim": 25}, "exceeds"),
        ({"type": "other"}, {}, "sampler.type"),
    ],
)
def test_invalid_family_configs_are_rejected(sampler, path, message) -> None:
    data = config_to_dict(load_config(BASE))
    data["sampler"].update(sampler)
    data["path"].update(path)
    with pytest.raises(ConfigError, match=message):
        parse_config_dict(data)


def test_latent_dim_override_is_validated() -> None:
    config = family_config(k=4)
    assert with_latent_dim(config, 8) == config
    with pytest.raises(ConfigError, match="cannot be overridden"):
        with_latent_dim(config, 24)
    with pytest.raises(ConfigError, match="exceeds"):
        with_latent_dim(load_config(BASE), 25)
    full_random = with_latent_dim(family_config(k=4, basis="random", latent_dim=4), 24)
    assert full_random.path.latent_dim == 24 and full_random.path.basis == "random"


def test_symmetry_augmentation_guard() -> None:
    require_symmetry_augmentation_allowed(with_latent_dim(load_config(BASE), 24))
    require_symmetry_augmentation_allowed(family_config(k=12, basis="random", latent_dim=24))
    with pytest.raises(ConfigError, match="invariant"):
        require_symmetry_augmentation_allowed(family_config(k=4))
    with pytest.raises(ConfigError, match="invariant"):
        require_symmetry_augmentation_allowed(family_config(k=4, basis="random", latent_dim=24))
    with pytest.raises(ConfigError, match="full latent space"):
        require_symmetry_augmentation_allowed(load_config(BASE))  # latent_dim 4


# --- environment -------------------------------------------------------------------------------


def test_family_environment_uses_the_family_basis_and_keeps_the_invariants() -> None:
    config = family_config(k=4)
    backend = FakeBackend()
    env = build_env(config, backend=backend)
    try:
        sampler = env.coefficient_sampler
        np.testing.assert_array_equal(env.parameterization.basis, realify_complex_basis(sampler.basis))
        assert env.action_space.shape == ((config.path.bezier_degree - 1) * 8,)
        spec = env.system_spec
        free = spec.free_coefficient_mask
        env.reset(seed=123)
        target = env.current_episode_context().target_coefficients
        start = env.start_coefficients
        linear = linear_interpolation_control_points(start, target, config.path.bezier_degree)
        zero = env.parameterization.build_control_points(start, target, np.zeros(env.action_space.shape))
        np.testing.assert_array_equal(zero, linear)
        action = np.random.default_rng(0).uniform(-1, 1, size=env.action_space.shape)
        points = env.parameterization.build_control_points(start, target, action)
        assert np.array_equal(points[0], start) and np.array_equal(points[-1], target)
        assert residual_outside(sampler.basis, (points - linear)[:, free].T) < 1e-12
        assert np.array_equal(points[:, spec.leading_indices], np.ones((points.shape[0], spec.leading_indices.size)))
    finally:
        env.close()


def test_family_targets_with_a_random_full_basis() -> None:
    config = family_config(k=4, basis="random", latent_dim=24)
    env = build_env(config, backend=FakeBackend())
    try:
        assert env.parameterization.basis.shape == (24, 24)
        env.reset(seed=5)
        target = env.current_episode_context().target_coefficients
        free = env.system_spec.free_coefficient_mask
        assert residual_outside(env.coefficient_sampler.basis, target[free][:, None]) < 1e-12
    finally:
        env.close()


# --- checkpoints -------------------------------------------------------------------------------


def _attention_checkpoint(tmp_path, config, env):
    spec = build_system_spec(config)
    config = replace(config, ppo=replace(config.ppo, actor_type="attention", attention_d_model=16,
                                         attention_heads=2, attention_layers=1, attention_ff_dim=16))
    torch.manual_seed(0)
    agent = ActorCritic(
        observation_shape=tuple(env.observation_space.shape),
        action_space=env.action_space,
        hidden_sizes=(8,),
        actor_type="attention",
        attention_kwargs=config.ppo.attention_kwargs,
        system_spec=spec,
        basis=env.parameterization.basis,
        bezier_degree=config.path.bezier_degree,
    )
    path = tmp_path / "last.pt"
    save_checkpoint(
        path,
        agent=agent,
        optimizer=torch.optim.Adam(agent.parameters()),
        config=replace(config, ppo=replace(config.ppo, hidden_sizes=(8,))),
        system_spec=spec,
        global_step=0,
        update=0,
        observation_shape=tuple(env.observation_space.shape),
        action_space=env.action_space,
        experiment_id="EXP-TEST",
        run_id="run",
        git_commit="abc",
        latent_basis=env.parameterization.basis,
    )
    return agent, load_checkpoint(path)


def test_family_checkpoint_round_trip_and_mismatch_detection(tmp_path) -> None:
    config = family_config(k=4)
    env = build_env(config, backend=FakeBackend())
    try:
        spec = build_system_spec(config)
        agent, checkpoint = _attention_checkpoint(tmp_path, config, env)
        assert checkpoint["target_family"] == {
            "sampler_type": "gaussian_subspace", "subspace_dim": 4, "family_seed": 7, "path_basis": "family",
        }
        kwargs = dict(system_spec=spec, observation_shape=tuple(env.observation_space.shape), action_space=env.action_space)
        validate_checkpoint(checkpoint, config=config, latent_basis=env.parameterization.basis, **kwargs)
        loaded = build_agent_from_checkpoint(checkpoint, action_space=env.action_space, device="cpu")
        obs = torch.randn(3, env.observation_space.shape[0])
        torch.testing.assert_close(loaded.deterministic_action(obs), agent.deterministic_action(obs))
        # Another family seed (same shapes) is rejected by the family record and by the basis hash.
        other = family_config(k=4, seed=8)
        with pytest.raises(CheckpointError, match="target family"):
            validate_checkpoint(checkpoint, config=other, **kwargs)
        with pytest.raises(CheckpointError, match="latent basis"):
            validate_checkpoint(checkpoint, config=config, latent_basis=np.eye(24, 8), **kwargs)
        # A uniform config with the same shapes is rejected.
        uniform = replace(load_config(BASE), path=replace(load_config(BASE).path, latent_dim=8))
        with pytest.raises(CheckpointError, match="target family"):
            validate_checkpoint(checkpoint, config=uniform, **kwargs)
    finally:
        env.close()


def test_legacy_checkpoint_without_family_keys_is_treated_as_uniform(tmp_path) -> None:
    config = load_config(BASE)
    space = gym.spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
    agent = ActorCritic(observation_shape=(24,), action_space=space, hidden_sizes=(8,))
    spec = build_system_spec(config)
    path = tmp_path / "legacy.pt"
    save_checkpoint(
        path, agent=agent, optimizer=torch.optim.Adam(agent.parameters()),
        config=replace(config, ppo=replace(config.ppo, hidden_sizes=(8,))), system_spec=spec,
        global_step=0, update=0, observation_shape=(24,), action_space=space,
        experiment_id="EXP-TEST", run_id="run", git_commit="abc",
    )
    checkpoint = load_checkpoint(path)
    del checkpoint["target_family"], checkpoint["latent_basis_sha256"]
    kwargs = dict(system_spec=spec, observation_shape=(24,), action_space=space)
    validate_checkpoint(checkpoint, config=config, latent_basis=np.eye(24, 4), **kwargs)
    with pytest.raises(CheckpointError, match="target family"):
        validate_checkpoint(checkpoint, config=family_config(k=2, basis="random", latent_dim=4), **kwargs)
