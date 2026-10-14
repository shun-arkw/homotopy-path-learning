from __future__ import annotations

import copy

import gymnasium as gym
import numpy as np
import pytest
import torch

from homotopy_path_learning.rl.cost_finetune import antithetic_action_gradients, finetune_step, sigma_at
from homotopy_path_learning.rl.behavior_cloning import predicted_actions
from homotopy_path_learning.rl.networks import ActorCritic

OBS, ACT = 6, 4
LOW, HIGH = -np.ones(ACT), np.ones(ACT)


def agent_for_test() -> ActorCritic:
    torch.manual_seed(0)
    return ActorCritic(
        observation_shape=(OBS,),
        action_space=gym.spaces.Box(-1.0, 1.0, shape=(ACT,), dtype=np.float32),
        hidden_sizes=(16, 16),
    )


def test_gradient_estimate_matches_quadratic_gradient():
    centers = np.array([[0.2, -0.1, 0.3, 0.0], [-0.3, 0.2, 0.1, 0.4]])
    means = np.array([[0.0, 0.0, 0.0, 0.0], [0.1, 0.1, -0.1, 0.0]])
    grads, info = antithetic_action_gradients(
        lambda b, a: float(np.sum((a - centers[b]) ** 2)),
        means, sigma=0.05, n_pairs=2000, rng=np.random.default_rng(0), low=LOW, high=HIGH,
    )
    true = 2 * (means - centers)
    for b in range(2):
        cos = grads[b] @ true[b] / (np.linalg.norm(grads[b]) * np.linalg.norm(true[b]))
        assert cos > 0.9
    assert grads.shape == (2, ACT) and grads.dtype == np.float64
    assert info["n_evaluations"] == 2 * 2 * 2000
    assert info["mean_grad_norm"] > 0


def test_box_clipping_reproducible_and_no_mutation():
    means = np.full((2, ACT), 0.95)
    original = means.copy()
    seen = []

    def cost(b, a):
        seen.append(a.copy())
        a[:] = 99.0  # must not affect later evaluations
        return float(np.sum(a * 0))

    antithetic_action_gradients(cost, means, sigma=0.5, n_pairs=3, rng=np.random.default_rng(1), low=LOW, high=HIGH)
    assert all(np.all(a <= 1.0) and np.all(a >= -1.0) for a in seen)
    np.testing.assert_array_equal(means, original)
    f = lambda b, a: float(np.sum(a**2))
    g1, _ = antithetic_action_gradients(f, means, sigma=0.1, n_pairs=3, rng=np.random.default_rng(2), low=LOW, high=HIGH)
    g2, _ = antithetic_action_gradients(f, means, sigma=0.1, n_pairs=3, rng=np.random.default_rng(2), low=LOW, high=HIGH)
    np.testing.assert_array_equal(g1, g2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sigma": 0.0}, {"sigma": float("nan")}, {"n_pairs": 0}, {"means": np.zeros(4)},
        {"means": np.full((1, ACT), np.nan)}, {"low": np.zeros(3)}, {"high": LOW}, {"rng": 0},
    ],
)
def test_invalid_inputs_rejected(kwargs):
    arguments = dict(means=np.zeros((1, ACT)), sigma=0.1, n_pairs=1, rng=np.random.default_rng(0), low=LOW, high=HIGH)
    arguments.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        antithetic_action_gradients(lambda b, a: 0.0, **arguments)
    with pytest.raises(ValueError):
        antithetic_action_gradients(lambda b, a: float("inf"), np.zeros((1, ACT)), sigma=0.1, n_pairs=1,
                                    rng=np.random.default_rng(0), low=LOW, high=HIGH)


def test_finetune_step_moves_against_gradient_and_keeps_critic():
    agent = agent_for_test()
    optimizer = torch.optim.Adam(agent.actor_mean.parameters(), lr=1e-2)
    rng = np.random.default_rng(0)
    obs = rng.uniform(-1, 1, (8, OBS)).astype(np.float32)
    grad = rng.normal(size=(8, ACT)).astype(np.float32)
    critic = copy.deepcopy(agent.critic.state_dict())
    x = torch.as_tensor(obs)

    def dot():
        with torch.no_grad():
            return float((predicted_actions(agent, x) * torch.as_tensor(grad)).sum(1).mean())

    before = dot()
    for _ in range(5):
        info = finetune_step(agent, obs, grad, optimizer)
    assert dot() < before
    assert info["bc_loss"] == 0.0
    for key, value in agent.critic.state_dict().items():
        assert torch.equal(value, critic[key])
    with pytest.raises(ValueError):
        finetune_step(agent, obs, grad[:, :3], optimizer)


def test_finetune_step_bc_term_with_augmentation():
    agent = agent_for_test()
    optimizer = torch.optim.Adam(agent.actor_mean.parameters(), lr=1e-2)
    rng = np.random.default_rng(1)
    obs = rng.uniform(-1, 1, (4, OBS)).astype(np.float32)
    grad = np.zeros((4, ACT), dtype=np.float32)
    bo = rng.uniform(-1, 1, (16, OBS)).astype(np.float32)
    ba = np.tanh(rng.normal(size=(16, ACT))).astype(np.float32)
    aug = [(np.eye(OBS), np.eye(ACT)), (-np.eye(OBS), 3.0 * np.eye(ACT))]
    first = finetune_step(agent, obs, grad, optimizer, bc_observations=bo, bc_actions=ba, bc_weight=1.0,
                          augmentations=aug, generator=torch.Generator().manual_seed(0))
    assert first["bc_loss"] > 0
    assert first["total_loss"] == pytest.approx(first["surrogate_loss"] + first["bc_loss"])
    with pytest.raises(ValueError):
        finetune_step(agent, obs, grad, optimizer, bc_weight=1.0)
    with pytest.raises(ValueError):
        finetune_step(agent, obs, grad, optimizer, bc_observations=bo, bc_actions=ba, bc_weight=1.0,
                      augmentations=[(np.eye(3), np.eye(ACT))])


def _reference_antithetic(cost_fn, means, *, sigma, n_pairs, rng, low, high):
    """Verbatim copy of the pre-refactor sequential estimator."""
    mu = np.array(means, dtype=np.float64, copy=True)
    lo = np.array(low, dtype=np.float64)
    hi = np.array(high, dtype=np.float64)
    gradients = np.zeros_like(mu)
    total = 0.0
    count = 0
    for b in range(mu.shape[0]):
        for _ in range(n_pairs):
            eps = rng.standard_normal(mu.shape[1])
            plus = float(cost_fn(b, np.clip(mu[b] + sigma * eps, lo, hi)))
            minus = float(cost_fn(b, np.clip(mu[b] - sigma * eps, lo, hi)))
            gradients[b] += (plus - minus) * eps
            total += plus + minus
            count += 2
        gradients[b] /= 2.0 * sigma * n_pairs
    return gradients, total / count


def test_refactored_estimator_identical_to_reference_and_split_functions():
    from homotopy_path_learning.rl.cost_finetune import antithetic_candidates, antithetic_gradients_from_costs

    means = np.random.default_rng(5).uniform(-0.9, 0.9, size=(3, ACT))
    centers = np.random.default_rng(6).uniform(-1, 1, size=(3, ACT))

    def cost(b, a):
        return float(np.sum((a - centers[b]) ** 2) + 0.1 * np.sin(a[0]))

    kwargs = dict(sigma=0.3, n_pairs=4, low=LOW, high=HIGH)
    ref_g, ref_mean = _reference_antithetic(cost, means, rng=np.random.default_rng(11), **kwargs)
    new_g, info = antithetic_action_gradients(cost, means, rng=np.random.default_rng(11), **kwargs)
    np.testing.assert_array_equal(new_g, ref_g)
    assert info["mean_cost_ratio"] == ref_mean

    rng = np.random.default_rng(11)
    eps, plus, minus = antithetic_candidates(means, rng=rng, **kwargs)
    assert eps.shape == plus.shape == minus.shape == (3, 4, ACT)
    plus_costs = np.array([[cost(b, plus[b, p]) for p in range(4)] for b in range(3)])
    minus_costs = np.array([[cost(b, minus[b, p]) for p in range(4)] for b in range(3)])
    g2, info2 = antithetic_gradients_from_costs(eps, plus_costs, minus_costs, sigma=0.3)
    np.testing.assert_array_equal(g2, ref_g)
    assert info2 == info
    # The generator is left in the same state as after the sequential estimator.
    seq_rng = np.random.default_rng(11)
    antithetic_action_gradients(cost, means, rng=seq_rng, **kwargs)
    assert rng.standard_normal() == seq_rng.standard_normal()


def test_sigma_at_linear_schedule_and_validation():
    kw = dict(iterations=5, sigma_start=0.1, sigma_end=0.02)
    values = [sigma_at(k, **kw) for k in range(1, 6)]
    assert values[0] == 0.1 and values[-1] == pytest.approx(0.02)
    np.testing.assert_allclose(np.diff(values), -0.02, atol=1e-12)
    assert sigma_at(3, **kw) == pytest.approx(0.06)
    # constant when sigma_end is None, one iteration uses sigma_start
    assert [sigma_at(k, iterations=4, sigma_start=0.05, sigma_end=None) for k in range(1, 5)] == [0.05] * 4
    assert sigma_at(1, iterations=1, sigma_start=0.1, sigma_end=0.02) == 0.1
    assert sigma_at(2, iterations=2, sigma_start=0.1, sigma_end=0.1) == 0.1
    for bad in (
        dict(iteration=0, iterations=5, sigma_start=0.1, sigma_end=0.05),
        dict(iteration=6, iterations=5, sigma_start=0.1, sigma_end=0.05),
        dict(iteration=1, iterations=0, sigma_start=0.1, sigma_end=None),
        dict(iteration=1, iterations=5, sigma_start=0.0, sigma_end=None),
        dict(iteration=1, iterations=5, sigma_start=0.1, sigma_end=0.0),
        dict(iteration=1, iterations=5, sigma_start=0.1, sigma_end=0.2),
        dict(iteration=1, iterations=5, sigma_start=0.1, sigma_end=-0.1),
    ):
        with pytest.raises(ValueError):
            sigma_at(**bad)
