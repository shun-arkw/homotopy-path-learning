from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

from homotopy_path_learning.rl.distributions import TanhDiagNormal


def test_tanh_distribution_samples_and_deterministic_actions_are_in_bounds() -> None:
    space = gym.spaces.Box(low=-2.0, high=3.0, shape=(4,), dtype=np.float32)
    distribution = TanhDiagNormal(space, device="cpu")
    mean = torch.zeros((16, 4))
    log_std = torch.zeros((16, 4))

    sample = distribution.sample(mean, log_std)
    deterministic = distribution.sample(mean, log_std, deterministic=True)

    assert torch.all(sample.action <= torch.as_tensor(space.high))
    assert torch.all(sample.action >= torch.as_tensor(space.low))
    assert torch.all(deterministic.action <= torch.as_tensor(space.high))
    assert torch.all(deterministic.action >= torch.as_tensor(space.low))
    assert torch.all(torch.isfinite(sample.log_prob))


def test_tanh_distribution_log_prob_recomputes_from_pre_tanh_action() -> None:
    space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    distribution = TanhDiagNormal(space, device="cpu")
    mean = torch.tensor([[0.1, -0.2]])
    log_std = torch.tensor([[-0.5, -0.5]])

    sample = distribution.sample(mean, log_std)
    recomputed = distribution.log_prob_from_pre_tanh(mean, log_std, sample.pre_tanh_action)

    torch.testing.assert_close(recomputed, sample.log_prob)


def test_tanh_distribution_handles_extreme_mean_without_nan_and_without_clipping() -> None:
    space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    distribution = TanhDiagNormal(space, device="cpu")
    mean = torch.tensor([[3.0]])
    log_std = torch.tensor([[-20.0]])

    sample = distribution.sample(mean, log_std, deterministic=True)

    assert torch.isfinite(sample.log_prob).all()
    assert sample.action.item() < 1.0
    assert sample.action.item() > -1.0


def test_tanh_distribution_rejects_bad_action_bounds() -> None:
    with pytest.raises(ValueError, match="finite"):
        TanhDiagNormal(
            gym.spaces.Box(low=-np.inf, high=1.0, shape=(1,), dtype=np.float32),
            device="cpu",
        )
    with pytest.raises(ValueError, match="positive"):
        TanhDiagNormal(
            gym.spaces.Box(low=1.0, high=1.0, shape=(1,), dtype=np.float32),
            device="cpu",
        )
