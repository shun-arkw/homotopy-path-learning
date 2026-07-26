from __future__ import annotations

from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
import torch

from homotopy_path_learning.config import load_config
from homotopy_path_learning.rl.networks import ActorCritic
from homotopy_path_learning.rl.ppo import clipped_policy_loss, ppo_update
from homotopy_path_learning.rl.rollout import collect_rollout, compute_gae


class OneStepQuadraticEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, *, fail_on_step: bool = False) -> None:
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.fail_on_step = fail_on_step
        self.reset_seeds: list[int | None] = []
        self._obs = np.zeros(2, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        del options
        self.reset_seeds.append(seed)
        self._obs = self.np_random.uniform(-1.0, 1.0, size=2).astype(np.float32)
        return self._obs.copy(), {}

    def step(self, action):
        if self.fail_on_step:
            raise RuntimeError("step failed")
        target = np.tanh(self._obs)
        reward = -float(np.sum((np.asarray(action, dtype=np.float32) - target) ** 2))
        info = {
            "linear_cost": 1.0,
            "bezier_cost": 1.0 - reward,
            "cost_improvement": reward,
            "linear_success": True,
            "bezier_success": True,
            "linear_n_success": 1,
            "bezier_n_success": 1,
            "linear_accepted_steps": 1,
            "bezier_accepted_steps": 1,
            "linear_rejected_steps": 0,
            "bezier_rejected_steps": 0,
        }
        return self._obs.copy(), reward, True, False, info


def small_agent() -> ActorCritic:
    return ActorCritic(
        observation_shape=(2,),
        action_space=gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        hidden_sizes=(8,),
        activation="tanh",
        actor_logstd_init=-0.5,
    )


def small_settings():
    config = load_config("experiments/multivariate_pham/configs/exp-0004-smoke.yaml")
    return replace(
        config.ppo,
        rollout_steps=8,
        minibatch_size=4,
        total_timesteps=8,
        update_epochs=2,
        target_kl=None,
    )


def test_compute_gae_one_step_terminal_matches_hand_calculation() -> None:
    advantages, returns = compute_gae(
        rewards=torch.tensor([2.0]),
        values=torch.tensor([0.5]),
        next_values=torch.tensor([99.0]),
        terminated=torch.tensor([True]),
        truncated=torch.tensor([False]),
        gamma=0.99,
        gae_lambda=0.95,
    )

    torch.testing.assert_close(advantages, torch.tensor([1.5]))
    torch.testing.assert_close(returns, torch.tensor([2.0]))


def test_compute_gae_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="identical"):
        compute_gae(
            rewards=torch.zeros(2),
            values=torch.zeros(1),
            next_values=torch.zeros(2),
            terminated=torch.zeros(2, dtype=torch.bool),
            truncated=torch.zeros(2, dtype=torch.bool),
            gamma=0.99,
            gae_lambda=0.95,
        )


def test_collect_rollout_resets_after_terminal_without_reusing_seed() -> None:
    env = OneStepQuadraticEnv()
    agent = small_agent()

    batch = collect_rollout(env, agent, rollout_steps=4, device="cpu", initial_seed=123)

    assert batch.observations.shape == (4, 2)
    assert batch.actions.shape == (4, 2)
    assert batch.pre_tanh_actions.shape == (4, 2)
    assert batch.rewards.shape == (4,)
    assert env.reset_seeds[0] == 123
    assert env.reset_seeds[1:] == [None, None, None, None]
    assert all("linear_cost" in metrics for metrics in batch.env_metrics)


def test_collect_rollout_does_not_swallow_environment_exceptions() -> None:
    with pytest.raises(RuntimeError, match="step failed"):
        collect_rollout(OneStepQuadraticEnv(fail_on_step=True), small_agent(), rollout_steps=1, device="cpu")


def test_ppo_update_returns_finite_metrics_and_changes_parameters() -> None:
    env = OneStepQuadraticEnv()
    agent = small_agent()
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
    batch = collect_rollout(env, agent, rollout_steps=8, device="cpu", initial_seed=123)
    before = [param.detach().clone() for param in agent.parameters()]

    metrics = ppo_update(
        agent=agent,
        optimizer=optimizer,
        batch=batch,
        settings=small_settings(),
        update_index=1,
        total_updates=1,
    )

    assert all(
        value is None or np.isfinite(float(value))
        for value in metrics.as_dict().values()
    )
    assert any(not torch.equal(old, new) for old, new in zip(before, agent.parameters()))


def test_clipped_policy_objective_matches_small_manual_case() -> None:
    loss = clipped_policy_loss(
        new_log_probs=torch.log(torch.tensor([1.3, 0.7])),
        old_log_probs=torch.zeros(2),
        advantages=torch.tensor([1.0, -1.0]),
        clip_coef=0.2,
    )

    expected = torch.tensor((-1.2 + 0.8) / 2.0)
    torch.testing.assert_close(loss, expected)
