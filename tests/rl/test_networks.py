from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from homotopy_path_learning.rl.networks import ActorCritic


def make_agent() -> ActorCritic:
    return ActorCritic(
        observation_shape=(3,),
        action_space=gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        hidden_sizes=(8, 8),
        activation="tanh",
        actor_logstd_init=-0.5,
    )


def test_actor_critic_shapes_for_single_and_batch_inputs() -> None:
    agent = make_agent()
    observation = torch.zeros(3)
    action, raw, log_prob, entropy, value = agent.get_action_and_value(observation)

    assert action.shape == (1, 2)
    assert raw.shape == (1, 2)
    assert log_prob.shape == (1,)
    assert entropy.shape == (1,)
    assert value.shape == (1,)

    batch_value = agent.get_value(torch.zeros((5, 3)))
    assert batch_value.shape == (5,)


def test_deterministic_action_is_reproducible() -> None:
    agent = make_agent()
    observation = torch.ones((1, 3))

    first = agent.deterministic_action(observation)
    second = agent.deterministic_action(observation)

    torch.testing.assert_close(first, second)


def test_state_dict_round_trip_preserves_deterministic_action() -> None:
    agent = make_agent()
    clone = make_agent()
    observation = torch.tensor([[0.25, -0.5, 0.75]])
    clone.load_state_dict(agent.state_dict())

    torch.testing.assert_close(
        agent.deterministic_action(observation),
        clone.deterministic_action(observation),
    )
