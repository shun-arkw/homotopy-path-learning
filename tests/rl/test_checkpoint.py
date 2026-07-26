from __future__ import annotations

from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
import torch

from homotopy_path_learning.config import load_config
from homotopy_path_learning.rl.checkpoint import (
    CheckpointError,
    build_agent_from_checkpoint,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint,
)
from homotopy_path_learning.rl.networks import ActorCritic
from homotopy_path_learning.systems import build_pham_spec


def test_checkpoint_round_trip_preserves_deterministic_action(tmp_path) -> None:
    config = load_config("experiments/multivariate_pham/configs/exp-0004-smoke.yaml")
    space = gym.spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
    agent = ActorCritic(
        observation_shape=(12,),
        action_space=space,
        hidden_sizes=(16,),
        actor_logstd_init=-0.5,
    )
    optimizer = torch.optim.Adam(agent.parameters())
    spec = build_pham_spec(config.system.degrees, config.system.free_exponents)
    checkpoint_path = tmp_path / "model.pt"

    save_checkpoint(
        checkpoint_path,
        agent=agent,
        optimizer=optimizer,
        config=replace(config, ppo=replace(config.ppo, hidden_sizes=(16,))),
        system_spec=spec,
        global_step=64,
        update=1,
        observation_shape=(12,),
        action_space=space,
        experiment_id=config.experiment.id,
        run_id="run",
        git_commit="abc",
    )
    checkpoint = load_checkpoint(checkpoint_path)
    validate_checkpoint(
        checkpoint,
        config=replace(config, ppo=replace(config.ppo, hidden_sizes=(16,))),
        system_spec=spec,
        observation_shape=(12,),
        action_space=space,
    )
    loaded_agent = build_agent_from_checkpoint(checkpoint, action_space=space, device="cpu")
    observation = torch.zeros((1, 12))

    torch.testing.assert_close(agent.deterministic_action(observation), loaded_agent.deterministic_action(observation))


def test_checkpoint_validation_rejects_shape_and_basis_mismatch(tmp_path) -> None:
    config = load_config("experiments/multivariate_pham/configs/exp-0004-smoke.yaml")
    space = gym.spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
    agent = ActorCritic(observation_shape=(12,), action_space=space)
    optimizer = torch.optim.Adam(agent.parameters())
    spec = build_pham_spec(config.system.degrees, config.system.free_exponents)
    path = tmp_path / "model.pt"
    save_checkpoint(
        path,
        agent=agent,
        optimizer=optimizer,
        config=config,
        system_spec=spec,
        global_step=1,
        update=1,
        observation_shape=(12,),
        action_space=space,
        experiment_id=config.experiment.id,
        run_id="run",
        git_commit="abc",
    )
    checkpoint = load_checkpoint(path)

    with pytest.raises(CheckpointError, match="observation"):
        validate_checkpoint(
            checkpoint,
            config=config,
            system_spec=spec,
            observation_shape=(10,),
            action_space=space,
        )
    with pytest.raises(CheckpointError, match="basis"):
        validate_checkpoint(
            checkpoint,
            config=replace(config, path=replace(config.path, basis_seed=999)),
            system_spec=spec,
            observation_shape=(12,),
            action_space=space,
        )
