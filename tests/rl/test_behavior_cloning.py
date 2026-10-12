from __future__ import annotations

import copy
from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
import torch

from homotopy_path_learning.config import build_system_spec, load_config
from homotopy_path_learning.rl.behavior_cloning import predicted_actions, train_behavior_cloning
from homotopy_path_learning.rl.checkpoint import (
    build_agent_from_checkpoint,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint,
)
from homotopy_path_learning.rl.networks import ActorCritic

OBS_DIM = 12
ACTION_DIM = 8
SMOKE_CONFIG = "experiments/multivariate_pham/configs/exp-0004-smoke.yaml"


def make_space() -> gym.spaces.Box:
    return gym.spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)


def make_agent(seed: int = 0, hidden=(16, 16)) -> ActorCritic:
    torch.manual_seed(seed)
    return ActorCritic(observation_shape=(OBS_DIM,), action_space=make_space(), hidden_sizes=hidden)


def make_data(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    obs = rng.uniform(-1.0, 1.0, size=(n, OBS_DIM)).astype(np.float32)
    weights = rng.normal(size=(OBS_DIM, ACTION_DIM)) * 0.5
    return obs, np.tanh(obs @ weights).astype(np.float32)


def run(agent, obs, act, **overrides):
    kwargs = dict(epochs=60, learning_rate=3e-3, batch_size=32, val_fraction=0.2, seed=0)
    kwargs.update(overrides)
    return train_behavior_cloning(agent, obs, act, **kwargs)


def test_validation_loss_decreases_and_best_weights_restored():
    obs, act = make_data()
    agent = make_agent()
    result, _ = run(agent, obs, act)

    assert result.n_train + result.n_val == 200
    assert len(result.history) == 60
    assert result.history[-1]["val_loss"] < 0.2 * result.history[0]["val_loss"]
    assert result.best_val_loss == min(item["val_loss"] for item in result.history)
    assert result.history[result.best_epoch - 1]["val_loss"] == result.best_val_loss
    # recompute the validation loss of the returned model on the same split
    order = np.random.default_rng(0).permutation(200)
    val_idx = order[result.n_train :]
    with torch.no_grad():
        pred = predicted_actions(agent, torch.as_tensor(obs[val_idx]))
        loss = torch.nn.functional.mse_loss(pred, torch.as_tensor(act[val_idx])).item()
    assert loss == pytest.approx(result.best_val_loss, abs=1e-6)


def test_critic_is_untouched():
    obs, act = make_data(50)
    agent = make_agent()
    before = copy.deepcopy(agent.critic.state_dict())
    run(agent, obs, act, epochs=3)
    for key, value in agent.critic.state_dict().items():
        assert torch.equal(value, before[key])


def test_determinism_and_inputs_not_mutated():
    obs, act = make_data(80)
    obs_copy, act_copy = obs.copy(), act.copy()
    first_agent, second_agent = make_agent(), make_agent()
    first, _ = run(first_agent, obs, act, epochs=5)
    second, _ = run(second_agent, obs, act, epochs=5)

    assert first.history == second.history
    np.testing.assert_array_equal(obs, obs_copy)
    np.testing.assert_array_equal(act, act_copy)
    x = torch.as_tensor(obs)
    assert torch.equal(predicted_actions(first_agent, x), predicted_actions(second_agent, x))


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"obs": lambda o, a: (o[:, :5], a)}, ValueError),
        ({"obs": lambda o, a: (o, a[:, :3])}, ValueError),
        ({"obs": lambda o, a: (o[:10], a)}, ValueError),
        ({"obs": lambda o, a: (o[0], a[0])}, ValueError),
        ({"obs": lambda o, a: (o[:1], a[:1])}, ValueError),
        ({"obs": lambda o, a: (o, a * 5.0)}, ValueError),
        ({"obs": lambda o, a: (np.where(o > 0.9, np.nan, o), a)}, ValueError),
        ({"kw": {"val_fraction": 0.0}}, ValueError),
        ({"kw": {"val_fraction": 1.0}}, ValueError),
        ({"kw": {"epochs": 0}}, ValueError),
        ({"kw": {"batch_size": -1}}, ValueError),
        ({"kw": {"learning_rate": 0.0}}, ValueError),
    ],
)
def test_invalid_inputs_rejected(change, error):
    obs, act = make_data(40)
    if "obs" in change:
        obs, act = change["obs"](obs, act)
    with pytest.raises(error):
        run(make_agent(), obs, act, **{"epochs": 2, **change.get("kw", {})})


def test_checkpoint_round_trip(tmp_path):
    obs, act = make_data(60)
    agent = make_agent(hidden=(8, 8))
    _, optimizer = run(agent, obs, act, epochs=3)
    config = load_config(SMOKE_CONFIG)
    config = replace(config, ppo=replace(config.ppo, hidden_sizes=(8, 8)))
    spec = build_system_spec(config)
    space = make_space()
    path = tmp_path / "ckpt" / "last.pt"
    save_checkpoint(
        path,
        agent=agent,
        optimizer=optimizer,
        config=config,
        system_spec=spec,
        global_step=0,
        update=0,
        observation_shape=(OBS_DIM,),
        action_space=space,
        experiment_id="EXP-TEST",
        run_id="bc",
        git_commit="test",
    )
    checkpoint = load_checkpoint(path)
    validate_checkpoint(checkpoint, config=config, system_spec=spec, observation_shape=(OBS_DIM,), action_space=space)
    loaded = build_agent_from_checkpoint(checkpoint, action_space=space, device="cpu")

    x = torch.as_tensor(obs)
    with torch.no_grad():
        assert torch.equal(predicted_actions(agent, x), predicted_actions(loaded, x))
