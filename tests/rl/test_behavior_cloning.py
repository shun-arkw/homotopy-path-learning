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


def test_zero_epochs_keeps_initial_weights():
    obs, act = make_data(80)
    agent = make_agent()
    initial = copy.deepcopy(agent.state_dict())
    result, optimizer = run(agent, obs, act, epochs=0)

    assert result.history == []
    assert result.best_epoch == 0
    assert result.n_train + result.n_val == 80
    for key, value in agent.state_dict().items():
        assert torch.equal(value, initial[key])
    order = np.random.default_rng(0).permutation(80)
    val_idx = order[result.n_train :]
    with torch.no_grad():
        pred = predicted_actions(agent, torch.as_tensor(obs[val_idx]))
        loss = torch.nn.functional.mse_loss(pred, torch.as_tensor(act[val_idx])).item()
    assert result.best_val_loss == pytest.approx(loss, abs=1e-7)
    assert np.isfinite(result.best_val_loss)
    assert isinstance(optimizer, torch.optim.Optimizer)


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
        ({"kw": {"epochs": -1}}, ValueError),
        ({"kw": {"epochs": 1.5}}, TypeError),
        ({"kw": {"epochs": True}}, TypeError),
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


def equivariant_weights() -> np.ndarray:
    """Weights ``(OBS_DIM, ACTION_DIM)`` commuting with the half-swap on obs and actions."""

    rng = np.random.default_rng(123)
    block = rng.normal(size=(OBS_DIM // 2, ACTION_DIM // 2)) * 0.5
    other = rng.normal(size=(OBS_DIM // 2, ACTION_DIM // 2)) * 0.5
    return np.block([[block, other], [other, block]])


def equivariant_task(n: int = 120, seed: int = 0):
    """Labels tanh(W obs) with W commuting with a coordinate swap + sign flip group."""

    rng = np.random.default_rng(seed)
    # Group on obs: identity and the map swapping the two halves of the observation.
    half = OBS_DIM // 2
    swap_obs = np.zeros((OBS_DIM, OBS_DIM), dtype=np.float32)
    swap_obs[:half, half:] = np.eye(half)
    swap_obs[half:, :half] = np.eye(half)
    swap_act = np.zeros((ACTION_DIM, ACTION_DIM), dtype=np.float32)
    swap_act[: ACTION_DIM // 2, ACTION_DIM // 2 :] = np.eye(ACTION_DIM // 2)
    swap_act[ACTION_DIM // 2 :, : ACTION_DIM // 2] = np.eye(ACTION_DIM // 2)
    w = equivariant_weights()
    obs = rng.uniform(-1, 1, size=(n, OBS_DIM)).astype(np.float32)
    act = np.tanh(obs @ w).astype(np.float32)
    identity = (np.eye(OBS_DIM, dtype=np.float32), np.eye(ACTION_DIM, dtype=np.float32))
    return obs, act, [identity, (swap_obs, swap_act)]


def test_augmentation_runs_is_deterministic_and_evaluates_original_data():
    obs, act, groups = equivariant_task()
    # swap equivariance of the labels
    # the labels are equivariant: label(M_obs obs) == M_act label(obs)
    np.testing.assert_allclose(
        np.tanh((obs @ groups[1][0].T) @ equivariant_weights()),
        act @ groups[1][1].T,
        atol=1e-6,
    )
    first_agent, second_agent = make_agent(), make_agent()
    first, _ = run(first_agent, obs, act, epochs=5, augmentations=groups)
    second, _ = run(second_agent, obs, act, epochs=5, augmentations=groups)
    plain, _ = run(make_agent(), obs, act, epochs=5)

    assert first.n_augmentations == 2
    assert first.clipped_label_fraction == 0.0
    assert first.history == second.history
    assert first.history != plain.history
    assert plain.n_augmentations == 0
    x = torch.as_tensor(obs)
    assert torch.equal(predicted_actions(first_agent, x), predicted_actions(second_agent, x))


def test_augmentation_clips_labels_and_rejects_bad_shapes():
    obs, act, groups = equivariant_task(60)
    scale = (np.eye(OBS_DIM, dtype=np.float32), 3.0 * np.eye(ACTION_DIM, dtype=np.float32))
    result, _ = run(make_agent(), obs, act, epochs=2, augmentations=[scale])
    assert result.clipped_label_fraction > 0.0

    identity_obs, identity_act = groups[0]
    for bad in (
        [(identity_obs[:5, :5], identity_act)],
        [(identity_obs, identity_act[:3, :3])],
        [(identity_obs, np.full_like(identity_act, np.nan))],
        [],
    ):
        with pytest.raises(ValueError):
            run(make_agent(), obs, act, epochs=1, augmentations=bad)


def multi_label_data(n: int = 60, k: int = 2, seed: int = 0):
    obs, act = make_data(n, seed)
    rng = np.random.default_rng(seed + 1)
    others = np.clip(act[:, None, :] + 0.1 * rng.normal(size=(n, k - 1, ACTION_DIM)), -1, 1).astype(np.float32)
    return obs, np.concatenate([act[:, None, :], others], axis=1)


def test_two_d_and_single_label_three_d_actions_are_identical():
    obs, act = make_data(80)
    a1, a2 = make_agent(), make_agent()
    first, _ = run(a1, obs, act, epochs=4)
    second, _ = run(a2, obs, act[:, None, :], epochs=4)
    assert first.history == second.history
    assert (first.label_reduction, first.n_labels_per_target) == ("mean", 1)
    x = torch.as_tensor(obs)
    assert torch.equal(predicted_actions(a1, x), predicted_actions(a2, x))


def test_min_reduction_with_identical_candidates_matches_single_label():
    obs, act = make_data(80)
    same = np.repeat(act[:, None, :], 2, axis=1)
    a1, a2 = make_agent(), make_agent()
    first, _ = run(a1, obs, act, epochs=4)
    second, _ = run(a2, obs, same, epochs=4, label_reduction="min")
    assert second.n_labels_per_target == 2 and second.label_reduction == "min"
    for one, two in zip(first.history, second.history):
        assert two["train_loss"] == pytest.approx(one["train_loss"], rel=1e-5, abs=1e-8)
        assert two["val_loss"] == pytest.approx(one["val_loss"], rel=1e-5, abs=1e-8)
        assert two["switch_fraction"] == 0.0  # identical candidates: the first index is always chosen
    assert "switch_fraction" not in first.history[0]
    x = torch.as_tensor(obs)
    np.testing.assert_allclose(
        predicted_actions(a1, x).detach().numpy(), predicted_actions(a2, x).detach().numpy(), atol=1e-5
    )


def test_min_loss_uses_nearest_candidate_by_hand():
    obs, act = make_data(40)
    far = np.where(act > 0, -1.0, 1.0)  # a distant second candidate
    labels = np.stack([act, far.astype(np.float32)], axis=1).astype(np.float32)
    agent = make_agent()
    result, _ = run(agent, obs, labels, epochs=1, label_reduction="min", learning_rate=1e-9)
    # (essentially) untrained network: recompute the validation loss by hand
    order = np.random.default_rng(0).permutation(40)
    val_idx = order[result.n_train :]
    with torch.no_grad():
        pred = predicted_actions(agent, torch.as_tensor(obs[val_idx])).numpy()
    per = ((pred[:, None, :] - labels[val_idx]) ** 2).mean(axis=2)
    assert result.history[-1]["val_loss"] == pytest.approx(per.min(axis=1).mean(), abs=1e-6)
    # the loss with a far second candidate equals the loss of the near candidate alone when it is nearer
    near_only = ((pred - act[val_idx]) ** 2).mean(axis=1)
    assert np.all(per.min(axis=1) <= near_only + 1e-12)


def test_min_reduction_learns_the_nearer_mode_and_records_switch_fraction():
    obs, act = make_data(100)
    labels = np.stack([act, -act], axis=1).astype(np.float32)
    agent = make_agent()
    result, _ = run(agent, obs, labels, epochs=40, label_reduction="min")
    assert result.history[-1]["val_loss"] < 0.2 * result.history[0]["val_loss"]
    assert all(0.0 <= item["switch_fraction"] <= 1.0 for item in result.history)


def test_split_is_by_target_and_mean_with_multiple_labels_is_error():
    obs, labels = multi_label_data(50, 3)
    result, _ = run(make_agent(), obs, labels, epochs=1, label_reduction="min", val_fraction=0.2)
    assert result.n_train + result.n_val == 50 and result.n_val == 10
    with pytest.raises(ValueError):
        run(make_agent(), obs, labels, epochs=1)  # default "mean" with K=3
    with pytest.raises(ValueError):
        run(make_agent(), obs, labels, epochs=1, label_reduction="max")
    with pytest.raises(ValueError):
        run(make_agent(), obs, labels[:, :0], epochs=1, label_reduction="min")
    with pytest.raises(ValueError):
        run(make_agent(), obs, labels[..., None], epochs=1, label_reduction="min")


def test_augmentation_applies_same_element_to_all_labels():
    obs, act, groups = equivariant_task(60)
    other = np.clip(act + 0.1, -1, 1).astype(np.float32)
    labels = np.stack([act, other], axis=1)
    swap_act = torch.as_tensor(groups[1][1])
    # reproduce the einsum used inside the trainer: the difference of the two augmented
    # candidates equals M_act applied to the difference before augmentation
    choice = torch.tensor([0, 1] * 30)
    stack = torch.stack([torch.as_tensor(groups[0][1]), swap_act])
    raw = torch.einsum("bij,bkj->bki", stack[choice], torch.as_tensor(labels))
    diff_after = raw[:, 1] - raw[:, 0]
    diff_before = torch.as_tensor(labels[:, 1] - labels[:, 0])
    expected = torch.einsum("bij,bj->bi", stack[choice], diff_before)
    torch.testing.assert_close(diff_after, expected)
    # and the trainer runs deterministically with K=2 augmentation
    r1, _ = run(make_agent(), obs, labels, epochs=3, augmentations=groups, label_reduction="min")
    r2, _ = run(make_agent(), obs, labels, epochs=3, augmentations=groups, label_reduction="min")
    assert r1.history == r2.history and r1.n_augmentations == 2 and r1.clipped_label_fraction == 0.0
    plain, _ = run(make_agent(), obs, labels, epochs=3, label_reduction="min")
    assert r1.history != plain.history


def test_augmentation_clip_fraction_counts_all_labels():
    obs, act, _ = equivariant_task(60)
    labels = np.stack([act, 0.5 * act], axis=1).astype(np.float32)
    scale = (np.eye(OBS_DIM, dtype=np.float32), 3.0 * np.eye(ACTION_DIM, dtype=np.float32))
    result, _ = run(make_agent(), obs, labels, epochs=1, augmentations=[scale], label_reduction="min")
    train_idx = np.random.default_rng(0).permutation(60)[: result.n_train]
    expected = (np.abs(3.0 * labels[train_idx]) > 1.0 + 1e-6).mean()
    assert result.clipped_label_fraction == pytest.approx(expected)
