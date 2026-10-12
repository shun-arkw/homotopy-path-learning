from __future__ import annotations

from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
import torch

from homotopy_path_learning.config import build_system_spec, load_config
from homotopy_path_learning.paths.parameterization import make_orthonormal_basis
from homotopy_path_learning.rl.attention import AttentionActor
from homotopy_path_learning.rl.checkpoint import build_agent_from_checkpoint, load_checkpoint, save_checkpoint
from homotopy_path_learning.rl.networks import ActorCritic
from homotopy_path_learning.systems.symmetry import enumerate_symmetries, free_real_matrix


CONFIG_DIR = "experiments/multivariate_pham/configs/"
SMALL = dict(d_model=16, n_heads=2, n_layers=2, ff_dim=32)


def delta_np(actor: AttentionActor, obs: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return actor.coefficient_delta(torch.as_tensor(obs)).numpy()


def make_spec(name: str):
    config = load_config(CONFIG_DIR + name)
    return config, build_system_spec(config)


def make_basis(spec, seed: int = 0) -> np.ndarray:
    real_dim = 2 * int(np.count_nonzero(spec.free_coefficient_mask))
    return make_orthonormal_basis(np.random.default_rng(seed), real_dim=real_dim, latent_dim=real_dim)


@pytest.mark.parametrize("name", ["exp-0007-n3-d222-seed0.yaml", "exp-0014-n4-d2222.yaml"])
def test_coefficient_delta_is_permutation_equivariant(name: str) -> None:
    config, spec = make_spec(name)
    torch.manual_seed(0)
    actor = AttentionActor(spec, make_basis(spec), 3, **SMALL).double()
    for head in (actor.edge_head, actor.node_head):  # make the output non-negligible
        torch.nn.init.normal_(head.weight, std=0.5)
    rng = np.random.default_rng(1)
    real_dim = 2 * int(np.count_nonzero(spec.free_coefficient_mask))
    obs = rng.uniform(-1.0, 1.0, size=(3, real_dim))
    delta = delta_np(actor, obs)
    assert delta.shape == (3, 2, real_dim)
    assert np.abs(delta).max() > 0.05
    elements = [g for g in enumerate_symmetries(spec) if g.kind in ("identity", "perm-only")]
    assert len(elements) >= 6
    checked_nontrivial = 0
    for g in elements:
        t = free_real_matrix(spec, g)
        moved = delta_np(actor, obs @ t.T)
        expected = delta @ t.T
        np.testing.assert_allclose(moved, expected, atol=1e-5, rtol=0.0)
        if not g.is_identity:
            checked_nontrivial += 1
            assert not np.allclose(delta, moved, atol=1e-3)
    assert checked_nontrivial >= 5


def test_shapes_dtype_and_small_initial_output() -> None:
    config, spec = make_spec("exp-0014-n4-d2222.yaml")
    actor = AttentionActor(spec, make_basis(spec), 3, **SMALL)
    real_dim = 2 * int(np.count_nonzero(spec.free_coefficient_mask))
    obs = torch.rand(5, real_dim) * 2 - 1
    out = actor(obs)
    assert out.shape == (5, 2 * real_dim)
    assert out.dtype == torch.float32
    assert out.abs().max() < 0.2
    assert actor.coefficient_delta(obs).shape == (5, 2, real_dim)
    with pytest.raises(ValueError, match="obs must have shape"):
        actor(torch.zeros(2, real_dim + 1))


def test_default_size_parameter_count_is_reported_and_basis_is_buffer() -> None:
    config, spec = make_spec("exp-0014-n4-d2222.yaml")
    actor = AttentionActor(spec, make_basis(spec), 2)
    assert sum(p.numel() for p in actor.parameters()) > 100_000
    assert "basis" in actor.state_dict()
    assert "basis" not in dict(actor.named_parameters())


def test_unsupported_support_and_nonsquare_basis_are_rejected() -> None:
    _, dense = make_spec("exp-0007-n2-d33-dense-seed0.yaml")
    real_dim = 2 * int(np.count_nonzero(dense.free_coefficient_mask))
    with pytest.raises(NotImplementedError):
        AttentionActor(dense, np.eye(real_dim), 3, **SMALL)
    _, spec = make_spec("exp-0004-smoke.yaml")
    with pytest.raises(ValueError, match="square"):
        AttentionActor(spec, np.eye(12)[:, :4], 3, **SMALL)


def test_actor_critic_default_is_unchanged_mlp_and_attention_needs_spec() -> None:
    space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    agent = ActorCritic(observation_shape=(3,), action_space=space)
    assert agent.actor_type == "mlp"
    assert isinstance(agent.actor_mean, torch.nn.Sequential)
    with pytest.raises(ValueError, match="requires"):
        ActorCritic(observation_shape=(12,), action_space=space, actor_type="attention")


def _attention_agent(config, spec, device="cpu"):
    real_dim = 2 * int(np.count_nonzero(spec.free_coefficient_mask))
    space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2 * real_dim,), dtype=np.float32)
    agent = ActorCritic(
        observation_shape=(real_dim,),
        action_space=space,
        hidden_sizes=(8,),
        actor_type="attention",
        attention_kwargs=SMALL,
        system_spec=spec,
        basis=make_basis(spec, seed=3),
        bezier_degree=3,
        device=device,
    )
    return agent, space, real_dim


def _save(path, agent, config, spec, space, real_dim):
    save_checkpoint(
        path,
        agent=agent,
        optimizer=torch.optim.Adam(agent.parameters()),
        config=config,
        system_spec=spec,
        global_step=0,
        update=0,
        observation_shape=(real_dim,),
        action_space=space,
        experiment_id="E",
        run_id="r",
        git_commit="abc",
    )


def test_attention_checkpoint_round_trip(tmp_path) -> None:
    config, spec = make_spec("exp-0004-smoke.yaml")
    agent, space, real_dim = _attention_agent(config, spec)
    config = replace(config, ppo=replace(config.ppo, hidden_sizes=(8,)))
    for p in agent.actor_mean.parameters():  # non-trivial weights
        torch.nn.init.normal_(p, std=0.3) if p.ndim > 1 else None
    config = replace(config, path=replace(config.path, bezier_degree=3, latent_dim=real_dim))
    _save(tmp_path / "m.pt", agent, config, spec, space, real_dim)
    checkpoint = load_checkpoint(tmp_path / "m.pt")
    assert checkpoint["actor_type"] == "attention"
    loaded = build_agent_from_checkpoint(checkpoint, action_space=space, device="cpu")
    assert loaded.actor_type == "attention"
    torch.testing.assert_close(loaded.actor_mean.basis, agent.actor_mean.basis)
    obs = torch.rand(4, real_dim) * 2 - 1
    agent.eval()
    torch.testing.assert_close(loaded.actor_mean(obs), agent.actor_mean(obs))
    torch.testing.assert_close(loaded.deterministic_action(obs), agent.deterministic_action(obs))


def test_legacy_mlp_checkpoint_without_actor_keys_loads_as_mlp(tmp_path) -> None:
    config, spec = make_spec("exp-0004-smoke.yaml")
    space = gym.spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
    agent = ActorCritic(observation_shape=(12,), action_space=space, hidden_sizes=(16,))
    config = replace(config, ppo=replace(config.ppo, hidden_sizes=(16,)))
    _save(tmp_path / "m.pt", agent, config, spec, space, 12)
    checkpoint = load_checkpoint(tmp_path / "m.pt")
    # Emulate a checkpoint written before this feature existed.
    for key in ("actor_type", "attention_kwargs"):
        checkpoint.pop(key)
    for key in ("actor_type", "attention_d_model", "attention_heads", "attention_layers", "attention_ff_dim"):
        checkpoint["ppo_config"].pop(key)
    loaded = build_agent_from_checkpoint(checkpoint, action_space=space, device="cpu")
    assert loaded.actor_type == "mlp"
    obs = torch.rand(1, 12)
    torch.testing.assert_close(loaded.deterministic_action(obs), agent.deterministic_action(obs))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_attention_cuda_matches_cpu_forward_and_bc_steps() -> None:
    from homotopy_path_learning.rl.behavior_cloning import predicted_actions, train_behavior_cloning

    config, spec = make_spec("exp-0004-smoke.yaml")
    cpu_agent, space, real_dim = _attention_agent(config, spec)
    for p in cpu_agent.actor_mean.parameters():
        if p.ndim > 1:
            torch.nn.init.normal_(p, std=0.3)
    cuda_agent, _, _ = _attention_agent(config, spec, device="cuda")
    cuda_agent.load_state_dict(cpu_agent.state_dict())
    rng = np.random.default_rng(0)
    obs = rng.uniform(-1, 1, size=(16, real_dim)).astype(np.float32)
    with torch.no_grad():
        cpu_out = cpu_agent.actor_mean(torch.as_tensor(obs))
        cuda_out = cuda_agent.actor_mean(torch.as_tensor(obs, device="cuda"))
    torch.testing.assert_close(cuda_out.cpu(), cpu_out, atol=1e-4, rtol=0.0)

    actions = rng.uniform(-0.5, 0.5, size=(16, 2 * real_dim)).astype(np.float32)
    kwargs = dict(epochs=2, learning_rate=1e-3, batch_size=8, val_fraction=0.25, seed=0)
    train_behavior_cloning(cpu_agent, obs, actions, device="cpu", **kwargs)
    result, _ = train_behavior_cloning(cuda_agent, obs, actions, device="cuda", **kwargs)
    assert result.n_train == 12
    with torch.no_grad():
        cpu_after = predicted_actions(cpu_agent, torch.as_tensor(obs))
        cuda_after = predicted_actions(cuda_agent, torch.as_tensor(obs, device="cuda"))
    torch.testing.assert_close(cuda_after.cpu(), cpu_after, atol=1e-3, rtol=0.0)
