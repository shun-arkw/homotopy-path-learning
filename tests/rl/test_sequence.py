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
from homotopy_path_learning.rl.sequence import SequenceActor


CONFIG_DIR = "experiments/multivariate_pham/configs/"
SMALL = dict(d_model=16, n_heads=2, n_layers=2, ff_dim=32)


def make_spec(name: str):
    config = load_config(CONFIG_DIR + name)
    return config, build_system_spec(config)


def real_dim_of(spec) -> int:
    return 2 * int(np.count_nonzero(spec.free_coefficient_mask))


def make_basis(spec, seed: int = 0) -> np.ndarray:
    real_dim = real_dim_of(spec)
    return make_orthonormal_basis(np.random.default_rng(seed), real_dim=real_dim, latent_dim=real_dim)


def n_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


@pytest.mark.parametrize(
    ("name", "bezier_degree"),
    [("exp-0014-n4-d2222.yaml", 2), ("exp-0014-n3-d333.yaml", 3)],
)
def test_output_shape_and_small_initial_output(name: str, bezier_degree: int) -> None:
    _, spec = make_spec(name)
    torch.manual_seed(0)
    actor = SequenceActor(spec, make_basis(spec), bezier_degree, **SMALL)
    real_dim = real_dim_of(spec)
    obs = torch.rand(5, real_dim) * 2 - 1
    out = actor(obs)
    assert out.shape == (5, (bezier_degree - 1) * real_dim)
    assert out.dtype == torch.float32
    assert out.abs().max() < 0.2
    assert actor.coefficient_delta(obs).shape == (5, bezier_degree - 1, real_dim)
    assert actor(obs[0]).shape == (1, (bezier_degree - 1) * real_dim)
    with pytest.raises(ValueError, match="obs must have shape"):
        actor(torch.zeros(2, real_dim + 1))


def test_output_is_basis_applied_to_coefficient_delta() -> None:
    _, spec = make_spec("exp-0014-n4-d2222.yaml")
    basis = make_basis(spec, seed=2)
    actor = SequenceActor(spec, basis, 3, **SMALL)
    for p in actor.parameters():
        if p.ndim > 1:
            torch.nn.init.normal_(p, std=0.3)
    obs = torch.rand(3, real_dim_of(spec)) * 2 - 1
    delta = actor.coefficient_delta(obs).detach().numpy()
    expected = np.einsum("bkm,mj->bkj", delta, basis.astype(np.float32)).reshape(3, -1)
    np.testing.assert_allclose(actor(obs).detach().numpy(), expected, atol=1e-5)


def test_works_on_dense_support_with_quadratic_terms() -> None:
    _, dense = make_spec("exp-0007-n2-d33-dense-seed0.yaml")
    real_dim = real_dim_of(dense)
    with pytest.raises(NotImplementedError):
        AttentionActor(dense, np.eye(real_dim), 3, **SMALL)  # the permutation-structured actor rejects it
    actor = SequenceActor(dense, make_basis(dense), 3, **SMALL)
    obs = torch.rand(4, real_dim) * 2 - 1
    out = actor(obs)
    assert out.shape == (4, 2 * real_dim)
    assert torch.isfinite(out).all()


def test_parameter_count_matches_attention_within_five_percent_and_basis_is_buffer() -> None:
    _, spec = make_spec("exp-0014-n4-d2222.yaml")
    basis = make_basis(spec)
    for kwargs in (dict(), dict(d_model=64, n_heads=4, n_layers=2, ff_dim=128)):
        seq = n_params(SequenceActor(spec, basis, 2, **kwargs))
        att = n_params(AttentionActor(spec, basis, 2, **kwargs))
        assert abs(seq - att) <= 0.05 * att, (kwargs, seq, att)
    actor = SequenceActor(spec, basis, 2)
    assert "basis" in actor.state_dict()
    assert "basis" not in dict(actor.named_parameters())


def test_invalid_arguments_are_rejected() -> None:
    _, spec = make_spec("exp-0004-smoke.yaml")
    with pytest.raises(ValueError, match="square"):
        SequenceActor(spec, np.eye(12)[:, :4], 3, **SMALL)
    with pytest.raises(ValueError, match="bezier_degree"):
        SequenceActor(spec, np.eye(12), 1, **SMALL)
    with pytest.raises(ValueError, match="divide"):
        SequenceActor(spec, np.eye(12), 3, d_model=16, n_heads=3, n_layers=1, ff_dim=8)


def _sequence_agent(spec, device="cpu"):
    real_dim = real_dim_of(spec)
    space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2 * real_dim,), dtype=np.float32)
    agent = ActorCritic(
        observation_shape=(real_dim,),
        action_space=space,
        hidden_sizes=(8,),
        actor_type="sequence",
        attention_kwargs=SMALL,
        system_spec=spec,
        basis=make_basis(spec, seed=3),
        bezier_degree=3,
        device=device,
    )
    return agent, space, real_dim


def test_actor_critic_sequence_needs_spec_and_rejects_unknown_type() -> None:
    space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    with pytest.raises(ValueError, match="requires"):
        ActorCritic(observation_shape=(12,), action_space=space, actor_type="sequence")
    with pytest.raises(ValueError, match="actor_type"):
        ActorCritic(observation_shape=(12,), action_space=space, actor_type="transformer")


def test_sequence_checkpoint_round_trip(tmp_path) -> None:
    config, spec = make_spec("exp-0004-smoke.yaml")
    agent, space, real_dim = _sequence_agent(spec)
    assert isinstance(agent.actor_mean, SequenceActor)
    for p in agent.actor_mean.parameters():  # non-trivial weights
        if p.ndim > 1:
            torch.nn.init.normal_(p, std=0.3)
    config = replace(
        config,
        ppo=replace(config.ppo, hidden_sizes=(8,), actor_type="sequence", attention_d_model=16, attention_heads=2,
                    attention_layers=2, attention_ff_dim=32),
        path=replace(config.path, bezier_degree=3, latent_dim=real_dim),
    )
    save_checkpoint(
        tmp_path / "m.pt",
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
    checkpoint = load_checkpoint(tmp_path / "m.pt")
    assert checkpoint["actor_type"] == "sequence"
    assert checkpoint["attention_kwargs"]["d_model"] == 16
    loaded = build_agent_from_checkpoint(checkpoint, action_space=space, device="cpu")
    assert loaded.actor_type == "sequence"
    assert isinstance(loaded.actor_mean, SequenceActor)
    torch.testing.assert_close(loaded.actor_mean.basis, agent.actor_mean.basis)
    obs = torch.rand(4, real_dim) * 2 - 1
    agent.eval()
    torch.testing.assert_close(loaded.actor_mean(obs), agent.actor_mean(obs))
    torch.testing.assert_close(loaded.deterministic_action(obs), agent.deterministic_action(obs))
