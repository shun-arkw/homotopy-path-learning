"""Checkpoint save/load helpers for Phase 6 PPO experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from homotopy_path_learning.config import Phase6Config, config_to_dict
from homotopy_path_learning.rl.networks import ActorCritic
from homotopy_path_learning.systems.spec import PolynomialSystemSpec


CHECKPOINT_FORMAT_VERSION = 1


class CheckpointError(ValueError):
    """Raised when a checkpoint is incompatible with the requested run."""


def system_spec_to_dict(spec: PolynomialSystemSpec) -> dict[str, Any]:
    """Serialize a polynomial system spec with zero-based Python indices."""

    return {
        "n_vars": int(spec.n_vars),
        "degrees": list(spec.degrees),
        "exponents": spec.exponents.tolist(),
        "offsets": spec.offsets.tolist(),
        "leading_indices": spec.leading_indices.tolist(),
        "constant_indices": spec.constant_indices.tolist(),
        "supports": [[list(exp) for exp in block] for block in spec.supports or ()],
    }


def _as_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(v) for v in value)


def save_checkpoint(
    path: str | Path,
    *,
    agent: ActorCritic,
    optimizer: torch.optim.Optimizer,
    config: Phase6Config,
    system_spec: PolynomialSystemSpec,
    global_step: int,
    update: int,
    observation_shape: tuple[int, ...],
    action_space: gym.spaces.Box,
    experiment_id: str,
    run_id: str,
    git_commit: str,
) -> None:
    """Save a PPO checkpoint sufficient for deterministic evaluation."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": int(global_step),
        "update": int(update),
        "ppo_config": config_to_dict(config)["ppo"],
        "config": config_to_dict(config),
        "observation_shape": tuple(int(v) for v in observation_shape),
        "action_shape": tuple(int(v) for v in action_space.shape),
        "action_low": np.array(action_space.low, dtype=np.float32, copy=True),
        "action_high": np.array(action_space.high, dtype=np.float32, copy=True),
        "experiment_id": experiment_id,
        "run_id": run_id,
        "basis_seed": int(config.path.basis_seed),
        "bezier_degree": int(config.path.bezier_degree),
        "latent_dim": int(config.path.latent_dim),
        "system_spec": system_spec_to_dict(system_spec),
        "git_commit": git_commit,
        "torch_version": torch.__version__,
    }
    torch.save(payload, output_path)


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint payload."""

    return torch.load(Path(path), map_location=map_location)


def validate_checkpoint(
    checkpoint: dict[str, Any],
    *,
    config: Phase6Config,
    system_spec: PolynomialSystemSpec,
    observation_shape: tuple[int, ...],
    action_space: gym.spaces.Box,
) -> None:
    """Reject checkpoints that do not match the requested environment."""

    if int(checkpoint.get("checkpoint_format_version", -1)) != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointError("checkpoint format version mismatch.")
    if _as_tuple(checkpoint["observation_shape"]) != tuple(observation_shape):
        raise CheckpointError("observation shape mismatch.")
    if _as_tuple(checkpoint["action_shape"]) != tuple(action_space.shape):
        raise CheckpointError("action shape mismatch.")
    if not np.array_equal(
        np.array(checkpoint["action_low"], dtype=np.float32),
        np.array(action_space.low, dtype=np.float32),
    ):
        raise CheckpointError("action low bounds mismatch.")
    if not np.array_equal(
        np.array(checkpoint["action_high"], dtype=np.float32),
        np.array(action_space.high, dtype=np.float32),
    ):
        raise CheckpointError("action high bounds mismatch.")
    if int(checkpoint["basis_seed"]) != int(config.path.basis_seed):
        raise CheckpointError("basis seed mismatch.")
    if int(checkpoint["bezier_degree"]) != int(config.path.bezier_degree):
        raise CheckpointError("Bezier degree mismatch.")
    if int(checkpoint["latent_dim"]) != int(config.path.latent_dim):
        raise CheckpointError("latent dimension mismatch.")
    if checkpoint["system_spec"] != system_spec_to_dict(system_spec):
        raise CheckpointError("system specification mismatch.")


def build_agent_from_checkpoint(
    checkpoint: dict[str, Any],
    *,
    action_space: gym.spaces.Box,
    device: torch.device | str,
) -> ActorCritic:
    """Construct and load an ``ActorCritic`` from checkpoint metadata."""

    ppo = checkpoint["ppo_config"]
    agent = ActorCritic(
        observation_shape=_as_tuple(checkpoint["observation_shape"]),
        action_space=action_space,
        hidden_sizes=tuple(int(v) for v in ppo["hidden_sizes"]),
        activation=str(ppo["activation"]),
        actor_logstd_init=float(ppo["actor_logstd_init"]),
        device=device,
    )
    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.to(device)
    agent.eval()
    return agent


__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "CheckpointError",
    "build_agent_from_checkpoint",
    "load_checkpoint",
    "save_checkpoint",
    "system_spec_to_dict",
    "validate_checkpoint",
]
