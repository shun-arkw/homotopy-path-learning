"""Strict YAML configuration for multivariate Pham experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from homotopy_path_learning.backends.types import TrackerConfig
from homotopy_path_learning.envs import BezierPhamEnv
from homotopy_path_learning.systems import ComplexUniformSampler, build_pham_spec
from homotopy_path_learning.systems.spec import PolynomialSystemSpec


class ConfigError(ValueError):
    """Raised when an experiment YAML file violates the Phase 6 schema."""


@dataclass(frozen=True)
class ExperimentSettings:
    id: str
    name: str
    seed: int
    device: str


@dataclass(frozen=True)
class SystemSettings:
    degrees: tuple[int, ...]
    free_exponents: tuple[tuple[tuple[int, ...], ...], ...]


@dataclass(frozen=True)
class SamplerSettings:
    type: str
    bound: float


@dataclass(frozen=True)
class PathSettings:
    bezier_degree: int
    latent_dim: int
    basis_seed: int
    action_limit: float


@dataclass(frozen=True)
class TrackerSettings:
    max_steps: int
    max_step_size: float
    max_initial_step_size: float
    min_step_size: float
    extended_precision: bool


@dataclass(frozen=True)
class EnvironmentSettings:
    reject_weight: float
    failure_penalty: float
    reward_scale: float
    warmup_backend: bool


@dataclass(frozen=True)
class PPOSettings:
    num_envs: int
    total_timesteps: int
    rollout_steps: int
    minibatch_size: int
    update_epochs: int
    learning_rate: float
    anneal_learning_rate: bool
    gamma: float
    gae_lambda: float
    clip_coef: float
    clip_value_loss: bool
    value_coef: float
    entropy_coef: float
    max_grad_norm: float
    normalize_advantage: bool
    target_kl: float | None
    hidden_sizes: tuple[int, ...]
    activation: str
    actor_logstd_init: float
    checkpoint_interval: int
    evaluation_interval_updates: int

    @property
    def num_updates(self) -> int:
        """Number of PPO updates implied by total timesteps and rollout length."""

        return self.total_timesteps // self.rollout_steps


@dataclass(frozen=True)
class EvaluationSettings:
    seed: int
    num_instances: int
    random_action_seed: int
    random_actions_per_instance: int
    deterministic_policy: bool


@dataclass(frozen=True)
class OutputSettings:
    root: str
    save_last_checkpoint: bool
    save_best_checkpoint: bool
    save_training_csv: bool


@dataclass(frozen=True)
class Phase6Config:
    experiment: ExperimentSettings
    system: SystemSettings
    sampler: SamplerSettings
    path: PathSettings
    tracker: TrackerSettings
    environment: EnvironmentSettings
    ppo: PPOSettings
    evaluation: EvaluationSettings
    output: OutputSettings


ROOT_KEYS = {
    "experiment",
    "system",
    "sampler",
    "path",
    "tracker",
    "environment",
    "ppo",
    "evaluation",
    "output",
}


def _mapping(value: Any, *, name: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a mapping.")
    actual = set(value)
    missing = keys - actual
    extra = actual - keys
    if missing:
        raise ConfigError(f"{name} is missing required keys: {sorted(missing)}.")
    if extra:
        raise ConfigError(f"{name} contains unknown keys: {sorted(extra)}.")
    return dict(value)


def _int(value: Any, *, name: str, positive: bool = False, nonnegative: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ConfigError(f"{name} must be an integer, got {value!r}.")
    normalized = int(value)
    if positive and normalized <= 0:
        raise ConfigError(f"{name} must be positive, got {normalized}.")
    if nonnegative and normalized < 0:
        raise ConfigError(f"{name} must be nonnegative, got {normalized}.")
    return normalized


def _float(
    value: Any,
    *,
    name: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ConfigError(f"{name} must be a finite float, got {value!r}.")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a finite float, got {value!r}.") from exc
    if not np.isfinite(normalized):
        raise ConfigError(f"{name} must be finite, got {value!r}.")
    if positive and normalized <= 0.0:
        raise ConfigError(f"{name} must be positive, got {normalized}.")
    if nonnegative and normalized < 0.0:
        raise ConfigError(f"{name} must be nonnegative, got {normalized}.")
    return normalized


def _bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be a bool, got {value!r}.")
    return value


def _str(value: Any, *, name: str, nonempty: bool = False) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{name} must be a string, got {value!r}.")
    if nonempty and value.strip() == "":
        raise ConfigError(f"{name} must not be empty.")
    return value


def _degrees(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ConfigError("system.degrees must be a list.")
    degrees = tuple(_int(item, name=f"system.degrees[{i}]", positive=True) for i, item in enumerate(value))
    if not degrees:
        raise ConfigError("system.degrees must not be empty.")
    return degrees


def _free_exponents(value: Any, *, n_vars: int) -> tuple[tuple[tuple[int, ...], ...], ...]:
    if not isinstance(value, (list, tuple)):
        raise ConfigError("system.free_exponents must be a list.")
    if len(value) != n_vars:
        raise ConfigError(
            f"system.free_exponents must contain {n_vars} equation blocks, got {len(value)}."
        )
    blocks: list[tuple[tuple[int, ...], ...]] = []
    for equation_index, block in enumerate(value):
        if not isinstance(block, (list, tuple)):
            raise ConfigError(f"system.free_exponents[{equation_index}] must be a list.")
        normalized_block: list[tuple[int, ...]] = []
        for exponent_index, exponent in enumerate(block):
            if not isinstance(exponent, (list, tuple)):
                raise ConfigError(
                    f"system.free_exponents[{equation_index}][{exponent_index}] must be a list."
                )
            if len(exponent) != n_vars:
                raise ConfigError(
                    "system.free_exponents"
                    f"[{equation_index}][{exponent_index}] must have length {n_vars}."
                )
            normalized_block.append(
                tuple(
                    _int(
                        component,
                        name=(
                            "system.free_exponents"
                            f"[{equation_index}][{exponent_index}][{component_index}]"
                        ),
                        nonnegative=True,
                    )
                    for component_index, component in enumerate(exponent)
                )
            )
        blocks.append(tuple(normalized_block))
    return tuple(blocks)


def parse_config_dict(data: dict[str, Any]) -> Phase6Config:
    """Parse and validate a Phase 6 YAML mapping."""

    root = _mapping(data, name="config", keys=ROOT_KEYS)
    exp_raw = _mapping(
        root["experiment"],
        name="experiment",
        keys={"id", "name", "seed", "device"},
    )
    system_raw = _mapping(root["system"], name="system", keys={"degrees", "free_exponents"})
    sampler_raw = _mapping(root["sampler"], name="sampler", keys={"type", "bound"})
    path_raw = _mapping(
        root["path"],
        name="path",
        keys={"bezier_degree", "latent_dim", "basis_seed", "action_limit"},
    )
    tracker_raw = _mapping(
        root["tracker"],
        name="tracker",
        keys={
            "max_steps",
            "max_step_size",
            "max_initial_step_size",
            "min_step_size",
            "extended_precision",
        },
    )
    env_raw = _mapping(
        root["environment"],
        name="environment",
        keys={"reject_weight", "failure_penalty", "reward_scale", "warmup_backend"},
    )
    ppo_raw = _mapping(
        root["ppo"],
        name="ppo",
        keys={
            "num_envs",
            "total_timesteps",
            "rollout_steps",
            "minibatch_size",
            "update_epochs",
            "learning_rate",
            "anneal_learning_rate",
            "gamma",
            "gae_lambda",
            "clip_coef",
            "clip_value_loss",
            "value_coef",
            "entropy_coef",
            "max_grad_norm",
            "normalize_advantage",
            "target_kl",
            "hidden_sizes",
            "activation",
            "actor_logstd_init",
            "checkpoint_interval",
            "evaluation_interval_updates",
        },
    )
    eval_raw = _mapping(
        root["evaluation"],
        name="evaluation",
        keys={
            "seed",
            "num_instances",
            "random_action_seed",
            "random_actions_per_instance",
            "deterministic_policy",
        },
    )
    output_raw = _mapping(
        root["output"],
        name="output",
        keys={"root", "save_last_checkpoint", "save_best_checkpoint", "save_training_csv"},
    )

    degrees = _degrees(system_raw["degrees"])
    system = SystemSettings(
        degrees=degrees,
        free_exponents=_free_exponents(system_raw["free_exponents"], n_vars=len(degrees)),
    )
    experiment = ExperimentSettings(
        id=_str(exp_raw["id"], name="experiment.id", nonempty=True),
        name=_str(exp_raw["name"], name="experiment.name", nonempty=True),
        seed=_int(exp_raw["seed"], name="experiment.seed"),
        device=_str(exp_raw["device"], name="experiment.device", nonempty=True),
    )
    sampler_type = _str(sampler_raw["type"], name="sampler.type", nonempty=True)
    if sampler_type != "complex_uniform":
        raise ConfigError("sampler.type must be 'complex_uniform'.")
    sampler = SamplerSettings(
        type=sampler_type,
        bound=_float(sampler_raw["bound"], name="sampler.bound", nonnegative=True),
    )
    path = PathSettings(
        bezier_degree=_int(path_raw["bezier_degree"], name="path.bezier_degree", positive=True),
        latent_dim=_int(path_raw["latent_dim"], name="path.latent_dim", positive=True),
        basis_seed=_int(path_raw["basis_seed"], name="path.basis_seed"),
        action_limit=_float(path_raw["action_limit"], name="path.action_limit", positive=True),
    )
    if path.bezier_degree < 2:
        raise ConfigError("path.bezier_degree must be at least 2.")
    tracker = TrackerSettings(
        max_steps=_int(tracker_raw["max_steps"], name="tracker.max_steps", positive=True),
        max_step_size=_float(tracker_raw["max_step_size"], name="tracker.max_step_size", positive=True),
        max_initial_step_size=_float(
            tracker_raw["max_initial_step_size"],
            name="tracker.max_initial_step_size",
            positive=True,
        ),
        min_step_size=_float(tracker_raw["min_step_size"], name="tracker.min_step_size", positive=True),
        extended_precision=_bool(tracker_raw["extended_precision"], name="tracker.extended_precision"),
    )
    environment = EnvironmentSettings(
        reject_weight=_float(env_raw["reject_weight"], name="environment.reject_weight", nonnegative=True),
        failure_penalty=_float(
            env_raw["failure_penalty"],
            name="environment.failure_penalty",
            positive=True,
        ),
        reward_scale=_float(env_raw["reward_scale"], name="environment.reward_scale", positive=True),
        warmup_backend=_bool(env_raw["warmup_backend"], name="environment.warmup_backend"),
    )
    target_kl_raw = ppo_raw["target_kl"]
    target_kl = None if target_kl_raw is None else _float(target_kl_raw, name="ppo.target_kl", positive=True)
    hidden_raw = ppo_raw["hidden_sizes"]
    if not isinstance(hidden_raw, list) or not hidden_raw:
        raise ConfigError("ppo.hidden_sizes must be a nonempty list.")
    hidden_sizes = tuple(_int(value, name=f"ppo.hidden_sizes[{i}]", positive=True) for i, value in enumerate(hidden_raw))
    activation = _str(ppo_raw["activation"], name="ppo.activation", nonempty=True)
    if activation not in {"tanh", "relu"}:
        raise ConfigError("ppo.activation must be 'tanh' or 'relu'.")
    ppo = PPOSettings(
        num_envs=_int(ppo_raw["num_envs"], name="ppo.num_envs", positive=True),
        total_timesteps=_int(ppo_raw["total_timesteps"], name="ppo.total_timesteps", positive=True),
        rollout_steps=_int(ppo_raw["rollout_steps"], name="ppo.rollout_steps", positive=True),
        minibatch_size=_int(ppo_raw["minibatch_size"], name="ppo.minibatch_size", positive=True),
        update_epochs=_int(ppo_raw["update_epochs"], name="ppo.update_epochs", positive=True),
        learning_rate=_float(ppo_raw["learning_rate"], name="ppo.learning_rate", positive=True),
        anneal_learning_rate=_bool(ppo_raw["anneal_learning_rate"], name="ppo.anneal_learning_rate"),
        gamma=_float(ppo_raw["gamma"], name="ppo.gamma", nonnegative=True),
        gae_lambda=_float(ppo_raw["gae_lambda"], name="ppo.gae_lambda", nonnegative=True),
        clip_coef=_float(ppo_raw["clip_coef"], name="ppo.clip_coef", positive=True),
        clip_value_loss=_bool(ppo_raw["clip_value_loss"], name="ppo.clip_value_loss"),
        value_coef=_float(ppo_raw["value_coef"], name="ppo.value_coef", nonnegative=True),
        entropy_coef=_float(ppo_raw["entropy_coef"], name="ppo.entropy_coef", nonnegative=True),
        max_grad_norm=_float(ppo_raw["max_grad_norm"], name="ppo.max_grad_norm", positive=True),
        normalize_advantage=_bool(ppo_raw["normalize_advantage"], name="ppo.normalize_advantage"),
        target_kl=target_kl,
        hidden_sizes=hidden_sizes,
        activation=activation,
        actor_logstd_init=_float(ppo_raw["actor_logstd_init"], name="ppo.actor_logstd_init"),
        checkpoint_interval=_int(ppo_raw["checkpoint_interval"], name="ppo.checkpoint_interval", positive=True),
        evaluation_interval_updates=_int(
            ppo_raw["evaluation_interval_updates"],
            name="ppo.evaluation_interval_updates",
            positive=True,
        ),
    )
    evaluation = EvaluationSettings(
        seed=_int(eval_raw["seed"], name="evaluation.seed"),
        num_instances=_int(eval_raw["num_instances"], name="evaluation.num_instances", positive=True),
        random_action_seed=_int(eval_raw["random_action_seed"], name="evaluation.random_action_seed"),
        random_actions_per_instance=_int(
            eval_raw["random_actions_per_instance"],
            name="evaluation.random_actions_per_instance",
            positive=True,
        ),
        deterministic_policy=_bool(eval_raw["deterministic_policy"], name="evaluation.deterministic_policy"),
    )
    output_root = _str(output_raw["root"], name="output.root", nonempty=True)
    output = OutputSettings(
        root=output_root,
        save_last_checkpoint=_bool(output_raw["save_last_checkpoint"], name="output.save_last_checkpoint"),
        save_best_checkpoint=_bool(output_raw["save_best_checkpoint"], name="output.save_best_checkpoint"),
        save_training_csv=_bool(output_raw["save_training_csv"], name="output.save_training_csv"),
    )

    config = Phase6Config(
        experiment=experiment,
        system=system,
        sampler=sampler,
        path=path,
        tracker=tracker,
        environment=environment,
        ppo=ppo,
        evaluation=evaluation,
        output=output,
    )
    validate_config(config)
    return config


def validate_config(config: Phase6Config) -> None:
    """Validate cross-field constraints for a parsed Phase 6 config."""

    if config.ppo.num_envs != 1:
        raise ConfigError("ppo.num_envs must be 1 in the initial Phase 6 implementation.")
    if config.ppo.total_timesteps < config.ppo.rollout_steps:
        raise ConfigError("ppo.total_timesteps must be >= ppo.rollout_steps.")
    if config.ppo.total_timesteps % config.ppo.rollout_steps != 0:
        raise ConfigError("ppo.total_timesteps must be divisible by ppo.rollout_steps.")
    if config.ppo.minibatch_size > config.ppo.rollout_steps:
        raise ConfigError("ppo.minibatch_size must be <= ppo.rollout_steps.")
    if config.ppo.rollout_steps % config.ppo.minibatch_size != 0:
        raise ConfigError("ppo.rollout_steps must be divisible by ppo.minibatch_size.")
    if config.tracker.min_step_size > config.tracker.max_step_size:
        raise ConfigError("tracker.min_step_size must be <= tracker.max_step_size.")
    if config.tracker.min_step_size > config.tracker.max_initial_step_size:
        raise ConfigError("tracker.min_step_size must be <= tracker.max_initial_step_size.")
    build_system_spec(config)


def load_config(path: str | Path) -> Phase6Config:
    """Load a strict Phase 6 YAML config from ``path``."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        raise ConfigError("config file must not be empty.")
    if not isinstance(data, dict):
        raise ConfigError("config root must be a mapping.")
    return parse_config_dict(data)


def config_to_dict(config: Phase6Config) -> dict[str, Any]:
    """Return a plain serializable dictionary for a config dataclass."""

    def normalize(value: Any) -> Any:
        if isinstance(value, tuple):
            return [normalize(item) for item in value]
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        return value

    return normalize(asdict(config))


def save_config(config: Phase6Config, path: str | Path) -> None:
    """Save ``config`` as YAML with deterministic key ordering preserved."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config_to_dict(config), handle, sort_keys=False)


def with_overrides(
    config: Phase6Config,
    *,
    total_timesteps: int | None = None,
    output_root: str | None = None,
    device: str | None = None,
) -> Phase6Config:
    """Return a validated config with the supported smoke CLI overrides."""

    updated = config
    if total_timesteps is not None:
        updated = replace(
            updated,
            ppo=replace(
                updated.ppo,
                total_timesteps=_int(total_timesteps, name="total_timesteps", positive=True),
            ),
        )
    if output_root is not None:
        if output_root.strip() == "":
            raise ConfigError("output_root override must be nonempty.")
        updated = replace(updated, output=replace(updated.output, root=output_root))
    if device is not None:
        updated = replace(
            updated,
            experiment=replace(
                updated.experiment,
                device=_str(device, name="device", nonempty=True),
            ),
        )
    validate_config(updated)
    return updated


def build_system_spec(config: Phase6Config) -> PolynomialSystemSpec:
    """Build the Pham polynomial system spec using the existing systems API."""

    return build_pham_spec(config.system.degrees, config.system.free_exponents)


def build_sampler(config: Phase6Config) -> ComplexUniformSampler:
    """Build the target coefficient sampler from config."""

    if config.sampler.type != "complex_uniform":
        raise ConfigError("only complex_uniform sampler is supported.")
    return ComplexUniformSampler(bound=config.sampler.bound)


def build_tracker_config(config: Phase6Config) -> TrackerConfig:
    """Build the backend tracker config from YAML settings."""

    return TrackerConfig(
        max_steps=config.tracker.max_steps,
        max_step_size=config.tracker.max_step_size,
        max_initial_step_size=config.tracker.max_initial_step_size,
        min_step_size=config.tracker.min_step_size,
        extended_precision=config.tracker.extended_precision,
    )


def build_env(config: Phase6Config, *, backend: object | None = None) -> BezierPhamEnv:
    """Build a ``BezierPhamEnv`` from config.

    Passing ``backend`` is intended for tests. If it is ``None``, the environment
    owns a Julia backend according to the Phase 5 environment rules.
    """

    return BezierPhamEnv(
        system_spec=build_system_spec(config),
        bezier_degree=config.path.bezier_degree,
        latent_dim=config.path.latent_dim,
        coefficient_sampler=build_sampler(config),
        tracker_config=build_tracker_config(config),
        reject_weight=config.environment.reject_weight,
        failure_penalty=config.environment.failure_penalty,
        reward_scale=config.environment.reward_scale,
        action_limit=config.path.action_limit,
        basis_seed=config.path.basis_seed,
        backend=backend,
        warmup_backend=config.environment.warmup_backend,
    )


__all__ = [
    "ConfigError",
    "EnvironmentSettings",
    "EvaluationSettings",
    "ExperimentSettings",
    "OutputSettings",
    "PPOSettings",
    "PathSettings",
    "Phase6Config",
    "SamplerSettings",
    "SystemSettings",
    "TrackerSettings",
    "build_env",
    "build_sampler",
    "build_system_spec",
    "build_tracker_config",
    "config_to_dict",
    "load_config",
    "parse_config_dict",
    "save_config",
    "validate_config",
    "with_overrides",
]
