from __future__ import annotations

from dataclasses import replace

import pytest

from homotopy_path_learning.config import (
    ConfigError,
    config_to_dict,
    load_config,
    parse_config_dict,
    save_config,
    with_overrides,
)


SMOKE_CONFIG = "experiments/multivariate_pham/configs/exp-0004-smoke.yaml"


def test_load_smoke_yaml_and_build_system_spec() -> None:
    config = load_config(SMOKE_CONFIG)

    assert config.experiment.id == "EXP-0004"
    assert config.system.degrees == (2, 2)
    assert config.path.bezier_degree == 3
    assert config.ppo.total_timesteps == 256
    assert config.ppo.num_updates == 4


def test_unknown_and_missing_keys_are_rejected() -> None:
    data = config_to_dict(load_config(SMOKE_CONFIG))
    data["ppo"]["extra"] = 1
    with pytest.raises(ConfigError, match="unknown"):
        parse_config_dict(data)

    data = config_to_dict(load_config(SMOKE_CONFIG))
    del data["system"]["degrees"]
    with pytest.raises(ConfigError, match="missing"):
        parse_config_dict(data)


def test_invalid_types_and_nonfinite_values_are_rejected() -> None:
    data = config_to_dict(load_config(SMOKE_CONFIG))
    data["ppo"]["rollout_steps"] = True
    with pytest.raises(ConfigError, match="integer"):
        parse_config_dict(data)

    data = config_to_dict(load_config(SMOKE_CONFIG))
    data["ppo"]["learning_rate"] = float("nan")
    with pytest.raises(ConfigError, match="finite"):
        parse_config_dict(data)


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"num_envs": 2}, "num_envs"),
        ({"total_timesteps": 63}, "total_timesteps"),
        ({"minibatch_size": 128}, "minibatch_size"),
        ({"minibatch_size": 48}, "divisible"),
    ],
)
def test_invalid_ppo_batch_settings_are_rejected(updates, match) -> None:
    data = config_to_dict(load_config(SMOKE_CONFIG))
    data["ppo"].update(updates)

    with pytest.raises(ConfigError, match=match):
        parse_config_dict(data)


def test_save_and_reload_config_round_trip(tmp_path) -> None:
    config = load_config(SMOKE_CONFIG)
    path = tmp_path / "config.yaml"

    save_config(config, path)
    loaded = load_config(path)

    assert loaded == config


def test_supported_overrides_are_validated() -> None:
    config = load_config(SMOKE_CONFIG)

    updated = with_overrides(config, total_timesteps=128, output_root="other", device="cpu")

    assert updated.ppo.total_timesteps == 128
    assert updated.output.root == "other"
    assert updated.experiment.device == "cpu"

    with pytest.raises(ConfigError, match="divisible"):
        with_overrides(config, total_timesteps=65)
