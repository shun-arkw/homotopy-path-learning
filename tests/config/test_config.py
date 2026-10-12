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


def test_optional_actor_keys_default_and_validate() -> None:
    config = load_config(SMOKE_CONFIG)
    assert config.ppo.actor_type == "mlp"
    assert config.ppo.attention_kwargs == {"d_model": 128, "n_heads": 4, "n_layers": 4, "ff_dim": 256}
    data = config_to_dict(config)
    assert data["ppo"]["actor_type"] == "mlp"
    for key in ("actor_type", "attention_d_model", "attention_heads", "attention_layers", "attention_ff_dim"):
        del data["ppo"][key]
    assert parse_config_dict(data) == config  # old YAML without the keys parses to the defaults

    data["ppo"]["actor_type"] = "attention"
    data["ppo"]["attention_d_model"] = 64
    assert parse_config_dict(data).ppo.attention_kwargs["d_model"] == 64
    data["ppo"]["actor_type"] = "transformer"
    with pytest.raises(ConfigError, match="actor_type"):
        parse_config_dict(data)
    data["ppo"]["actor_type"] = "attention"
    data["ppo"]["attention_heads"] = 5
    with pytest.raises(ConfigError, match="divisible"):
        parse_config_dict(data)


def test_all_existing_yaml_configs_still_parse() -> None:
    from pathlib import Path

    import yaml

    paths = sorted(Path("experiments/multivariate_pham/configs").glob("*.yaml"))
    assert paths
    parsed = 0
    for path in paths:
        if "performance" in yaml.safe_load(path.read_text(encoding="utf-8")):
            continue  # Phase 7 performance config has its own loader
        assert load_config(path).ppo.actor_type == "mlp"
        parsed += 1
    assert parsed > 30
