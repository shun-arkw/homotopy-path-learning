from __future__ import annotations

import pytest
import yaml

from homotopy_path_learning.performance.reporting import (
    load_performance_config,
    parse_performance_config_dict,
)


def _config_dict():
    with open("experiments/multivariate_pham/configs/exp-0008-phase7-performance.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_performance_config_loads_existing_profile_yaml() -> None:
    config = load_performance_config("experiments/multivariate_pham/configs/exp-0008-phase7-performance.yaml")

    assert config.phase6.experiment.id == "EXP-0008"
    assert config.performance.warmup_runs == 5
    assert config.performance.fixed_problem_count == 100


def test_performance_config_rejects_unknown_keys() -> None:
    data = _config_dict()
    data["performance"]["unknown"] = 1

    with pytest.raises(ValueError, match="unknown"):
        parse_performance_config_dict(data)


def test_existing_phase6_config_shape_is_preserved() -> None:
    config = load_performance_config("experiments/multivariate_pham/configs/exp-0008-phase7-performance.yaml")
    phase6 = config.phase6

    assert phase6.path.bezier_degree == 3
    assert phase6.path.latent_dim == 4
    assert phase6.ppo.num_envs == 1
