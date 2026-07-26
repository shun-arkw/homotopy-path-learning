from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest
import yaml


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.phase7
def test_phase7_profile_cli_generates_outputs(tmp_path) -> None:
    source_config = Path("experiments/multivariate_pham/configs/exp-0008-phase7-performance.yaml")
    data = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    data["performance"]["warmup_runs"] = 0
    data["performance"]["measure_runs"] = 1
    data["performance"]["trials"] = 1
    data["performance"]["fixed_problem_count"] = 2
    config_path = tmp_path / "phase7-profile.yaml"
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    env = os.environ.copy()
    env.setdefault("PYTHON_JULIACALL_EXE", "/usr/local/bin/julia")
    env.setdefault("PYTHON_JULIACALL_PROJECT", str(Path.cwd() / "julia"))
    env.setdefault("JULIAPKG_OFF", "1")
    env.setdefault("JULIA_CONDAPKG_OFF", "1")
    env.setdefault("JULIA_PYTHONCALL_INSTALL", "never")

    baseline_dir = tmp_path / "baseline"
    optimized_dir = tmp_path / "optimized"
    subprocess.run(
        [
            "python3",
            "experiments/multivariate_pham/profile.py",
            "--config",
            str(config_path),
            "--output-dir",
            str(baseline_dir),
            "--mode",
            "baseline",
        ],
        check=True,
        env=env,
    )
    subprocess.run(
        [
            "python3",
            "experiments/multivariate_pham/profile.py",
            "--config",
            str(config_path),
            "--output-dir",
            str(optimized_dir),
            "--mode",
            "optimized",
            "--baseline",
            str(baseline_dir / "performance.json"),
        ],
        check=True,
        env=env,
    )

    for directory in (baseline_dir, optimized_dir):
        assert (directory / "config.yaml").is_file()
        assert (directory / "performance.json").is_file()
        assert (directory / "performance.csv").is_file()
        assert (directory / "equivalence.json").is_file()
        assert (directory / "profiling" / "python.txt").is_file()
        assert (directory / "profiling" / "julia.txt").is_file()
        assert (directory / "summary.md").is_file()

    payload = json.loads((optimized_dir / "performance.json").read_text(encoding="utf-8"))
    assert payload["mode"] == "optimized"
    assert payload["equivalence"]["tracking"]["differing_items"] == 0
    assert payload["equivalence"]["evaluation_rows"]["differing_items"] == 0
    assert payload["performance_comparison"]
