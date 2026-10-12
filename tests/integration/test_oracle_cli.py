from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


pytestmark = [pytest.mark.integration, pytest.mark.slow]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_oracle_cli_smoke(tmp_path) -> None:
    env = os.environ.copy()
    env["PYTHON_JULIACALL_EXE"] = env.get("PYTHON_JULIACALL_EXE", "/usr/local/bin/julia")
    env["PYTHON_JULIACALL_PROJECT"] = env.get("PYTHON_JULIACALL_PROJECT", str(Path.cwd() / "julia"))
    output_root = tmp_path / "outputs"
    run_dir = output_root / "EXP-TEST" / "oracle-smoke"

    subprocess.run(
        [
            sys.executable,
            "experiments/multivariate_pham/oracle.py",
            "--config",
            "experiments/multivariate_pham/configs/exp-0004-smoke.yaml",
            "--experiment-id",
            "EXP-TEST",
            "--run-id",
            "oracle-smoke",
            "--latent-dim",
            "full",
            "--budget",
            "6",
            "--n-random",
            "3",
            "--num-instances",
            "2",
            "--output-root",
            str(output_root),
            "--allow-dirty",
        ],
        check=True,
        env=env,
    )

    rows = read_csv_rows(run_dir / "oracle.csv")
    assert len(rows) == 2
    for row in rows:
        assert int(row["latent_dim"]) == 12
        assert int(row["action_dim"]) == 24
        assert int(row["n_evaluations"]) == 6
        # The zero action is the linear path, so it must reproduce the linear cost.
        assert float(row["zero_action_cost"]) == float(row["linear_cost"])
        assert float(row["oracle_cost"]) <= float(row["linear_cost"])
    summary = read_csv_rows(run_dir / "oracle_summary.csv")
    assert [row["method"] for row in summary] == ["Linear", "Oracle"]
    assert np.load(run_dir / "oracle_actions.npy").shape == (2, 24)
    assert np.load(run_dir / "oracle_cost_traces.npy").shape == (2, 6)
    assert np.load(run_dir / "latent_basis.npy").shape == (12, 12)
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["experiment_id"] == "EXP-TEST"
    assert metadata["oracle"]["latent_dim"] == 12
    assert "oracle.csv" in metadata["output_files"]
