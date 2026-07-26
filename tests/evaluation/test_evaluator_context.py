from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

from homotopy_path_learning.backends.types import TrackerConfig, TrackingResult
from homotopy_path_learning.config import load_config
from homotopy_path_learning.envs import BezierPhamEnv
from homotopy_path_learning.evaluation import evaluator
from homotopy_path_learning.systems import build_pham_spec


def _tracking_result(
    *,
    accepted: Iterable[int] = (4, 5, 6, 7),
    rejected: Iterable[int] = (0, 0, 0, 0),
) -> TrackingResult:
    accepted_array = np.array(tuple(accepted), dtype=np.int64)
    rejected_array = np.array(tuple(rejected), dtype=np.int64)
    success = np.ones(accepted_array.shape, dtype=np.bool_)
    return TrackingResult(
        success=True,
        n_paths=accepted_array.size,
        n_success=accepted_array.size,
        n_failed=0,
        accepted_steps=int(np.sum(accepted_array)),
        rejected_steps=int(np.sum(rejected_array)),
        per_path_accepted_steps=accepted_array,
        per_path_rejected_steps=rejected_array,
        path_success=success,
        endpoints=np.ones((accepted_array.size, 2), dtype=np.complex128),
        residual_norms=np.full(accepted_array.size, 1e-12, dtype=np.float64),
        failure_codes=tuple("" for _ in success),
    )


class CountingBackend:
    def __init__(self) -> None:
        self.track_calls = 0
        self.initialize_calls = 0

    def initialize(self, system_spec, bezier_degree: int, tracker_config: TrackerConfig) -> None:
        del system_spec, bezier_degree, tracker_config
        self.initialize_calls += 1

    def warmup(self) -> TrackingResult:
        return _tracking_result()

    def track(self, control_points: np.ndarray) -> TrackingResult:
        del control_points
        self.track_calls += 1
        return _tracking_result()

    def close(self) -> None:
        return None


class ZeroAgent:
    def deterministic_action(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.zeros((observation.shape[0], 8), dtype=torch.float32, device=observation.device)


def test_evaluate_policy_rows_reuses_linear_context_once_per_target(monkeypatch) -> None:
    config = load_config("experiments/multivariate_pham/configs/exp-0004-smoke.yaml")
    backend = CountingBackend()
    spec = build_pham_spec(
        (2, 2),
        (
            ((1, 0), (0, 1), (0, 0)),
            ((1, 0), (0, 1), (0, 0)),
        ),
    )
    env = BezierPhamEnv(system_spec=spec, backend=backend)
    monkeypatch.setattr(evaluator, "build_env", lambda _config: env)

    rows = evaluator.evaluate_policy_rows(
        config=config,
        run_id="run",
        agent=ZeroAgent(),
        device="cpu",
        methods=("Linear", "RandomBezier", "LearnedBezier"),
        num_instances=2,
        evaluation_seed=100,
        random_action_seed=200,
    )

    assert len(rows) == 6
    assert backend.track_calls == 6
    for target_seed in (100, 101):
        seed_rows = [row for row in rows if row["target_seed"] == target_seed]
        linear_costs = {float(row["cost_improvement_vs_linear"]) for row in seed_rows if row["method"] == "Linear"}
        assert linear_costs == {0.0}
