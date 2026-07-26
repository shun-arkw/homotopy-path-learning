from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pytest

from homotopy_path_learning.backends.types import TrackerConfig, TrackingResult
from homotopy_path_learning.systems import build_pham_spec


SUPPORT_2X2 = (
    ((1, 0), (0, 1), (0, 0)),
    ((1, 0), (0, 1), (0, 0)),
)


def make_smoke_spec():
    return build_pham_spec((2, 2), SUPPORT_2X2)


def make_tracking_result(
    *,
    accepted: Iterable[int] = (4, 5, 6, 7),
    rejected: Iterable[int] = (0, 1, 0, 1),
    success: Iterable[bool] | None = None,
    n_vars: int = 2,
) -> TrackingResult:
    accepted_array = np.array(tuple(accepted), dtype=np.int64)
    rejected_array = np.array(tuple(rejected), dtype=np.int64)
    if success is None:
        success_array = np.ones(accepted_array.shape, dtype=np.bool_)
    else:
        success_array = np.array(tuple(success), dtype=np.bool_)
    n_paths = int(success_array.size)
    return TrackingResult(
        success=bool(np.all(success_array)),
        n_paths=n_paths,
        n_success=int(np.count_nonzero(success_array)),
        n_failed=int(n_paths - np.count_nonzero(success_array)),
        accepted_steps=int(np.sum(accepted_array)),
        rejected_steps=int(np.sum(rejected_array)),
        per_path_accepted_steps=accepted_array,
        per_path_rejected_steps=rejected_array,
        path_success=success_array,
        endpoints=np.ones((n_paths, n_vars), dtype=np.complex128),
        residual_norms=np.full(n_paths, 1e-12, dtype=np.float64),
        failure_codes=tuple("" if flag else "failed" for flag in success_array),
    )


class FixedSampler:
    def __init__(self, coefficients: np.ndarray) -> None:
        self.coefficients = np.array(coefficients, dtype=np.complex128, copy=True)

    def sample(self, rng: np.random.Generator, spec) -> np.ndarray:
        del rng, spec
        return self.coefficients.copy()


class FakeBackend:
    def __init__(
        self,
        results: Iterable[TrackingResult] | None = None,
        *,
        raise_on_track_call: int | None = None,
    ) -> None:
        self.results = list(results or [])
        self.raise_on_track_call = raise_on_track_call
        self.initialize_calls = 0
        self.warmup_calls = 0
        self.track_calls = 0
        self.close_calls = 0
        self.closed = False
        self.system_spec = None
        self.bezier_degree = None
        self.tracker_config = None
        self.tracked_control_points: list[np.ndarray] = []

    def initialize(self, system_spec, bezier_degree: int, tracker_config: TrackerConfig) -> None:
        self.initialize_calls += 1
        self.system_spec = system_spec
        self.bezier_degree = bezier_degree
        self.tracker_config = tracker_config

    def warmup(self) -> TrackingResult:
        self.warmup_calls += 1
        return make_tracking_result()

    def track(self, control_points: np.ndarray) -> TrackingResult:
        self.track_calls += 1
        self.tracked_control_points.append(np.array(control_points, dtype=np.complex128, copy=True))
        if self.raise_on_track_call == self.track_calls:
            raise RuntimeError("backend failure")
        if self.results:
            return self.results.pop(0)
        return make_tracking_result()

    def close(self) -> None:
        self.close_calls += 1
        self.closed = True


@pytest.fixture
def smoke_spec():
    return make_smoke_spec()


@pytest.fixture
def target_coefficients() -> np.ndarray:
    return np.array(
        [
            1.0 + 0.0j,
            0.25 + 0.5j,
            -0.5 + 0.75j,
            0.1 - 0.2j,
            1.0 + 0.0j,
            -0.3 + 0.4j,
            0.6 - 0.1j,
            -0.7 + 0.2j,
        ],
        dtype=np.complex128,
    )
