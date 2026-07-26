from __future__ import annotations

import numpy as np
import pytest

from homotopy_path_learning.backends.types import TrackerConfig, TrackingResult


def test_tracker_config_defaults_are_valid() -> None:
    config = TrackerConfig()

    assert config.max_steps == 50_000
    assert config.max_step_size == 0.05
    assert config.max_initial_step_size == 0.05
    assert config.min_step_size == 1e-12
    assert config.extended_precision is False


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"max_steps": 0}, ValueError, "max_steps"),
        ({"max_steps": True}, TypeError, "max_steps"),
        ({"max_step_size": 0.0}, ValueError, "max_step_size"),
        ({"max_initial_step_size": float("inf")}, ValueError, "max_initial_step_size"),
        ({"min_step_size": float("nan")}, ValueError, "min_step_size"),
        ({"min_step_size": 0.1, "max_step_size": 0.05}, ValueError, "min_step_size"),
        (
            {"min_step_size": 0.1, "max_initial_step_size": 0.05},
            ValueError,
            "min_step_size",
        ),
        ({"extended_precision": np.bool_(False)}, TypeError, "extended_precision"),
    ],
)
def test_tracker_config_rejects_invalid_values(
    kwargs: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        TrackerConfig(**kwargs)


def test_tracking_result_copies_and_validates_arrays() -> None:
    accepted = np.array([2, 1], dtype=np.int64)
    rejected = np.array([0, 0], dtype=np.int64)
    success = np.array([True, True], dtype=np.bool_)
    endpoints = np.array([[1.0 + 0.0j], [-1.0 + 0.0j]], dtype=np.complex128)
    residuals = np.array([1e-12, 2e-12], dtype=np.float64)

    result = TrackingResult(
        success=True,
        n_paths=2,
        n_success=2,
        n_failed=0,
        accepted_steps=3,
        rejected_steps=0,
        per_path_accepted_steps=accepted,
        per_path_rejected_steps=rejected,
        path_success=success,
        endpoints=endpoints,
        residual_norms=residuals,
        failure_codes=("", ""),
    )
    accepted[0] = 99
    endpoints[0, 0] = 99.0 + 0.0j

    assert result.per_path_accepted_steps.dtype == np.int64
    assert result.endpoints.dtype == np.complex128
    np.testing.assert_array_equal(result.per_path_accepted_steps, np.array([2, 1]))
    np.testing.assert_array_equal(
        result.endpoints,
        np.array([[1.0 + 0.0j], [-1.0 + 0.0j]], dtype=np.complex128),
    )


def test_tracking_result_rejects_inconsistent_counts() -> None:
    with pytest.raises(ValueError, match="n_success"):
        TrackingResult(
            success=True,
            n_paths=2,
            n_success=1,
            n_failed=0,
            accepted_steps=0,
            rejected_steps=0,
            per_path_accepted_steps=np.zeros(2, dtype=np.int64),
            per_path_rejected_steps=np.zeros(2, dtype=np.int64),
            path_success=np.ones(2, dtype=np.bool_),
            endpoints=np.zeros((2, 1), dtype=np.complex128),
            residual_norms=np.zeros(2, dtype=np.float64),
            failure_codes=("", ""),
        )
