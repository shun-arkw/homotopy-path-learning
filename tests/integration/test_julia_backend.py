from __future__ import annotations

import os

import numpy as np
import pytest

from homotopy_path_learning.backends.julia import (
    BackendClosedError,
    BackendStateError,
    JuliaBackendValidationError,
    JuliaTrackerBackend,
)
from homotopy_path_learning.backends.types import TrackerConfig, TrackingResult
from homotopy_path_learning.paths import (
    BezierParameterization,
    linear_interpolation_control_points,
)
from homotopy_path_learning.systems import build_pham_spec, start_system_coefficients


pytestmark = pytest.mark.integration


SUPPORT_2X2 = (
    ((1, 0), (0, 1), (0, 0)),
    ((1, 0), (0, 1), (0, 0)),
)


def make_spec():
    return build_pham_spec((2, 2), SUPPORT_2X2)


def diagonal_target_coefficients() -> np.ndarray:
    return np.array([1, 0, 0, -2, 1, 0, 0, -2], dtype=np.complex128)


def linear_control_points() -> np.ndarray:
    spec = make_spec()
    return linear_interpolation_control_points(
        start_system_coefficients(spec),
        diagonal_target_coefficients(),
        3,
    )


def assert_same_tracking_result(left: TrackingResult, right: TrackingResult) -> None:
    assert left.success == right.success
    assert left.n_paths == right.n_paths
    assert left.n_success == right.n_success
    assert left.n_failed == right.n_failed
    assert left.accepted_steps == right.accepted_steps
    assert left.rejected_steps == right.rejected_steps
    np.testing.assert_array_equal(left.path_success, right.path_success)
    np.testing.assert_array_equal(left.per_path_accepted_steps, right.per_path_accepted_steps)
    np.testing.assert_array_equal(left.per_path_rejected_steps, right.per_path_rejected_steps)
    np.testing.assert_allclose(left.endpoints, right.endpoints, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        left.residual_norms,
        right.residual_norms,
        atol=1e-12,
        rtol=1e-12,
    )
    assert left.failure_codes == right.failure_codes


def test_environment_variables_are_explicit() -> None:
    assert os.environ["PYTHON_JULIACALL_EXE"].endswith("/julia")
    assert os.environ["PYTHON_JULIACALL_PROJECT"].endswith("/julia")


def test_julia_backend_initializes_tracks_and_exposes_state_snapshot() -> None:
    spec = make_spec()
    control_points = linear_control_points()

    with JuliaTrackerBackend() as backend:
        backend.initialize(spec, 3, TrackerConfig())
        snapshot = backend.state_snapshot()

        assert snapshot["initialized"] is True
        assert snapshot["nvars"] == 2
        assert snapshot["bezier_degree"] == 3
        np.testing.assert_array_equal(snapshot["degrees"], np.array([2, 2], dtype=np.int64))
        np.testing.assert_array_equal(snapshot["offsets"], spec.offsets + 1)
        np.testing.assert_array_equal(snapshot["leading_indices"], spec.leading_indices + 1)
        np.testing.assert_array_equal(snapshot["constant_indices"], spec.constant_indices + 1)
        np.testing.assert_array_equal(snapshot["exponents"], spec.exponents)
        assert snapshot["starts"].shape == (4, 2)

        result = backend.track(control_points)
        assert isinstance(result, TrackingResult)
        assert result.per_path_accepted_steps.shape == (4,)
        assert result.per_path_rejected_steps.shape == (4,)
        assert result.path_success.shape == (4,)
        assert result.endpoints.shape == (4, 2)
        assert result.endpoints.dtype == np.complex128
        assert result.residual_norms.shape == (4,)
        assert result.residual_norms.dtype == np.float64
        assert len(result.failure_codes) == 4
        assert result.success
        assert result.n_paths == 4
        assert result.n_success == 4
        assert result.n_failed == 0
        assert np.max(result.residual_norms) < 1e-8

        snapshot = backend.state_snapshot()
        np.testing.assert_array_equal(snapshot["control_points"], control_points)


def test_repeated_tracking_is_reproducible() -> None:
    spec = make_spec()
    control_points = linear_control_points()

    with JuliaTrackerBackend() as backend:
        backend.initialize(spec, 3, TrackerConfig())
        first = backend.track(control_points)
        second = backend.track(control_points)

    assert_same_tracking_result(first, second)


def test_warmup_does_not_change_tracking_result() -> None:
    spec = make_spec()
    control_points = linear_control_points()

    with JuliaTrackerBackend() as backend:
        backend.initialize(spec, 3, TrackerConfig())
        before = backend.track(control_points)
        warmup_result = backend.warmup()
        after = backend.track(control_points)

    assert warmup_result.n_paths == 4
    assert_same_tracking_result(before, after)


def test_zero_latent_action_matches_linear_control_points_tracking() -> None:
    spec = make_spec()
    start = start_system_coefficients(spec)
    target = diagonal_target_coefficients()
    linear_points = linear_interpolation_control_points(start, target, 3)
    parameterization = BezierParameterization.from_spec(
        spec,
        bezier_degree=3,
        latent_dim=4,
        rng=np.random.default_rng(0),
    )
    zero_points = parameterization.build_control_points(
        start,
        target,
        np.zeros(parameterization.action_dim, dtype=np.float64),
    )
    np.testing.assert_array_equal(zero_points, linear_points)

    with JuliaTrackerBackend() as backend:
        backend.initialize(spec, 3, TrackerConfig())
        zero_result = backend.track(zero_points)
        linear_result = backend.track(linear_points)

    assert_same_tracking_result(zero_result, linear_result)


def test_lifecycle_errors_and_close_clears_julia_state() -> None:
    spec = make_spec()
    control_points = linear_control_points()
    backend = JuliaTrackerBackend()

    with pytest.raises(BackendStateError, match="not initialized"):
        backend.track(control_points)
    with pytest.raises(BackendStateError, match="not initialized"):
        backend.warmup()

    backend.initialize(spec, 3, TrackerConfig())
    assert backend.julia_state_initialized()
    backend.close()
    assert not backend.julia_state_initialized()
    backend.close()

    with pytest.raises(BackendClosedError, match="closed"):
        backend.track(control_points)
    with pytest.raises(BackendClosedError, match="closed"):
        backend.warmup()


def test_invalid_control_points_are_rejected_before_julia_call() -> None:
    spec = make_spec()
    control_points = linear_control_points()

    with JuliaTrackerBackend() as backend:
        backend.initialize(spec, 3, TrackerConfig())
        before = backend.state_snapshot()["control_points"]
        bad = control_points.copy()
        bad[1, spec.leading_indices[0]] = 1.0 + 0.5j

        with pytest.raises(JuliaBackendValidationError, match="leading"):
            backend.track(bad)

        after = backend.state_snapshot()["control_points"]
        np.testing.assert_array_equal(after, before)


def test_reinitialization_clears_previous_state_safely() -> None:
    spec = make_spec()
    backend = JuliaTrackerBackend()
    try:
        backend.initialize(spec, 3, TrackerConfig())
        backend.initialize(spec, 3, TrackerConfig(max_steps=60_000))
        assert backend.julia_state_initialized()
        result = backend.track(linear_control_points())
        assert result.success
    finally:
        backend.close()
