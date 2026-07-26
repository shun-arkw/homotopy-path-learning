from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from homotopy_path_learning.backends.julia import (
    JuliaResultConversionError,
    tracking_result_from_julia,
)


@dataclass
class PseudoJuliaResult:
    success: bool
    n_paths: int
    n_success: int
    n_failed: int
    accepted_steps: int
    rejected_steps: int
    per_path_accepted_steps: np.ndarray
    per_path_rejected_steps: np.ndarray
    path_success: np.ndarray
    endpoints: np.ndarray
    residual_norms: np.ndarray
    failure_codes: tuple[str, ...]


def valid_pseudo_result() -> PseudoJuliaResult:
    return PseudoJuliaResult(
        success=True,
        n_paths=4,
        n_success=4,
        n_failed=0,
        accepted_steps=10,
        rejected_steps=1,
        per_path_accepted_steps=np.array([2, 3, 2, 3], dtype=np.int64),
        per_path_rejected_steps=np.array([0, 0, 1, 0], dtype=np.int64),
        path_success=np.array([True, True, True, True], dtype=np.bool_),
        endpoints=np.ones((4, 2), dtype=np.complex128),
        residual_norms=np.array([1e-12, 2e-12, 3e-12, 4e-12], dtype=np.float64),
        failure_codes=("", "", "", ""),
    )


def test_tracking_result_from_julia_converts_valid_payload() -> None:
    result = tracking_result_from_julia(valid_pseudo_result(), n_vars=2)

    assert result.success is True
    assert result.n_paths == 4
    assert result.n_success == 4
    assert result.n_failed == 0
    assert result.per_path_accepted_steps.shape == (4,)
    assert result.per_path_rejected_steps.shape == (4,)
    assert result.path_success.shape == (4,)
    assert result.endpoints.shape == (4, 2)
    assert result.endpoints.dtype == np.complex128
    assert result.residual_norms.shape == (4,)
    assert result.residual_norms.dtype == np.float64
    assert result.failure_codes == ("", "", "", "")


def test_tracking_result_from_julia_rejects_bad_endpoint_shape() -> None:
    payload = valid_pseudo_result()
    payload.endpoints = np.ones((4, 3), dtype=np.complex128)

    with pytest.raises(JuliaResultConversionError, match="endpoints"):
        tracking_result_from_julia(payload, n_vars=2)


def test_tracking_result_from_julia_rejects_bad_vector_shape() -> None:
    payload = valid_pseudo_result()
    payload.residual_norms = np.zeros((4, 1), dtype=np.float64)

    with pytest.raises(JuliaResultConversionError, match="TrackingResult"):
        tracking_result_from_julia(payload, n_vars=2)


def test_tracking_result_from_julia_rejects_count_mismatch() -> None:
    payload = valid_pseudo_result()
    payload.n_failed = 1

    with pytest.raises(JuliaResultConversionError, match="TrackingResult"):
        tracking_result_from_julia(payload, n_vars=2)


def test_tracking_result_from_julia_copies_arrays() -> None:
    payload = valid_pseudo_result()
    result = tracking_result_from_julia(payload, n_vars=2)

    payload.endpoints[0, 0] = 99.0 + 0.0j
    payload.per_path_accepted_steps[0] = 99

    assert result.endpoints[0, 0] == 1.0 + 0.0j
    assert result.per_path_accepted_steps[0] == 2
