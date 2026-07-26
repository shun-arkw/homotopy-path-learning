from __future__ import annotations

import numpy as np
import pytest

from homotopy_path_learning.backends.julia import (
    JuliaBackendValidationError,
    system_spec_to_julia_arrays,
    validate_control_points_for_backend,
    validate_system_arrays,
)
from homotopy_path_learning.paths import linear_interpolation_control_points
from homotopy_path_learning.systems import build_pham_spec, start_system_coefficients


SUPPORT_2X2 = (
    ((1, 0), (0, 1), (0, 0)),
    ((1, 0), (0, 1), (0, 0)),
)


@pytest.fixture
def spec():
    return build_pham_spec((2, 2), SUPPORT_2X2)


@pytest.fixture
def target_coeffs() -> np.ndarray:
    return np.array([1, 0, 0, -2, 1, 0, 0, -2], dtype=np.complex128)


@pytest.fixture
def control_points(spec, target_coeffs) -> np.ndarray:
    return linear_interpolation_control_points(
        start_system_coefficients(spec),
        target_coeffs,
        3,
    )


def test_system_spec_to_julia_arrays_copies_expected_shapes_and_dtypes(spec) -> None:
    arrays = system_spec_to_julia_arrays(spec)

    assert arrays.degrees.shape == (2,)
    assert arrays.degrees.dtype == np.int64
    assert arrays.exponents.shape == (8, 2)
    assert arrays.exponents.dtype == np.int64
    assert arrays.offsets.shape == (3,)
    assert arrays.offsets.dtype == np.int64
    assert arrays.leading_indices.shape == (2,)
    assert arrays.constant_indices.shape == (2,)
    np.testing.assert_array_equal(arrays.offsets, np.array([0, 4, 8], dtype=np.int64))
    arrays.offsets[0] = 99
    assert spec.offsets[0] == 0


def test_validate_system_arrays_rejects_bad_shape(spec) -> None:
    with pytest.raises(JuliaBackendValidationError, match="exponents"):
        validate_system_arrays(
            degrees=np.array(spec.degrees, dtype=np.int64),
            exponents=np.zeros((8, 3), dtype=np.int64),
            offsets=spec.offsets,
            leading_indices=spec.leading_indices,
            constant_indices=spec.constant_indices,
        )


def test_validate_system_arrays_rejects_bad_dtype(spec) -> None:
    with pytest.raises(TypeError, match="degrees"):
        validate_system_arrays(
            degrees=np.array(spec.degrees, dtype=np.int32),
            exponents=spec.exponents,
            offsets=spec.offsets,
            leading_indices=spec.leading_indices,
            constant_indices=spec.constant_indices,
        )


def test_validate_system_arrays_rejects_bad_offsets(spec) -> None:
    bad_offsets = spec.offsets.copy()
    bad_offsets[0] = 1
    with pytest.raises(JuliaBackendValidationError, match="offsets"):
        validate_system_arrays(
            degrees=np.array(spec.degrees, dtype=np.int64),
            exponents=spec.exponents,
            offsets=bad_offsets,
            leading_indices=spec.leading_indices,
            constant_indices=spec.constant_indices,
        )

    bad_offsets = spec.offsets.copy()
    bad_offsets[1] = bad_offsets[0]
    with pytest.raises(JuliaBackendValidationError, match="strictly"):
        validate_system_arrays(
            degrees=np.array(spec.degrees, dtype=np.int64),
            exponents=spec.exponents,
            offsets=bad_offsets,
            leading_indices=spec.leading_indices,
            constant_indices=spec.constant_indices,
        )


def test_validate_system_arrays_rejects_out_of_range_and_wrong_block_indices(spec) -> None:
    bad_leading = spec.leading_indices.copy()
    bad_leading[0] = spec.n_coeffs
    with pytest.raises(JuliaBackendValidationError, match=r"\[0, M\)"):
        validate_system_arrays(
            degrees=np.array(spec.degrees, dtype=np.int64),
            exponents=spec.exponents,
            offsets=spec.offsets,
            leading_indices=bad_leading,
            constant_indices=spec.constant_indices,
        )

    bad_leading = spec.leading_indices.copy()
    bad_leading[0] = spec.leading_indices[1]
    with pytest.raises(JuliaBackendValidationError, match="equation block"):
        validate_system_arrays(
            degrees=np.array(spec.degrees, dtype=np.int64),
            exponents=spec.exponents,
            offsets=spec.offsets,
            leading_indices=bad_leading,
            constant_indices=spec.constant_indices,
        )


def test_validate_control_points_converts_dtype_without_mutating_input(spec, control_points) -> None:
    source = np.asarray(control_points, dtype=np.complex64)
    source_before = source.copy()

    validated = validate_control_points_for_backend(
        source,
        system_spec=spec,
        bezier_degree=3,
    )

    assert validated.dtype == np.complex128
    np.testing.assert_allclose(validated, control_points)
    np.testing.assert_array_equal(source, source_before)
    validated[0, 0] = 99.0 + 0.0j
    assert source[0, 0] != 99.0 + 0.0j


def test_validate_control_points_rejects_nan_and_inf(spec, control_points) -> None:
    bad = control_points.copy()
    bad[1, 1] = np.nan + 0.0j
    with pytest.raises(JuliaBackendValidationError, match="NaN or Inf"):
        validate_control_points_for_backend(bad, system_spec=spec, bezier_degree=3)

    bad = control_points.copy()
    bad[1, 1] = 0.0 + np.inf * 1j
    with pytest.raises(JuliaBackendValidationError, match="NaN or Inf"):
        validate_control_points_for_backend(bad, system_spec=spec, bezier_degree=3)


def test_validate_control_points_rejects_changed_leading_coefficient(spec, control_points) -> None:
    bad = control_points.copy()
    bad[1, spec.leading_indices[0]] = 1.0 + 0.1j

    with pytest.raises(JuliaBackendValidationError, match="leading"):
        validate_control_points_for_backend(bad, system_spec=spec, bezier_degree=3)


def test_validate_control_points_rejects_bad_start_endpoint(spec, control_points) -> None:
    bad = control_points.copy()
    bad[0, spec.constant_indices[0]] = -2.0 + 0.0j

    with pytest.raises(JuliaBackendValidationError, match=r"control_points\[0\]"):
        validate_control_points_for_backend(bad, system_spec=spec, bezier_degree=3)


def test_validate_control_points_rejects_bad_shape(spec, control_points) -> None:
    with pytest.raises(JuliaBackendValidationError, match="shape"):
        validate_control_points_for_backend(
            control_points[:-1],
            system_spec=spec,
            bezier_degree=3,
        )
