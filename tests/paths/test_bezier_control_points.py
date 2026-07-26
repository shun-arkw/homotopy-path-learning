from __future__ import annotations

import numpy as np
import pytest

from homotopy_path_learning.paths import (
    BezierParameterization,
    evaluate_bezier_control_points,
    free_real_vector_to_complex_delta,
    linear_interpolation_control_points,
    make_orthonormal_basis,
    validate_control_points,
)
from homotopy_path_learning.systems import build_pham_spec, start_system_coefficients


SUPPORT_2X2 = (
    ((1, 0), (0, 1), (0, 0)),
    ((1, 0), (0, 1), (0, 0)),
)


@pytest.fixture
def spec():
    return build_pham_spec((2, 2), SUPPORT_2X2)


@pytest.fixture
def start_coeffs(spec):
    return start_system_coefficients(spec)


@pytest.fixture
def target_coeffs():
    return np.array(
        [
            1.0 + 0.0j,
            0.25 + 0.125j,
            -0.5 + 0.75j,
            0.1 - 0.2j,
            1.0 + 0.0j,
            -0.3 + 0.4j,
            0.6 - 0.1j,
            -0.7 + 0.2j,
        ],
        dtype=np.complex128,
    )


def test_from_spec_uses_smoke_dimensions_and_reproducible_basis(spec) -> None:
    first = BezierParameterization.from_spec(
        spec,
        bezier_degree=3,
        latent_dim=4,
        rng=np.random.default_rng(11),
    )
    second = BezierParameterization.from_spec(
        spec,
        bezier_degree=3,
        latent_dim=4,
        rng=np.random.default_rng(11),
    )

    assert spec.n_vars == 2
    assert spec.degrees == (2, 2)
    assert spec.n_coeffs == 8
    np.testing.assert_array_equal(spec.leading_indices, np.array([0, 4], dtype=np.int64))
    assert first.bezier_degree == 3
    assert first.latent_dim == 4
    assert first.action_dim == 8
    assert first.basis.shape == (12, 4)
    np.testing.assert_allclose(first.basis.T @ first.basis, np.eye(4), atol=1e-12)
    np.testing.assert_array_equal(first.basis, second.basis)
    np.testing.assert_array_equal(first.free_indices, np.array([1, 2, 3, 5, 6, 7]))
    np.testing.assert_array_equal(first.leading_indices, np.array([0, 4]))


def test_from_spec_rejects_latent_dim_larger_than_free_real_dim(spec) -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        BezierParameterization.from_spec(
            spec,
            bezier_degree=3,
            latent_dim=13,
            rng=np.random.default_rng(0),
        )


def test_build_control_points_does_not_mutate_inputs(spec, start_coeffs, target_coeffs) -> None:
    parameterization = BezierParameterization.from_spec(
        spec,
        bezier_degree=3,
        latent_dim=4,
        rng=np.random.default_rng(3),
    )
    action = np.linspace(-0.2, 0.2, parameterization.action_dim, dtype=np.float64)
    start_before = start_coeffs.copy()
    target_before = target_coeffs.copy()
    action_before = action.copy()

    control_points = parameterization.build_control_points(start_coeffs, target_coeffs, action)

    np.testing.assert_array_equal(start_coeffs, start_before)
    np.testing.assert_array_equal(target_coeffs, target_before)
    np.testing.assert_array_equal(action, action_before)
    np.testing.assert_array_equal(control_points[0], start_before)
    np.testing.assert_array_equal(control_points[-1], target_before)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_build_control_points_rejects_nan_or_inf_action(
    spec, start_coeffs, target_coeffs, bad_value
) -> None:
    parameterization = BezierParameterization.from_spec(
        spec,
        bezier_degree=3,
        latent_dim=4,
        rng=np.random.default_rng(0),
    )
    action = np.zeros(parameterization.action_dim, dtype=np.float64)
    action[0] = bad_value

    with pytest.raises(ValueError, match="action"):
        parameterization.build_control_points(start_coeffs, target_coeffs, action)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_bezier_parameterization_rejects_nan_or_inf_basis(spec, bad_value) -> None:
    free_indices = np.flatnonzero(spec.free_coefficient_mask).astype(np.int64)
    basis = make_orthonormal_basis(
        np.random.default_rng(4),
        real_dim=2 * free_indices.size,
        latent_dim=4,
    )
    basis[0, 0] = bad_value

    with pytest.raises(ValueError, match="basis"):
        BezierParameterization(
            bezier_degree=3,
            basis=basis,
            free_indices=free_indices,
            leading_indices=spec.leading_indices,
        )


def test_bezier_parameterization_rejects_nonorthonormal_basis(spec) -> None:
    free_indices = np.flatnonzero(spec.free_coefficient_mask).astype(np.int64)
    basis = make_orthonormal_basis(
        np.random.default_rng(5),
        real_dim=2 * free_indices.size,
        latent_dim=4,
    )
    basis[:, 1] = basis[:, 0]

    with pytest.raises(ValueError, match="orthonormal"):
        BezierParameterization(
            bezier_degree=3,
            basis=basis,
            free_indices=free_indices,
            leading_indices=spec.leading_indices,
        )


def test_bezier_parameterization_rejects_free_and_leading_index_overlap(spec) -> None:
    free_indices = np.array([0, 1, 2, 3, 5, 6], dtype=np.int64)
    basis = make_orthonormal_basis(
        np.random.default_rng(6),
        real_dim=2 * free_indices.size,
        latent_dim=4,
    )

    with pytest.raises(ValueError, match="disjoint"):
        BezierParameterization(
            bezier_degree=3,
            basis=basis,
            free_indices=free_indices,
            leading_indices=spec.leading_indices,
        )


def test_free_real_vector_to_complex_delta_rejects_index_overlap(spec) -> None:
    free_indices = np.array([0, 1, 2, 3, 5, 6], dtype=np.int64)
    free_real = np.zeros(2 * free_indices.size, dtype=np.float64)

    with pytest.raises(ValueError, match="disjoint"):
        free_real_vector_to_complex_delta(
            free_real,
            n_coeffs=spec.n_coeffs,
            free_indices=free_indices,
            leading_indices=spec.leading_indices,
        )


def test_free_real_vector_to_complex_delta_rejects_nan_or_inf(spec) -> None:
    free_indices = np.flatnonzero(spec.free_coefficient_mask).astype(np.int64)
    free_real = np.zeros(2 * free_indices.size, dtype=np.float64)
    free_real[-1] = np.inf

    with pytest.raises(ValueError, match="free_real_vector"):
        free_real_vector_to_complex_delta(
            free_real,
            n_coeffs=spec.n_coeffs,
            free_indices=free_indices,
            leading_indices=spec.leading_indices,
        )


def test_validate_control_points_rejects_bad_shape(spec, start_coeffs, target_coeffs) -> None:
    control_points = linear_interpolation_control_points(start_coeffs, target_coeffs, 3)

    with pytest.raises(ValueError, match="shape"):
        validate_control_points(
            control_points[:-1],
            start_coeffs=start_coeffs,
            target_coeffs=target_coeffs,
            leading_indices=spec.leading_indices,
            bezier_degree=3,
        )


@pytest.mark.parametrize("bad_value", [np.nan + 0.0j, np.inf + 0.0j, 0.0 + np.inf * 1j])
def test_validate_control_points_rejects_nan_or_inf_control_points(
    spec, start_coeffs, target_coeffs, bad_value
) -> None:
    control_points = linear_interpolation_control_points(start_coeffs, target_coeffs, 3)
    control_points[1, 1] = bad_value

    with pytest.raises(ValueError, match="control_points"):
        validate_control_points(
            control_points,
            start_coeffs=start_coeffs,
            target_coeffs=target_coeffs,
            leading_indices=spec.leading_indices,
            bezier_degree=3,
        )


def test_evaluate_bezier_control_points_rejects_nan_control_point(
    start_coeffs, target_coeffs
) -> None:
    control_points = linear_interpolation_control_points(start_coeffs, target_coeffs, 3)
    control_points[1, 1] = np.nan + 0.0j

    with pytest.raises(ValueError, match="control_points"):
        evaluate_bezier_control_points(control_points, 0.5)


@pytest.mark.parametrize("bad_t", [-0.1, 1.1, np.nan, np.inf])
def test_evaluate_bezier_control_points_rejects_invalid_t(start_coeffs, target_coeffs, bad_t) -> None:
    control_points = linear_interpolation_control_points(start_coeffs, target_coeffs, 3)

    with pytest.raises(ValueError, match="t"):
        evaluate_bezier_control_points(control_points, bad_t)
