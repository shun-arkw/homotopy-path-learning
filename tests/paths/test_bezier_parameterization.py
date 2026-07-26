from __future__ import annotations

import numpy as np
import pytest

from homotopy_path_learning.paths import (
    BezierParameterization,
    evaluate_bezier_control_points,
    free_real_vector_to_complex_delta,
    leading_coefficient_mask,
    linear_interpolation_control_points,
    make_orthonormal_basis,
    nonleading_coefficient_mask,
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


def test_make_orthonormal_basis_has_reproducible_orthonormal_columns() -> None:
    first = make_orthonormal_basis(np.random.default_rng(7), real_dim=12, latent_dim=4)
    second = make_orthonormal_basis(np.random.default_rng(7), real_dim=12, latent_dim=4)

    assert first.shape == (12, 4)
    assert first.dtype == np.float64
    np.testing.assert_allclose(first.T @ first, np.eye(4), atol=1e-12)
    np.testing.assert_array_equal(first, second)


def test_make_orthonormal_basis_rejects_too_many_columns() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        make_orthonormal_basis(np.random.default_rng(0), real_dim=3, latent_dim=4)


def test_leading_and_nonleading_masks(spec) -> None:
    leading = leading_coefficient_mask(spec.n_coeffs, spec.leading_indices)
    nonleading = nonleading_coefficient_mask(spec.n_coeffs, spec.leading_indices)

    np.testing.assert_array_equal(np.flatnonzero(leading), spec.leading_indices)
    np.testing.assert_array_equal(np.flatnonzero(nonleading), np.array([1, 2, 3, 5, 6, 7]))
    assert not np.any(leading & nonleading)


def test_linear_interpolation_control_points_have_expected_shape_and_endpoints(
    start_coeffs, target_coeffs
) -> None:
    control_points = linear_interpolation_control_points(start_coeffs, target_coeffs, 3)

    assert control_points.shape == (4, 8)
    assert control_points.dtype == np.complex128
    np.testing.assert_array_equal(control_points[0], start_coeffs)
    np.testing.assert_array_equal(control_points[-1], target_coeffs)
    np.testing.assert_allclose(
        control_points[1],
        (2.0 / 3.0) * start_coeffs + (1.0 / 3.0) * target_coeffs,
    )
    np.testing.assert_allclose(
        control_points[2],
        (1.0 / 3.0) * start_coeffs + (2.0 / 3.0) * target_coeffs,
    )


def test_zero_action_builds_exact_linear_control_points(spec, start_coeffs, target_coeffs) -> None:
    parameterization = BezierParameterization.from_spec(
        spec,
        bezier_degree=3,
        latent_dim=4,
        rng=np.random.default_rng(0),
    )
    zero_action = np.zeros(parameterization.action_dim, dtype=np.float64)

    control_points = parameterization.build_control_points(
        start_coeffs,
        target_coeffs,
        zero_action,
    )

    expected = linear_interpolation_control_points(start_coeffs, target_coeffs, 3)
    np.testing.assert_array_equal(control_points, expected)
    np.testing.assert_array_equal(
        control_points[:, spec.leading_indices],
        np.ones((4, 2), dtype=np.complex128),
    )


def test_zero_action_bezier_curve_evaluates_to_straight_line(
    spec, start_coeffs, target_coeffs
) -> None:
    parameterization = BezierParameterization.from_spec(
        spec,
        bezier_degree=3,
        latent_dim=4,
        rng=np.random.default_rng(1),
    )
    control_points = parameterization.build_control_points(
        start_coeffs,
        target_coeffs,
        np.zeros(parameterization.action_dim, dtype=np.float64),
    )

    for t in (0.0, 0.125, 0.5, 0.875, 1.0):
        expected = (1.0 - t) * start_coeffs + t * target_coeffs
        np.testing.assert_allclose(evaluate_bezier_control_points(control_points, t), expected)


def test_nonzero_action_perturbs_only_nonleading_coefficients(
    spec, start_coeffs, target_coeffs
) -> None:
    parameterization = BezierParameterization.from_spec(
        spec,
        bezier_degree=3,
        latent_dim=4,
        rng=np.random.default_rng(2),
    )
    action = np.linspace(-0.25, 0.5, parameterization.action_dim, dtype=np.float64)

    control_points = parameterization.build_control_points(start_coeffs, target_coeffs, action)
    linear_points = linear_interpolation_control_points(start_coeffs, target_coeffs, 3)

    np.testing.assert_array_equal(control_points[0], start_coeffs)
    np.testing.assert_array_equal(control_points[-1], target_coeffs)
    np.testing.assert_array_equal(
        control_points[:, spec.leading_indices],
        np.ones((4, 2), dtype=np.complex128),
    )
    assert np.linalg.norm(control_points[1:-1, spec.free_coefficient_mask] - linear_points[1:-1, spec.free_coefficient_mask]) > 0


def test_free_real_vector_to_complex_delta_uses_free_re_concat_im_layout(spec) -> None:
    free_indices = np.flatnonzero(spec.free_coefficient_mask).astype(np.int64)
    free_real = np.zeros(2 * free_indices.size, dtype=np.float64)
    free_real[0] = 0.5
    free_real[free_indices.size] = -0.25

    delta = free_real_vector_to_complex_delta(
        free_real,
        n_coeffs=spec.n_coeffs,
        free_indices=free_indices,
        leading_indices=spec.leading_indices,
    )

    assert delta.dtype == np.complex128
    assert delta[free_indices[0]] == 0.5 - 0.25j
    np.testing.assert_array_equal(delta[spec.leading_indices], np.zeros(2, dtype=np.complex128))


def test_known_basis_action_maps_to_expected_intermediate_delta(
    spec, start_coeffs, target_coeffs
) -> None:
    free_indices = np.flatnonzero(spec.free_coefficient_mask).astype(np.int64)
    basis = np.zeros((2 * free_indices.size, 2), dtype=np.float64)
    basis[0, 0] = 1.0
    basis[free_indices.size, 1] = 1.0
    parameterization = BezierParameterization(
        bezier_degree=2,
        basis=basis,
        free_indices=free_indices,
        leading_indices=spec.leading_indices,
    )

    control_points = parameterization.build_control_points(
        start_coeffs,
        target_coeffs,
        np.array([0.5, -0.25], dtype=np.float64),
    )

    expected = linear_interpolation_control_points(start_coeffs, target_coeffs, 2)
    expected[1, free_indices[0]] += 0.5 - 0.25j
    np.testing.assert_array_equal(control_points, expected)


def test_action_shape_is_validated(spec, start_coeffs, target_coeffs) -> None:
    parameterization = BezierParameterization.from_spec(
        spec,
        bezier_degree=3,
        latent_dim=4,
        rng=np.random.default_rng(0),
    )

    with pytest.raises(ValueError, match="action"):
        parameterization.build_control_points(start_coeffs, target_coeffs, np.zeros((2, 4)))

    with pytest.raises(ValueError, match="action"):
        parameterization.build_control_points(
            start_coeffs,
            target_coeffs,
            np.zeros(parameterization.action_dim + 1),
        )


def test_target_leading_coefficients_must_be_fixed(spec, start_coeffs, target_coeffs) -> None:
    parameterization = BezierParameterization.from_spec(
        spec,
        bezier_degree=3,
        latent_dim=4,
        rng=np.random.default_rng(0),
    )
    bad_target = target_coeffs.copy()
    bad_target[spec.leading_indices[1]] = 1.5 + 0.0j

    with pytest.raises(ValueError, match="leading"):
        parameterization.build_control_points(
            start_coeffs,
            bad_target,
            np.zeros(parameterization.action_dim),
        )


def test_validate_control_points_rejects_changed_leading_coefficient(
    spec, start_coeffs, target_coeffs
) -> None:
    control_points = linear_interpolation_control_points(start_coeffs, target_coeffs, 3)
    control_points[1, spec.leading_indices[0]] = 1.0 + 1.0j

    with pytest.raises(ValueError, match="leading"):
        validate_control_points(
            control_points,
            start_coeffs=start_coeffs,
            target_coeffs=target_coeffs,
            leading_indices=spec.leading_indices,
            bezier_degree=3,
        )
