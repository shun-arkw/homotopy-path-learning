from __future__ import annotations

import numpy as np
import pytest

from homotopy_path_learning.systems import (
    ComplexUniformSampler,
    PolynomialSystemSpec,
    build_pham_spec,
    pham_start_solutions,
    sample_target_coefficients,
    start_system_coefficients,
)


SUPPORT_2X2 = (
    ((1, 0), (0, 1), (0, 0)),
    ((1, 0), (0, 1), (0, 0)),
)


def evaluate_system(
    spec: PolynomialSystemSpec, coefficients: np.ndarray, point: np.ndarray
) -> np.ndarray:
    values = np.zeros(spec.n_equations, dtype=np.complex128)
    for equation_index in range(spec.n_equations):
        start = int(spec.offsets[equation_index])
        stop = int(spec.offsets[equation_index + 1])
        for coefficient, exponent in zip(coefficients[start:stop], spec.exponents[start:stop]):
            values[equation_index] += coefficient * np.prod(point ** exponent)
    return values


def test_build_valid_2x2_quadratic_pham_spec() -> None:
    spec = build_pham_spec((2, 2), SUPPORT_2X2)

    assert spec.n_vars == 2
    assert spec.n_equations == 2
    assert spec.degrees == (2, 2)
    assert spec.n_coeffs == 8
    assert spec.M == 8
    np.testing.assert_array_equal(
        spec.exponents,
        np.array(
            [
                (2, 0),
                (1, 0),
                (0, 1),
                (0, 0),
                (0, 2),
                (1, 0),
                (0, 1),
                (0, 0),
            ],
            dtype=np.int64,
        ),
    )
    np.testing.assert_array_equal(spec.offsets, np.array([0, 4, 8], dtype=np.int64))
    np.testing.assert_array_equal(spec.leading_indices, np.array([0, 4], dtype=np.int64))
    np.testing.assert_array_equal(spec.constant_indices, np.array([3, 7], dtype=np.int64))
    assert spec.equation_exponents == (
        ((2, 0), (1, 0), (0, 1), (0, 0)),
        ((0, 2), (1, 0), (0, 1), (0, 0)),
    )


def test_invalid_degree_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        build_pham_spec((2, 0), SUPPORT_2X2)


def test_negative_exponent_is_rejected() -> None:
    supports = (((-1, 0), (0, 0)), ((1, 0), (0, 1), (0, 0)))
    with pytest.raises(ValueError, match="nonnegative"):
        build_pham_spec((2, 2), supports)


def test_duplicate_nonleading_exponent_is_rejected() -> None:
    supports = (((1, 0), (1, 0), (0, 0)), ((1, 0), (0, 1), (0, 0)))
    with pytest.raises(ValueError, match="duplicate"):
        build_pham_spec((2, 2), supports)


def test_duplicate_leading_monomial_is_rejected() -> None:
    supports = (((2, 0), (0, 0)), ((1, 0), (0, 1), (0, 0)))
    with pytest.raises(ValueError, match="leading exponent"):
        build_pham_spec((2, 2), supports)


def test_missing_leading_monomial_is_rejected() -> None:
    with pytest.raises(ValueError, match="leading monomial"):
        PolynomialSystemSpec(
            n_vars=1,
            degrees=(2,),
            exponents=np.array([[1], [0]], dtype=np.int64),
            offsets=np.array([0, 2], dtype=np.int64),
            leading_indices=np.array([0], dtype=np.int64),
            constant_indices=np.array([1], dtype=np.int64),
        )


def test_missing_constant_term_is_rejected() -> None:
    with pytest.raises(ValueError, match="constant"):
        build_pham_spec((2, 2), (((1, 0),), ((1, 0), (0, 1), (0, 0))))


def test_nonleading_total_degree_must_be_below_leading_degree() -> None:
    with pytest.raises(ValueError, match="total degree"):
        build_pham_spec((2, 2), (((1, 1), (0, 0)), ((1, 0), (0, 1), (0, 0))))


def test_start_coefficients_represent_x_i_power_d_i_minus_one() -> None:
    spec = build_pham_spec((2, 2), SUPPORT_2X2)
    coeffs = start_system_coefficients(spec)

    assert coeffs.dtype == np.complex128
    np.testing.assert_array_equal(
        coeffs,
        np.array([1, 0, 0, -1, 1, 0, 0, -1], dtype=np.complex128),
    )


def test_target_coefficients_keep_leading_entries_fixed() -> None:
    spec = build_pham_spec((2, 2), SUPPORT_2X2)
    coeffs = sample_target_coefficients(spec, np.random.default_rng(0), bound=0.5)

    np.testing.assert_array_equal(coeffs[spec.leading_indices], np.ones(2, dtype=np.complex128))
    assert np.all(np.abs(coeffs[spec.free_coefficient_mask].real) <= 0.5)
    assert np.all(np.abs(coeffs[spec.free_coefficient_mask].imag) <= 0.5)


def test_same_seed_reproduces_target_coefficients() -> None:
    spec = build_pham_spec((2, 2), SUPPORT_2X2)
    sampler = ComplexUniformSampler(bound=0.25)

    first = sampler.sample(np.random.default_rng(123), spec)
    second = sampler.sample(np.random.default_rng(123), spec)

    np.testing.assert_array_equal(first, second)


def test_start_solution_count_is_product_of_degrees() -> None:
    solutions = pham_start_solutions((2, 3))

    assert solutions.shape == (6, 2)
    assert solutions.dtype == np.complex128


def test_start_solutions_have_small_residual_on_start_system() -> None:
    spec = build_pham_spec((2, 2), SUPPORT_2X2)
    coeffs = start_system_coefficients(spec)
    solutions = pham_start_solutions(spec.degrees)

    for point in solutions:
        residual = evaluate_system(spec, coeffs, point)
        assert np.linalg.norm(residual) <= 1e-12
