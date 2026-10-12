from __future__ import annotations

import numpy as np
import pytest

from homotopy_path_learning.config import build_system_spec, load_config
from homotopy_path_learning.paths.parameterization import BezierParameterization, make_orthonormal_basis
from homotopy_path_learning.systems.sampling import start_system_coefficients
from homotopy_path_learning.systems.symmetry import (
    SymmetryElement,
    coefficient_map,
    enumerate_symmetries,
    free_real_matrix,
    latent_action_matrix,
    transform_coefficients,
)

CONFIG_DIR = "experiments/multivariate_pham/configs/"
CASES = [
    ("exp-0007-n3-d222-seed0.yaml", 96),
    ("exp-0007-n2-d33-seed0.yaml", 36),
    ("exp-0007-n2-d33-dense-seed0.yaml", 36),
    ("exp-0004-smoke.yaml", 16),
]


def spec_of(name: str):
    return build_system_spec(load_config(CONFIG_DIR + name))


def random_target(spec, rng):
    c = np.zeros(spec.n_coeffs, dtype=np.complex128)
    free = spec.free_coefficient_mask
    c[free] = rng.uniform(-1, 1, int(free.sum())) + 1j * rng.uniform(-1, 1, int(free.sum()))
    c[spec.leading_indices] = 1.0
    return c


@pytest.mark.parametrize(("name", "size"), CASES)
def test_group_size_and_identity_first(name, size):
    spec = spec_of(name)
    elements = enumerate_symmetries(spec)
    assert len(elements) == size
    assert len(set(elements)) == size
    assert elements[0].is_identity
    assert sum(e.is_identity for e in elements) == 1
    c = random_target(spec, np.random.default_rng(0))
    np.testing.assert_array_equal(transform_coefficients(spec, elements[0], c), c)
    assert elements == enumerate_symmetries(spec)


@pytest.mark.parametrize(("name", "size"), CASES)
def test_start_system_and_leading_invariant(name, size):
    spec = spec_of(name)
    start = start_system_coefficients(spec)
    c = random_target(spec, np.random.default_rng(1))
    for g in enumerate_symmetries(spec):
        np.testing.assert_array_equal(transform_coefficients(spec, g, start), start)
        out = transform_coefficients(spec, g, c)
        np.testing.assert_array_equal(out[spec.leading_indices], np.ones(spec.n_vars, dtype=np.complex128))
        _, phase = coefficient_map(spec, g)
        np.testing.assert_array_equal(phase[spec.leading_indices], np.ones(spec.n_vars, dtype=np.complex128))


@pytest.mark.parametrize(("name", "size"), CASES[:3])
def test_free_real_matrix_orthogonal_and_consistent(name, size):
    spec = spec_of(name)
    free = np.flatnonzero(spec.free_coefficient_mask)
    rng = np.random.default_rng(2)
    c = random_target(spec, rng)
    x = np.concatenate([c[free].real, c[free].imag])
    for g in enumerate_symmetries(spec):
        r = free_real_matrix(spec, g)
        np.testing.assert_allclose(r.T @ r, np.eye(r.shape[0]), atol=1e-12)
        out = transform_coefficients(spec, g, c)[free]
        np.testing.assert_allclose(r @ x, np.concatenate([out.real, out.imag]), atol=1e-12)


@pytest.mark.parametrize("name", ["exp-0004-smoke.yaml", "exp-0007-n2-d33-seed0.yaml"])
def test_latent_action_matrix_consistent(name):
    config = load_config(CONFIG_DIR + name)
    spec = build_system_spec(config)
    real_dim = 2 * int(spec.free_coefficient_mask.sum())
    degree = config.path.bezier_degree
    param = BezierParameterization.from_spec(
        spec, bezier_degree=degree, latent_dim=real_dim, rng=np.random.default_rng(3)
    )
    rng = np.random.default_rng(4)
    start = start_system_coefficients(spec)
    target = random_target(spec, rng)
    z = rng.uniform(-1, 1, param.action_dim)
    points = param.build_control_points(start, target, z)
    for g in enumerate_symmetries(spec):
        m = latent_action_matrix(spec, g, param.basis, degree)
        assert m.shape == (param.action_dim, param.action_dim)
        expected = param.build_control_points(start, transform_coefficients(spec, g, target), m @ z)
        np.testing.assert_allclose(transform_coefficients(spec, g, points), expected, atol=1e-12)


def test_inverse_returns_original_and_input_not_mutated():
    spec = spec_of("exp-0007-n2-d33-seed0.yaml")
    c = random_target(spec, np.random.default_rng(5))
    stack = np.stack([c, 2 * c])
    original = stack.copy()
    elements = enumerate_symmetries(spec)
    for g in elements:
        image = transform_coefficients(spec, g, stack)
        np.testing.assert_array_equal(stack, original)
        assert image.shape == stack.shape
        inverses = [h for h in elements if np.allclose(transform_coefficients(spec, h, image), stack, atol=1e-12)]
        assert len(inverses) >= 1


def test_invalid_inputs_rejected():
    spec = spec_of("exp-0007-n2-d33-seed0.yaml")
    c = random_target(spec, np.random.default_rng(6))
    identity = enumerate_symmetries(spec)[0]
    with pytest.raises(ValueError):
        transform_coefficients(spec, identity, c[:-1])
    with pytest.raises(ValueError):
        transform_coefficients(spec, identity, c.reshape(1, 1, -1))
    with pytest.raises(ValueError, match="permutation"):
        coefficient_map(spec, SymmetryElement((0, 0), (0, 0), False))
    with pytest.raises(ValueError, match="root_exponents"):
        coefficient_map(spec, SymmetryElement((0, 1), (3, 0), False))
    with pytest.raises(ValueError, match="length"):
        coefficient_map(spec, SymmetryElement((0, 1), (0,), False))
    small = make_orthonormal_basis(np.random.default_rng(0), real_dim=12, latent_dim=4)
    with pytest.raises(ValueError, match="not closed"):
        latent_action_matrix(spec, identity, small, 3)
    with pytest.raises(ValueError):
        latent_action_matrix(spec, identity, np.eye(11), 3)


def test_incompatible_permutation_rejected():
    # degrees (3, 2) cannot be permuted
    from homotopy_path_learning.systems import build_pham_spec

    spec = build_pham_spec((3, 2), [[(0, 0), (1, 0)], [(0, 0), (0, 1)]])
    assert all(g.permutation == (0, 1) for g in enumerate_symmetries(spec))
    with pytest.raises(ValueError, match="compatible"):
        coefficient_map(spec, SymmetryElement((1, 0), (0, 0), False))
