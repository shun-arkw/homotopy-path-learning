"""Real vector parameterizations and Bezier control points for coefficients."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from homotopy_path_learning.systems.spec import PolynomialSystemSpec


def complex_coefficients_to_real(coefficients: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Convert ``complex[M]`` coefficients to ``real[2M]``.

    The layout is ``[Re(c_0), ..., Re(c_{M-1}), Im(c_0), ..., Im(c_{M-1})]``.
    The returned array is one-dimensional with dtype ``float64``.
    """

    coeffs = np.asarray(coefficients, dtype=np.complex128)
    if coeffs.ndim != 1:
        raise ValueError(f"coefficients must have shape (M,), got {coeffs.shape}.")
    return np.concatenate([coeffs.real, coeffs.imag]).astype(np.float64, copy=False)


def real_to_complex_coefficients(vector: npt.ArrayLike) -> npt.NDArray[np.complex128]:
    """Convert ``real[2M]`` back to ``complex[M]`` coefficients.

    The input layout must be ``[Re(c_0), ..., Re(c_{M-1}), Im(c_0), ...,
    Im(c_{M-1})]``. The returned array has shape ``(M,)`` and dtype
    ``complex128``.
    """

    values = np.asarray(vector, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"vector must have shape (2M,), got {values.shape}.")
    if values.shape[0] % 2 != 0:
        raise ValueError(f"vector length must be even, got {values.shape[0]}.")

    half = values.shape[0] // 2
    return (values[:half] + 1j * values[half:]).astype(np.complex128)


complex_coeffs_to_real = complex_coefficients_to_real
real_to_complex_coeffs = real_to_complex_coefficients


def _is_integer_value(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _validate_positive_integer(name: str, value: object) -> int:
    if not _is_integer_value(value):
        raise TypeError(f"{name} must be an integer, got {value!r}.")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive, got {normalized}.")
    return normalized


def _validate_complex_vector(
    name: str,
    values: npt.ArrayLike,
    *,
    expected_length: int | None = None,
) -> npt.NDArray[np.complex128]:
    vector = np.asarray(values, dtype=np.complex128)
    if vector.ndim != 1:
        raise ValueError(f"{name} must have shape (M,), got {vector.shape}.")
    if expected_length is not None and vector.shape != (expected_length,):
        raise ValueError(f"{name} must have shape ({expected_length},), got {vector.shape}.")
    if not np.all(np.isfinite(vector.real)) or not np.all(np.isfinite(vector.imag)):
        raise ValueError(f"{name} must not contain NaN or Inf.")
    return vector


def _validate_real_vector(
    name: str,
    values: npt.ArrayLike,
    *,
    expected_length: int,
) -> npt.NDArray[np.float64]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.shape != (expected_length,):
        raise ValueError(f"{name} must have shape ({expected_length},), got {vector.shape}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must not contain NaN or Inf.")
    return vector


def _validate_index_array(name: str, values: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    if not isinstance(values, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray.")
    if values.dtype != np.int64:
        raise TypeError(f"{name} must have dtype int64, got {values.dtype}.")
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got ndim={values.ndim}.")
    if values.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if np.any(values < 0):
        raise ValueError(f"{name} must contain only nonnegative indices.")
    if np.unique(values).size != values.size:
        raise ValueError(f"{name} must not contain duplicate indices.")
    return values


def leading_coefficient_mask(
    n_coeffs: int, leading_indices: npt.NDArray[np.int64]
) -> npt.NDArray[np.bool_]:
    """Return a boolean mask selecting fixed leading coefficient entries."""

    n_coeffs = _validate_positive_integer("n_coeffs", n_coeffs)
    indices = _validate_index_array("leading_indices", leading_indices)
    if np.any(indices >= n_coeffs):
        raise ValueError("leading_indices must be within the coefficient vector.")

    mask = np.zeros(n_coeffs, dtype=np.bool_)
    mask[indices] = True
    return mask


def nonleading_coefficient_mask(
    n_coeffs: int, leading_indices: npt.NDArray[np.int64]
) -> npt.NDArray[np.bool_]:
    """Return a boolean mask selecting coefficients that may be perturbed."""

    return ~leading_coefficient_mask(n_coeffs, leading_indices)


def make_orthonormal_basis(
    rng: np.random.Generator,
    *,
    real_dim: int,
    latent_dim: int,
) -> npt.NDArray[np.float64]:
    """Generate ``U`` with orthonormal columns and shape ``(real_dim, latent_dim)``.

    ``rng`` must be a ``numpy.random.Generator``. No global random state is used.
    The implementation uses a reduced QR decomposition and fixes column signs
    from the diagonal of ``R`` for deterministic output under a fixed seed.
    """

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an instance of numpy.random.Generator.")
    real_dim = _validate_positive_integer("real_dim", real_dim)
    latent_dim = _validate_positive_integer("latent_dim", latent_dim)
    if latent_dim > real_dim:
        raise ValueError(
            f"latent_dim={latent_dim} cannot exceed real_dim={real_dim} for "
            "orthonormal columns."
        )

    raw = rng.standard_normal(size=(real_dim, latent_dim))
    q, r = np.linalg.qr(raw, mode="reduced")
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return (q * signs).astype(np.float64, copy=False)


def linear_interpolation_control_points(
    start_coeffs: npt.ArrayLike,
    target_coeffs: npt.ArrayLike,
    bezier_degree: int,
) -> npt.NDArray[np.complex128]:
    """Return Bezier control points on the straight line between endpoints.

    The returned array has shape ``(bezier_degree + 1, M)`` and dtype
    ``complex128``. Row ``k`` is
    ``(1 - k / bezier_degree) * start_coeffs + (k / bezier_degree) * target_coeffs``.
    """

    degree = _validate_positive_integer("bezier_degree", bezier_degree)
    start = _validate_complex_vector("start_coeffs", start_coeffs)
    target = _validate_complex_vector("target_coeffs", target_coeffs, expected_length=start.size)

    weights = np.linspace(0.0, 1.0, num=degree + 1, dtype=np.float64)
    control_points = (1.0 - weights[:, None]) * start[None, :] + weights[:, None] * target[
        None, :
    ]
    return control_points.astype(np.complex128, copy=False)


def validate_control_points(
    control_points: npt.ArrayLike,
    *,
    start_coeffs: npt.ArrayLike,
    target_coeffs: npt.ArrayLike,
    leading_indices: npt.NDArray[np.int64],
    bezier_degree: int,
) -> npt.NDArray[np.complex128]:
    """Validate coefficient-space Bezier control point invariants.

    Expected shape is ``(bezier_degree + 1, M)`` with dtype ``complex128`` after
    conversion. The endpoints must equal ``start_coeffs`` and ``target_coeffs``
    exactly, all leading coefficient entries must be exactly ``1 + 0j``, and no
    entry may contain NaN or Inf.
    """

    degree = _validate_positive_integer("bezier_degree", bezier_degree)
    start = _validate_complex_vector("start_coeffs", start_coeffs)
    target = _validate_complex_vector("target_coeffs", target_coeffs, expected_length=start.size)
    indices = _validate_index_array("leading_indices", leading_indices)
    if np.any(indices >= start.size):
        raise ValueError("leading_indices must be within the coefficient vector.")

    points = np.asarray(control_points, dtype=np.complex128)
    expected_shape = (degree + 1, start.size)
    if points.ndim != 2 or points.shape != expected_shape:
        raise ValueError(f"control_points must have shape {expected_shape}, got {points.shape}.")
    if not np.all(np.isfinite(points.real)) or not np.all(np.isfinite(points.imag)):
        raise ValueError("control_points must not contain NaN or Inf.")
    if not np.array_equal(points[0], start):
        raise ValueError("control_points[0] must equal start_coeffs.")
    if not np.array_equal(points[-1], target):
        raise ValueError("control_points[-1] must equal target_coeffs.")
    if not np.array_equal(start[indices], np.ones(indices.size, dtype=np.complex128)):
        raise ValueError("start_coeffs leading entries must be exactly 1 + 0j.")
    if not np.array_equal(target[indices], np.ones(indices.size, dtype=np.complex128)):
        raise ValueError("target_coeffs leading entries must be exactly 1 + 0j.")
    expected_leading = np.ones((degree + 1, indices.size), dtype=np.complex128)
    if not np.array_equal(points[:, indices], expected_leading):
        raise ValueError("all control point leading entries must be exactly 1 + 0j.")
    return points


def free_real_vector_to_complex_delta(
    free_real_vector: npt.ArrayLike,
    *,
    n_coeffs: int,
    free_indices: npt.NDArray[np.int64],
    leading_indices: npt.NDArray[np.int64],
) -> npt.NDArray[np.complex128]:
    """Expand a real free-coordinate perturbation to a full complex vector.

    ``free_real_vector`` has shape ``(2 * D_free,)`` and layout
    ``[Re(delta_free), Im(delta_free)]`` in the order of ``free_indices``. The
    returned vector has shape ``(M,)`` and dtype ``complex128``. Leading entries
    are forced to zero.
    """

    n_coeffs = _validate_positive_integer("n_coeffs", n_coeffs)
    free = _validate_index_array("free_indices", free_indices)
    leading = _validate_index_array("leading_indices", leading_indices)
    if np.any(free >= n_coeffs) or np.any(leading >= n_coeffs):
        raise ValueError("free_indices and leading_indices must be within the coefficient vector.")
    if np.intersect1d(free, leading).size != 0:
        raise ValueError("free_indices must be disjoint from leading_indices.")

    free_real = _validate_real_vector(
        "free_real_vector",
        free_real_vector,
        expected_length=2 * free.size,
    )
    free_delta = real_to_complex_coefficients(free_real)
    delta = np.zeros(n_coeffs, dtype=np.complex128)
    delta[free] = free_delta
    delta[leading] = 0.0 + 0.0j
    return delta


def evaluate_bezier_control_points(
    control_points: npt.ArrayLike, t: float
) -> npt.NDArray[np.complex128]:
    """Evaluate coefficient-space Bezier control points at ``t`` via De Casteljau."""

    points = np.asarray(control_points, dtype=np.complex128)
    if points.ndim != 2:
        raise ValueError(f"control_points must have shape (d_b + 1, M), got {points.shape}.")
    if points.shape[0] < 2:
        raise ValueError("control_points must contain at least two rows.")
    if not np.all(np.isfinite(points.real)) or not np.all(np.isfinite(points.imag)):
        raise ValueError("control_points must not contain NaN or Inf.")

    parameter = float(t)
    if not np.isfinite(parameter) or parameter < 0.0 or parameter > 1.0:
        raise ValueError(f"t must be finite and in [0, 1], got {t!r}.")

    working = points.copy()
    for level in range(points.shape[0] - 1, 0, -1):
        working[:level] = (1.0 - parameter) * working[:level] + parameter * working[1 : level + 1]
    return working[0].astype(np.complex128, copy=False)


@dataclass(frozen=True)
class BezierParameterization:
    """Latent parameterization of coefficient-space Bezier control points.

    ``basis`` is the latent matrix ``U`` with shape ``(2 * D_free, m)`` and
    dtype ``float64``. Rows use the same real representation as
    ``complex_coefficients_to_real`` but only for ``free_indices``:
    first all real parts in free-index order, then all imaginary parts. The
    flattened action has shape ``((bezier_degree - 1) * m,)`` and is reshaped
    internally to one latent vector per intermediate control point.
    """

    bezier_degree: int
    basis: npt.NDArray[np.float64]
    free_indices: npt.NDArray[np.int64]
    leading_indices: npt.NDArray[np.int64]

    def __post_init__(self) -> None:
        degree = _validate_positive_integer("bezier_degree", self.bezier_degree)
        object.__setattr__(self, "bezier_degree", degree)

        if not isinstance(self.basis, np.ndarray):
            raise TypeError("basis must be a numpy.ndarray.")
        if self.basis.dtype != np.float64:
            raise TypeError(f"basis must have dtype float64, got {self.basis.dtype}.")
        if self.basis.ndim != 2:
            raise ValueError(f"basis must be two-dimensional, got ndim={self.basis.ndim}.")
        if not np.all(np.isfinite(self.basis)):
            raise ValueError("basis must not contain NaN or Inf.")

        free = _validate_index_array("free_indices", self.free_indices)
        leading = _validate_index_array("leading_indices", self.leading_indices)
        if np.intersect1d(free, leading).size != 0:
            raise ValueError("free_indices must be disjoint from leading_indices.")

        expected_real_dim = 2 * free.size
        if self.basis.shape[0] != expected_real_dim:
            raise ValueError(
                f"basis must have shape ({expected_real_dim}, latent_dim), "
                f"got {self.basis.shape}."
            )
        if self.basis.shape[1] <= 0:
            raise ValueError("basis must have at least one latent column.")
        gram = self.basis.T @ self.basis
        identity = np.eye(self.basis.shape[1], dtype=np.float64)
        if not np.allclose(gram, identity, rtol=1e-10, atol=1e-10):
            raise ValueError("basis columns must be orthonormal.")

    @classmethod
    def from_spec(
        cls,
        spec: PolynomialSystemSpec,
        *,
        bezier_degree: int,
        latent_dim: int,
        rng: np.random.Generator,
    ) -> "BezierParameterization":
        """Create a parameterization using all non-leading coefficients as free."""

        free_indices = np.flatnonzero(spec.free_coefficient_mask).astype(np.int64)
        basis = make_orthonormal_basis(
            rng,
            real_dim=2 * free_indices.size,
            latent_dim=latent_dim,
        )
        return cls(
            bezier_degree=bezier_degree,
            basis=basis,
            free_indices=free_indices,
            leading_indices=spec.leading_indices.copy(),
        )

    @property
    def latent_dim(self) -> int:
        """Latent dimension ``m``."""

        return int(self.basis.shape[1])

    @property
    def n_free_coeffs(self) -> int:
        """Number of complex coefficients that may be perturbed."""

        return int(self.free_indices.size)

    @property
    def action_dim(self) -> int:
        """Flattened action dimension ``(bezier_degree - 1) * latent_dim``."""

        return (self.bezier_degree - 1) * self.latent_dim

    def build_control_points(
        self,
        start_coeffs: npt.ArrayLike,
        target_coeffs: npt.ArrayLike,
        action: npt.ArrayLike,
    ) -> npt.NDArray[np.complex128]:
        """Build all Bezier control points from a flattened latent action."""

        start = _validate_complex_vector("start_coeffs", start_coeffs)
        target = _validate_complex_vector("target_coeffs", target_coeffs, expected_length=start.size)
        max_index = int(max(np.max(self.free_indices), np.max(self.leading_indices)))
        if max_index >= start.size:
            raise ValueError("free_indices and leading_indices must fit coefficient vectors.")
        expected_leading = np.ones(self.leading_indices.size, dtype=np.complex128)
        if not np.array_equal(start[self.leading_indices], expected_leading):
            raise ValueError("start_coeffs leading entries must be exactly 1 + 0j.")
        if not np.array_equal(target[self.leading_indices], expected_leading):
            raise ValueError("target_coeffs leading entries must be exactly 1 + 0j.")

        action_vector = _validate_real_vector("action", action, expected_length=self.action_dim)
        control_points = linear_interpolation_control_points(start, target, self.bezier_degree)

        if self.bezier_degree > 1:
            action_rows = action_vector.reshape(self.bezier_degree - 1, self.latent_dim)
            for row_index, z_k in enumerate(action_rows, start=1):
                free_real_delta = self.basis @ z_k
                delta = free_real_vector_to_complex_delta(
                    free_real_delta,
                    n_coeffs=start.size,
                    free_indices=self.free_indices,
                    leading_indices=self.leading_indices,
                )
                control_points[row_index] += delta

        control_points[:, self.leading_indices] = 1.0 + 0.0j
        return validate_control_points(
            control_points,
            start_coeffs=start,
            target_coeffs=target,
            leading_indices=self.leading_indices,
            bezier_degree=self.bezier_degree,
        )


__all__ = [
    "BezierParameterization",
    "complex_coefficients_to_real",
    "complex_coeffs_to_real",
    "evaluate_bezier_control_points",
    "free_real_vector_to_complex_delta",
    "leading_coefficient_mask",
    "linear_interpolation_control_points",
    "make_orthonormal_basis",
    "nonleading_coefficient_mask",
    "real_to_complex_coefficients",
    "real_to_complex_coeffs",
    "validate_control_points",
]
