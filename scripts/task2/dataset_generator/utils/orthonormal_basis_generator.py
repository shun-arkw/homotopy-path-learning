import numpy as np
from typing import Optional, Any
import time 


class OrthonormalBasisGenerator:
    """Generate orthonormal vectors orthogonal to a given direction vector.

    This class uses Gram-Schmidt orthonormalization to construct an
    orthonormal set of vectors that lie in the subspace orthogonal to
    a given direction vector.
    """

    def __init__(self, tol: float = 1e-8, max_attempts: int = 1000) -> None:
        """Initialize the generator.

        Args:
            tol: Tolerance used to decide whether a vector is too close
                to the zero vector and to judge unit-length checks.
            max_attempts: Maximum number of attempts to generate
                orthonormal vectors.
        """
        self.tol = tol
        self.max_attempts = max_attempts

    def generate(
        self,
        unit_direction_vector: np.ndarray,
        seed: Optional[int] = None,
        num_orthonormal_vectors: Optional[int] = None,
        normalize_direction: bool = True,
    ) -> np.ndarray:
        """Generate orthonormal vectors orthogonal to a direction vector.

        The input direction vector can optionally be normalized
        (controlled by ``normalize_direction``), and Gram-Schmidt
        orthonormalization is applied to produce
        ``num_orthonormal_vectors`` orthonormal vectors in the
        orthogonal subspace.

        Args:
            unit_direction_vector: Direction vector of shape ``(dim,)``.
                It must be non-zero. If ``normalize_direction`` is True,
                it does not need to be exactly unit length, since it is
                normalized inside this function. If
                ``normalize_direction`` is False, it must already be
                (approximately) unit length.
            seed: Seed for the random number generator. If ``None``, a
                random seed is used.
            num_orthonormal_vectors: Number of orthonormal vectors
                (orthogonal to the direction) to generate.
                If ``None``, it is set to ``dim - 1``.
            normalize_direction: If True, normalize
                ``unit_direction_vector`` before constructing the
                orthonormal basis. If False, the direction vector is
                not modified and is required to be unit length within
                the given tolerance. Default is True.

        Returns:
            np.ndarray: Array of shape
                ``(num_orthonormal_vectors, dim)`` containing orthonormal
                vectors (not including ``unit_direction_vector``), each
                orthogonal to ``unit_direction_vector``.

        Raises:
            ValueError: If ``unit_direction_vector`` is not 1D, is
                (numerically) zero, if it is not unit length when
                ``normalize_direction`` is False, if
                ``num_orthonormal_vectors`` is outside ``[1, dim - 1]``,
                or if an orthonormal set cannot be generated within
                ``max_attempts`` resamplings.
        """
        rng = np.random.default_rng(seed)  # random number generator

        # Ensure we have a 1D float array
        u = np.asarray(unit_direction_vector, dtype=float)
        if u.ndim != 1:
            raise ValueError("unit_direction_vector must be a 1D array.")

        dim = u.shape[0]

        # Check norm and optionally normalize the direction vector
        norm_u = np.linalg.norm(u)
        if norm_u < self.tol:
            raise ValueError("unit_direction_vector must be non-zero.")

        if normalize_direction:
            # Normalize to unit length
            u = u / norm_u
        else:
            # Require that the input is already (approximately) unit length
            if abs(norm_u - 1.0) > self.tol:
                raise ValueError(
                    "When normalize_direction is False, "
                    "unit_direction_vector must be approximately unit length "
                    f"(norm ≈ 1), but got norm={norm_u}."
                )

        # At this point, u is guaranteed to be (approximately) unit.
        u_norm_sq = 1.0  # for clarity; could also use float(np.dot(u, u))

        # Determine the number of orthonormal vectors
        if num_orthonormal_vectors is None:
            num_orthonormal_vectors = dim - 1
        elif not (1 <= num_orthonormal_vectors <= dim - 1):
            raise ValueError(
                f"num_orthonormal_vectors must be between 1 and {dim - 1}, "
                f"but got {num_orthonormal_vectors}"
            )

        # Generate orthonormal vectors using Gram–Schmidt orthonormalization
        orthonormal_vectors = []
        attempts = 0
        while (
            len(orthonormal_vectors) < num_orthonormal_vectors
            and attempts < self.max_attempts
        ):
            attempts += 1
            # Sample a random vector
            v = rng.normal(size=dim)

            # Remove the component along the direction vector
            # (u is unit, so this is just projecting and subtracting)
            alpha = np.dot(v, u) / u_norm_sq
            v = v - alpha * u

            # Remove components along already constructed orthonormal vectors
            for basis_vec in orthonormal_vectors:
                v = v - np.dot(v, basis_vec) * basis_vec

            norm_v = np.linalg.norm(v)
            if norm_v < self.tol:
                # The vector is too small; resample
                continue

            # Normalize and accept
            v = v / norm_v
            orthonormal_vectors.append(v)

        if len(orthonormal_vectors) < num_orthonormal_vectors:
            raise ValueError(
                f"Failed to generate {num_orthonormal_vectors} orthonormal "
                f"vectors within max_attempts={self.max_attempts}."
            )

        return np.stack(orthonormal_vectors, axis=0)


class OrthonormalBasisValidator:
    """Validate an orthonormal system including a given direction vector.

    This class checks whether a direction vector together with a set of
    orthonormal vectors forms an orthonormal system. Internally it
    builds a matrix whose rows are the vectors, and checks that the
    Gram matrix (Q Q^T) is close to the identity.
    """

    def __init__(self, tol: float = 1e-8) -> None:
        """Initialize the validator.

        Args:
            tol: Numerical tolerance for all checks. Any deviation
                larger than this value is considered a failure.
        """
        self.tol = tol

    def validate(
        self,
        unit_direction_vector: np.ndarray,
        orthonormal_vectors: np.ndarray,
        normalize_direction: bool = True,
        raise_on_error: bool = False,
    ) -> dict[str, Any]:
        """Validate that direction + orthonormal vectors form an orthonormal system.

        The direction vector and the orthonormal vectors are stacked as rows
        into a single matrix Q of shape (m + 1, dim). The Gram matrix
        G = Q Q^T is then compared against the identity matrix.

        Args:
            unit_direction_vector: Direction vector of shape ``(dim,)``.
                If ``normalize_direction`` is True, it will be
                normalized inside this function. If False, it must be
                (approximately) unit length.
            orthonormal_vectors: Orthonormal vectors of shape
                ``(m, dim)``, typically the output of
                ``OrthonormalBasisGenerator.generate(...)``. These
                vectors are assumed to be orthonormal among themselves
                and orthogonal to ``unit_direction_vector``.
            normalize_direction: If True, normalize
                ``unit_direction_vector`` before constructing Q. If
                False, the direction vector is not modified and is
                required to be unit length within the given tolerance.
            raise_on_error: If True, raise ``ValueError`` when the
                validation fails. If False, no exception is raised and
                the result is only reported in the returned dictionary.

        Returns:
            dict: A dictionary containing:
                - ``ok`` (bool): True if all checks passed within the
                  tolerance, False otherwise.
                - ``max_abs_gram_error`` (float): Maximum absolute
                  deviation of the Gram matrix from the identity.
                - ``dim`` (int): Ambient dimension.
                - ``num_orthonormal_vectors`` (int): Number of
                  orthonormal vectors (m).
                - ``system_size`` (int): Total number of vectors
                  (m + 1, including the direction).

        Raises:
            ValueError: If input shapes are invalid, if the direction
                vector is (numerically) zero, if it is not unit length
                when ``normalize_direction`` is False, or if
                ``raise_on_error`` is True and validation fails.
        """
        # Convert inputs to arrays
        u = np.asarray(unit_direction_vector, dtype=float)
        U = np.asarray(orthonormal_vectors, dtype=float)

        if u.ndim != 1:
            raise ValueError("unit_direction_vector must be a 1D array.")
        if U.ndim != 2:
            raise ValueError(
                "orthonormal_vectors must be a 2D array of shape (m, dim)."
            )

        dim = u.shape[0]
        m, dim_U = U.shape
        if dim != dim_U:
            raise ValueError(
                f"Dimension mismatch: unit_direction_vector has dim={dim}, "
                f"but orthonormal_vectors has dim={dim_U}."
            )

        # Check norm and optionally normalize the direction vector
        norm_u = np.linalg.norm(u)
        if norm_u < self.tol:
            raise ValueError("unit_direction_vector must be non-zero.")

        if normalize_direction:
            u = u / norm_u
        else:
            if abs(norm_u - 1.0) > self.tol:
                raise ValueError(
                    "When normalize_direction is False, "
                    "unit_direction_vector must be approximately unit length "
                    f"(norm ≈ 1), but got norm={norm_u}."
                )

        # Stack direction and orthonormal vectors as rows: Q shape (m + 1, dim)
        Q = np.vstack([u, U])

        # Gram matrix of row vectors: G = Q Q^T
        G = Q @ Q.T  # shape (m + 1, m + 1)
        I = np.eye(m + 1)

        # Difference from identity
        G_diff = G - I
        max_abs_gram_error = float(np.max(np.abs(G_diff))) if G_diff.size > 0 else 0.0

        ok = max_abs_gram_error <= self.tol

        result: dict[str, Any] = {
            "ok": ok,
            "max_abs_gram_error": max_abs_gram_error,
            "dim": dim,
            "num_orthonormal_vectors": m,
            "system_size": m + 1,
        }

        if raise_on_error and not ok:
            raise ValueError(
                "Orthonormal system validation failed: "
                f"max_abs_gram_error={max_abs_gram_error:.3e}, "
                f"tol={self.tol:.3e}."
            )

        return result


def main():
    direction_vector = np.array([1, -1, -1, 1, 1])
    unit_direction_vector = direction_vector / np.linalg.norm(direction_vector)
    print("Unit direction vector")
    print(unit_direction_vector)

    orthonormal_basis_generator = OrthonormalBasisGenerator(tol=1e-8)
    time_start = time.time()
    orthonormal_vectors = orthonormal_basis_generator.generate(unit_direction_vector, seed=42)
    time_end = time.time()
    print("Orthonormal vectors")
    print(orthonormal_vectors)
    print(f"Time taken: {time_end - time_start} seconds")

    orthonormal_basis_validator = OrthonormalBasisValidator(tol=1e-8)
    time_start = time.time()
    result = orthonormal_basis_validator.validate(unit_direction_vector, orthonormal_vectors, normalize_direction=True, raise_on_error=False)
    time_end = time.time()
    print(f"Result: {result}")
    print(f"Time taken: {time_end - time_start} seconds")

    # print(unit_direction_vector.reshape(1, -1))
    # print(unit_orthogonal_vectors.shape)
    # orthogonal_matrix = np.concatenate([unit_direction_vector.reshape(1, -1), unit_orthogonal_vectors], axis=0)
    # print("Orthogonal matrix")
    # print(orthogonal_matrix)

    # print(orthogonal_matrix.T @ orthogonal_matrix)

if __name__ == "__main__":
    main()