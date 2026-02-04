import numpy as np
from typing import Optional
from utils.orthonormal_basis_generator import OrthonormalBasisGenerator


class ControlPointsGenerator:
    """Generate control points for paths between two points in R^d (d-dimensional real space).

    This class generates a sequence of control points from a start point
    to a target point. The path is parameterized along the direction from
    start to target, and small perturbations are added in orthonormal
    directions using an orthonormal basis constructed by
    `OrthonormalBasisGenerator`.
    """

    def __init__(
        self,
        zero_vec_tol: float = 1e-8,
        max_attempts: int = 1000,
        basis_generator: Optional[OrthonormalBasisGenerator] = None,
    ) -> None:
        """Initialize the generator.

        Args:
            zero_vec_tol: Tolerance used to decide whether the direction
                (target - start) is too small.
            max_attempts: Maximum number of attempts to generate orthonormal vectors.
            basis_generator: Optional instance of `OrthonormalBasisGenerator`.
                If ``None``, a new instance is created on first use.
        """
        self.tol = zero_vec_tol
        self.max_attempts = max_attempts
        self._basis_generator = basis_generator  # may be None

    @property
    def basis_generator(self) -> OrthonormalBasisGenerator:
        """Return the basis generator, creating it lazily if needed."""
        if self._basis_generator is None:
            self._basis_generator = OrthonormalBasisGenerator(tol=self.tol, max_attempts=self.max_attempts)
        return self._basis_generator

    def generate(
        self,
        start_point: np.ndarray,
        target_point: np.ndarray,
        num_segments: int,
        noise_std: float = 0.1,
        num_orthonormal_vectors: Optional[int] = None,
        seed: Optional[int] = None,
        orthonormal_basis: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate a sequence of control points from start to target.

        The generated control points lie in a tube around the straight
        line segment from `start_point` to `target_point`. The parameter
        along the main direction is monotonically increasing, ensuring
        that the path always progresses from start to target without
        backtracking.

        Args:
            start_point: Start point of shape ``(dim,)``.
            target_point: Target point of shape ``(dim,)``.
            num_segments: Number of segments. The number of control
                points will be ``num_segments + 1`` (including both
                endpoints).
            noise_std: Standard deviation of the noise in orthonormal
                directions. Larger values allow the path to deviate more
                from the straight line.
            num_orthonormal_vectors: Number of orthonormal vectors
                (orthogonal to the main direction) used to perturb the
                path. If ``None``, this is set to ``dim - 1``. Must
                satisfy ``1 <= num_orthonormal_vectors <= dim - 1``.
            seed: Seed for the random number generator. If ``None``, a
                random seed is used.
            orthonormal_basis: Precomputed orthonormal basis of shape
                ``(m, dim)`` (where ``m`` equals ``num_orthonormal_vectors``)
                that is orthogonal to the direction ``target_point - start_point``.
                If provided, it is reused and generation is faster. If
                ``None``, the basis is computed internally.

        Returns:
            np.ndarray: Control points as an array of shape
                ``(num_segments + 1, dim)``. The first row equals
            ``start_point`` and the last row equals ``target_point``.

        Raises:
            ValueError: If the input points have incompatible shapes,
                if ``start_point`` and ``target_point`` are (numerically)
                identical, if ``dim < 2``, or if ``num_segments < 1``.

        """
        if num_segments < 1:
            raise ValueError("num_segments must be >= 1.")

        # Convert input to float arrays
        p_start = np.asarray(start_point, dtype=float)
        p_target = np.asarray(target_point, dtype=float)

        if p_start.ndim != 1 or p_target.ndim != 1:
            raise ValueError("start_point and target_point must be 1D arrays.")
        if p_start.shape != p_target.shape:
            raise ValueError("start_point and target_point must have the same shape.")

        dim = p_start.shape[0]
        if dim < 2:
            raise ValueError(
                "Dimension must be at least 2 to define orthonormal directions."
            )

        # Direction from start to target
        direction_vector = p_target - p_start
        L = np.linalg.norm(direction_vector)
        if L < self.tol:
            raise ValueError("start_point and target_point are too close or identical.")

        # Unit direction vector (main direction of the path)
        unit_direction_vector = direction_vector / L

        # Orthonormal basis in the orthogonal subspace.
        # Note: `orthonormal_basis` does NOT include `unit_direction_vector` itself.
        if orthonormal_basis is None:
            orthonormal_basis = self.basis_generator.generate(
                unit_direction_vector=unit_direction_vector,
                seed=seed,
                num_orthonormal_vectors=num_orthonormal_vectors,
                normalize_direction=False,  # direction vector is already unit
            )  # shape: (m, dim)
        else:
            orthonormal_basis = np.asarray(orthonormal_basis, dtype=float)
            if orthonormal_basis.ndim != 2 or orthonormal_basis.shape[1] != dim:
                raise ValueError(
                    "orthonormal_basis must have shape (m, dim) "
                    f"with dim={dim}, but got {orthonormal_basis.shape}."
                )
            if num_orthonormal_vectors is not None and orthonormal_basis.shape[0] != num_orthonormal_vectors:
                raise ValueError(
                    "Provided orthonormal_basis rows must match num_orthonormal_vectors."
                )

        m = orthonormal_basis.shape[0]

        rng = np.random.default_rng(seed)

        # Sample monotonically increasing parameters in [0, 1]
        ts = np.sort(rng.uniform(0.0, 1.0, size=num_segments + 1))
        ts[0] = 0.0
        ts[-1] = 1.0

        # Scale to [0, L] along the main direction
        a = ts * L  # shape: (num_segments + 1,)

        # Sample noise in orthonormal directions
        B = rng.normal(loc=0.0, scale=noise_std, size=(num_segments + 1, m))
        B[0, :] = 0.0  # fix start
        B[-1, :] = 0.0  # fix target

        # Base points on the straight line from p_start to p_target
        base = p_start + np.outer(a, unit_direction_vector)  # shape: (num_segments + 1, dim)

        # Orthonormal perturbations
        orth = B @ orthonormal_basis  # shape: (num_segments + 1, dim)

        return base + orth