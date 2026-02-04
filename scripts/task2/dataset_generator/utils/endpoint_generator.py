import numpy as np
from typing import Optional, Tuple
from .control_point_validator import ControlPointValidator
import time


class EndpointGenerator:
    """
    Generate valid (start_point, target_point) pairs under coordinate and norm constraints.

    Each coordinate is sampled uniformly from [coord_min, coord_max]. Points must pass
    ControlPointValidator.validate. The L2 distance between start_point and target_point
    must lie within [norm_min, norm_max].
    """

    def __init__(
        self,
        validator: ControlPointValidator,
        dim: int,
        coord_min: float,
        coord_max: float,
        norm_min: float,
        norm_max: float,
        max_attempts: int = 10000,
        seed: Optional[int] = None,
    ) -> None:
        """
        Args:
            validator: ControlPointValidator used to accept or reject sampled points.
            dim: Dimension of the ambient space.
            coord_min: Lower bound for uniform sampling of each coordinate.
            coord_max: Upper bound for uniform sampling of each coordinate.
            norm_min: Minimum allowed L2 distance between start_point and target_point.
            norm_max: Maximum allowed L2 distance between start_point and target_point.
            max_attempts: Maximum sampling attempts for start/target generation.
            seed: Optional RNG seed.
        """
        if dim < 1:
            raise ValueError("dim must be >= 1.")
        if coord_min >= coord_max:
            raise ValueError("coord_min must be less than coord_max.")
        if norm_min < 0 or norm_min >= norm_max:
            raise ValueError("Require 0 <= norm_min < norm_max.")

        self.validator = validator
        self.dim = dim
        self.coord_min = coord_min
        self.coord_max = coord_max
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.max_attempts = max_attempts
        self.rng = np.random.default_rng(seed)

    def _sample_point(self) -> np.ndarray:
        """Sample a single point uniformly within coordinate bounds."""
        return self.rng.uniform(self.coord_min, self.coord_max, size=self.dim)

    def _sample_valid_point(self) -> np.ndarray:
        """Sample a point that passes validator checks."""
        for _ in range(self.max_attempts):
            candidate = self._sample_point()
            if self.validator.validate(candidate).ok:
                return candidate
        raise RuntimeError("Failed to sample a valid point within max_attempts.")

    def _sample_valid_target(self, start_point: np.ndarray) -> np.ndarray:
        """Sample a target_point that satisfies norm and validator constraints."""
        for _ in range(self.max_attempts):
            candidate = self._sample_point()
            dist = np.linalg.norm(candidate - start_point)
            if not (self.norm_min <= dist <= self.norm_max):
                continue
            if self.validator.validate(candidate).ok:
                return candidate
        raise RuntimeError("Failed to sample a valid target_point within max_attempts.")

    def generate_pair(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate one (start_point, target_point) pair."""
        start_point = self._sample_valid_point()
        target_point = self._sample_valid_target(start_point)
        return start_point, target_point

    def generate_pairs(self, count: int) -> list[Tuple[np.ndarray, np.ndarray]]:
        """Generate multiple (start_point, target_point) pairs."""
        if count < 1:
            raise ValueError("count must be >= 1.")
        return [self.generate_pair() for _ in range(count)]


def main():
    validator = ControlPointValidator(imag_tol=1e-8, distinct_tol=1e-6)
    endpoint_generator = EndpointGenerator(
        validator=validator, 
        dim=6, 
        coord_min=-10, 
        coord_max=10, 
        norm_min=0, 
        norm_max=300,
        max_attempts=100000,
        seed=None,
    )

    for i in range(100):
        time_start = time.time()
        start_point, target_point = endpoint_generator.generate_pair()
        time_end = time.time()
        print(f"Time taken: {time_end - time_start} seconds")
        print(start_point)
        print(target_point)

if __name__ == "__main__":
    main()