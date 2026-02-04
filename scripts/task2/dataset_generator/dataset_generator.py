from dataclasses import dataclass
from typing import Optional

import time
import numpy as np
import sympy as sp
from joblib import Parallel, delayed

from utils.condition_length_calculator import calculate_linear_condition_length_numeric
from utils.control_point_validator import ControlPointValidator, ValidationResult, build_coeffs
from utils.control_points_generator import ControlPointsGenerator
from utils.endpoint_generator import EndpointGenerator


@dataclass
class ConditionLengthResult:
    control_points: np.ndarray
    segment_condition_lengths: list[float]
    total_condition_length: float

    @property
    def is_valid(self) -> bool:
        return np.isfinite(self.total_condition_length)


class DatasetGenerator:
    """
    Orchestrates parallel generation of multiple control-point sequences
    for a single (start, target) pair.

    Future responsibilities (validation, condition-length computation) can
    be added here. Currently, it computes the orthonormal basis once per
    pair and generates multiple sequences in parallel.
    """

    def __init__(
        self,
        control_points_generator: ControlPointsGenerator,
        control_point_validator: ControlPointValidator,
        n_jobs: int = -1,
        backend: str = "multiprocessing",
        verbose: int = 0,
    ) -> None:
        self.control_points_generator = control_points_generator
        self.control_point_validator = control_point_validator
        self.n_jobs = n_jobs
        self.backend = backend
        self.verbose = verbose
        self._default_backend = backend

    def _total_condition_length_for_sequence(
        self,
        control_points: np.ndarray,
    ) -> ConditionLengthResult:
        """Compute total condition length along a control-point sequence."""
        total_condition_length = 0.0
        segment_condition_lengths: list[float] = []
        for i in range(len(control_points) - 1):
            start_point = control_points[i]
            target_point = control_points[i + 1]
            start_coeffs = build_coeffs(start_point)
            target_coeffs = build_coeffs(target_point)

            try:
                condition_length = calculate_linear_condition_length_numeric(
                    start_coeffs=start_coeffs, 
                    target_coeffs=target_coeffs,
                    zero_tol=1e-12,
                    num_samples=200,
                    use_root_finder=True,
                    root_tol=1e-12,
                )
                total_condition_length += float(condition_length)
                segment_condition_lengths.append(float(condition_length))
            except Exception:
                # If Sympy discriminant or integration fails, mark as invalid
                return ConditionLengthResult(
                    control_points=control_points,
                    segment_condition_lengths=segment_condition_lengths,
                    total_condition_length=np.nan,
                )
        return ConditionLengthResult(
            control_points=control_points,
            segment_condition_lengths=segment_condition_lengths,
            total_condition_length=total_condition_length,
        )

    def compute_condition_lengths(
        self,
        control_point_sequences: list[np.ndarray],
        n_jobs: Optional[int] = None,
        backend: Optional[str] = None,
        verbose: Optional[int] = None,
        drop_invalid: bool = True,
    ) -> list[ConditionLengthResult]:
        """
        Compute total condition length for each control-point sequence in parallel.
        """
        if not control_point_sequences:
            return []
        jobs = (
            delayed(self._total_condition_length_for_sequence)(seq)
            for seq in control_point_sequences
        )
        results = Parallel(
            n_jobs=self.n_jobs if n_jobs is None else n_jobs,
            backend=self.backend if backend is None else backend,
            verbose=self.verbose if verbose is None else verbose,
        )(jobs)
        if drop_invalid:
            results = [r for r in results if r.is_valid]
        return results

    def _generate_and_validate(
        self,
        p_start: np.ndarray,
        p_target: np.ndarray,
        segment_count: int,
        noise_std: float,
        num_orthonormal_vectors: Optional[int],
        seed: int,
        orthonormal_basis: np.ndarray,
        control_points_generator: "ControlPointsGenerator",
        validator: ControlPointValidator,
    ) -> tuple[np.ndarray, list[ValidationResult]]:
        """Helper for joblib: generate one sequence and validate its control points."""
        control_points = control_points_generator.generate(
            start_point=p_start,
            target_point=p_target,
            num_segments=segment_count,
            noise_std=noise_std,
            num_orthonormal_vectors=num_orthonormal_vectors,
            seed=seed,
            orthonormal_basis=orthonormal_basis,
        )
        validation_results: list[ValidationResult] = []
        for cp in control_points:
            res = validator.validate(cp)
            validation_results.append(res)
            if not res.ok:
                break  # early exit if one control point fails
        return control_points, validation_results

    def generate_control_point_sequences(
        self,
        start_point: np.ndarray,
        target_point: np.ndarray,
        num_sequences: int,
        min_num_segments: int,
        max_num_segments: int,
        noise_std: float = 0.1,
        num_orthonormal_vectors: Optional[int] = None,
        seed: Optional[int] = None,
        drop_invalid: bool = True,
        validator: Optional[ControlPointValidator] = None,
    ) -> list[np.ndarray]:
        """
        Generate multiple control-point sequences in parallel for one
        start/target pair. The segment count for each sequence is uniformly
        sampled from [min_num_segments, max_num_segments].

        Args:
            drop_invalid: If True, discard sequences that fail validation.
            validator: Override the default validator passed at init. If None, the
                instance-level validator is used.
        """
        if num_sequences < 1:
            raise ValueError("num_sequences must be >= 1.")
        if min_num_segments < 1 or max_num_segments < min_num_segments:
            raise ValueError("segment bounds must satisfy 1 <= min <= max.")

        # Basic input validation
        p_start = np.asarray(start_point, dtype=float)
        p_target = np.asarray(target_point, dtype=float)
        if p_start.ndim != 1 or p_target.ndim != 1:
            raise ValueError("start_point and target_point must be 1D arrays.")
        if p_start.shape != p_target.shape:
            raise ValueError("start_point and target_point must have the same shape.")

        dim = p_start.shape[0]
        if dim < 2:
            raise ValueError("Dimension must be at least 2.")

        direction_vector = p_target - p_start
        L = np.linalg.norm(direction_vector)
        if L < self.control_points_generator.tol:
            raise ValueError("start_point and target_point are too close or identical.")

        unit_direction_vector = direction_vector / L

        # Compute the orthonormal basis once per (start, target) pair
        orthonormal_basis = self.control_points_generator.basis_generator.generate(
            unit_direction_vector=unit_direction_vector,
            seed=seed,
            num_orthonormal_vectors=num_orthonormal_vectors,
            normalize_direction=False,  # already unit
        )

        rng = np.random.default_rng(seed)
        segment_counts = rng.integers(
            low=min_num_segments, high=max_num_segments + 1, size=num_sequences
        )
        seq_seeds = rng.integers(
            low=0, high=np.iinfo(np.int32).max, size=num_sequences, dtype=np.int64
        )

        active_validator = validator or self.control_point_validator

        jobs = (
            delayed(self._generate_and_validate)(
                p_start=p_start,
                p_target=p_target,
                segment_count=int(segment_counts[i]),
                noise_std=noise_std,
                num_orthonormal_vectors=num_orthonormal_vectors,
                seed=int(seq_seeds[i]),
                orthonormal_basis=orthonormal_basis,
                control_points_generator=self.control_points_generator,
                validator=active_validator,
            )
            for i in range(num_sequences)
        )

        control_point_sequences = Parallel(
            n_jobs=self.n_jobs, backend=self.backend, verbose=self.verbose
        )(jobs)

        # Split results
        sequences: list[np.ndarray] = []
        for seq, val in control_point_sequences:
            if drop_invalid and not all(r.ok for r in val):
                continue
            sequences.append(seq)
        return sequences


def main():
    # start_point = np.array([-15, 85, -225, 274, -120])
    # target_point = np.array([-15.5, 91.1, -251.41, 321.6105, -149.73651])
    # start_point = np.array([-2, -1, 2]) 
    # target_point = np.array([-1, -4, 4])
    # start_point = np.array([-1,-1])
    # target_point = np.array([1,-1])
    # start_point = np.array([-4,1])
    # target_point = np.array([4,1])
    min_num_segments = 2
    max_num_segments = 2
    noise_std = 10
    seed = 42
    x = sp.symbols('x')

    validator = ControlPointValidator(imag_tol=1e-8, distinct_tol=1e-6)
    endpoint_generator = EndpointGenerator(
        validator=validator, 
        dim=6, 
        coord_min=-10, 
        coord_max=10, 
        norm_min=10, 
        norm_max=100,
        max_attempts=10000,
        seed=None,
    )
    start_point, target_point = endpoint_generator.generate_pair()
    print(f"\nStart point: {start_point}")
    print(f"Start polynomial: {sp.Poly(build_coeffs(start_point), x).as_expr()}\n")
    print(f"Target point: {target_point}")
    print(f"Target polynomial: {sp.Poly(build_coeffs(target_point), x).as_expr()}\n")
    print(f"Coefficient distance: {np.linalg.norm(target_point - start_point)}")

    start_coeffs = build_coeffs(start_point)
    target_coeffs = build_coeffs(target_point)
    linear_condition_length = calculate_linear_condition_length_numeric(
        start_coeffs,
        target_coeffs,
        zero_tol=1e-12,
        num_samples=200,
        use_root_finder=True,
        root_tol=1e-12,
    )
    print(f"Condition length: {linear_condition_length}\n")

    control_points_generator = ControlPointsGenerator(zero_vec_tol=1e-8, max_attempts=1000)
    control_point_validator = ControlPointValidator(imag_tol=1e-8, distinct_tol=1e-6)

    dataset_generator = DatasetGenerator(
        control_points_generator=control_points_generator,
        control_point_validator=control_point_validator,
        n_jobs=-1,
        backend="multiprocessing",
        verbose=1,
    )
    
    num_sequences = 10000
    time_start = time.time()
    control_points_sequences = dataset_generator.generate_control_point_sequences(
        start_point=start_point,
        target_point=target_point,
        num_sequences=num_sequences,
        min_num_segments=min_num_segments,
        max_num_segments=max_num_segments,
        noise_std=noise_std,
        seed=seed,
        drop_invalid=True,        # drop invalid sequences
    )
    time_end = time.time()
    print(f"Time taken: {time_end - time_start} seconds")
    print(f"Generated sequences (kept): {len(control_points_sequences)}\n")


    # Compute total condition lengths (in parallel) for kept sequences
    cl_time_start = time.time()
    condition_length_results = dataset_generator.compute_condition_lengths(
        control_points_sequences,
        drop_invalid=True,
        n_jobs=-1,
        verbose=1,
    )
    cl_time_end = time.time()
    print(f"Time taken: {cl_time_end - cl_time_start} seconds")
    print(f"Condition lengths computed for {len(condition_length_results)} sequences\n")
    if condition_length_results:
        best_result = min(
            condition_length_results, key=lambda r: r.total_condition_length
        )
        print(f"min condition length: {best_result.total_condition_length}")
        print("control points:")
        print(best_result.control_points)
        print("segment condition lengths:")
        print(best_result.segment_condition_lengths)
        print(max(condition_length_results, key=lambda r: r.total_condition_length).total_condition_length)


if __name__ == "__main__":
    main()