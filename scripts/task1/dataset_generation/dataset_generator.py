# dataset_generator.py
from __future__ import annotations

from dataclasses import dataclass
import argparse
import os
import numpy as np
from joblib import Parallel, delayed
from utils import DatasetWriter, DatasetVisualizer
from math import ceil


# ============
# Math helpers
# ============

def calculate_discriminant(a: float, b: float) -> float:
    """Return the discriminant D = a^2 - 4b for the quadratic x^2 + a x + b."""
    return a * a - 4.0 * b


def calculate_quad_roots(coeffs: tuple[float, float]) -> tuple[float, float]:
    """
    Calculate the real roots (λ1, λ2) of x^2 + a x + b = 0
    assuming D >= 0 by construction.

    The roots are returned in descending order: λ1 >= λ2.
    If a tiny negative discriminant appears due to roundoff, it is clamped to 0.
    
    Args:
        coeffs: Tuple (a, b) representing the quadratic coefficients
        
    Returns:
        Tuple (λ1, λ2) with λ1 >= λ2
    """
    a, b = coeffs
    D = calculate_discriminant(a, b)
    if D < 0.0:
        # Numerical safety: clamp tiny negative values to 0
        if D > -1e-12:
            D = 0.0
        else:
            # This should not happen if the caller respects the domain constraints.
            raise ValueError(f"Negative discriminant encountered: D={D} for a={a}, b={b}")
    s = np.sqrt(D)
    root1 = (-a + s) / 2.0 # λ1
    root2 = (-a - s) / 2.0 # λ2
    if root2 > root1:
        root1, root2 = root2, root1
    return root1, root2


def reconstruct_coeffs_from_roots(roots: tuple[float, float] | np.ndarray) -> tuple[float, float] | np.ndarray:
    """
    Reconstruct polynomial coefficients from given roots.
    
    Given roots (λ1, λ2), this function returns the coefficients (a, b) of the 
    monic quadratic polynomial that has these roots:
        (x - λ1)(x - λ2) = x^2 - (λ1 + λ2)x + λ1*λ2
    
    The function uses the convention x^2 + ax + b, so:
        a = -(λ1 + λ2)
        b = λ1 * λ2
    
    Note: λ1 and λ2 correspond to root1 and root2 respectively.
    
    Args:
        roots: Either a tuple (root1, root2) or numpy array of shape (2,) 
               containing the roots in descending order (λ1 ≥ λ2)
        
    Returns:
        Coefficients (a, b) in the same format as input (tuple or numpy array)
    """
    if isinstance(roots, tuple):
        root1, root2 = roots
        return (-(root1 + root2), root1 * root2)
    else:
        # numpy array case
        root1, root2 = roots[0], roots[1]
        return np.array([-(root1 + root2), root1 * root2])


# ==================
# Data record types
# ==================

@dataclass
class OneStepPair:
    """
    A one-step training example for Task 1.1: predicting roots at the next step.
    
    This represents a single step along a homotopy path between two quadratic 
    polynomials. Given quadratic polynomials P (with unknown roots) and Q (with 
    known roots), the homotopy path is defined as:
        H(t) = (1 - t) * Q + t * P
    
    where H(0) = Q and H(1) = P.

    The homotopy path is discretized into M+1 steps (j = 0, 1, ..., M), where each 
    step H(t_j) represents a quadratic polynomial:
        H(t_j) = x² + a_j x + b_j
    
    Each polynomial H(t_j) has roots (λ1_j, λ2_j) with λ1_j ≥ λ2_j.

    Attributes
    ----------
    coeffs_curr: tuple[float, float]
        Coefficients (a_j, b_j) at current step t_j.
    roots_curr: tuple[float, float]
        Roots (λ1_j, λ2_j) at current step t_j.
    coeffs_next: tuple[float, float]
        Next step coefficients (a_{j+1}, b_{j+1}) at next step t_{j+1}.
    roots_next: tuple[float, float]
        Roots (λ1_{j+1}, λ2_{j+1}) at next step t_{j+1}.
    """
    coeffs_curr: tuple[float, float]
    roots_curr: tuple[float, float]
    coeffs_next: tuple[float, float]
    roots_next: tuple[float, float]


@dataclass
class PathRecord:
    """
    A full linear-homotopy path in coefficient space from coeffs_start to coeffs_end.

    Attributes
    ----------
    coeffs_along_path : np.ndarray
        Array of shape (num_steps+1, 2) with polynomial coefficients (a, b)
        along the homotopy path from t=0 to t=1.
    roots_along_path : np.ndarray
        Array of shape (num_steps+1, 2) with the corresponding real roots (λ1, λ2)
        at each discretized t. By convention λ1 >= λ2.
    coeffs_start : tuple[float, float]
        Coefficients (a, b) at t=0. If use_solver_free=True, these are reconstructed.
    coeffs_end : tuple[float, float]
        Coefficients (a, b) at t=1. If use_solver_free=True, these are reconstructed.
    roots_start : tuple[float, float]
        Roots (λ1, λ2) at t=0.
    roots_end : tuple[float, float]
        Roots (λ1, λ2) at t=1.
    """
    coeffs_along_path: np.ndarray
    roots_along_path: np.ndarray
    coeffs_start: tuple[float, float]
    coeffs_end: tuple[float, float]
    roots_start: tuple[float, float]
    roots_end: tuple[float, float]


# =====================
# Dataset configuration
# =====================

@dataclass
class Domain:
    """
    Coefficient domain Ω for quadratic polynomials.
    
    The domain is defined as:
        Ω = { (a, b) ∈ [-B, B]² | b < a²/4 - τ/4 }
    
    where:
        - B is the maximum absolute value for coefficients (max_coeff_abs)
        - τ is the discriminant margin (discriminant_margin > 0)
    
    This domain guarantees that every quadratic polynomial x² + ax + b with 
    coefficients (a, b) ∈ Ω has a strictly positive discriminant:
        D = a² - 4b ≥ τ > 0
    
    Therefore, all such polynomials have two distinct real roots.
    """
    max_coeff_abs: float = 10.0          # B: coefficients constrained to [-B, B]
    discriminant_margin: float = 3.0     # τ: margin away from D=0

    def inside(self, a: float, b: float) -> bool:
        """Return True iff (a, b) lies inside Ω."""
        if abs(a) > self.max_coeff_abs or abs(b) > self.max_coeff_abs:
            return False
        return b < (a * a) / 4.0 - self.discriminant_margin / 4.0


@dataclass
class SamplerCfg:
    """
    Sampling and homotopy configuration for the quadratic case.
    """
    # Required parameters for machine learning
    train_size: int                      # desired number of training pairs (required)
    test_size: int                       # desired number of test pairs (required)
    # Optional parameters with defaults
    num_steps: int = 5                   # number of discretization steps per path (j = 0..num_steps)
    min_coeff_distance: float = 1.0      # lower bound on ||coeffs_end - coeffs_start||_2
    max_coeff_distance: float = 6.0      # upper bound on ||coeffs_end - coeffs_start||_2
    n_check: int = 32                    # #grid points used to verify the entire segment stays in Ω
    root_seed: int | None = 42
    use_solver_free: bool = True
    # If True: reconstruct coefficients from analytic roots for all points on the path.


# =====================
# DatasetGenerator core
# =====================

class DatasetGenerator:
    """
    Generate Task 1.1 data using linear homotopy in the (a, b) coefficient space.

    Pipeline per path
    -----------------
    1) Sampling:
       - Sample coeffs_end ∈ Ω (polynomial P with unknown roots by design).
       - Sample coeffs_start ∈ Ω nearby (polynomial Q). Its roots are considered "known"
         in the sense that we will compute them explicitly with a solver.
    2) Build the linear homotopy H(t) between coeffs_start and coeffs_end and compute
       the roots at all discretized steps using a solver abstraction.
    3) Optional (solver-free flavor): Reconstruct coefficients from the computed roots
       at all steps and use those reconstructed coefficients for the dataset.

    Notes
    -----
    * Roots are computed with the closed-form quadratic formula (no external solver).
    * joblib is used to parallelize per-path sampling.
    * All roots are ordered λ1 >= λ2 for consistency.
    """

    def __init__(self, domain: Domain, cfg: SamplerCfg, n_jobs: int = -1, backend: str = "multiprocessing", verbose: int = 0):
        self.domain = domain
        self.cfg = cfg
        self.n_jobs = n_jobs
        self.backend = backend
        self.verbose = verbose
        self._hc_pairs: list[OneStepPair] = []
        self._train_pairs: list[OneStepPair] = []
        self._test_pairs: list[OneStepPair] = []
        self._hc_paths: list[PathRecord] = []
        self._train_hc_paths: list[PathRecord] = []
        self._test_hc_paths: list[PathRecord] = []

    # --------- Public API ---------

    def generate(self) -> None:
        """
        Generate the dataset according to the current configuration.

        Side effects
        ------------
        Populates:
          - self._hc_paths : list[PathRecord]
          - self._hc_pairs : list[OneStepPair] (Task 1.1 pairs)
        """
        # Sample independent hc_paths in parallel with per-task RNG seeds.
        # Each worker also builds one-step pairs for its path to avoid a second pass.
        # Decide how many hc_paths to sample
        num_hc_paths = self._calc_num_hc_paths()

        results: list[tuple[PathRecord, list[OneStepPair]]] = Parallel(
            n_jobs=self.n_jobs, backend=self.backend, verbose=self.verbose
        )(
            delayed(self._make_single_hc_path)(seed=self.cfg.root_seed + idx)
            for idx in range(num_hc_paths)
        )

        # Unpack results
        self._hc_paths = [r[0] for r in results]
        self._hc_pairs = [p for _, plist in results for p in plist]

        # Always perform path-level split into train/test using requested sizes.
        # This ensures that pairs from the same homotopy path are never split across train/test,
        # preventing data leakage and enabling proper evaluation with independent data.
        self._perform_path_level_split() 

    def get_pairs(self) -> list[OneStepPair]:
        """Return Task 1.1 one-step training pairs."""
        return self._hc_pairs

    def get_hc_paths(self) -> list[PathRecord]:
        """Return per-hc_path time series (coefficients and roots)."""
        return self._hc_paths

    def get_train_pairs(self) -> list[OneStepPair]:
        """Return training pairs after path-level split (empty if not split)."""
        return self._train_pairs

    def get_test_pairs(self) -> list[OneStepPair]:
        """Return test pairs after path-level split (empty if not split)."""
        return self._test_pairs

    def get_train_hc_paths(self) -> list[PathRecord]:
        """Return training homotopy paths after path-level split."""
        return self._train_hc_paths

    def get_test_hc_paths(self) -> list[PathRecord]:
        """Return test homotopy paths after path-level split."""
        return self._test_hc_paths

    # --------- Internals ---------
    def _calc_num_hc_paths(self) -> int:
        """Return the number of homotopy paths to sample.

        Calculate the minimum number of paths needed to satisfy train_size and test_size
        requirements using ceil(size/num_steps).
        """

        train_size = max(int(self.cfg.train_size), 0)
        test_size = max(int(self.cfg.test_size), 0)
        
        if train_size == 0 and test_size == 0:
            raise ValueError("Both train_size and test_size cannot be 0. At least one must be specified.")

        num_steps = max(int(self.cfg.num_steps), 1)
        required_num_hc_paths = (ceil(train_size / num_steps) if train_size else 0) + (ceil(test_size / num_steps) if test_size else 0)
        return max(required_num_hc_paths, 1)

    def _make_single_hc_path(self, seed: int) -> tuple[PathRecord, list[OneStepPair]]:
        """
        Build one homotopy path coeffs_start → coeffs_end inside Ω and its Task 1.1 pairs.

        Steps
        -----
        1) Sample endpoints (coeffs_end first, then a valid coeffs_start near it).
        2) Discretize the linear homotopy and compute roots at all steps via solver.
        3) Optionally reconstruct coefficients from those roots at all steps.
        """
        rng = np.random.default_rng(seed)

        # 1) Sample endpoints
        coeffs_end = self._sample_coeffs(rng)
        coeffs_start = self._sample_coeff_start_near(coeffs_end, rng)

        # 2) Build discretized homotopy and compute roots along the path via solver.
        coeffs_along_path = self._calc_coeffs_along_path(coeffs_start, coeffs_end)
        roots_along_path = np.empty((self.cfg.num_steps + 1, 2), dtype=float)
        for j, coeffs_j in enumerate(coeffs_along_path):
            roots_along_path[j] = calculate_quad_roots(coeffs_j)

        # 3) Optionally reconstruct coefficients at ALL steps from their roots
        if self.cfg.use_solver_free:
            coeffs_reconstructed = np.empty_like(coeffs_along_path)
            for j in range(self.cfg.num_steps + 1):
                roots_j = roots_along_path[j]  # numpy array of shape (2,)
                coeffs_j = reconstruct_coeffs_from_roots(roots_j)
                coeffs_reconstructed[j, 0] = coeffs_j[0]
                coeffs_reconstructed[j, 1] = coeffs_j[1]

            # Update coefficients along homotopy path using reconstructed coefficients
            coeffs_along_path = coeffs_reconstructed
            coeffs_start = (float(coeffs_along_path[0, 0]), float(coeffs_along_path[0, 1]))
            coeffs_end = (float(coeffs_along_path[-1, 0]), float(coeffs_along_path[-1, 1]))

        roots_start = (float(roots_along_path[0, 0]), float(roots_along_path[0, 1]))
        roots_end = (float(roots_along_path[-1, 0]), float(roots_along_path[-1, 1]))

        hc_path = PathRecord(
            coeffs_along_path=coeffs_along_path,
            roots_along_path=roots_along_path,
            coeffs_start=coeffs_start,
            coeffs_end=coeffs_end,
            roots_start=roots_start,
            roots_end=roots_end,
        )

        hc_pairs = self._pick_pairs_from_path(hc_path)
        return hc_path, hc_pairs

    def _calc_coeffs_along_path(self, coeffs_start: tuple[float, float], coeffs_end: tuple[float, float]) -> np.ndarray:
        """
        Calculate coefficients along the homotopy path between polynomials Q 
        (with coefficients `coeffs_start`) and P (with coefficients `coeffs_end`).
        
        Formula: H(t_j) = (1 - t_j) * Q + t_j * P,
        sampled at t_j = j / M for j = 0, 1, ..., M, where M is the number of steps.

        Args:
            coeffs_start: Coefficients of polynomial Q
            coeffs_end: Coefficients of polynomial P

        Returns:
            Array of shape (num_steps+1, 2) with columns [a, b].
        """
        (a_start, b_start), (a_end, b_end) = coeffs_start, coeffs_end
        t = np.linspace(0.0, 1.0, self.cfg.num_steps + 1)
        a = (1.0 - t) * a_start + t * a_end
        b = (1.0 - t) * b_start + t * b_end
        return np.stack([a, b], axis=1)

    def _pick_pairs_from_path(self, hc_path: PathRecord) -> list[OneStepPair]:
        """Pick Task 1.1 one-step pairs from a single hc_path.

        This runs without Python-side math and mostly uses NumPy indexing,
        so it benefits from thread-level parallelism.
        """
        num_steps = self.cfg.num_steps
        coeffs_along_path = hc_path.coeffs_along_path
        roots = hc_path.roots_along_path
        pairs: list[OneStepPair] = []
        for j in range(num_steps):
            coeffs_curr = (float(coeffs_along_path[j, 0]), float(coeffs_along_path[j, 1]))
            roots_curr = (float(roots[j, 0]), float(roots[j, 1]))
            coeffs_next = (float(coeffs_along_path[j + 1, 0]), float(coeffs_along_path[j + 1, 1]))
            roots_next = (float(roots[j + 1, 0]), float(roots[j + 1, 1]))
            pairs.append(OneStepPair(
                coeffs_curr=coeffs_curr,
                roots_curr=roots_curr,
                coeffs_next=coeffs_next,
                roots_next=roots_next,
            ))
        return pairs

    # --------- Helpers ---------
    

    def _perform_path_level_split(self) -> None:
        """Split pairs by whole hc_path units to meet train_size/test_size.

        This method ensures data leakage prevention by maintaining strict separation
        between train and test sets at the hc_path level:
        - hc_path A's information is used only for training
        - hc_path B's information is used only for testing
        - This enables evaluation with completely independent data

        Process:
        - Shuffle hc_paths deterministically from cfg.seed.
        - Convert requested pair sizes to required numbers of hc_paths using ceil(size/num_steps).
        - Assign full paths to train then test; if the final assigned path would exceed
          the requested size, take only the required leading pairs from that path.
        - This avoids leakage across splits because an hc_path never contributes to both
          splits, except possibly the single truncated path at each boundary which is still
          assigned exclusively to one split.
        """
        rng = np.random.default_rng(self.cfg.root_seed)

        # Index hc_paths and associated hc_pairs per hc_path
        pairs_by_path: list[list[OneStepPair]] = []
        idx = 0
        for _ in self._hc_paths:
            pairs_by_path.append(self._hc_pairs[idx: idx + self.cfg.num_steps])
            idx += self.cfg.num_steps

        hc_path_indices = np.arange(len(self._hc_paths)) # e.g., [0, 1, 2, 3, 4] , where len(self._hc_paths) = 5
        rng.shuffle(hc_path_indices)

        remaining_train_size = max(self.cfg.train_size, 0)
        remaining_test_size = max(self.cfg.test_size, 0)

        train_pairs: list[OneStepPair] = []
        test_pairs: list[OneStepPair] = []
        train_hc_paths: list[PathRecord] = []
        test_hc_paths: list[PathRecord] = []

        # Assign to train first, then test, consuming pairs from each path until the remaining size is zero
        for i in hc_path_indices:
            if remaining_train_size > 0:
                selected_pairs, remaining_train_size = self._consume_pairs_until_zero(pairs_by_path[i], remaining_train_size)
                if selected_pairs:
                    train_pairs.extend(selected_pairs)
                    train_hc_paths.append(self._hc_paths[i])
                continue
            if remaining_test_size > 0:
                selected_pairs, remaining_test_size = self._consume_pairs_until_zero(pairs_by_path[i], remaining_test_size)
                if selected_pairs:
                    test_pairs.extend(selected_pairs)
                    test_hc_paths.append(self._hc_paths[i])
                continue
            break

        self._train_pairs = train_pairs
        self._test_pairs = test_pairs
        self._train_hc_paths = train_hc_paths
        self._test_hc_paths = test_hc_paths

    def _consume_pairs_until_zero(self, pairs_by_path: list[OneStepPair], remain: int) -> tuple[list[OneStepPair], int]:
        """
        Select pairs from a single homotopy path until `remain` reaches zero.

        This function takes pairs from the beginning of `pairs_by_path` until either
        all pairs are taken or `remain` reaches zero. It never reorders elements.

        Args:
            pairs_by_path: Pairs belonging to one homotopy path (ordered along t_j).
            remain: Remaining number of pairs needed (non-negative).

        Returns:
            A tuple (selected_pairs, updated_remain) where:
              - selected_pairs is a list of pairs taken from the start of `pairs_by_path`.
              - updated_remain is the remaining count after consuming selected_pairs.
        """
        if remain <= 0:
            return [], 0
        if len(pairs_by_path) <= remain:
            return pairs_by_path, remain - len(pairs_by_path)
        return pairs_by_path[:remain], 0

    def _sample_coeffs(self, rng: np.random.Generator, max_attempts: int = 20_000) -> tuple[float, float]:
        """
        Rejection-sample a coefficient vector (a, b) uniformly over the area of Ω.

        Algorithm:
        - Draw (a, b) uniformly from the square [-B, B]^2 where B = domain.max_coeff_abs.
        - Accept the sample if (a, b) lies inside Ω, i.e., Domain.inside(a, b) is True;
          otherwise, resample up to `max_attempts` times.
        This procedure yields an area-uniform distribution on Ω.
        """
        B = self.domain.max_coeff_abs
        for _ in range(max_attempts):
            a = rng.uniform(-B, B)
            b = rng.uniform(-B, B)
            if self.domain.inside(a, b):
                return a, b
        raise RuntimeError("Rejection sampling for Ω did not converge. Consider relaxing the domain.")

    def _sample_coeff_start_near(self, coeffs_end: tuple[float, float], rng: np.random.Generator,
                                 max_attempts: int = 30_000) -> tuple[float, float]:
        """
        Sample coeffs_start ∈ Ω such that:
          - min_coeff_distance ≤ ||coeffs_end - coeffs_start||_2 ≤ max_coeff_distance,
          - the straight segment between coeffs_start and coeffs_end stays inside Ω
            (verified on a uniform grid of n_check points).
        """
        for _ in range(max_attempts):
            coeffs_start = self._sample_coeffs(rng)
            d = np.hypot(coeffs_end[0] - coeffs_start[0], coeffs_end[1] - coeffs_start[1]) # L2 distance between endpoints
            if not (self.cfg.min_coeff_distance <= d <= self.cfg.max_coeff_distance):
                continue
            if self._is_path_valid(coeffs_start, coeffs_end):
                return coeffs_start
        raise RuntimeError("Failed to find a valid coeffs_start given coeffs_end. "
                           "Consider relaxing distance/path constraints.")

    def _is_path_valid(self, coeffs_start: tuple[float, float], coeffs_end: tuple[float, float]) -> bool:
        """
        Check if the homotopy path between polynomial Q (with coefficients coeffs_start) 
        and polynomial P (with coefficients coeffs_end) remains entirely inside 
        the domain Ω (verified on a uniform grid of n_check points).
        """
        (a_start, b_start), (a_end, b_end) = coeffs_start, coeffs_end
        ts = np.linspace(0.0, 1.0, self.cfg.n_check)
        for t in ts:
            a = (1.0 - t) * a_start + t * a_end
            b = (1.0 - t) * b_start + t * b_end
            if not self.domain.inside(a, b):
                return False
        return True

    


# ==============
# Example usage
# ==============
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Task 1.1 dataset and save outputs")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save datasets and plots")
    parser.add_argument("--save-hc-paths", action="store_true", help="Also save full HC paths as jsonl")
    parser.add_argument("--save-hc-endpoints", action="store_true", help="Also save HC endpoints as txt and jsonl")
    # Required parameters for machine learning
    parser.add_argument("--train-size", type=int, required=True, help="Number of training pairs (required)")
    parser.add_argument("--test-size", type=int, required=True, help="Number of test pairs (required)")
    # Optional overrides for quick experimentation
    parser.add_argument("--num-hc-steps", type=int, default=5)
    parser.add_argument("--min-dist", type=float, default=1.0)
    parser.add_argument("--max-dist", type=float, default=6.0)
    parser.add_argument("--n-check", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--solver-free", action="store_true", help="Reconstruct coefficients from analytic roots for all path points")
    parser.add_argument("--no-solver-free", dest="solver_free", action="store_false")
    parser.set_defaults(solver_free=True)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--backend", type=str, default="multiprocessing")
    parser.add_argument("--verbose", type=int, default=0)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    domain = Domain(max_coeff_abs=10.0, discriminant_margin=3.0)
    cfg = SamplerCfg(
        num_steps=args.num_hc_steps,
        min_coeff_distance=args.min_dist,
        max_coeff_distance=args.max_dist,
        n_check=args.n_check,
        root_seed=args.seed,
        use_solver_free=args.solver_free,
        train_size=args.train_size,
        test_size=args.test_size,
    )

    dataset_generator = DatasetGenerator(
        domain=domain,
        cfg=cfg,
        n_jobs=args.n_jobs,
        backend=args.backend,
        verbose=args.verbose,
    )

    dataset_generator.generate()

    # Get train/test split data
    train_hc_pairs = dataset_generator.get_train_pairs()
    test_hc_pairs = dataset_generator.get_test_pairs()
    train_hc_paths = dataset_generator.get_train_hc_paths()
    test_hc_paths = dataset_generator.get_test_hc_paths()

    print(f"#hc_paths: train={len(train_hc_paths)}, test={len(test_hc_paths)}")
    print(f"#pairs (Task 1.1): train={len(train_hc_pairs)}, test={len(test_hc_pairs)}")
    
    if train_hc_pairs:
        print("example train pair:", train_hc_pairs[0])
    if test_hc_pairs:
        print("example test pair:", test_hc_pairs[0])

    # Save datasets
    writer = DatasetWriter(output_dir=args.output_dir)
    
    # Save train/test splits
    if train_hc_pairs:
        train_txt, train_jsonl = writer.save_pairs(train_hc_pairs, base_filename="hc_pairs_train")
        print(f"Saved train pairs:")
        print(f"  - {train_txt} ({len(train_hc_pairs)} pairs)")
        print(f"  - {train_jsonl} ({len(train_hc_pairs)} pairs)")
    
    if test_hc_pairs:
        test_txt, test_jsonl = writer.save_pairs(test_hc_pairs, base_filename="hc_pairs_test")
        print(f"Saved test pairs:")
        print(f"  - {test_txt} ({len(test_hc_pairs)} pairs)")
        print(f"  - {test_jsonl} ({len(test_hc_pairs)} pairs)")

    if args.save_hc_paths:
        # Save train/test HC paths separately
        if train_hc_paths:
            train_jsonl = writer.save_hc_paths(train_hc_paths, filename="hc_paths_train.jsonl")
            print(f"Saved train HC paths: {train_jsonl}")
        
        if test_hc_paths:
            test_jsonl = writer.save_hc_paths(test_hc_paths, filename="hc_paths_test.jsonl")
            print(f"Saved test HC paths: {test_jsonl}")

    if args.save_hc_endpoints:
        # Save train/test endpoints separately
        if train_hc_paths:
            train_txt, train_jsonl = writer.save_hc_endpoints(train_hc_paths, base_filename="hc_endpoints_train")
            print(f"Saved train HC endpoints:")
            print(f"  - {train_txt}")
            print(f"  - {train_jsonl}")
        
        if test_hc_paths:
            test_txt, test_jsonl = writer.save_hc_endpoints(test_hc_paths, base_filename="hc_endpoints_test")
            print(f"Saved test HC endpoints:")
            print(f"  - {test_txt}")
            print(f"  - {test_jsonl}")

    # Visualizations
    viz = DatasetVisualizer(max_coeff_abs=domain.max_coeff_abs, discriminant_margin=domain.discriminant_margin)
    plot_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot train data (full homotopy paths)
    if train_hc_paths:
        train_roots_png = viz.plot_roots_paths(train_hc_paths, plot_dir, filename="hc_roots_paths_train.png")
        train_coeffs_png = viz.plot_coeffs_paths(train_hc_paths, plot_dir, filename="hc_coefficients_paths_train.png")
        if train_roots_png:
            print(f"Saved train plot: {train_roots_png}")
        if train_coeffs_png:
            print(f"Saved train plot: {train_coeffs_png}")
    
    # Plot test data (full homotopy paths)
    if test_hc_paths:
        test_roots_png = viz.plot_roots_paths(test_hc_paths, plot_dir, filename="hc_roots_paths_test.png")
        test_coeffs_png = viz.plot_coeffs_paths(test_hc_paths, plot_dir, filename="hc_coefficients_paths_test.png")
        if test_roots_png:
            print(f"Saved test plot: {test_roots_png}")
        if test_coeffs_png:
            print(f"Saved test plot: {test_coeffs_png}")
    
    # Plot HC paths (start/end points from PathRecord)
    if train_hc_paths:
        train_hc_roots_png = viz.plot_roots_endpoints(train_hc_paths, plot_dir, filename="hc_roots_endpoints_train.png")
        train_hc_coeffs_png = viz.plot_coeffs_endpoints(train_hc_paths, plot_dir, filename="hc_coefficients_endpoints_train.png")
        if train_hc_roots_png:
            print(f"Saved train HC plot: {train_hc_roots_png}")
        if train_hc_coeffs_png:
            print(f"Saved train HC plot: {train_hc_coeffs_png}")
    
    if test_hc_paths:
        test_hc_roots_png = viz.plot_roots_endpoints(test_hc_paths, plot_dir, filename="hc_roots_endpoints_test.png")
        test_hc_coeffs_png = viz.plot_coeffs_endpoints(test_hc_paths, plot_dir, filename="hc_coefficients_endpoints_test.png")
        if test_hc_roots_png:
            print(f"Saved test HC plot: {test_hc_roots_png}")
        if test_hc_coeffs_png:
            print(f"Saved test HC plot: {test_hc_coeffs_png}")
        