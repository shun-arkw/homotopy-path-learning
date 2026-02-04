# dataset_generator2.py (split-generation version)
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
    """Return the discriminant D = a^2 - 4b for the quadratic polynomial x^2 + a x + b."""
    return a * a - 4.0 * b


def calculate_quad_roots(coeffs: tuple[float, float]) -> tuple[float, float]:
    """
    Calculate the real roots (root1, root2) of x^2 + a x + b = 0
    assuming D >= 0 by construction.

    The roots are returned in descending order: root1 >= root2.
    If a tiny negative discriminant appears due to roundoff, it is clamped to 0.
    
    Args:
        coeffs: Tuple (a, b) representing the quadratic coefficients
        
    Returns:
        Tuple (root1, root2) with root1 >= root2
    """
    a, b = coeffs
    D = calculate_discriminant(a, b)
    if D < 0.0:
        if D > -1e-12:
            D = 0.0
        else:
            raise ValueError(f"Negative discriminant encountered: D={D} for a={a}, b={b}")
    s = np.sqrt(D)
    root1 = (-a + s) / 2.0
    root2 = (-a - s) / 2.0
    if root2 > root1:
        root1, root2 = root2, root1
    return root1, root2


def reconstruct_coeffs_from_roots(roots: tuple[float, float] | np.ndarray) -> tuple[float, float] | np.ndarray:
    """
    Given roots (root1, root2), return the coefficients (a, b) of the monic quadratic:
        (x - root1)(x - root2) = x^2 - (root1 + root2) x + root1*root2.
    We use the convention x^2 + a x + b, hence:
        a = -(root1 + root2),  b = root1 * root2.
    """
    if isinstance(roots, tuple):
        root1, root2 = roots
        return (-(root1 + root2), root1 * root2)
    else:
        root1, root2 = roots[0], roots[1]
        return np.array([-(root1 + root2), root1 * root2])


# ==================
# Data record types
# ==================

@dataclass
class OneStepPair:
    """
    A Task 1.1 one-step training example.

    Input  : (H(t_j) = (a_j, b_j), roots at t_j = (λ1_j, λ2_j), H(t_{j+1}) = (a_{j+1}, b_{j+1}))
    Target : roots at next step t_{j+1} = (λ1_{j+1}, λ2_{j+1})

    All tuples use the convention λ1 >= λ2.
    """
    coeffs_curr: tuple[float, float]
    roots_curr: tuple[float, float]
    coeffs_next: tuple[float, float]
    roots_next: tuple[float, float]


@dataclass
class PathRecord:
    """
    A full linear-homotopy path in coefficient space from coeffs_start to coeffs_end.
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
    Coefficient domain Ω for quadratics:
        Ω = { (a, b) ∈ [-B, B]^2 | b < a^2/4 - τ/4 },  with τ > 0 (discriminant margin).

    This guarantees a strictly positive discriminant D = a^2 - 4b >= τ > 0
    for every point of Ω, i.e., two distinct real roots everywhere in Ω.
    """
    max_coeff_abs: float = 10.0
    discriminant_margin: float = 3.0

    def inside(self, a: float, b: float) -> bool:
        if abs(a) > self.max_coeff_abs or abs(b) > self.max_coeff_abs:
            return False
        return b < (a * a) / 4.0 - self.discriminant_margin / 4.0


@dataclass
class SamplerCfg:
    """Sampling and homotopy configuration for the quadratic case."""
    train_size: int
    test_size: int
    num_steps: int = 5
    min_coeff_distance: float = 1.0
    max_coeff_distance: float = 6.0
    n_check: int = 32
    root_seed: int | None = 42
    use_solver_free: bool = True


# =====================
# DatasetGenerator core (split generation)
# =====================

class DatasetGenerator:
    """
    Split-generation version: independently generate train and test sets using
    separate RNG seeds while keeping the same I/O interface as the original.
    """

    def __init__(self, domain: Domain, cfg: SamplerCfg, n_jobs: int = -1, backend: str = "multiprocessing", verbose: int = 0):
        self.domain = domain
        self.cfg = cfg
        self.n_jobs = n_jobs
        self.backend = backend
        self.verbose = verbose

    # --------- Public API ---------

    def generate_set(self, *, size: int, base_seed: int) -> tuple[list[OneStepPair], list[PathRecord]]:
        """
        Generate a dataset of the requested number of pairs by sampling whole paths
        and concatenating their one-step pairs until the requested size is met.
        """
        if size <= 0:
            return [], []

        steps = max(int(self.cfg.num_steps), 1)
        num_paths = ceil(size / steps)

        results: list[tuple[PathRecord, list[OneStepPair]]] = Parallel(
            n_jobs=self.n_jobs, backend=self.backend, verbose=self.verbose
        )(
            delayed(self._make_single_hc_path)(seed=base_seed + idx)
            for idx in range(num_paths)
        )

        hc_paths: list[PathRecord] = [r[0] for r in results]
        all_pairs: list[OneStepPair] = [p for _, plist in results for p in plist]

        # Truncate pairs to requested size; keep the minimal number of paths that cover them
        pairs = all_pairs[:size]
        used_path_count = ceil(len(pairs) / steps)
        paths_used = hc_paths[:used_path_count]
        return pairs, paths_used

    # --------- Internals ---------

    def _make_single_hc_path(self, seed: int) -> tuple[PathRecord, list[OneStepPair]]:
        rng = np.random.default_rng(seed)

        coeffs_end = self._sample_coeffs(rng)
        coeffs_start = self._sample_coeff_start_near(coeffs_end, rng)

        coeffs_along_path = self._linear_homotopy(coeffs_start, coeffs_end)
        roots_along_path = np.empty((self.cfg.num_steps + 1, 2), dtype=float)
        for j, coeffs_j in enumerate(coeffs_along_path):
            roots_along_path[j] = calculate_quad_roots(coeffs_j)

        if self.cfg.use_solver_free:
            coeffs_reconstructed = np.empty_like(coeffs_along_path)
            for j in range(self.cfg.num_steps + 1):
                roots_j = roots_along_path[j]
                coeffs_j = reconstruct_coeffs_from_roots(roots_j)
                coeffs_reconstructed[j, 0] = coeffs_j[0]
                coeffs_reconstructed[j, 1] = coeffs_j[1]
            coeffs_along_path = coeffs_reconstructed

        if self.cfg.use_solver_free:
            coeffs_start_out = (float(coeffs_along_path[0, 0]), float(coeffs_along_path[0, 1]))
            coeffs_end_out = (float(coeffs_along_path[-1, 0]), float(coeffs_along_path[-1, 1]))
        else:
            coeffs_start_out = coeffs_start
            coeffs_end_out = coeffs_end

        roots_start = (float(roots_along_path[0, 0]), float(roots_along_path[0, 1]))
        roots_end = (float(roots_along_path[-1, 0]), float(roots_along_path[-1, 1]))

        path_record = PathRecord(
            coeffs_along_path=coeffs_along_path,
            roots_along_path=roots_along_path,
            coeffs_start=coeffs_start_out,
            coeffs_end=coeffs_end_out,
            roots_start=roots_start,
            roots_end=roots_end,
        )

        pairs = self._pairs_from_hc_path(path_record)
        return path_record, pairs

    def _pairs_from_hc_path(self, hc_path: PathRecord) -> list[OneStepPair]:
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

    def _sample_coeffs(self, rng: np.random.Generator, max_attempts: int = 20_000) -> tuple[float, float]:
        B = self.domain.max_coeff_abs
        for _ in range(max_attempts):
            a = rng.uniform(-B, B)
            b = rng.uniform(-B, B)
            if self.domain.inside(a, b):
                return a, b
        raise RuntimeError("Rejection sampling for Ω did not converge. Consider relaxing the domain.")

    def _sample_coeff_start_near(self, coeffs_end: tuple[float, float], rng: np.random.Generator,
                                 max_attempts: int = 30_000) -> tuple[float, float]:
        for _ in range(max_attempts):
            coeffs_start = self._sample_coeffs(rng)
            d = np.hypot(coeffs_end[0] - coeffs_start[0], coeffs_end[1] - coeffs_start[1])
            if not (self.cfg.min_coeff_distance <= d <= self.cfg.max_coeff_distance):
                continue
            if self._segment_inside(coeffs_start, coeffs_end):
                return coeffs_start
        raise RuntimeError("Failed to find a valid coeffs_start given coeffs_end. "
                           "Consider relaxing distance/path constraints.")

    def _segment_inside(self, coeffs_start: tuple[float, float], coeffs_end: tuple[float, float]) -> bool:
        (as_, bs), (ae, be) = coeffs_start, coeffs_end
        ts = np.linspace(0.0, 1.0, self.cfg.n_check)
        for t in ts:
            a = (1.0 - t) * as_ + t * ae
            b = (1.0 - t) * bs  + t * be
            if not self.domain.inside(a, b):
                return False
        return True

    def _linear_homotopy(self, coeffs_start: tuple[float, float], coeffs_end: tuple[float, float]) -> np.ndarray:
        (as_, bs), (ae, be) = coeffs_start, coeffs_end
        t = np.linspace(0.0, 1.0, self.cfg.num_steps + 1)
        a = (1.0 - t) * as_ + t * ae
        b = (1.0 - t) * bs  + t * be
        return np.stack([a, b], axis=1)


# ==============
# Example usage (CLI is identical to dataset_generator.py)
# ==============
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Task 1.1 dataset and save outputs (split generation)")
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

    # Derive independent seeds for train and test from a single base seed
    # to guarantee independence while keeping reproducibility simple.
    seed_base = int(args.seed)
    # Use simple deterministic offsets (keeps CLI output identical to original)
    train_seed = seed_base * 2 + 1
    test_seed = seed_base * 2 + 2

    train_hc_pairs, train_hc_paths = dataset_generator.generate_set(size=args.train_size, base_seed=train_seed)
    test_hc_pairs, test_hc_paths = dataset_generator.generate_set(size=args.test_size, base_seed=test_seed)

    print(f"#hc_paths: train={len(train_hc_paths)}, test={len(test_hc_paths)}")
    print(f"#pairs (Task 1.1): train={len(train_hc_pairs)}, test={len(test_hc_pairs)}")
    
    if train_hc_pairs:
        print("example train pair:", train_hc_pairs[0])
    if test_hc_pairs:
        print("example test pair:", test_hc_pairs[0])

    writer = DatasetWriter(output_dir=args.output_dir)
    
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
        if train_hc_paths:
            train_jsonl = writer.save_hc_paths(train_hc_paths, filename="hc_paths_train.jsonl")
            print(f"Saved train HC paths: {train_jsonl}")
        
        if test_hc_paths:
            test_jsonl = writer.save_hc_paths(test_hc_paths, filename="hc_paths_test.jsonl")
            print(f"Saved test HC paths: {test_jsonl}")

    if args.save_hc_endpoints:
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

    viz = DatasetVisualizer(max_coeff_abs=domain.max_coeff_abs, discriminant_margin=domain.discriminant_margin)
    plot_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    if train_hc_paths:
        train_roots_png = viz.plot_roots_paths(train_hc_paths, plot_dir, filename="hc_roots_paths_train.png")
        train_coeffs_png = viz.plot_coeffs_paths(train_hc_paths, plot_dir, filename="hc_coefficients_paths_train.png")
        if train_roots_png:
            print(f"Saved train plot: {train_roots_png}")
        if train_coeffs_png:
            print(f"Saved train plot: {train_coeffs_png}")
    
    if test_hc_paths:
        test_roots_png = viz.plot_roots_paths(test_hc_paths, plot_dir, filename="hc_roots_paths_test.png")
        test_coeffs_png = viz.plot_coeffs_paths(test_hc_paths, plot_dir, filename="hc_coefficients_paths_test.png")
        if test_roots_png:
            print(f"Saved test plot: {test_roots_png}")
        if test_coeffs_png:
            print(f"Saved test plot: {test_coeffs_png}")
    
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


