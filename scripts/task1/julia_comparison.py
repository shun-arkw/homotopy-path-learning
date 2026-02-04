import os
import time
import logging
from typing import List, Tuple, Dict

# Preload juliacall BEFORE torch to avoid potential segfaults when this module is imported in workers
try:
    from juliacall import Main as _jl_preload  # noqa: F401
except Exception:
    _jl_preload = None  # Julia may be unavailable; handled at runtime

import torch
from joblib import Parallel, delayed

from utils import mse_sorted, evaluate_accuracy_within_threshold


class JuliaHomotopyComparison:
    """Thin wrapper around HomotopyContinuation.jl for quadratic homotopy tracking.

    Responsibilities:
        - Lazily initialize Julia and required packages.
        - Define the Julia homotopy solver function once for reuse.
        - Track a linear homotopy from t=0 to t=1 between two quadratics.
        - Compare Julia solutions against ML predictions using RMSE metrics.

    This module is imported only when comparison is requested to avoid
    introducing Julia dependencies into unrelated workflows.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.julia_initialized = False
        self.jl = None

    def _initialize_julia(self) -> None:
        if self.julia_initialized:
            return
        try:
            from juliacall import Main as jl  # lazy import
            self.jl = jl
        except ImportError:
            self.logger.warning("JuliaCall not available. Julia comparison disabled.")
            self.jl = None
            self.julia_initialized = False
            return

        try:
            self.logger.info("Initializing Julia and defining solver (one-time)...")
            t0 = time.time()
            self.jl.seval("using HomotopyContinuation, DynamicPolynomials")
            # Define the solver function once. Guard redefinition for repeated runs.
            self.jl.seval(
                """
                if !isdefined(Main, :hc_solve_quadratic)
                    function hc_solve_quadratic(a0::Float64, b0::Float64, r1::Float64, r2::Float64, a1::Float64, b1::Float64)
                        @polyvar x t
                        Q = [x^2 + a0*x + b0]
                        P = [x^2 + a1*x + b1]
                        H  = (1 - t) .* Q .+ t .* P
                        Hsys = System(H; variables=[x], parameters=[t])
                        starts = [[r1+0im], [r2+0im]]
                        opts = HomotopyContinuation.TrackerOptions(;
                            max_steps=50_000,
                            max_step_size=0.05,
                            max_initial_step_size=0.05,
                            min_step_size=1e-10,
                            min_rel_step_size=1e-10,
                            extended_precision=true,
                        )
                        return solve(Hsys, starts; start_parameters=[0.0], target_parameters=[1.0], tracker_options=opts)
                    end
                end
                """
            )
            t1 = time.time()
            self.logger.info(f"Julia initialization completed in {t1 - t0:.3f}s")
            self.julia_initialized = True
        except Exception as e:
            self.logger.error(f"Failed to initialize Julia: {e}")
            self.julia_initialized = False

    def solve_quadratic_homotopy(
        self,
        a_start: float,
        b_start: float,
        r1_start: float,
        r2_start: float,
        a_end: float,
        b_end: float,
        num_steps: int = 10,
    ) -> Tuple[List[float], float]:
        """
        Solve a linear homotopy between two quadratics from t=0 to t=1.

        Args:
            a_start: Linear coefficient of the start polynomial.
            b_start: Constant term of the start polynomial.
            r1_start: Root 1 of the start polynomial (r1_start >= r2_start).
            r2_start: Root 2 of the start polynomial (r1_start >= r2_start).
            a_end: Linear coefficient of the target polynomial.
            b_end: Constant term of the target polynomial.
            num_steps: Unused, kept for API compatibility.

        Returns:
            A tuple (real_roots_at_t1, elapsed_seconds).
        """
        self._initialize_julia()
        if not self.julia_initialized or self.jl is None:
            return [], 0.0

        try:
            t0 = time.time()
            result = self.jl.hc_solve_quadratic(a_start, b_start, r1_start, r2_start, a_end, b_end)
            t1 = time.time()

            real_solutions = self.jl.seval("real_solutions")(result)

            roots: List[float] = []
            if real_solutions is not None and len(real_solutions) > 0:
                for sol in real_solutions:
                    if len(sol) > 0:
                        try:
                            roots.append(float(sol[0]))
                        except (ValueError, TypeError):
                            pass

            if len(roots) == 0:
                all_solutions = self.jl.seval("solutions")(result)
                if all_solutions is not None and len(all_solutions) > 0:
                    has_complex = False
                    for sol in all_solutions:
                        if len(sol) > 0:
                            try:
                                if hasattr(sol[0], 'imag') and abs(float(sol[0].imag)) > 1e-10:
                                    has_complex = True
                                    break
                            except (ValueError, TypeError, AttributeError):
                                pass
                    if has_complex:
                        self.logger.warning(
                            f"Only complex solutions found for homotopy a_start={a_start}, b_start={b_start}, a_end={a_end}, b_end={b_end}, skipping this sample"
                        )
                        return [], t1 - t0

            return roots, t1 - t0

        except Exception as e:
            self.logger.error(
                f"Julia homotopy solve failed for a_start={a_start}, b_start={b_start}, a_end={a_end}, b_end={b_end}: {e}"
            )
            return [], 0.0

    def compare_with_julia(
        self,
        samples: List[List[float]],
        targets: List[List[float]],
        thresholds: List[float],
        num_steps: int = 10,
        max_samples: int | None = None,
        n_jobs: int | None = None,
        backend: str | None = None,
        prefer: str | None = None,
        verbose: int | None = None,
    ) -> Dict:
        """
        Compare ML predictions with Julia HomotopyContinuation results.

        Args:
            samples: [a_start, b_start, r1_start, r2_start, a_end, b_end]
            targets: [r1_end, r2_end]
            thresholds: RMSE thresholds for accuracy calculation
            num_steps: Unused by Julia, kept for API compatibility
            max_samples: Max number of samples to process (None for all)
        """
        self._initialize_julia()
        if not self.julia_initialized or self.jl is None:
            self.logger.warning("Julia not available, skipping comparison")
            return {}

        self.logger.info("=== Julia HomotopyContinuation Comparison ===")

        if max_samples is not None:
            samples = samples[:max_samples]
            targets = targets[:max_samples]

        julia_roots: List[List[float]] = []
        julia_times: List[float] = []
        valid_targets: List[List[float]] = []

        total_julia_time = 0.0
        num_skipped = 0
        num_complex_skipped = 0

        # Parallel or sequential execution
        if max_samples is None:
            work_samples = samples
            work_targets = targets
        else:
            work_samples = samples[:max_samples]
            work_targets = targets[:max_samples]

        # Resolve parallel settings from CLI args only (no env fallback)
        if n_jobs is None:
            n_jobs = 1
        if backend is None:
            backend = "loky"
        if prefer is None:
            prefer = "processes"
        if verbose is None:
            verbose = 0

        if n_jobs == 1:
            results = [
                self.solve_quadratic_homotopy(a0, b0, r1, r2, a1, b1, num_steps)
                for (a0, b0, r1, r2, a1, b1) in work_samples
            ]
        else:
            results = Parallel(n_jobs=n_jobs, backend=backend, prefer=prefer, verbose=verbose)(
                delayed(_solve_worker)(inp, num_steps) for inp in work_samples
            )

        for (roots, solve_time), gt in zip(results, work_targets):
            if len(roots) == 2:
                roots_sorted = sorted(roots, reverse=True)
                julia_roots.append(roots_sorted)
                julia_times.append(solve_time)
                valid_targets.append(gt)
                total_julia_time += solve_time
            else:
                num_skipped += 1
                if len(roots) == 0:
                    num_complex_skipped += 1

        if not julia_roots:
            self.logger.warning("No valid Julia solutions found")
            return {}

        julia_tensor = torch.tensor(julia_roots, dtype=torch.float32)
        gt_tensor = torch.tensor(valid_targets, dtype=torch.float32)
        # Overall RMSE: match training code by using mse_sorted (which sorts ASC internally)
        julia_rmse = torch.sqrt(mse_sorted(julia_tensor, gt_tensor)).item()
        julia_accuracies: Dict[float, float] = {}
        for threshold in thresholds:
            acc = evaluate_accuracy_within_threshold(julia_tensor, gt_tensor, threshold)
            julia_accuracies[threshold] = acc

        self.logger.info(f"Julia samples processed: {len(julia_roots)} / {len(samples)}")
        self.logger.info(f"Julia skipped: {num_skipped} (complex: {num_complex_skipped})")
        self.logger.info(f"Julia RMSE(sorted): {julia_rmse:.6f}")
        for threshold in sorted(thresholds, reverse=True):
            self.logger.info(f"Julia RMSE_Accuracy(threshold={threshold}): {julia_accuracies[threshold]:.2f}%")
        self.logger.info(f"Julia total solve time: {total_julia_time:.3f}s")
        if len(julia_roots) > 0:
            self.logger.info(f"Julia average solve time: {total_julia_time/len(julia_roots):.6f}s")

        return {
            'julia_rmse': julia_rmse,
            'julia_accuracies': julia_accuracies,
            'julia_times': julia_times,
            'total_julia_time': total_julia_time,
            'num_processed': len(julia_roots),
            'num_skipped': num_skipped,
            'num_complex_skipped': num_complex_skipped,
        }

# -------------------------------
# Parallel execution helpers
# -------------------------------

_COMPARATOR = None  # process-local singleton


class _NullLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


def _get_comparator_for_process():
    global _COMPARATOR
    if _COMPARATOR is None:
        _COMPARATOR = JuliaHomotopyComparison(_NullLogger())
    return _COMPARATOR


def _solve_worker(sample: List[float], num_steps: int) -> Tuple[List[float], float]:
    a_start, b_start, r1_start, r2_start, a_end, b_end = sample
    comp = _get_comparator_for_process()
    return comp.solve_quadratic_homotopy(a_start, b_start, r1_start, r2_start, a_end, b_end, num_steps)


