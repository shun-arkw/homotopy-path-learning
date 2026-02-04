# -------------------------------------------------------------------
# Task 1.2: Implement a multi-steps polynomial root-finding algorithm 
#           based on machine learning along a fixed homotopy path.
# -------------------------------------------------------------------

"""Multi-steps polynomial root finding (Task 1.2).

Julia comparison is implemented in julia_comparison.py and imported lazily
only when requested to avoid unconditional Julia dependency.
"""

# Import juliacall BEFORE torch when Julia comparison is requested to avoid
# potential segfaults (see PyTorch issue #78829).
import sys
if any(arg == "--compare-julia" for arg in sys.argv):
    try:
        from juliacall import Main as _jl  # pre-load Julia runtime safely
    except Exception:
        # If Julia is not available, we simply proceed; comparison will be skipped
        pass

import argparse
import os
import random
import logging
import time
from datetime import datetime
import pytz
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import (
    DataLoader,
    mse_sorted,
    load_model_and_scaler,
    evaluate_accuracy_within_threshold,
)
from one_step_root_finding import ModelTrainer


def setup_logging(save_path: str, timezone: str = 'Europe/Paris') -> tuple[logging.Logger, datetime]:
    os.makedirs(save_path, exist_ok=True)
    tz = pytz.timezone(timezone)
    now_local = datetime.now(tz)
    timestamp = now_local.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(save_path, f"multisteps_log_{timestamp}.log")

    class CleanFormatter(logging.Formatter):
        def format(self, record):
            # For all messages, just show the message without timestamp
            return record.getMessage()

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(CleanFormatter())
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])
    return logging.getLogger(__name__), now_local
    

class HomotopyRootFinder:
    """Multi-step polynomial root finding using homotopy continuation method."""
    
    def __init__(self, trainer: ModelTrainer, num_steps: int = 10):
        """
        Initialize the homotopy root finder.
        
        Args:
            trainer: ModelTrainer instance for one-step predictions
            num_steps: Number of steps for homotopy path (num_steps >= 1)
        """
        self.trainer = trainer
        self.num_steps = num_steps
    
    def _linear_homotopy_coeffs(self, a_start: float, b_start: float, a_end: float, b_end: float, t: float) -> tuple[float, float]:
        """Calculate linear homotopy coefficients at parameter t.

        Args:
            a_start: Starting quadratic coefficient a
            b_start: Starting quadratic coefficient b
            a_end: Ending quadratic coefficient a
            b_end: Ending quadratic coefficient b
            t: Parameter value in [0, 1]

        Returns:
            Tuple (a_t, b_t) with the linear homotopy coefficients at parameter t.
        """
        a_t = (1.0 - t) * a_start + t * a_end
        b_t = (1.0 - t) * b_start + t * b_end
        return a_t, b_t
    
    def _discriminant(self, a: float, b: float) -> float:
        """Calculate discriminant for quadratic polynomial x^2 + a*x + b."""
        return a * a - 4.0 * b
    
    def _normalize_roots_desc(self, r1: float, r2: float) -> tuple[float, float, bool]:
        """Return roots in descending order (λ1 >= λ2); bool indicates reordering."""
        if r1 >= r2:
            return r1, r2, False
        return r2, r1, True
    
    def find_roots(
        self,
        samples: list[list[float]],
        targets: list[list[float]],
        mean: torch.Tensor,
        std: torch.Tensor,
        skip_if_nonreal: bool,
        save_trajectories: bool,
        save_path: str,
        logger: logging.Logger,
        thresholds: list[float],
    ):
        """
        Find roots using multi-step homotopy continuation method.
        
        Args:
            samples: List of input samples [a_start, b_start, r1_start, r2_start, a_end, b_end]
                - a_start, b_start: Starting quadratic coefficients x^2 + a x + b
                - r1_start, r2_start: Starting roots (descending: r1_start ≥ r2_start)
                - a_end, b_end: Ending quadratic coefficients x^2 + a x + b
            targets: List of target roots [r1_end, r2_end] (final roots at t=1)
            mean: Mean values for input normalization
            std: Standard deviation values for input normalization
            skip_if_nonreal: Skip samples if intermediate discriminant <= 0
            save_trajectories: Save trajectory files for each sample
            save_path: Directory to save results
            logger: Logger instance
            thresholds: List of RMSE thresholds for accuracy calculation
        """
        assert self.num_steps >= 1

        # Generate homotopy parameter values
        t_values = []
        for i in range(self.num_steps + 1):
            t = i / self.num_steps
            t_values.append(t)

        preds_final = []
        gts = []
        num_skipped_nonreal = 0
        start_time = time.time()

        if save_trajectories:
            os.makedirs(os.path.join(save_path, "trajectories"), exist_ok=True)

        # Process each sample
        for idx, (inp, gt) in enumerate(zip(samples, targets)):
            a_start, b_start, r1_start, r2_start, a_end, b_end = inp
            # Ground truth kept as ascending list; we will compare with sorted predictions in metrics
            gt_sorted = sorted(gt)

            # Initial state at t=0
            a_curr, b_curr = a_start, b_start
            # Ensure initial roots follow the convention (lambda1 >= lambda2)
            r1_curr, r2_curr, reordered0 = self._normalize_roots_desc(r1_start, r2_start)
            if reordered0:
                logger.debug(f"Reordered initial roots for sample {idx}: ({r1_start:.6f}, {r2_start:.6f}) -> ({r1_curr:.6f}, {r2_curr:.6f})")

            traj = []
            traj.append([0.0, r1_curr, r2_curr])

            valid_path = True
            for t in t_values[1:]:
                a_next, b_next = self._linear_homotopy_coeffs(a_start, b_start, a_end, b_end, t)
                if skip_if_nonreal and (self._discriminant(a_next, b_next) <= 0 or self._discriminant(a_curr, b_curr) <= 0):
                    valid_path = False
                    break

                x = torch.tensor([[a_curr, b_curr, r1_curr, r2_curr, a_next, b_next]], dtype=torch.float32)
                x = x.to(mean.device)  # Move to same device as mean and std
                pred = self.trainer.predict(x, mean, std)[0].tolist()
                # Guard against NaN/Inf in prediction
                if not torch.isfinite(torch.tensor(pred)).all().item():
                    valid_path = False
                    logger.debug(f"Invalid prediction (NaN/Inf) at sample {idx}, t={t:.3f}; skipping path.")
                    break
                # predict() now returns descending order; keep descending for internal state
                r1_next, r2_next = pred[0], pred[1]  # already descending

                a_curr, b_curr = a_next, b_next
                # Keep the convention: (lambda1 >= lambda2) which matches descending order
                r1_curr, r2_curr, reordered = self._normalize_roots_desc(r1_next, r2_next)
                if reordered:
                    logger.debug(f"Reordered predicted roots for sample {idx}, t={t:.3f}: ({r1_next:.6f}, {r2_next:.6f}) -> ({r1_curr:.6f}, {r2_curr:.6f})")
                if save_trajectories:
                    traj.append([t, r1_curr, r2_curr])

            if not valid_path:
                num_skipped_nonreal += 1
                continue

            preds_final.append([r1_curr, r2_curr])
            gts.append(gt_sorted)

            if save_trajectories:
                traj_path = os.path.join(save_path, "trajectories", f"sample_{idx:06d}.txt")
                with open(traj_path, "w", encoding="utf-8") as f:
                    for t, rr1, rr2 in traj:
                        f.write(f"{t:.8f} {rr1:.8f} {rr2:.8f}\n")

        # Calculate evaluation metrics
        if preds_final:
            pred_tensor = torch.tensor(preds_final, dtype=torch.float32)
            gt_tensor = torch.tensor(gts, dtype=torch.float32)
            rmse = torch.sqrt(mse_sorted(pred_tensor, gt_tensor)).item()
            # Accuracy: both roots within threshold in RMSE sense per sample
            accuracies = {}
            for threshold in thresholds:
                acc = evaluate_accuracy_within_threshold(pred_tensor, gt_tensor, threshold)
                accuracies[threshold] = acc
        else:
            rmse = float("nan")
            accuracies = {threshold: float("nan") for threshold in thresholds}

        elapsed = time.time() - start_time
        logger.info("=== Multi-steps Evaluation ===")
        logger.info(f"Samples processed: {len(preds_final)} / {len(samples)}")
        logger.info(f"Skipped (non-real along path): {num_skipped_nonreal}")
        logger.info(f"Number of steps: {self.num_steps}")
        logger.info(f"RMSE(sorted): {rmse:.6f}")
        for threshold in sorted(thresholds, reverse=True):
            acc = accuracies[threshold]
            logger.info(f"RMSE_Accuracy(threshold={threshold}): {acc:.2f}%")
        logger.info(f"Elapsed time: {elapsed:.3f}s")

        # Save summary
        with open(os.path.join(save_path, "summary.txt"), "w", encoding="utf-8") as f:
            f.write(f"processed\t{len(preds_final)}\n")
            f.write(f"skipped_nonreal\t{num_skipped_nonreal}\n")
            f.write(f"num_steps\t{self.num_steps}\n")
            f.write(f"rmse_sorted\t{rmse:.8f}\n")
            for threshold in sorted(thresholds, reverse=True):
                acc = accuracies[threshold]
                f.write(f"rmse_acc_{threshold}\t{acc:.4f}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--scaler-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--threshold", type=float, nargs="+", default=[0.01], 
                        help="RMSE accuracy threshold(s) for evaluation (can specify multiple)")
    parser.add_argument("--skip-if-nonreal", action="store_true", help="Skip samples if any intermediate discriminant <= 0")
    parser.add_argument("--save-trajectories", action="store_true")
    parser.add_argument("--timezone", type=str, default="Europe/Paris",
                        help="Timezone for logging (default: Europe/Paris)")
    parser.add_argument("--compare-julia", action="store_true",
                        help="Compare with Julia HomotopyContinuation library")
    # Parallel processing options for Julia comparison
    parser.add_argument("--julia-n-jobs", type=int, default=None,
                        help="Number of parallel jobs for Julia HC comparison (default from env or 1)")
    parser.add_argument("--julia-backend", type=str, default=None,
                        help="joblib backend for Julia HC comparison (loky/threading; default from env or loky)")
    parser.add_argument("--julia-prefer", type=str, default=None,
                        help="joblib prefer for Julia HC comparison (processes/threads; default from env or processes)")
    parser.add_argument("--julia-verbose", type=int, default=None,
                        help="joblib verbose for Julia HC comparison (0=silent; default from env or 0)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    logger, now_local = setup_logging(args.save_path, args.timezone)
    
    # Set debug level if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)

    logger.info("=== Multi-Steps Root Finding (Task 1.2) ===")
    logger.info(f"Location: {args.timezone}")
    logger.info(f"Start time: {now_local.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    logger.info(f"Dataset: {args.dataset_path}")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Scaler: {args.scaler_path}")
    logger.info(f"Save path: {args.save_path}")
    logger.info(f"Number of steps: {args.num_steps}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Thresholds: {args.threshold}")
    logger.info(f"Skip non-real along path: {args.skip_if_nonreal}")
    logger.info(f"Save trajectories: {args.save_trajectories}")
    logger.info(f"Compare with Julia: {args.compare_julia}")

    # Reproducibility (not critical for pure inference but align with other scripts)
    random.seed(42)
    torch.manual_seed(42)

    # Load data
    loader = DataLoader()
    samples, targets = loader.load_data(args.dataset_path, args.max_samples)
    logger.info(f"Loaded {len(samples)} samples from dataset")

    # Load model and scaler
    model, mean, std = load_model_and_scaler(args.model_path, args.scaler_path, args.device)
    
    # Create trainer for prediction
    trainer = ModelTrainer(model, device=args.device)
    
    # Create homotopy root finder
    root_finder = HomotopyRootFinder(trainer, num_steps=args.num_steps)

    # Run multi-steps tracking
    root_finder.find_roots(
        samples=samples,
        targets=targets,
        mean=mean,
        std=std,
        skip_if_nonreal=args.skip_if_nonreal,
        save_trajectories=args.save_trajectories,
        save_path=args.save_path,
        logger=logger,
        thresholds=args.threshold,
    )
    
    # Julia comparison if requested
    if args.compare_julia:
        # Lazy import to avoid Julia dependency unless needed
        from julia_comparison import JuliaHomotopyComparison
        julia_comparison = JuliaHomotopyComparison(logger)
        julia_results = julia_comparison.compare_with_julia(
            samples=samples,
            targets=targets,
            thresholds=args.threshold,
            num_steps=args.num_steps,
            max_samples=args.max_samples,
            n_jobs=args.julia_n_jobs,
            backend=args.julia_backend,
            prefer=args.julia_prefer,
            verbose=args.julia_verbose,
        )
        
        # Save Julia comparison results
        if julia_results:
            julia_summary_path = os.path.join(args.save_path, "julia_comparison.txt")
            with open(julia_summary_path, "w", encoding="utf-8") as f:
                f.write(f"julia_rmse\t{julia_results['julia_rmse']:.8f}\n")
                f.write(f"julia_total_time\t{julia_results['total_julia_time']:.6f}\n")
                f.write(f"julia_avg_time\t{julia_results['total_julia_time']/julia_results['num_processed']:.8f}\n")
                f.write(f"julia_processed\t{julia_results['num_processed']}\n")
                f.write(f"julia_skipped\t{julia_results['num_skipped']}\n")
                f.write(f"julia_complex_skipped\t{julia_results['num_complex_skipped']}\n")
                for threshold in sorted(args.threshold, reverse=True):
                    acc = julia_results['julia_accuracies'][threshold]
                    f.write(f"julia_acc_{threshold}\t{acc:.4f}\n")
            
            logger.info(f"Julia comparison results saved to {julia_summary_path}")


if __name__ == "__main__":
    main()
