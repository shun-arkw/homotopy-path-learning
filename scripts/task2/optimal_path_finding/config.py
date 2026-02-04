from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LossConfig:
    """Configuration for the optimization loss (shared base).

    Attributes:
        samples_per_segment: Number of sample points M per segment.
        lambda_smooth: Weight for the smoothness regularizer (second finite difference).
            NOTE:
                This term is *not* part of the condition length itself; it is an optional
                optimizer stabilizer to discourage zig-zag paths. If you want to minimize
                (discretized) condition length as-is, set this to 0.0.
        eps_soft: Small epsilon to avoid division by zero and log(0).
        delta_soft: Parameter for softabs. See notes below.
        disc_eps: Optional stabilization passed into discriminant calculation.
        lead_eps: Optional stabilization for the leading coefficient magnitude term.
        disc_backend: Backend for discriminant computation. Options:
            - "complex": k x k complex Sylvester matrix slogdet (fast; default).
            - "real_block": 2k x 2k real block matrix slogdet (slower; useful for cross-checks).

    Notes:
        What is "softabs"?
            The discriminant D(p) is complex in general. When we only need its magnitude
            (for condition-based weighting), we use |D|. However, |D| is non-smooth at 0,
            and using 1/|D| or log|D| can create numerical issues near D=0.

            A smooth approximation is:

                softabs(z) = sqrt(|z|^2 + delta_soft)

            where |z| = sqrt(Re(z)^2 + Im(z)^2) and delta_soft > 0.
            This avoids non-differentiability at 0 and prevents division by zero.

        Why the 1/degree power?
            Some formulations weight the integrand by |D(p(t))|^{1/degree},
            equivalently using (softabs + eps)^{1/degree} as a smooth proxy.
            In log-domain this becomes a simple scaling by 1/degree.

        Why log-domain?
            The discriminant magnitude can vary over many orders of magnitude. Computing
            weights via log|D| is typically much more stable than using |D| directly.
    """

    samples_per_segment: int = 16
    # Default to 0: optimize condition-length-like loss directly unless the user opts in.
    lambda_smooth: float = 0.0
    eps_soft: float = 1e-12
    delta_soft: float = 1e-12
    disc_eps: float = 0.0
    lead_eps: float = 1e-24
    disc_backend: str = "complex"


@dataclass
class BezierLossConfig(LossConfig):
    """Loss configuration for Bezier coefficient paths.

    Attributes:
        bezier_eval_method: How to evaluate Bezier curves T(t).
            - "casteljau": De Casteljau recursion (O(d^2), numerically stable; default).
            - "bernstein": Bernstein basis linear combination (O(d), often faster for d≈20+,
              but can be less stable for large d / extreme t near 0 or 1).
        alpha: Weight for the L2-norm regularizer of the first derivative ||T'(t)||_{L2}.
        beta: Weight for the L2-norm regularizer of the second derivative ||T''(t)||_{L2}.
    """

    bezier_eval_method: str = "casteljau"
    # L2 regularization on Bezier derivatives (0 disables).
    alpha: float = 0.0
    beta: float = 0.0


@dataclass
class PiecewiseLossConfig(LossConfig):
    """Loss configuration for piecewise-linear coefficient paths.

    This currently matches the shared base exactly; it exists to keep APIs explicit and
    provide a home for piecewise-only options as they evolve.
    """

    # No extra fields yet.


@dataclass
class OptimConfig:
    """Configuration for the gradient-based optimization loop.

    Attributes:
        lr: Learning rate.
        steps: Number of optimization steps.
        print_every: Print logs every this many steps (0 disables printing).
        grad_clip: If not None, clip gradient norm to this value.
    """

    lr: float = 1e-2
    steps: int = 500
    print_every: int = 50
    grad_clip: float | None = 1.0


