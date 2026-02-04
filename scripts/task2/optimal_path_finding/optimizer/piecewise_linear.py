from __future__ import annotations

from typing import Any

import torch

from ..config import OptimConfig, PiecewiseLossConfig
from ..utils.complex_repr import complex_norm_ri, p_to_monic_poly_coeffs_ri, to_ri
from ..utils.discriminant_calculator import discriminant_univariate_logabs
from ..utils.log_stability import make_uniform_ts, log_softabs_from_logabs, log_softabs_plus_eps
from .init import initialize_control_points_linear, initialize_control_points_qr


class PiecewiseLinearHCPathOptimizer(torch.nn.Module):
    """Optimizes a piecewise linear path in complex coefficient space using gradients.

    We optimize intermediate control points (P_1, ..., P_{K-1}) while fixing endpoints
    P_0 = p_start and P_K = p_target.

    Each segment is linear:
        gamma_k(t) = (1 - t) P_k + t P_{k+1}, t in [0, 1].

    The main loss term approximates a condition-length-like integral using M samples
    per segment. The discriminant magnitude is computed via
    discriminant_univariate_logabs, and we only use log|Disc| for stability.

    Implementation detail:
        All computations are done in a real representation (Re, Im), i.e. in R^{2*degree}.
    """

    def __init__(
        self,
        p_start: torch.Tensor,
        p_target: torch.Tensor,
        num_segments: int,
        *,
        init: str = "linear",
        init_imag_noise_scale: float = 0.0,
        init_seed: int | None = None,
        loss_cfg: PiecewiseLossConfig = PiecewiseLossConfig(),
    ):
        """Initializes the optimizer module.

        Args:
            p_start: Start coefficient vector in one of these formats:
                - complex tensor of shape (degree,)
                - real tensor of shape (degree, 2) in (Re, Im)
                - real tensor of shape (degree,) (treated as purely real)
            p_target: Target coefficient vector with the same shape rule as p_start.
            num_segments: Number of segments K (control points are K+1).
            init: Initialization method: "linear" or "qr".
            loss_cfg: Loss configuration.

        Raises:
            ValueError: If shapes are incompatible.
        """
        super().__init__()
        if num_segments < 1:
            raise ValueError("num_segments must be >= 1.")
        self.K = int(num_segments)
        self.loss_cfg = loss_cfg

        p_start_ri = to_ri(p_start).detach()
        p_target_ri = to_ri(p_target).detach()

        if p_start_ri.shape != p_target_ri.shape:
            raise ValueError("p_start and p_target must have the same shape after conversion to (Re, Im).")
        if p_start_ri.ndim != 2 or p_start_ri.shape[-1] != 2:
            raise ValueError("Expected a single vector with shape (degree, 2) or complex shape (degree,).")

        # Here, degree is the polynomial degree for monic f(x)=x^degree+...,
        # and p has length degree.
        self.degree = int(p_start_ri.shape[0])

        self.register_buffer("P0", p_start_ri)
        self.register_buffer("PK", p_target_ri)

        if init == "qr":
            P_init = initialize_control_points_qr(self.P0, self.PK, self.K)
        elif init == "linear":
            P_init = initialize_control_points_linear(
                self.P0,
                self.PK,
                self.K,
                imag_noise_scale=init_imag_noise_scale,
                seed=init_seed,
            )
        else:
            raise ValueError("init must be 'linear' or 'qr'.")

        if self.K >= 2:
            self.P_mid = torch.nn.Parameter(P_init[1:-1].clone())  # shape (K-1, degree, 2)
        else:
            self.P_mid = None

    def control_points(self) -> torch.Tensor:
        """Returns the full control point list P_0..P_K.

        Returns:
            Tensor of shape (K+1, degree, 2) in (Re, Im) format.
        """
        if self.K == 1:
            return torch.stack([self.P0, self.PK], dim=0)
        return torch.cat([self.P0.unsqueeze(0), self.P_mid, self.PK.unsqueeze(0)], dim=0)

    def _smoothness_loss(self, P: torch.Tensor) -> torch.Tensor:
        """Second-difference smoothness regularizer."""
        if self.K <= 1:
            return torch.zeros((), device=P.device, dtype=P.dtype)

        d2 = P[2:] - 2.0 * P[1:-1] + P[:-2]  # shape (K-1, degree, 2)
        re = d2[..., 0]
        im = d2[..., 1]
        return (re * re + im * im).sum()

    def _condition_length_like_loss(self, P: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Condition-length-like loss term computed with one batched discriminant call."""
        cfg = self.loss_cfg
        device, dtype = P.device, P.dtype

        K = self.K
        M = int(cfg.samples_per_segment)

        # Segment vectors and lengths
        P0s = P[:-1]  # (K, degree, 2)
        P1s = P[1:]  # (K, degree, 2)
        dP = P1s - P0s  # (K, degree, 2)
        seg_len = complex_norm_ri(dP)  # (K,)

        # Sample points t in (0, 1)
        ts = make_uniform_ts(M, device=device, dtype=dtype)  # (M,)

        # gamma[k, m] = (1 - t_m) P_k + t_m P_{k+1}
        t = ts.view(1, M, 1, 1)  # (1, M, 1, 1)
        gamma = (1.0 - t) * P0s.unsqueeze(1) + t * P1s.unsqueeze(1)  # (K, M, degree, 2)

        # Flatten to a single batch for discriminant evaluation
        gamma_flat = gamma.reshape(K * M, self.degree, 2)  # (K*M, degree, 2)
        a_ri = p_to_monic_poly_coeffs_ri(gamma_flat)  # (K*M, degree+1, 2)

        # One batched call: returns log|Disc|, shape (K*M,)
        disc_logabs = discriminant_univariate_logabs(
            a_ri,
            eps=cfg.disc_eps,
            lead_eps=cfg.lead_eps,
            backend=getattr(cfg, "disc_backend", "complex"),
        )

        disc_logabs = disc_logabs.view(K, M)  # (K, M)
        min_logabs_disc = disc_logabs.min()

        # Build log(softabs + eps) stably
        log_softabs = log_softabs_from_logabs(disc_logabs, cfg.delta_soft)
        log_softabs_eps = log_softabs_plus_eps(log_softabs, cfg.eps_soft)  # log(softabs + eps)

        # Incorporate the 1/degree root: denom = (softabs + eps)^{1/degree}
        degree_f = torch.tensor(float(self.degree), device=device, dtype=dtype)
        log_denom = log_softabs_eps / degree_f

        # inv-only weight: w = 1 / denom = exp(-log_denom)
        w = torch.exp(-log_denom)

        w_mean = w.mean(dim=1)  # (K,)
        loss = (seg_len * w_mean).sum()

        diag = {
            "min_logabs_disc": min_logabs_disc,
            "mean_seg_len": seg_len.mean().detach(),
        }
        return loss, diag

    def forward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Computes the total loss (main term + smoothness regularizer)."""
        P = self.control_points()
        L_cl, diag = self._condition_length_like_loss(P)
        L_sm = self._smoothness_loss(P)
        L = L_cl + self.loss_cfg.lambda_smooth * L_sm

        diag.update(
            {
                "loss_cl": L_cl.detach(),
                "loss_smooth": L_sm.detach(),
                "loss_total": L.detach(),
            }
        )
        return L, diag


def optimize_piecewise_linear_path(
    p_start: torch.Tensor,
    p_target: torch.Tensor,
    num_segments: int,
    *,
    init: str = "linear",
    init_imag_noise_scale: float = 0.0,
    init_seed: int | None = None,
    loss_cfg: PiecewiseLossConfig = PiecewiseLossConfig(),
    optim_cfg: OptimConfig = OptimConfig(),
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Optimizes a piecewise linear coefficient path for one endpoint pair.

    Returns:
        (P_best, info)
          - P_best: (K+1, degree, 2) in (Re, Im) format
          - info: dict
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PiecewiseLinearHCPathOptimizer(
        p_start=p_start.to(device=device),
        p_target=p_target.to(device=device),
        num_segments=num_segments,
        init=init,
        init_imag_noise_scale=init_imag_noise_scale,
        init_seed=init_seed,
        loss_cfg=loss_cfg,
    ).to(device=device, dtype=dtype)

    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        P = model.control_points().detach()
        return P, {"note": "K=1, no optimization variables.", "K": int(num_segments), "degree": int(model.degree)}

    opt = torch.optim.Adam(params, lr=optim_cfg.lr)
    best: dict[str, Any] = {"loss": float("inf"), "P": None, "diag": None}

    for step in range(int(optim_cfg.steps)):
        opt.zero_grad(set_to_none=True)
        loss, diag = model()
        # IMPORTANT: `diag` corresponds to the current control points (pre-step).
        # Update best *before* applying `opt.step()` to keep best["P"] aligned with best["diag"].
        with torch.no_grad():
            if float(diag["loss_total"].item()) < best["loss"]:
                best["loss"] = float(diag["loss_total"].item())
                best["P"] = model.control_points().detach().clone()
                best["diag"] = {k: float(v.item()) if torch.is_tensor(v) and v.numel() == 1 else v for k, v in diag.items()}

        loss.backward()
        if optim_cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(params, max_norm=float(optim_cfg.grad_clip))
        opt.step()

        if optim_cfg.print_every and (step % int(optim_cfg.print_every) == 0 or step == int(optim_cfg.steps) - 1):
            print(
                f"[step {step:04d}] "
                f"loss={diag['loss_total'].item():.6e} "
                f"cl={diag['loss_cl'].item():.6e} "
                f"sm={diag['loss_smooth'].item():.6e} "
                f"min_log|Disc|={diag['min_logabs_disc'].item():.6e}"
            )

    info = {
        "best_loss": best["loss"],
        "best_diag": best["diag"],
        "device": str(device),
        "dtype": str(dtype),
        "K": int(num_segments),
        "degree": int(model.degree),
        "init": init,
        "init_imag_noise_scale": float(init_imag_noise_scale),
        "init_seed": init_seed,
        "loss_cfg": loss_cfg,
        "optim_cfg": optim_cfg,
    }
    return best["P"], info


