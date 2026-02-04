from __future__ import annotations

from typing import Any

import torch

from ..config import BezierLossConfig, OptimConfig
from ..utils.bezier import (
    bezier_eval,
    bezier_derivative_control_points,
)
from ..utils.complex_repr import complex_norm_ri, p_to_monic_poly_coeffs_ri, to_ri
from ..utils.condition_length.bezier_torch_screen import min_disc_logabs_on_bezier_torch
from ..utils.discriminant_calculator import discriminant_univariate_logabs
from ..utils.log_stability import make_uniform_ts, log_softabs_from_logabs, log_softabs_plus_eps
from .init import initialize_control_points_bezier_linear


class BezierHCPathOptimizer(torch.nn.Module):
    """Optimizes a Bezier coefficient path using gradients (endpoints fixed)."""

    def __init__(
        self,
        p_start: torch.Tensor,
        p_target: torch.Tensor,
        *,
        bezier_degree: int = 3,
        init_imag_noise_scale: float = 0.0,
        init_seed: int | None = None,
        loss_cfg: BezierLossConfig = BezierLossConfig(),
    ):
        super().__init__()
        d = int(bezier_degree)
        if d < 1:
            raise ValueError("bezier_degree must be >= 1.")
        self.d = d
        self.loss_cfg = loss_cfg

        p_start_ri = to_ri(p_start).detach()
        p_target_ri = to_ri(p_target).detach()

        if p_start_ri.shape != p_target_ri.shape:
            raise ValueError("p_start and p_target must have the same shape after conversion to (Re, Im).")
        if p_start_ri.ndim != 2 or p_start_ri.shape[-1] != 2:
            raise ValueError("Expected a single vector with shape (degree, 2) or complex shape (degree,).")

        self.degree = int(p_start_ri.shape[0])
        self.register_buffer("P0", p_start_ri)
        self.register_buffer("Pd", p_target_ri)

        P_init = initialize_control_points_bezier_linear(
            self.P0,
            self.Pd,
            self.d,
            imag_noise_scale=init_imag_noise_scale,
            seed=init_seed,
        )

        if self.d >= 2:
            self.P_mid = torch.nn.Parameter(P_init[1:-1].clone())  # (d-1, degree, 2)
        else:
            self.P_mid = None

    def control_points(self) -> torch.Tensor:
        """Returns Bezier control points P_0..P_d, shape (d+1, degree, 2)."""
        if self.d == 1:
            return torch.stack([self.P0, self.Pd], dim=0)
        return torch.cat([self.P0.unsqueeze(0), self.P_mid, self.Pd.unsqueeze(0)], dim=0)

    def _bezier_condition_length_loss(self, P_ctrl: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the total Bezier loss (condition length + optional derivative L2 regularizers)."""
        cfg = self.loss_cfg
        device, dtype = P_ctrl.device, P_ctrl.dtype

        M = int(cfg.samples_per_segment)
        ts = make_uniform_ts(M, device=device, dtype=dtype)  # (M,)

        T = bezier_eval(P_ctrl, ts, method=cfg.bezier_eval_method)  # (M, degree, 2)
        Q_ctrl = bezier_derivative_control_points(P_ctrl)  # (d, degree, 2)
        Tp = bezier_eval(Q_ctrl, ts, method=cfg.bezier_eval_method)  # (M, degree, 2)
        speed = complex_norm_ri(Tp)  # (M,)

        a_ri = p_to_monic_poly_coeffs_ri(T)  # (M, degree+1, 2)
        disc_logabs = discriminant_univariate_logabs(
            a_ri,
            eps=cfg.disc_eps,
            lead_eps=cfg.lead_eps,
            backend=getattr(cfg, "disc_backend", "complex"),
        )  # (M,)

        # Stable log(softabs + eps)
        log_softabs = log_softabs_from_logabs(disc_logabs, cfg.delta_soft)
        log_softabs_eps = log_softabs_plus_eps(log_softabs, cfg.eps_soft)
        degree_f = torch.tensor(float(self.degree), device=device, dtype=dtype)
        log_denom = log_softabs_eps / degree_f

        # inv-only weight: w = 1 / (softabs(Disc) + eps)^{1/degree}
        w = torch.exp(-log_denom)

        # ------------------------------------------------------------
        # Main term: discrete Bezier condition length-like objective
        # ------------------------------------------------------------
        L_cl = (speed * w).mean()

        # ------------------------------------------------------------
        # Optional L2 regularizers on Bezier derivatives
        #   ||T'||_{L2}  ≈ sqrt(E[||T'(t)||^2])
        #   ||T''||_{L2} ≈ sqrt(E[||T''(t)||^2])
        # ------------------------------------------------------------
        L_d1_l2 = torch.sqrt((speed * speed).mean())

        # Bezier degree d = P_ctrl.shape[0]-1. Second derivative exists if d >= 2.
        if P_ctrl.shape[0] >= 3:
            R_ctrl = bezier_derivative_control_points(Q_ctrl)  # (d-1, degree, 2)
            Tpp = bezier_eval(R_ctrl, ts, method=cfg.bezier_eval_method)  # (M, degree, 2)
            accel = complex_norm_ri(Tpp)  # (M,)
            L_d2_l2 = torch.sqrt((accel * accel).mean())
        else:
            L_d2_l2 = torch.zeros((), device=device, dtype=dtype)

        loss = L_cl + float(cfg.alpha) * L_d1_l2 + float(cfg.beta) * L_d2_l2
        diag = {
            "min_logabs_disc_samples": disc_logabs.min().detach(),
            "mean_speed": speed.mean().detach(),
            "mean_accel": accel.mean().detach(),
            "loss_cl": L_cl.detach(),
            "loss_d1_l2": L_d1_l2.detach(),
            "loss_d2_l2": L_d2_l2.detach(),
        }
        return loss, diag

    def forward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        P_ctrl = self.control_points()
        L, diag = self._bezier_condition_length_loss(P_ctrl)
        diag.update({"loss_total": L.detach()})
        return L, diag


def optimize_bezier_path(
    p_start: torch.Tensor,
    p_target: torch.Tensor,
    *,
    bezier_degree: int = 3,
    init_imag_noise_scale: float = 0.0,
    init_seed: int | None = None,
    loss_cfg: BezierLossConfig = BezierLossConfig(),
    optim_cfg: OptimConfig = OptimConfig(),
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
    # Torch-only screen (heuristic) during optimization: evaluated for logging only.
    screen_disc_coarse_samples: int = 257,
    screen_disc_refine_steps: int = 2,
    screen_disc_refine_samples: int = 257,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Optimize a Bezier coefficient path by gradient descent (endpoints fixed)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BezierHCPathOptimizer(
        p_start=p_start.to(device=device),
        p_target=p_target.to(device=device),
        bezier_degree=int(bezier_degree),
        init_imag_noise_scale=float(init_imag_noise_scale),
        init_seed=init_seed,
        loss_cfg=loss_cfg,
    ).to(device=device, dtype=dtype)

    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        P = model.control_points().detach()
        return P, {"note": "bezier_degree=1, no optimization variables.", "degree": int(model.degree), "d": int(model.d)}

    opt = torch.optim.Adam(params, lr=optim_cfg.lr)

    best: dict[str, Any] = {"loss": float("inf"), "P": None, "diag": None}

    for step in range(int(optim_cfg.steps)):
        opt.zero_grad(set_to_none=True)
        loss, diag = model()

        # Track best (pre-step)
        with torch.no_grad():
            cur_loss = float(loss.item())
            if cur_loss < best["loss"]:
                best["loss"] = cur_loss
                best["P"] = model.control_points().detach().clone()
                best["diag"] = {k: float(v.item()) if torch.is_tensor(v) and v.numel() == 1 else v for k, v in diag.items()}

        loss.backward()
        if optim_cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(params, max_norm=float(optim_cfg.grad_clip))
        opt.step()

        if optim_cfg.print_every and (step % int(optim_cfg.print_every) == 0 or step == int(optim_cfg.steps) - 1):
            # Heuristic Disc screen (no backprop, occasional)
            with torch.no_grad():
                P_cur = model.control_points().detach()
                min_logabs, _t = min_disc_logabs_on_bezier_torch(
                    P_cur,
                    loss_cfg=loss_cfg,
                    coarse_samples=int(screen_disc_coarse_samples),
                    refine_steps=int(screen_disc_refine_steps),
                    refine_samples=int(screen_disc_refine_samples),
                )
            print(
                f"[bezier step {step:04d}] "
                f"loss={float(diag['loss_total'].item()):.6e} "
                f"cl={float(diag['loss_cl'].item()):.6e} "
                f"d1_l2={float(diag['loss_d1_l2'].item()):.6e} "
                f"d2_l2={float(diag['loss_d2_l2'].item()):.6e} "
                f"min_log|Disc|_samples={float(diag['min_logabs_disc_samples'].item()):.6e} "
                f"min_log|Disc|_screen={float(min_logabs.item()):.6e}"
            )

    info = {
        "best_loss": best["loss"],
        "best_diag": best["diag"],
        "device": str(device),
        "dtype": str(dtype),
        "degree": int(model.degree),
        "bezier_degree": int(model.d),
        "init_imag_noise_scale": float(init_imag_noise_scale),
        "init_seed": init_seed,
        "loss_cfg": loss_cfg,
        "optim_cfg": optim_cfg,
        "screen_disc_coarse_samples": int(screen_disc_coarse_samples),
        "screen_disc_refine_steps": int(screen_disc_refine_steps),
        "screen_disc_refine_samples": int(screen_disc_refine_samples),
    }
    return best["P"], info


