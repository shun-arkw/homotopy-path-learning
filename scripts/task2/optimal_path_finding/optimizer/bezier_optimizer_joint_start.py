from __future__ import annotations

import math
from typing import Any

import torch

from ..config import BezierLossConfig, OptimConfig
from ..utils.bezier import bezier_eval, bezier_derivative_control_points
from ..utils.complex_repr import complex_norm_ri, p_to_monic_poly_coeffs_ri, to_ri
from ..utils.condition_length.bezier_torch_screen import min_disc_logabs_on_bezier_torch
from ..utils.discriminant_calculator import discriminant_univariate_logabs
from ..utils.log_stability import make_uniform_ts, log_softabs_from_logabs, log_softabs_plus_eps
from .init import initialize_control_points_bezier_linear


def _p_start_complex_from_c_s(c: torch.Tensor, s: torch.Tensor, degree: int) -> torch.Tensor:
    """Build p_start (complex) for monic polynomial:

        G(x) = (x - c)^n - s^n,  n=degree

    Returned p has shape (degree,) representing (a_{n-1}, ..., a_0).
    """
    n = int(degree)
    if n < 1:
        raise ValueError("degree must be >= 1")

    # full coefficients in descending powers: [1, a_{n-1}, ..., a_0]
    full = torch.zeros(n + 1, dtype=c.dtype, device=c.device)
    minus_c = -c
    for j in range(0, n + 1):
        full[j] = math.comb(n, j) * (minus_c**j)
    full[-1] = full[-1] - (s**n)

    # drop leading 1 -> p = (a_{n-1}, ..., a_0)
    return full[1:]


class BezierHCPathOptimizerJointStart(torch.nn.Module):
    """Jointly optimizes start polynomial params + Bezier mid control points.

    - Target endpoint is fixed.
    - Start endpoint is generated from parameters (c, rho, theta):
        G(x) = (x - c)^n - s^n,  s = rho * exp(i*theta)
    """

    def __init__(
        self,
        p_target: torch.Tensor,
        *,
        bezier_degree: int = 3,
        init_imag_noise_scale: float = 0.0,
        init_seed: int | None = None,
        loss_cfg: BezierLossConfig = BezierLossConfig(),
        # start poly init
        c_re_init: float = 0.0,
        c_im_init: float = 0.0,
        rho_init: float = 1.0,
        theta_init: float = 0.0,
        rho_eps: float = 1e-8,
    ):
        super().__init__()
        d = int(bezier_degree)
        if d < 1:
            raise ValueError("bezier_degree must be >= 1.")
        self.d = d
        self.loss_cfg = loss_cfg
        self.rho_eps = float(rho_eps)

        p_target_ri = to_ri(p_target).detach()
        if p_target_ri.ndim != 2 or p_target_ri.shape[-1] != 2:
            raise ValueError("Expected p_target shape (degree, 2) or complex shape (degree,).")

        self.degree = int(p_target_ri.shape[0])
        self.register_buffer("Pd", p_target_ri)

        # Parameters for c (complex), rho>0, theta (rotation)
        dtype = p_target_ri.dtype
        device = p_target_ri.device
        self.c_re = torch.nn.Parameter(torch.tensor(float(c_re_init), device=device, dtype=dtype))
        self.c_im = torch.nn.Parameter(torch.tensor(float(c_im_init), device=device, dtype=dtype))
        self.log_rho = torch.nn.Parameter(torch.tensor(math.log(float(rho_init)), device=device, dtype=dtype))
        self.theta = torch.nn.Parameter(torch.tensor(float(theta_init), device=device, dtype=dtype))

        # Initialize mid control points using current start/target
        with torch.no_grad():
            P0_init = self.p_start_ri().detach()
            P_init = initialize_control_points_bezier_linear(
                P0_init,
                self.Pd,
                self.d,
                imag_noise_scale=init_imag_noise_scale,
                seed=init_seed,
            )
        if self.d >= 2:
            self.P_mid = torch.nn.Parameter(P_init[1:-1].clone())  # (d-1, degree, 2)
        else:
            self.P_mid = None

    def p_start_ri(self) -> torch.Tensor:
        """Current p_start in (Re,Im) with shape (degree,2)."""
        c = torch.complex(self.c_re, self.c_im)
        rho = torch.exp(self.log_rho) + self.rho_eps
        s = rho * torch.complex(torch.cos(self.theta), torch.sin(self.theta))
        p_start_c = _p_start_complex_from_c_s(c=c, s=s, degree=self.degree)  # (degree,)
        return to_ri(p_start_c)

    def control_points(self) -> torch.Tensor:
        """Returns Bezier control points P_0..P_d, shape (d+1, degree, 2)."""
        P0 = self.p_start_ri()
        if self.d == 1:
            return torch.stack([P0, self.Pd], dim=0)
        return torch.cat([P0.unsqueeze(0), self.P_mid, self.Pd.unsqueeze(0)], dim=0)

    def _bezier_condition_length_loss(self, P_ctrl: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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

        log_softabs = log_softabs_from_logabs(disc_logabs, cfg.delta_soft)
        log_softabs_eps = log_softabs_plus_eps(log_softabs, cfg.eps_soft)
        degree_f = torch.tensor(float(self.degree), device=device, dtype=dtype)
        w = torch.exp(-(log_softabs_eps / degree_f))  # inv-only weight

        L_cl = (speed * w).mean()
        L_d1_l2 = torch.sqrt((speed * speed).mean())

        accel = torch.zeros_like(speed)
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


def optimize_bezier_path_joint_start(
    p_target: torch.Tensor,
    *,
    bezier_degree: int = 3,
    init_imag_noise_scale: float = 0.0,
    init_seed: int | None = None,
    loss_cfg: BezierLossConfig = BezierLossConfig(),
    optim_cfg: OptimConfig = OptimConfig(),
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
    # start init
    c_re_init: float = 0.0,
    c_im_init: float = 0.0,
    rho_init: float = 1.0,
    theta_init: float = 0.0,
    rho_eps: float = 1e-8,
    # Torch-only screen (heuristic) during optimization: evaluated for logging only.
    screen_disc_coarse_samples: int = 257,
    screen_disc_refine_steps: int = 2,
    screen_disc_refine_samples: int = 257,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Jointly optimize start polynomial params + Bezier mid control points (target endpoint fixed)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BezierHCPathOptimizerJointStart(
        p_target=p_target.to(device=device),
        bezier_degree=int(bezier_degree),
        init_imag_noise_scale=float(init_imag_noise_scale),
        init_seed=init_seed,
        loss_cfg=loss_cfg,
        c_re_init=float(c_re_init),
        c_im_init=float(c_im_init),
        rho_init=float(rho_init),
        theta_init=float(theta_init),
        rho_eps=float(rho_eps),
    ).to(device=device, dtype=dtype)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=optim_cfg.lr)

    best: dict[str, Any] = {"loss": float("inf"), "P": None, "diag": None, "start": None}

    for step in range(int(optim_cfg.steps)):
        opt.zero_grad(set_to_none=True)
        loss, diag = model()

        with torch.no_grad():
            cur_loss = float(loss.item())
            if cur_loss < best["loss"]:
                best["loss"] = cur_loss
                best["P"] = model.control_points().detach().clone()
                best["diag"] = {k: float(v.item()) if torch.is_tensor(v) and v.numel() == 1 else v for k, v in diag.items()}
                rho = float((torch.exp(model.log_rho) + model.rho_eps).item())
                best["start"] = {
                    "c_re": float(model.c_re.item()),
                    "c_im": float(model.c_im.item()),
                    "rho": rho,
                    "theta": float(model.theta.item()),
                }

        loss.backward()
        if optim_cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(params, max_norm=float(optim_cfg.grad_clip))
        opt.step()

        if optim_cfg.print_every and (step % int(optim_cfg.print_every) == 0 or step == int(optim_cfg.steps) - 1):
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
                f"[bezier joint step {step:04d}] "
                f"loss={float(diag['loss_total'].item()):.6e} "
                f"cl={float(diag['loss_cl'].item()):.6e} "
                f"min_log|Disc|_samples={float(diag['min_logabs_disc_samples'].item()):.6e} "
                f"min_log|Disc|_screen={float(min_logabs.item()):.6e}"
            )

    info = {
        "best_loss": best["loss"],
        "best_diag": best["diag"],
        "best_start": best["start"],
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


