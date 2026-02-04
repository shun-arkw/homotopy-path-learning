from __future__ import annotations

from typing import Any, Sequence

import torch

from ..config import OptimConfig, PiecewiseLossConfig
from ..utils.complex_repr import complex_norm_ri, p_to_monic_poly_coeffs_ri, to_ri
from ..utils.discriminant_calculator import discriminant_univariate_logabs
from ..utils.log_stability import make_uniform_ts, log_softabs_from_logabs, log_softabs_plus_eps
from .init import initialize_control_points_linear_batched


class BatchedPiecewiseLinearHCPathOptimizer(torch.nn.Module):
    """Batched variant of PiecewiseLinearHCPathOptimizer (optimizes R paths in parallel).

    We optimize intermediate control points for each path independently, but run the
    discriminant evaluation and loss computation in one batched call on GPU.
    """

    def __init__(
        self,
        p_start: torch.Tensor,
        p_target: torch.Tensor,
        num_segments: int,
        num_paths: int,
        *,
        init: str = "linear",
        init_imag_noise_scale: float = 0.0,
        init_seeds: Sequence[int] | None = None,
        loss_cfg: PiecewiseLossConfig = PiecewiseLossConfig(),
    ):
        super().__init__()
        if num_segments < 1:
            raise ValueError("num_segments must be >= 1.")
        if num_paths < 1:
            raise ValueError("num_paths must be >= 1.")

        self.K = int(num_segments)
        self.R = int(num_paths)
        self.loss_cfg = loss_cfg

        p_start_ri = to_ri(p_start).detach()
        p_target_ri = to_ri(p_target).detach()

        # Normalize shapes to (R, degree, 2)
        if p_start_ri.ndim == 2:
            p_start_ri = p_start_ri.unsqueeze(0).expand(self.R, -1, -1)
        if p_target_ri.ndim == 2:
            p_target_ri = p_target_ri.unsqueeze(0).expand(self.R, -1, -1)

        if p_start_ri.ndim != 3 or p_start_ri.shape[-1] != 2:
            raise ValueError("Expected p_start to be a vector (degree,) or (degree,2) or batched (R,degree,2).")
        if p_target_ri.shape != p_start_ri.shape:
            raise ValueError("p_target must match p_start shape after conversion to (Re, Im).")
        if p_start_ri.shape[0] != self.R:
            raise ValueError("Batch size of p_start/p_target must match num_paths (R).")

        self.degree = int(p_start_ri.shape[1])
        self.register_buffer("P0", p_start_ri)
        self.register_buffer("PK", p_target_ri)

        if init == "qr":
            raise NotImplementedError("Batched QR initialization is not implemented yet.")
        if init != "linear":
            raise ValueError("init must be 'linear' (or 'qr' in the future).")

        P_init = initialize_control_points_linear_batched(
            self.P0,
            self.PK,
            self.K,
            imag_noise_scale=init_imag_noise_scale,
            seeds=init_seeds,
        )

        if self.K >= 2:
            self.P_mid = torch.nn.Parameter(P_init[:, 1:-1].clone())  # (R, K-1, degree, 2)
        else:
            self.P_mid = None

    def control_points(self) -> torch.Tensor:
        """Returns control points for all paths: shape (R, K+1, degree, 2)."""
        if self.K == 1:
            return torch.stack([self.P0, self.PK], dim=1)  # (R, 2, degree, 2)
        return torch.cat([self.P0.unsqueeze(1), self.P_mid, self.PK.unsqueeze(1)], dim=1)

    def _smoothness_loss_vec(self, P: torch.Tensor) -> torch.Tensor:
        """Per-path smoothness loss. Returns shape (R,)."""
        if self.K <= 1:
            return torch.zeros((self.R,), device=P.device, dtype=P.dtype)
        d2 = P[:, 2:] - 2.0 * P[:, 1:-1] + P[:, :-2]  # (R, K-1, degree, 2)
        re = d2[..., 0]
        im = d2[..., 1]
        return (re * re + im * im).sum(dim=(-1, -2, -3))

    def _condition_length_like_loss_vec(self, P: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Per-path condition-length-like loss (vector), computed in one batched discriminant call."""
        cfg = self.loss_cfg
        device, dtype = P.device, P.dtype

        R, K = self.R, self.K
        M = int(cfg.samples_per_segment)

        # Segment vectors and lengths: (R, K)
        P0s = P[:, :-1]  # (R, K, degree, 2)
        P1s = P[:, 1:]  # (R, K, degree, 2)
        dP = P1s - P0s
        seg_len = complex_norm_ri(dP)  # (R, K)

        ts = make_uniform_ts(M, device=device, dtype=dtype)  # (M,)
        t = ts.view(1, 1, M, 1, 1)  # (1,1,M,1,1)
        gamma = (1.0 - t) * P0s.unsqueeze(2) + t * P1s.unsqueeze(2)  # (R, K, M, degree, 2)

        gamma_flat = gamma.reshape(R * K * M, self.degree, 2)  # (R*K*M, degree, 2)
        a_ri = p_to_monic_poly_coeffs_ri(gamma_flat)  # (R*K*M, degree+1, 2)

        disc_logabs = discriminant_univariate_logabs(
            a_ri,
            eps=cfg.disc_eps,
            lead_eps=cfg.lead_eps,
            backend=getattr(cfg, "disc_backend", "complex"),
        )
        disc_logabs = disc_logabs.view(R, K, M)  # (R, K, M)

        # Stable log(softabs + eps)
        log_softabs = log_softabs_from_logabs(disc_logabs, cfg.delta_soft)
        log_softabs_eps = log_softabs_plus_eps(log_softabs, cfg.eps_soft)

        degree_f = torch.tensor(float(self.degree), device=device, dtype=dtype)
        log_denom = log_softabs_eps / degree_f

        # inv-only weight: w = 1 / denom = exp(-log_denom)
        w = torch.exp(-log_denom)  # (R, K, M)

        w_mean = w.mean(dim=2)  # (R, K)
        loss_vec = (seg_len * w_mean).sum(dim=1)  # (R,)

        diag = {
            "min_logabs_disc_vec": disc_logabs.amin(dim=(1, 2)).detach(),  # (R,)
            "mean_seg_len_vec": seg_len.mean(dim=1).detach(),  # (R,)
        }
        return loss_vec, diag

    def forward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Returns (scalar_loss, diagnostics). diagnostics includes per-path vectors."""
        P = self.control_points()
        L_cl_vec, diag = self._condition_length_like_loss_vec(P)
        L_sm_vec = self._smoothness_loss_vec(P)
        L_vec = L_cl_vec + self.loss_cfg.lambda_smooth * L_sm_vec

        diag.update(
            {
                "loss_cl_vec": L_cl_vec.detach(),
                "loss_smooth_vec": L_sm_vec.detach(),
                "loss_total_vec": L_vec.detach(),
                "loss_total_mean": L_vec.mean().detach(),
                "loss_total_sum": L_vec.sum().detach(),
            }
        )
        # Aggregate to a scalar for backward (SUM; avoids implicit 1/R scaling).
        return L_vec.sum(), diag


def optimize_piecewise_linear_paths_batched(
    p_start: torch.Tensor,
    p_target: torch.Tensor,
    num_segments: int,
    *,
    num_paths: int = 8,
    init: str = "linear",
    init_imag_noise_scale: float = 0.0,
    init_seeds: Sequence[int] | None = None,
    loss_cfg: PiecewiseLossConfig = PiecewiseLossConfig(),
    optim_cfg: OptimConfig = OptimConfig(),
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
    return_cpu: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Optimizes R piecewise linear paths in parallel (batched on GPU)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if init_seeds is None:
        init_seeds = list(range(int(num_paths)))
    else:
        init_seeds = list(init_seeds)
        if len(init_seeds) != int(num_paths):
            raise ValueError("init_seeds must have length num_paths.")

    model = BatchedPiecewiseLinearHCPathOptimizer(
        p_start=p_start.to(device=device),
        p_target=p_target.to(device=device),
        num_segments=num_segments,
        num_paths=int(num_paths),
        init=init,
        init_imag_noise_scale=init_imag_noise_scale,
        init_seeds=init_seeds,
        loss_cfg=loss_cfg,
    ).to(device=device, dtype=dtype)

    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        P = model.control_points().detach()
        if return_cpu:
            P = P.cpu()
        return P, {
            "note": "K=1, no optimization variables.",
            "R": int(num_paths),
            "K": int(num_segments),
            "degree": int(model.degree),
            "device": str(device),
            "dtype": str(dtype),
        }

    opt = torch.optim.Adam(params, lr=optim_cfg.lr)

    R = int(num_paths)
    best_loss = torch.full((R,), float("inf"), device=device, dtype=torch.float64)
    best_P = model.control_points().detach().clone()  # (R, K+1, degree, 2)
    best_min_logabs_disc = torch.full((R,), float("inf"), device=device, dtype=torch.float64)
    best_loss_cl = torch.full((R,), float("inf"), device=device, dtype=torch.float64)
    best_loss_smooth = torch.full((R,), float("inf"), device=device, dtype=torch.float64)

    for step in range(int(optim_cfg.steps)):
        opt.zero_grad(set_to_none=True)
        loss, diag = model()
        # IMPORTANT: `diag` corresponds to the current control points (pre-step).
        # Update best *before* applying `opt.step()` to keep best_* aligned with best_P.
        with torch.no_grad():
            loss_vec = diag["loss_total_vec"]  # (R,)
            mask = loss_vec < best_loss
            if mask.any():
                curP = model.control_points().detach()
                best_loss = torch.where(mask, loss_vec.to(best_loss.dtype), best_loss)
                best_loss_cl = torch.where(mask, diag["loss_cl_vec"].to(best_loss_cl.dtype), best_loss_cl)
                best_loss_smooth = torch.where(mask, diag["loss_smooth_vec"].to(best_loss_smooth.dtype), best_loss_smooth)
                best_P[mask] = curP[mask]
                best_min_logabs_disc = torch.where(mask, diag["min_logabs_disc_vec"].to(best_min_logabs_disc.dtype), best_min_logabs_disc)

        loss.backward()
        if optim_cfg.grad_clip is not None:
            # IMPORTANT: clip_grad_norm_ would couple paths (global norm). Clip PER PATH instead.
            g = model.P_mid.grad
            if g is not None:
                g_flat = g.reshape(R, -1)
                norms = torch.linalg.vector_norm(g_flat, ord=2, dim=1)  # (R,)
                max_norm = float(optim_cfg.grad_clip)
                scale = (max_norm / (norms + 1e-12)).clamp(max=1.0)  # (R,)
                g.mul_(scale.view(R, 1, 1, 1))
        opt.step()

        if optim_cfg.print_every and (step % int(optim_cfg.print_every) == 0 or step == int(optim_cfg.steps) - 1):
            with torch.no_grad():
                msg = (
                    f"[batched step {step:04d}] "
                    f"loss_mean={diag['loss_total_mean'].item():.6e} "
                    f"best_loss_mean={best_loss.mean().item():.6e} "
                    f"min_log|Disc|_min={diag['min_logabs_disc_vec'].min().item():.6e}"
                )
            print(msg)

    P_out = best_P.detach()
    if return_cpu:
        P_out = P_out.cpu()
        best_loss_out = best_loss.detach().cpu()
        best_min_disc_out = best_min_logabs_disc.detach().cpu()
        best_loss_cl_out = best_loss_cl.detach().cpu()
        best_loss_smooth_out = best_loss_smooth.detach().cpu()
    else:
        best_loss_out = best_loss.detach()
        best_min_disc_out = best_min_logabs_disc.detach()
        best_loss_cl_out = best_loss_cl.detach()
        best_loss_smooth_out = best_loss_smooth.detach()

    info = {
        "best_loss_vec": best_loss_out,
        "best_loss_cl_vec": best_loss_cl_out,
        "best_loss_smooth_vec": best_loss_smooth_out,
        "best_min_logabs_disc_vec": best_min_disc_out,
        "device": str(device),
        "dtype": str(dtype),
        "R": int(num_paths),
        "K": int(num_segments),
        "degree": int(model.degree),
        "init": init,
        "init_imag_noise_scale": float(init_imag_noise_scale),
        "init_seeds": list(init_seeds),
        "loss_cfg": loss_cfg,
        "optim_cfg": optim_cfg,
    }
    return P_out, info


