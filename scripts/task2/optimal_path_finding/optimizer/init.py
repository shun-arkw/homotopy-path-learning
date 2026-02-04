from __future__ import annotations

from typing import Sequence

import torch


def initialize_control_points_qr(
    p_start_ri: torch.Tensor,
    p_target_ri: torch.Tensor,
    num_segments: int,
    *,
    noise_scale: float = 0.05,
    seed: int | None = None,
) -> torch.Tensor:
    """Initializes control points using QR-based orthogonal perturbations (placeholder).

    Intended design (to be implemented later):
      1) Map C^degree to R^{2*degree} and compute the unit direction u from start to target.
      2) Use QR decomposition to build an orthonormal basis of the orthogonal complement u^⊥.
      3) Add small random perturbations in u^⊥ to intermediate points to avoid
         getting stuck near discriminant singularities.

    Args:
        p_start_ri: Start point, shape (degree, 2).
        p_target_ri: Target point, shape (degree, 2).
        num_segments: Number of line segments K (control points are K+1).
        noise_scale: Magnitude of orthogonal perturbations.
        seed: Optional random seed.

    Returns:
        Control points in (Re, Im) format with shape (K+1, degree, 2).

    Raises:
        NotImplementedError: This is a placeholder.
    """
    raise NotImplementedError("QR-based initialization will be implemented later.")


def initialize_control_points_linear(
    p_start_ri: torch.Tensor,
    p_target_ri: torch.Tensor,
    num_segments: int,
    *,
    imag_noise_scale: float = 0.0,
    seed: int | None = None,
) -> torch.Tensor:
    """Initializes control points by simple linear interpolation.

    Args:
        p_start_ri: Start point, shape (degree, 2).
        p_target_ri: Target point, shape (degree, 2).
        num_segments: Number of line segments K.
        imag_noise_scale: If > 0, add N(0, imag_noise_scale^2) noise to the *imaginary*
            part of intermediate control points (P_1..P_{K-1}) only. This enables detours
            into complex coefficient space even when endpoints are purely real.
        seed: Optional random seed for the initialization noise.

    Returns:
        Control points P_ri with shape (K+1, degree, 2).
    """
    K = int(num_segments)
    ts = torch.linspace(0.0, 1.0, K + 1, device=p_start_ri.device, dtype=p_start_ri.dtype)
    ts = ts.view(K + 1, 1, 1)
    P = (1.0 - ts) * p_start_ri.unsqueeze(0) + ts * p_target_ri.unsqueeze(0)

    # Optional: allow complex detours by perturbing only imaginary parts of midpoints.
    if imag_noise_scale and imag_noise_scale > 0 and K >= 2:
        gen = None
        if seed is not None:
            gen = torch.Generator(device=P.device)
            gen.manual_seed(int(seed))
        noise = torch.randn((K - 1, P.shape[1]), device=P.device, dtype=P.dtype, generator=gen)
        P[1:-1, :, 1] = P[1:-1, :, 1] + float(imag_noise_scale) * noise

    return P


def initialize_control_points_linear_batched(
    p_start_ri: torch.Tensor,
    p_target_ri: torch.Tensor,
    num_segments: int,
    *,
    imag_noise_scale: float = 0.0,
    seeds: Sequence[int] | None = None,
) -> torch.Tensor:
    """Batched linear initialization for multiple paths.

    Args:
        p_start_ri: Start points, shape (R, degree, 2).
        p_target_ri: Target points, shape (R, degree, 2).
        num_segments: Number of segments K.
        imag_noise_scale: If > 0, add noise to imaginary part of intermediate points.
        seeds: Optional list/sequence of length R for per-path deterministic noise.

    Returns:
        Control points tensor of shape (R, K+1, degree, 2).
    """
    if p_start_ri.ndim != 3 or p_start_ri.shape[-1] != 2:
        raise ValueError("p_start_ri must have shape (R, degree, 2).")
    if p_target_ri.shape != p_start_ri.shape:
        raise ValueError("p_target_ri must have the same shape as p_start_ri.")

    R = int(p_start_ri.shape[0])
    K = int(num_segments)
    device, dtype = p_start_ri.device, p_start_ri.dtype

    ts = torch.linspace(0.0, 1.0, K + 1, device=device, dtype=dtype).view(1, K + 1, 1, 1)
    P = (1.0 - ts) * p_start_ri.unsqueeze(1) + ts * p_target_ri.unsqueeze(1)  # (R, K+1, degree, 2)

    if imag_noise_scale and imag_noise_scale > 0 and K >= 2:
        if seeds is not None and len(seeds) != R:
            raise ValueError("If provided, seeds must have length R.")
        # Per-path seeded noise (loop is fine; heavy work is in the batched loss).
        for r in range(R):
            gen = None
            if seeds is not None:
                gen = torch.Generator(device=device)
                gen.manual_seed(int(seeds[r]))
            noise = torch.randn((K - 1, P.shape[2]), device=device, dtype=dtype, generator=gen)
            P[r, 1:-1, :, 1] = P[r, 1:-1, :, 1] + float(imag_noise_scale) * noise

    return P


def initialize_control_points_bezier_linear(
    p_start_ri: torch.Tensor,
    p_target_ri: torch.Tensor,
    bezier_degree: int,
    *,
    imag_noise_scale: float = 0.0,
    seed: int | None = None,
) -> torch.Tensor:
    """Linear interpolation initialization for Bezier control points.

    Endpoints are p_start and p_target. Intermediate control points are placed on the
    straight line, optionally adding N(0, imag_noise_scale^2) noise to imaginary parts.

    Args:
        p_start_ri: (degree, 2)
        p_target_ri: (degree, 2)
        bezier_degree: d >= 1 (number of control points is d+1)
        imag_noise_scale: noise scale for imaginary part of intermediate points
        seed: optional RNG seed

    Returns:
        P_ctrl_ri: (d+1, degree, 2)
    """
    d = int(bezier_degree)
    if d < 1:
        raise ValueError("bezier_degree must be >= 1.")
    if p_start_ri.shape != p_target_ri.shape:
        raise ValueError("p_start_ri and p_target_ri must have the same shape.")
    if p_start_ri.ndim != 2 or p_start_ri.shape[-1] != 2:
        raise ValueError("p_start_ri must have shape (degree, 2).")

    ts = torch.linspace(0.0, 1.0, d + 1, device=p_start_ri.device, dtype=p_start_ri.dtype).view(d + 1, 1, 1)
    P = (1.0 - ts) * p_start_ri.unsqueeze(0) + ts * p_target_ri.unsqueeze(0)

    if imag_noise_scale and imag_noise_scale > 0 and d >= 2:
        gen = None
        if seed is not None:
            gen = torch.Generator(device=P.device)
            gen.manual_seed(int(seed))
        noise = torch.randn((d - 1, P.shape[1]), device=P.device, dtype=P.dtype, generator=gen)
        P[1:-1, :, 1] = P[1:-1, :, 1] + float(imag_noise_scale) * noise

    return P


