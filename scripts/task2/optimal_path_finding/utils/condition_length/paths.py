from __future__ import annotations

import torch

from ..complex_repr import to_ri


def build_linear_control_points(
    p_start: torch.Tensor,
    p_target: torch.Tensor,
    num_segments: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Build control points for a straight-line path, but represented as K segments.

    This returns points on the straight line from p_start to p_target, uniformly spaced
    in parameter, with shape (K+1, degree, 2) in (Re, Im).

    Notes:
        We intentionally represent the straight line using the same number of segments K
        as the optimized polyline. This makes it easy to compare both paths using exactly
        the same discretization scheme (same K and same per-segment samples M).
    """
    if num_segments < 1:
        raise ValueError("num_segments must be >= 1.")

    p0 = to_ri(p_start)
    p1 = to_ri(p_target)
    if p0.shape != p1.shape:
        raise ValueError("p_start and p_target must have the same shape after conversion to (Re, Im).")
    if p0.ndim != 2 or p0.shape[-1] != 2:
        raise ValueError("Expected a single vector with shape (degree, 2) or complex shape (degree,).")

    if device is None:
        device = p0.device
    if dtype is None:
        dtype = p0.dtype

    p0 = p0.to(device=device, dtype=dtype)
    p1 = p1.to(device=device, dtype=dtype)

    K = int(num_segments)
    ts = torch.linspace(0.0, 1.0, K + 1, device=device, dtype=dtype).view(K + 1, 1, 1)
    return (1.0 - ts) * p0.unsqueeze(0) + ts * p1.unsqueeze(0)


