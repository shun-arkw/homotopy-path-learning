"""Complex (Re, Im) representation helpers. No dependency on task2."""
from __future__ import annotations

import torch


def to_ri(x: torch.Tensor) -> torch.Tensor:
    """Converts to real tensor with explicit (Re, Im) last dimension."""
    if torch.is_complex(x):
        return torch.stack([x.real, x.imag], dim=-1)
    if x.ndim >= 2 and x.shape[-1] == 2:
        return x
    return torch.stack([x, torch.zeros_like(x)], dim=-1)


def pack_ri(x_re: torch.Tensor, x_im: torch.Tensor) -> torch.Tensor:
    """Packs separate real/imag tensors into (..., 2) (Re, Im)."""
    return torch.stack([x_re, x_im], dim=-1)


def complex_abs_from_ri(x_re: torch.Tensor, x_im: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """|x| = sqrt(x_re^2 + x_im^2 + eps)."""
    return torch.sqrt(x_re * x_re + x_im * x_im + (eps if eps > 0 else 0.0))


def complex_norm_ri(x_ri: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """Euclidean norm of complex vector in (Re, Im) format."""
    re = x_ri[..., 0]
    im = x_ri[..., 1]
    sq = (re * re + im * im).sum(dim=-1)
    if eps > 0:
        sq = sq + eps
    return torch.sqrt(sq)
