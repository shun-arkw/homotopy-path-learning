from __future__ import annotations

import torch


def to_ri(x: torch.Tensor) -> torch.Tensor:
    """Converts a complex-valued vector into a real tensor with explicit real/imag parts.

    This project frequently represents complex-valued quantities using a real tensor with
    an explicit last dimension for the real and imaginary parts:

        x_ri[..., j, 0] = Re(x[..., j])
        x_ri[..., j, 1] = Im(x[..., j])
        where j ∈ {0, ..., degree - 1}.

    Args:
        x: Input tensor. Supported formats:
            - Complex tensor of shape (..., degree)
            - Real tensor of shape (..., degree, 2) interpreted as (Re, Im)
            - Real tensor of shape (..., degree) interpreted as purely real (Im = 0)

    Returns:
        Real tensor of shape (..., degree, 2) in (Re, Im) format.
    """
    if torch.is_complex(x):
        return torch.stack([x.real, x.imag], dim=-1)
    if x.ndim >= 2 and x.shape[-1] == 2:
        return x
    return torch.stack([x, torch.zeros_like(x)], dim=-1)


def pack_ri(x_re: torch.Tensor, x_im: torch.Tensor) -> torch.Tensor:
    """Packs separate real/imag tensors into a single (Re, Im) representation.

    This project often stores complex-valued quantities as a real tensor with an explicit
    last dimension for real and imaginary parts. Given separate real and imaginary tensors,
    this function packs them into the convention:

        x_ri[..., j, 0] = x_re[..., j]
        x_ri[..., j, 1] = x_im[..., j]
        where j ∈ {0, ..., degree - 1}.

    Unlike `to_ri`, this function does not accept complex inputs and does not perform any
    normalization or shape interpretation; it simply stacks the two provided tensors.

    Args:
        x_re: Real-part tensor of shape (..., degree).
        x_im: Imag-part tensor of shape (..., degree). Must have the same shape as `x_re`.

    Returns:
        Real tensor of shape (..., degree, 2) in (Re, Im) format.
    """
    return torch.stack([x_re, x_im], dim=-1)


def complex_abs_from_ri(x_re: torch.Tensor, x_im: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """Computes |x| = sqrt(x_re^2 + x_im^2 + eps).

    Args:
        x_re: Real-part tensor.
        x_im: Imaginary-part tensor. Must be broadcast-compatible with `x_re`.
        eps: Small nonnegative constant added inside sqrt for numerical stability.

    Returns:
        Real tensor containing the complex magnitude with the broadcasted shape of `x_re` and `x_im`.
    """
    return torch.sqrt(x_re * x_re + x_im * x_im + (eps if eps > 0 else 0.0))


def complex_norm_ri(x_ri: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """Computes the Euclidean norm of a complex vector represented in (Re, Im).

    We interpret C^degree as R^{2*degree}. For x = (x_1, ..., x_degree) with x_j = a_j + i b_j,
    the norm is:

        ||x|| = sqrt(sum_j (a_j^2 + b_j^2))

    Args:
        x_ri: Real tensor of shape (..., degree, 2) in (Re, Im) format.
        eps: Small nonnegative constant added inside sqrt for numerical stability.

    Returns:
        Real tensor of shape (...) with the Euclidean norm.
    """
    re = x_ri[..., 0]
    im = x_ri[..., 1]
    sq = (re * re + im * im).sum(dim=-1)
    if eps > 0:
        sq = sq + eps
    return torch.sqrt(sq)


def p_to_monic_poly_coeffs_ri(p_ri: torch.Tensor) -> torch.Tensor:
    """Builds monic polynomial coefficients from p = (a_{degree-1}, ..., a_0).

    We consider monic polynomials:
        f(x) = x^degree + a_{degree-1} x^{degree-1} + ... + a_0

    The optimizer state uses:
        p = (a_{degree-1}, ..., a_0) in C^degree

    This function converts p into the full coefficient list:
        a = [1, a_{degree-1}, ..., a_0]

    Args:
        p_ri: Real tensor of shape (..., degree, 2) in (Re, Im) format.

    Returns:
        Real tensor of shape (..., degree+1, 2) representing [1, a_{degree-1}, ..., a_0].
    """
    batch_shape = p_ri.shape[:-2]
    one = torch.zeros((*batch_shape, 1, 2), dtype=p_ri.dtype, device=p_ri.device)
    one[..., 0, 0] = 1.0
    return torch.cat([one, p_ri], dim=-2)


