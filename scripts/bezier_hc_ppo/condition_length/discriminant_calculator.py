"""Univariate polynomial discriminant (log|Disc|). Uses .complex_repr only."""
from __future__ import annotations

import torch

from .complex_repr import complex_abs_from_ri, pack_ri, to_ri


def discriminant_univariate_logabs(
    a: torch.Tensor,
    *,
    eps: float = 0.0,
    lead_eps: float = 0.0,
    backend: str = "complex",
) -> torch.Tensor:
    """Returns log|Disc(f)|. a: coeffs (see to_ri). lead_eps: stabilization for |a_n|."""
    a_ri = to_ri(a)
    a_re, a_im = a_ri[..., 0], a_ri[..., 1]
    n = a_re.shape[-1] - 1
    if n <= 0:
        return torch.zeros(a_re.shape[:-1], dtype=a_re.dtype, device=a_re.device)
    fp_re, fp_im = poly_derivative_coeffs_ri(a_re, a_im)
    if backend == "complex":
        res_logabs = resultant_univariate_logabs_complex(a_re, a_im, fp_re, fp_im, eps=eps)
    elif backend in ("real_block", "real", "block"):
        res_logabs = resultant_univariate_logabs_real_block(a_re, a_im, fp_re, fp_im, eps=eps)
    else:
        raise ValueError("backend must be 'complex' or 'real_block'.")
    a_lead_abs = complex_abs_from_ri(a_re[..., 0], a_im[..., 0], eps=lead_eps)
    return res_logabs - torch.log(a_lead_abs)


def poly_derivative_coeffs(a: torch.Tensor) -> torch.Tensor:
    """Backward-compatible wrapper: (..., n+1) -> (..., n) derivative coeffs."""
    a_ri = to_ri(a)
    a_re, a_im = a_ri[..., 0], a_ri[..., 1]
    da_re, da_im = poly_derivative_coeffs_ri(a_re, a_im)
    if torch.is_complex(a):
        return da_re + 1j * da_im
    if a.ndim >= 2 and a.shape[-1] == 2:
        return pack_ri(da_re, da_im)
    return da_re


def resultant_univariate_logabs_complex(
    a_re: torch.Tensor,
    a_im: torch.Tensor,
    b_re: torch.Tensor,
    b_im: torch.Tensor,
    *,
    eps: float = 0.0,
) -> torch.Tensor:
    """log|Res(f,g)| via complex Sylvester slogdet."""
    S = sylvester_matrix_univariate_complex(a_re, a_im, b_re, b_im)
    if eps and eps > 0:
        k = S.shape[-1]
        I = torch.eye(k, dtype=S.dtype, device=S.device).expand_as(S)
        S = S + eps * I
    _, logabs = torch.linalg.slogdet(S)
    return logabs


def resultant_univariate_logabs_real_block(
    a_re: torch.Tensor,
    a_im: torch.Tensor,
    b_re: torch.Tensor,
    b_im: torch.Tensor,
    *,
    eps: float = 0.0,
) -> torch.Tensor:
    """log|Res(f,g)| via real 2k x 2k block matrix."""
    M = sylvester_matrix_univariate_real_block(a_re, a_im, b_re, b_im)
    if eps and eps > 0:
        k2 = M.shape[-1]
        I = torch.eye(k2, dtype=M.dtype, device=M.device).expand_as(M)
        M = M + eps * I
    _, logabs2 = torch.linalg.slogdet(M)
    return 0.5 * logabs2


def _complex_dtype_from_real(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.float32:
        return torch.complex64
    if dtype == torch.float64:
        return torch.complex128
    return torch.complex64


def sylvester_matrix_univariate_complex(
    a_re: torch.Tensor,
    a_im: torch.Tensor,
    b_re: torch.Tensor,
    b_im: torch.Tensor,
) -> torch.Tensor:
    """Build complex Sylvester matrix S(a,b). a_*, b_* (..., deg+1) descending."""
    m = a_re.shape[-1] - 1
    n = b_re.shape[-1] - 1
    if m < 0 or n < 0:
        raise ValueError("degrees must be >= 0")
    batch_shape = torch.broadcast_shapes(a_re.shape[:-1], b_re.shape[:-1])
    a_re = a_re.expand(*batch_shape, m + 1)
    a_im = a_im.expand(*batch_shape, m + 1)
    b_re = b_re.expand(*batch_shape, n + 1)
    b_im = b_im.expand(*batch_shape, n + 1)
    real_dtype = torch.promote_types(
        torch.promote_types(a_re.dtype, a_im.dtype),
        torch.promote_types(b_re.dtype, b_im.dtype),
    )
    device = a_re.device
    cdtype = _complex_dtype_from_real(real_dtype)
    a = torch.complex(a_re.to(real_dtype), a_im.to(real_dtype))
    b = torch.complex(b_re.to(real_dtype), b_im.to(real_dtype))
    a, b = a.to(dtype=cdtype), b.to(dtype=cdtype)
    k = m + n
    S_top = torch.zeros((*batch_shape, n, k), dtype=cdtype, device=device)
    if n > 0:
        cols_top = (
            torch.arange(m + 1, device=device).unsqueeze(0)
            + torch.arange(n, device=device).unsqueeze(1)
        )
        cols_top = cols_top.expand(*batch_shape, n, m + 1)
        S_top.scatter_(dim=-1, index=cols_top, src=a.unsqueeze(-2).expand(*batch_shape, n, m + 1))
    S_bot = torch.zeros((*batch_shape, m, k), dtype=cdtype, device=device)
    if m > 0:
        cols_bot = (
            torch.arange(n + 1, device=device).unsqueeze(0)
            + torch.arange(m, device=device).unsqueeze(1)
        )
        cols_bot = cols_bot.expand(*batch_shape, m, n + 1)
        S_bot.scatter_(dim=-1, index=cols_bot, src=b.unsqueeze(-2).expand(*batch_shape, m, n + 1))
    return torch.cat([S_top, S_bot], dim=-2)


def sylvester_matrix_univariate_ri(
    a_re: torch.Tensor, a_im: torch.Tensor,
    b_re: torch.Tensor, b_im: torch.Tensor,
):
    """Build Sylvester S(a,b)=A+iB as (A, B) real."""
    m = a_re.shape[-1] - 1
    n = b_re.shape[-1] - 1
    if m < 0 or n < 0:
        raise ValueError("degrees must be >= 0")
    batch_shape = torch.broadcast_shapes(a_re.shape[:-1], b_re.shape[:-1])
    a_re = a_re.expand(*batch_shape, m + 1)
    a_im = a_im.expand(*batch_shape, m + 1)
    b_re = b_re.expand(*batch_shape, n + 1)
    b_im = b_im.expand(*batch_shape, n + 1)
    dtype = torch.promote_types(
        torch.promote_types(a_re.dtype, a_im.dtype),
        torch.promote_types(b_re.dtype, b_im.dtype),
    )
    device = a_re.device
    k = m + n
    A_top = torch.zeros((*batch_shape, n, k), dtype=dtype, device=device)
    B_top = torch.zeros((*batch_shape, n, k), dtype=dtype, device=device)
    if n > 0:
        cols_top = (
            torch.arange(m + 1, device=device).unsqueeze(0)
            + torch.arange(n, device=device).unsqueeze(1)
        )
        cols_top = cols_top.expand(*batch_shape, n, m + 1)
        A_top.scatter_(dim=-1, index=cols_top, src=a_re.to(dtype).unsqueeze(-2).expand(*batch_shape, n, m + 1))
        B_top.scatter_(dim=-1, index=cols_top, src=a_im.to(dtype).unsqueeze(-2).expand(*batch_shape, n, m + 1))
    A_bot = torch.zeros((*batch_shape, m, k), dtype=dtype, device=device)
    B_bot = torch.zeros((*batch_shape, m, k), dtype=dtype, device=device)
    if m > 0:
        cols_bot = (
            torch.arange(n + 1, device=device).unsqueeze(0)
            + torch.arange(m, device=device).unsqueeze(1)
        )
        cols_bot = cols_bot.expand(*batch_shape, m, n + 1)
        A_bot.scatter_(dim=-1, index=cols_bot, src=b_re.to(dtype).unsqueeze(-2).expand(*batch_shape, m, n + 1))
        B_bot.scatter_(dim=-1, index=cols_bot, src=b_im.to(dtype).unsqueeze(-2).expand(*batch_shape, m, n + 1))
    A = torch.cat([A_top, A_bot], dim=-2)
    B = torch.cat([B_top, B_bot], dim=-2)
    return A, B


def real_block_from_complex_parts(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """[[A,-B],[B,A]]."""
    return torch.cat([torch.cat([A, -B], dim=-1), torch.cat([B, A], dim=-1)], dim=-2)


def sylvester_matrix_univariate_real_block(
    a_re: torch.Tensor, a_im: torch.Tensor,
    b_re: torch.Tensor, b_im: torch.Tensor,
) -> torch.Tensor:
    """2k x 2k real block Sylvester."""
    A, B = sylvester_matrix_univariate_ri(a_re, a_im, b_re, b_im)
    return real_block_from_complex_parts(A, B)


def poly_derivative_coeffs_ri(a_re: torch.Tensor, a_im: torch.Tensor):
    """(..., n+1) descending -> (..., n) derivative coeffs."""
    if a_re.ndim < 1 or a_im.ndim < 1:
        raise ValueError("a_re and a_im must be tensors with last dim = degree+1")
    if a_re.shape != a_im.shape:
        raise ValueError("a_re and a_im must have the same shape")
    n = a_re.shape[-1] - 1
    if n <= 0:
        z = torch.zeros((*a_re.shape[:-1], 1), dtype=a_re.dtype, device=a_re.device)
        return z, z
    mult = torch.arange(n, 0, -1, device=a_re.device, dtype=a_re.dtype)
    da_re = a_re[..., :-1] * mult
    da_im = a_im[..., :-1] * mult
    return da_re, da_im
