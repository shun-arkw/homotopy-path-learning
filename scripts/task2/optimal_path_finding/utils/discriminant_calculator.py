import torch
import time

from .complex_repr import complex_abs_from_ri, pack_ri, to_ri


# ============================================================
# Public API (users should start reading here)
# ============================================================

def discriminant_univariate_logabs(
    a: torch.Tensor,
    *,
    eps: float = 0.0,
    lead_eps: float = 0.0,
    backend: str = "complex",
) -> torch.Tensor:
    """Returns only log|Disc(f)| (skips exp), for performance-critical callers.

    Args:
        a: Polynomial coefficients in one of the accepted formats (see `to_ri`).
        eps: Optional stabilization added to Sylvester matrix (or its real block form).
        lead_eps: Stabilization for |a_n| in the division term.
        backend: "complex" (fast k x k complex slogdet) or "real_block" (2k x 2k real block slogdet).
    """
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
    """
    Backward-compatible wrapper.
    If a is complex or (...,n+1,2), returns (...,n,2).
    If a is real (...,n+1), returns (...,n).
    """
    a_ri = to_ri(a)
    a_re, a_im = a_ri[..., 0], a_ri[..., 1]
    da_re, da_im = poly_derivative_coeffs_ri(a_re, a_im)
    if torch.is_complex(a):
        return da_re + 1j * da_im
    if a.ndim >= 2 and a.shape[-1] == 2:
        return pack_ri(da_re, da_im)
    return da_re


# ============================================================
# Resultant (logabs) backends
# ============================================================

def resultant_univariate_logabs_complex(
    a_re: torch.Tensor,
    a_im: torch.Tensor,
    b_re: torch.Tensor,
    b_im: torch.Tensor,
    *,
    eps: float = 0.0,
) -> torch.Tensor:
    """Compute log|Res_x(f,g)| using complex Sylvester matrix slogdet (no exp)."""
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
    """
    Compute log|Res_x(f,g)| using the real 2k x 2k block matrix formulation.

    Identity:
      det([[A,-B],[B,A]]) = |det(A+iB)|^2  =>  log|det(A+iB)| = 0.5 * log(det(block))
    """
    M = sylvester_matrix_univariate_real_block(a_re, a_im, b_re, b_im)
    if eps and eps > 0:
        k2 = M.shape[-1]
        I = torch.eye(k2, dtype=M.dtype, device=M.device).expand_as(M)
        M = M + eps * I
    _, logabs2 = torch.linalg.slogdet(M)  # real
    return 0.5 * logabs2


# ============================================================
# Sylvester matrix construction
# ============================================================

def _complex_dtype_from_real(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.float32:
        return torch.complex64
    if dtype == torch.float64:
        return torch.complex128
    # fallback: let PyTorch decide (may error for unsupported dtypes)
    return torch.complex64


def sylvester_matrix_univariate_complex(
    a_re: torch.Tensor,
    a_im: torch.Tensor,
    b_re: torch.Tensor,
    b_im: torch.Tensor,
) -> torch.Tensor:
    """
    Build complex Sylvester matrix S(a,b) directly as a complex tensor.

    This avoids the 2x-sized real block matrix construction and is usually much faster on GPU.
    Shapes follow `sylvester_matrix_univariate_ri`:
      a_* shape (..., m+1), b_* shape (..., n+1) (descending powers)
    Returns:
      S with shape (..., m+n, m+n) and complex dtype.
    """
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
    a = a.to(dtype=cdtype)
    b = b.to(dtype=cdtype)

    k = m + n
    S_top = torch.zeros((*batch_shape, n, k), dtype=cdtype, device=device)
    if n > 0:
        cols_top = (
            torch.arange(m + 1, device=device).unsqueeze(0)
            + torch.arange(n, device=device).unsqueeze(1)
        )  # (n, m+1)
        cols_top = cols_top.expand(*batch_shape, n, m + 1)
        S_top.scatter_(dim=-1, index=cols_top, src=a.unsqueeze(-2).expand(*batch_shape, n, m + 1))

    S_bot = torch.zeros((*batch_shape, m, k), dtype=cdtype, device=device)
    if m > 0:
        cols_bot = (
            torch.arange(n + 1, device=device).unsqueeze(0)
            + torch.arange(m, device=device).unsqueeze(1)
        )  # (m, n+1)
        cols_bot = cols_bot.expand(*batch_shape, m, n + 1)
        S_bot.scatter_(dim=-1, index=cols_bot, src=b.unsqueeze(-2).expand(*batch_shape, m, n + 1))

    return torch.cat([S_top, S_bot], dim=-2)


def sylvester_matrix_univariate_ri(a_re: torch.Tensor, a_im: torch.Tensor,
                                  b_re: torch.Tensor, b_im: torch.Tensor):
    """
    Build Sylvester matrix S(a,b) = A + iB, returned as (A,B).

    a_* shape (..., m+1), b_* shape (..., n+1), descending powers.
    Returns:
      A, B with shape (..., m+n, m+n)
    """
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

    # ------------------------------------------------------------
    # Vectorized construction (no Python loops):
    # Top block rows i=0..n-1 place a_* at cols i..i+m
    # Bottom block rows i=0..m-1 place b_* at cols i..i+n
    # ------------------------------------------------------------
    k = m + n  # matrix size

    # Top block: shape (..., n, k)
    A_top = torch.zeros((*batch_shape, n, k), dtype=dtype, device=device)
    B_top = torch.zeros((*batch_shape, n, k), dtype=dtype, device=device)
    if n > 0:
        cols_top = (torch.arange(m + 1, device=device).unsqueeze(0) + torch.arange(n, device=device).unsqueeze(1))  # (n, m+1)
        cols_top = cols_top.expand(*batch_shape, n, m + 1)
        A_top.scatter_(dim=-1, index=cols_top, src=a_re.to(dtype).unsqueeze(-2).expand(*batch_shape, n, m + 1))
        B_top.scatter_(dim=-1, index=cols_top, src=a_im.to(dtype).unsqueeze(-2).expand(*batch_shape, n, m + 1))

    # Bottom block: shape (..., m, k)
    A_bot = torch.zeros((*batch_shape, m, k), dtype=dtype, device=device)
    B_bot = torch.zeros((*batch_shape, m, k), dtype=dtype, device=device)
    if m > 0:
        cols_bot = (torch.arange(n + 1, device=device).unsqueeze(0) + torch.arange(m, device=device).unsqueeze(1))  # (m, n+1)
        cols_bot = cols_bot.expand(*batch_shape, m, n + 1)
        A_bot.scatter_(dim=-1, index=cols_bot, src=b_re.to(dtype).unsqueeze(-2).expand(*batch_shape, m, n + 1))
        B_bot.scatter_(dim=-1, index=cols_bot, src=b_im.to(dtype).unsqueeze(-2).expand(*batch_shape, m, n + 1))

    A = torch.cat([A_top, A_bot], dim=-2)
    B = torch.cat([B_top, B_bot], dim=-2)
    return A, B


def real_block_from_complex_parts(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Map complex matrix S=A+iB (A,B real) to real block matrix:
      [[A, -B],
       [B,  A]]
    """
    top = torch.cat([A, -B], dim=-1)
    bot = torch.cat([B,  A], dim=-1)
    return torch.cat([top, bot], dim=-2)


def sylvester_matrix_univariate_real_block(
    a_re: torch.Tensor,
    a_im: torch.Tensor,
    b_re: torch.Tensor,
    b_im: torch.Tensor,
) -> torch.Tensor:
    """Build the 2k x 2k real block Sylvester matrix [[A,-B],[B,A]]."""
    A, B = sylvester_matrix_univariate_ri(a_re, a_im, b_re, b_im)
    return real_block_from_complex_parts(A, B)


# ============================================================
# derivative coefficients (real/imag separated)
# ============================================================

def poly_derivative_coeffs_ri(a_re: torch.Tensor, a_im: torch.Tensor):
    """
    Derivative coefficients for f with complex coefficients, stored as (Re,Im).

    Input:
      a_re, a_im: (..., n+1) descending powers [a_n, ..., a_0]
    Output:
      da_re, da_im: (..., n) descending powers for f'(x)
    """
    if a_re.ndim < 1 or a_im.ndim < 1:
        raise ValueError("a_re and a_im must be tensors with last dim = degree+1")
    if a_re.shape != a_im.shape:
        raise ValueError("a_re and a_im must have the same shape")

    n = a_re.shape[-1] - 1
    if n <= 0:
        z = torch.zeros((*a_re.shape[:-1], 1), dtype=a_re.dtype, device=a_re.device)
        return z, z

    mult = torch.arange(n, 0, -1, device=a_re.device, dtype=a_re.dtype)  # [n, ..., 1]
    da_re = a_re[..., :-1] * mult
    da_im = a_im[..., :-1] * mult
    return da_re, da_im


# ============================================================
# sanity checks
# ============================================================

if __name__ == "__main__":
    # Example 1: real quadratic, Disc = 1
    a = torch.tensor([1.0, -3.0, 2.0], requires_grad=True)
    disc_logabs = discriminant_univariate_logabs(a)
    disc_abs = torch.exp(disc_logabs)
    print("real disc_abs =", disc_abs.item(), "disc_logabs =", disc_logabs.item())

    loss = (disc_abs - 1.0) ** 2
    loss.backward()
    print("grad(real) =", a.grad)

    # Example 2: complex coefficients given as (Re,Im)
    # f(x) = x^2 + (1+i)x + (2-3i)
    a2_ri = torch.tensor([[1.0, 0.0],
                          [1.0, 1.0],
                          [2.0, -3.0]], requires_grad=True)  # shape (3,2) = (n+1,2)
    disc_logabs2 = discriminant_univariate_logabs(a2_ri)
    disc_abs2 = torch.exp(disc_logabs2)
    print("complex disc_abs =", disc_abs2.item(), "disc_logabs =", disc_logabs2.item())

    loss2 = disc_logabs2 ** 2
    loss2.backward()
    print("grad(complex, re/im packed) =", a2_ri.grad)

    # Example 3: complex tensor input
    a3 = torch.tensor([1.0 + 0.0j, -6.0 + 0.5j, 11.0 - 0.3j, -6.0 + 0.0j], requires_grad=True)
    t0 = time.time()
    disc_logabs3 = discriminant_univariate_logabs(a3, eps=0.0)
    disc_abs3 = torch.exp(disc_logabs3)
    t1 = time.time()
    print("complex-tensor disc_abs =", disc_abs3.item(), "time =", (t1 - t0))
