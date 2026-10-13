"""Sequence Transformer actor (comparison baseline without permutation structure).

Implements section 10 of ``docs/design/implementation/attention_policy.md``
(EXP-0022). Each free coefficient is one token, in the order of
``np.flatnonzero(spec.free_coefficient_mask)``. Tokens carry only
``[Re, Im]`` of the coefficient plus a learned positional embedding, so the
network has no built-in permutation symmetry. The same block layout as
:class:`~homotopy_path_learning.rl.attention.AttentionActor` is used (two
pre-LN self-attention sublayers over all tokens, then a pre-LN FFN), which keeps
the parameter count comparable. It works for any support.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from homotopy_path_learning.rl.attention import _AttentionSublayer
from homotopy_path_learning.systems.spec import PolynomialSystemSpec


class _SequenceBlock(nn.Module):
    """Two full self-attention sublayers and a token-wise FFN (pre-LN, residual)."""

    def __init__(self, d_model: int, n_heads: int, ff_dim: int) -> None:
        super().__init__()
        self.attn1 = _AttentionSublayer(d_model, n_heads)
        self.attn2 = _AttentionSublayer(d_model, n_heads)
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ff_dim), nn.GELU(), nn.Linear(ff_dim, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` has shape ``(batch, D_free, d_model)``; the output has the same shape."""

        x = self.attn1(x)
        x = self.attn2(x)
        return x + self.ff(self.ff_norm(x))


class SequenceActor(nn.Module):
    """Sequence Transformer actor producing the pre-tanh latent action mean.

    Args:
        spec: system support (any support is accepted).
        basis: latent basis ``U``, array of shape ``(2 D_free, 2 D_free)``
            (square; ``ValueError`` otherwise). Stored as a float32 buffer
            ``basis`` and therefore part of the state dict.
        bezier_degree: ``d_b >= 2``; the action has ``d_b - 1`` blocks.
        d_model, n_heads, n_layers, ff_dim: model size (default 128, 4, 4, 256).

    ``forward(obs)`` takes ``obs`` of shape ``(batch, 2 D_free)`` laid out as
    ``[Re c_free, Im c_free]`` (``np.flatnonzero(spec.free_coefficient_mask)``
    order) and returns ``(batch, (d_b - 1) * 2 D_free)`` (the model dtype,
    float32 by default): the concatenation over ``k`` of ``u_k = U^T delta_k``.
    """

    def __init__(
        self,
        spec: PolynomialSystemSpec,
        basis: npt.ArrayLike,
        bezier_degree: int,
        *,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        ff_dim: int = 256,
    ) -> None:
        super().__init__()
        free_dim = int(np.count_nonzero(spec.free_coefficient_mask))
        real_dim = 2 * free_dim
        basis_array = np.asarray(basis, dtype=np.float64)
        if basis_array.shape != (real_dim, real_dim):
            raise ValueError(
                f"basis must be square with shape ({real_dim}, {real_dim}) (full latent space); "
                f"got {basis_array.shape}."
            )
        if isinstance(bezier_degree, bool) or int(bezier_degree) < 2:
            raise ValueError(f"bezier_degree must be an integer >= 2, got {bezier_degree!r}.")
        if min(d_model, n_heads, n_layers, ff_dim) <= 0 or d_model % n_heads != 0:
            raise ValueError("d_model, n_heads, n_layers, ff_dim must be positive and n_heads must divide d_model.")
        self.free_dim = free_dim
        self.real_dim = real_dim
        self.n_blocks = int(bezier_degree) - 1
        self.action_dim = self.n_blocks * real_dim
        self.register_buffer("basis", torch.as_tensor(basis_array, dtype=torch.float32))

        self.embed = nn.Linear(2, d_model)
        self.pos_embed = nn.Parameter(torch.randn(free_dim, d_model) * 0.02)
        self.blocks = nn.ModuleList(_SequenceBlock(d_model, n_heads, ff_dim) for _ in range(n_layers))
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2 * self.n_blocks)
        nn.init.orthogonal_(self.head.weight, 0.01)
        nn.init.constant_(self.head.bias, 0.0)

    def coefficient_delta(self, obs: torch.Tensor) -> torch.Tensor:
        """Return ``delta_k`` before the basis, shape ``(batch, d_b - 1, 2 D_free)``.

        Each row is a real free vector ``[Re delta_free, Im delta_free]`` in
        free-coefficient order (the perturbation of control point ``k``).
        """

        dtype = self.embed.weight.dtype
        x = obs.to(dtype=dtype)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2 or x.shape[1] != self.real_dim:
            raise ValueError(f"obs must have shape (batch, {self.real_dim}), got {tuple(obs.shape)}.")
        d = self.free_dim
        feat = torch.stack([x[:, :d], x[:, d:]], dim=-1)  # (B, D, 2): [Re, Im] per free coefficient
        h = self.embed(feat) + self.pos_embed
        for block in self.blocks:
            h = block(h)
        out = self.head(self.final_norm(h))  # (B, D, 2K)
        out = out.reshape(x.shape[0], d, self.n_blocks, 2)
        delta = torch.cat([out[..., 0], out[..., 1]], dim=1)  # (B, 2D, K)
        return delta.transpose(1, 2)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the pre-tanh action mean, shape ``(batch, (d_b - 1) * 2 D_free)``."""

        delta = self.coefficient_delta(obs)
        u = delta @ self.basis.to(delta.dtype)  # row form of u_k = U^T delta_k
        return u.reshape(delta.shape[0], self.action_dim)


__all__ = ["SequenceActor"]
