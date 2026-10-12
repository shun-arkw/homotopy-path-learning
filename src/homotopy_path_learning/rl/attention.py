"""Permutation-equivariant attention actor for linear-plus-constant supports.

Implements ``docs/design/implementation/attention_policy.md`` (ADR-0007, stage
1). The free coefficients of the target system are arranged as a matrix
``A`` (``A[i, j]`` = coefficient of ``x_j`` in equation ``i``) and a vector
``c`` (constant terms). Edge tokens ``(i, j)`` and node tokens ``i`` are
processed by ``L`` pre-LN blocks of row attention, column attention and a
token-wise feed-forward network. The heads output coefficient perturbations
``delta_k`` for each intermediate Bezier control point; the mean of the latent
action is ``u_k = U^T delta_k``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from homotopy_path_learning.systems.spec import PolynomialSystemSpec


EDGE_FEATURES = 5
NODE_FEATURES = 3


def linear_constant_index_maps(
    spec: PolynomialSystemSpec,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Locate ``A[i, j]`` and ``c[i]`` inside the free-coefficient vector.

    Returns ``(edge_pos, node_pos)`` with shapes ``(n, n)`` and ``(n,)`` and
    dtype ``int64``. Entry ``edge_pos[i, j]`` is the position, within
    ``np.flatnonzero(spec.free_coefficient_mask)``, of the coefficient of
    ``x_j`` in equation ``i``; ``node_pos[i]`` is the position of the constant
    term of equation ``i``. The maps are computed from ``spec.exponents`` only
    (no assumption on the coefficient order).

    Raises:
        NotImplementedError: if the free support of any equation is not
            exactly ``{x_1, ..., x_n, 1}``.
    """

    n = spec.n_vars
    free = np.flatnonzero(spec.free_coefficient_mask)
    position = {int(q): p for p, q in enumerate(free)}
    edge_pos = np.full((n, n), -1, dtype=np.int64)
    node_pos = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        count = 0
        for q in range(int(spec.offsets[i]), int(spec.offsets[i + 1])):
            if q == int(spec.leading_indices[i]):
                continue
            count += 1
            alpha = spec.exponents[q]
            total = int(alpha.sum())
            if total == 0:
                node_pos[i] = position[q]
            elif total == 1:
                edge_pos[i, int(np.argmax(alpha))] = position[q]
            else:
                raise NotImplementedError(
                    "AttentionActor supports only free monomials {x_1, ..., x_n, 1}; "
                    f"equation {i} has the free exponent {tuple(int(v) for v in alpha)}."
                )
        if count != n + 1 or np.any(edge_pos[i] < 0) or node_pos[i] < 0:
            raise NotImplementedError(
                "AttentionActor requires every equation to contain each of x_1, ..., x_n "
                f"and the constant term as free monomials; equation {i} does not."
            )
    return edge_pos, node_pos


class _AttentionSublayer(nn.Module):
    """Pre-LN multi-head self-attention with a residual connection."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` has shape ``(batch * groups, seq, d_model)``."""

        h = self.norm(x)
        out, _ = self.attn(h, h, h, need_weights=False)
        return x + out


class _Block(nn.Module):
    """Row attention, column attention and token-wise FFN (all pre-LN, residual)."""

    def __init__(self, d_model: int, n_heads: int, ff_dim: int) -> None:
        super().__init__()
        self.row = _AttentionSublayer(d_model, n_heads)
        self.col = _AttentionSublayer(d_model, n_heads)
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ff_dim), nn.GELU(), nn.Linear(ff_dim, d_model))

    def forward(self, edge: torch.Tensor, node: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``edge``: ``(B, n, n, d)``; ``node``: ``(B, n, d)``."""

        batch, n, _, d = edge.shape
        # Row attention: sequence of equation i = [node i, edge (i, 1..n)].
        seq = torch.cat([node.unsqueeze(2), edge], dim=2).reshape(batch * n, n + 1, d)
        seq = self.row(seq).reshape(batch, n, n + 1, d)
        node, edge = seq[:, :, 0], seq[:, :, 1:]
        # Column attention: sequence of variable j = [node j, edge (1..n, j)].
        seq = torch.cat([node.unsqueeze(2), edge.transpose(1, 2)], dim=2).reshape(batch * n, n + 1, d)
        seq = self.col(seq).reshape(batch, n, n + 1, d)
        node, edge = seq[:, :, 0], seq[:, :, 1:].transpose(1, 2)
        edge = edge + self.ff(self.ff_norm(edge))
        node = node + self.ff(self.ff_norm(node))
        return edge, node


class AttentionActor(nn.Module):
    """Equivariant Transformer actor producing the pre-tanh latent action mean.

    Args:
        spec: system support; every equation's free monomials must be exactly
            ``{x_1, ..., x_n, 1}`` (``NotImplementedError`` otherwise).
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
        edge_pos, node_pos = linear_constant_index_maps(spec)
        n = spec.n_vars
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
        self.n_vars = n
        self.free_dim = free_dim
        self.real_dim = real_dim
        self.n_blocks = int(bezier_degree) - 1
        self.action_dim = self.n_blocks * real_dim

        # Token t (edge i*n+j first, then node i) sits at free position token_to_free[t].
        token_to_free = np.concatenate([edge_pos.reshape(-1), node_pos])
        free_to_token = np.argsort(token_to_free)
        degrees = torch.tensor(spec.degrees, dtype=torch.float32)
        self.register_buffer("edge_pos", torch.as_tensor(edge_pos.reshape(-1)), persistent=False)
        self.register_buffer("node_pos", torch.as_tensor(node_pos), persistent=False)
        self.register_buffer("free_to_token", torch.as_tensor(free_to_token), persistent=False)
        self.register_buffer("degrees", degrees, persistent=False)
        self.register_buffer("diagonal", torch.eye(n, dtype=torch.float32), persistent=False)
        self.register_buffer("basis", torch.as_tensor(basis_array, dtype=torch.float32))

        self.edge_embed = nn.Linear(EDGE_FEATURES, d_model)
        self.node_embed = nn.Linear(NODE_FEATURES, d_model)
        self.blocks = nn.ModuleList(_Block(d_model, n_heads, ff_dim) for _ in range(n_layers))
        self.final_norm = nn.LayerNorm(d_model)
        out = 2 * self.n_blocks
        self.edge_head = nn.Linear(d_model, out)
        self.node_head = nn.Linear(d_model, out)
        for head in (self.edge_head, self.node_head):
            nn.init.orthogonal_(head.weight, 0.01)
            nn.init.constant_(head.bias, 0.0)

    def _tokens(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = obs.shape[0]
        n, d = self.n_vars, self.free_dim
        re, im = obs[:, :d], obs[:, d:]
        a_re = re[:, self.edge_pos].reshape(batch, n, n)
        a_im = im[:, self.edge_pos].reshape(batch, n, n)
        deg = self.degrees.to(obs.dtype)
        diag = self.diagonal.to(obs.dtype).expand(batch, n, n)
        d_i = deg.view(1, n, 1).expand(batch, n, n)
        d_j = deg.view(1, 1, n).expand(batch, n, n)
        edge = torch.stack([a_re, a_im, diag, d_i, d_j], dim=-1)
        node = torch.stack(
            [re[:, self.node_pos], im[:, self.node_pos], deg.expand(batch, n)], dim=-1
        )
        return edge, node

    def coefficient_delta(self, obs: torch.Tensor) -> torch.Tensor:
        """Return ``delta_k`` before the basis, shape ``(batch, d_b - 1, 2 D_free)``.

        Each row is a real free vector ``[Re delta_free, Im delta_free]`` in
        free-coefficient order (the perturbation of control point ``k``).
        """

        dtype = self.edge_embed.weight.dtype
        x = obs.to(dtype=dtype)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2 or x.shape[1] != self.real_dim:
            raise ValueError(f"obs must have shape (batch, {self.real_dim}), got {tuple(obs.shape)}.")
        batch, n = x.shape[0], self.n_vars
        edge_feat, node_feat = self._tokens(x)
        edge = self.edge_embed(edge_feat)
        node = self.node_embed(node_feat)
        for block in self.blocks:
            edge, node = block(edge, node)
        edge = self.final_norm(edge)
        node = self.final_norm(node)
        edge_out = self.edge_head(edge).reshape(batch, n * n, self.n_blocks, 2)
        node_out = self.node_head(node).reshape(batch, n, self.n_blocks, 2)
        tokens = torch.cat([edge_out, node_out], dim=1)  # (B, D, K, 2) in token order
        free = tokens[:, self.free_to_token]  # free-coefficient order
        delta = torch.cat([free[..., 0], free[..., 1]], dim=1)  # (B, 2D, K)
        return delta.transpose(1, 2)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the pre-tanh action mean, shape ``(batch, (d_b - 1) * 2 D_free)``."""

        delta = self.coefficient_delta(obs)
        u = delta @ self.basis.to(delta.dtype)  # row form of u_k = U^T delta_k
        return u.reshape(delta.shape[0], self.action_dim)


__all__ = ["AttentionActor", "linear_constant_index_maps"]
