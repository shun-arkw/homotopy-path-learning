"""Log-domain stability helpers. No dependency on task2."""
from __future__ import annotations

import torch


def make_uniform_ts(M: int, device, dtype) -> torch.Tensor:
    """Returns M uniformly spaced interior points in (0, 1). Midpoints (m + 0.5)/M."""
    m = torch.arange(M, device=device, dtype=dtype)
    return (m + 0.5) / M


def log_softabs_from_logabs(logabs: torch.Tensor, delta: float) -> torch.Tensor:
    """log(softabs(z)) from log(|z|). softabs(z) = sqrt(|z|^2 + delta)."""
    if delta <= 0:
        return logabs
    log_delta = torch.log(torch.tensor(delta, device=logabs.device, dtype=logabs.dtype))
    return 0.5 * torch.logaddexp(2.0 * logabs, log_delta)


def log_softabs_plus_eps(log_softabs: torch.Tensor, eps: float) -> torch.Tensor:
    """log(softabs(z) + eps) stably."""
    if eps <= 0:
        return log_softabs
    log_eps = torch.log(torch.tensor(eps, device=log_softabs.device, dtype=log_softabs.dtype))
    return torch.logaddexp(log_softabs, log_eps)
