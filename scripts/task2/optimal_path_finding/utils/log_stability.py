from __future__ import annotations

import torch


def make_uniform_ts(M: int, device, dtype) -> torch.Tensor:
    """Returns M uniformly spaced interior points in (0, 1).

    Uses midpoints (m + 0.5)/M to avoid endpoints.

    Args:
        M: Number of samples.
        device: Torch device.
        dtype: Torch dtype.

    Returns:
        Tensor of shape (M,) with values in (0, 1).
    """
    m = torch.arange(M, device=device, dtype=dtype)
    return (m + 0.5) / M


def log_softabs_from_logabs(logabs: torch.Tensor, delta: float) -> torch.Tensor:
    """Computes log(softabs(z)) from log(|z|) in a numerically stable way.

    We define:
        softabs(z) = sqrt(|z|^2 + delta)

    Then:
        log(softabs(z)) = 0.5 * log(|z|^2 + delta)
                        = 0.5 * log(exp(2*log|z|) + delta)
                        = 0.5 * logaddexp(2*log|z|, log(delta))

    Args:
        logabs: Tensor containing log(|z|).
        delta: Positive constant delta in softabs.

    Returns:
        Tensor containing log(softabs(z)).
    """
    if delta <= 0:
        return logabs
    log_delta = torch.log(torch.tensor(delta, device=logabs.device, dtype=logabs.dtype))
    return 0.5 * torch.logaddexp(2.0 * logabs, log_delta)


def log_softabs_plus_eps(log_softabs: torch.Tensor, eps: float) -> torch.Tensor:
    """Computes log(softabs(z) + eps) stably.

    We use:
        log(softabs + eps) = logaddexp(log(softabs), log(eps))

    Args:
        log_softabs: Tensor containing log(softabs(z)).
        eps: Nonnegative epsilon.

    Returns:
        Tensor containing log(softabs(z) + eps).
    """
    if eps <= 0:
        return log_softabs
    log_eps = torch.log(torch.tensor(eps, device=log_softabs.device, dtype=log_softabs.dtype))
    return torch.logaddexp(log_softabs, log_eps)


