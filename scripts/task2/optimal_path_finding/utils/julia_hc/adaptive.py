from __future__ import annotations

import numpy as np

from .tracker import JuliaTrackerOptions


def _clamp_int(x: float, lo: int, hi: int) -> int:
    xi = int(np.ceil(float(x)))
    if xi < int(lo):
        return int(lo)
    if xi > int(hi):
        return int(hi)
    return xi


def _clamp_float(x: float, lo: float, hi: float) -> float:
    xf = float(x)
    if xf < float(lo):
        return float(lo)
    if xf > float(hi):
        return float(hi)
    return float(xf)


def adaptive_julia_tracker_opts_per_segment(
    cl_per_seg: np.ndarray,
    *,
    base_opts: JuliaTrackerOptions,
    alpha_steps: float = 0.5,
    beta_step_size: float = 0.5,
    min_steps: int = 5,
    max_steps: int = 50_000,
    min_max_step_size: float = 1e-4,
    max_max_step_size: float = 0.2,
    eps: float = 1e-30,
) -> list[JuliaTrackerOptions]:
    """
    Map per-segment condition lengths to per-segment Julia tracker options.

    Heuristic:
      - smaller condition length => fewer max_steps, larger max_step_size
      - non-finite (inf/nan) => fall back to base options (likely to fail anyway)
    """

    cl = np.asarray(cl_per_seg, dtype=np.float64).reshape(-1)
    finite = np.isfinite(cl)
    if finite.any():
        cl_ref = float(np.median(np.maximum(cl[finite], float(eps))))
    else:
        cl_ref = 1.0

    opts_list: list[JuliaTrackerOptions] = []
    for v in cl.tolist():
        if not np.isfinite(v) or v <= 0.0:
            opts_list.append(base_opts)
            continue
        vv = max(float(v), float(eps))
        scale_steps = (vv / cl_ref) ** float(alpha_steps)
        scale_step = (cl_ref / vv) ** float(beta_step_size)

        opts_list.append(
            JuliaTrackerOptions(
                max_steps=_clamp_int(base_opts.max_steps * scale_steps, int(min_steps), int(max_steps)),
                max_step_size=_clamp_float(
                    base_opts.max_step_size * scale_step, float(min_max_step_size), float(max_max_step_size)
                ),
                max_initial_step_size=_clamp_float(
                    base_opts.max_initial_step_size * scale_step, float(min_max_step_size), float(max_max_step_size)
                ),
                min_step_size=float(base_opts.min_step_size),
                min_rel_step_size=float(base_opts.min_rel_step_size),
                extended_precision=bool(base_opts.extended_precision),
            )
        )
    return opts_list


