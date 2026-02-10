# hc_envs/julia_backend.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np
from juliacall import Main as jl


@dataclass(frozen=True)
class BezierUnivarConfig:
    degree: int
    bezier_degree: int
    seed: int
    compute_newton_iters: bool = False
    max_steps: int = 50_000
    max_step_size: float = 0.05 # 0.05
    max_initial_step_size: float = 0.05 # 0.05
    min_step_size: float = 1e-12
    min_rel_step_size: float = 1e-12
    extended_precision: bool = False


@dataclass(frozen=True)
class LinearUnivarConfig:
    degree: int
    seed: int
    compute_newton_iters: bool = False
    max_steps: int = 50_000
    max_step_size: float = 0.05 # 0.05
    max_initial_step_size: float = 0.05 # 0.05
    min_step_size: float = 1e-12
    min_rel_step_size: float = 1e-12
    extended_precision: bool = False


class JuliaBackend:
    """
    Process-wide backend for Julia init + warmup.
    - Assumes a single Julia session per process (juliacall Main).
    - Caches initialization by (degree, bezier_degree, extended_precision, compute_newton_iters).
    """

    def __init__(self) -> None:
        self.jl = jl
        self._ready: Dict[Tuple[int, int, bool, bool], bool] = {}
        self._ready_linear: Dict[Tuple[int, bool, bool], bool] = {}

    def ensure_ready(self, cfg: BezierUnivarConfig, warmup_ctrl: Optional[np.ndarray] = None) -> None:
        key = (cfg.degree, cfg.bezier_degree, cfg.extended_precision, cfg.compute_newton_iters)
        if self._ready.get(key, False):
            return

        # Initialize Julia-side cached state for the given (degree, bezier_degree).
        self.jl.init_bezier_univar(
            degree=int(cfg.degree),
            bezier_degree=int(cfg.bezier_degree),
            seed=int(cfg.seed),
            compute_newton_iters=bool(cfg.compute_newton_iters),
            extended_precision=bool(cfg.extended_precision),
            max_steps=int(cfg.max_steps),
            max_step_size=float(cfg.max_step_size),
            max_initial_step_size=float(cfg.max_initial_step_size),
            min_step_size=float(cfg.min_step_size),
            min_rel_step_size=float(cfg.min_rel_step_size),
        )

        # Warmup: run once to trigger JIT compilation on the exact execution path.
        if warmup_ctrl is None:
            warmup_ctrl = self._make_dummy_ctrl(cfg.degree, cfg.bezier_degree, cfg.seed)

        _ = self.jl.track_bezier_paths_univar(
            int(cfg.degree),
            int(cfg.bezier_degree),
            warmup_ctrl,
            compute_newton_iters=bool(cfg.compute_newton_iters),
        )

        self._ready[key] = True

    def ensure_ready_linear(self, cfg: LinearUnivarConfig, warmup_endpoints: Optional[np.ndarray] = None) -> None:
        key = (cfg.degree, cfg.extended_precision, cfg.compute_newton_iters)
        if self._ready_linear.get(key, False):
            return

        self.jl.init_linear_univar(
            degree=int(cfg.degree),
            seed=int(cfg.seed),
            compute_newton_iters=bool(cfg.compute_newton_iters),
            extended_precision=bool(cfg.extended_precision),
            max_steps=int(cfg.max_steps),
            max_step_size=float(cfg.max_step_size),
            max_initial_step_size=float(cfg.max_initial_step_size),
            min_step_size=float(cfg.min_step_size),
            min_rel_step_size=float(cfg.min_rel_step_size),
        )

        if warmup_endpoints is None:
            warmup_endpoints = self._make_dummy_linear_endpoints(cfg.degree, cfg.seed)

        _ = self.jl.track_linear_paths_univar(
            int(cfg.degree),
            warmup_endpoints[0],
            warmup_endpoints[1],
            compute_newton_iters=bool(cfg.compute_newton_iters),
        )

        self._ready_linear[key] = True

    @staticmethod
    def _make_dummy_ctrl(degree: int, bezier_degree: int, seed: int) -> np.ndarray:
        """
        Dummy control points with the correct shape for JIT warmup.
        Success is not required; avoid NaNs by using small random values.
        """
        rng = np.random.default_rng(seed)
        D = degree + 1
        d = bezier_degree
        re = 1e-3 * rng.standard_normal(size=(d + 1, D))
        im = 1e-3 * rng.standard_normal(size=(d + 1, D))
        ctrl = (re + 1j * im).astype(np.complex128)
        return ctrl

    @staticmethod
    def _make_dummy_linear_endpoints(degree: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        D = degree + 1
        re = 1e-3 * rng.standard_normal(size=(2, D))
        im = 1e-3 * rng.standard_normal(size=(2, D))
        endpoints = (re + 1j * im).astype(np.complex128)
        return endpoints


# ---- Module-level singleton / one-time include ----

_BACKEND: Optional[JuliaBackend] = None
_INCLUDED: bool = False


def get_backend(
    include_path: str = 'include("scripts/bezier_hc_ppo/bezier_univar.jl")',
    include_linear_path: str = 'include("scripts/bezier_hc_ppo/linear_univar.jl")',
) -> JuliaBackend:
    """
    Return a process-wide singleton backend.
    Also ensures Julia code is included exactly once per process.
    """
    global _BACKEND, _INCLUDED

    if _BACKEND is None:
        _BACKEND = JuliaBackend()

    if not _INCLUDED:
        # Include Julia definitions once per process.
        jl.seval(include_path)
        jl.seval(include_linear_path)
        _INCLUDED = True

    return _BACKEND
