"""Process pool that evaluates episode resets and fixed-action costs in parallel.

Each worker process owns its own ``BezierPhamEnv`` (and Julia backend). The
results of a task depend only on its inputs, so they do not depend on the number
of workers or on which worker runs a task.
"""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from homotopy_path_learning.config import Phase6Config, build_env
from homotopy_path_learning.envs.bezier_pham import BezierPhamEpisodeContext

_ENV = None


def _init_worker(config: Phase6Config) -> None:
    """Set the Julia runtime variables, import Julia, and build this worker's env."""

    global _ENV
    repository_root = Path(__file__).resolve().parents[3]
    os.environ.setdefault("PYTHON_JULIACALL_EXE", "/usr/local/bin/julia")
    os.environ.setdefault("PYTHON_JULIACALL_PROJECT", str(repository_root / "julia"))
    os.environ.setdefault("JULIAPKG_OFF", "1")
    os.environ.setdefault("JULIA_CONDAPKG_OFF", "1")
    os.environ.setdefault("JULIA_PYTHONCALL_INSTALL", "never")
    from juliacall import Main as _jl  # noqa: F401

    _ENV = build_env(config)


def _reset_task(seed: int) -> tuple[NDArray[np.float32], BezierPhamEpisodeContext]:
    observation, _ = _ENV.reset(seed=int(seed))
    return np.asarray(observation, dtype=np.float32), _ENV.current_episode_context()


def _cost_task(item: tuple[BezierPhamEpisodeContext, NDArray[np.float64]]) -> float:
    context, action = item
    return float(_ENV.evaluate_action_against_context(context, action).tracking_cost.mean)


class EnvPool:
    """``workers`` spawned processes, each with its own environment.

    Use as a context manager so the pool is always shut down. Spawn (not fork)
    is used because the Julia runtime is not fork-safe.
    """

    def __init__(self, config: Phase6Config, workers: int) -> None:
        if int(workers) < 1:
            raise ValueError("workers must be at least 1.")
        self.workers = int(workers)
        self._pool = None
        # Spawned children re-import the main script, which must find its sibling
        # modules; the CLI bootstrap removes its own directory from sys.path.
        main_file = getattr(sys.modules.get("__main__"), "__file__", None)
        saved = list(sys.path)
        if main_file:
            sys.path.insert(0, str(Path(main_file).resolve().parent))
        try:
            self._pool = multiprocessing.get_context("spawn").Pool(
                self.workers, initializer=_init_worker, initargs=(config,)
            )
        finally:
            sys.path[:] = saved

    def _chunk(self, n: int) -> int:
        return max(1, -(-n // (self.workers * 4)))

    def reset_many(
        self, seeds: Sequence[int]
    ) -> tuple[list[NDArray[np.float32]], list[BezierPhamEpisodeContext]]:
        """Reset the environment at each seed; return observations ``(obs_dim,)`` float32 and contexts."""

        results = self._pool.map(_reset_task, [int(s) for s in seeds], chunksize=self._chunk(len(seeds)))
        return [r[0] for r in results], [r[1] for r in results]

    def mean_costs(
        self, contexts: Sequence[BezierPhamEpisodeContext], actions: Sequence[NDArray[np.float64]]
    ) -> list[float]:
        """Mean tracking cost of ``actions[i]`` against ``contexts[i]`` (order preserved)."""

        if len(contexts) != len(actions):
            raise ValueError("contexts and actions must have the same length.")
        items = [(c, np.asarray(a, dtype=np.float64)) for c, a in zip(contexts, actions)]
        return self._pool.map(_cost_task, items, chunksize=self._chunk(len(items)))

    def close(self) -> None:
        """Terminate the worker processes (idempotent)."""

        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

    def __enter__(self) -> "EnvPool":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


__all__ = ["EnvPool"]
