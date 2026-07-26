"""Small timing primitives for explicit Phase 7 performance runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import statistics
import time
from typing import Callable, TypeVar

import numpy as np


T = TypeVar("T")


@dataclass(frozen=True)
class TimingStats:
    """Summary statistics for wall-clock durations measured in nanoseconds.

    All time fields are stored in seconds. ``std_seconds`` uses population
    standard deviation with ``ddof=0``. ``throughput_per_second`` is
    ``items_per_sample * count / total_seconds``.
    """

    name: str
    count: int
    total_seconds: float
    mean_seconds: float
    median_seconds: float
    std_seconds: float
    min_seconds: float
    max_seconds: float
    p50_seconds: float
    p95_seconds: float
    throughput_per_second: float
    problem_seconds: float
    path_seconds: float

    def to_dict(self) -> dict[str, float | int | str]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)


def _finite_nonnegative_durations(durations_ns: list[int] | tuple[int, ...]) -> np.ndarray:
    if not durations_ns:
        raise ValueError("durations_ns must not be empty.")
    values = np.array(durations_ns, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("durations_ns must be one-dimensional.")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("durations_ns must contain finite nonnegative values.")
    return values


def summarize_durations_ns(
    name: str,
    durations_ns: list[int] | tuple[int, ...],
    *,
    items_per_sample: int = 1,
    paths_per_sample: int | None = None,
) -> TimingStats:
    """Summarize non-empty nanosecond durations.

    Args:
        name: Stable metric name.
        durations_ns: One or more nonnegative durations in nanoseconds.
        items_per_sample: Number of problems or operations represented by one
            timing sample. Used for throughput and ``problem_seconds``.
        paths_per_sample: Optional number of paths represented by one sample.
            If omitted, ``path_seconds`` equals ``problem_seconds``.
    """

    if not isinstance(name, str) or name.strip() == "":
        raise ValueError("name must be a non-empty string.")
    if not isinstance(items_per_sample, int) or isinstance(items_per_sample, bool):
        raise TypeError("items_per_sample must be an integer.")
    if items_per_sample <= 0:
        raise ValueError("items_per_sample must be positive.")
    if paths_per_sample is not None:
        if not isinstance(paths_per_sample, int) or isinstance(paths_per_sample, bool):
            raise TypeError("paths_per_sample must be an integer or None.")
        if paths_per_sample <= 0:
            raise ValueError("paths_per_sample must be positive.")

    durations = _finite_nonnegative_durations(durations_ns)
    seconds = durations / 1_000_000_000.0
    total = float(np.sum(seconds))
    count = int(seconds.size)
    throughput = float(items_per_sample * count / total) if total > 0.0 else float("inf")
    if not np.isfinite(throughput):
        throughput = 0.0
    problem_seconds = float(np.mean(seconds) / items_per_sample)
    path_divisor = paths_per_sample if paths_per_sample is not None else items_per_sample
    path_seconds = float(np.mean(seconds) / path_divisor)
    stats = TimingStats(
        name=name,
        count=count,
        total_seconds=total,
        mean_seconds=float(np.mean(seconds)),
        median_seconds=float(np.median(seconds)),
        std_seconds=float(np.std(seconds, ddof=0)),
        min_seconds=float(np.min(seconds)),
        max_seconds=float(np.max(seconds)),
        p50_seconds=float(np.percentile(seconds, 50)),
        p95_seconds=float(np.percentile(seconds, 95)),
        throughput_per_second=throughput,
        problem_seconds=problem_seconds,
        path_seconds=path_seconds,
    )
    for key, value in stats.to_dict().items():
        if key != "name" and not np.isfinite(float(value)):
            raise ValueError(f"timing statistic {key} must be finite.")
    return stats


def timed_call(func: Callable[[], T]) -> tuple[T, int]:
    """Call ``func`` and return ``(result, elapsed_ns)``."""

    start = time.perf_counter_ns()
    result = func()
    elapsed = time.perf_counter_ns() - start
    if elapsed < 0:
        raise RuntimeError("perf_counter_ns returned a negative duration.")
    return result, int(elapsed)


def measure_repeated(
    name: str,
    func: Callable[[], T],
    *,
    warmup: int,
    repeat: int,
    trials: int,
    items_per_sample: int = 1,
    paths_per_sample: int | None = None,
) -> tuple[TimingStats, T]:
    """Measure ``func`` after warmup and return stats plus the last result."""

    if warmup < 0 or repeat <= 0 or trials <= 0:
        raise ValueError("warmup must be nonnegative and repeat/trials must be positive.")
    last_result: T | None = None
    for _ in range(warmup):
        last_result = func()
    durations: list[int] = []
    for _ in range(trials):
        for _ in range(repeat):
            last_result, elapsed = timed_call(func)
            durations.append(elapsed)
    return (
        summarize_durations_ns(
            name,
            durations,
            items_per_sample=items_per_sample,
            paths_per_sample=paths_per_sample,
        ),
        last_result,
    )


__all__ = [
    "TimingStats",
    "measure_repeated",
    "summarize_durations_ns",
    "timed_call",
]
