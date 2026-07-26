from __future__ import annotations

import numpy as np
import pytest

from homotopy_path_learning.performance import measure_repeated, summarize_durations_ns, timed_call


def test_timing_stats_are_finite_and_match_manual_values() -> None:
    stats = summarize_durations_ns(
        "metric",
        [1_000_000_000, 2_000_000_000, 3_000_000_000],
        items_per_sample=2,
        paths_per_sample=4,
    )

    assert stats.count == 3
    assert stats.mean_seconds == 2.0
    assert stats.median_seconds == 2.0
    assert stats.std_seconds == pytest.approx(float(np.std([1.0, 2.0, 3.0], ddof=0)))
    assert stats.p50_seconds == 2.0
    assert stats.p95_seconds == pytest.approx(float(np.percentile([1.0, 2.0, 3.0], 95)))
    assert stats.throughput_per_second == 1.0
    assert stats.problem_seconds == 1.0
    assert stats.path_seconds == 0.5


def test_empty_timing_results_are_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        summarize_durations_ns("empty", [])


def test_timed_call_and_measure_repeated() -> None:
    result, elapsed = timed_call(lambda: "ok")
    assert result == "ok"
    assert elapsed >= 0

    calls = {"count": 0}

    def run() -> int:
        calls["count"] += 1
        return calls["count"]

    stats, last = measure_repeated("repeat", run, warmup=1, repeat=2, trials=3)
    assert stats.count == 6
    assert last == 7
    assert calls["count"] == 7
