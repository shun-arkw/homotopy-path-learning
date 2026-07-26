"""Typed backend protocol for homotopy path trackers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from homotopy_path_learning.backends.types import TrackerConfig, TrackingResult
from homotopy_path_learning.systems.spec import PolynomialSystemSpec


@runtime_checkable
class TrackerBackend(Protocol):
    """Common lifecycle for path tracking backends."""

    def initialize(
        self,
        system_spec: PolynomialSystemSpec,
        bezier_degree: int,
        tracker_config: TrackerConfig,
    ) -> None:
        """Prepare a fixed system support and Bezier degree for tracking."""

    def warmup(self) -> TrackingResult:
        """Run a backend warmup track using the currently initialized state."""

    def track(
        self,
        control_points: np.ndarray,
    ) -> TrackingResult:
        """Track all start solutions for Bezier control points."""

    def close(self) -> None:
        """Release backend state. Calling this multiple times is safe."""


__all__ = ["TrackerBackend"]
