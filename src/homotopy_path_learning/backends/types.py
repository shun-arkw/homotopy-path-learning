"""Shared backend data types."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


def _is_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _require_positive_int(name: str, value: object) -> int:
    if not _is_int(value):
        raise TypeError(f"{name} must be an integer, got {value!r}.")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive, got {normalized}.")
    return normalized


def _require_positive_finite_float(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a finite positive float, got {value!r}.")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite positive float, got {value!r}.") from exc
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    if normalized <= 0.0:
        raise ValueError(f"{name} must be positive, got {normalized}.")
    return normalized


SUPPORTED_PREDICTOR_ORDERS = (1, 3)


def _require_open_unit_float(name: str, value: object, *, upper_inclusive: bool) -> float:
    normalized = _require_positive_finite_float(name, value)
    if normalized > 1.0 or (not upper_inclusive and normalized == 1.0):
        bound = "<= 1" if upper_inclusive else "< 1"
        raise ValueError(f"{name} must be {bound}, got {normalized}.")
    return normalized


@dataclass(frozen=True)
class TrackerConfig:
    """Configuration for HomotopyContinuation.jl tracking.

    Attributes:
        max_steps: Positive maximum number of predictor-corrector steps.
        max_step_size: Positive finite maximum step size.
        max_initial_step_size: Positive finite maximum initial step size.
        min_step_size: Positive finite minimum step size. It must not exceed
            ``max_step_size`` or ``max_initial_step_size``.
        extended_precision: Boolean flag forwarded to the Julia tracker
            options.
        a, beta_a, beta_omega_p, beta_tau, strict_beta_tau, min_newton_iters:
            HomotopyContinuation.jl ``TrackerParameters`` (``a``, ``β_a``,
            ``β_ω_p``, ``β_τ``, ``strict_β_τ``, ``min_newton_iters``). The
            defaults are ``DEFAULT_TRACKER_PARAMETERS``; ``strict_beta_tau=None``
            means HomotopyContinuation.jl's rule ``min(0.75 * beta_tau, 0.4)``.
        predictor_order: Highest Taylor order passed to the predictor, 1 (the
            default; first-order data only, as in all experiments up to
            EXP-0052) or 3 (what the Pade (2,1) predictor needs).
    """

    max_steps: int = 50_000
    max_step_size: float = 0.05
    max_initial_step_size: float = 0.05
    min_step_size: float = 1e-12
    extended_precision: bool = False
    a: float = 0.125
    beta_a: float = 1.0
    beta_omega_p: float = 3.0
    beta_tau: float = 0.4
    strict_beta_tau: float | None = None
    min_newton_iters: int = 2
    predictor_order: int = 1

    def __post_init__(self) -> None:
        max_steps = _require_positive_int("max_steps", self.max_steps)
        max_step_size = _require_positive_finite_float("max_step_size", self.max_step_size)
        max_initial_step_size = _require_positive_finite_float(
            "max_initial_step_size",
            self.max_initial_step_size,
        )
        min_step_size = _require_positive_finite_float("min_step_size", self.min_step_size)
        if min_step_size > max_step_size:
            raise ValueError("min_step_size must be <= max_step_size.")
        if min_step_size > max_initial_step_size:
            raise ValueError("min_step_size must be <= max_initial_step_size.")
        if not isinstance(self.extended_precision, bool):
            raise TypeError("extended_precision must be a bool.")
        a = _require_open_unit_float("a", self.a, upper_inclusive=False)
        beta_a = _require_positive_finite_float("beta_a", self.beta_a)
        beta_omega_p = _require_positive_finite_float("beta_omega_p", self.beta_omega_p)
        beta_tau = _require_open_unit_float("beta_tau", self.beta_tau, upper_inclusive=True)
        strict_beta_tau = (
            None
            if self.strict_beta_tau is None
            else _require_open_unit_float("strict_beta_tau", self.strict_beta_tau, upper_inclusive=True)
        )
        min_newton_iters = _require_positive_int("min_newton_iters", self.min_newton_iters)
        predictor_order = _require_positive_int("predictor_order", self.predictor_order)
        if predictor_order not in SUPPORTED_PREDICTOR_ORDERS:
            raise ValueError(
                f"predictor_order must be one of {SUPPORTED_PREDICTOR_ORDERS}, got {predictor_order}."
            )

        object.__setattr__(self, "a", a)
        object.__setattr__(self, "beta_a", beta_a)
        object.__setattr__(self, "beta_omega_p", beta_omega_p)
        object.__setattr__(self, "beta_tau", beta_tau)
        object.__setattr__(self, "strict_beta_tau", strict_beta_tau)
        object.__setattr__(self, "min_newton_iters", min_newton_iters)
        object.__setattr__(self, "predictor_order", predictor_order)
        object.__setattr__(self, "max_steps", max_steps)
        object.__setattr__(self, "max_step_size", max_step_size)
        object.__setattr__(self, "max_initial_step_size", max_initial_step_size)
        object.__setattr__(self, "min_step_size", min_step_size)


@dataclass(frozen=True)
class TrackingResult:
    """Python-owned tracking result arrays.

    All NumPy arrays are copied during construction so a result never keeps a
    direct reference to Julia-owned memory.

    Attributes:
        success: Overall success flag, equal to ``n_failed == 0``.
        n_paths: Total number of tracked paths.
        n_success: Number of successful paths.
        n_failed: Number of failed paths.
        accepted_steps: Total accepted step count.
        rejected_steps: Total rejected step count.
        per_path_accepted_steps: Integer array with shape ``(n_paths,)``.
        per_path_rejected_steps: Integer array with shape ``(n_paths,)``.
        path_success: Boolean array with shape ``(n_paths,)``.
        endpoints: Complex endpoint array with shape ``(n_paths, n_vars)`` and
            dtype ``complex128``.
        residual_norms: Residual norm array with shape ``(n_paths,)`` and dtype
            ``float64``.
        failure_codes: Tuple of strings with length ``n_paths``.
    """

    success: bool
    n_paths: int
    n_success: int
    n_failed: int
    accepted_steps: int
    rejected_steps: int
    per_path_accepted_steps: npt.NDArray[np.integer]
    per_path_rejected_steps: npt.NDArray[np.integer]
    path_success: npt.NDArray[np.bool_]
    endpoints: npt.NDArray[np.complex128]
    residual_norms: npt.NDArray[np.float64]
    failure_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.success, bool):
            raise TypeError("success must be a bool.")

        n_paths = _require_nonnegative_int("n_paths", self.n_paths)
        n_success = _require_nonnegative_int("n_success", self.n_success)
        n_failed = _require_nonnegative_int("n_failed", self.n_failed)
        accepted_steps = _require_nonnegative_int("accepted_steps", self.accepted_steps)
        rejected_steps = _require_nonnegative_int("rejected_steps", self.rejected_steps)

        if n_success + n_failed != n_paths:
            raise ValueError("n_success + n_failed must equal n_paths.")
        if self.success != (n_failed == 0):
            raise ValueError("success must equal (n_failed == 0).")

        accepted = np.array(self.per_path_accepted_steps, dtype=np.int64, copy=True)
        rejected = np.array(self.per_path_rejected_steps, dtype=np.int64, copy=True)
        path_success = np.array(self.path_success, dtype=np.bool_, copy=True)
        endpoints = np.array(self.endpoints, dtype=np.complex128, copy=True)
        residuals = np.array(self.residual_norms, dtype=np.float64, copy=True)
        failure_codes = tuple(str(code) for code in self.failure_codes)

        expected_vector_shape = (n_paths,)
        if accepted.shape != expected_vector_shape:
            raise ValueError(
                "per_path_accepted_steps must have shape "
                f"{expected_vector_shape}, got {accepted.shape}."
            )
        if rejected.shape != expected_vector_shape:
            raise ValueError(
                "per_path_rejected_steps must have shape "
                f"{expected_vector_shape}, got {rejected.shape}."
            )
        if path_success.shape != expected_vector_shape:
            raise ValueError(
                f"path_success must have shape {expected_vector_shape}, got {path_success.shape}."
            )
        if residuals.shape != expected_vector_shape:
            raise ValueError(
                f"residual_norms must have shape {expected_vector_shape}, got {residuals.shape}."
            )
        if endpoints.ndim != 2 or endpoints.shape[0] != n_paths:
            raise ValueError(
                "endpoints must have shape (n_paths, n_vars), got "
                f"{endpoints.shape}."
            )
        if len(failure_codes) != n_paths:
            raise ValueError(
                f"failure_codes must have length n_paths={n_paths}, got {len(failure_codes)}."
            )
        if np.any(accepted < 0) or np.any(rejected < 0):
            raise ValueError("per-path step counts must be nonnegative.")
        if int(np.sum(accepted)) != accepted_steps:
            raise ValueError("accepted_steps must equal sum(per_path_accepted_steps).")
        if int(np.sum(rejected)) != rejected_steps:
            raise ValueError("rejected_steps must equal sum(per_path_rejected_steps).")
        if int(np.count_nonzero(path_success)) != n_success:
            raise ValueError("n_success must equal count(path_success).")
        successful_residuals = residuals[path_success]
        if not np.all(np.isfinite(successful_residuals)):
            raise ValueError("residual_norms for successful paths must be finite.")

        object.__setattr__(self, "n_paths", n_paths)
        object.__setattr__(self, "n_success", n_success)
        object.__setattr__(self, "n_failed", n_failed)
        object.__setattr__(self, "accepted_steps", accepted_steps)
        object.__setattr__(self, "rejected_steps", rejected_steps)
        object.__setattr__(self, "per_path_accepted_steps", accepted)
        object.__setattr__(self, "per_path_rejected_steps", rejected)
        object.__setattr__(self, "path_success", path_success)
        object.__setattr__(self, "endpoints", endpoints)
        object.__setattr__(self, "residual_norms", residuals)
        object.__setattr__(self, "failure_codes", failure_codes)


def _require_nonnegative_int(name: str, value: object) -> int:
    if not _is_int(value):
        raise TypeError(f"{name} must be an integer, got {value!r}.")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be nonnegative, got {normalized}.")
    return normalized


__all__ = ["TrackerConfig", "TrackingResult"]
