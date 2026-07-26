"""Python-to-Julia backend using :mod:`juliacall`.

The module intentionally does not import ``juliacall`` at import time. The
Julia runtime is loaded lazily by :class:`JuliaTrackerBackend`, after
``PYTHON_JULIACALL_EXE`` and ``PYTHON_JULIACALL_PROJECT`` have been configured.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
import threading
from typing import Any

import numpy as np
import numpy.typing as npt

from homotopy_path_learning.backends.types import TrackerConfig, TrackingResult
from homotopy_path_learning.systems.sampling import start_system_coefficients
from homotopy_path_learning.systems.spec import PolynomialSystemSpec


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_JULIA_PROJECT = _REPOSITORY_ROOT / "julia"
_DEFAULT_JULIA_EXE = "/usr/local/bin/julia"


class JuliaBackendError(RuntimeError):
    """Base class for Julia backend errors."""


class JuliaBackendValidationError(ValueError):
    """Raised when Python-side boundary validation fails."""


class BackendStateError(JuliaBackendError):
    """Raised when a backend method is called in the wrong lifecycle state."""


class BackendClosedError(BackendStateError):
    """Raised when a closed backend is used again."""


class JuliaRuntimeInitializationError(JuliaBackendError):
    """Raised when ``juliacall`` or the Julia runtime cannot start."""


class JuliaModuleLoadError(JuliaBackendError):
    """Raised when the HomotopyPathLearning Julia module cannot be loaded."""


class JuliaAPIError(JuliaBackendError):
    """Raised when a Julia public API call fails."""


class JuliaTrackingError(JuliaAPIError):
    """Raised when a Julia tracking API call raises an exception."""


class JuliaResultConversionError(JuliaBackendError):
    """Raised when a Julia tracking result cannot be converted safely."""


@dataclass(frozen=True)
class JuliaSystemArrays:
    """Python-owned arrays passed to ``from_python_spec`` on the Julia side.

    Attributes:
        degrees: Shape ``(n_vars,)``, dtype ``int64``.
        exponents: Shape ``(M, n_vars)``, dtype ``int64``.
        offsets: Shape ``(n_vars + 1,)``, dtype ``int64``. Python-side
            zero-based block offsets.
        leading_indices: Shape ``(n_vars,)``, dtype ``int64``. Python-side
            zero-based leading coefficient indices.
        constant_indices: Shape ``(n_vars,)``, dtype ``int64``. Python-side
            zero-based constant coefficient indices.
    """

    degrees: npt.NDArray[np.int64]
    exponents: npt.NDArray[np.int64]
    offsets: npt.NDArray[np.int64]
    leading_indices: npt.NDArray[np.int64]
    constant_indices: npt.NDArray[np.int64]


@dataclass(frozen=True)
class _JuliaRuntime:
    jl: Any
    make_tracker_options: Any
    init_bezier_pham: Any
    track_bezier_paths: Any
    warmup: Any
    clear_state: Any
    state_snapshot: Any
    state_initialized: Any
    tracking_result_payload: Any


_RUNTIME_LOCK = threading.RLock()
_RUNTIME: _JuliaRuntime | None = None


def _is_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _require_positive_int(name: str, value: object) -> int:
    if not _is_int(value):
        raise TypeError(f"{name} must be an integer, got {value!r}.")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive, got {normalized}.")
    return normalized


def _require_numpy_int64(name: str, value: object, *, ndim: int) -> npt.NDArray[np.int64]:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray.")
    if value.dtype != np.int64:
        raise TypeError(f"{name} must have dtype int64, got {value.dtype}.")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got ndim={value.ndim}.")
    return np.array(value, dtype=np.int64, copy=True)


def validate_system_arrays(
    *,
    degrees: npt.NDArray[np.int64],
    exponents: npt.NDArray[np.int64],
    offsets: npt.NDArray[np.int64],
    leading_indices: npt.NDArray[np.int64],
    constant_indices: npt.NDArray[np.int64],
) -> JuliaSystemArrays:
    """Validate Python-side arrays for the Julia boundary.

    The returned arrays are independent ``int64`` copies. Indices remain
    zero-based; Julia converts them to one-based indices in ``from_python_spec``.
    """

    degree_array = _require_numpy_int64("degrees", degrees, ndim=1)
    exponent_array = _require_numpy_int64("exponents", exponents, ndim=2)
    offset_array = _require_numpy_int64("offsets", offsets, ndim=1)
    leading_array = _require_numpy_int64("leading_indices", leading_indices, ndim=1)
    constant_array = _require_numpy_int64("constant_indices", constant_indices, ndim=1)

    n_vars = int(degree_array.shape[0])
    if n_vars <= 0:
        raise JuliaBackendValidationError("degrees must not be empty.")
    if np.any(degree_array <= 0):
        raise JuliaBackendValidationError("degrees must contain positive integers.")
    if exponent_array.shape[1] != n_vars:
        raise JuliaBackendValidationError(
            f"exponents must have shape (M, {n_vars}), got {exponent_array.shape}."
        )
    if np.any(exponent_array < 0):
        raise JuliaBackendValidationError("exponents must be nonnegative.")

    n_coeffs = int(exponent_array.shape[0])
    if offset_array.shape != (n_vars + 1,):
        raise JuliaBackendValidationError(
            f"offsets must have shape ({n_vars + 1},), got {offset_array.shape}."
        )
    if int(offset_array[0]) != 0:
        raise JuliaBackendValidationError(f"offsets[0] must be 0, got {offset_array[0]}.")
    if int(offset_array[-1]) != n_coeffs:
        raise JuliaBackendValidationError(
            f"offsets[-1] must equal M={n_coeffs}, got {offset_array[-1]}."
        )
    if np.any(np.diff(offset_array) <= 0):
        raise JuliaBackendValidationError("offsets must be strictly increasing.")

    expected_index_shape = (n_vars,)
    if leading_array.shape != expected_index_shape:
        raise JuliaBackendValidationError(
            f"leading_indices must have shape {expected_index_shape}, got {leading_array.shape}."
        )
    if constant_array.shape != expected_index_shape:
        raise JuliaBackendValidationError(
            "constant_indices must have shape "
            f"{expected_index_shape}, got {constant_array.shape}."
        )
    for name, values in (
        ("leading_indices", leading_array),
        ("constant_indices", constant_array),
    ):
        if np.any(values < 0) or np.any(values >= n_coeffs):
            raise JuliaBackendValidationError(f"{name} must be in [0, M).")

    for equation_index in range(n_vars):
        block_start = int(offset_array[equation_index])
        block_stop = int(offset_array[equation_index + 1])
        leading = int(leading_array[equation_index])
        constant = int(constant_array[equation_index])
        if not block_start <= leading < block_stop:
            raise JuliaBackendValidationError(
                f"leading_indices[{equation_index}] is outside its equation block."
            )
        if not block_start <= constant < block_stop:
            raise JuliaBackendValidationError(
                f"constant_indices[{equation_index}] is outside its equation block."
            )

    return JuliaSystemArrays(
        degrees=degree_array,
        exponents=exponent_array,
        offsets=offset_array,
        leading_indices=leading_array,
        constant_indices=constant_array,
    )


def system_spec_to_julia_arrays(spec: PolynomialSystemSpec) -> JuliaSystemArrays:
    """Build validated Python-owned boundary arrays from a system spec."""

    if not isinstance(spec, PolynomialSystemSpec):
        raise TypeError("system_spec must be a PolynomialSystemSpec.")
    return validate_system_arrays(
        degrees=np.array(spec.degrees, dtype=np.int64, copy=True),
        exponents=spec.exponents,
        offsets=spec.offsets,
        leading_indices=spec.leading_indices,
        constant_indices=spec.constant_indices,
    )


def validate_control_points_for_backend(
    control_points: npt.ArrayLike,
    *,
    system_spec: PolynomialSystemSpec,
    bezier_degree: int,
) -> npt.NDArray[np.complex128]:
    """Validate Bezier control points before calling Julia.

    The returned array has shape ``(bezier_degree + 1, M)`` and dtype
    ``complex128``. Safe dtype conversion is allowed, and the returned array is
    a Python-owned copy independent of the caller's input. Because the Phase 4
    backend is initialized without a target coefficient vector, this validation
    checks the start endpoint and fixed leading coefficients; target endpoint
    ownership remains with the caller that constructs the control points.
    """

    if not isinstance(system_spec, PolynomialSystemSpec):
        raise TypeError("system_spec must be a PolynomialSystemSpec.")
    degree = _require_positive_int("bezier_degree", bezier_degree)
    points = np.array(control_points, dtype=np.complex128, copy=True)
    expected_shape = (degree + 1, system_spec.n_coeffs)
    if points.ndim != 2 or points.shape != expected_shape:
        raise JuliaBackendValidationError(
            f"control_points must have shape {expected_shape}, got {points.shape}."
        )
    if not np.all(np.isfinite(points.real)) or not np.all(np.isfinite(points.imag)):
        raise JuliaBackendValidationError("control_points must not contain NaN or Inf.")

    start_coeffs = start_system_coefficients(system_spec)
    if not np.array_equal(points[0], start_coeffs):
        raise JuliaBackendValidationError("control_points[0] must equal the start coefficients.")
    expected_leading = np.ones(
        (degree + 1, system_spec.leading_indices.size),
        dtype=np.complex128,
    )
    if not np.array_equal(points[:, system_spec.leading_indices], expected_leading):
        raise JuliaBackendValidationError(
            "all control point leading coefficients must be exactly 1 + 0j."
        )
    return points


def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value[name]
    try:
        return getattr(value, name)
    except AttributeError:
        pass
    try:
        return value[name]
    except Exception as exc:  # pragma: no cover - depends on foreign objects.
        raise AttributeError(f"could not read field {name!r}.") from exc


def _failure_codes_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        return (str(value),)
    try:
        return tuple(str(code) for code in np.array(value, dtype=object, copy=True).reshape(-1))
    except ValueError:
        return tuple(str(code) for code in value)


def tracking_result_from_julia(value: Any, *, n_vars: int) -> TrackingResult:
    """Convert a Julia tracking result or payload to a Python ``TrackingResult``.

    Julia matrices are copied with ``np.array(..., copy=True)``. No view into
    Julia-owned memory is kept in the returned object.
    """

    expected_n_vars = _require_positive_int("n_vars", n_vars)
    try:
        result = TrackingResult(
            success=bool(_field(value, "success")),
            n_paths=int(_field(value, "n_paths")),
            n_success=int(_field(value, "n_success")),
            n_failed=int(_field(value, "n_failed")),
            accepted_steps=int(_field(value, "accepted_steps")),
            rejected_steps=int(_field(value, "rejected_steps")),
            per_path_accepted_steps=np.array(
                _field(value, "per_path_accepted_steps"),
                dtype=np.int64,
                copy=True,
            ),
            per_path_rejected_steps=np.array(
                _field(value, "per_path_rejected_steps"),
                dtype=np.int64,
                copy=True,
            ),
            path_success=np.array(_field(value, "path_success"), dtype=np.bool_, copy=True),
            endpoints=np.array(_field(value, "endpoints"), dtype=np.complex128, copy=True),
            residual_norms=np.array(
                _field(value, "residual_norms"),
                dtype=np.float64,
                copy=True,
            ),
            failure_codes=_failure_codes_tuple(_field(value, "failure_codes")),
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise JuliaResultConversionError("could not convert Julia TrackingResult.") from exc

    if result.endpoints.shape[1] != expected_n_vars:
        raise JuliaResultConversionError(
            f"endpoints must have shape (n_paths, {expected_n_vars}), "
            f"got {result.endpoints.shape}."
        )
    return result


def _configure_juliacall_environment(
    *,
    julia_executable: str | os.PathLike[str] | None,
    julia_project: str | os.PathLike[str] | None,
) -> None:
    if "juliacall" in sys.modules:
        return
    if julia_executable is not None and "PYTHON_JULIACALL_EXE" not in os.environ:
        os.environ["PYTHON_JULIACALL_EXE"] = str(julia_executable)
    else:
        os.environ.setdefault("PYTHON_JULIACALL_EXE", _DEFAULT_JULIA_EXE)

    if julia_project is not None and "PYTHON_JULIACALL_PROJECT" not in os.environ:
        os.environ["PYTHON_JULIACALL_PROJECT"] = str(julia_project)
    else:
        os.environ.setdefault("PYTHON_JULIACALL_PROJECT", str(_JULIA_PROJECT))

    os.environ.setdefault("JULIAPKG_OFF", "1")
    os.environ.setdefault("JULIA_CONDAPKG_OFF", "1")
    os.environ.setdefault("JULIA_PYTHONCALL_INSTALL", "never")


def _load_julia_runtime(
    *,
    julia_executable: str | os.PathLike[str] | None = None,
    julia_project: str | os.PathLike[str] | None = None,
) -> _JuliaRuntime:
    global _RUNTIME
    with _RUNTIME_LOCK:
        if _RUNTIME is not None:
            return _RUNTIME

        _configure_juliacall_environment(
            julia_executable=julia_executable,
            julia_project=julia_project,
        )
        try:
            from juliacall import Main as jl  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - exercised in integration failure.
            raise JuliaRuntimeInitializationError(
                "failed to initialize juliacall. Ensure PYTHON_JULIACALL_EXE points "
                "to the reference Julia executable and PYTHON_JULIACALL_PROJECT "
                "points to the repository julia project."
            ) from exc

        try:
            jl.seval("using HomotopyPathLearning")
        except Exception as exc:  # pragma: no cover - exercised in integration failure.
            raise JuliaModuleLoadError("failed to load Julia module HomotopyPathLearning.") from exc

        try:
            _RUNTIME = _JuliaRuntime(
                jl=jl,
                make_tracker_options=jl.seval("HomotopyPathLearning.make_tracker_options"),
                init_bezier_pham=jl.seval("HomotopyPathLearning.init_bezier_pham!"),
                track_bezier_paths=jl.seval("HomotopyPathLearning.track_bezier_paths!"),
                warmup=jl.seval("HomotopyPathLearning.warmup!"),
                clear_state=jl.seval("HomotopyPathLearning.clear_state!"),
                state_snapshot=jl.seval("HomotopyPathLearning.bezier_pham_state_snapshot"),
                state_initialized=jl.seval("HomotopyPathLearning.bezier_pham_state_initialized"),
                tracking_result_payload=jl.seval("HomotopyPathLearning.tracking_result_payload"),
            )
        except Exception as exc:  # pragma: no cover - exercised in integration failure.
            raise JuliaModuleLoadError("failed to bind HomotopyPathLearning Julia API.") from exc

        return _RUNTIME


def _make_tracker_options(runtime: _JuliaRuntime, config: TrackerConfig) -> Any:
    try:
        return runtime.make_tracker_options(
            max_steps=config.max_steps,
            max_step_size=config.max_step_size,
            max_initial_step_size=config.max_initial_step_size,
            min_step_size=config.min_step_size,
            extended_precision=config.extended_precision,
        )
    except Exception as exc:
        raise JuliaAPIError("failed to construct Julia tracker options.") from exc


def _snapshot_to_python(snapshot: Any) -> dict[str, Any]:
    return {
        "initialized": bool(_field(snapshot, "initialized")),
        "nvars": int(_field(snapshot, "nvars")),
        "bezier_degree": int(_field(snapshot, "bezier_degree")),
        "degrees": np.array(_field(snapshot, "degrees"), dtype=np.int64, copy=True),
        "exponents": np.array(_field(snapshot, "exponents"), dtype=np.int64, copy=True),
        "offsets": np.array(_field(snapshot, "offsets"), dtype=np.int64, copy=True),
        "leading_indices": np.array(
            _field(snapshot, "leading_indices"),
            dtype=np.int64,
            copy=True,
        ),
        "constant_indices": np.array(
            _field(snapshot, "constant_indices"),
            dtype=np.int64,
            copy=True,
        ),
        "control_points": np.array(
            _field(snapshot, "control_points"),
            dtype=np.complex128,
            copy=True,
        ),
        "starts": np.array(_field(snapshot, "starts"), dtype=np.complex128, copy=True),
    }


class JuliaTrackerBackend:
    """Tracker backend backed by the Phase 3 Julia public API."""

    def __init__(
        self,
        *,
        julia_executable: str | os.PathLike[str] | None = None,
        julia_project: str | os.PathLike[str] | None = None,
    ) -> None:
        self._julia_executable = julia_executable
        self._julia_project = julia_project
        self._runtime: _JuliaRuntime | None = None
        self._system_spec: PolynomialSystemSpec | None = None
        self._bezier_degree: int | None = None
        self._initialized = False
        self._closed = False

    def __enter__(self) -> "JuliaTrackerBackend":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    @property
    def initialized(self) -> bool:
        """Whether this Python backend instance has initialized Julia state."""

        return self._initialized and not self._closed

    @property
    def closed(self) -> bool:
        """Whether ``close()`` has been called on this backend instance."""

        return self._closed

    def initialize(
        self,
        system_spec: PolynomialSystemSpec,
        bezier_degree: int,
        tracker_config: TrackerConfig,
    ) -> None:
        """Initialize and cache the Julia Pham homotopy state."""

        if self._closed:
            raise BackendClosedError("cannot initialize a closed JuliaTrackerBackend.")
        if not isinstance(tracker_config, TrackerConfig):
            raise TypeError("tracker_config must be a TrackerConfig.")
        degree = _require_positive_int("bezier_degree", bezier_degree)
        arrays = system_spec_to_julia_arrays(system_spec)
        runtime = _load_julia_runtime(
            julia_executable=self._julia_executable,
            julia_project=self._julia_project,
        )
        options = _make_tracker_options(runtime, tracker_config)

        if self._initialized:
            try:
                runtime.clear_state()
            except Exception as exc:
                raise JuliaAPIError("failed to clear previous Julia state before reinitializing.") from exc

        try:
            runtime.init_bezier_pham(
                arrays.degrees,
                arrays.exponents,
                arrays.offsets,
                arrays.leading_indices,
                arrays.constant_indices,
                degree,
                tracker_options=options,
            )
        except Exception as exc:
            self._initialized = False
            raise JuliaAPIError("failed to initialize Julia Bezier Pham backend state.") from exc

        self._runtime = runtime
        self._system_spec = system_spec
        self._bezier_degree = degree
        self._initialized = True

    def warmup(self) -> TrackingResult:
        """Run Julia ``warmup!`` after initialization."""

        runtime = self._require_active_runtime()
        try:
            julia_result = runtime.warmup()
            payload = runtime.tracking_result_payload(julia_result)
        except Exception as exc:
            raise JuliaTrackingError("Julia warmup failed.") from exc
        return tracking_result_from_julia(payload, n_vars=self._require_system_spec().n_vars)

    def track(self, control_points: np.ndarray) -> TrackingResult:
        """Track all paths for Python-owned Bezier control points."""

        runtime = self._require_active_runtime()
        spec = self._require_system_spec()
        degree = self._require_bezier_degree()
        points = validate_control_points_for_backend(
            control_points,
            system_spec=spec,
            bezier_degree=degree,
        )
        try:
            julia_result = runtime.track_bezier_paths(points)
            payload = runtime.tracking_result_payload(julia_result)
        except Exception as exc:
            raise JuliaTrackingError("Julia path tracking failed.") from exc
        return tracking_result_from_julia(payload, n_vars=spec.n_vars)

    def close(self) -> None:
        """Clear Julia state. Multiple calls are safe."""

        if self._closed:
            return
        runtime = self._runtime
        try:
            if runtime is not None:
                runtime.clear_state()
        except Exception as exc:
            raise JuliaAPIError("failed to clear Julia backend state.") from exc
        finally:
            self._initialized = False
            self._system_spec = None
            self._bezier_degree = None
            self._closed = True

    def julia_state_initialized(self) -> bool:
        """Return whether the process-global Julia state is currently initialized."""

        runtime = self._runtime
        if runtime is None:
            return False
        try:
            return bool(runtime.state_initialized())
        except Exception as exc:
            raise JuliaAPIError("failed to inspect Julia backend state.") from exc

    def state_snapshot(self) -> dict[str, Any]:
        """Return a read-only Python copy of the current Julia state snapshot."""

        runtime = self._require_active_runtime()
        try:
            snapshot = runtime.state_snapshot()
        except Exception as exc:
            raise JuliaAPIError("failed to inspect Julia backend state.") from exc
        return _snapshot_to_python(snapshot)

    def _require_active_runtime(self) -> _JuliaRuntime:
        if self._closed:
            raise BackendClosedError("JuliaTrackerBackend has been closed.")
        if not self._initialized or self._runtime is None:
            raise BackendStateError("JuliaTrackerBackend is not initialized.")
        return self._runtime

    def _require_system_spec(self) -> PolynomialSystemSpec:
        if self._system_spec is None:
            raise BackendStateError("JuliaTrackerBackend has no system spec.")
        return self._system_spec

    def _require_bezier_degree(self) -> int:
        if self._bezier_degree is None:
            raise BackendStateError("JuliaTrackerBackend has no Bezier degree.")
        return self._bezier_degree


__all__ = [
    "BackendClosedError",
    "BackendStateError",
    "JuliaAPIError",
    "JuliaBackendError",
    "JuliaBackendValidationError",
    "JuliaModuleLoadError",
    "JuliaResultConversionError",
    "JuliaRuntimeInitializationError",
    "JuliaSystemArrays",
    "JuliaTrackerBackend",
    "JuliaTrackingError",
    "system_spec_to_julia_arrays",
    "tracking_result_from_julia",
    "validate_control_points_for_backend",
    "validate_system_arrays",
]
