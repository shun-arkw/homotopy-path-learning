"""One-step Gymnasium environment for Bezier Pham homotopy paths."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import numpy.typing as npt

from homotopy_path_learning.backends import JuliaTrackerBackend, TrackerBackend
from homotopy_path_learning.backends.types import TrackerConfig, TrackingResult
from homotopy_path_learning.envs.costs import TrackingCost, reward_from_costs, tracking_cost
from homotopy_path_learning.paths.parameterization import (
    BezierParameterization,
    complex_coefficients_to_real,
    linear_interpolation_control_points,
)
from homotopy_path_learning.systems.sampling import (
    CoefficientSampler,
    ComplexUniformSampler,
    start_system_coefficients,
)
from homotopy_path_learning.systems.spec import PolynomialSystemSpec


def _is_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _require_int(name: str, value: object) -> int:
    if not _is_int(value):
        raise TypeError(f"{name} must be an integer, got {value!r}.")
    return int(value)


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a finite float, got {value!r}.")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite float, got {value!r}.") from exc
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return normalized


class BezierPhamEnv(gym.Env[npt.NDArray[np.float32], npt.NDArray[np.float32]]):
    """One-step Bezier control-point optimization environment.

    The observation is the target system's free complex coefficients in real
    form with shape ``(2 * n_free_coeffs,)`` and dtype ``float32``:
    ``[Re(c_free), Im(c_free)]``. The action has shape
    ``((bezier_degree - 1) * latent_dim,)`` and dtype ``float32`` in the
    Gymnasium space; it is converted to ``float64`` before expanding through
    :class:`BezierParameterization`.

    If ``backend`` is ``None``, the environment constructs and owns a
    :class:`JuliaTrackerBackend`; ``close()`` closes that backend. If a backend
    is injected, it is initialized by the environment but remains owned by the
    caller and is not closed by ``BezierPhamEnv.close()``.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        system_spec: PolynomialSystemSpec,
        bezier_degree: int = 3,
        latent_dim: int = 4,
        coefficient_sampler: CoefficientSampler | None = None,
        tracker_config: TrackerConfig | None = None,
        reject_weight: float = 1.0,
        failure_penalty: float = 100_000.0,
        reward_scale: float = 1.0,
        action_limit: float = 1.0,
        basis_seed: int = 0,
        backend: TrackerBackend | None = None,
        warmup_backend: bool = False,
        render_mode: None = None,
    ) -> None:
        if render_mode is not None:
            raise ValueError("BezierPhamEnv does not implement render modes.")
        if not isinstance(system_spec, PolynomialSystemSpec):
            raise TypeError("system_spec must be a PolynomialSystemSpec.")
        degree = _require_int("bezier_degree", bezier_degree)
        if degree < 2:
            raise ValueError(f"bezier_degree must be at least 2, got {degree}.")
        latent = _require_int("latent_dim", latent_dim)
        if latent <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent}.")
        seed_value = _require_int("basis_seed", basis_seed)
        if not isinstance(warmup_backend, bool):
            raise TypeError("warmup_backend must be a bool.")

        rho = _finite_float("reject_weight", reject_weight)
        penalty = _finite_float("failure_penalty", failure_penalty)
        scale = _finite_float("reward_scale", reward_scale)
        limit = _finite_float("action_limit", action_limit)
        if rho < 0.0:
            raise ValueError(f"reject_weight must be nonnegative, got {rho}.")
        if penalty <= 0.0:
            raise ValueError(f"failure_penalty must be positive, got {penalty}.")
        if scale <= 0.0:
            raise ValueError(f"reward_scale must be positive, got {scale}.")
        if limit <= 0.0:
            raise ValueError(f"action_limit must be positive, got {limit}.")

        config = tracker_config if tracker_config is not None else TrackerConfig()
        if not isinstance(config, TrackerConfig):
            raise TypeError("tracker_config must be a TrackerConfig.")
        sampler = coefficient_sampler if coefficient_sampler is not None else ComplexUniformSampler()
        if not hasattr(sampler, "sample"):
            raise TypeError("coefficient_sampler must provide a sample(rng, spec) method.")

        self.system_spec = system_spec
        self.bezier_degree = degree
        self.latent_dim = latent
        self.coefficient_sampler = sampler
        self.tracker_config = config
        self.reject_weight = rho
        self.failure_penalty = penalty
        self.reward_scale = scale
        self.action_limit = limit
        self.basis_seed = seed_value
        self.render_mode = render_mode

        self.start_coefficients = start_system_coefficients(system_spec)
        self.free_indices = np.flatnonzero(system_spec.free_coefficient_mask).astype(np.int64)
        self.parameterization = BezierParameterization.from_spec(
            system_spec,
            bezier_degree=degree,
            latent_dim=latent,
            rng=np.random.default_rng(seed_value),
        )

        self.action_space = spaces.Box(
            low=-limit,
            high=limit,
            shape=(self.parameterization.action_dim,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=self._observation_low(),
            high=self._observation_high(),
            shape=(2 * self.free_indices.size,),
            dtype=np.float32,
        )

        if backend is None:
            self._backend: TrackerBackend = JuliaTrackerBackend()
            self._owns_backend = True
        else:
            self._backend = backend
            self._owns_backend = False
        self._backend.initialize(system_spec, degree, config)
        if warmup_backend:
            self._backend.warmup()

        self._closed = False
        self._has_reset = False
        self._terminated = False
        self._target_coefficients: npt.NDArray[np.complex128] | None = None
        self._observation: npt.NDArray[np.float32] | None = None
        self._linear_result: TrackingResult | None = None
        self._linear_cost: TrackingCost | None = None
        self._linear_control_points: npt.NDArray[np.complex128] | None = None

    @property
    def backend(self) -> TrackerBackend:
        """Backend used by the environment."""

        return self._backend

    @property
    def owns_backend(self) -> bool:
        """Whether ``close()`` will close the backend."""

        return self._owns_backend

    @property
    def closed(self) -> bool:
        """Whether the environment has been closed."""

        return self._closed

    @property
    def target_coefficients(self) -> npt.NDArray[np.complex128] | None:
        """Current target coefficients as a copy, or ``None`` before reset."""

        if self._target_coefficients is None:
            return None
        return np.array(self._target_coefficients, dtype=np.complex128, copy=True)

    @property
    def linear_control_points(self) -> npt.NDArray[np.complex128] | None:
        """Current episode's cached linear control points as a copy."""

        if self._linear_control_points is None:
            return None
        return np.array(self._linear_control_points, dtype=np.complex128, copy=True)

    @property
    def linear_cost(self) -> float | None:
        """Current episode's cached mean linear cost."""

        if self._linear_cost is None:
            return None
        return float(self._linear_cost.mean)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[npt.NDArray[np.float32], dict[str, Any]]:
        """Start a new one-step episode and cache the linear baseline cost."""

        if self._closed:
            raise RuntimeError("cannot reset a closed BezierPhamEnv.")
        super().reset(seed=seed)
        del options

        target = self._sample_target_coefficients()
        observation = self._observation_from_target(target)
        zero_action = np.zeros(self.parameterization.action_dim, dtype=np.float64)
        linear_points = self.parameterization.build_control_points(
            self.start_coefficients,
            target,
            zero_action,
        )
        expected_linear = linear_interpolation_control_points(
            self.start_coefficients,
            target,
            self.bezier_degree,
        )
        if not np.array_equal(linear_points, expected_linear):
            raise RuntimeError("zero action did not produce linear interpolation control points.")

        linear_result = self._backend.track(linear_points)
        linear_cost_value = tracking_cost(
            linear_result,
            reject_weight=self.reject_weight,
            failure_penalty=self.failure_penalty,
        )

        self._target_coefficients = target
        self._observation = observation
        self._linear_control_points = linear_points
        self._linear_result = linear_result
        self._linear_cost = linear_cost_value
        self._has_reset = True
        self._terminated = False

        return observation.copy(), self._reset_info()

    def step(
        self,
        action: npt.ArrayLike,
    ) -> tuple[npt.NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Apply one latent action and terminate the episode."""

        self._require_step_ready()
        assert self._target_coefficients is not None
        assert self._observation is not None
        assert self._linear_result is not None
        assert self._linear_cost is not None

        action_vector = self._validate_action(action)
        control_points = self.parameterization.build_control_points(
            self.start_coefficients,
            self._target_coefficients,
            action_vector,
        )
        bezier_result = self._backend.track(control_points)
        bezier_cost_value = tracking_cost(
            bezier_result,
            reject_weight=self.reject_weight,
            failure_penalty=self.failure_penalty,
        )
        reward = reward_from_costs(
            linear_cost=self._linear_cost.mean,
            bezier_cost=bezier_cost_value.mean,
            reward_scale=self.reward_scale,
        )

        self._terminated = True
        info = self._step_info(
            control_points=control_points,
            bezier_result=bezier_result,
            bezier_cost=bezier_cost_value,
            reward=reward,
        )
        return self._observation.copy(), reward, True, False, info

    def render(self) -> None:
        """Rendering is intentionally not implemented for the initial environment."""

        return None

    def close(self) -> None:
        """Close the environment and its owned backend, if any."""

        if self._closed:
            return
        try:
            if self._owns_backend:
                self._backend.close()
        finally:
            self._closed = True

    def _observation_low(self) -> float:
        if isinstance(self.coefficient_sampler, ComplexUniformSampler):
            return float(-self.coefficient_sampler.bound)
        return float("-inf")

    def _observation_high(self) -> float:
        if isinstance(self.coefficient_sampler, ComplexUniformSampler):
            return float(self.coefficient_sampler.bound)
        return float("inf")

    def _sample_target_coefficients(self) -> npt.NDArray[np.complex128]:
        target = np.array(
            self.coefficient_sampler.sample(self.np_random, self.system_spec),
            dtype=np.complex128,
            copy=True,
        )
        expected_shape = (self.system_spec.n_coeffs,)
        if target.shape != expected_shape:
            raise ValueError(
                f"target coefficients must have shape {expected_shape}, got {target.shape}."
            )
        if not np.all(np.isfinite(target.real)) or not np.all(np.isfinite(target.imag)):
            raise ValueError("target coefficients must not contain NaN or Inf.")
        expected_leading = np.ones(self.system_spec.leading_indices.size, dtype=np.complex128)
        if not np.array_equal(target[self.system_spec.leading_indices], expected_leading):
            raise ValueError("target leading coefficients must be exactly 1 + 0j.")
        return target

    def _observation_from_target(
        self,
        target: npt.NDArray[np.complex128],
    ) -> npt.NDArray[np.float32]:
        free_coefficients = target[self.free_indices]
        observation = complex_coefficients_to_real(free_coefficients).astype(np.float32)
        expected_shape = self.observation_space.shape
        if observation.shape != expected_shape:
            raise RuntimeError(
                f"observation must have shape {expected_shape}, got {observation.shape}."
            )
        if not np.all(np.isfinite(observation)):
            raise ValueError("observation must not contain NaN or Inf.")
        return observation

    def _validate_action(self, action: npt.ArrayLike) -> npt.NDArray[np.float64]:
        action_vector = np.array(action, dtype=np.float64, copy=True)
        expected_shape = (self.parameterization.action_dim,)
        if action_vector.shape != expected_shape:
            raise ValueError(
                f"action must have shape {expected_shape}, got {action_vector.shape}."
            )
        if not np.all(np.isfinite(action_vector)):
            raise ValueError("action must not contain NaN or Inf.")
        if np.any(action_vector < -self.action_limit) or np.any(action_vector > self.action_limit):
            raise ValueError("action must lie within the action_space bounds.")
        return action_vector

    def _require_step_ready(self) -> None:
        if self._closed:
            raise RuntimeError("cannot step a closed BezierPhamEnv.")
        if not self._has_reset:
            raise RuntimeError("reset() must be called before step().")
        if self._terminated:
            raise RuntimeError("BezierPhamEnv is a one-step environment; call reset() first.")

    def _reset_info(self) -> dict[str, Any]:
        assert self._target_coefficients is not None
        assert self._linear_result is not None
        assert self._linear_cost is not None
        assert self._linear_control_points is not None
        return {
            "target_coefficients": self._target_coefficients.copy(),
            "linear_control_points": self._linear_control_points.copy(),
            "linear_cost": float(self._linear_cost.mean),
            "linear_success": bool(self._linear_result.success),
            "linear_n_paths": int(self._linear_result.n_paths),
            "linear_n_success": int(self._linear_result.n_success),
            "linear_n_failed": int(self._linear_result.n_failed),
            "linear_accepted_steps": int(self._linear_result.accepted_steps),
            "linear_rejected_steps": int(self._linear_result.rejected_steps),
            "linear_per_path_costs": self._linear_cost.per_path.copy(),
            "linear_residual_norms": self._linear_result.residual_norms.copy(),
            "linear_failure_codes": tuple(self._linear_result.failure_codes),
        }

    def _step_info(
        self,
        *,
        control_points: npt.NDArray[np.complex128],
        bezier_result: TrackingResult,
        bezier_cost: TrackingCost,
        reward: float,
    ) -> dict[str, Any]:
        assert self._target_coefficients is not None
        assert self._linear_result is not None
        assert self._linear_cost is not None
        improvement = float(self._linear_cost.mean - bezier_cost.mean)
        return {
            "target_coefficients": self._target_coefficients.copy(),
            "control_points": control_points.copy(),
            "linear_cost": float(self._linear_cost.mean),
            "bezier_cost": float(bezier_cost.mean),
            "cost_improvement": improvement,
            "reward": float(reward),
            "linear_success": bool(self._linear_result.success),
            "bezier_success": bool(bezier_result.success),
            "linear_n_success": int(self._linear_result.n_success),
            "linear_n_failed": int(self._linear_result.n_failed),
            "bezier_n_success": int(bezier_result.n_success),
            "bezier_n_failed": int(bezier_result.n_failed),
            "linear_accepted_steps": int(self._linear_result.accepted_steps),
            "linear_rejected_steps": int(self._linear_result.rejected_steps),
            "bezier_accepted_steps": int(bezier_result.accepted_steps),
            "bezier_rejected_steps": int(bezier_result.rejected_steps),
            "linear_per_path_costs": self._linear_cost.per_path.copy(),
            "bezier_per_path_costs": bezier_cost.per_path.copy(),
            "bezier_residual_norms": bezier_result.residual_norms.copy(),
            "bezier_failure_codes": tuple(bezier_result.failure_codes),
        }


__all__ = ["BezierPhamEnv"]
