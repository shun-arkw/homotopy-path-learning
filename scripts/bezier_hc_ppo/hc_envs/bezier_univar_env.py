# hc_envs/bezier_univar_env.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Sequence
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from hc_envs.julia_backend import get_backend, BezierUnivarConfig, LinearUnivarConfig


def complex_coeffs_to_real(x: np.ndarray) -> np.ndarray:
    """
    x: complex[num_coeffs] -> real[2*num_coeffs] (Re concat Im)
    """
    x = np.asarray(x, dtype=np.complex128)
    return np.concatenate([x.real, x.imag], axis=0)


def real_to_complex_coeffs(v: np.ndarray) -> np.ndarray:
    """
    v: real[2*num_coeffs] -> complex[num_coeffs]
    """
    v = np.asarray(v, dtype=np.float64)
    D2 = v.shape[0]
    assert D2 % 2 == 0
    D = D2 // 2
    return (v[:D] + 1j * v[D:]).astype(np.complex128)


def make_orthonormal_U(rng: np.random.Generator, dim: int, m: int) -> np.ndarray:
    """
    U: real[dim, latent_dim] with orthonormal columns via QR (m = latent_dim).
    """
    A = rng.standard_normal(size=(dim, m))
    Q, _ = np.linalg.qr(A)
    return Q[:, :m].astype(np.float64)


def clip_rows_l2(z: np.ndarray, alpha: float, eps: float = 1e-12) -> np.ndarray:
    """
    Clip each row z[k] so that ||z[k]||_2 <= alpha.
    z: real[(bezier_degree-1), latent_dim]
    """
    if alpha <= 0:
        return z
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    scale = np.minimum(1.0, alpha / (norms + eps))
    return z * scale


@dataclass
class TargetCoeffConfig:
    """
    Config for sampling target polynomial coefficients.
    Real and imaginary parts are sampled independently with either:
    - Gaussian: N(mean_*, std_*)
    - Uniform: [low_*, high_*]
    """
    dist_real: Literal["gaussian", "uniform"] = "gaussian"
    dist_imag: Literal["gaussian", "uniform"] = "gaussian"
    # Gaussian params
    mean_real: float = 0.0
    mean_imag: float = 0.0
    std_real: float = 0.5
    std_imag: float = 0.5
    # Uniform params
    low_real: float = -0.5
    high_real: float = 0.5
    low_imag: float = -0.5
    high_imag: float = 0.5


@dataclass
class ProblemInstance:
    """
    Per-episode fixed data: start_coeffs, target_coeffs (complex[num_coeffs]), gamma (complex).
    """
    start_coeffs: np.ndarray
    target_coeffs: np.ndarray
    gamma: complex


class BezierHomotopyUnivarEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        degree: int,
        bezier_degree: int,
        latent_dim_m: int,
        episode_len_T: int,
        alpha_z: float,
        failure_penalty: float,
        rho_reject: float = 1.0,
        terminal_linear_bonus: bool = True,
        terminal_linear_bonus_coef: float = 1.0,
        terminal_z0_bonus: bool = False,
        terminal_z0_bonus_coef: float = 1.0,
        step_reward_scale: float = 1.0,
        reward_mode: Literal["delta", "baseline", "terminal", "best"] = "delta",
        tracking_cost_mode: Literal["steps", "eaj"] = "steps",
        require_z0_success: bool = False,
        z0_max_tries: int = 10,
        gamma_trick: bool = True,
        seed: int = 0,
        extended_precision: bool = False,
        compute_newton_iters: bool = False,  # Prefer False during training
        include_progress_features: bool = True,
        f_coeff_config: Optional[TargetCoeffConfig] = None,
        fixed_instances: Optional[Sequence[ProblemInstance]] = None,
        # Advanced: you can override the Julia include expression if needed
        julia_include_seval: str = 'include("scripts/bezier_hc_ppo/bezier_univar.jl")',
        # HC TrackerParameters (shared linear/bezier): a, β_a, β_ω_p, β_τ, strict_β_τ, min_newton_iters
        hc_a: float = 0.125,
        hc_beta_a: float = 1.0,
        hc_beta_omega_p: float = 0.8,
        hc_beta_tau: float = 0.85,
        hc_strict_beta_tau: float = 0.8,
        hc_min_newton_iters: int = 1,
        # HC TrackerOptions (shared linear/bezier)
        hc_max_steps: int = 50_000,
        hc_max_step_size: float = float("inf"),
        hc_max_initial_step_size: float = float("inf"),
        hc_min_step_size: float = 1e-12,
        hc_extended_precision: bool = False,
    ):
        super().__init__()
        assert bezier_degree >= 2, "bezier_degree must be >= 2 (2, 3, 4, ... supported)."
        assert latent_dim_m > 0
        assert episode_len_T > 0

        self.f_coeff_config = f_coeff_config if f_coeff_config is not None else TargetCoeffConfig()
        self.degree = int(degree)
        self.bezier_degree = int(bezier_degree)
        self.num_coeffs = self.degree + 1
        self.latent_dim = int(latent_dim_m)
        self.episode_length = int(episode_len_T)

        self.alpha_z = float(alpha_z)
        self.failure_penalty = float(failure_penalty)
        self.rho = float(rho_reject)
        self.terminal_linear_bonus = bool(terminal_linear_bonus)
        self.terminal_linear_bonus_coef = float(terminal_linear_bonus_coef)
        self.terminal_z0_bonus = bool(terminal_z0_bonus)
        self.terminal_z0_bonus_coef = float(terminal_z0_bonus_coef)
        self.step_reward_scale = float(step_reward_scale)
        reward_mode = str(reward_mode).lower()
        if reward_mode not in ("delta", "baseline", "terminal", "best"):
            raise ValueError(
                "reward_mode must be 'delta', 'baseline', 'terminal', or 'best', "
                f"got: {reward_mode!r}"
            )
        self.reward_mode = reward_mode
        mode = str(tracking_cost_mode).lower()
        if mode not in ("steps", "eaj"):
            raise ValueError(
                f"tracking_cost_mode must be 'steps' or 'eaj', got: {tracking_cost_mode!r}"
            )
        self.tracking_cost_mode = mode
        self.require_z0_success = bool(require_z0_success)
        self.z0_max_tries = int(z0_max_tries)
        self.gamma_trick = bool(gamma_trick)

        self.seed0 = int(seed)
        self.rng = np.random.default_rng(self.seed0)
        self.fixed_instances = list(fixed_instances) if fixed_instances else None
        self.fixed_instance_idx = 0

        self.extended_precision = bool(extended_precision)
        self.compute_newton_iters = bool(compute_newton_iters)
        self.hc_a = float(hc_a)
        self.hc_beta_a = float(hc_beta_a)
        self.hc_beta_omega_p = float(hc_beta_omega_p)
        self.hc_beta_tau = float(hc_beta_tau)
        self.hc_strict_beta_tau = float(hc_strict_beta_tau)
        self.hc_min_newton_iters = int(hc_min_newton_iters)
        self.hc_max_steps = int(hc_max_steps)
        self.hc_max_step_size = float(hc_max_step_size)
        self.hc_max_initial_step_size = float(hc_max_initial_step_size)
        self.hc_min_step_size = float(hc_min_step_size)
        self.hc_extended_precision = bool(hc_extended_precision)

        # Get a process-wide backend and include Julia definitions exactly once.
        self.backend = get_backend(include_path=julia_include_seval)

        backend_cfg = BezierUnivarConfig(
            degree=self.degree,
            bezier_degree=self.bezier_degree,
            seed=self.seed0,
            compute_newton_iters=self.compute_newton_iters,
            extended_precision=self.hc_extended_precision,
            max_steps=self.hc_max_steps,
            max_step_size=self.hc_max_step_size,
            max_initial_step_size=self.hc_max_initial_step_size,
            min_step_size=self.hc_min_step_size,
            hc_a=self.hc_a,
            hc_beta_a=self.hc_beta_a,
            hc_beta_omega_p=self.hc_beta_omega_p,
            hc_beta_tau=self.hc_beta_tau,
            hc_strict_beta_tau=self.hc_strict_beta_tau,
            hc_min_newton_iters=self.hc_min_newton_iters,
        )
        self.backend.ensure_ready(backend_cfg)
        self.needs_linear_baseline = self.terminal_linear_bonus or self.reward_mode == "best"
        if self.needs_linear_baseline:
            linear_cfg = LinearUnivarConfig(
                degree=self.degree,
                seed=self.seed0,
                compute_newton_iters=self.compute_newton_iters,
                extended_precision=self.hc_extended_precision,
                max_steps=self.hc_max_steps,
                max_step_size=self.hc_max_step_size,
                max_initial_step_size=self.hc_max_initial_step_size,
                min_step_size=self.hc_min_step_size,
                hc_a=self.hc_a,
                hc_beta_a=self.hc_beta_a,
                hc_beta_omega_p=self.hc_beta_omega_p,
                hc_beta_tau=self.hc_beta_tau,
                hc_strict_beta_tau=self.hc_strict_beta_tau,
                hc_min_newton_iters=self.hc_min_newton_iters,
            )
            self.backend.ensure_ready_linear(linear_cfg)

        # Fixed basis U: (2*num_coeffs x latent_dim)
        self.U = make_orthonormal_U(self.rng, dim=2 * self.num_coeffs, m=self.latent_dim)

        # Multi-step internal state
        self.z = np.zeros((self.bezier_degree - 1, self.latent_dim), dtype=np.float64)
        self.prev_tracking_cost: Optional[float] = None
        self.current_tracking_cost: Optional[float] = None
        self.best_tracking_cost: Optional[float] = None
        self.best_z: Optional[np.ndarray] = None
        self.best_tracking_info: Optional[dict] = None
        self.linear_tracking_cost: Optional[float] = None
        self.z0_tracking_cost: Optional[float] = None
        self.t = 0
        self.inst: Optional[ProblemInstance] = None

        # Action: flat Δz
        self.action_dim = (self.bezier_degree - 1) * self.latent_dim
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )

        # Obs: problem features + progress features
        self.include_progress = bool(include_progress_features)
        base_obs_dim = (2 * self.num_coeffs) + (2 * self.num_coeffs) + 2
        if self.include_progress and self.reward_mode == "best":
            prog_obs_dim = self.action_dim + 5
        else:
            prog_obs_dim = (self.action_dim + 2) if self.include_progress else 0
        obs_dim = base_obs_dim + prog_obs_dim

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    # ---------------------------
    # Problem sampling
    # ---------------------------
    def _sample_instance(self) -> ProblemInstance:
        """
        Sample a problem instance:
        - start_coeffs = x^degree - 1
        - target_coeffs sampled from TargetCoeffConfig
        - gamma = exp(i theta)
        """
        num_coeffs = self.num_coeffs
        deg = self.degree
        cfg = self.f_coeff_config

        # Sample target real part
        if cfg.dist_real == "gaussian":
            re = cfg.mean_real + cfg.std_real * self.rng.standard_normal(num_coeffs)
        else:
            re = self.rng.uniform(cfg.low_real, cfg.high_real, size=num_coeffs)

        # Sample target imaginary part
        if cfg.dist_imag == "gaussian":
            im = cfg.mean_imag + cfg.std_imag * self.rng.standard_normal(num_coeffs)
        else:
            im = self.rng.uniform(cfg.low_imag, cfg.high_imag, size=num_coeffs)

        target_coeffs = (re + 1j * im).astype(np.complex128)

        # Start system: x^deg - 1 (descending coeff order [a_deg, ..., a_0])
        start_coeffs = np.zeros(num_coeffs, dtype=np.complex128)
        start_coeffs[0] = 1.0 + 0.0j # coefficient of x^deg
        start_coeffs[deg] = -1.0 + 0.0j # coefficient of x^0

        # Gamma: exp(i theta) when gamma_trick else 1
        if not self.gamma_trick:
            gamma = 1.0 + 0.0j
            return ProblemInstance(start_coeffs=start_coeffs, target_coeffs=target_coeffs, gamma=gamma)

        def _sample_gamma() -> complex:
            theta = float(self.rng.uniform(0.0, 2.0 * np.pi))
            return np.cos(theta) + 1j * np.sin(theta)

        if not self.require_z0_success:
            gamma = _sample_gamma()
            return ProblemInstance(start_coeffs=start_coeffs, target_coeffs=target_coeffs, gamma=gamma)

        max_tries = max(1, self.z0_max_tries)
        last_gamma = None
        for _ in range(max_tries):
            gamma = _sample_gamma()
            inst = ProblemInstance(start_coeffs=start_coeffs, target_coeffs=target_coeffs, gamma=gamma)
            if self._check_z0_success(inst):
                return inst
            last_gamma = gamma

        # Fallback: return the last sampled gamma even if it failed.
        if last_gamma is None:
            last_gamma = _sample_gamma()
        return ProblemInstance(start_coeffs=start_coeffs, target_coeffs=target_coeffs, gamma=last_gamma)

    # ---------------------------
    # Control points construction
    # ---------------------------
    def _build_control_points(self) -> np.ndarray:
        """
        ctrl shape: (bezier_degree+1, num_coeffs) complex128
        P0 = gamma * start_coeffs
        Pd = target_coeffs
        Pk = barPk + perturb(U z_k) for k=1..bezier_degree-1
        """
        assert self.inst is not None
        start_coeffs = self.inst.start_coeffs
        target_coeffs = self.inst.target_coeffs
        gamma = self.inst.gamma

        num_coeffs = self.num_coeffs
        d = self.bezier_degree
        ctrl = np.zeros((d + 1, num_coeffs), dtype=np.complex128)

        P0 = gamma * start_coeffs
        Pd = target_coeffs
        ctrl[0, :] = P0
        ctrl[d, :] = Pd

        tilde_P0 = complex_coeffs_to_real(P0)
        tilde_Pd = complex_coeffs_to_real(Pd)

        for k in range(1, d):
            t = k / d
            tilde_bar = (1.0 - t) * tilde_P0 + t * tilde_Pd
            tilde_pk = tilde_bar + self.U @ self.z[k - 1]
            ctrl[k, :] = real_to_complex_coeffs(tilde_pk)

        return ctrl

    def _build_control_points_for(self, inst: ProblemInstance, z_override: np.ndarray) -> np.ndarray:
        """
        Build control points for a given instance and z (no side effects).
        """
        start_coeffs = inst.start_coeffs
        target_coeffs = inst.target_coeffs
        gamma = inst.gamma

        num_coeffs = self.num_coeffs
        d = self.bezier_degree
        ctrl = np.zeros((d + 1, num_coeffs), dtype=np.complex128)

        P0 = gamma * start_coeffs
        Pd = target_coeffs
        ctrl[0, :] = P0
        ctrl[d, :] = Pd

        tilde_P0 = complex_coeffs_to_real(P0)
        tilde_Pd = complex_coeffs_to_real(Pd)

        for k in range(1, d):
            t = k / d
            tilde_bar = (1.0 - t) * tilde_P0 + t * tilde_Pd
            tilde_pk = tilde_bar + self.U @ z_override[k - 1]
            ctrl[k, :] = real_to_complex_coeffs(tilde_pk)

        return ctrl

    def _check_z0_success(self, inst: ProblemInstance) -> bool:
        z0 = np.zeros((self.bezier_degree - 1, self.latent_dim), dtype=np.float64)
        ctrl = self._build_control_points_for(inst, z0)
        out = self.backend.jl.track_bezier_paths_univar(
            int(self.degree),
            int(self.bezier_degree),
            ctrl,
            compute_newton_iters=bool(self.compute_newton_iters),
        )
        return bool(out.success_flag)

    def _make_obs(self) -> np.ndarray:
        assert self.inst is not None
        target_coeffs = self.inst.target_coeffs
        start_coeffs = self.inst.start_coeffs
        gamma = self.inst.gamma

        obs_parts = [
            complex_coeffs_to_real(target_coeffs),
            complex_coeffs_to_real(start_coeffs),
            np.array([gamma.real, gamma.imag], dtype=np.float64),
        ]

        if self.include_progress:
            z_flat = self.z.reshape(-1)
            t_frac = 0.0 if self.episode_length <= 1 else (self.t / self.episode_length)
            obs_parts.append(z_flat.astype(np.float64))
            if self.reward_mode == "best":
                denom = self._baseline_denom()
                current_cost = (
                    self.current_tracking_cost
                    if self.current_tracking_cost is not None
                    else self.linear_tracking_cost
                )
                best_cost = (
                    self.best_tracking_cost
                    if self.best_tracking_cost is not None
                    else self.linear_tracking_cost
                )
                current_ratio = 1.0 if current_cost is None else float(current_cost) / denom
                best_ratio = 1.0 if best_cost is None else float(best_cost) / denom
                gap_ratio = float(current_ratio - best_ratio)
                norm_z_ratio = float(np.linalg.norm(self.z) / max(self.alpha_z, 1e-12))
                obs_parts.append(
                    np.array(
                        [current_ratio, best_ratio, gap_ratio, t_frac, norm_z_ratio],
                        dtype=np.float64,
                    )
                )
            else:
                prev_cost = 0.0 if self.prev_tracking_cost is None else float(self.prev_tracking_cost)
                obs_parts.append(np.array([t_frac, prev_cost], dtype=np.float64))

        obs = np.concatenate(obs_parts, axis=0).astype(np.float32)
        return obs

    def _tracking_metrics_from_out(self, out):
        success = bool(out.success_flag)
        acc = int(out.total_accepted_steps)
        rej = int(out.total_rejected_steps)
        attempts = int(out.total_step_attempts)
        newton_iters = int(getattr(out, "total_newton_iterations", 0))
        eaj_calls = int(getattr(out, "total_evaluate_and_jacobian_calls", 0))
        return success, acc, rej, attempts, newton_iters, eaj_calls

    def _compute_tracking_cost(
        self,
        success: bool,
        accepted_steps: int,
        rejected_steps: int,
        evaluate_and_jacobian_calls: int,
    ) -> float:
        if not success:
            return float(self.failure_penalty)
        if self.tracking_cost_mode == "steps":
            return float(accepted_steps + self.rho * rejected_steps)
        return float(evaluate_and_jacobian_calls)

    def _tracking_info_from_out(self, out) -> dict:
        success, acc, rej, attempts, newton_iters, eaj_calls = self._tracking_metrics_from_out(
            out
        )
        tracking_cost = self._compute_tracking_cost(success, acc, rej, eaj_calls)
        return {
            "success": success,
            "tracking_cost": tracking_cost,
            "accepted_steps": acc,
            "rejected_steps": rej,
            "total_step_attempts": attempts,
            "total_newton_iterations": newton_iters,
            "total_evaluate_and_jacobian_calls": eaj_calls,
            "runtime_sec": float(getattr(out, "runtime_sec", 0.0)),
            "tracking_time_sec": float(getattr(out, "tracking_time_sec", 0.0)),
        }

    def _baseline_denom(self) -> float:
        baseline = self.linear_tracking_cost
        if baseline is None or not np.isfinite(float(baseline)) or float(baseline) <= 0.0:
            return max(float(self.failure_penalty), 1.0)
        return max(float(baseline), 1e-12)

    def _compute_linear_tracking_info(self) -> dict:
        assert self.inst is not None
        start_coeffs = self.inst.start_coeffs
        target_coeffs = self.inst.target_coeffs
        gamma = self.inst.gamma
        start_path = gamma * start_coeffs
        out = self.backend.jl.track_linear_paths_univar(
            int(self.degree),
            start_path,
            target_coeffs,
            compute_newton_iters=bool(self.compute_newton_iters),
        )
        return self._tracking_info_from_out(out)

    def _compute_linear_tracking_cost(self) -> float:
        return float(self._compute_linear_tracking_info()["tracking_cost"])

    def _compute_z0_tracking_cost(self) -> float:
        assert self.inst is not None
        z0 = np.zeros((self.bezier_degree - 1, self.latent_dim), dtype=np.float64)
        ctrl = self._build_control_points_for(self.inst, z0)
        out = self.backend.jl.track_bezier_paths_univar(
            int(self.degree),
            int(self.bezier_degree),
            ctrl,
            compute_newton_iters=bool(self.compute_newton_iters),
        )
        success, acc, rej, _, _, eaj_calls = self._tracking_metrics_from_out(out)
        return self._compute_tracking_cost(success, acc, rej, eaj_calls)

    def set_fixed_instances(self, instances: Sequence[ProblemInstance], reset_idx: bool = True) -> None:
        """
        Provide a fixed list of ProblemInstance objects for evaluation.
        When set, reset() will cycle through these instances in order.
        """
        if not instances:
            self.fixed_instances = None
            self.fixed_instance_idx = 0
            return
        self.fixed_instances = list(instances)
        if reset_idx:
            self.fixed_instance_idx = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            # Keep deterministic behavior when Gym provides a seed.
            self.rng = np.random.default_rng(int(seed))

        if self.fixed_instances:
            idx = self.fixed_instance_idx % len(self.fixed_instances)
            self.inst = self.fixed_instances[idx]
            self.fixed_instance_idx += 1
        else:
            self.inst = self._sample_instance()
        self.z[...] = 0.0
        self.prev_tracking_cost = None
        self.current_tracking_cost = None
        self.best_tracking_cost = None
        self.best_z = self.z.copy()
        self.best_tracking_info = None
        self.linear_tracking_cost = None
        self.z0_tracking_cost = None
        self.t = 0

        if self.needs_linear_baseline:
            linear_info = self._compute_linear_tracking_info()
            self.linear_tracking_cost = float(linear_info["tracking_cost"])
            if self.reward_mode == "best":
                self.current_tracking_cost = self.linear_tracking_cost
                self.best_tracking_cost = self.linear_tracking_cost
                self.best_tracking_info = linear_info
        if self.terminal_z0_bonus:
            self.z0_tracking_cost = self._compute_z0_tracking_cost()

        obs = self._make_obs()
        info = {"degree": self.degree, "bezier_degree": self.bezier_degree}
        return obs, info

    def step(self, action: np.ndarray):
        assert self.inst is not None
        action = np.asarray(action, dtype=np.float32)
        assert action.shape == (self.action_dim,)

        # Δz update (assume action in [-1, 1], e.g., from tanh; scale here if needed)
        dz = action.astype(np.float64).reshape(self.bezier_degree - 1, self.latent_dim)

        # Optional: environment-side action scaling
        # dz = dz * 0.1

        self.z = clip_rows_l2(self.z + dz, alpha=self.alpha_z)
        proposed_z = self.z.copy()

        # Build ctrl and call Julia
        ctrl = self._build_control_points()
        out = self.backend.jl.track_bezier_paths_univar(
            int(self.degree),
            int(self.bezier_degree),
            ctrl,
            compute_newton_iters=bool(self.compute_newton_iters),
        )

        proposal_info = self._tracking_info_from_out(out)
        success = bool(proposal_info["success"])
        tracking_cost = float(proposal_info["tracking_cost"])

        if self.reward_mode == "delta":
            # Existing shaping: reward only reductions from the previous tried path.
            if self.prev_tracking_cost is None:
                reward = 0.0
            else:
                reward = float(self.prev_tracking_cost - tracking_cost) * self.step_reward_scale
        elif self.reward_mode == "baseline":
            baseline = (
                self.linear_tracking_cost
                if self.linear_tracking_cost is not None
                else self.z0_tracking_cost
            )
            if baseline is None:
                reward = -tracking_cost * self.step_reward_scale
            else:
                reward = float(baseline - tracking_cost) * self.step_reward_scale
        elif self.reward_mode == "best":
            prev_best = (
                float(self.best_tracking_cost)
                if self.best_tracking_cost is not None
                else self._baseline_denom()
            )
            improved = tracking_cost < prev_best
            if improved:
                self.best_tracking_cost = tracking_cost
                self.best_z = proposed_z.copy()
                self.best_tracking_info = proposal_info
            elif self.best_z is not None:
                # Treat each action as a proposal around the accepted best path.
                self.z = self.best_z.copy()
            new_best = (
                float(self.best_tracking_cost)
                if self.best_tracking_cost is not None
                else prev_best
            )
            reward = self.step_reward_scale * max(0.0, prev_best - new_best) / self._baseline_denom()
        else:
            reward = 0.0
        self.prev_tracking_cost = tracking_cost
        self.current_tracking_cost = tracking_cost

        self.t += 1
        terminated = (self.t >= self.episode_length)
        truncated = False

        if self.reward_mode == "best":
            reported_info = self.best_tracking_info or proposal_info
            reported_cost = float(reported_info["tracking_cost"])
            if terminated and self.terminal_linear_bonus and self.linear_tracking_cost is not None:
                reward += self.terminal_linear_bonus_coef * (
                    self.linear_tracking_cost - reported_cost
                ) / self._baseline_denom()
            if terminated and self.terminal_z0_bonus and self.z0_tracking_cost is not None:
                reward += self.terminal_z0_bonus_coef * (
                    self.z0_tracking_cost - reported_cost
                ) / self._baseline_denom()
        else:
            reported_info = proposal_info
            reported_cost = tracking_cost
            if terminated and self.terminal_linear_bonus and self.linear_tracking_cost is not None:
                reward += self.terminal_linear_bonus_coef * (
                    self.linear_tracking_cost - tracking_cost
                )
            if terminated and self.terminal_z0_bonus and self.z0_tracking_cost is not None:
                reward += self.terminal_z0_bonus_coef * (
                    self.z0_tracking_cost - tracking_cost
                )

        obs = self._make_obs()
        info = {
            "success": bool(reported_info["success"]),
            "tracking_cost": reported_cost,
            "proposal_success": success,
            "proposal_tracking_cost": tracking_cost,
            "best_tracking_cost": (
                float(self.best_tracking_cost)
                if self.best_tracking_cost is not None
                else None
            ),
            "linear_tracking_cost": (
                float(self.linear_tracking_cost)
                if self.linear_tracking_cost is not None
                else None
            ),
            "z0_tracking_cost": (
                float(self.z0_tracking_cost) if self.z0_tracking_cost is not None else None
            ),
            "accepted_steps": int(reported_info["accepted_steps"]),
            "rejected_steps": int(reported_info["rejected_steps"]),
            "total_step_attempts": int(reported_info["total_step_attempts"]),
            "total_newton_iterations": int(reported_info["total_newton_iterations"]),
            "total_evaluate_and_jacobian_calls": int(
                reported_info["total_evaluate_and_jacobian_calls"]
            ),
            "tracking_cost_mode": self.tracking_cost_mode,
            "reward_mode": self.reward_mode,
            "runtime_sec": float(reported_info["runtime_sec"]),
            "tracking_time_sec": float(reported_info["tracking_time_sec"]),
            "norm_z": float(np.linalg.norm(self.z)),
        }
        return obs, reward, terminated, truncated, info
