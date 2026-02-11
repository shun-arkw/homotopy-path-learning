# hc_envs/register_env.py
import os
import gymnasium as gym

from hc_envs.bezier_univar_env import TargetCoeffConfig


def _parse_float_or_inf(value: str) -> float:
    """Parse env var as float; 'inf' or 'infinity' -> float('inf')."""
    v = (value or "").strip().lower()
    if v in ("inf", "infinity"):
        return float("inf")
    return float(value)


def _env_kwargs_from_envvars() -> dict:
    # Read config from environment variables so subprocess can pass them.
    f_coeff_config = TargetCoeffConfig(
        dist_real=os.environ.get("BH_TARGET_DIST_REAL", "gaussian"),
        dist_imag=os.environ.get("BH_TARGET_DIST_IMAG", "gaussian"),
        mean_real=float(os.environ.get("BH_TARGET_MEAN_REAL", "0.0")),
        mean_imag=float(os.environ.get("BH_TARGET_MEAN_IMAG", "0.0")),
        std_real=float(os.environ.get("BH_TARGET_STD_REAL", "0.5")),
        std_imag=float(os.environ.get("BH_TARGET_STD_IMAG", "0.5")),
        low_real=float(os.environ.get("BH_TARGET_LOW_REAL", "-0.5")),
        high_real=float(os.environ.get("BH_TARGET_HIGH_REAL", "0.5")),
        low_imag=float(os.environ.get("BH_TARGET_LOW_IMAG", "-0.5")),
        high_imag=float(os.environ.get("BH_TARGET_HIGH_IMAG", "0.5")),
    )
    return dict(
        degree=int(os.environ.get("BH_DEGREE", "10")),
        bezier_degree=int(os.environ.get("BH_BEZIER_DEGREE", "2")),
        latent_dim_m=int(os.environ.get("BH_M", "8")),
        episode_len_T=int(os.environ.get("BH_T", "8")),
        alpha_z=float(os.environ.get("BH_ALPHA_Z", "2.0")),
        failure_penalty=float(os.environ.get("BH_FAILURE_PENALTY", "1000000")),
        rho_reject=float(os.environ.get("BH_RHO_REJECT", "1.0")),
        terminal_linear_bonus=(os.environ.get("BH_TERMINAL_LINEAR_BONUS", "1") == "1"),
        terminal_linear_bonus_coef=float(os.environ.get("BH_TERMINAL_LINEAR_BONUS_COEF", "1.0")),
        terminal_z0_bonus=(os.environ.get("BH_TERMINAL_Z0_BONUS", "0") == "1"),
        terminal_z0_bonus_coef=float(os.environ.get("BH_TERMINAL_Z0_BONUS_COEF", "1.0")),
        step_reward_scale=float(os.environ.get("BH_STEP_REWARD_SCALE", "1.0")),
        require_z0_success=(os.environ.get("BH_REQUIRE_Z0_SUCCESS", "0") == "1"),
        z0_max_tries=int(os.environ.get("BH_Z0_MAX_TRIES", "10")),
        gamma_trick=(os.environ.get("BH_GAMMA_TRICK", "1") == "1"),
        seed=int(os.environ.get("BH_SEED", "0")),
        extended_precision=(os.environ.get("BH_EXTENDED_PRECISION", "0") == "1"),
        compute_newton_iters=(os.environ.get("BH_COMPUTE_NEWTON_ITERS", "0") == "1"),
        include_progress_features=True,
        f_coeff_config=f_coeff_config,
        # HC TrackerParameters (shared linear/bezier)
        hc_a=float(os.environ.get("BH_HC_A", "0.125")),
        hc_beta_a=float(os.environ.get("BH_HC_BETA_A", "1.0")),
        hc_beta_omega_p=float(os.environ.get("BH_HC_BETA_OMEGA_P", "0.8")),
        hc_beta_tau=float(os.environ.get("BH_HC_BETA_TAU", "0.85")),
        hc_strict_beta_tau=float(os.environ.get("BH_HC_STRICT_BETA_TAU", "0.8")),
        hc_min_newton_iters=int(os.environ.get("BH_HC_MIN_NEWTON_ITERS", "1")),
        # HC TrackerOptions (shared linear/bezier)
        hc_max_steps=int(os.environ.get("BH_HC_MAX_STEPS", "50000")),
        hc_max_step_size=_parse_float_or_inf(os.environ.get("BH_HC_MAX_STEP_SIZE", "inf")),
        hc_max_initial_step_size=_parse_float_or_inf(os.environ.get("BH_HC_MAX_INITIAL_STEP_SIZE", "inf")),
        hc_min_step_size=float(os.environ.get("BH_HC_MIN_STEP_SIZE", "1e-12")),
        hc_extended_precision=(os.environ.get("BH_HC_EXTENDED_PRECISION", "0") == "1"),
    )


BEZIER_ENV_ID = "BezierHomotopyUnivar-v0"


def register_bezier_env() -> None:
    if BEZIER_ENV_ID in gym.registry:
        return

    gym.register(
        id=BEZIER_ENV_ID,
        entry_point="hc_envs.bezier_univar_env:BezierHomotopyUnivarEnv",
        kwargs=_env_kwargs_from_envvars(),
    )


def unregister_bezier_env() -> None:
    """Remove Bezier env from gym registry so it can be re-registered with new kwargs (e.g. BH_COMPUTE_NEWTON_ITERS=0)."""
    # Registry key may be short id or namespace/id depending on gymnasium version.
    for key in (BEZIER_ENV_ID, f"hc_envs.register_env/{BEZIER_ENV_ID}"):
        if key in gym.registry:
            try:
                del gym.registry[key]
            except TypeError:
                reg = gym.registry
                if hasattr(reg, "env_specs"):
                    reg.env_specs.pop(key, None)
                elif hasattr(reg, "_env_specs"):
                    reg._env_specs.pop(key, None)
            break


# Register on import so that `module:EnvID` works in a fresh subprocess.
register_bezier_env()
