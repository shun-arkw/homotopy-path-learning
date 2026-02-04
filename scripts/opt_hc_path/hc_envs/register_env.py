# hc_envs/register_env.py
import os
import gymnasium as gym

from hc_envs.bezier_univar_env import TargetCoeffConfig


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
        seed=int(os.environ.get("BH_SEED", "0")),
        extended_precision=(os.environ.get("BH_EXTENDED_PRECISION", "0") == "1"),
        compute_newton_iters=False,  # Keep False during training
        include_progress_features=True,
        f_coeff_config=f_coeff_config,
    )


def register_bezier_env() -> None:
    env_id = "BezierHomotopyUnivar-v0"
    if env_id in gym.registry:
        return

    gym.register(
        id=env_id,
        entry_point="hc_envs.bezier_univar_env:BezierHomotopyUnivarEnv",
        kwargs=_env_kwargs_from_envvars(),
    )


# Register on import so that `module:EnvID` works in a fresh subprocess.
register_bezier_env()
