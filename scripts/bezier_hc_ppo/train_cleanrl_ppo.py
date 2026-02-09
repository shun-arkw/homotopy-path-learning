# scripts/train_cleanrl_ppo.py
import argparse
import subprocess
import sys
import os


def main():
    parser = argparse.ArgumentParser()

    # Env / problem parameters
    parser.add_argument("--degree", type=int, default=10)
    parser.add_argument("--bezier-degree", type=int, default=2)
    parser.add_argument("--latent-dim", type=int, default=10, dest="latent_dim")
    parser.add_argument("--episode-len", type=int, default=8, dest="episode_len")
    parser.add_argument("--alpha-z", type=float, default=2.0)
    parser.add_argument("--failure-penalty", type=float, default=1e6, dest="failure_penalty")
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument(
        "--terminal-linear-bonus",
        action="store_true",
        dest="terminal_linear_bonus",
        help="Add terminal reward based on linear baseline cost.",
    )
    parser.add_argument(
        "--terminal-linear-bonus-coef",
        type=float,
        default=1.0,
        dest="terminal_linear_bonus_coef",
    )
    parser.add_argument(
        "--terminal-z0-bonus",
        action="store_true",
        dest="terminal_z0_bonus",
        help="Add terminal reward based on z=0 Bezier cost.",
    )
    parser.add_argument(
        "--terminal-z0-bonus-coef",
        type=float,
        default=1.0,
        dest="terminal_z0_bonus_coef",
    )
    parser.add_argument(
        "--terminal-z0-bonus-scale",
        type=float,
        default=100.0,
        dest="terminal_z0_bonus_scale",
    )
    parser.add_argument(
        "--step-reward-scale",
        type=float,
        default=1.0,
        dest="step_reward_scale",
    )
    parser.add_argument(
        "--require-z0-success",
        action="store_true",
        dest="require_z0_success",
        help="Resample gamma until z=0 Bezier succeeds.",
    )
    parser.add_argument(
        "--z0-max-tries",
        type=int,
        default=10,
        dest="z0_max_tries",
    )
    parser.add_argument("--seed", type=int, default=0)

    # TargetCoeffConfig: sampling of target polynomial coefficients
    parser.add_argument("--target-dist-real", type=str, default="gaussian", choices=("gaussian", "uniform"), dest="target_dist_real")
    parser.add_argument("--target-dist-imag", type=str, default="gaussian", choices=("gaussian", "uniform"), dest="target_dist_imag")
    parser.add_argument("--target-mean-real", type=float, default=0.0, dest="target_mean_real")
    parser.add_argument("--target-mean-imag", type=float, default=0.0, dest="target_mean_imag")
    parser.add_argument("--target-std-real", type=float, default=0.5, dest="target_std_real")
    parser.add_argument("--target-std-imag", type=float, default=0.5, dest="target_std_imag")
    parser.add_argument("--target-low-real", type=float, default=-0.5, dest="target_low_real")
    parser.add_argument("--target-high-real", type=float, default=0.5, dest="target_high_real")
    parser.add_argument("--target-low-imag", type=float, default=-0.5, dest="target_low_imag")
    parser.add_argument("--target-high-imag", type=float, default=0.5, dest="target_high_imag")

    # PPO parameters you may want to change frequently
    parser.add_argument("--total-timesteps", type=int, default=100_000, dest="total_timesteps")
    parser.add_argument("--num-steps", type=int, default=256, dest="num_steps")
    parser.add_argument("--num-envs", type=int, default=1, dest="num_envs")
    parser.add_argument("--learning-rate", type=float, default=3e-4, dest="learning_rate")
    parser.add_argument("--update-epochs", type=int, default=10, dest="update_epochs")
    parser.add_argument("--num-minibatches", type=int, default=32, dest="num_minibatches")
    parser.add_argument("--gamma", type=float, default=0.99, dest="gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95, dest="gae_lambda")

    # Logging / tracking (forwarded to CleanRL tyro Args)
    parser.add_argument("--eval-interval", type=int, default=10, dest="eval_interval")
    parser.add_argument("--eval-num-instances", type=int, default=256, dest="eval_num_instances")
    parser.add_argument("--eval-seed", type=int, default=0, dest="eval_seed")
    parser.add_argument("--eval-linear-baseline", action="store_true", dest="eval_linear_baseline")
    parser.add_argument("--save-model", action="store_true", dest="save_model", help="Save model to save_dir/run_name after training.")
    parser.add_argument("--save-dir", type=str, default="runs", dest="save_dir", help="Base directory for run logs and saved model.")
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--wandb-project-name", type=str, default=None, dest="wandb_project_name")
    parser.add_argument("--wandb-entity", type=str, default=None, dest="wandb_entity")

    args, unknown = parser.parse_known_args()

    # Pass env parameters to the subprocess via environment variables.
    os.environ["BH_DEGREE"] = str(args.degree)
    os.environ["BH_BEZIER_DEGREE"] = str(args.bezier_degree)
    os.environ["BH_M"] = str(args.latent_dim)
    os.environ["BH_T"] = str(args.episode_len)
    os.environ["BH_ALPHA_Z"] = str(args.alpha_z)
    os.environ["BH_FAILURE_PENALTY"] = str(args.failure_penalty)
    os.environ["BH_RHO_REJECT"] = str(args.rho)
    os.environ["BH_TERMINAL_LINEAR_BONUS"] = "1" if args.terminal_linear_bonus else "0"
    os.environ["BH_TERMINAL_LINEAR_BONUS_COEF"] = str(args.terminal_linear_bonus_coef)
    os.environ["BH_TERMINAL_Z0_BONUS"] = "1" if args.terminal_z0_bonus else "0"
    os.environ["BH_TERMINAL_Z0_BONUS_COEF"] = str(args.terminal_z0_bonus_coef)
    os.environ["BH_TERMINAL_Z0_BONUS_SCALE"] = str(args.terminal_z0_bonus_scale)
    os.environ["BH_STEP_REWARD_SCALE"] = str(args.step_reward_scale)
    os.environ["BH_REQUIRE_Z0_SUCCESS"] = "1" if args.require_z0_success else "0"
    os.environ["BH_Z0_MAX_TRIES"] = str(args.z0_max_tries)
    os.environ["BH_SEED"] = str(args.seed)
    os.environ["BH_EXTENDED_PRECISION"] = "0"
    # TargetCoeffConfig
    os.environ["BH_TARGET_DIST_REAL"] = args.target_dist_real
    os.environ["BH_TARGET_DIST_IMAG"] = args.target_dist_imag
    os.environ["BH_TARGET_MEAN_REAL"] = str(args.target_mean_real)
    os.environ["BH_TARGET_MEAN_IMAG"] = str(args.target_mean_imag)
    os.environ["BH_TARGET_STD_REAL"] = str(args.target_std_real)
    os.environ["BH_TARGET_STD_IMAG"] = str(args.target_std_imag)
    os.environ["BH_TARGET_LOW_REAL"] = str(args.target_low_real)
    os.environ["BH_TARGET_HIGH_REAL"] = str(args.target_high_real)
    os.environ["BH_TARGET_LOW_IMAG"] = str(args.target_low_imag)
    os.environ["BH_TARGET_HIGH_IMAG"] = str(args.target_high_imag)

    cmd = [
        sys.executable,
        "scripts/bezier_hc_ppo/ppo_continuous_action.py",
        "--env-id", "hc_envs.register_env:BezierHomotopyUnivar-v0",
        "--seed", str(args.seed),
        "--num-envs", str(args.num_envs),
        "--total-timesteps", str(args.total_timesteps),
        "--num-steps", str(args.num_steps),
        "--learning-rate", str(args.learning_rate),
        "--update-epochs", str(args.update_epochs),
        "--num-minibatches", str(args.num_minibatches),
        "--gamma", str(args.gamma),
        "--gae-lambda", str(args.gae_lambda),
        "--eval-interval", str(args.eval_interval),
        "--eval-num-instances", str(args.eval_num_instances),
        "--eval-seed", str(args.eval_seed),
    ]
    if args.eval_linear_baseline:
        cmd.append("--eval-linear-baseline")
    if args.save_model:
        cmd.append("--save-model")
    cmd += ["--save-dir", args.save_dir]

    # Forward tracking flags explicitly.
    if args.track:
        cmd.append("--track")
    if args.wandb_project_name is not None:
        cmd += ["--wandb-project-name", args.wandb_project_name]
    if args.wandb_entity is not None:
        cmd += ["--wandb-entity", args.wandb_entity]

    # Forward any extra args as-is (e.g., --capture-video, --ent-coef, etc.).
    cmd += unknown

    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
