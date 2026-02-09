# scripts/bezier_hc_ppo/eval_saved_model.py
"""
Load a trained PPO model and evaluate on a fixed validation set.
Compares Bezier (policy) vs linear path (single gamma) on the same instances.

Usage (from repo root, env params must match training e.g. run_ppo.sh):
  python3 scripts/bezier_hc_ppo/eval_saved_model.py --model-path runs/.../ppo_continuous_action.cleanrl_model

Or with explicit env params (defaults match run_ppo.sh):
  python3 scripts/bezier_hc_ppo/eval_saved_model.py --model-path RUNS/.../model.cleanrl_model \\
    --degree 20 --bezier-degree 3 --num-instances 256
"""
import argparse
import json
import os
import sys


def _env_args(parser):
    """Add env/config args with defaults matching run_ppo.sh."""
    parser.add_argument("--model-path", type=str, required=True, help="Path to .cleanrl_model")
    parser.add_argument("--num-instances", type=int, default=1024, dest="num_instances")
    parser.add_argument("--eval-seed", type=int, default=0, dest="eval_seed")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-results", type=str, default=None, dest="save_results", help="Write JSON results to this path")

    # Env (must match training)
    parser.add_argument("--degree", type=int, default=20)
    parser.add_argument("--bezier-degree", type=int, default=3)
    parser.add_argument("--latent-dim", type=int, default=None, dest="latent_dim")
    parser.add_argument("--episode-len", type=int, default=1, dest="episode_len")
    parser.add_argument("--alpha-z", type=float, default=2.0)
    parser.add_argument("--failure-penalty", type=float, default=3000, dest="failure_penalty")
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--terminal-linear-bonus", action="store_true", dest="terminal_linear_bonus")
    parser.add_argument("--terminal-linear-bonus-coef", type=float, default=3.0, dest="terminal_linear_bonus_coef")
    parser.add_argument("--terminal-z0-bonus", action="store_true", dest="terminal_z0_bonus")
    parser.add_argument("--terminal-z0-bonus-coef", type=float, default=2.0, dest="terminal_z0_bonus_coef")
    parser.add_argument("--terminal-z0-bonus-scale", type=float, default=25.0, dest="terminal_z0_bonus_scale")
    parser.add_argument("--step-reward-scale", type=float, default=0.2, dest="step_reward_scale")
    parser.add_argument("--require-z0-success", action="store_true", dest="require_z0_success")
    parser.add_argument("--z0-max-tries", type=int, default=20, dest="z0_max_tries")
    parser.add_argument("--target-dist-real", type=str, default="uniform", dest="target_dist_real")
    parser.add_argument("--target-dist-imag", type=str, default="uniform", dest="target_dist_imag")
    parser.add_argument("--target-mean-real", type=float, default=0.0, dest="target_mean_real")
    parser.add_argument("--target-mean-imag", type=float, default=0.0, dest="target_mean_imag")
    parser.add_argument("--target-std-real", type=float, default=0.5, dest="target_std_real")
    parser.add_argument("--target-std-imag", type=float, default=0.5, dest="target_std_imag")
    parser.add_argument("--target-low-real", type=float, default=-5, dest="target_low_real")
    parser.add_argument("--target-high-real", type=float, default=5, dest="target_high_real")
    parser.add_argument("--target-low-imag", type=float, default=-5, dest="target_low_imag")
    parser.add_argument("--target-high-imag", type=float, default=5, dest="target_high_imag")
    parser.add_argument("--gamma", type=float, default=0.99)


def set_env_from_args(args):
    """Set BH_* so register_env builds the same env as training."""
    os.environ.setdefault("BH_DEGREE", str(args.degree))
    os.environ.setdefault("BH_BEZIER_DEGREE", str(args.bezier_degree))
    m = args.latent_dim if args.latent_dim is not None else args.degree
    os.environ.setdefault("BH_M", str(m))
    os.environ.setdefault("BH_T", str(args.episode_len))
    os.environ.setdefault("BH_ALPHA_Z", str(args.alpha_z))
    os.environ.setdefault("BH_FAILURE_PENALTY", str(args.failure_penalty))
    os.environ.setdefault("BH_RHO_REJECT", str(args.rho))
    os.environ.setdefault("BH_TERMINAL_LINEAR_BONUS", "1" if args.terminal_linear_bonus else "0")
    os.environ.setdefault("BH_TERMINAL_LINEAR_BONUS_COEF", str(args.terminal_linear_bonus_coef))
    os.environ.setdefault("BH_TERMINAL_Z0_BONUS", "1" if args.terminal_z0_bonus else "0")
    os.environ.setdefault("BH_TERMINAL_Z0_BONUS_COEF", str(args.terminal_z0_bonus_coef))
    os.environ.setdefault("BH_TERMINAL_Z0_BONUS_SCALE", str(args.terminal_z0_bonus_scale))
    os.environ.setdefault("BH_STEP_REWARD_SCALE", str(args.step_reward_scale))
    os.environ.setdefault("BH_REQUIRE_Z0_SUCCESS", "1" if args.require_z0_success else "0")
    os.environ.setdefault("BH_Z0_MAX_TRIES", str(args.z0_max_tries))
    os.environ.setdefault("BH_SEED", str(args.seed))
    os.environ.setdefault("BH_EXTENDED_PRECISION", "0")
    os.environ.setdefault("BH_TARGET_DIST_REAL", args.target_dist_real)
    os.environ.setdefault("BH_TARGET_DIST_IMAG", args.target_dist_imag)
    os.environ.setdefault("BH_TARGET_MEAN_REAL", str(args.target_mean_real))
    os.environ.setdefault("BH_TARGET_MEAN_IMAG", str(args.target_mean_imag))
    os.environ.setdefault("BH_TARGET_STD_REAL", str(args.target_std_real))
    os.environ.setdefault("BH_TARGET_STD_IMAG", str(args.target_std_imag))
    os.environ.setdefault("BH_TARGET_LOW_REAL", str(args.target_low_real))
    os.environ.setdefault("BH_TARGET_HIGH_REAL", str(args.target_high_real))
    os.environ.setdefault("BH_TARGET_LOW_IMAG", str(args.target_low_imag))
    os.environ.setdefault("BH_TARGET_HIGH_IMAG", str(args.target_high_imag))


def _load_run_config(model_path: str) -> dict | None:
    """Load config.json from the same directory as model_path if present."""
    run_dir = os.path.dirname(os.path.abspath(model_path))
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.isfile(config_path):
        return None
    with open(config_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved PPO model: Bezier vs linear (single gamma).")
    _env_args(parser)
    args = parser.parse_args()

    # Override env args from run's config.json if present (avoids shape mismatch)
    run_config = _load_run_config(args.model_path)
    if run_config:
        for key, value in run_config.items():
            if hasattr(args, key):
                setattr(args, key, value)

    set_env_from_args(args)

    # Import after env vars so register_env sees them. Juliacall before torch to reduce segfault risk.
    from juliacall import Main as _jl  # noqa: F401
    import torch
    import gymnasium as gym
    from ppo_continuous_action import make_env, Agent
    from eval_utils import (
        build_fixed_eval_instances,
        run_fixed_eval,
        run_linear_baseline_eval,
    )

    env_id = "hc_envs.register_env:BezierHomotopyUnivar-v0"
    run_name = "eval"
    gamma = args.gamma

    eval_env = make_env(env_id, 0, False, run_name, gamma)()
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, False, run_name, gamma)])
    device = torch.device(args.device)

    agent = Agent(envs).to(device)
    state = torch.load(args.model_path, map_location=device)
    agent.load_state_dict(state)
    agent.eval()

    instances = build_fixed_eval_instances(eval_env, args.num_instances, args.eval_seed)
    eval_env.unwrapped.set_fixed_instances(instances, reset_idx=True)

    bezier = run_fixed_eval(eval_env, agent, device, args.num_instances)
    eval_env.unwrapped.set_fixed_instances(instances, reset_idx=True)
    linear = run_linear_baseline_eval(eval_env, args.num_instances)

    out = {
        "model_path": args.model_path,
        "num_instances": args.num_instances,
        "eval_seed": args.eval_seed,
        "bezier": bezier,
        "linear": linear,
    }

    print("eval_saved_model (Bezier vs linear single-gamma)")
    print(f"  model_path = {args.model_path}")
    print(f"  num_instances = {args.num_instances}  eval_seed = {args.eval_seed}")
    print("  Bezier (policy):")
    for k, v in (bezier or {}).items():
        print(f"    {k} = {v}")
    print("  Linear (single gamma):")
    for k, v in (linear or {}).items():
        print(f"    {k} = {v}")

    if args.save_results:
        with open(args.save_results, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  results saved to {args.save_results}")

    return out


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
