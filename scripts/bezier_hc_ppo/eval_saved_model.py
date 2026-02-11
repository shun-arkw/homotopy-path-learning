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
    parser.add_argument("--top-k", type=int, default=1, dest="top_k", help="Number of top improvement (linear - Bezier) instances to report (default: 1)")
    parser.add_argument("--worst-k", type=int, default=0, dest="worst_k", help="Number of worst improvement instances to report (0 = disabled, default: 0)")
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
    parser.add_argument("--step-reward-scale", type=float, default=0.2, dest="step_reward_scale")
    parser.add_argument("--require-z0-success", action="store_true", dest="require_z0_success")
    parser.add_argument("--z0-max-tries", type=int, default=20, dest="z0_max_tries")
    parser.add_argument(
        "--hc-gamma-trick",
        action="store_true",
        dest="hc_gamma_trick",
        help="Enable homotopy gamma trick (BH_GAMMA_TRICK=1).",
    )
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
    parser.add_argument(
        "--compute-newton-iters",
        type=str,
        default="true",
        choices=["true", "false"],
        dest="compute_newton_iters",
        help="Count Newton iterations: true (slower, reports total_newton_iterations_*), false (faster).",
    )


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
    os.environ.setdefault("BH_STEP_REWARD_SCALE", str(args.step_reward_scale))
    os.environ.setdefault("BH_REQUIRE_Z0_SUCCESS", "1" if args.require_z0_success else "0")
    os.environ.setdefault("BH_Z0_MAX_TRIES", str(args.z0_max_tries))
    os.environ.setdefault("BH_GAMMA_TRICK", "1" if args.hc_gamma_trick else "0")
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


def _apply_hc_tracker_config(run_config: dict) -> None:
    """Set BH_HC_* env vars from config.json hc_tracker_params and hc_tracker_options."""
    def _set(k: str, v) -> None:
        env_key = "BH_" + k.upper()
        if isinstance(v, bool):
            os.environ.setdefault(env_key, "1" if v else "0")
        else:
            os.environ.setdefault(env_key, str(v))

    for section in ("hc_tracker_params", "hc_tracker_options"):
        for key, value in (run_config.get(section) or {}).items():
            _set(key, value)


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved PPO model: Bezier vs linear (single gamma).")
    _env_args(parser)
    # Get model_path first so we can load run config, then set config as parser defaults so CLI overrides
    args_pre, _ = parser.parse_known_args()
    run_config = _load_run_config(args_pre.model_path)
    if run_config:
        flat = {}
        for key, value in run_config.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    if hasattr(args_pre, k):
                        flat[k] = v
            elif hasattr(args_pre, key):
                flat[key] = value
        if flat:
            parser.set_defaults(**flat)
    args = parser.parse_args()

    # Apply saved HC tracker config so eval uses same tracker as training
    if run_config:
        _apply_hc_tracker_config(run_config)
    # Newton iteration counting (env reads BH_COMPUTE_NEWTON_ITERS on register)
    os.environ["BH_COMPUTE_NEWTON_ITERS"] = "1" if (args.compute_newton_iters == "true") else "0"
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

    bezier = run_fixed_eval(
        eval_env, agent, device, args.num_instances, return_per_instance=True
    )
    eval_env.unwrapped.set_fixed_instances(instances, reset_idx=True)
    linear = run_linear_baseline_eval(
        eval_env, args.num_instances, return_per_instance=True
    )

    # Aggregate stats for output (exclude per-instance lists from printed summary)
    def _summary(d: dict):
        exclude = {"total_step_attempts_list", "total_newton_iterations_list"}
        return {k: v for k, v in (d or {}).items() if k not in exclude}

    out = {
        "model_path": args.model_path,
        "num_instances": args.num_instances,
        "eval_seed": args.eval_seed,
        "bezier": _summary(bezier),
        "linear": _summary(linear),
    }

    # When Newton iters are measured, tracking is slower. Re-run with BH_COMPUTE_NEWTON_ITERS=0 and overwrite tracking_time_sec_* only.
    if args.compute_newton_iters == "true":
        from hc_envs.register_env import unregister_bezier_env, register_bezier_env

        prev = os.environ.get("BH_COMPUTE_NEWTON_ITERS")
        os.environ["BH_COMPUTE_NEWTON_ITERS"] = "0"
        try:
            unregister_bezier_env()
            register_bezier_env()
            eval_env_tt = make_env(env_id, 0, False, run_name, gamma)()
            eval_env_tt.unwrapped.set_fixed_instances(instances, reset_idx=True)
            bezier_tt = run_fixed_eval(
                eval_env_tt, agent, device, args.num_instances, return_per_instance=False
            )
            eval_env_tt.unwrapped.set_fixed_instances(instances, reset_idx=True)
            linear_tt = run_linear_baseline_eval(
                eval_env_tt, args.num_instances, return_per_instance=False
            )
            if out.get("bezier") and bezier_tt:
                for key in list(out["bezier"].keys()):
                    if "tracking_time_sec" in key and key in bezier_tt:
                        out["bezier"][key] = bezier_tt[key]
            if out.get("linear") and linear_tt:
                for key in list(out["linear"].keys()):
                    if "tracking_time_sec" in key and key in linear_tt:
                        out["linear"][key] = linear_tt[key]
            if hasattr(eval_env_tt, "close"):
                eval_env_tt.close()
        except Exception as e:
            print(f"  Warning: in-process tracking_time re-eval failed: {e}", file=sys.stderr)
        finally:
            if prev is not None:
                os.environ["BH_COMPUTE_NEWTON_ITERS"] = prev
            else:
                os.environ.pop("BH_COMPUTE_NEWTON_ITERS", None)

    print("eval_saved_model (Bezier vs linear single-gamma)")
    print(f"  model_path = {args.model_path}")
    print(f"  num_instances = {args.num_instances}  eval_seed = {args.eval_seed}")
    print("  Bezier (policy):")
    for k, v in (out["bezier"] or {}).items():
        print(f"    {k} = {v}")
    print("  Linear (single gamma):")
    for k, v in (out["linear"] or {}).items():
        print(f"    {k} = {v}")

    # Improvement = linear - Bezier (positive = Bezier is better); top-k by improvement
    if bezier and linear and "total_step_attempts_list" in bezier and "total_step_attempts_list" in linear:
        blist = bezier["total_step_attempts_list"]
        llist = linear["total_step_attempts_list"]
        n = min(len(blist), len(llist))
        if n > 0:
            improvements = [llist[i] - blist[i] for i in range(n)]
            k = max(1, min(args.top_k, n))
            top_indices = sorted(range(n), key=lambda i: improvements[i], reverse=True)[:k]
            top_k_list = [
                {
                    "rank": rank + 1,
                    "instance_index": int(idx),
                    "improvement_linear_minus_bezier": float(improvements[idx]),
                    "bezier_total_step_attempts": float(blist[idx]),
                    "linear_total_step_attempts": float(llist[idx]),
                }
                for rank, idx in enumerate(top_indices)
            ]
            out["top_k_improvement"] = top_k_list
            if k == 1:
                out["max_improvement"] = top_k_list[0]
            print(f"  Top-{k} improvement (linear - Bezier) [largest gain for Bezier]:")
            for rank, idx in enumerate(top_indices):
                print(f"    rank = {rank + 1}  instance_index = {idx}  improvement = {improvements[idx]:.4f}  Bezier = {blist[idx]:.4f}  Linear = {llist[idx]:.4f}")

            # Worst-k: smallest improvement (Bezier barely better or worse than Linear)
            if args.worst_k > 0:
                kw = min(args.worst_k, n)
                worst_indices = sorted(range(n), key=lambda i: improvements[i])[:kw]
                worst_k_list = [
                    {
                        "rank": rank + 1,
                        "instance_index": int(idx),
                        "improvement_linear_minus_bezier": float(improvements[idx]),
                        "bezier_total_step_attempts": float(blist[idx]),
                        "linear_total_step_attempts": float(llist[idx]),
                    }
                    for rank, idx in enumerate(worst_indices)
                ]
                out["worst_k_improvement"] = worst_k_list
                print(f"  Top-{kw} worst improvement (linear - Bezier) [smallest gain or Bezier worse]:")
                for rank, idx in enumerate(worst_indices):
                    print(f"    rank = {rank + 1}  instance_index = {idx}  improvement = {improvements[idx]:.4f}  Bezier = {blist[idx]:.4f}  Linear = {llist[idx]:.4f}")

    # Newton iterations: same Top-k / Worst-k when per-instance lists exist (--compute-newton-iters true)
    if bezier and linear and "total_newton_iterations_list" in bezier and "total_newton_iterations_list" in linear:
        bnlist = bezier["total_newton_iterations_list"]
        lnlist = linear["total_newton_iterations_list"]
        nn = min(len(bnlist), len(lnlist))
        if nn > 0:
            nimprovements = [lnlist[i] - bnlist[i] for i in range(nn)]
            nk = max(1, min(args.top_k, nn))
            ntop_indices = sorted(range(nn), key=lambda i: nimprovements[i], reverse=True)[:nk]
            out["top_k_newton_improvement"] = [
                {
                    "rank": rank + 1,
                    "instance_index": int(idx),
                    "improvement_linear_minus_bezier": float(nimprovements[idx]),
                    "bezier_total_newton_iterations": float(bnlist[idx]),
                    "linear_total_newton_iterations": float(lnlist[idx]),
                }
                for rank, idx in enumerate(ntop_indices)
            ]
            print(f"  Top-{nk} improvement (linear - Bezier) [Newton iterations]:")
            for rank, idx in enumerate(ntop_indices):
                print(f"    rank = {rank + 1}  instance_index = {idx}  improvement = {nimprovements[idx]:.4f}  Bezier = {bnlist[idx]:.4f}  Linear = {lnlist[idx]:.4f}")
            if args.worst_k > 0:
                nkw = min(args.worst_k, nn)
                nworst_indices = sorted(range(nn), key=lambda i: nimprovements[i])[:nkw]
                out["worst_k_newton_improvement"] = [
                    {
                        "rank": rank + 1,
                        "instance_index": int(idx),
                        "improvement_linear_minus_bezier": float(nimprovements[idx]),
                        "bezier_total_newton_iterations": float(bnlist[idx]),
                        "linear_total_newton_iterations": float(lnlist[idx]),
                    }
                    for rank, idx in enumerate(nworst_indices)
                ]
                print(f"  Top-{nkw} worst improvement (linear - Bezier) [Newton iterations]:")
                for rank, idx in enumerate(nworst_indices):
                    print(f"    rank = {rank + 1}  instance_index = {idx}  improvement = {nimprovements[idx]:.4f}  Bezier = {bnlist[idx]:.4f}  Linear = {lnlist[idx]:.4f}")

    if args.save_results:
        with open(args.save_results, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  results saved to {args.save_results}")

        # Always write human-readable eval.log next to the JSON
        log_path = os.path.join(os.path.dirname(args.save_results), "eval.log")
        lines = [
            "# Evaluation results",
            f"model_path: {out['model_path']}",
            f"num_instances: {out['num_instances']}",
            f"eval_seed: {out['eval_seed']}",
            "",
            "## Bezier vs Linear",
        ]
        b, l = out.get("bezier") or {}, out.get("linear") or {}
        all_keys = sorted(set(b) | set(l))
        # Conventional stat order: mean, median, std, min, max
        _stat_suffix_order = ("mean", "median", "std", "min", "max")

        def _metric_sort_key(k: str):
            for i, suf in enumerate(_stat_suffix_order):
                if k.endswith("_" + suf):
                    return (i, k)
            return (len(_stat_suffix_order), k)

        # Group order: success_rate; accepted/rejected; step_attempts; tracking_cost; newton; tracking_time_sec (separated)
        group1 = [k for k in all_keys if k == "success_rate"]
        group2 = sorted(
            [k for k in all_keys if k in ("accepted_steps_mean", "rejected_steps_mean")],
            key=_metric_sort_key,
        )
        group3 = sorted([k for k in all_keys if "total_step_attempts" in k], key=_metric_sort_key)
        group4 = [k for k in all_keys if k == "tracking_cost_mean"]
        group5 = sorted([k for k in all_keys if "total_newton_iterations" in k], key=_metric_sort_key)
        group6 = sorted([k for k in all_keys if "tracking_time_sec" in k], key=_metric_sort_key)
        key_groups = [g for g in (group1, group2, group3, group4, group5, group6) if g]
        def _label_len(key):
            if "tracking_time_sec" in key:
                return len(key.replace("tracking_time_sec", "tracking_time", 1)) + 5  # "(ms)" suffix
            return len(key)
        name_w = max(len("metric"), max(_label_len(k) for k in all_keys) if all_keys else 0)
        col_w = 22  # wide enough for "value (XX.XX%)"
        no_improvement_keys = ("success_rate", "accepted_steps_mean", "rejected_steps_mean")
        sep_line = "  " + "-" * (name_w + 2 + col_w * 2 + 2)
        lines.append(f"  {'metric':<{name_w}}  {'Bezier':>{col_w}}  {'Linear':>{col_w}}")
        lines.append(sep_line)
        for gi, keys in enumerate(key_groups):
            for k in keys:
                bv, lv = b.get(k, ""), l.get(k, "")
                # Time metrics: display in ms (sec * 1000), 2 decimal places. Others: 1 decimal place.
                is_time = "tracking_time_sec" in k
                fmt = ".4f" if is_time else ".1f"
                scale = 1000.0 if is_time else 1.0
                if isinstance(bv, (int, float)):
                    sb = f"{float(bv) * scale:{fmt}}"
                else:
                    sb = "-" if bv == "" else str(bv)
                if isinstance(lv, (int, float)):
                    sl = f"{float(lv) * scale:{fmt}}"
                else:
                    sl = "-" if lv == "" else str(lv)
                # Improvement = (Linear - Bezier) / Linear * 100. Append in parentheses after Bezier when applicable.
                if k not in no_improvement_keys and isinstance(lv, (int, float)) and isinstance(bv, (int, float)) and lv != 0:
                    imp = (lv - bv) / lv * 100
                    sb = f"{sb} ({imp:5.2f}%)"
                label = f"{k.replace('tracking_time_sec', 'tracking_time', 1)} (ms)" if is_time else k
                lines.append(f"  {label:<{name_w}}  {sb:>{col_w}}  {sl:>{col_w}}")
            if gi < len(key_groups) - 1:
                lines.append(sep_line)
        if out.get("top_k_improvement"):
            lines.extend(["", "## Top-k improvement (linear - Bezier) [Steps]"])
            lines.append("  rank  instance_index  improvement  bezier_steps  linear_steps")
            for e in out["top_k_improvement"]:
                lines.append(
                    f"  {e['rank']:<4}  {e['instance_index']:<14}  {e['improvement_linear_minus_bezier']:<11}  {e['bezier_total_step_attempts']:<12}  {e['linear_total_step_attempts']}"
                )
        if out.get("worst_k_improvement"):
            lines.extend(["", "## Worst-k improvement (linear - Bezier) [Steps]"])
            lines.append("  rank  instance_index  improvement  bezier_steps  linear_steps")
            for e in out["worst_k_improvement"]:
                lines.append(
                    f"  {e['rank']:<4}  {e['instance_index']:<14}  {e['improvement_linear_minus_bezier']:<11}  {e['bezier_total_step_attempts']:<12}  {e['linear_total_step_attempts']}"
                )
        if out.get("top_k_newton_improvement"):
            lines.extend(["", "## Top-k improvement (linear - Bezier) [Newton iterations]"])
            lines.append("  rank  instance_index  improvement  bezier_newton  linear_newton")
            for e in out["top_k_newton_improvement"]:
                lines.append(
                    f"  {e['rank']:<4}  {e['instance_index']:<14}  {e['improvement_linear_minus_bezier']:<11}  {e['bezier_total_newton_iterations']:<14}  {e['linear_total_newton_iterations']}"
                )
        if out.get("worst_k_newton_improvement"):
            lines.extend(["", "## Worst-k improvement (linear - Bezier) [Newton iterations]"])
            lines.append("  rank  instance_index  improvement  bezier_newton  linear_newton")
            for e in out["worst_k_newton_improvement"]:
                lines.append(
                    f"  {e['rank']:<4}  {e['instance_index']:<14}  {e['improvement_linear_minus_bezier']:<11}  {e['bezier_total_newton_iterations']:<14}  {e['linear_total_newton_iterations']}"
                )
        with open(log_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"  log saved to {log_path}")

    return out


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
