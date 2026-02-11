# Evaluation helpers for fixed validation set (policy eval and linear baseline).
# Used by ppo_continuous_action.py; can be reused by standalone eval scripts.
import numpy as np
import torch

from hc_envs.julia_backend import LinearUnivarConfig


def build_fixed_eval_instances(eval_env, num_instances: int, seed: int):
    """Build a fixed set of problem instances for validation using the env's sampler."""
    base_env = eval_env.unwrapped
    base_env.rng = np.random.default_rng(int(seed))
    instances = [base_env._sample_instance() for _ in range(int(num_instances))]
    base_env.set_fixed_instances(instances, reset_idx=True)
    return instances


def run_fixed_eval(
    eval_env,
    agent,
    device,
    num_instances: int,
    force_action_zero: bool = False,
    return_per_instance: bool = False,
) -> dict:
    """Run policy (agent) on fixed eval instances; return mean metrics dict.
    If return_per_instance=True, adds "total_step_attempts_list" and "total_newton_iterations_list" to the returned dict.
    """
    successes = []
    tracking_costs = []
    tracking_time_secs = []
    total_attempts = []
    total_newton_iters = []
    accepted_steps = []
    rejected_steps = []
    zero_action = None
    if force_action_zero:
        zero_action = np.zeros(eval_env.action_space.shape, dtype=np.float32)
    with torch.no_grad():
        for _ in range(int(num_instances)):
            obs, _ = eval_env.reset()
            done = False
            last_info = None
            while not done:
                if force_action_zero:
                    action = zero_action
                else:
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
                    if obs_tensor.ndim == 1:
                        obs_tensor = obs_tensor.unsqueeze(0)
                    action_mean = agent.actor_mean(obs_tensor)
                    action = action_mean.cpu().numpy()[0]
                obs, _, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                last_info = info
            if last_info is not None:
                successes.append(float(last_info.get("success", 0.0)))
                tracking_costs.append(float(last_info.get("tracking_cost", 0.0)))
                tracking_time_secs.append(float(last_info.get("tracking_time_sec", 0.0)))
                total_attempts.append(float(last_info.get("total_step_attempts", 0.0)))
                total_newton_iters.append(float(last_info.get("total_newton_iterations", 0.0)))
                accepted_steps.append(float(last_info.get("accepted_steps", 0.0)))
                rejected_steps.append(float(last_info.get("rejected_steps", 0.0)))
    if not successes:
        return {}
    total_attempts_arr = np.array(total_attempts, dtype=np.float64)
    out = {
        "success_rate": float(np.mean(successes)),
        "tracking_cost_mean": float(np.mean(tracking_costs)),
        "total_step_attempts_mean": float(np.mean(total_attempts)),
        "total_step_attempts_median": float(np.median(total_attempts_arr)),
        "total_step_attempts_min": float(np.min(total_attempts_arr)),
        "total_step_attempts_max": float(np.max(total_attempts_arr)),
        "total_step_attempts_std": float(np.std(total_attempts_arr)),
        "accepted_steps_mean": float(np.mean(accepted_steps)),
        "rejected_steps_mean": float(np.mean(rejected_steps)),
    }
    if tracking_time_secs:
        tarr = np.array(tracking_time_secs, dtype=np.float64)
        out["tracking_time_sec_mean"] = float(np.mean(tracking_time_secs))
        out["tracking_time_sec_median"] = float(np.median(tarr))
        out["tracking_time_sec_std"] = float(np.std(tarr))
        out["tracking_time_sec_min"] = float(np.min(tarr))
        out["tracking_time_sec_max"] = float(np.max(tarr))
    else:
        out["tracking_time_sec_mean"] = 0.0
    if total_newton_iters:
        newton_arr = np.array(total_newton_iters, dtype=np.float64)
        out["total_newton_iterations_mean"] = float(np.mean(total_newton_iters))
        out["total_newton_iterations_median"] = float(np.median(newton_arr))
        out["total_newton_iterations_min"] = float(np.min(newton_arr))
        out["total_newton_iterations_max"] = float(np.max(newton_arr))
        out["total_newton_iterations_std"] = float(np.std(newton_arr))
    if return_per_instance:
        out["total_step_attempts_list"] = list(total_attempts)
        if total_newton_iters:
            out["total_newton_iterations_list"] = list(total_newton_iters)
    return out


def run_linear_baseline_eval(
    eval_env, num_instances: int, return_per_instance: bool = False
) -> dict:
    """Run linear-path baseline via Julia linear homotopy; return mean metrics dict.
    If return_per_instance=True, adds "total_step_attempts_list" and "total_newton_iterations_list" to the returned dict.
    """
    successes = []
    tracking_costs = []
    tracking_time_secs = []
    total_attempts = []
    total_newton_iters = []
    accepted_steps = []
    rejected_steps = []
    base_env = eval_env.unwrapped
    backend = base_env.backend
    linear_cfg = LinearUnivarConfig(
        degree=base_env.degree,
        seed=base_env.seed0,
        compute_newton_iters=base_env.compute_newton_iters,
        extended_precision=base_env.hc_extended_precision,
        max_steps=base_env.hc_max_steps,
        max_step_size=base_env.hc_max_step_size,
        max_initial_step_size=base_env.hc_max_initial_step_size,
        min_step_size=base_env.hc_min_step_size,
        hc_a=base_env.hc_a,
        hc_beta_a=base_env.hc_beta_a,
        hc_beta_omega_p=base_env.hc_beta_omega_p,
        hc_beta_tau=base_env.hc_beta_tau,
        hc_strict_beta_tau=base_env.hc_strict_beta_tau,
        hc_min_newton_iters=base_env.hc_min_newton_iters,
    )
    backend.ensure_ready_linear(linear_cfg)
    for _ in range(int(num_instances)):
        obs, _ = eval_env.reset()
        inst = base_env.inst
        if inst is None:
            continue
        start_coeffs = inst.start_coeffs
        target_coeffs = inst.target_coeffs
        gamma = inst.gamma
        start_path = gamma * start_coeffs
        out = backend.jl.track_linear_paths_univar(
            int(base_env.degree),
            start_path,
            target_coeffs,
            compute_newton_iters=bool(base_env.compute_newton_iters),
        )
        success = bool(out.success_flag)
        acc = int(out.total_accepted_steps)
        rej = int(out.total_rejected_steps)
        attempts = int(out.total_step_attempts)
        newton_iters = int(getattr(out, "total_newton_iterations", 0))
        tracking_time_sec = float(getattr(out, "tracking_time_sec", 0.0))
        if success:
            tracking_cost = float(acc + base_env.rho * rej)
        else:
            tracking_cost = float(base_env.failure_penalty)

        successes.append(float(success))
        tracking_costs.append(tracking_cost)
        tracking_time_secs.append(tracking_time_sec)
        total_attempts.append(float(attempts))
        total_newton_iters.append(float(newton_iters))
        accepted_steps.append(float(acc))
        rejected_steps.append(float(rej))
    if not successes:
        return {}
    total_attempts_arr = np.array(total_attempts, dtype=np.float64)
    out = {
        "success_rate": float(np.mean(successes)),
        "tracking_cost_mean": float(np.mean(tracking_costs)),
        "total_step_attempts_mean": float(np.mean(total_attempts)),
        "total_step_attempts_median": float(np.median(total_attempts_arr)),
        "total_step_attempts_min": float(np.min(total_attempts_arr)),
        "total_step_attempts_max": float(np.max(total_attempts_arr)),
        "total_step_attempts_std": float(np.std(total_attempts_arr)),
        "accepted_steps_mean": float(np.mean(accepted_steps)),
        "rejected_steps_mean": float(np.mean(rejected_steps)),
    }
    if tracking_time_secs:
        tarr = np.array(tracking_time_secs, dtype=np.float64)
        out["tracking_time_sec_mean"] = float(np.mean(tracking_time_secs))
        out["tracking_time_sec_median"] = float(np.median(tarr))
        out["tracking_time_sec_std"] = float(np.std(tarr))
        out["tracking_time_sec_min"] = float(np.min(tarr))
        out["tracking_time_sec_max"] = float(np.max(tarr))
    else:
        out["tracking_time_sec_mean"] = 0.0
    if total_newton_iters:
        newton_arr = np.array(total_newton_iters, dtype=np.float64)
        out["total_newton_iterations_mean"] = float(np.mean(total_newton_iters))
        out["total_newton_iterations_median"] = float(np.median(newton_arr))
        out["total_newton_iterations_min"] = float(np.min(newton_arr))
        out["total_newton_iterations_max"] = float(np.max(newton_arr))
        out["total_newton_iterations_std"] = float(np.std(newton_arr))
    if return_per_instance:
        out["total_step_attempts_list"] = list(total_attempts)
        if total_newton_iters:
            out["total_newton_iterations_list"] = list(total_newton_iters)
    return out
