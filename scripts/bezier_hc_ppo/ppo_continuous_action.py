# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime

# Import juliacall before torch to reduce segfault risk.
from juliacall import Main as _jl  # noqa: F401

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from eval_utils import (
    build_fixed_eval_instances,
    run_fixed_eval,
    run_linear_baseline_eval,
)

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str | None = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    save_dir: str = "runs"
    """base directory for run logs and saved model (run subdir is created inside)"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = None
    """the target KL divergence threshold"""
    eval_interval: int = 0
    """evaluate on fixed validation set every this many iterations after update (0 to disable)"""
    eval_num_instances: int = 16
    """number of fixed validation instances"""
    eval_seed: int = 0
    """seed for generating fixed validation instances"""
    eval_linear_baseline: bool = False
    """evaluate linear-path baseline on fixed validation set"""
    eval_zero_action: bool = False
    """evaluate Bezier tracker with action=0 on fixed validation set"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)

        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), None)
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


def _mean_scalar(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return float(np.mean(x))
    return float(x)


if __name__ == "__main__":
    print("RUNNING:", __file__)

    args = tyro.cli(Args)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_tz = os.environ.get("RUN_TZ", "Europe/Paris")
    if ZoneInfo is None:
        logging.warning("zoneinfo is unavailable; using local time for run naming.")
        run_dt = datetime.now()
        run_tz_label = "local"
    else:
        try:
            run_dt = datetime.now(ZoneInfo(run_tz))
            run_tz_label = run_tz
        except Exception:
            logging.warning(f"Invalid RUN_TZ='{run_tz}'. Falling back to local time.")
            run_dt = datetime.now()
            run_tz_label = "local"
    run_name = f"run_{run_dt.strftime('%Y%m%d_%H%M%S')}"
    logging.info(f"run_name={run_name} (timezone={run_tz_label})")

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(os.path.join(args.save_dir, run_name))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if device.type == "cuda":
        device_str = f"{device} ({torch.cuda.get_device_name(0)})"
    else:
        device_str = str(device)
    logging.info(f"device={device_str}")
    writer.add_text("config/device", device_str, 0)

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    eval_env = None
    eval_instances = None
    linear_baseline_done = False
    eval_z0_done = False
    if args.eval_interval > 0 and args.eval_num_instances > 0:
        eval_env = make_env(args.env_id, 0, False, run_name, args.gamma)()
        eval_instances = build_fixed_eval_instances(eval_env, args.eval_num_instances, args.eval_seed)

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            # Log custom env metrics every step (robust to vector env info formats).
            if "success" in infos:
                writer.add_scalar("env/success_mean", _mean_scalar(infos["success"]), global_step)
            if "tracking_cost" in infos:
                writer.add_scalar("env/tracking_cost_mean", _mean_scalar(infos["tracking_cost"]), global_step)
            if "accepted_steps" in infos:
                writer.add_scalar("env/accepted_steps_mean", _mean_scalar(infos["accepted_steps"]), global_step)
            if "rejected_steps" in infos:
                writer.add_scalar("env/rejected_steps_mean", _mean_scalar(infos["rejected_steps"]), global_step)
            if "total_step_attempts" in infos:
                writer.add_scalar("env/total_step_attempts_mean", _mean_scalar(infos["total_step_attempts"]), global_step)
            if "runtime_sec" in infos:
                writer.add_scalar("env/runtime_sec_mean", _mean_scalar(infos["runtime_sec"]), global_step)
            if "tracking_time_sec" in infos:
                writer.add_scalar("env/tracking_time_sec_mean", _mean_scalar(infos["tracking_time_sec"]), global_step)
            if "norm_z" in infos:
                writer.add_scalar("env/norm_z_mean", _mean_scalar(infos["norm_z"]), global_step)

            # Log episode-level statistics if provided via `final_info`.
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if not info:
                        continue

                    # Log episodic return/length from RecordEpisodeStatistics wrapper if present.
                    if "episode" in info:
                        ep = info["episode"]
                        if isinstance(ep, dict) and "r" in ep and "l" in ep:
                            print(f"global_step={global_step}, episodic_return={ep['r']}")
                            writer.add_scalar("charts/episodic_return", ep["r"], global_step)
                            writer.add_scalar("charts/episodic_length", ep["l"], global_step)

                    # Log custom metrics at episode end if present.
                    if "success" in info:
                        writer.add_scalar("charts/success", float(info["success"]), global_step)
                    if "tracking_cost" in info:
                        writer.add_scalar("charts/tracking_cost", float(info["tracking_cost"]), global_step)
                    if "accepted_steps" in info:
                        writer.add_scalar("charts/accepted_steps", float(info["accepted_steps"]), global_step)
                    if "rejected_steps" in info:
                        writer.add_scalar("charts/rejected_steps", float(info["rejected_steps"]), global_step)
                    if "total_step_attempts" in info:
                        writer.add_scalar("charts/total_step_attempts", float(info["total_step_attempts"]), global_step)
                    if "runtime_sec" in info:
                        writer.add_scalar("charts/runtime_sec", float(info["runtime_sec"]), global_step)
                    if "tracking_time_sec" in info:
                        writer.add_scalar("charts/tracking_time_sec", float(info["tracking_time_sec"]), global_step)
                    if "norm_z" in info:
                        writer.add_scalar("charts/norm_z", float(info["norm_z"]), global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # Evaluate on fixed validation set right after this iteration's update (0 = disabled).
        if (
            eval_env is not None
            and args.eval_interval > 0
            and iteration % args.eval_interval == 0
        ):
            if eval_instances is not None:
                eval_env.unwrapped.set_fixed_instances(eval_instances, reset_idx=True)
            eval_metrics = run_fixed_eval(eval_env, agent, device, args.eval_num_instances)
            if eval_metrics:
                writer.add_scalar("eval/success_rate", eval_metrics["success_rate"], global_step)
                writer.add_scalar("eval/tracking_cost_mean", eval_metrics["tracking_cost_mean"], global_step)
                writer.add_scalar("eval/total_step_attempts_mean", eval_metrics["total_step_attempts_mean"], global_step)
                writer.add_scalar("eval/accepted_steps_mean", eval_metrics["accepted_steps_mean"], global_step)
                writer.add_scalar("eval/rejected_steps_mean", eval_metrics["rejected_steps_mean"], global_step)
                logging.info(
                    "eval | "
                    f"global_step={global_step} | "
                    f"success_rate={eval_metrics['success_rate']:.3f} | "
                    f"tracking_cost_mean={eval_metrics['tracking_cost_mean']:.3f} | "
                    f"total_step_attempts_mean={eval_metrics['total_step_attempts_mean']:.1f}"
                )
            if args.eval_zero_action and not eval_z0_done:
                if eval_instances is not None:
                    eval_env.unwrapped.set_fixed_instances(eval_instances, reset_idx=True)
                eval_z0_metrics = run_fixed_eval(
                    eval_env,
                    agent,
                    device,
                    args.eval_num_instances,
                    force_action_zero=True,
                )
                if eval_z0_metrics:
                    writer.add_scalar(
                        "eval_z0/success_rate", eval_z0_metrics["success_rate"], global_step
                    )
                    writer.add_scalar(
                        "eval_z0/tracking_cost_mean",
                        eval_z0_metrics["tracking_cost_mean"],
                        global_step,
                    )
                    writer.add_scalar(
                        "eval_z0/total_step_attempts_mean",
                        eval_z0_metrics["total_step_attempts_mean"],
                        global_step,
                    )
                    writer.add_scalar(
                        "eval_z0/accepted_steps_mean",
                        eval_z0_metrics["accepted_steps_mean"],
                        global_step,
                    )
                    writer.add_scalar(
                        "eval_z0/rejected_steps_mean",
                        eval_z0_metrics["rejected_steps_mean"],
                        global_step,
                    )
                    logging.info(
                        "eval_z0 | "
                        f"global_step={global_step} | "
                        f"success_rate={eval_z0_metrics['success_rate']:.3f} | "
                        f"tracking_cost_mean={eval_z0_metrics['tracking_cost_mean']:.3f} | "
                        f"total_step_attempts_mean={eval_z0_metrics['total_step_attempts_mean']:.1f}"
                    )
                eval_z0_done = True
            if args.eval_linear_baseline and not linear_baseline_done:
                if eval_instances is not None:
                    eval_env.unwrapped.set_fixed_instances(eval_instances, reset_idx=True)
                linear_metrics = run_linear_baseline_eval(eval_env, args.eval_num_instances)
                if linear_metrics:
                    writer.add_scalar("eval_linear/success_rate", linear_metrics["success_rate"], global_step)
                    writer.add_scalar("eval_linear/tracking_cost_mean", linear_metrics["tracking_cost_mean"], global_step)
                    writer.add_scalar(
                        "eval_linear/total_step_attempts_mean",
                        linear_metrics["total_step_attempts_mean"],
                        global_step,
                    )
                    writer.add_scalar("eval_linear/accepted_steps_mean", linear_metrics["accepted_steps_mean"], global_step)
                    writer.add_scalar("eval_linear/rejected_steps_mean", linear_metrics["rejected_steps_mean"], global_step)
                    logging.info(
                        "eval_linear | "
                        f"global_step={global_step} | "
                        f"success_rate={linear_metrics['success_rate']:.3f} | "
                        f"tracking_cost_mean={linear_metrics['tracking_cost_mean']:.3f} | "
                        f"total_step_attempts_mean={linear_metrics['total_step_attempts_mean']:.1f}"
                    )
                linear_baseline_done = True

    if args.save_model:
        run_dir = os.path.join(args.save_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        model_path = os.path.join(run_dir, f"{args.exp_name}.cleanrl_model")
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        # Save full experiment config in nested structure (env, target_coeff, ppo, eval_logging)
        hc_gamma_trick = os.environ.get("BH_GAMMA_TRICK", "1") == "1"
        config = {
            "env": {
                "degree": int(os.environ.get("BH_DEGREE", "10")),
                "bezier_degree": int(os.environ.get("BH_BEZIER_DEGREE", "2")),
                "latent_dim": int(os.environ.get("BH_M", "8")),
                "episode_len": int(os.environ.get("BH_T", "8")),
                "alpha_z": float(os.environ.get("BH_ALPHA_Z", "2.0")),
                "failure_penalty": float(os.environ.get("BH_FAILURE_PENALTY", "1000000")),
                "rho": float(os.environ.get("BH_RHO_REJECT", "1.0")),
                "seed": int(os.environ.get("BH_SEED", "0")),
                "terminal_linear_bonus": os.environ.get("BH_TERMINAL_LINEAR_BONUS", "0") == "1",
                "terminal_linear_bonus_coef": float(os.environ.get("BH_TERMINAL_LINEAR_BONUS_COEF", "1.0")),
                "terminal_z0_bonus": os.environ.get("BH_TERMINAL_Z0_BONUS", "0") == "1",
                "terminal_z0_bonus_coef": float(os.environ.get("BH_TERMINAL_Z0_BONUS_COEF", "1.0")),
                "step_reward_scale": float(os.environ.get("BH_STEP_REWARD_SCALE", "1.0")),
                "require_z0_success": os.environ.get("BH_REQUIRE_Z0_SUCCESS", "0") == "1",
                "z0_max_tries": int(os.environ.get("BH_Z0_MAX_TRIES", "10")),
                "hc_gamma_trick": hc_gamma_trick,
            },
            "target_coeff": {
                "target_dist_real": os.environ.get("BH_TARGET_DIST_REAL", "gaussian"),
                "target_dist_imag": os.environ.get("BH_TARGET_DIST_IMAG", "gaussian"),
                "target_mean_real": float(os.environ.get("BH_TARGET_MEAN_REAL", "0.0")),
                "target_mean_imag": float(os.environ.get("BH_TARGET_MEAN_IMAG", "0.0")),
                "target_std_real": float(os.environ.get("BH_TARGET_STD_REAL", "0.5")),
                "target_std_imag": float(os.environ.get("BH_TARGET_STD_IMAG", "0.5")),
                "target_low_real": float(os.environ.get("BH_TARGET_LOW_REAL", "-0.5")),
                "target_high_real": float(os.environ.get("BH_TARGET_HIGH_REAL", "0.5")),
                "target_low_imag": float(os.environ.get("BH_TARGET_LOW_IMAG", "-0.5")),
                "target_high_imag": float(os.environ.get("BH_TARGET_HIGH_IMAG", "0.5")),
            },
            "ppo": {
                "gamma": args.gamma,
                "total_timesteps": args.total_timesteps,
                "num_steps": args.num_steps,
                "num_envs": args.num_envs,
                "learning_rate": args.learning_rate,
                "update_epochs": args.update_epochs,
                "num_minibatches": args.num_minibatches,
                "gae_lambda": args.gae_lambda,
            },
            "eval_logging": {
                "eval_interval": args.eval_interval,
                "eval_num_instances": args.eval_num_instances,
                "eval_seed": args.eval_seed,
                "eval_linear_baseline": args.eval_linear_baseline,
                "eval_zero_action": args.eval_zero_action,
                "save_model": args.save_model,
                "track": args.track,
                "wandb_project_name": args.wandb_project_name or "",
                "wandb_entity": args.wandb_entity or "",
            },
        }
        config_path = os.path.join(run_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"config saved to {config_path}")

    envs.close()
    writer.close()
