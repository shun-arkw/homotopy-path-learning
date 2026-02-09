#!/bin/bash
# =============================================================================
# EVAL SAVED PPO MODEL (BEZIER VS LINEAR SINGLE-GAMMA)
# =============================================================================
# Runs scripts/bezier_hc_ppo/eval_saved_model.py with configurable arguments.
# Keep section 1 and 2 in sync with sh/run_ppo.sh so eval uses the same env.
#
# Usage:
#   bash sh/eval_model.sh
#   bash sh/eval_model.sh /path/to/ppo_continuous_action.cleanrl_model
#
# CONFIGURATION SECTIONS:
# 1. Model path              - path to .cleanrl_model (or pass as $1)
# 2. Env / problem parameters - degree, bezier_degree, episode_len, etc. (match run_ppo.sh)
# 3. TargetCoeffConfig        - target polynomial coefficient sampling (match run_ppo.sh)
# 4. Eval options             - num_instances, eval_seed, save_results, device
# 5. Run evaluation           - Execute eval_saved_model.py
# =============================================================================


# =============================================================================
# 1. MODEL PATH
# =============================================================================
# Set to your saved model, or pass as first argument: bash sh/eval_model.sh RUNS/.../model.cleanrl_model
model_path="runs/hc_envs.register_env:BezierHomotopyUnivar-v0__ppo_continuous_action__0__1770260251/ppo_continuous_action.cleanrl_model"

# =============================================================================
# 2. ENV / PROBLEM PARAMETERS (match run_ppo.sh section 1)
# =============================================================================
degree=10
bezier_degree=3
latent_dim=$degree
episode_len=1
alpha_z=2.0
failure_penalty=3000
rho=1.0
seed=0
terminal_linear_bonus=false
terminal_linear_bonus_coef=3.0
terminal_z0_bonus=true
terminal_z0_bonus_coef=2.0
terminal_z0_bonus_scale=25.0
step_reward_scale=0.2
require_z0_success=true
z0_max_tries=20

# =============================================================================
# 3. TARGET COEFF CONFIG (match run_ppo.sh section 2)
# =============================================================================
target_dist_real="uniform"
target_dist_imag="uniform"
target_mean_real=0.0
target_mean_imag=0.0
target_std_real=0.5
target_std_imag=0.5
target_low_real=-5
target_high_real=5
target_low_imag=-5
target_high_imag=5

# =============================================================================
# 4. EVAL OPTIONS
# =============================================================================
num_instances=1024
eval_seed=123
device="cpu"
# Optional: write JSON results to this path (leave empty to skip)
save_results=""

# =============================================================================
# 5. RUN EVALUATION
# =============================================================================
python3 scripts/bezier_hc_ppo/eval_saved_model.py \
    --model-path "$model_path" \
    --num-instances "$num_instances" \
    --eval-seed "$eval_seed" \
    --device "$device" \
    --degree "$degree" \
    --bezier-degree "$bezier_degree" \
    --latent-dim "$latent_dim" \
    --episode-len "$episode_len" \
    --alpha-z "$alpha_z" \
    --failure-penalty "$failure_penalty" \
    --rho "$rho" \
    --seed "$seed" \
    $([ "$terminal_linear_bonus" = true ] && echo "--terminal-linear-bonus") \
    --terminal-linear-bonus-coef "$terminal_linear_bonus_coef" \
    $([ "$terminal_z0_bonus" = true ] && echo "--terminal-z0-bonus") \
    --terminal-z0-bonus-coef "$terminal_z0_bonus_coef" \
    --terminal-z0-bonus-scale "$terminal_z0_bonus_scale" \
    --step-reward-scale "$step_reward_scale" \
    $([ "$require_z0_success" = true ] && echo "--require-z0-success") \
    --z0-max-tries "$z0_max_tries" \
    --target-dist-real "$target_dist_real" \
    --target-dist-imag "$target_dist_imag" \
    --target-mean-real "$target_mean_real" \
    --target-mean-imag "$target_mean_imag" \
    --target-std-real "$target_std_real" \
    --target-std-imag "$target_std_imag" \
    --target-low-real "$target_low_real" \
    --target-high-real "$target_high_real" \
    --target-low-imag "$target_low_imag" \
    --target-high-imag "$target_high_imag" \
    $([ -n "$save_results" ] && echo "--save-results" "$save_results")
