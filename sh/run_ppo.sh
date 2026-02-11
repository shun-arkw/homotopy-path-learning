#!/bin/bash
# =============================================================================
# PPO TRAINING FOR BEZIER HOMOTOPY UNIVAR ENV
# =============================================================================
# Runs scripts/bezier_hc_ppo/train_cleanrl_ppo.py with configurable arguments.
#
# Usage: bash sh/run_ppo.sh
#
# CONFIGURATION SECTIONS:
# 1. Env / problem parameters     - degree, bezier_degree, episode_len, etc.
# 2. TargetCoeffConfig             - target polynomial coefficient sampling
# 3. PPO parameters                - total_timesteps, num_steps, learning_rate, etc.
# 4. Eval & logging                - eval_interval, eval_num_instances, wandb
# 5. Result directory / timezone   - output layout and run timestamp timezone
# 6. Run training                  - Execute train_cleanrl_ppo.py
# =============================================================================


# =============================================================================
# 1. ENV / PROBLEM PARAMETERS
# =============================================================================
degree=30
bezier_degree=3
latent_dim=$degree
episode_len=1
alpha_z=2.0
failure_penalty=3000
rho=1.0
seed=0
terminal_linear_bonus=true
terminal_linear_bonus_coef=10.0 # 10.0, 20.0, 30,0
terminal_z0_bonus=false # true
terminal_z0_bonus_coef=2.0
step_reward_scale=0.2
require_z0_success=true
z0_max_tries=20
hc_gamma_trick=false

# =============================================================================
# 2. TARGET COEFF CONFIG (sampling of target polynomial coefficients)
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
# 3. PPO PARAMETERS
# =============================================================================
total_timesteps=10000000 # 1000000
num_steps=2048
num_envs=1
learning_rate=0.0003
update_epochs=10
num_minibatches=32
gamma=0.99
gae_lambda=0.95

# =============================================================================
# 4. EVAL & LOGGING
# =============================================================================
eval_interval=10
eval_num_instances=1024
eval_seed=0
eval_linear_baseline=true
eval_zero_action=true
save_model=true
track=false
wandb_project_name="BezierHomotopyUnivar-PPO"
wandb_entity=""

# =============================================================================
# 5. RESULT DIRECTORY / TIMEZONE
# =============================================================================
# Timezone used by ppo_continuous_action.py to generate run names.
# Examples: Europe/Paris, Asia/Tokyo, UTC
run_tz="${RUN_TZ:-Europe/Paris}"
export RUN_TZ="$run_tz"

# Save runs under: result/degree{d}_bezier{b}_ep{t}/run_YYYYMMDD_HHMMSS
result_root="results/bezier_ppo/univar"
setting_tag="degree${degree}_bezier${bezier_degree}_ep${episode_len}"
save_dir="${result_root}/${setting_tag}"
mkdir -p "$save_dir"

# =============================================================================
# 6. RUN TRAINING
# =============================================================================
python3 scripts/bezier_hc_ppo/train_cleanrl_ppo.py \
    --degree "$degree" \
    --bezier-degree "$bezier_degree" \
    --latent-dim "$latent_dim" \
    --episode-len "$episode_len" \
    --alpha-z "$alpha_z" \
    --failure-penalty "$failure_penalty" \
    --rho "$rho" \
    $([ "$terminal_linear_bonus" = true ] && echo "--terminal-linear-bonus") \
    --terminal-linear-bonus-coef "$terminal_linear_bonus_coef" \
    $([ "$terminal_z0_bonus" = true ] && echo "--terminal-z0-bonus") \
    --terminal-z0-bonus-coef "$terminal_z0_bonus_coef" \
    --step-reward-scale "$step_reward_scale" \
    $([ "$require_z0_success" = true ] && echo "--require-z0-success") \
    --z0-max-tries "$z0_max_tries" \
    $([ "$hc_gamma_trick" = true ] && echo "--hc-gamma-trick") \
    --seed "$seed" \
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
    --total-timesteps "$total_timesteps" \
    --num-steps "$num_steps" \
    --num-envs "$num_envs" \
    --learning-rate "$learning_rate" \
    --update-epochs "$update_epochs" \
    --num-minibatches "$num_minibatches" \
    --gamma "$gamma" \
    --gae-lambda "$gae_lambda" \
    --eval-interval "$eval_interval" \
    --eval-num-instances "$eval_num_instances" \
    --eval-seed "$eval_seed" \
    $([ "$eval_linear_baseline" = true ] && echo "--eval-linear-baseline") \
    $([ "$eval_zero_action" = true ] && echo "--eval-zero-action") \
    $([ "$save_model" = true ] && echo "--save-model") \
    --save-dir "$save_dir" \
    $([ "$track" = true ] && echo "--track") \
    $([ -n "$wandb_project_name" ] && echo "--wandb-project-name" "$wandb_project_name") \
    $([ -n "$wandb_entity" ] && echo "--wandb-entity" "$wandb_entity")
