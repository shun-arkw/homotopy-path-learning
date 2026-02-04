#!/bin/bash
# =============================================================================
# PPO TRAINING FOR BEZIER HOMOTOPY UNIVAR ENV
# =============================================================================
# Runs scripts/opt_hc_path/train_cleanrl_ppo.py with configurable arguments.
#
# Usage: bash sh/run_ppo.sh
#
# CONFIGURATION SECTIONS:
# 1. Env / problem parameters     - degree, bezier_degree, episode_len, etc.
# 2. TargetCoeffConfig             - target polynomial coefficient sampling
# 3. PPO parameters                - total_timesteps, num_steps, learning_rate, etc.
# 4. Eval & logging                - eval_interval, eval_num_instances, wandb
# 5. Run training                  - Execute train_cleanrl_ppo.py
# =============================================================================


# =============================================================================
# 1. ENV / PROBLEM PARAMETERS
# =============================================================================
degree=20
bezier_degree=3
latent_dim=$degree
episode_len=8
alpha_z=2.0
failure_penalty=1000000
rho=1.0
seed=0

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
total_timesteps=1000000
num_steps=512
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
eval_num_instances=512
eval_seed=0
track=true
wandb_project_name="BezierHomotopyUnivar-PPO"
wandb_entity=""

# =============================================================================
# 5. RUN TRAINING
# =============================================================================
python3 scripts/opt_hc_path/train_cleanrl_ppo.py \
    --degree "$degree" \
    --bezier-degree "$bezier_degree" \
    --latent-dim "$latent_dim" \
    --episode-len "$episode_len" \
    --alpha-z "$alpha_z" \
    --failure-penalty "$failure_penalty" \
    --rho "$rho" \
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
    $([ "$track" = true ] && echo "--track") \
    $([ -n "$wandb_project_name" ] && echo "--wandb-project-name" "$wandb_project_name") \
    $([ -n "$wandb_entity" ] && echo "--wandb-entity" "$wandb_entity")
