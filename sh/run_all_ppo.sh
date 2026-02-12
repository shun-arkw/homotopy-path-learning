#!/bin/bash
# =============================================================================
# RUN MULTIPLE PPO EXPERIMENTS IN PARALLEL (grid over degree, bezier_degree, hc_beta_omega_p)
# =============================================================================
# Self-contained: does not source or modify run_ppo.sh. Calls train_cleanrl_ppo.py
# directly with the same argument structure.
#
# Usage: bash sh/run_all_ppo.sh
#   Optional: first argument overrides NUM_GPUS. Each run uses one GPU via CUDA_VISIBLE_DEVICES.
#
# Grid (edit arrays below if needed): e.g. degree x bezier_degree x hc_beta_omega_p = 36 runs.
# Each run's stdout/stderr is written to a separate log file so outputs do not mix.
# =============================================================================

# -----------------------------------------------------------------------------
# Per-GPU-server overrides (defaults: NUM_GPUS=1, JULIA_NUM_THREADS=8)
# Edit the next lines when using a different GPU server.
# -----------------------------------------------------------------------------
NUM_GPUS=1
JULIA_THREADS=8
RUN_TZ_DEFAULT="Europe/Paris" # Europe/Paris, Asia/Tokyo, UTC

# Override NUM_GPUS by first argument if given (e.g. bash sh/run_all_ppo.sh 4)
[[ -n "${1:-}" ]] && NUM_GPUS="$1"
if [[ ! "$NUM_GPUS" =~ ^[0-9]+$ ]] || (( NUM_GPUS < 1 )); then
    echo "Usage: bash sh/run_all_ppo.sh (optional first arg: NUM_GPUS >= 1)" >&2
    exit 1
fi

export RUN_TZ="${RUN_TZ:-$RUN_TZ_DEFAULT}"
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-$JULIA_THREADS}"
export PYTHON_JULIACALL_HANDLE_SIGNALS=yes

# -----------------------------------------------------------------------------
# Shared config (fixed for all runs; only degree, bezier_degree, hc_beta_omega_p vary)
# -----------------------------------------------------------------------------
episode_len=1
alpha_z=2.0
failure_penalty=3000
rho=1.0
seed=0
terminal_linear_bonus=true
terminal_linear_bonus_coef=10.0
terminal_z0_bonus=false
terminal_z0_bonus_coef=2.0
step_reward_scale=0.2
require_z0_success=true
z0_max_tries=20
hc_gamma_trick=false

hc_a=0.125
hc_beta_a=1.0
hc_beta_tau=0.85
hc_strict_beta_tau=0.8
hc_min_newton_iters=1
hc_max_steps=50000
hc_max_step_size="inf"
hc_max_initial_step_size="inf"
hc_min_step_size=1e-12
hc_extended_precision=false

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

total_timesteps=1000000 # 1000000
num_steps=2048
num_envs=1
learning_rate=0.0003
update_epochs=10
num_minibatches=32
gamma=0.99
gae_lambda=0.95

eval_interval=10
eval_num_instances=1024
eval_seed=0
eval_linear_baseline=true
eval_zero_action=true
save_model=true
track=false
wandb_project_name="BezierHomotopyUnivar-PPO"
wandb_entity=""

result_root="results/bezier_ppo/univar"

# -----------------------------------------------------------------------------
# Grid to sweep (degree x bezier_degree x hc_beta_omega_p)
# -----------------------------------------------------------------------------
degrees=(5 10) # (5 10 20 30 40 50)
bezier_degrees=(2) # (2 3)
omega_ps=(0.8) # (0.8 1.0)

# Per-job log directory (stdout/stderr from each parallel run; train.log remains inside each run's save_dir)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs_run_all"
mkdir -p "$LOG_DIR"

# Run a single PPO experiment (degree, bezier_degree, hc_beta_omega_p, gpu_id, log_path)
run_one_ppo() {
    local degree="$1"
    local bezier_degree="$2"
    local hc_beta_omega_p="$3"
    local gpu_id="$4"
    local log_path="$5"
    local latent_dim="$degree"
    local setting_tag="degree${degree}_bezier${bezier_degree}_ep${episode_len}"
    local hc_tracking_tag="omega${hc_beta_omega_p}_tau${hc_beta_tau}_strict${hc_strict_beta_tau}"
    local save_dir="${result_root}/${setting_tag}/${hc_tracking_tag}"
    mkdir -p "$save_dir"
    CUDA_VISIBLE_DEVICES="$gpu_id" python3 scripts/bezier_hc_ppo/train_cleanrl_ppo.py \
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
        --hc-a "$hc_a" \
        --hc-beta-a "$hc_beta_a" \
        --hc-beta-omega-p "$hc_beta_omega_p" \
        --hc-beta-tau "$hc_beta_tau" \
        --hc-strict-beta-tau "$hc_strict_beta_tau" \
        --hc-min-newton-iters "$hc_min_newton_iters" \
        --hc-max-steps "$hc_max_steps" \
        --hc-max-step-size "$hc_max_step_size" \
        --hc-max-initial-step-size "$hc_max_initial_step_size" \
        --hc-min-step-size "$hc_min_step_size" \
        $([ "$hc_extended_precision" = true ] && echo "--hc-extended-precision") \
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
        $([ -n "$wandb_entity" ] && echo "--wandb-entity" "$wandb_entity") \
        >> "$log_path" 2>&1
}

# Build list of (degree, bezier_degree, omega_p) triples
triples=()
for d in "${degrees[@]}"; do
    for b in "${bezier_degrees[@]}"; do
        for o in "${omega_ps[@]}"; do
            triples+=("$d $b $o")
        done
    done
done

total="${#triples[@]}"
echo "run_all_ppo: ${total} experiments, max ${NUM_GPUS} concurrent GPU(s). Logs: ${LOG_DIR}"
started=0

for triple in "${triples[@]}"; do
    while (( $(jobs -r 2>/dev/null | wc -l) >= NUM_GPUS )); do
        sleep 2
    done
    read -r d b o <<< "$triple"
    gpu_id=$((started % NUM_GPUS))
    # Log filename: d5_b2_o0.6_gpu0.log (safe, no spaces)
    log_name="d${d}_b${b}_o${o}_gpu${gpu_id}.log"
    log_path="${LOG_DIR}/${log_name}"
    started=$((started + 1))
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${started}/${total}: degree=${d} bezier_degree=${b} hc_beta_omega_p=${o} GPU=${gpu_id} -> ${log_name}"
    run_one_ppo "$d" "$b" "$o" "$gpu_id" "$log_path" &
done

wait
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All ${total} runs finished."
