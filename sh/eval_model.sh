#!/bin/bash
# =============================================================================
# EVAL SAVED PPO MODEL (BEZIER VS LINEAR SINGLE-GAMMA)
# =============================================================================
# Runs scripts/bezier_hc_ppo/eval_saved_model.py with configurable arguments.
# The evaluator automatically loads config.json from the model directory, so
# training/evaluation environment settings stay consistent.
#
# Usage:
#   bash sh/eval_model.sh
#
# Configure input_path, num_instances, device, etc. in the script below.
# =============================================================================

set -euo pipefail

# =============================================================================
# 1. CONFIGURATION (edit these)
# =============================================================================
# Experiment setting (must match the run you want to evaluate; same as run_ppo.sh)
degree=10
bezier_degree=3
episode_len=1

# Run subdir: run_YYYYMMDD_HHMMSS (same format as run_ppo.sh output)
run_date="20260210"
run_time="182424"

# Base dir for results (match run_ppo.sh result_root)
result_root="results/bezier_ppo/univar"
setting_tag="degree${degree}_bezier${bezier_degree}_ep${episode_len}"
run_subdir="run_${run_date}_${run_time}"
input_path="${result_root}/${setting_tag}/${run_subdir}"

num_instances=1024
eval_seed=1234
top_k=10
worst_k=10
device="cpu"
# Newton iteration counting: true=count (slower, reports total_newton_iterations_*), false=do not (faster)
compute_newton_iters=false
# Leave empty to save to <run_dir>/eval_results.json
save_results=""

# =============================================================================
# 2. RESOLVE PATHS
# =============================================================================

if [[ -d "$input_path" ]]; then
    run_dir="${input_path%/}"
    model_path="$run_dir/ppo_continuous_action.cleanrl_model"
elif [[ -f "$input_path" ]]; then
    model_path="$input_path"
    run_dir="$(dirname "$model_path")"
else
    echo "Error: input path not found: $input_path"
    exit 1
fi

if [[ ! -f "$model_path" ]]; then
    echo "Error: model file not found: $model_path"
    exit 1
fi

config_path="$run_dir/config.json"
if [[ -f "$config_path" ]]; then
    echo "Using run config: $config_path"
else
    echo "Warning: config.json not found in run directory."
    echo "Evaluation will fall back to eval_saved_model.py defaults/CLI options."
fi

if [[ -z "$save_results" ]]; then
    save_results="$run_dir/eval_results.json"
fi

# =============================================================================
# 3. RUN EVALUATION
# =============================================================================
python3 scripts/bezier_hc_ppo/eval_saved_model.py \
    --model-path "$model_path" \
    --num-instances "$num_instances" \
    --eval-seed "$eval_seed" \
    --top-k "$top_k" \
    --worst-k "$worst_k" \
    --device "$device" \
    --compute-newton-iters "$compute_newton_iters" \
    --save-results "$save_results"
