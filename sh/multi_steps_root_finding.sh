#!/bin/bash
# =============================================================================
# MULTI-STEPS POLYNOMIAL ROOT-FINDING ALGORITHM
# =============================================================================
# This script implements a multi-steps polynomial root-finding algorithm 
# based on machine learning along a fixed homotopy path for quadratic polynomials x^2 + a*x + b.
#
# Usage: bash sh/multi_steps_root_finding.sh
#
# CONFIGURATION SECTIONS:
# 1.  Dataset Configuration           - Parameters to compose dataset paths
# 2.  Multi-Steps Configuration       - Homotopy path and evaluation settings
# 3.  Output & Logging Configuration  - Output directory and logging settings
# 4.  Optional Parameters             - Advanced options and flags
# 5.  Run Multi-Steps                 - Execute the multi-steps script
# =============================================================================

# =============================================================================
# 1. DATASET CONFIGURATION
# =============================================================================
# Parameters used to construct dataset directories and file paths (shared with one-step)
max_coeff_abs=10.0               # domain size: domain_<max_coeff_abs>
discriminant_margin=3.0          # margin: margin_<discriminant_margin>
num_hc_steps=20                  # steps_<num_hc_steps>
min_coeff_distance=5.5           # distance_<min>_to_<max>
max_coeff_distance=6.0           # distance_<min>_to_<max>
solver_free=true                 # true -> solver_free, false -> solver_used

# Dataset size parameters (used in directory naming)
# Use separate sizes for dataset (endpoints) and model artifacts
data_train_size=0                # dataset train size used in directory name (can be 0)
data_test_size=20000             # dataset test size used in directory name

# Pretrained model was trained on potentially different sizes
model_train_size=100000          # one-step training samples used for model
model_test_size=10000            # one-step test samples used for model

# Build dataset directory and file paths (this script uses the TEST endpoints file)
solver_dir=$([ "$solver_free" = true ] && echo "solver_free" || echo "solver_used")
dataset_dir="dataset/domain_${max_coeff_abs}_margin_${discriminant_margin}/steps_${num_hc_steps}/distance_${min_coeff_distance}_to_${max_coeff_distance}/train_${data_train_size}_test_${data_test_size}/${solver_dir}"
dataset_path="${dataset_dir}/hc_endpoints_test.txt"

# Pre-trained model artifacts paths from one-step training (sizes may differ from dataset sizes)
model_base_dir="results/one_step_root_finding/domain_${max_coeff_abs}_margin_${discriminant_margin}/steps_${num_hc_steps}/distance_${min_coeff_distance}_to_${max_coeff_distance}/train_${model_train_size}_test_${model_test_size}/${solver_dir}"
model_path="${model_base_dir}/model.pt"
scaler_path="${model_base_dir}/scaler.pt"

# =============================================================================
# 2. MULTI-STEPS CONFIGURATION
# =============================================================================
# Parameters for the multi-steps homotopy continuation algorithm

# Homotopy path parameters
num_steps=$num_hc_steps # number of steps for the homotopy path
device="cuda" # device to use: "cuda" or "cpu"

# Evaluation parameters
thresholds=(0.01 0.001 0.0001 0.00001) # RMSE accuracy thresholds (can specify multiple)
max_samples=1000 # set e.g. 10000 to limit processed samples (empty -> all)

# Processing options
skip_if_nonreal=true # skip samples if intermediate discriminant <= 0
save_trajectories=true # save per-sample trajectories
compare_julia=false # compare with Julia HomotopyContinuation library
debug=false # enable debug logging

# =============================================================================
# 3. OUTPUT & LOGGING CONFIGURATION
# =============================================================================
# Directory to save results and logging settings

# Output directory (record both dataset sizes and model sizes for traceability)
save_path="results/multi_steps_root_finding/domain_${max_coeff_abs}_margin_${discriminant_margin}/steps_${num_hc_steps}/distance_${min_coeff_distance}_to_${max_coeff_distance}/data_train_${data_train_size}_test_${data_test_size}__model_train_${model_train_size}_test_${model_test_size}/${solver_dir}"

# Logging parameters
timezone="Europe/Paris" # timezone for logging timestamps

# =============================================================================
# 3.5. PROCESSING CONFIGURATION
# =============================================================================
# Parallel processing and performance settings (dataset_generation.sh style)
# These will be passed to Python via CLI args

# Number of parallel jobs for Julia HC (use small numbers like 2-4; each process initializes Julia)
n_jobs=-1               # 1 for sequential; set to 2-4 for parallel
# Joblib backend and preference (advanced)
backend="multiprocessing"        # loky|threading
prefer="processes"    # processes|threads
verbose=1           # 0=silent

# =============================================================================
# 4. OPTIONAL PARAMETERS
# =============================================================================
# Advanced options and conditional argument assembly

# Optional args assembly
optional_args=()
if [ ${#thresholds[@]} -gt 0 ]; then
    optional_args+=(--threshold "${thresholds[@]}")
fi
if [ -n "$max_samples" ]; then
    optional_args+=(--max-samples "$max_samples")
fi
if [ "$skip_if_nonreal" = true ]; then
    optional_args+=(--skip-if-nonreal)
fi
if [ "$save_trajectories" = true ]; then
    optional_args+=(--save-trajectories)
fi
if [ "$compare_julia" = true ]; then
    optional_args+=(--compare-julia)
fi
if [ "$debug" = true ]; then
    optional_args+=(--debug)
fi

# =============================================================================
# 5. RUN MULTI-STEPS
# =============================================================================
# Run the multi-steps root finding script with all configured parameters

python3 scripts/task1/multi_steps_root_finding.py \
    --dataset-path "$dataset_path" \
    --model-path "$model_path" \
    --scaler-path "$scaler_path" \
    --save-path "$save_path" \
    --num-steps "$num_steps" \
    --device "$device" \
    --timezone "$timezone" \
    --julia-n-jobs "$n_jobs" \
    --julia-backend "$backend" \
    --julia-prefer "$prefer" \
    --julia-verbose "$verbose" \
    "${optional_args[@]}"