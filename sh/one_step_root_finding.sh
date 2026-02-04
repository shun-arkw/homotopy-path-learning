#!/bin/bash
# =============================================================================
# ONE-STEP POLYNOMIAL ROOT-FINDING ALGORITHM
# =============================================================================
# This script implements a one-time step polynomial root-finding algorithm 
# based on machine learning for quadratic polynomials x^2 + a*x + b.
#
# Usage: bash sh/one_step_root_finding.sh
#
# CONFIGURATION SECTIONS:
# 1.  Dataset Configuration            - Parameters to compose dataset paths
# 2.  Model and Training Configuration - Architecture, training, and evaluation settings
# 3.  Output & Logging Configuration   - Output directory and logging settings
# 4.  Optional Parameters              - Advanced options and flags
# 5.  Run Training                     - Execute the training script
# =============================================================================

# =============================================================================
# 1. DATASET CONFIGURATION
# =============================================================================
# Parameters used to construct dataset directories and file paths
max_coeff_abs=10.0               # domain size: domain_<max_coeff_abs>
discriminant_margin=3.0          # margin: margin_<discriminant_margin>
num_hc_steps=20                  # steps_<num_hc_steps>
min_coeff_distance=5.5           # distance_<min>_to_<max>
max_coeff_distance=6.0           # distance_<min>_to_<max>
solver_free=true                 # true -> solver_free, false -> solver_used
train_size=100000                # number of training samples (train_<train_size>)
test_size=10000                  # number of test samples (test_<test_size>)

# =============================================================================
# 2. MODEL AND TRAINING CONFIGURATION
# =============================================================================
# Neural network architecture parameters
hidden=128 # number of hidden units per layer
depth=6 # number of hidden layers

# Training hyperparameters
epochs=512 # number of training epochs
batch_size=512 # batch size for training
lr=0.001 # learning rate
lr_scheduler="cosine" # learning rate scheduler: cosine or linear
seed=42 # random seed for reproducibility
device="cuda" # device to use: "cuda" or "cpu"

# Evaluation parameters
# Support multiple thresholds like: (0.01 0.001 0.0001 0.00001)
thresholds=(0.01 0.001 0.0001 0.00001)

# =============================================================================
# 3. OUTPUT & LOGGING CONFIGURATION
# =============================================================================
# Compose dataset and result paths from parameters
solver_dir=$([ "$solver_free" = true ] && echo "solver_free" || echo "solver_used")
dataset_dir="dataset/domain_${max_coeff_abs}_margin_${discriminant_margin}/steps_${num_hc_steps}/distance_${min_coeff_distance}_to_${max_coeff_distance}/train_${train_size}_test_${test_size}/${solver_dir}"
train_dataset_path="${dataset_dir}/hc_pairs_train.txt"
test_dataset_path="${dataset_dir}/hc_pairs_test.txt"

# Directory to save model artifacts and logs
save_path="results/one_step_root_finding/domain_${max_coeff_abs}_margin_${discriminant_margin}/steps_${num_hc_steps}/distance_${min_coeff_distance}_to_${max_coeff_distance}/train_${train_size}_test_${test_size}/${solver_dir}"

# Logging controls
epoch_log_interval=64 # log every N epochs (0 to disable)
timezone="Europe/Paris" # timezone used for timestamping logs

# =============================================================================
# 4. OPTIONAL PARAMETERS
# =============================================================================
# Advanced options (keep false unless you know you need it)
no_filter_disc=false # if true, disables discriminant filtering inside the training script

# =============================================================================
# 5. RUN TRAINING
# =============================================================================
# Execute training with the configured parameters

python3 scripts/task1/one_step_root_finding.py \
    --train-dataset-path "$train_dataset_path" \
    --test-dataset-path "$test_dataset_path" \
    --train-size "$train_size" \
    --test-size "$test_size" \
    --save-path "$save_path" \
    --hidden "$hidden" \
    --depth "$depth" \
    --epochs "$epochs" \
    --batch-size "$batch_size" \
    --lr "$lr" \
    --seed "$seed" \
    --device "$device" \
    --threshold "${thresholds[@]}" \
    --epoch-log-interval "$epoch_log_interval" \
    --timezone "$timezone" \
    --lr-scheduler "$lr_scheduler" \
    $([ "$no_filter_disc" = true ] && echo "--no-filter-disc")
