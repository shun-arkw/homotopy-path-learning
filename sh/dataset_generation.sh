#!/bin/bash
# =============================================================================
# DATASET GENERATION SCRIPT
# =============================================================================
# This script generates Task 1.1 datasets using linear homotopy in the (a, b) 
# coefficient space for quadratic polynomials x^2 + a*x + b.
#
# Usage: bash sh/dataset_generation.sh
#
# CONFIGURATION SECTIONS:
# 1.  Domain Configuration           - Coefficient domain Ω parameters
# 2.  Sampling Configuration         - Homotopy path and sampling parameters
# 3.  Output Configuration           - Output directory and file settings
# 4.  Processing Configuration       - Parallel processing and performance settings
# 5.  Optional Parameters            - Advanced options and flags
# 6.  Execute Dataset Generation     - Run the dataset generation with all parameters
# =============================================================================

# =============================================================================
# 1. DOMAIN CONFIGURATION
# =============================================================================
# Parameters for the coefficient domain Ω = { (a, b) ∈ [-B, B]^2 | b < a^2/4 - τ/4 }

# Domain bounds
max_coeff_abs=10.0 # B: coefficients constrained to [-B, B]
discriminant_margin=3.0 # τ: margin away from D=0 (ensures D = a^2 - 4b >= τ > 0)

# =============================================================================
# 2. SAMPLING CONFIGURATION
# =============================================================================
# Parameters for homotopy path sampling and generation

# Homotopy path parameters
num_hc_steps=5 # number of discretization steps per path (j = 0..num_steps)

# Distance constraints
min_coeff_distance=5.5 # minimum distance between coeffs_start and coeffs_end
max_coeff_distance=6.0 # maximum distance between coeffs_start and coeffs_end

# Path validation
n_check=32 # number of grid points used to verify the entire segment stays in Ω

# Random seed
seed=42 # random seed for reproducibility

# Solver-free mode
solver_free=true # reconstruct coefficients from analytic roots for all path points

# Train/test split (required for machine learning)
train_size=0 # desired number of training pairs (required)
test_size=5000 # desired number of test pairs (required)

# =============================================================================
# 3. OUTPUT CONFIGURATION
# =============================================================================
# Directory and file settings for saving datasets and visualizations

# Output directory structure
output_base_dir="dataset"
domain_config="domain_${max_coeff_abs}_margin_${discriminant_margin}"
path_config="steps_${num_hc_steps}"
distance_config="distance_${min_coeff_distance}_to_${max_coeff_distance}"
split_config="train_${train_size}_test_${test_size}"
solver_config="solver_$(if [ "$solver_free" = true ]; then echo "free"; else echo "dependent"; fi)"

# Full output directory path
output_dir="${output_base_dir}/${domain_config}/${path_config}/${distance_config}/${split_config}/${solver_config}"

# File saving options
save_hc_paths=false # also save full HC paths as jsonl
save_hc_endpoints=true # also save HC endpoints as txt and jsonl

# =============================================================================
# 4. PROCESSING CONFIGURATION
# =============================================================================
# Parallel processing and performance settings

# Parallel processing
n_jobs=-1 # number of parallel jobs (-1 for all available cores)
backend="multiprocessing" # backend for parallel processing: "multiprocessing" or "threading"
verbose=0 # verbosity level (0=silent, 1=progress, 2=debug)

# =============================================================================
# 5. OPTIONAL PARAMETERS
# =============================================================================
# Advanced options and conditional argument assembly

# Optional args assembly
optional_args=()
if [ "$solver_free" = true ]; then
    optional_args+=(--solver-free)
else
    optional_args+=(--no-solver-free)
fi
if [ "$save_hc_paths" = true ]; then
    optional_args+=(--save-hc-paths)
fi
if [ "$save_hc_endpoints" = true ]; then
    optional_args+=(--save-hc-endpoints)
fi

# =============================================================================
# 6. EXECUTE DATASET GENERATION
# =============================================================================
# Run the dataset generation script with all configured parameters

echo "Generating dataset with the following configuration:"
echo "  Domain: max_coeff_abs=${max_coeff_abs}, discriminant_margin=${discriminant_margin}"
echo "  Paths: num_hc_steps=${num_hc_steps}"
echo "  Distance: min=${min_coeff_distance}, max=${max_coeff_distance}"
echo "  Train/Test: train_size=${train_size}, test_size=${test_size}"
echo "  Output: ${output_dir}"
echo "  Solver-free: ${solver_free}"
echo "  Save HC paths: ${save_hc_paths}"
echo "  Save HC endpoints: ${save_hc_endpoints}"
echo ""

python3 scripts/task1/dataset_generation/dataset_generator.py \
    --output-dir "$output_dir" \
    --num-hc-steps "$num_hc_steps" \
    --min-dist "$min_coeff_distance" \
    --max-dist "$max_coeff_distance" \
    --n-check "$n_check" \
    --seed "$seed" \
    --train-size "$train_size" \
    --test-size "$test_size" \
    --n-jobs "$n_jobs" \
    --backend "$backend" \
    --verbose "$verbose" \
    "${optional_args[@]}"
