# Multivariate Pham Experiments

This directory contains experiment-specific CLIs and YAML configuration for the
multivariate Pham implementation. Reusable PPO, evaluation, and checkpoint code
lives under `src/homotopy_path_learning/`.

## EXP-0004 Smoke

Run the Phase 6 smoke pipeline inside the reference Docker container:

```bash
export PYTHON_JULIACALL_EXE="$(command -v julia)"
export PYTHON_JULIACALL_PROJECT="$PWD/julia"

RUN_ID="phase6-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="outputs/EXP-0004/${RUN_ID}"

python3 experiments/multivariate_pham/train.py \
  --config experiments/multivariate_pham/configs/exp-0004-smoke.yaml \
  --run-id "$RUN_ID" \
  --device cpu

python3 experiments/multivariate_pham/evaluate.py \
  --run-dir "$RUN_DIR" \
  --checkpoint "$RUN_DIR/checkpoints/last.pt"

python3 experiments/multivariate_pham/benchmark.py \
  --run-dir "$RUN_DIR" \
  --checkpoint "$RUN_DIR/checkpoints/last.pt"

python3 experiments/multivariate_pham/analyze.py \
  --run-dir "$RUN_DIR"
```

The smoke run verifies that PPO rollout collection, updates, checkpointing,
deterministic evaluation, benchmark CSV generation, and Markdown analysis work
end to end. It is not intended to establish a performance conclusion.

## EXP-0008 Phase 7 Performance

Run baseline and optimized performance measurements inside the reference Docker
container:

```bash
export PYTHON_JULIACALL_EXE="$(command -v julia)"
export PYTHON_JULIACALL_PROJECT="$PWD/julia"

BASELINE_ID="phase7-baseline-$(date -u +%Y%m%dT%H%M%SZ)"
BASELINE_DIR="outputs/EXP-0008/${BASELINE_ID}"

python3 experiments/multivariate_pham/profile.py \
  --config experiments/multivariate_pham/configs/exp-0008-phase7-performance.yaml \
  --output-dir "$BASELINE_DIR" \
  --mode baseline

OPTIMIZED_ID="phase7-optimized-$(date -u +%Y%m%dT%H%M%SZ)"
OPTIMIZED_DIR="outputs/EXP-0008/${OPTIMIZED_ID}"

python3 experiments/multivariate_pham/profile.py \
  --config experiments/multivariate_pham/configs/exp-0008-phase7-performance.yaml \
  --output-dir "$OPTIMIZED_DIR" \
  --mode optimized \
  --baseline "$BASELINE_DIR/performance.json"
```

The profiler writes `performance.json`, `performance.csv`,
`equivalence.json`, `profiling/python.txt`, `profiling/julia.txt`, and
`summary.md`. The optimized Phase 7 path reuses one linear tracking result per
target seed during fixed evaluation; Tracker reuse, batch API, and threaded mode
are not enabled by default.
