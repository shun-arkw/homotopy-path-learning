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
