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

## Oracle Headroom (EXP-0009)

`oracle.py` does not train a policy. For each fixed evaluation instance of a
PPO config (`evaluation.seed + i`), it searches the latent action box directly
(zero action, uniform random samples, then a (1+1) evolution strategy) and
keeps the cheapest Bezier path found. The result estimates how much a policy
could improve over the linear path at most. Because the search is finite, it
is a lower bound on the attainable improvement, not the true optimum.

```bash
python3 experiments/multivariate_pham/oracle.py \
  --config experiments/multivariate_pham/configs/exp-0007-n3-d222-seed0.yaml \
  --experiment-id EXP-0009 \
  --run-id "exp-0009-n3-d222-m4-seed0-$(date -u +%Y%m%dT%H%M%SZ)" \
  --budget 500 --n-random 100 \
  --compare-run-dir outputs/EXP-0007/exp-0007-n3-d222-seed0-20260927T145848Z
```

`--latent-dim full` uses a square basis (`m = 2 * n_free_coeffs`) that spans
every real direction of the free coefficients. The run writes `config.yaml`
(with `experiment.id` and `path.latent_dim` overridden), `oracle.csv`,
`oracle_summary.csv`, `oracle_actions.npy`, `oracle_cost_traces.npy`,
`latent_basis.npy`, and `metadata.json` to `outputs/<experiment-id>/<run-id>/`.
Each `--compare-run-dir` adds a `LearnedBezier[<run>]` row to the summary.
That run's `evaluation/evaluation.csv` must contain the same target seeds and
the same linear costs.

## Learning from Oracle Labels (EXP-0010)

EXP-0010 trains a policy that needs one forward pass at inference time, using
per-instance oracle actions as labels (behavior cloning), and adds diagnostics.

1. Generate labels on training instances disjoint from the evaluation seeds.
   `--evaluation-seed` replaces `evaluation.seed` as the base of the instance
   seeds; the run also stores `oracle_observations.npy` (`(N, obs_dim)`).

   ```bash
   python3 experiments/multivariate_pham/oracle.py --config <config.yaml> \
     --experiment-id EXP-0010 --run-id labels-a --evaluation-seed 900000 \
     --num-instances 1000 --budget 500 --n-random 100
   ```

2. Constant-action baseline: search one action shared by all problems on
   training targets `train-seed + i`, then evaluate it on the evaluation
   instances (reported as `ConstantBezier`). `--oracle-run-dir` adds a
   comparison summary (`constant_summary.csv`) against a per-instance oracle run.

   ```bash
   python3 experiments/multivariate_pham/oracle.py --config <config.yaml> --mode constant \
     --experiment-id EXP-0010 --run-id constant --train-seed 900000 --num-train 200 \
     --oracle-run-dir <oracle-run-on-eval-seeds>
   ```

3. Behavior cloning: fits `actor_mean` to the labels (MSE on the tanh-bounded
   action, best validation epoch kept), writes a normal checkpoint, evaluates
   it on the config's evaluation seeds, and writes `bc_summary.csv` when
   `--oracle-run-dir` is given. `--label-run-dir` may be repeated.

   ```bash
   python3 experiments/multivariate_pham/behavior_cloning.py --config <config.yaml> \
     --label-run-dir <labels-a> --label-run-dir <labels-b> --hidden-sizes 64,64 \
     --experiment-id EXP-0010 --run-id bc --oracle-run-dir <oracle-run-on-eval-seeds>
   ```

4. Multimodality: compare oracle runs of the same instances with different
   `--search-seed` (at least two) and test whether near-best actions are
   similar and whether their midpoints stay near-best.

   ```bash
   python3 experiments/multivariate_pham/oracle_modes.py \
     --oracle-run-dir <oracle-seed0> --oracle-run-dir <oracle-seed1> \
     --tolerance 0.02 --output-dir outputs/EXP-0010/modes
   ```

## Symmetry (EXP-0011)

`symmetry_check.py` verifies numerically that the tracking cost of a Bezier
path is invariant under the symmetry group of the Pham family (variable and
equation permutations, roots of unity, conjugation; see
`docs/design/mathematics/symmetry.md`). It tracks the original control points
and every transformed copy for the zero action and random actions.

```bash
python3 experiments/multivariate_pham/symmetry_check.py --config <config.yaml> \
  --num-instances 20 --random-actions 2 --output-dir outputs/EXP-0011/symmetry-check
```

`behavior_cloning.py --latent-dim full --augment-symmetry` trains on
symmetry-transformed samples. It requires labels from per-instance oracle
runs that used the full latent space (`oracle.py --latent-dim full`) and the
same config, so that the latent basis agrees.

## Cost Fine-Tuning (EXP-0013)

`finetune.py` starts from a full-latent behavior-cloning run and minimizes the
tracking cost directly. For each training problem it estimates the gradient of
the cost ratio `J / J_lin` with respect to the action by antithetic finite
differences around the policy mean and back-propagates it through
`actor_mean`. An optional behavior-cloning term (`--bc-weight`,
`--label-run-dir`, `--augment-symmetry`) regularizes towards oracle labels.
The weights with the lowest validation `Delta J` (iteration 0 included) are
kept. Inference is unchanged: one forward pass and one tracking.

```bash
python3 experiments/multivariate_pham/finetune.py --config <config.yaml> --latent-dim full \
  --init-run-dir <bc-run> --iterations 2000 --batch-problems 32 --pairs 4 --sigma 0.1 \
  --learning-rate 1e-4 --experiment-id EXP-0013 --run-id <run-id> --oracle-run-dir <oracle-full-run>
```

Outputs: `finetune_metrics.csv`, `val_metrics.csv`, `checkpoints/last.pt`,
`evaluation/evaluation.csv`, and `finetune_summary.csv` when `--oracle-run-dir`
is given.

## Attention Policy (ADR-0007 stage 1)

`ppo.actor_type: attention` (optional YAML key, default `mlp`; further optional
keys `attention_d_model`, `attention_heads`, `attention_layers`,
`attention_ff_dim`) selects the permutation-equivariant Transformer actor of
`docs/design/implementation/attention_policy.md`. It supports only supports
whose free monomials are `{x_1, ..., x_n, 1}` and requires the full latent
space (`--latent-dim full`). The critic stays an MLP.

```bash
python3 experiments/multivariate_pham/behavior_cloning.py --config <config.yaml> --latent-dim full \
  --label-run-dir <labels> --actor-type attention --attention-d-model 128 --attention-heads 4 \
  --attention-layers 4 --attention-ff-dim 256 --augment-symmetry --device cuda \
  --experiment-id <EXP> --run-id <run-id>
```

`finetune.py` inherits the actor type and size from `--init-run-dir`.
`--device cuda` runs the policy forward/backward on the GPU (tracking stays on
the CPU); checkpoints load on any device. Checkpoints store `actor_type` and
`attention_kwargs`; older checkpoints load as MLP.

### Evaluating on a held-out set

`evaluate.py --output-subdir NAME` writes `<run-dir>/NAME/evaluation.csv`
instead of `evaluation/evaluation.csv`. A non-default subdirectory is never
overwritten, and the evaluation is recorded in `metadata.json` under
`extra_evaluations`.

```bash
python3 experiments/multivariate_pham/evaluate.py --run-dir <run> --checkpoint <run>/checkpoints/last.pt \
  --evaluation-seed 41000 --num-instances 500 --output-subdir evaluation_test41000
```

### Comparing runs

```bash
python3 experiments/multivariate_pham/compare_runs.py --evaluation-subdir evaluation_test41000 \
  --group mlp=<mlp-s0>,<mlp-s1>,<mlp-s2> --group attn=<a-s0>,<a-s1>,<a-s2> \
  --oracle-run-dir <oracle-run> --pair mlp,attn --bootstrap 2000 --seed 0 --output-dir <out>
```

Runs must be evaluated on the same target seeds with identical Linear costs
(checked, also against the oracle). Groups average the per-problem policy costs
over their members. Outputs: `comparison_summary.csv` (Delta J, positive rate,
captured fraction per run/group), `comparison_pairs.csv` (B - A with 95%
paired percentile-bootstrap intervals over problems) and `metadata.json`.

### Inference timing

```bash
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 experiments/multivariate_pham/timing.py --config <config.yaml> --latent-dim full \
  --run-dir <mlp-run> --run-dir <attention-run> --num-instances 100 --output-dir <out>
```

Run `timing.py` as a single process on an otherwise idle machine (no other
experiment running). It reports per-problem median times over `--repeats`
(`timing.csv`) and a summary with ratios to the linear path
(`timing_summary.csv`).

