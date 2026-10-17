"""Count which term of the HomotopyContinuation.jl step size control sets each proposed step size.

usage: step_size_terms.py --run-dir RUN --evaluation-seed SEED [--num-instances N]

For each target (seeds from ``fixed_evaluation_seeds(SEED, N)``), the linear path and the learned Bezier path of the
run's network (``checkpoints/last.pt``, CPU) are tracked again with HC.jl's path iterator and the run's own tracker
options. After every accepted step, the step size that HC.jl proposes for the next step is compared (relative
tolerance 1e-9) with the candidates of ``update_stepsize!`` in HC.jl 2.15.2 (``src/tracker.jl``):

* ``delta_s1``: (sqrt(1 + 2 h(beta_a a)) - 1) / (omega e))^(1/p) / beta_omega_p (Newton convergence region),
* ``delta_s2``: beta_tau * tau (trust region of the predictor), ``strict``: strict_beta_tau * tau,
* ``max_step_size``, ``10x_previous``, ``previous`` (after a failed step),
* ``other``: none of these.

The value proposed after the last step of a path is not used and is excluded. The script also checks, for every
iterate, whether the second and third Taylor coefficients of the solution path are zero and whether the trust
region equals |x| / max(|x|, |x'|, |x''|, |x'''|) (HC.jl's fallback value). It prints a summary and writes no file.

Use development or training targets only; do not use the test sets for this diagnostic.
Report: docs/reports/2026-10-17-step-size-settings.md (section 5.3).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import preload_julia  # noqa: E402

preload_julia()

from juliacall import Main as jl  # noqa: E402

from homotopy_path_learning.config import build_env, load_config  # noqa: E402
from homotopy_path_learning.evaluation.datasets import fixed_evaluation_seeds  # noqa: E402
from homotopy_path_learning.evaluation.evaluator import _action_from_agent, load_policy_for_config  # noqa: E402

CATEGORIES = ("delta_s1", "delta_s2", "strict", "max_step_size", "10x_previous", "previous", "other")

JULIA_CODE = r"""
function step_size_terms()
  S = HomotopyPathLearning._BEZIER_PHAM_STATE[]
  H = HomotopyPathLearning._homotopy_for_tracking(S.homotopy)
  opts = S.tracker_options
  P = opts.parameters
  a = P.β_a * P.a
  ha = 2a * (sqrt(4a^2 + 1) - 2a)
  counts = Dict{String,Int}()
  n_iterates = 0; n_zero_taylor = 0; n_fallback_tau = 0; n_rejected = 0
  for k in 1:size(S.starts, 1)
    tracker = HomotopyContinuation.Tracker(H; options = opts)
    x = ComplexF64.(vec(S.starts[k, :]))
    prev_acc = 0; prev_rej = 0; first = true
    labels = String[]
    for _ in HomotopyContinuation.iterator(tracker, x, 1.0, 0.0)
      st = tracker.state; pr = tracker.predictor
      tn = pr.tx_norm
      n_iterates += 1
      n_zero_taylor += (tn[3] == 0.0 && tn[4] == 0.0)
      n_fallback_tau += isapprox(st.τ, tn[1] / maximum(tn); rtol = 1e-9)
      accepted = st.accepted_steps > prev_acc
      rejected = st.rejected_steps > prev_rej
      prev_acc = st.accepted_steps; prev_rej = st.rejected_steps
      if first
        first = false
        continue
      end
      if rejected
        n_rejected += 1
        continue
      end
      accepted || continue
      Δs = st.segment_stepper.Δs
      ω = clamp(st.ω + 2(st.ω - st.ω_prev), st.ω, 8st.ω)
      d1 = ((sqrt(1 + 2ha) - 1) / (ω * pr.local_error))^(1 / pr.order) / P.β_ω_p
      candidates = (("delta_s1", d1), ("delta_s2", P.β_τ * st.τ), ("strict", P.strict_β_τ * st.τ),
                    ("max_step_size", opts.max_step_size), ("10x_previous", 10 * st.Δs_prev),
                    ("previous", st.Δs_prev))
      label = "other"
      for (name, value) in candidates
        if isfinite(value) && isapprox(Δs, value; rtol = 1e-9)
          label = name
          break
        end
      end
      push!(labels, label)
    end
    isempty(labels) || pop!(labels)  # the value proposed after the last step is not used
    for label in labels
      counts[label] = get(counts, label, 0) + 1
    end
  end
  return (counts, n_iterates, n_zero_taylor, n_fallback_tau, n_rejected)
end
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evaluation-seed", type=int, required=True)
    parser.add_argument("--num-instances", type=int, default=20)
    args = parser.parse_args()

    jl.seval("using HomotopyPathLearning, HomotopyContinuation")
    jl.seval(JULIA_CODE)
    config = load_config(args.run_dir / "config.yaml")
    agent = load_policy_for_config(checkpoint_path=args.run_dir / "checkpoints" / "last.pt", config=config, device="cpu")
    env = build_env(config)

    totals = {name: Counter() for name in ("Linear", "LearnedBezier")}
    checks = {name: Counter() for name in totals}
    for seed in fixed_evaluation_seeds(seed=args.evaluation_seed, num_instances=args.num_instances):
        observation, _ = env.reset(seed=seed)
        context = env.current_episode_context()
        action = _action_from_agent(agent, observation, device="cpu")
        for name, a in (("Linear", 0.0 * action), ("LearnedBezier", action)):
            env.evaluate_action_against_context(context, a)  # leaves these control points in the Julia state
            counts, n_iterates, n_zero_taylor, n_fallback_tau, n_rejected = jl.step_size_terms()
            for key in counts.keys():
                totals[name][str(key)] += int(counts[key])
            checks[name].update(iterates=int(n_iterates), zero_taylor=int(n_zero_taylor),
                                fallback_tau=int(n_fallback_tau), rejected=int(n_rejected))

    print(f"run={args.run_dir.name} evaluation_seed={args.evaluation_seed} num_instances={args.num_instances}")
    for name in totals:
        total = sum(totals[name].values())
        c = checks[name]
        print(f"{name}: proposals={total} rejected_steps={c['rejected']} iterates={c['iterates']} "
              f"zero_2nd_3rd_taylor={c['zero_taylor']} fallback_tau={c['fallback_tau']}")
        for key in CATEGORIES:
            n = totals[name][key]
            print(f"  {key:14s} {n:7d} {100.0 * n / total:6.1f}%")


if __name__ == "__main__":
    main()
