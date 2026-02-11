# python3 scripts/bezier_hc_ppo/bench_julia_bezier_univar.py --degree 20 --bezier-degree 3 --n-eval 10
import time
import numpy as np
from juliacall import Main as jl

from dataclasses import dataclass, asdict

@dataclass
class BezierUnivarConfig:
    degree: int
    bezier_degree: int
    seed: int
    compute_newton_iters: bool = False
    max_steps: int = 50_000
    max_step_size: float = 0.05
    max_initial_step_size: float = 0.05
    min_step_size: float = 1e-12
    extended_precision: bool = False


def now():
    return time.perf_counter()

def randomize_middle(ctrl, rng, degree, bezier_degree, sigma_mid=0.2):
    # rows j=1..bezier_degree-1 = middle control points
    for j in range(1, bezier_degree):
        ctrl[j, :] = sigma_mid * (rng.standard_normal(degree + 1) + 1j * rng.standard_normal(degree + 1))
        ctrl[j, 0] += 1.0  # keep leading near 1


def randomize_target_system_univar(ctrl, rng, degree, bezier_degree, sigma_f=0.3, min_lead=1e-3):
    # row bezier_degree = F endpoint (random coeffs)
    ctrl[bezier_degree, :] = sigma_f * (rng.standard_normal(degree + 1) + 1j * rng.standard_normal(degree + 1))
    if abs(ctrl[bezier_degree, 0]) < min_lead:
        ctrl[bezier_degree, 0] += 1.0 + 0.0j


def set_total_degree_start_system_univar(ctrl, degree):
    # row 0 = G endpoint: x^degree - 1 (coeffs a_n..a_0)
    ctrl[0, :] = 0.0 + 0.0j
    ctrl[0, 0] = 1.0 + 0.0j       # a_n
    ctrl[0, degree] = -1.0 + 0.0j  # a_0


def main(degree=10, bezier_degree=4, n_eval=200, seed=0, compute_newton_iters=False):
    rng = np.random.default_rng(seed)
    config = BezierUnivarConfig(
        degree=int(degree),
        bezier_degree=int(bezier_degree),
        compute_newton_iters=bool(compute_newton_iters),
        seed=int(seed),
        extended_precision=False,
    )

    t0 = now()
    jl.seval('include("scripts/bezier_hc_ppo/bezier_univar.jl")')
    t_include = now() - t0

    t1 = now()
    jl.init_bezier_univar(**asdict(config))
    t_init = now() - t1

    # ctrl shape: (bezier_degree+1, degree+1), row j = j-th control point's coeffs [a_n..a_0]
    ctrl = np.zeros((config.bezier_degree + 1, config.degree + 1), dtype=np.complex128)

    degree, bezier_degree = config.degree, config.bezier_degree

    # G endpoint at j=0: x^degree - 1
    set_total_degree_start_system_univar(ctrl, degree)

    # warmup (same compute_newton_iters as main loop for fair timing)
    randomize_middle(ctrl, rng, degree, bezier_degree, sigma_mid=0.2)
    randomize_target_system_univar(ctrl, rng, degree, bezier_degree, sigma_f=0.3)
    t_2 = now()
    _ = jl.track_bezier_paths_univar(degree, bezier_degree, ctrl, compute_newton_iters=compute_newton_iters)
    t_warmup = now() - t_2

    runtime_sec_lst = []
    tracking_time_sec_lst = []
    total_step_attempts_lst = []
    total_accepted_steps_lst = []
    total_rejected_steps_lst = []
    success_flag_lst = []
    total_newton_iterations_lst = [] if compute_newton_iters else None

    t3 = now()
    for i in range(n_eval):
        randomize_middle(ctrl, rng, degree, bezier_degree, sigma_mid=0.2)
        randomize_target_system_univar(ctrl, rng, degree, bezier_degree, sigma_f=0.3)
        out = jl.track_bezier_paths_univar(degree, bezier_degree, ctrl, compute_newton_iters=compute_newton_iters)

        runtime_sec_lst.append(float(out.runtime_sec))
        tracking_time_sec_lst.append(float(out.tracking_time_sec))
        total_step_attempts_lst.append(int(out.total_step_attempts))
        total_accepted_steps_lst.append(int(out.total_accepted_steps))
        total_rejected_steps_lst.append(int(out.total_rejected_steps))
        success_flag_lst.append(bool(out.success_flag))
        if compute_newton_iters:
            total_newton_iterations_lst.append(int(out.total_newton_iterations))

        if (i + 1) % 20 == 0:
            line = f"iter={i+1}  runtime_sec={runtime_sec_lst[-1]:.6f}  tracking_time_sec={tracking_time_sec_lst[-1]:.6f}  steps={total_step_attempts_lst[-1]}  ok={success_flag_lst[-1]}"
            if compute_newton_iters:
                line += f"  newton_iters={total_newton_iterations_lst[-1]}"
            print(line)

    wall = now() - t3
    mean_runtime_sec = float(np.mean(runtime_sec_lst))
    mean_tracking_time_sec = float(np.mean(tracking_time_sec_lst))
    mean_total_step_attempts = float(np.mean(total_step_attempts_lst))
    mean_total_accepted_steps = float(np.mean(total_accepted_steps_lst))
    mean_total_rejected_steps = float(np.mean(total_rejected_steps_lst))
    success_rate = float(np.mean(success_flag_lst))
    per_step_ms = (mean_runtime_sec / mean_total_step_attempts) * 1e3 if mean_total_step_attempts > 0 else 0.0

    print("\n[juliacall_bezier_univar_bench]")
    print(f"  include_sec={t_include:.6f}")
    print(f"  init_sec={t_init:.6f}")
    print(f"  warmup_sec={t_warmup:.6f}")
    print(f"  loop_wall_sec={wall:.6f}  n_eval={n_eval}")
    print(f"  degree={degree}  bezier_degree={bezier_degree}")
    print(f"  mean_runtime_sec={mean_runtime_sec:.6f}")
    print(f"  mean_tracking_time_sec={mean_tracking_time_sec:.6f}")
    print(f"  success_rate={success_rate*100:.1f}%")
    print(f"  mean_total_step_attempts={mean_total_step_attempts:.1f}  est_per_step_ms={per_step_ms:.4f}")
    print(f"  mean_total_accepted_steps={mean_total_accepted_steps:.1f}  mean_total_rejected_steps={mean_total_rejected_steps:.1f}")
    if compute_newton_iters and total_newton_iterations_lst:
        mean_ni = float(np.mean(total_newton_iterations_lst))
        print(f"  mean_total_newton_iterations={mean_ni:.1f}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--compute-newton-iters", action="store_true", help="Compute and report total Newton iterations per eval (slower, uses debug=true)")
    ap.add_argument("--degree", type=int, default=20, help="Polynomial degree")
    ap.add_argument("--bezier-degree", type=int, default=3, help="Bezier degree")
    ap.add_argument("--n-eval", type=int, default=5, help="Number of evals")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")
    args = ap.parse_args()
    main(
        degree=args.degree,
        bezier_degree=args.bezier_degree,
        n_eval=args.n_eval,
        seed=args.seed,
        compute_newton_iters=args.compute_newton_iters,
    )
