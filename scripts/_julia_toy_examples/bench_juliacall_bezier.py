# bench_juliacall_bezier.py
import time
import numpy as np
from juliacall import Main as jl

def now():
    return time.perf_counter()

def main(m=4, n=200, seed=0):
    np.random.seed(seed)

    t0 = now()
    jl.seval('include("scripts/_julia_toy_examples/bezier_server.jl")')
    t_include = now() - t0

    # init (one-time)
    t1 = now()
    jl.init_bezier(m, seed=seed, extended_precision=False)
    t_init = now() - t1

    # ctrl buffer (Python side)
    ctrl = np.zeros((2, 3, m + 1), dtype=np.complex128)

    # endpointsは固定したいなら，Julia側build_ctrlと同じ値を入れる
    # 例：G endpoint at j=0
    ctrl[0,0,0] = 1+0j   # x^3
    ctrl[0,1,0] = 0+0j   # x*y
    ctrl[0,2,0] = -1+0j  # 1
    ctrl[1,0,0] = 1+0j   # y^3
    ctrl[1,1,0] = 0+0j
    ctrl[1,2,0] = -1+0j

    # F endpoint at j=m
    ctrl[0,0,m] = 1+0j
    ctrl[0,1,m] = 0.3+0.2j
    ctrl[0,2,m] = 0.1-0.4j
    ctrl[1,0,m] = 1+0j
    ctrl[1,1,m] = -0.25+0.15j
    ctrl[1,2,m] = -0.2+0.3j

    def randomize_middle():
        # j=1..m-1
        for j in range(1, m):
            ctrl[:, :, j] = 0.2 * (np.random.randn(2,3) + 1j*np.random.randn(2,3))
            ctrl[0,0,j] += 1.0
            ctrl[1,0,j] += 1.0

    # warmup
    randomize_middle()
    jl.set_ctrl_(m, ctrl) if hasattr(jl, "set_ctrl_") else jl.set_ctrl_bang(m, ctrl)  # fallback
    _ = jl.eval_once(m)

    times = []
    steps = []
    oks = []

    t2 = now()
    for i in range(n):
        randomize_middle()
        # set_ctrl + solve をまとめて計測
        ta = now()
        jl.set_ctrl_bang(m, ctrl)
        out = jl.eval_once(m)
        tb = now()

        times.append(tb - ta)
        steps.append(int(out.total_steps))
        oks.append(bool(out.ok))

        if (i+1) % 20 == 0:
            print(f"iter={i+1}  dt={times[-1]:.6f}s  steps={steps[-1]}  ok={oks[-1]}")

    wall = now() - t2
    mean_t = float(np.mean(times))
    mean_s = float(np.mean(steps))
    ok_rate = float(np.mean(oks))
    per_step_ms = (mean_t / mean_s) * 1e3 if mean_s > 0 else 0.0

    print("\n[juliacall_bezier_bench]")
    print(f"  include_sec={t_include:.6f}")
    print(f"  init_sec={t_init:.6f}")
    print(f"  loop_wall_sec={wall:.6f}  n={n}")
    print(f"  mean_eval_sec={mean_t:.6f}  ok_rate={ok_rate*100:.1f}%")
    print(f"  mean_total_steps={mean_s:.1f}  est_per_step_ms={per_step_ms:.4f}")

if __name__ == "__main__":
    main(m=4, n=200, seed=0)
