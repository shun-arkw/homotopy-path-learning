# python3 -m scripts._julia_toy_examples.hc_root_cluster_newton_iters
# 例2: 根クラスタ（疑似重根）: (x-1)(x-(1+ε))(x+2) で ε を小さくしていく
#
# 目的:
# - HomotopyContinuation.jl の track(debug=true) ログから "iters → N" を抜き出し、
#   根クラスタで Newton 反復が増えたり不安定化しやすい様子を観測する。
#
# 注意:
# - juliacall は torch 等より先に import すること（他スクリプトと同様）

from __future__ import annotations

import argparse
import re
from statistics import mean
from time import perf_counter

from juliacall import Main as jl


def seval(s: str):
    return jl.seval(s)


def _parse_iters(log: str) -> list[int]:
    return list(map(int, re.findall(r"iters\s*→\s*(\d+)", log)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eps1", type=float, default=1e-12, help="ターゲット多項式の根分離 ε（固定）")
    p.add_argument(
        "--extended-precision",
        action="store_true",
        help="TrackerOptions(extended_precision=true) を使う",
    )
    p.add_argument("--max-steps", type=int, default=200_000)
    p.add_argument("--min-step-size", type=float, default=1e-14)
    p.add_argument(
        "--no-warmup",
        action="store_true",
        help="warmup（JIT/初期化を馴らすための捨て実行）をしない",
    )
    args = p.parse_args()

    # --- basic info ---
    seval("using HomotopyContinuation")
    print("Julia VERSION:", seval("string(VERSION)"))
    print("HomotopyContinuation.jl:", seval("string(Base.pkgversion(HomotopyContinuation))"))
    print("Active project:", seval("Base.active_project()"))

    eps1 = args.eps1
    extprec = "true" if args.extended_precision else "false"

    # --- target polynomial (root cluster / pseudo multiple root) ---
    seval(
        rf"""
    using HomotopyContinuation
    @var x

    eps1 = {eps1}

    # ターゲット多項式を固定: f(x) = (x-1)(x-(1+eps1))(x+2)
    P = System([(x - 1) * (x - (1 + eps1)) * (x + 2)];
               variables=[x])

    opts = TrackerOptions(;
        extended_precision={extprec},
        max_steps={args.max_steps},
        min_step_size={args.min_step_size},
        min_rel_step_size={args.min_step_size},
    )

    # total degree homotopy で start system/solutions と tracker を自動生成
    tracker, start_iter = total_degree(P; tracker_options=opts)
    starts = collect(start_iter)
    """
    )

    print()
    print("--- experiment settings ---")
    print("eps1:", eps1)
    print("extended_precision:", args.extended_precision)
    print("max_steps:", args.max_steps)
    print("min_step_size:", args.min_step_size)

    # --- run track(debug=true) per path and parse Newton iters ---
    seval(
        "logs = String[]; return_codes = Any[]; residuals = Any[]; accuracies = Any[]; solutions = Any[]; runtimes_sec = Float64[]"
    )
    n_paths = int(seval("length(starts)"))
    print("num_paths (total_degree):", n_paths)

    # warmup: 1回だけ捨て実行して JIT/初期化の影響を本計測から外す
    # （debug=true で測っているので、ここも debug=true に合わせる）
    if not args.no_warmup and n_paths > 0:
        seval(
            r"""
        pipe = Pipe()
        redirect_stdout(pipe) do
            track(tracker, starts[1]; debug=true)
        end
        close(pipe.in)
        _ = String(read(pipe)) # 捨てる
        """
        )

    t_total0 = perf_counter()
    for i in range(n_paths):
        seval(
            rf"""
        pipe = Pipe()
        redirect_stdout(pipe) do
            t0 = time_ns()
            r = track(tracker, starts[{i+1}]; debug=true)
            push!(runtimes_sec, (time_ns() - t0) / 1e9)
            push!(return_codes, r.return_code)
            push!(residuals, r.residual)
            push!(accuracies, r.accuracy)
            push!(solutions, r.solution)
        end
        close(pipe.in)
        push!(logs, String(read(pipe)))
        """
        )
    t_total1 = perf_counter()

    logs = seval("logs")
    return_codes = seval("string.(return_codes)")
    residuals = seval("residuals")
    accuracies = seval("accuracies")
    solutions = seval("solutions")
    runtimes_sec = list(seval("runtimes_sec"))
    total_iters_all_paths = 0
    total_iters_events_all_paths = 0
    n_success = 0
    for i, log in enumerate(logs, start=1):
        iters = _parse_iters(log)
        print()
        print(f"--- path {i} ---")
        print("runtime_sec (track+debug):", runtimes_sec[i - 1] if i - 1 < len(runtimes_sec) else "n/a")
        rc = return_codes[i - 1]
        is_ok = (rc == "success")
        if is_ok:
            n_success += 1
        print("return_code:", rc, "(success)" if is_ok else "")
        # 参考: 数値品質の目安
        try:
            print("residual:", residuals[i - 1])
            print("accuracy:", accuracies[i - 1])
        except Exception:
            pass
        # 参考: 到達した解（1変数なので長さ1のベクトル）
        try:
            print("solution:", solutions[i - 1])
        except Exception:
            pass
        print("iters count:", len(iters))
        if iters:
            s = sum(iters)
            total_iters_all_paths += s
            total_iters_events_all_paths += len(iters)
            print("iters sum:", s)
            print("iters min/max/mean:", min(iters), max(iters), mean(iters))
        else:
            print("iters: (no matches; check debug log formatting)")

    print()
    print("--- total over all paths ---")
    print("success paths:", n_success, "/", n_paths)
    print("iters events total:", total_iters_events_all_paths)
    print("iters sum total:", total_iters_all_paths)
    if runtimes_sec:
        print("runtime_sec total (sum of per-path):", sum(runtimes_sec))
    print("walltime_sec total (python, includes overhead):", t_total1 - t_total0)
    print("DONE.")


if __name__ == "__main__":
    main()


