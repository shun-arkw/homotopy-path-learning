# python3 -m scripts._julia_toy_examples.hc_probe_newton_iters
# 方針A: debugログを取得し，Python側で "iters → N" を正規表現で抜いて集計する

from juliacall import Main as jl  # torch より先に import すること
import re


def seval(s: str):
    return jl.seval(s)


def main():
    # --- basic info ---
    seval("using HomotopyContinuation")
    print("Julia VERSION:", seval("string(VERSION)"))
    print("HomotopyContinuation.jl:", seval("string(Base.pkgversion(HomotopyContinuation))"))
    print("Active project:", seval("Base.active_project()"))
    print("DEPOT_PATH:", seval("string.(DEPOT_PATH)"))
    print("LOAD_PATH:", seval("string.(LOAD_PATH)"))

    # --- minimal HC instance ---
    seval(r"""
    using HomotopyContinuation
    @var x t

    # H(x,t) = (1-t)*(x^2-1) + t*(x^2-2) = x^2 - (1+t)
    F = System([(1 - t) * (x^2 - 1) + t * (x^2 - 2)];
               variables=[x], parameters=[t])

    starts = [[1.0], [-1.0]]
    start_params  = [0.0]
    target_params = [1.0]
    opts = TrackerOptions()
    """)

    # --- solve() once and inspect result object ---
    seval(r"""
    res = solve(F, starts;
        start_parameters=start_params,
        target_parameters=target_params,
        tracker_options=opts,
    )
    rr = results(res)
    """)
    print("\n--- solve(...) done ---")
    print("Result fields:", seval("propertynames(rr[1])"))

    print("\n--- last_path_point inspection ---")
    print("typeof(last_path_point):", seval("string(typeof(rr[1].last_path_point))"))
    print("last_path_point value:", seval("string(rr[1].last_path_point)"))

    # --- build tracker from explicit parameter homotopy ---
    seval(r"""
    H = parameter_homotopy(F; start_parameters=start_params, target_parameters=target_params)
    tracker = Tracker(H; options=opts)
    """)
    print("\n--- Tracker(H; options=opts) constructed ---")

    # --- run track with debug=true and capture stdout via Pipe ---
    seval(r"""
    pipe = Pipe()
    redirect_stdout(pipe) do
        track(tracker, starts[1], 1.0, 0.0; debug=true)
    end
    close(pipe.in)
    log = String(read(pipe))
    """)
    log = seval("log")

    print("\n--- track(debug=true) log ---\n")
    print(log)

    # --- Python-side parse of Newton iters ---
    iters = list(map(int, re.findall(r"iters\s*→\s*(\d+)", log)))
    print("\n--- parsed Newton iters ---")
    print("iters list:", iters)
    print("count:", len(iters))
    print("sum:", sum(iters))
    print("mean:", (sum(iters) / len(iters)) if iters else float("nan"))

    # --- small helper: show only lines around iters (optional) ---
    lines = log.splitlines()
    focus = [ln for ln in lines if ("iters" in ln or "NewtonCorrectorResult" in ln)]
    if focus:
        print("\n--- focused lines ---")
        for ln in focus:
            print(ln)

    print("\nDONE.")


if __name__ == "__main__":
    main()
