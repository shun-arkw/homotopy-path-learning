from juliacall import Main as jl
import cmath

a_end, b_end = 1.574004859917217, -4.326494374235970

jl.seval(f"""
using HomotopyContinuation, DynamicPolynomials
@polyvar x

# target polynomial as a System
P = System([x^2 + {a_end}*x + {b_end}]; variables = [x])

# tracker options
opts = HomotopyContinuation.TrackerOptions(;
    max_steps=50_000,
    max_step_size=0.05,
    max_initial_step_size=0.05,
    min_step_size=1e-10,
    min_rel_step_size=1e-10,
    extended_precision=true,
)

# 1) 高レベル solve（最終結果）
all_results = solve(P;
    start_system    = :total_degree,
    tracker_options = opts,
)

# 2) total_degree からトラッカー＋start 解を取得
tracker, start_iter = total_degree(P; tracker_options = opts)

# collect して普通のベクトルに
starts = collect(start_iter)

# 3) 各 start 解からパスを追跡して PathResult を得る
path_results = track.(tracker, starts)

# パスごとのステップ数など
path_steps           = [steps(r) for r in path_results]
path_accepted_steps  = [accepted_steps(r) for r in path_results]
path_rejected_steps  = [rejected_steps(r) for r in path_results]
""")

# ---- Python 側 ----
print("all_results:", jl.seval("results(all_results)"))
print()
print("num_solutions:", jl.seval("nsolutions(all_results)"))
print("real_solutions:", jl.seval("real_solutions(all_results)"))
print("solutions:", jl.seval("solutions(all_results)"))

print("num_steps:", jl.seval("path_steps"))
print("num_accepted_steps:", jl.seval("path_accepted_steps"))
print("num_rejected_steps:", jl.seval("path_rejected_steps"))

# analytic roots
_disc = a_end*a_end - 4*b_end
r1_end = (-a_end + cmath.sqrt(_disc)) / 2
r2_end = (-a_end - cmath.sqrt(_disc)) / 2
print("analytic:", [r1_end, r2_end])
