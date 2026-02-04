from juliacall import Main as jl
import cmath


a_start, b_start = -2.845281577082634, -0.517576427371104
a_end, b_end = 1.574004859917217, -4.326494374235970
r1_start, r2_start = 3.0168437996388886, -0.17156222255625453

jl.seval(f"""
using HomotopyContinuation, DynamicPolynomials
@polyvar x t

# Q, P
Q = [x^2 + {a_start}*x + {b_start}]
P = [x^2 + {a_end}*x + {b_end}]

# H(x,t) = (1-t)Q + tP
H  = (1 - t) .* Q .+ t .* P
Hsys = System(H; variables=[x], parameters=[t])

# start solutions
starts = [[{r1_start}+0im], [{r2_start}+0im]]

# tracker options
opts = HomotopyContinuation.TrackerOptions(;
    max_steps=50_000,
    max_step_size=0.05,
    max_initial_step_size=0.05,
    min_step_size=1e-10,
    min_rel_step_size=1e-10,
    extended_precision=true,
)

# solve homotopy
all_results = solve(Hsys, starts;
            start_parameters = [0.0],
            target_parameters = [1.0],
            tracker_options = opts)
""")

print("all_results:", jl.seval("results(all_results)"))
print()
print("num_solutions:", jl.seval("nsolutions(all_results)"))
print("real_solutions:", jl.seval("real_solutions(all_results)"))
print("solutions:", jl.seval("solutions(all_results)"))
print("num_steps:", jl.seval("[steps(r) for r in results(all_results)]"))
print("num_accepted_steps:", jl.seval("[accepted_steps(r) for r in results(all_results)]"))
print("num_rejected_steps:", jl.seval("[rejected_steps(r) for r in results(all_results)]"))


# analytic roots
_disc = a_end*a_end - 4*b_end
r1_end = (-a_end + cmath.sqrt(_disc)) / 2
r2_end = (-a_end - cmath.sqrt(_disc)) / 2
print("analytic:", [r1_end, r2_end])

print(jl.seval("VERSION"))