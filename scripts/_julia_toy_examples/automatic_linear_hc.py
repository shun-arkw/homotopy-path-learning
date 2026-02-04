from juliacall import Main as jl
import cmath

a_end, b_end = 1.574004859917217, -4.326494374235970

jl.seval(f"""
using HomotopyContinuation, DynamicPolynomials
@polyvar x t

# target system F(x) = x^2 + a_end*x + b_end
F = System([x^2 + {a_end}*x + {b_end}]; variables=[x])

# total degree start system G(x) = x^2 - 1 （標準的な選び方）
G = System([x^2 - 1]; variables=[x])

# straight-line homotopy H(x,t) = γ*t*G(x) + (1-t)*F(x)
gamma = exp(2im * pi * 0.3141592653589793)  # 例：γ-trick 用のランダム位相
H = StraightLineHomotopy(G, F; gamma)

# tracker options
opts = HomotopyContinuation.TrackerOptions(;
    max_steps=50_000,
    max_step_size=0.05,
    max_initial_step_size=0.05,
    min_step_size=1e-10,
    min_rel_step_size=1e-10,
    extended_precision=true,
)

# Tracker を構成
tracker = Tracker(H; options=opts)

# start solutions for G(x) = 0  →  x = ±1 が total degree の標準解
start_sols = [[1.0 + 0im], [-1.0 + 0im]]

# 各パスについて詳細情報を取得
infos = [path_info(tracker, s, 1.0, 0.0) for s in start_sols]

# どんなフィールドがあるか確認（t の列・x の列のフィールド名を探す）
fieldnames_infos = [fieldnames(typeof(info)) for info in infos]
""")

print("fieldnames for each info struct:")
print(jl.seval("fieldnames_infos"))
