# Juliaカスタムホモトピーの実装方針

## 1．対象

HomotopyContinuation.jlの`AbstractHomotopy`を継承し，固定サポート上の係数がベジェ曲線に従って変化する$n$変数$n$本の多項式系を実装する．

## 2．ファイル分割

```text
julia/src/
├── HomotopyPathLearning.jl
├── system_spec.jl
├── bernstein.jl
├── bezier_homotopy.jl
├── start_solutions.jl
├── tracking.jl
└── api.jl
```

## 3．内部構造

概念的な構造体は次のとおりである．実際のフィールド型は，HomotopyContinuation.jlの要求と性能測定後に調整する．

```julia
struct PolynomialSystemSpec
    nvars::Int
    degrees::Vector{Int}
    exponents::Matrix{Int}
    offsets::Vector{Int}
    leading_indices::Vector{Int}
    constant_indices::Vector{Int}
end

mutable struct BezierPhamHomotopy <: AbstractHomotopy
    spec::PolynomialSystemSpec
    bezier_degree::Int
    control_points::Matrix{ComplexF64}
    coefficient_differences::Vector{Matrix{ComplexF64}}
    coefficient_buffers::Vector{Vector{ComplexF64}}
    monomial_buffer::Vector{ComplexF64}
end
```

## 4．パラメータ変換

数学上のベジェパラメータを$\tau$，HomotopyContinuation.jlから渡されるパラメータを$t$とし，

$$
\tau=1-\operatorname{Re}(t)
$$

とする．$t=1$で開始系，$t=0$で目的系となる．

## 5．実装する主要メソッド

- `Base.size(H)`
- `ModelKit.variables(H)`
- `ModelKit.parameters(H)`
- `ModelKit.evaluate!`
- `ModelKit.evaluate_and_jacobian!`
- `ModelKit.taylor!`

既存一変数実装のBernstein基底評価と制御点有限差分の処理は再利用できるが，多項式値とヤコビ行列は多変数向けに新規実装する．

## 6．多項式評価

第$i$方程式を

$$
H_i(\boldsymbol{x},\tau)
=
\sum_{q=o_i}^{o_{i+1}-1}
c_q(\tau)\boldsymbol{x}^{\boldsymbol{\alpha}_q}
$$

として評価する．

初期実装では，正しさを優先して指数ごとの累乗を直接計算する．正しさを確認した後，次の順で最適化する．

1．各変数の$x_j^k$を事前計算する．
2．全単項式値をキャッシュする．
3．偏微分単項式を単項式値から再利用する．
4．スレッドごとにバッファを確保し，追跡中のアロケーションを削減する．

## 7．ヤコビ行列

$$
\frac{\partial H_i}{\partial x_j}
=
\sum_{q=o_i}^{o_{i+1}-1}
c_q(\tau)\alpha_{q,j}
\boldsymbol{x}^{\boldsymbol{\alpha}_q-\boldsymbol{e}_j}
$$

を評価する．$\alpha_{q,j}=0$の項はスキップする．

ゼロ除算を避けるため，単項式値を$x_j$で割って偏微分を求める方法は，$x_j=0$の場合を適切に扱える実装が完成するまで使用しない．

## 8．$t$方向Taylor係数

ベジェ制御点の有限差分を用いて，係数曲線の高階微分を評価する．$\tau=1-t$であるため，$t$に関する$k$階微分には$(-1)^k$が付く．

$$
\frac{\partial^k H_i}{\partial t^k}
=
(-1)^k
\sum_q c_q^{(k)}(\tau)
\boldsymbol{x}^{\boldsymbol{\alpha}_q}
$$

既存一変数実装の`compute_diffs!`，Bernstein重み，階乗係数の処理を一般の係数数$M$へ拡張する．

## 9．開始解

開始系

$$
G_i(\boldsymbol{x})=x_i^{d_i}-1
$$

の解を，各変数の$d_i$乗根の直積として生成する．返却形状は概念的に

```text
(n_paths, n_vars)
```

であり，

$$
N_{\mathrm{path}}=\prod_{i=1}^n d_i
$$

である．

## 10．全パス追跡

全開始解を同一のホモトピーで追跡する．初期実装では逐次追跡で正しさを確認し，その後に既存一変数実装と同様のスレッド並列化を導入する．

追跡ごとに，次を記録する．

- 成功または失敗
- 受理ステップ数
- 棄却ステップ数
- 終点近似解
- 目的系に対する残差ノルム
- 失敗理由

## 11．API境界

`api.jl`以外の内部関数をPythonから直接呼び出さない．`api.jl`は，初期化，制御点更新，追跡，ウォームアップ，状態破棄の公開関数を提供する．
