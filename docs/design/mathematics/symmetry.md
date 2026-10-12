# Pham型多項式系族の対称性

## 1．目的

係数から最適な行動への写像を学習するとき，問題族が持つ対称性を学習器に使わせる．本書は，その対称性の群と，係数・潜在行動へ誘導される実線形写像を定義する．実装は`src/homotopy_path_learning/systems/symmetry.py`である．

## 2．群の元

$n$変数$n$本の系$F=\{f_i\}$の係数ベクトル$\boldsymbol{c}\in\mathbb{C}^M$は，`PolynomialSystemSpec`の順序（方程式ごとのブロック，ブロック内は先頭係数が最初）に従う．群の元を

$$
g=(\sigma,\boldsymbol{k},\mathrm{conj}),\qquad
\sigma\in S_n,\quad k_j\in\{0,\ldots,d_j-1\},\quad \mathrm{conj}\in\{0,1\}
$$

とする．$\omega_j=\exp(2\pi\mathrm{i}\,k_j/d_j)$とおく．$\sigma$は変数と方程式に同時に作用し，旧添字$j$は新添字$\sigma_j$へ移る．

## 3．作用

系$F$への作用は，次の順に適用する．

1. 共役：すべての係数$c$を$\overline{c}$に置き換える．
2. スケーリング：$f_i(\boldsymbol{x})\mapsto f_i(\omega_1x_1,\ldots,\omega_nx_n)$．単項式$\boldsymbol{x}^{\boldsymbol{\alpha}}$の係数に
$$
\mathrm{phase}(\boldsymbol{\alpha})=\prod_{j}\omega_j^{\alpha_j}
$$
を掛ける．
3. 置換：旧方程式$i$の単項式$\boldsymbol{x}^{\boldsymbol{\alpha}}$の係数を，新方程式$\sigma_i$の単項式$\boldsymbol{y}^{\boldsymbol{\beta}}$の係数へ移す．ここで$\beta_{\sigma_j}=\alpha_j$である．

実装は，係数ベクトルに対して`out[target[m]] = phase[m] * c'[m]`（$c'$は共役後の係数）の形で作用させる（`coefficient_map`）．位相は$2\pi\sum_jk_j\alpha_j/d_j$を有理数として厳密に約分し，$1/4$の整数倍のときは$\pm1,\pm\mathrm{i}$を正確な値で返す．

## 4．適合条件

次を満たす$\sigma$だけを許す．

- すべての$i$で$d_{\sigma_i}=d_i$．
- すべての$i$と，方程式$i$の支持に属する$\boldsymbol{\alpha}$について，置換後の$\boldsymbol{\beta}$が方程式$\sigma_i$の支持に属する．

$\boldsymbol{k}$と共役には追加の条件はない．

## 5．不変量

- 先頭単項式$x_i^{d_i}$は$x_{\sigma_i}^{d_{\sigma_i}}$に移り，その位相は$\omega_i^{d_i}=1$である．よって先頭係数は$1+0\mathrm{j}$のまま保たれる．位相は厳密に$1$で計算するので，この等式は浮動小数点でも厳密に成り立つ．
- 開始系$g_i=x_i^{d_i}-1$は，定数項の位相が$1$，共役で$-1$が不変なので，不変である．
- 作用は係数の実線形写像であり，複素数の絶対値を保つので，自由係数の実ベクトルには直交行列として作用する．

## 6．誘導される写像

### 6.1．自由係数

自由係数の実ベクトル$\boldsymbol{x}=[\mathrm{Re}\,\boldsymbol{c}_{\mathrm{free}},\mathrm{Im}\,\boldsymbol{c}_{\mathrm{free}}]$（`np.flatnonzero(spec.free_coefficient_mask)`の順）に対する直交行列$R_g\in\mathbb{R}^{2D\times2D}$が定まる（`free_real_matrix`）．この並びは，環境の観測（`BezierPhamEnv._observation_from_target`が`complex_coefficients_to_real(target[free_indices])`を返す）および潜在基底$U$の行の並びと一致する．したがって観測$\boldsymbol{o}$は$R_g\boldsymbol{o}$へ写る．

### 6.2．潜在行動

ベジェ制御点は，線形補間に潜在行動$\boldsymbol{z}_k$による摂動$U\boldsymbol{z}_k$を加えて作る．開始系と先頭係数は不変で，作用は実線形なので，制御点に$g$を作用させたものは，目的系に$g$を作用させたときの制御点で，摂動を$R_gU\boldsymbol{z}_k$に置き換えたものに等しい．$U$が$2D\times2D$の直交行列（$m=2D$，全潜在空間）なら，
$$
R_gU\boldsymbol{z}_k=U\,(U^\top R_gU)\boldsymbol{z}_k
$$
と書けるので，潜在行動は$U^\top R_gU$で写る．平坦化した行動$(d_b-1)m$次元には，この行列の$(d_b-1)$個のブロック対角行列が作用する（`latent_action_matrix`）．

$m<2D$（例えば$m=4$のランダム部分空間）では，$R_gU\boldsymbol{z}$が一般に$U$の列空間の外に出る．部分空間が群の作用で閉じていないため，潜在行動に作用する行列は定義できない．全潜在空間（`--latent-dim full`）が必要な理由である．

## 7．群の位数

| 系 | 支持 | 位数 |
|---|---|---:|
| n3-d222 | 各方程式に$x_1,x_2,x_3,1$ | 96（$3!\cdot2^3\cdot2$） |
| n2-d33 | 各方程式に$x_1,x_2,1$ | 36（$2!\cdot3^2\cdot2$） |
| n2-d33-dense | 各方程式に$x_1^2,x_1x_2,x_2^2,x_1,x_2,1$ | 36 |
| n2-d22（EXP-0004 smoke） | 疎な支持 | 16（$2!\cdot2^2\cdot2$） |

## 8．追跡コストの不変性

数学的には，対称な系の追跡パスは対応する解の間で同型であり，ステップ数が一致すると期待される．しかし，ステップ幅の適応制御と浮動小数点の丸めがあるため，追跡器のステップ数が厳密に不変かは経験的な問題である．これは`experiments/multivariate_pham/symmetry_check.py`で数値的に確認する．
