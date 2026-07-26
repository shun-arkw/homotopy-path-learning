# Gymnasium環境の実装方針

## 1．クラス

```python
class BezierPhamEnv(gym.Env[np.ndarray, np.ndarray]):
    ...
```

環境は，多項式系の数学処理やJuliaの詳細を直接実装せず，次のコンポーネントを組み合わせる．

- `PolynomialSystemSpec`
- `CoefficientSampler`
- `BezierParameterization`
- `TrackerBackend`
- `RewardConfig`

## 2．エピソード長

最初の多変数実験は，1回の行動で全中間制御点を決定する1ステップ環境とする．

```text
episode_length = 1
```

この場合，`step`で追跡と報酬計算を実行し，`terminated=True`を返す．複数ステップで制御点を逐次改善する環境は，1ステップ版の正しさと学習可能性を確認した後に追加する．

## 3．観測

開始系とサポートは固定であるため，自由な目的係数だけを観測に含める．

$$
\boldsymbol{s}
=
\begin{pmatrix}
\operatorname{Re}\boldsymbol{c}_{F,\mathrm{free}}\\
\operatorname{Im}\boldsymbol{c}_{F,\mathrm{free}}
\end{pmatrix}
$$

観測次元は$2D_{\mathrm{free}}$である．必要に応じて，係数を係数分布のboundで正規化する．

## 4．行動

中間制御点数は$d_b-1$，潜在次元は$m$である．

```text
action shape = ((d_b - 1) * m,)
```

方策の出力範囲は`Box(-1, 1, ...)`とし，環境内部で摂動スケールを適用する．

## 5．`reset`

`reset`では，次を行う．

1．乱数生成器を初期化する．
2．目的係数をサンプリングする．
3．完全な目的係数ベクトルを構成する．
4．線形パスの追跡コストを計算する．
5．観測を返す．

線形コストの計算が高価な場合，固定データセットまたは決定的な問題IDに対してキャッシュを導入する．

## 6．`step`

`step`では，次を行う．

1．行動の形状と有限性を検証する．
2．ベジェ制御点を構成する．
3．先頭係数固定制約を検証する．
4．Juliaバックエンドで全パスを追跡する．
5．追跡コストを計算する．
6．報酬を計算する．
7．診断情報を`info`へ格納する．

## 7．コスト

第$\ell$パスのコストを

$$
J_\ell
=
\begin{cases}
N_{\mathrm{acc}}^{(\ell)}
+\rho N_{\mathrm{rej}}^{(\ell)},
&\text{成功時},\\
M_{\mathrm{fail}},
&\text{失敗時}
\end{cases}
$$

とする．問題単位のコストは，最初は全パスの平均とする．

$$
J
=
\frac{1}{N_{\mathrm{path}}}
\sum_{\ell=1}^{N_{\mathrm{path}}}J_\ell
$$

## 8．報酬

最初の報酬は

$$
r=\mu(J_{\mathrm{lin}}-J_{\mathrm{bez}})
$$

とする．コストのスケールが問題ごとに大きく異なる場合は，後続実験として

$$
r=
\log\frac{J_{\mathrm{lin}}+\varepsilon}
{J_{\mathrm{bez}}+\varepsilon}
$$

を比較する．

## 9．`info`

少なくとも次を返す．

```python
info = {
    "linear_cost": float,
    "bezier_cost": float,
    "improvement": float,
    "n_paths": int,
    "n_success": int,
    "n_failed": int,
    "accepted_steps": int,
    "rejected_steps": int,
    "max_residual": float,
}
```

## 10．失敗時の挙動

Julia呼び出し自体が例外で終了した場合と，数値追跡の一部が失敗した場合を区別する．

- 数値追跡の失敗は通常の環境遷移として扱い，失敗ペナルティを与える．
- API不整合，shape不正，Julia内部例外などは，初期開発中は例外として表面化させる．
- 長時間学習時には，回復可能なバックエンド例外を分類し，ログへ保存したうえで大きなペナルティを返す設計を検討する．
