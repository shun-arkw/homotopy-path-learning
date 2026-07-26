# ベジェ係数パスの実装

## 1．制御点

ベジェ次数を$d_b$とし，完全な係数ベクトルの次元を$M$とする．制御点配列の形状は

```text
(d_b + 1, M)
```

である．

端点は

$$
P_0=\boldsymbol{c}_G,
\qquad
P_{d_b}=\boldsymbol{c}_F
$$

に固定する．

## 2．線形補間上の基準点

中間制御点の基準を

$$
\overline{P}_k
=
\left(1-\frac{k}{d_b}\right)\boldsymbol{c}_G
+
\frac{k}{d_b}\boldsymbol{c}_F
$$

とする．すべての中間制御点を$\overline{P}_k$へ置くと，ベジェ曲線は端点間の直線と一致する．

## 3．潜在変数による摂動

先頭係数を除く実自由度を$D_{\mathrm{free}}$とする．複素自由係数を実数化した空間の次元は$2D_{\mathrm{free}}$である．潜在次元$m$を導入し，固定行列

$$
U\in\mathbb{R}^{2D_{\mathrm{free}}\times m}
$$

を用いる．

PPOの出力$\boldsymbol{z}_k\in\mathbb{R}^m$から，

$$
\mathcal{R}(\boldsymbol{\delta}_{k,\mathrm{free}})
=U\boldsymbol{z}_k
$$

を生成する．完全な$\boldsymbol{\delta}_k\in\mathbb{C}^M$へ戻す際，先頭係数成分を$0$に固定する．

## 4．推奨API

```python
@dataclass(frozen=True)
class BezierParameterization:
    bezier_degree: int
    basis: np.ndarray
    free_indices: np.ndarray
    leading_indices: np.ndarray

    def action_dim(self) -> int:
        return (self.bezier_degree - 1) * self.basis.shape[1]

    def build_control_points(
        self,
        start_coeffs: np.ndarray,
        target_coeffs: np.ndarray,
        action: np.ndarray,
    ) -> np.ndarray:
        ...
```

## 5．行動の形状

Gymnasiumの行動は1次元配列とする．

```text
shape: ((d_b - 1) * m,)
```

内部で

```python
action.reshape(d_b - 1, m)
```

へ変換する．

## 6．スケーリングとクリッピング

無制限な摂動は，判別集合への接近，係数ノルムの増大，学習の不安定化を招く可能性がある．最初は次のいずれかを用いる．

### 固定スケール

$$
\mathcal{R}(\boldsymbol{\delta}_{k,\mathrm{free}})
=
\lambda U\tanh(\boldsymbol{z}_k)
$$

### 端点距離に比例するスケール

$$
\mathcal{R}(\boldsymbol{\delta}_{k,\mathrm{free}})
=
\lambda
\|\boldsymbol{c}_F-\boldsymbol{c}_G\|_2
U\tanh(\boldsymbol{z}_k)
$$

初期実験では固定スケールを採用し，$\lambda$を設定ファイルで管理する．

## 7．不変条件

`build_control_points`の出力に対し，次を検証する．

- `control_points[0] == start_coeffs`
- `control_points[-1] == target_coeffs`
- 全制御点の先頭係数成分が$1$
- NaNまたはInfを含まない
- 形状が`(d_b + 1, M)`

## 8．線形パス

専用の別実装を初期段階では作らず，ゼロ行動により同じベジェホモトピー実装上で線形パスを表現する．これにより，線形パスと学習パスでTracker設定と数値評価器を完全に一致させる．
