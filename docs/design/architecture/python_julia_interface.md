# Python–Juliaインターフェース仕様

## 1．基本方針

PythonからJuliaへは，Pythonのクラスを直接渡さず，数値配列とスカラー値だけを渡す．Julia側は初期化時に多項式系仕様を受け取り，追跡ごとには制御点のみを受け取る．

Phase 4のPython–Julia接続では，参照Docker環境内の
`juliacall==0.9.31`を使用する．`juliacall`のimport前に，
`PYTHON_JULIACALL_EXE`を参照Docker内のJulia実行ファイルへ，
`PYTHON_JULIACALL_PROJECT`をリポジトリの`julia/`プロジェクトへ
設定する．実行時に`Pkg.add`を呼ばず，Juliaの自動取得も許可しない．

## 2．多項式系仕様

### `degrees`

```text
shape: (n,)
dtype: int64
```

第$i$方程式の先頭単項式$x_i^{d_i}$の次数を保持する．

### `exponents`

```text
shape: (M, n)
dtype: int64
```

全方程式の候補指数を平坦化した配列である．第$q$行は，多重指数$\boldsymbol{\alpha}_q$を表す．先頭単項式を含む．

### `offsets`

```text
shape: (n + 1,)
dtype: int64
```

Python側では0始まりとし，第$i$方程式に属する係数添字を

```text
offsets[i] <= q < offsets[i + 1]
```

で表す．Julia側では，初期化時に1始まりの添字へ変換する．

### `leading_indices`

```text
shape: (n,)
dtype: int64
```

各方程式の先頭単項式に対応する平坦化係数添字である．Python側では0始まりとする．

### `constant_indices`

```text
shape: (n,)
dtype: int64
```

各方程式の定数項に対応する平坦化係数添字である．

## 3．係数ベクトル

### `start_coeffs`

```text
shape: (M,)
dtype: complex128
```

開始系$G_i(\boldsymbol{x})=x_i^{d_i}-1$の係数ベクトルである．先頭係数は$1$，定数係数は$-1$，その他は$0$である．

### `target_coeffs`

```text
shape: (M,)
dtype: complex128
```

目的系の係数ベクトルである．先頭係数は$1$に固定する．自由係数だけをサンプリングし，完全な係数ベクトルへ埋め込む．

## 4．ベジェ制御点

### `control_points`

```text
shape: (d_b + 1, M)
dtype: complex128
```

第0行が開始係数$\boldsymbol{c}_G$，最終行が目的係数$\boldsymbol{c}_F$である．各中間行が中間制御点である．全行について，`leading_indices`の成分は$1$でなければならない．

## 5．初期化API

Python側から，概念的には次の関数を呼び出す．

```python
backend.initialize(
    degrees=degrees,
    exponents=exponents,
    offsets=offsets,
    leading_indices=leading_indices,
    constant_indices=constant_indices,
    bezier_degree=bezier_degree,
    tracker_config=tracker_config,
)
```

Julia側の公開関数名は，例えば次とする．

```julia
init_bezier_pham!(
    degrees,
    exponents,
    offsets,
    leading_indices,
    constant_indices,
    bezier_degree;
    tracker_options...,
)
```

初期化時に，次をキャッシュする．

- 多項式系仕様
- 開始解集合
- ベジェ係数評価用バッファ
- 多項式値とヤコビ行列の作業バッファ
- HomotopyContinuation.jlのTracker
- スレッドごとの追跡状態

## 6．追跡API

Python側から，概念的には次を呼ぶ．

```python
result = backend.track(control_points)
```

Julia側は次のような公開関数を持つ．

```julia
track_bezier_paths!(control_points)
```

## 7．追跡結果

追跡結果は，Pythonで次のデータ構造へ変換できる形式で返す．

```python
@dataclass(frozen=True)
class TrackingResult:
    success: bool
    n_paths: int
    n_success: int
    n_failed: int
    accepted_steps: int
    rejected_steps: int
    per_path_accepted_steps: np.ndarray
    per_path_rejected_steps: np.ndarray
    path_success: np.ndarray
    endpoints: np.ndarray
    residual_norms: np.ndarray
    failure_codes: tuple[str, ...]
```

### 必須フィールド

- 全体成功フラグ
- パス数
- 成功数と失敗数
- 受理ステップ総数
- 棄却ステップ総数
- パスごとの成功フラグ
- パスごとの受理・棄却ステップ数

### デバッグ時に必要なフィールド

- 終点近似解
- 目的系に対する残差ノルム
- 失敗理由
- 最終追跡パラメータ
- 最小到達ステップ幅

Python側の`TrackingResult`へ変換する際，Juliaが所有する配列への参照は
保持せず，NumPy配列としてコピーする．

## 8．キャッシュキー

Julia側の状態は，少なくとも次から構成されるキーで管理する．

```text
(
    degrees,
    exponents,
    offsets,
    bezier_degree,
    tracker_options,
)
```

Pythonのプロセス内でサポートが固定される最初の実装では，状態を1個だけ保持してもよい．ただし，将来的な複数サポート対応を妨げないインターフェースにする．

## 9．検証

Python–Julia境界では，次を必ず検証する．

- 配列の次元とdtype
- `offsets[-1] == M`
- 各指数が非負整数であること
- `leading_indices`と`constant_indices`が各方程式の範囲内にあること
- 制御点の先頭係数がすべて$1$であること
- `control_points.shape[0] == bezier_degree + 1`
