# 多項式系のデータ表現

## 1．`PolynomialSystemSpec`

固定した多項式系族を表す不変オブジェクトを定義する．

```python
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class PolynomialSystemSpec:
    n_vars: int
    degrees: tuple[int, ...]
    exponents: npt.NDArray[np.int64]
    offsets: npt.NDArray[np.int64]
    leading_indices: npt.NDArray[np.int64]
    constant_indices: npt.NDArray[np.int64]

    @property
    def n_equations(self) -> int:
        return self.n_vars

    @property
    def n_coeffs(self) -> int:
        return int(self.exponents.shape[0])
```

$n$変数$n$本の正方系のみを最初の対象とするため，`n_equations == n_vars`とする．

## 2．検証条件

初期化時に，少なくとも次を検証する．

- `n_vars > 0`
- `len(degrees) == n_vars`
- すべての$d_i$が正整数であること
- `exponents.shape == (M, n_vars)`
- `offsets.shape == (n_vars + 1,)`
- `offsets[0] == 0`
- `offsets[-1] == M`
- `offsets`が狭義単調増加であること
- 指数が非負整数であること
- 各方程式に$x_i^{d_i}$がちょうど1個存在すること
- 各方程式に定数項がちょうど1個存在すること
- 先頭単項式以外が固定した単項式順序で$x_i^{d_i}$より小さいこと

最初の実装では次数適合順序を前提とし，簡易条件

$$
|\boldsymbol{\alpha}|<d_i
$$

で検証してよい．任意の単項式順序への対応は後続課題とする．

## 3．係数ベクトル

完全な係数ベクトルは，先頭係数を含む$\mathbb{C}^M$のベクトルとする．固定成分を含めることで，PythonとJuliaの多項式評価に同じ配列を利用できる．

```python
start_coeffs: np.ndarray   # shape (M,), complex128
target_coeffs: np.ndarray  # shape (M,), complex128
```

先頭係数は常に$1$である．開始系の定数係数は$-1$である．

## 4．自由係数マスク

```python
free_mask = np.ones(M, dtype=bool)
free_mask[leading_indices] = False
```

必要に応じて，さらに固定する係数をマスクから除く．観測には固定成分を含めず，次の実数ベクトルを使用してよい．

```python
free_target = target_coeffs[free_mask]
observation = np.concatenate([free_target.real, free_target.imag])
```

## 5．Pham系仕様の生成

`systems/pham.py`に，仕様生成と検証を実装する．

```python
def build_pham_spec(
    degrees: tuple[int, ...],
    supports: tuple[tuple[tuple[int, ...], ...], ...],
) -> PolynomialSystemSpec:
    ...
```

`supports[i]`は，第$i$方程式の先頭項を除く候補指数集合である．関数内部で先頭指数$d_i\boldsymbol{e}_i$を追加し，定数項の存在を確認し，平坦化配列とoffsetを構成する．

## 6．係数サンプラー

`systems/sampling.py`に，係数分布を抽象化する．

```python
class CoefficientSampler(Protocol):
    def sample(
        self,
        rng: np.random.Generator,
        spec: PolynomialSystemSpec,
    ) -> np.ndarray:
        ...
```

最初は複素一様分布を実装する．

```python
@dataclass(frozen=True)
class ComplexUniformSampler:
    bound: float
```

自由係数の実部と虚部を独立に$\operatorname{Unif}[-b,b]$から生成し，先頭係数を$1$へ上書きする．

## 7．開始解

開始解集合はPython側でも生成できるが，追跡に使用する正式な開始解はJulia側で生成する．Python側の生成関数は，形状検証とテストに利用する．

```python
def pham_start_solutions(
    degrees: tuple[int, ...],
) -> np.ndarray:
    """Return shape (prod(degrees), n_vars)."""
```
