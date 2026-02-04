"""
discriminant_univariate_logabs (Torch/Sylvester) vs sympy.discriminant の精度・ランタイム比較。

実行例:
  python scripts/task2/optimal_path_finding/test_discriminant.py --trials 200 --coeff-range 5
  python scripts/task2/optimal_path_finding/test_discriminant.py --trials 200 --coeff-range 5 --coeff-range-im 5

メモ:
- SymPy は整数係数の判別式を厳密計算できるので、比較の「真値」側として使う。
- Torch 側は float64 で計算する（行列式系は float32 だと誤差が出やすい）。
- discriminant_calculator.py の実装に合わせ、比較は |Disc| と log|Disc| を基本にする（複素係数でも定義できる）。
"""

from __future__ import annotations
from scripts.task2.optimal_path_finding.utils.discriminant_calculator import discriminant_univariate_logabs
import argparse
import importlib.util
import math
import os
import time
from dataclasses import dataclass
from typing import List, Tuple

import sympy as sp
import torch


# def _load_discriminant_module():
#     """同ディレクトリの discriminant_calculator.py をパス指定で import する。"""
#     here = os.path.dirname(os.path.abspath(__file__))
#     path = os.path.join(here, "discriminant_calculator.py")
#     spec = importlib.util.spec_from_file_location("discriminant_calculator", path)
#     if spec is None or spec.loader is None:
#         raise RuntimeError(f"Failed to create import spec for: {path}")
#     mod = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(mod)  # type: ignore[attr-defined]
#     return mod


@dataclass
class DegreeResult:
    deg: int
    trials: int
    torch_time_s: float
    sympy_time_s: float
    abs_abs_err_median: float
    abs_abs_err_max: float
    abs_rel_err_median: float
    abs_rel_err_max: float
    logabs_err_median: float
    logabs_err_max: float
    sympy_logabs_median: float
    sympy_zero: int
    torch_zero: int
    bad_abs_value_count: int  # |Disc| の float 比較ができない（nan/inf 等）
    bad_log_count: int  # log 比較ができない（±inf/nan 等）


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        # SymPy の巨大整数などで float 変換が落ちた場合は inf 扱い
        return float("inf")


def _rel_err(a: float, b: float) -> float:
    # |a-b| / max(1, |b|)
    denom = max(1.0, abs(b))
    return abs(a - b) / denom


def _int_sign_logabs(n: int) -> Tuple[int, float]:
    """
    Python int の符号と log|n|（自然対数）を float で返す。
    巨大整数でも overflow せず、上位ビットから近似して log を取る。
    """
    if n == 0:
        return 0, float("-inf")
    sign = 1 if n > 0 else -1
    m = -n if n < 0 else n

    # log(m) を bit_length を使って近似:
    # m = 2^(bl-1) * frac, frac in [1,2)
    bl = m.bit_length()
    k = min(53, bl)  # float64 の仮数に収まる上位ビット数
    shift = bl - k
    top = m >> shift  # 上位 k ビット
    frac = top / float(1 << (k - 1))
    log2m = (bl - 1) + math.log2(frac)
    return sign, log2m * math.log(2.0)


def _int_logabs_pos(n: int) -> float:
    """n>0 のとき log(n) を返す（巨大整数でも overflow しにくい近似）。"""
    if n <= 0:
        raise ValueError("n must be > 0")
    _, logabs = _int_sign_logabs(n)
    return logabs


def _generate_int_coeffs(deg: int, coeff_range: int, g: torch.Generator) -> torch.Tensor:
    """
    次数 deg の整数係数（降べき）を生成。先頭係数は 0 にしない。
    """
    # [-R, R] の整数
    a = torch.randint(-coeff_range, coeff_range + 1, (deg + 1,), generator=g, dtype=torch.int64)
    # 先頭係数が 0 なら非ゼロにする
    if int(a[0].item()) == 0:
        a0 = torch.randint(1, coeff_range + 1, (1,), generator=g, dtype=torch.int64)
        if bool(torch.randint(0, 2, (1,), generator=g).item()):
            a0 = -a0
        a[0] = a0
    return a


def _generate_int_coeffs_complex(
    deg: int, coeff_range_re: int, coeff_range_im: int, g: torch.Generator
) -> torch.Tensor:
    """
    次数 deg の複素整数係数（降べき）を生成。
    shape (deg+1,2) where [...,0]=Re, [...,1]=Im
    先頭係数は 0+0i にしない。
    """
    re = torch.randint(-coeff_range_re, coeff_range_re + 1, (deg + 1,), generator=g, dtype=torch.int64)
    im = torch.randint(-coeff_range_im, coeff_range_im + 1, (deg + 1,), generator=g, dtype=torch.int64)
    if int(re[0].item()) == 0 and int(im[0].item()) == 0:
        a0 = torch.randint(1, coeff_range_re + 1, (1,), generator=g, dtype=torch.int64)
        if bool(torch.randint(0, 2, (1,), generator=g).item()):
            a0 = -a0
        re[0] = a0
    return torch.stack([re, im], dim=-1)


def _sympy_discriminant_from_coeffs(a_int_desc: torch.Tensor) -> Tuple[int, int]:
    """
    a_int_desc:
      - real: shape (deg+1,), int64, 降べき
      - complex: shape (deg+1,2), int64, [...,0]=Re, [...,1]=Im
    戻り値: SymPy による判別式（Gaussian 整数想定）を (re,im) の Python int で返す。
    """
    x = sp.Symbol("x")
    if a_int_desc.ndim == 1:
        deg = a_int_desc.numel() - 1
        coeffs_re = [int(v.item()) for v in a_int_desc]
        poly = sum(sp.Integer(coeffs_re[i]) * x ** (deg - i) for i in range(deg + 1))
    elif a_int_desc.ndim == 2 and a_int_desc.shape[-1] == 2:
        deg = a_int_desc.shape[0] - 1
        coeffs_re = [int(v.item()) for v in a_int_desc[:, 0]]
        coeffs_im = [int(v.item()) for v in a_int_desc[:, 1]]
        poly = sum(
            (sp.Integer(coeffs_re[i]) + sp.I * sp.Integer(coeffs_im[i])) * x ** (deg - i) for i in range(deg + 1)
        )
    else:
        raise ValueError("a_int_desc must be (deg+1,) or (deg+1,2)")

    disc = sp.discriminant(poly, x)
    disc_re = sp.re(disc)
    disc_im = sp.im(disc)
    if not (disc_re.is_Integer and disc_im.is_Integer):
        disc_re = sp.simplify(disc_re)
        disc_im = sp.simplify(disc_im)
    if not (disc_re.is_Integer and disc_im.is_Integer):
        raise RuntimeError(f"discriminant is not Gaussian integer: {disc!r}")
    return int(disc_re), int(disc_im)


def _gaussian_abs_logabs(re: int, im: int) -> Tuple[float, float]:
    """(re,im) の複素数の |.| と log|.| を返す。巨大数では abs は inf になり得る。"""
    if re == 0 and im == 0:
        return 0.0, float("-inf")
    r2 = re * re + im * im
    logabs = 0.5 * _int_logabs_pos(r2)
    abs_v = math.exp(logabs) if logabs < 709.0 else float("inf")
    return abs_v, logabs


def run_benchmark(
    *,
    deg_min: int,
    deg_max: int,
    trials: int,
    coeff_range: int,
    coeff_range_im: int,
    seed: int,
    device: str,
    eps: float,
) -> List[DegreeResult]:
    # disc_mod = _load_discriminant_module()
    # discriminant_univariate_logabs = disc_mod.discriminant_univariate_logabs

    torch_device = torch.device(device)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    results: List[DegreeResult] = []

    for deg in range(deg_min, deg_max + 1):
        # 係数は整数で共通に生成し、Torch 側は float64 にして計算
        if coeff_range_im > 0:
            coeffs_int: List[torch.Tensor] = [
                _generate_int_coeffs_complex(deg, coeff_range, coeff_range_im, g) for _ in range(trials)
            ]
            A = torch.stack(coeffs_int, dim=0).to(dtype=torch.float64, device=torch_device)  # (trials, deg+1,2)
        else:
            coeffs_int = [_generate_int_coeffs(deg, coeff_range, g) for _ in range(trials)]
            A = torch.stack(coeffs_int, dim=0).to(dtype=torch.float64, device=torch_device)  # (trials, deg+1)

        # Torch |Disc|, log|Disc|（バッチ計算）
        torch_start = time.perf_counter()
        disc_logabs_t = discriminant_univariate_logabs(A, eps=eps, lead_eps=0.0)
        disc_abs_t = torch.exp(disc_logabs_t)
        # 同期（GPU の場合）
        if torch_device.type == "cuda":
            torch.cuda.synchronize()
        torch_time = time.perf_counter() - torch_start

        disc_abs_cpu = disc_abs_t.detach().to("cpu")
        disc_logabs_cpu = disc_logabs_t.detach().to("cpu")

        # SymPy 判別式（逐次）
        sympy_start = time.perf_counter()
        disc_s_list: List[Tuple[int, int]] = [_sympy_discriminant_from_coeffs(a) for a in coeffs_int]
        sympy_time = time.perf_counter() - sympy_start

        # 誤差集計（巨大で float 化が溢れる場合は bad 扱い）
        abs_abs_errs: List[float] = []
        abs_rel_errs: List[float] = []
        logabs_errs: List[float] = []
        sympy_logabs_list: List[float] = []
        sympy_zero = 0
        torch_zero = 0
        bad_abs_value = 0
        bad_log = 0
        for i in range(trials):
            # SymPy: Disc は Gaussian 整数 (re,im) として受け、|.| と log|.| を作る
            s_re, s_im = disc_s_list[i]
            s_abs, s_logabs = _gaussian_abs_logabs(s_re, s_im)

            # Torch
            t_abs = _safe_float(disc_abs_cpu[i].item())
            t_logabs = float(disc_logabs_cpu[i].item())

            if s_abs == 0.0:
                sympy_zero += 1
            if t_abs == 0.0 or t_logabs == float("-inf"):
                torch_zero += 1

            # |Disc| の float 比較（inf/nan は除外）
            if (not math.isfinite(t_abs)) or (not math.isfinite(s_abs)):
                bad_abs_value += 1
            else:
                abs_abs_errs.append(abs(t_abs - s_abs))
                abs_rel_errs.append(_rel_err(t_abs, s_abs))

            # log|Disc| 比較（0 だと -inf になるので、有限同士のみ）
            if (not math.isfinite(t_logabs)) or (not math.isfinite(s_logabs)):
                bad_log += 1
            else:
                logabs_errs.append(abs(t_logabs - s_logabs))
                sympy_logabs_list.append(s_logabs)

        abs_abs_errs_sorted = sorted(abs_abs_errs)
        abs_rel_errs_sorted = sorted(abs_rel_errs)
        logabs_errs_sorted = sorted(logabs_errs)
        sympy_logabs_sorted = sorted(sympy_logabs_list)

        def median(xs: List[float]) -> float:
            if not xs:
                return float("nan")
            m = len(xs) // 2
            return xs[m] if (len(xs) % 2 == 1) else 0.5 * (xs[m - 1] + xs[m])

        results.append(
            DegreeResult(
                deg=deg,
                trials=trials,
                torch_time_s=torch_time,
                sympy_time_s=sympy_time,
                abs_abs_err_median=median(abs_abs_errs_sorted),
                abs_abs_err_max=max(abs_abs_errs_sorted) if abs_abs_errs_sorted else float("nan"),
                abs_rel_err_median=median(abs_rel_errs_sorted),
                abs_rel_err_max=max(abs_rel_errs_sorted) if abs_rel_errs_sorted else float("nan"),
                logabs_err_median=median(logabs_errs_sorted),
                logabs_err_max=max(logabs_errs_sorted) if logabs_errs_sorted else float("nan"),
                sympy_logabs_median=median(sympy_logabs_sorted),
                sympy_zero=sympy_zero,
                torch_zero=torch_zero,
                bad_abs_value_count=bad_abs_value,
                bad_log_count=bad_log,
            )
        )

    return results


def _print_table(results: List[DegreeResult]) -> None:
    # ざっくり見やすい表
    header = (
        "deg  trials  torch[s]   sympy[s]   speedup   |D|_abs_err_med  |D|_abs_err_max  |D|_rel_err_med  |D|_rel_err_max  "
        "logabs_err_med  logabs_err_max  logabs_sym_med   sym0  t0  badAbs  badLog"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        speedup = (r.sympy_time_s / r.torch_time_s) if r.torch_time_s > 0 else float("inf")
        print(
            f"{r.deg:>3d}  {r.trials:>6d}  {r.torch_time_s:>8.4f}  {r.sympy_time_s:>8.4f}  {speedup:>8.2f}  "
            f"{r.abs_abs_err_median:>15.3e}  {r.abs_abs_err_max:>15.3e}  {r.abs_rel_err_median:>14.3e}  {r.abs_rel_err_max:>14.3e}  "
            f"{r.logabs_err_median:>13.3e}  {r.logabs_err_max:>13.3e}  {r.sympy_logabs_median:>13.3e}  "
            f"{r.sympy_zero:>5d}  {r.torch_zero:>2d}  {r.bad_abs_value_count:>6d}  {r.bad_log_count:>6d}"
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--deg-min", type=int, default=2)
    p.add_argument("--deg-max", type=int, default=10)
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--coeff-range", type=int, default=5, help="係数を [-R, R] の整数で生成")
    p.add_argument(
        "--coeff-range-im",
        type=int,
        default=0,
        help="虚部係数を [-Rim, Rim] の整数で生成（>0 なら複素係数モード）",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu", help="cpu / cuda")
    p.add_argument("--eps", type=float, default=0.0, help="resultant の det 安定化用に eps*I を足す")
    args = p.parse_args()

    if args.deg_min < 2 or args.deg_max < args.deg_min:
        raise ValueError("degree range is invalid (need 2 <= deg_min <= deg_max)")
    if args.trials <= 0:
        raise ValueError("trials must be > 0")
    if args.coeff_range <= 0:
        raise ValueError("coeff-range must be > 0")
    if args.coeff_range_im < 0:
        raise ValueError("coeff-range-im must be >= 0")

    results = run_benchmark(
        deg_min=args.deg_min,
        deg_max=args.deg_max,
        trials=args.trials,
        coeff_range=args.coeff_range,
        coeff_range_im=args.coeff_range_im,
        seed=args.seed,
        device=args.device,
        eps=args.eps,
    )
    _print_table(results)


if __name__ == "__main__":
    main()
