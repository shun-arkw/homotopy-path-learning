# EXP-0008 Results

このファイルは参照Docker`homotopy-continuation`で実行したPhase 7性能測定の
事実のみを記録する．

## 実行環境

- Python：3.10.12
- Julia：1.11.1
- HomotopyContinuation.jl：2.15.2
- juliacall：0.9.31
- PyTorch：2.3.0+cu121
- Gymnasium：1.2.3
- Julia thread数：未設定

## Run

- Baseline run：`phase7-baseline-20260726T205422Z`
- Baseline directory：`outputs/EXP-0008/phase7-baseline-20260726T205422Z`
- Optimized run：`phase7-optimized-20260726T210054Z`
- Optimized directory：`outputs/EXP-0008/phase7-optimized-20260726T210054Z`

## 主要結果

| Metric | Baseline median s | Optimized median s | Speedup | Time reduction |
|---|---:|---:|---:|---:|
| `track_same_control_points` | 0.0001965015 | 0.0001958705 | 1.003222 | 0.321% |
| `env_reset` | 0.0004535495 | 0.0004520535 | 1.003309 | 0.330% |
| `env_step` | 0.0009736690 | 0.0009622770 | 1.011839 | 1.170% |
| `fixed_evaluation` | 0.1528274820 | 0.1044146820 | 1.463659 | 31.678% |
| `benchmark` | 0.2582341080 | 0.1577842590 | 1.636628 | 38.899% |

## 数値同値性

固定100問題，3手法，合計300 tracking recordsで比較した．

- path success差分：0
- accepted/rejected steps差分：0
- failure code差分：0
- endpoint最大差分：0.0
- residual最大差分：0.0
- 非有限値：0
- elapsed timeとrun IDを除く評価行差分：0

## 採用した最適化

評価器で同一target seedに対して`env.reset(seed)`を1回だけ実行し，
そのepisode contextに保存された目的係数，線形制御点，線形追跡結果，`J_lin`を
Linear，RandomBezier，LearnedBezierで共有するようにした．

変更前はbenchmarkで1 target seedあたりLinear，RandomBezier，LearnedBezierの
各評価ごとに線形追跡を行っていたため，線形追跡は3回だった．
変更後は1回に削減した．

## 生成物

両runで次を生成した．

- `config.yaml`
- `performance.json`
- `performance.csv`
- `equivalence.json`
- `profiling/python.txt`
- `profiling/julia.txt`
- `summary.md`

この測定結果は小規模smoke設定に限定される．
