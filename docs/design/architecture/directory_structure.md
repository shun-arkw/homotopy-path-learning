# 推奨ディレクトリ構成

## 1．全体構成

```text
homotopy-path-learning/
├── README.md
├── pyproject.toml
├── .gitignore
│
├── docs/
│   ├── README.md
│   ├── design/
│   │   ├── README.md
│   │   ├── mathematics/
│   │   ├── architecture/
│   │   ├── implementation/
│   │   └── decisions/
│   ├── experiments/
│   │   ├── README.md
│   │   └── templates/
│   ├── references/
│   └── reports/
│
├── src/
│   └── homotopy_path_learning/
│       ├── __init__.py
│       ├── config.py
│       ├── systems/
│       ├── paths/
│       ├── backends/
│       ├── envs/
│       ├── rl/
│       ├── performance/
│       └── evaluation/
│
├── julia/
│   ├── Project.toml
│   ├── Manifest.toml
│   ├── src/
│   └── test/
│
├── experiments/
│   ├── multivariate_pham/
│   │   ├── README.md
│   │   ├── configs/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── benchmark.py
│   │   ├── analyze.py
│   │   ├── oracle.py
│   │   ├── behavior_cloning.py
│   │   ├── oracle_modes.py
│   │   └── profile.py
│   └── univariate/          （未作成．一変数実験は別ブランチ・タグで保存）
│       └── README.md
│
├── tests/
├── scripts/
├── docker/
├── data/
└── outputs/
```

## 2．`src/homotopy_path_learning/`

再利用可能なPythonコードを配置する．実験IDや特定のハイパーパラメータを直接埋め込まない．

```text
src/homotopy_path_learning/
├── config.py
├── systems/
│   ├── spec.py
│   ├── pham.py
│   ├── sampling.py
│   └── start_solutions.py
├── paths/
│   └── parameterization.py
├── backends/
│   ├── base.py
│   ├── types.py
│   └── julia.py
├── envs/
│   ├── bezier_pham.py
│   └── costs.py
├── rl/
│   ├── distributions.py
│   ├── networks.py
│   ├── rollout.py
│   ├── ppo.py
│   ├── checkpoint.py
│   ├── behavior_cloning.py
│   └── train_utils.py
├── performance/
│   ├── timing.py
│   ├── comparison.py
│   └── reporting.py
└── evaluation/
    ├── datasets.py
    ├── metrics.py
    ├── evaluator.py
    ├── analyzer.py
    └── oracle.py
```

Phase 6では，`rl/`を環境非依存のPPO実装とし，Bezier/Pham固有の情報は
`experiments/multivariate_pham/`のCLIまたは`evaluation/`で扱う．
PPO本体には`linear_cost`，`bezier_cost`，多項式系仕様，Juliaバックエンドを埋め込まない．
Phase 7では，明示的に実行する性能測定と同値性比較を`performance/`へ配置する．
通常の学習・評価経路へ無条件のプロファイリング処理は追加しない．

## 3．`julia/`

Juliaコードを独立したJuliaプロジェクトとして管理する．

```text
julia/
├── Project.toml
├── Manifest.toml
├── src/
│   ├── HomotopyPathLearning.jl
│   ├── system_spec.jl
│   ├── bernstein.jl
│   ├── bezier_homotopy.jl
│   ├── start_solutions.jl
│   ├── tracking.jl
│   └── api.jl
└── test/
    ├── runtests.jl
    ├── test_helpers.jl
    ├── test_system_spec.jl
    ├── test_bezier_homotopy.jl
    ├── test_jacobian.jl
    ├── test_taylor.jl
    ├── test_start_solutions.jl
    └── test_tracking.jl
```

## 4．`experiments/`と`docs/experiments/`の違い

`experiments/`には実行コードと設定ファイルを配置する．`docs/experiments/`には人間向けの実験記録を配置する．

```text
experiments/multivariate_pham/configs/exp-XXXX-<variant>[-seedK].yaml
docs/experiments/EXP-XXXX-<name>/README.md
outputs/EXP-XXXX/<run-id>/
```

上記3か所を同じ実験IDで対応付ける．1つの実験IDに複数の設定ファイルが
あってよい．命名と`experiment.id`の確認手順の詳細は
`docs/experiments/README.md`を参照．

## 5．一変数コードの扱い

一変数実験は，多変数実装の予備実験として，現在のブランチまたはタグで保存する．新構成へ一変数専用コードをそのまま移動せず，複素数変換，ベジェ制御点構築，PPO，Juliaバックエンドなど，多変数でも再利用する機能だけを抽出する．

推奨手順は次のとおりである．

1．現在の`experiment/ai4math2026`をタグで保存する．
2．多変数化用の新しいブランチを作成する．
3．新構成を作成する．
4．既存コードから再利用可能な処理を段階的に移植する．
5．各段階で一変数版と多変数版の結果を比較し，退行を防ぐ．
