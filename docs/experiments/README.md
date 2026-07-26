# Experiment Records

本ディレクトリは，個々の実験の目的，仮説，条件，結果，考察を記録する．設計仕様は`docs/design/`へ記載し，本ディレクトリには実験固有の情報だけを記載する．

## ディレクトリ名

```text
EXP-0001-n2-d2-smoke-test/
EXP-0002-n2-d2-linear-baseline/
EXP-0003-n2-d2-ppo-db3/
```

実験IDを主キーとし，日付は文書内に記載する．

## 対応関係

```text
実行設定:
experiments/multivariate_pham/configs/exp-0003.yaml

実験記録:
docs/experiments/EXP-0003-n2-d2-ppo-db3/

生成物:
outputs/EXP-0003/
```

## 推奨ファイル

```text
EXP-XXXX-name/
├── README.md
├── results.md
└── notes.md
```

- `README.md`は，実験開始前に目的，仮説，設定，評価方法を記載する．
- `results.md`は，実験終了後に定量結果と結論を記載する．
- `notes.md`は，実行中の問題，修正，観察を時系列で記載する．

新しい実験を開始する際は，`templates/experiment_record.md`をコピーして使用する．

## EXP-0004

Phase 6の短時間PPO smoke testは次で管理する．

```text
実行設定:
experiments/multivariate_pham/configs/exp-0004-smoke.yaml

実験CLI:
experiments/multivariate_pham/train.py
experiments/multivariate_pham/evaluate.py
experiments/multivariate_pham/benchmark.py
experiments/multivariate_pham/analyze.py

実験記録:
docs/experiments/EXP-0004-short-ppo-smoke/

生成物:
outputs/EXP-0004/<run-id>/
```

EXP-0004は学習パイプラインの接続確認であり，性能改善の結論を目的としない．
