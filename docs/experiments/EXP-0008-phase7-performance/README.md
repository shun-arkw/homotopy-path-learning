# EXP-0008：Phase 7性能測定

## 目的

Phase 1からPhase 6までの多変数Pham型ベジェホモトピー追跡基盤について，
数値結果と再現性を維持したまま，測定に基づいて安全な性能改善だけを採用する．

この実験は小規模smoke系での性能確認であり，大規模な多変数系全般の性能結論を
目的としない．

## 測定条件

- 設定ファイル：`experiments/multivariate_pham/configs/exp-0008-phase7-performance.yaml`
- 変数数：2
- 次数：`[2, 2]`
- ベジェ次数：3
- 潜在次元：4
- パス数：4
- 固定評価問題数：100
- warmup回数：5
- 測定回数：30
- trial数：3
- 時間計測：Pythonは`time.perf_counter_ns()`
- speedup定義：`baseline_time / optimized_time`

## 採用基準

各最適化は，変更前のbaseline測定，数値同値性確認，変更後の測定を行い，
数値結果が変化せず，主要測定で明確な改善がある場合のみ採用する．

Phase 7では，まず評価時に同一target seedの線形追跡を重複して実行しないことを
最適化候補とする．Juliaカーネル最適化，Tracker再利用，batch API，threaded modeは，
安全性と効果を直接確認できない限り採用しない．
