# EXP-0004：短時間PPO smoke

## 目的

Phase 5までに実装した1ステップ`BezierPhamEnv`へ，環境非依存のPyTorch PPO実装を接続し，
学習，チェックポイント保存，固定seed評価，benchmark CSV生成，Markdown分析生成までの
一連のPhase 6パイプラインが参照Docker内で完了することを確認する．

この実験は接続確認であり，学習済み方策が線形パスを上回ることを目的としない．

## 設定

- 実験ID：EXP-0004
- 設定ファイル：`experiments/multivariate_pham/configs/exp-0004-smoke.yaml`
- 変数数：2
- 次数：`[2, 2]`
- ベジェ次数：3
- 潜在次元：4
- 観測：自由目的係数の`[Re(c_free), Im(c_free)]`
- 行動：tanh-squashed Gaussianによる有界連続行動
- PPO総step数：256
- rollout長：64
- 評価instance数：16

## 評価方法

固定評価seed`evaluation.seed + instance_index`を使用し，同じ目的係数に対して
Linear，RandomBezier，LearnedBezierを比較する．

生成物は`outputs/EXP-0004/<run-id>/`へ保存する．
