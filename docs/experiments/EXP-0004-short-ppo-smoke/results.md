# EXP-0004 Results

このファイルはPhase 6実装時に実行したsmoke学習の事実を記録する．
短時間smokeの結果から性能上の結論は出さない．

## 参照Dockerでの実行

- 実行環境：`homotopy-continuation`
- Python：3.10.12
- Julia：1.11.1
- PyTorch：2.3.0+cu121
- Gymnasium：1.2.3
- juliacall：0.9.31
- HomotopyContinuation.jl：2.15.2
- 総学習timesteps：256
- PPO update数：4
- 評価instance数：16

実行後に確認した生成物：

- `config.yaml`
- `metadata.json`
- `latent_basis.npy`
- `train_metrics.csv`
- `evaluation_seeds.json`
- `checkpoints/last.pt`
- `checkpoints/best.pt`
- `evaluation/evaluation.csv`
- `benchmark/benchmark.csv`
- `benchmark/benchmark_summary.csv`
- `analysis/summary.md`

## 観測された範囲

smoke実行では，学習，checkpoint保存，checkpoint読込，評価CSV生成，
benchmark CSV生成，summary Markdown生成が完了した．

性能改善の有無はこの実験の結論対象ではない．
