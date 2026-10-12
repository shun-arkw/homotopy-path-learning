# 学習・評価コードの実装方針

## 1．配置

```text
experiments/multivariate_pham/
├── README.md
├── configs/
├── train.py
├── evaluate.py
├── benchmark.py
├── analyze.py
├── oracle.py
├── behavior_cloning.py
└── oracle_modes.py
```

`behavior_cloning.py`はoracleの行動を教師とする教師あり学習のCLI，`oracle_modes.py`は探索シードの異なるoracle実行間で最良行動の多峰性を調べるCLIである．

再利用可能なPPO実装は`src/homotopy_path_learning/rl/`へ配置し，実験固有の設定読み込みと入出力だけを`experiments/`へ置く．

## 2．設定ファイル

YAMLで管理する．例を次に示す．

```yaml
experiment:
  id: EXP-0001
  name: n2-d2-smoke-test
  seed: 0

system:
  family: pham
  degrees: [2, 2]
  supports:
    - [[1, 0], [0, 1], [0, 0]]
    - [[1, 0], [0, 1], [0, 0]]
  coefficient_sampler:
    type: complex_uniform
    bound: 5.0

path:
  type: bezier
  degree: 3
  latent_dim: 4
  perturbation_scale: 0.1

tracker:
  max_steps: 50000
  min_step_size: 1.0e-12
  extended_precision: false

reward:
  rejected_step_weight: 1.0
  failure_penalty: 3000.0
  scale: 1.0

training:
  total_timesteps: 10000
  learning_rate: 3.0e-4
  rollout_steps: 256

output:
  root: outputs/EXP-0001
```

## 3．学習コード

`train.py`は次を行う．

1．YAMLを読み込む．
2．設定値を型付きconfigへ変換する．
3．多項式系仕様を生成して検証する．
4．Juliaバックエンドを初期化する．
5．環境を作成する．
6．PPOを学習する．
7．定期的にチェックポイントと評価結果を保存する．
8．実行時設定，Git commit，環境情報を出力先へコピーする．

## 4．評価データセット

学習時とは独立した固定シードで目的係数を生成し，評価データセットとして保存する．少なくとも次を分離する．

- training distribution
- validation instances
- test instances

評価時に毎回係数を再生成する場合でも，生成シードとサンプル順序を固定する．

## 5．比較対象

初期評価では，次を比較する．

1．線形パス
2．ランダムな潜在行動によるベジェパス
3．学習済み方策によるベジェパス

必要に応じて，制御点を問題ごとに直接最適化したoracle的手法を追加し，方策の性能上限を調べる．

oracleは`src/homotopy_path_learning/evaluation/oracle.py`（探索と集計）と`experiments/multivariate_pham/oracle.py`（CLI）で実装する．方策を学習せず，評価問題ごとに潜在行動の箱$[-a_{\max},a_{\max}]^{(d_b-1)m}$を直接探索する．探索は，ゼロ行動（線形パス），一様乱数サンプル，(1+1)進化戦略（1/5成功則）の順に行う．評価問題は学習済み方策の評価と同じ固定シードを用い，問題ごとに方策のコストと比較できるようにする．探索回数は有限なので，oracleの改善量は到達可能な改善の下界である．

さらに次の比較対象を追加する．

- 定数行動ベースライン：全問題に共通の1つの潜在行動を，訓練問題（評価問題とは別のシード）の平均コストを最小化するように上記と同じ探索で求める．`experiments/multivariate_pham/oracle.py --mode constant`で実装し，評価問題上のコストを`ConstantBezier`として報告する．問題ごとの適応が必要かどうかを切り分けるために用いる．
- oracleを教師とした教師あり学習（behavior cloning）：別シードの問題に対するoracleの観測と最良行動の組を教師データとし，`ActorCritic.actor_mean`を$\tanh$で有界化した出力との二乗誤差で学習する（`src/homotopy_path_learning/rl/behavior_cloning.py`，CLIは`experiments/multivariate_pham/behavior_cloning.py`）．検証損失が最小のエポックの重みを採用し，通常のチェックポイントとして保存するので，既存の評価コードでそのまま評価できる．

推論時の制約として，学習済み方策を用いる手法では推論時は方策の順伝播1回と追跡1回のみとし，問題ごとの探索や追加の追跡を行わない．oracleは性能上限を測るための参照であり，この制約を満たさない．

## 6．評価指標

問題単位で次を集計する．

- 全パス追跡成功率
- パス単位成功率
- 総受理ステップ数
- 総棄却ステップ数
- 平均パスコスト
- 最大パスコスト
- 残差ノルム
- 実行時間

データセット全体では，平均，中央値，標準偏差，最小値，最大値，分位点を出力する．

## 7．出力構成

1回のrunの出力は`outputs/EXP-XXXX/<run-id>/`に次の構成で保存される
（Phase 6実装時点．`docs/design/architecture/data_flow.md`と同じ）．

```text
outputs/EXP-XXXX/<run-id>/
├── config.yaml
├── metadata.json
├── latent_basis.npy
├── train_metrics.csv
├── evaluation_seeds.json
├── checkpoints/
│   ├── last.pt
│   ├── best.pt
│   └── step_XXXXXXXX.pt
├── evaluation/evaluation.csv
├── benchmark/benchmark.csv
├── benchmark/benchmark_summary.csv
└── analysis/summary.md
```

`outputs/`は原則としてGit管理対象外とする．論文や報告に使用する小規模な集計結果だけを，必要に応じて別途管理する．
