# Codex向け実装計画

## 1．目的

既存の一変数ベジェホモトピー最適化コードを参考にしつつ，多変数多項式系を主対象とする再利用可能な実装を新しいディレクトリ構成で作成する．最初の対象は，固定したPham型サポートをもち，先頭係数を$1$に固定した$n$変数$n$本の正方多項式系である．

## 2．参照対象

既存実装は，次のブランチを参照する．

```text
https://github.com/shun-arkw/homotopy-path-learning/tree/experiment/ai4math2026
```

既存コードを直接大規模に書き換えるのではなく，再利用可能な処理を新構成へ移植する．一変数コードの動作を壊さない．

## 3．実装上の優先順位

1．正しさ
2．テスト可能性
3．PythonとJuliaの責務分離
4．再現性
5．性能

初期段階では，過度な最適化やスレッド並列化を避ける．

## 4．実装フェーズ

### Phase 0：現行コードの保存

- 現在のブランチの動作確認手順を記録する．
- 一変数実験を再現できるタグまたはブランチを保存する．
- 新しい多変数化ブランチを作成する．

### Phase 1：ディレクトリとPythonデータモデル

作成対象は次である．

```text
src/homotopy_path_learning/systems/
src/homotopy_path_learning/paths/
src/homotopy_path_learning/backends/
src/homotopy_path_learning/envs/
src/homotopy_path_learning/rl/
src/homotopy_path_learning/evaluation/
tests/
```

実装内容は次である．

- `PolynomialSystemSpec`
- Pham型サポートの構築と検証
- 開始係数ベクトル生成
- 複素一様係数サンプラー
- 開始解のPython参照実装
- 複素係数と実数ベクトルの相互変換

受入条件は，Python単体テストが成功することである．

### Phase 2：ベジェ制御点

実装内容は次である．

- 線形補間制御点
- 潜在基底$U$
- 潜在行動から複素摂動への変換
- 先頭係数固定マスク
- 制御点のshapeと不変条件の検証

受入条件は，ゼロ行動で厳密な直線パスになり，すべての制御点で先頭係数が$1$となることである．

### Phase 3：Juliaプロジェクトと多変数ホモトピー

作成対象は次である．

```text
julia/Project.toml
julia/src/HomotopyPathLearning.jl
julia/src/system_spec.jl
julia/src/bernstein.jl
julia/src/bezier_homotopy.jl
julia/src/start_solutions.jl
julia/src/tracking.jl
julia/src/api.jl
julia/test/
```

実装内容は次である．

- 多項式系仕様
- ベジェ係数評価
- `AbstractHomotopy`実装
- 多項式値評価
- ヤコビ行列評価
- $t$方向Taylor係数
- 開始解の直積生成
- 逐次全パス追跡

受入条件は，Julia単体テストおよび$n=2$，$(d_1,d_2)=(2,2)$の追跡テストが成功することである．

### Phase 4：Python–Julia統合

実装内容は次である．

- 型付きバックエンドインターフェース
- `juliacall`による初期化
- 制御点の送信
- `TrackingResult`への変換
- Julia状態のウォームアップとキャッシュ
- エラー分類

受入条件は，同一入力で再現可能な追跡結果が得られ，ゼロ行動が線形ベースラインと一致することである．

### Phase 5：Gymnasium環境

実装内容は次である．

- 1ステップの`BezierPhamEnv`
- 目的係数サンプリング
- 観測生成
- 行動から制御点生成
- 線形コストの計算
- ベジェコストと報酬の計算
- 診断用`info`

初期環境は1エピソードにつき1回だけ`step()`を受け付ける．
`reset()`では目的係数をサンプリングし，ゼロ行動から生成した線形補間制御点を同じJuliaバックエンドで追跡して，
線形コスト`J_lin`をエピソード内へキャッシュする．

観測は目的係数の自由係数のみを用いる．順序は`free_coefficient_mask`で抽出した複素係数に対し，
`[Re(c_free), Im(c_free)]`の実数表現とし，dtypeは`float32`である．
行動shapeは`((bezier_degree - 1) * latent_dim,)`であり，Gymnasium空間では`float32`，
制御点生成時には`float64`へ変換する．

パス単位コストは成功時
`accepted_steps_l + reject_weight * rejected_steps_l`，
失敗時`failure_penalty`とし，全パス平均を`J_lin`または`J_bez`とする．
報酬は`reward_scale * (J_lin - J_bez)`であり，追加のclip，正規化，ボーナスは導入しない．
数値的な追跡失敗は有限の失敗コストとして扱うが，Juliaランタイム障害，返却payload不正，
shape不正，NaN/Infなどのプログラム上の例外は握りつぶさない．

受入条件は，環境チェッカーに合格し，ランダム行動で100エピソード程度をクラッシュせず実行できることである．
参照Docker内では次を実行する．

```bash
python3 -m pytest -q -m "not integration and not slow"
python3 -m pytest -q -m "integration and not slow"
python3 -m pytest -q -m "integration and slow"
```

### Phase 6：PPOと実験CLI

実装内容は次である．

- 既存PPOコードの環境非依存化
- YAML設定
- `train.py`
- `evaluate.py`
- `benchmark.py`
- `analyze.py`
- チェックポイントとメタデータ保存

PPO実装は`BezierPhamEnv`を直接importしない環境非依存のPyTorch実装とする．
初期Phase 6では単一のGymnasium環境を逐次的に使用し，`num_envs == 1`のみを許可する．
行動分布は有界Box行動向けの`tanh`変換付き対角正規分布を用い，action clippingは使用しない．
報酬clip，報酬正規化，観測正規化は導入しない．

YAML設定は`experiment`，`system`，`sampler`，`path`，`tracker`，
`environment`，`ppo`，`evaluation`，`output`のグループを持つ．
未知キー，必須キー欠落，不正shape，不正batch設定は読込時に拒否する．

出力は`outputs/<experiment-id>/<run-id>/`に保存し，少なくとも
`config.yaml`，`metadata.json`，`latent_basis.npy`，`train_metrics.csv`，
`evaluation_seeds.json`，`checkpoints/last.pt`，`checkpoints/best.pt`，
`evaluation/evaluation.csv`，`benchmark/benchmark.csv`，
`benchmark/benchmark_summary.csv`，`analysis/summary.md`を生成する．
checkpointにはmodel/optimizer state，PPO設定，観測・行動shape，行動範囲，
多項式系仕様，ベジェ次数，潜在次元，basis seed，Git commit，PyTorch version，
format versionを含める．

固定評価データは`evaluation.seed + instance_index`で生成したseed列として保存する．
`evaluate.py`はLinearとLearnedBezierを，`benchmark.py`はLinear，RandomBezier，
LearnedBezierを同じ目的係数で比較する．

受入条件は，smoke設定で短時間学習が完了し，評価CSVと集計Markdownが生成されることである．
smoke実行は次の4 CLIで行う．

```bash
python3 experiments/multivariate_pham/train.py --config experiments/multivariate_pham/configs/exp-0004-smoke.yaml --run-id <run-id> --device cpu
python3 experiments/multivariate_pham/evaluate.py --run-dir outputs/EXP-0004/<run-id> --checkpoint outputs/EXP-0004/<run-id>/checkpoints/last.pt
python3 experiments/multivariate_pham/benchmark.py --run-dir outputs/EXP-0004/<run-id> --checkpoint outputs/EXP-0004/<run-id>/checkpoints/last.pt
python3 experiments/multivariate_pham/analyze.py --run-dir outputs/EXP-0004/<run-id>
```

### Phase 7：性能改善

正しさ確認後に，次を段階的に導入する．

- 単項式値のキャッシュ
- 変数べきの事前計算
- Julia追跡バッファの再利用
- 複数パスのスレッド並列化
- 線形コストのキャッシュ
- 評価データのバッチ処理

各最適化の前後で数値結果が変化しないことをテストする．
Phase 7では，明示的な性能測定CLIを追加し，baselineとoptimizedの
`performance.json`，`performance.csv`，`equivalence.json`，`summary.md`を保存する．
測定条件は`experiments/multivariate_pham/configs/exp-0008-phase7-performance.yaml`で管理し，
`python3 experiments/multivariate_pham/profile.py --mode baseline|optimized`で実行する．

初期の採用済み最適化は，固定評価時の同一target seedに対する線形追跡重複の削減である．
Gymnasiumの`reset()`/`step()`の意味は変更せず，評価専用のepisode contextで
目的係数，線形制御点，線形追跡結果，`J_lin`を共有する．
Tracker再利用，Python-Julia batch API，threaded mode，Juliaカーネル最適化は，
安全性または実測改善を確認できるまで既定経路へ導入しない．

## 5．Codexへの制約

- 数学仕様を独自に変更しない．
- 先頭係数を時間変化させない．
- 初期実装では$\gamma$-trickを導入しない．
- Python側とJulia側で係数の並び順を統一する．
- Pythonの0始まりとJuliaの1始まりをAPI境界で明示的に変換する．
- 逆行列を明示的に計算しない．
- Juliaの多項式評価とヤコビ行列評価にはテストを付ける．
- 学習が動くことだけを理由に，追跡失敗や例外を握りつぶさない．
- 既存一変数コードを削除しない．

## 6．最初の実験設定

最初のsmoke testは次とする．

$$
n=2,
\qquad
(d_1,d_2)=(2,2),
\qquad
d_b=3
$$

候補指数集合は両方程式について

$$
\mathcal{A}_i
=
\{(1,0),(0,1),(0,0)\}
$$

とする．開始パス数は$4$である．潜在次元は$m=4$程度から開始する．

## 7．完了条件

最初の多変数実装は，次を満たした時点で完了とする．

- 推奨ディレクトリ構成が作成されている．
- PythonとJuliaの全テストが成功する．
- 固定サポートのランダム目的系を生成できる．
- 4本の開始パスを追跡できる．
- ゼロ行動が線形パスを表す．
- PPOの短時間学習が実行できる．
- 学習済み方策と線形パスを固定評価データで比較できる．
- 実験設定，生結果，集計結果を実験ID単位で保存できる．
