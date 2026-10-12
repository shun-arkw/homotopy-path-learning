# Experiment Records

本ディレクトリは，個々の実験の目的，仮説，条件，結果，考察を記録する．設計仕様は`docs/design/`へ記載し，本ディレクトリには実験固有の情報だけを記載する．複数の実験を横断した報告は`docs/reports/`に置く．

## ディレクトリ名

```text
EXP-0005-linear-vs-bezier/
EXP-0006-bezier-degree/
EXP-0007-scale-separation/
EXP-0009-oracle-headroom/
EXP-0010-oracle-imitation/
EXP-0011-symmetry/
EXP-0012-bc-saturation/
```

実験IDを主キーとし，日付は文書内に記載する．

## 記録の構成

```text
EXP-XXXX-name/
├── README.md     （必須）
├── results.md    （任意）
└── notes.md      （任意）
```

- 1実験につき`README.md`1ファイルを基本とする．実験開始前に目的・仮説・設定・評価方法を書き，実験後に結果・観察・考察・結論・次の実験を追記する．
- 同じ実験IDの中で追加検証を行った場合は，既存の節を書き換えず，新しい節（例：「16．追加検証：…」）として末尾に追記する．追加検証の結果が以前の結論を修正する場合も，元の結論は残し，新しい節で修正内容を明記する．
- `results.md`と`notes.md`は，記録が長くなった場合や実行中の時系列メモを分けたい場合に任意で使う（EXP-0004，EXP-0008はこの形式）．
- 新しい実験を開始する際は，`templates/experiment_record.md`をコピーして使用する．

## 対応関係と命名

```text
実行設定:  experiments/multivariate_pham/configs/exp-XXXX-<variant>[-seedK].yaml
実験記録:  docs/experiments/EXP-XXXX-<name>/README.md
生成物:    outputs/EXP-XXXX/<run-id>/
```

- **設定ファイル名**：`exp-XXXX-<variant>[-seedK].yaml`とする（例：`exp-0007-n2-d33-seed1.yaml`）．1つの実験IDに複数の設定ファイルがあってよい．
- **`experiment.id`の一致**：設定ファイル内の`experiment.id`は，ファイル名の`XXXX`と必ず一致させる．出力先`outputs/EXP-XXXX/`はこの値で決まるため，既存の設定ファイルをコピーして作る場合は特に注意し，作成後に次で確認する．

  ```bash
  grep -n "id: EXP" experiments/multivariate_pham/configs/exp-XXXX-*.yaml
  ```

  例外として，`oracle.py`のように既存の設定を読み込み，`--experiment-id`で`experiment.id`を上書きするCLIがある．この場合は設定ファイルを新設せず，上書き後の設定を各runの`config.yaml`に保存し，実験記録に読み込んだ設定ファイルを明記する．

- **run-id**：`exp-XXXX-<variant>[-seedK]-<UTC時刻>`を推奨形式とする（例：`exp-0007-n2-d33-seed1-20260927T145848Z`）．再実行時も同じ形式を使い，`retry`などの接頭辞で区別しない（区別はUTC時刻で行う）．
- **`outputs/`**：Git管理対象外とする．

## 中断したrunの扱い

コンテナ停止やプロセス停止などで中断したrunの出力は，次をすべて確認した後に削除する．

1. 同じ設定の正式runが正常終了している（ログのexit code 0，`GLOBAL_STEP`が`total_timesteps`に到達，`checkpoints/last.pt`が存在）．
2. 中断runが未完了である（`last.pt`がない，または`train_metrics.csv`の最終stepが目標未満）．
3. 実験記録から中断runが参照されていない．

削除するときは対象パスを明示し，ワイルドカードでまとめて消さない．中断が起きた事実と原因は，実験記録の「実行中の問題」節に残す．

## 判断基準と手順

- **再現性**：結論を出す前に，原則として学習seedを3つ（`experiment.seed`と`path.basis_seed`を共変させる）で再現性を確認する．評価seed（`evaluation.seed`，`random_action_seed`）は同じ実験内の比較対象間で揃える．1seedだけの結果は，記録・報告の中でその旨を明記し，「傾向の確認」として扱う．
- **変数を1つずつ変える**：比較では原則として1回に1つの設定だけを変える．複数を同時に変えた場合は，要因が分離できていないことを明記する．
- **事前チェック**：新しい問題設定（系・`bound`・support）で本番の学習を始める前に，目的系を100件程度サンプリングし，線形パス（ゼロ行動）の追跡成功率を確認する．
- **指標の表記**：
  - $\Delta J := (\overline{J}_{\text{method}}-\overline{J}_{\mathrm{lin}})/\overline{J}_{\mathrm{lin}}\times100$（%）．負の値が改善．
  - Positive rate：評価問題のうち，その手法のコストがLinearより小さかった問題の割合．$\Delta J$自体の改善率ではない．
- **並列実行**：参照Dockerで複数の学習を同時に動かす場合の手順は`AGENTS.md`を参照（`OPENBLAS_NUM_THREADS=1`等の指定が必須）．

## 実験の索引

### EXP-0004

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

### EXP-0005：線形パスとの比較

$n=2,(2,2)$で，学習済みベジェパスと線形パスを比較する．`bound`と学習ステップ数の2×2要因分離を含む．

```text
実行設定:  experiments/multivariate_pham/configs/exp-0005-*.yaml
実験記録:  docs/experiments/EXP-0005-linear-vs-bezier/
生成物:    outputs/EXP-0005/<run-id>/
```

### EXP-0006：ベジェ次数の比較

$d_b\in\{2,3,4\}$を，n2-d22（$b=3.0$），n2-d33（$b=1.0$），n2-d33（$b=3.0$）の3つのベースで比較する．

```text
実行設定:  experiments/multivariate_pham/configs/exp-0006-*.yaml
           （d_b=3の一部はexp-0005，exp-0007の設定を再利用）
実験記録:  docs/experiments/EXP-0006-bezier-degree/
生成物:    outputs/EXP-0006/<run-id>/
```

### EXP-0007：変数数拡大と次数拡大の要因分離

次数拡大（n2-d33〜n2-d66）と変数数拡大（n3-d222）を比較し，n3-d222の不安定さの原因（学習量，entropy，ネットワーク容量，観測次元）を調べる．

```text
実行設定:  experiments/multivariate_pham/configs/exp-0007-*.yaml
実験記録:  docs/experiments/EXP-0007-scale-separation/
生成物:    outputs/EXP-0007/<run-id>/
```

### EXP-0008

Phase 7の性能測定は次で管理する．

```text
実行設定:
experiments/multivariate_pham/configs/exp-0008-phase7-performance.yaml

実験CLI:
experiments/multivariate_pham/profile.py

実験記録:
docs/experiments/EXP-0008-phase7-performance/

生成物:
outputs/EXP-0008/<run-id>/
```

EXP-0008は測定駆動の性能改善確認であり，小規模smoke系の結果から
大規模問題全般の性能結論を出さない．

### EXP-0009：oracleによる改善余地の推定

方策を学習せず，評価問題ごとに潜在行動を直接最適化して，改善余地を推定する（n3-d222，n2-d33，n2-d33-dense）．PPOがその改善余地のうちどれだけを得ているか（到達率）を調べる．

```text
実行設定:  EXP-0007の設定を読み込み，experiment.idとpath.latent_dimをCLIで上書き
           （上書き後の設定は各runのconfig.yamlに保存）
実験CLI:   experiments/multivariate_pham/oracle.py
実験記録:  docs/experiments/EXP-0009-oracle-headroom/
生成物:    outputs/EXP-0009/<run-id>/
```

### EXP-0010：oracleを教師とした教師あり学習と診断

oracleの最良行動を教師として方策ネットワークを教師あり学習し（behavior cloning），推論の速い方法としての性能をPPOと比べる．定数行動ベースラインと，最良行動の多峰性も調べる．

```text
実行設定:  EXP-0007の設定を読み込み，CLIでexperiment.idなどを上書き
実験CLI:   experiments/multivariate_pham/oracle.py（教師データ，定数行動）
           experiments/multivariate_pham/behavior_cloning.py
           experiments/multivariate_pham/oracle_modes.py
実験記録:  docs/experiments/EXP-0010-oracle-imitation/
生成物:    outputs/EXP-0010/<run-id>/（実行スクリプトはoutputs/EXP-0010/scripts/）
```

### EXP-0011：対称性の検証と，対称性による拡張つきの教師あり学習

Pham型系の対称性（変数・方程式の置換，1の冪根倍，複素共役）で追跡コストが不変かを数値で確かめる．そのうえで，全次元の行動空間で，対称性による教師データの拡張つきの教師あり学習を行う．

```text
実行設定:  EXP-0007の設定を読み込み，CLIでexperiment.idとpath.latent_dimを上書き
実験CLI:   experiments/multivariate_pham/symmetry_check.py
           experiments/multivariate_pham/oracle.py（--latent-dim full）
           experiments/multivariate_pham/behavior_cloning.py（--augment-symmetry）
実験記録:  docs/experiments/EXP-0011-symmetry/
生成物:    outputs/EXP-0011/<run-id>/（実行スクリプトはoutputs/EXP-0011/scripts/）
```

### EXP-0012：拡張つき教師あり学習の飽和の確認

EXP-0011の拡張つき教師あり学習について，エポック数，ネットワークの大きさ（幅・深さ），教師データ量を1要因ずつ変え，改善の余地を調べる．

```text
実行設定:  EXP-0007の設定を読み込み，CLIで上書き（ラベルはEXP-0011の全次元ラベルを再利用）
実験CLI:   experiments/multivariate_pham/behavior_cloning.py
実験記録:  docs/experiments/EXP-0012-bc-saturation/
生成物:    outputs/EXP-0012/<run-id>/（実行スクリプトはoutputs/EXP-0012/scripts/）
```
