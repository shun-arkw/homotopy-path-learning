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
EXP-0013-cost-finetune/
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

## 実験用の作業ツリー

実験は，コミット済みのコードから起動する（各CLIは起動時に作業ツリーがクリーンかを検査し，`metadata.json`に`git_dirty`を記録する）．実験の実行中に記録を書くと作業ツリーがdirtyになるため，実験はメインの作業ツリーとは別の，実験専用の作業ツリーから起動する．

```bash
# ホスト側で作成する（git 2.34では相対パスのオプションがないため，リンクを手で相対パスに直す）
git worktree add --detach .run-worktree <commit>
echo ".run-worktree/" >> .git/info/exclude
printf 'gitdir: ../.git/worktrees/-run-worktree\n' > .run-worktree/.git
```

- 起動は`/app/.run-worktree`で行い，`PYTHONPATH=/app/.run-worktree/src`として作業ツリー側のコードを使う．
- 出力は`--output-root /app/outputs`で，従来どおりメインの`outputs/`に書き出す．
- 新しいコミットで実験する場合は，作業ツリーのHEADを`git -C .run-worktree checkout --detach <commit>`で切り替える．

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

### EXP-0013：追跡コストを直接使った微調整

拡張つき教師あり学習の方策を初期値に，同じ問題で対になる摂動の追跡コストから勾配を推定し，方策を微調整する．モデル選択は検証用問題の追跡コストで行う．

```text
実行設定:  EXP-0007の設定を読み込み，CLIで上書き（初期方策はEXP-0012のrun）
実験CLI:   experiments/multivariate_pham/finetune.py
実験記録:  docs/experiments/EXP-0013-cost-finetune/
生成物:    outputs/EXP-0013/<run-id>/（実行スクリプトはoutputs/EXP-0013/scripts/）
```

### EXP-0014：規模拡大（n3-d333，n4-d2222）

EXP-0010〜0013で整えた標準パイプライン（フル潜在oracleのラベル，対称性拡張つきの教師あり学習，追跡コストによる微調整）を，パス数の多い系（n3-d333）と変数数の多い系（n4-d2222）に適用する．

```text
実行設定:  experiments/multivariate_pham/configs/exp-0014-{n3-d333,n4-d2222}.yaml
実験CLI:   experiments/multivariate_pham/oracle.py
           experiments/multivariate_pham/behavior_cloning.py
           experiments/multivariate_pham/finetune.py
実験記録:  docs/experiments/EXP-0014-scale-up/
生成物:    outputs/EXP-0014/<run-id>/（実行スクリプトはoutputs/EXP-0014/scripts/）
```

### EXP-0015：全次元の行動空間でのPPO（n3-d222）

BCと微調整の優位が，全次元の行動空間を用いたことだけで説明できるかを確認するため，n3-d222でPPOを全次元の行動空間（$m=24$）で学習する．

```text
実行設定:  experiments/multivariate_pham/configs/exp-0015-n3-d222-full-seed{0,1,2}.yaml
           （EXP-0007のn3-d222と潜在次元以外は同一）
実験CLI:   experiments/multivariate_pham/train.py，evaluate.py
実験記録:  docs/experiments/EXP-0015-ppo-full-latent/
生成物:    outputs/EXP-0015/<run-id>/（実行スクリプトはoutputs/EXP-0015/scripts/）
```

### EXP-0016：ベジェ次数を変えた標準パイプライン（n3-d222）

n3-d222で，ベジェ次数$d_b$を2と4にしたときのフル潜在oracleを調べ，$d_b=2$では標準パイプライン（ラベル，拡張つきBC，微調整）を実行して$d_b=3$と比べる．

```text
実行設定:  experiments/multivariate_pham/configs/exp-0016-n3-d222-db{2,4}.yaml
           （EXP-0007のn3-d222とベジェ次数以外は同一）
実験CLI:   experiments/multivariate_pham/oracle.py
           experiments/multivariate_pham/behavior_cloning.py
           experiments/multivariate_pham/finetune.py
実験記録:  docs/experiments/EXP-0016-bezier-degree-pipeline/
生成物:    outputs/EXP-0016/<run-id>/（実行スクリプトはoutputs/EXP-0016/scripts/）
```

### EXP-0017：n4-d2222での$d_b=2$の標準パイプライン

n4-d2222で，ベジェ次数$d_b=2$のフル潜在oracleを調べ，標準パイプライン（ラベル，拡張つきBC，微調整）を実行して$d_b=3$（EXP-0014）と比べる．

```text
実行設定:  experiments/multivariate_pham/configs/exp-0017-n4-d2222-db2.yaml
実験CLI:   experiments/multivariate_pham/oracle.py
           experiments/multivariate_pham/behavior_cloning.py
           experiments/multivariate_pham/finetune.py
実験記録:  docs/experiments/EXP-0017-n4-d2222-db2/
生成物:    outputs/EXP-0017/<run-id>/（実行スクリプトはoutputs/EXP-0017/scripts/）
```

### EXP-0018：n4-d3333のoracle改善余地の予備調査（$d_b=2$と$d_b=3$）

n4-d3333（81パス）で，フル潜在oracleによる改善余地を$d_b=2$と$d_b=3$で調べる．

```text
実行設定:  experiments/multivariate_pham/configs/exp-0018-n4-d3333-db{2,3}.yaml
実験CLI:   experiments/multivariate_pham/oracle.py
実験記録:  docs/experiments/EXP-0018-n4-d3333-prelim/
生成物:    outputs/EXP-0018/<run-id>/（実行スクリプトはoutputs/EXP-0018/scripts/）
```

### EXP-0019：教師データの量とMLPの教師あり学習（n4-d2222，$d_b=2$）

教師データを1万件から3万件に増やすと，n4-d2222でのMLPの教師あり学習が改善するかを調べる．

```text
実行設定:  experiments/multivariate_pham/configs/exp-0017-n4-d2222-db2.yaml（EXP-0017と同じ）
実験CLI:   experiments/multivariate_pham/oracle.py
           experiments/multivariate_pham/behavior_cloning.py
           experiments/multivariate_pham/compare_runs.py
実験記録:  docs/experiments/EXP-0019-n4-bc-data-scaling/
生成物:    outputs/EXP-0019/<run-id>/（実行スクリプトはoutputs/EXP-0019/scripts/）
```

### EXP-0020：置換に同変なattention方策とMLPの比較（n4-d2222，$d_b=2$）

ADR-0007の第1段階のattention方策を，EXP-0017と同じ教師データ・学習量で学習・微調整し，新しい試験問題（評価seed 41000〜41499）でMLPと比べる．推論時間も計測する．

```text
実行設定:  experiments/multivariate_pham/configs/exp-0017-n4-d2222-db2.yaml（CLIで--actor-type attention）
実装設計:  docs/design/implementation/attention_policy.md
実験CLI:   experiments/multivariate_pham/oracle.py
           experiments/multivariate_pham/behavior_cloning.py
           experiments/multivariate_pham/finetune.py
           experiments/multivariate_pham/evaluate.py
           experiments/multivariate_pham/compare_runs.py
           experiments/multivariate_pham/timing.py
実験記録:  docs/experiments/EXP-0020-attention-policy/
生成物:    outputs/EXP-0020/<run-id>/（実行スクリプトはoutputs/EXP-0020/scripts/）
```

### EXP-0021：attention方策のn3系への適用（n3-d222，n3-d333）

EXP-0020のattention方策を，MLPの標準パイプラインと同じ教師データでn3-d222（$d_b=2$）とn3-d333（$d_b=3$）に適用し，新しい試験問題でMLPと比べる．

```text
実行設定:  experiments/multivariate_pham/configs/exp-0016-n3-d222-db2.yaml
           experiments/multivariate_pham/configs/exp-0014-n3-d333.yaml
実験記録:  docs/experiments/EXP-0021-attention-n3/
生成物:    outputs/EXP-0021/<run-id>/（実行スクリプトはoutputs/EXP-0021/scripts/）
```

### EXP-0022：attention方策の改善の原因の切り分け（n4-d2222）

パラメータ数をそろえたMLP，系列Transformer（置換の対称性を構造として持たない），小さいattentionと比べ，EXP-0020の改善の原因を切り分ける．

```text
実行設定:  experiments/multivariate_pham/configs/exp-0017-n4-d2222-db2.yaml
実装設計:  docs/design/implementation/attention_policy.md（10節）
実験記録:  docs/experiments/EXP-0022-attention-ablation/
生成物:    outputs/EXP-0022/<run-id>/（実行スクリプトはoutputs/EXP-0022/scripts/）
```

### EXP-0023：n4-d3333の本番パイプライン（attentionとMLP）

n4-d3333（$d_b=2$）で，教師データの生成から微調整までをMLPとattention方策の両方で行い，試験問題で比べる．外部サーバで実行する．

```text
実行設定:  experiments/multivariate_pham/configs/exp-0018-n4-d3333-db2.yaml
実験記録:  docs/experiments/EXP-0023-n4-d3333-pipeline/
生成物:    外部サーバのoutputs/EXP-0023/<run-id>/（実行スクリプトはoutputs/EXP-0023/scripts/）
```

### EXP-0024：attention方策の微調整の延長（2000反復→4000反復）

EXP-0020とEXP-0021のattention＋微調整の方策から，さらに2000反復の微調整を行い，試験問題で比べる．

```text
実行設定:  EXP-0020，EXP-0021と同じ
実験記録:  docs/experiments/EXP-0024-finetune-extension/
生成物:    outputs/EXP-0024/<run-id>/（実行スクリプトはoutputs/EXP-0024/scripts/）
```

### EXP-0025：n4-d3333での小さいattentionと推論時間

n4-d3333で小さいattentionの方策を学習し，EXP-0023の方策とあわせて推論時間を計測する．外部サーバで実行する．

```text
実行設定:  experiments/multivariate_pham/configs/exp-0018-n4-d3333-db2.yaml
実験記録:  docs/experiments/EXP-0025-n4-d3333-small-attention/
生成物:    外部サーバのoutputs/EXP-0025/<run-id>/（実行スクリプトはoutputs/EXP-0025/scripts/）
```

### EXP-0026：attention方策のn2-d33への適用

n2-d33（$d_b=3$）で，MLPの標準パイプラインと同じ教師データでattention方策を学習し，新しい試験問題でMLPと比べる．

```text
実行設定:  experiments/multivariate_pham/configs/exp-0007-n2-d33-seed0.yaml
実験記録:  docs/experiments/EXP-0026-attention-n2-d33/
生成物:    outputs/EXP-0026/<run-id>/（実行スクリプトはoutputs/EXP-0026/scripts/）
```

### EXP-0027：教師あり学習なしでのattention方策の微調整（n4-d2222）

教師データを使わず，初期化したままのattention方策から微調整だけを行い，教師あり学習ありの場合と比べる．

```text
実行設定:  experiments/multivariate_pham/configs/exp-0017-n4-d2222-db2.yaml
実験記録:  docs/experiments/EXP-0027-finetune-without-bc/
生成物:    outputs/EXP-0027/<run-id>/（実行スクリプトはoutputs/EXP-0027/scripts/）
```

### EXP-0028：n5系の予備調査と教師データの生成

n5-d22222とn5-d33333（$d_b=2$）で，線形パスの追跡の安定性とoracleの改善余地を調べ，条件を満たせば教師データ1万件を生成する．外部サーバで実行する．

```text
実行設定:  experiments/multivariate_pham/configs/exp-0028-n5-d{22222,33333}-db2.yaml
実験記録:  docs/experiments/EXP-0028-n5-prelim-labels/
生成物:    外部サーバのoutputs/EXP-0028/<run-id>/（実行スクリプトはoutputs/EXP-0028/scripts/）
```
