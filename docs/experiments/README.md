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

### EXP-0029：n5-d33333でのoracleの探索の強化

n5-d33333の20問題で，多点スタート（5回）と長い探索（予算10000）により，oracleの値がどれだけ改善するかを調べる．

```text
実行設定:  experiments/multivariate_pham/configs/exp-0028-n5-d33333-db2.yaml
実験記録:  docs/experiments/EXP-0029-oracle-search-n5-d33333/
生成物:    outputs/EXP-0029/<run-id>/（実行スクリプトはoutputs/EXP-0029/scripts/）
```

### EXP-0030：教師データの質と量（多点スタートのoracle，n4-d2222）

多点スタートのoracleで作った2000件，1回の探索の2000件，1回の探索の1万件の教師データで，attention方策の教師あり学習と微調整の結果を比べる．

```text
実行設定:  experiments/multivariate_pham/configs/exp-0017-n4-d2222-db2.yaml
実験記録:  docs/experiments/EXP-0030-label-quality-vs-quantity/
生成物:    outputs/EXP-0030/<run-id>/（実行スクリプトはoutputs/EXP-0030/scripts/）
```

### EXP-0031：n5-d22222のパイプライン（MLPとattention）

n5-d22222（$d_b=2$）で，EXP-0028の教師データを使って，MLPとattention方策の教師あり学習と微調整を行い，試験問題で比べる．外部サーバで実行する．

```text
実行設定:  experiments/multivariate_pham/configs/exp-0028-n5-d22222-db2.yaml
実験記録:  docs/experiments/EXP-0031-n5-d22222-pipeline/
生成物:    外部サーバのoutputs/EXP-0031/<run-id>/（実行スクリプトはoutputs/EXP-0031/scripts/）
```

### EXP-0032：多点スタートの教師データ1万件（n4-d2222）

n4-d2222の1万問題で，探索seed 0〜4の5回のうち最良をとった教師データを作り，attention方策の教師あり学習と微調整の結果を，1回の探索の1万件（EXP-0020）と比べる．

```text
実行設定:  experiments/multivariate_pham/configs/exp-0017-n4-d2222-db2.yaml
実験記録:  docs/experiments/EXP-0032-multistart-labels-10k/
生成物:    outputs/EXP-0032/<run-id>/（実行スクリプトはoutputs/EXP-0032/scripts/）
```

### EXP-0033：線形パスで難しい問題と到達できる上限（n5）

n5-d22222とn5-d33333で，多点スタートの強い探索（$d_b=2$と$3$）により，今の設定で到達できる上限を測る．あわせて，線形パスの$J$が上位10%の問題の分布で，n5-d22222のパイプラインを行い，通常の分布で学習した方策と比べる（n5-d33333は条件つき）．外部サーバで実行する．

```text
実行設定:  experiments/multivariate_pham/configs/exp-0028-n5-d22222-db2.yaml，exp-0028-n5-d33333-db2.yaml，
           exp-0033-n5-d22222-db3.yaml，exp-0033-n5-d33333-db3.yaml
実験記録:  docs/experiments/EXP-0033-hard-problems-n5/
生成物:    外部サーバのoutputs/EXP-0033/<run-id>/（実行スクリプトはoutputs/EXP-0033/scripts/）
```

### EXP-0034：n5-d33333のパイプライン（attention）

変数5つ，次数3の系（n5-d33333）で，EXP-0028の教師データ1万件からattention方策の教師あり学習と微調整まで通し，パイプラインが通用するかを確かめる．EXP-0033の難しい問題の分布との比較の基準にもなる．外部サーバで実行する．

```text
実行設定:  experiments/multivariate_pham/configs/exp-0028-n5-d33333-db2.yaml
実験記録:  docs/experiments/EXP-0034-n5-d33333-pipeline/
生成物:    外部サーバのoutputs/EXP-0034/<run-id>/（実行スクリプトはoutputs/EXP-0034/scripts/）
```

### EXP-0035：方策を初期値にした探索による上限の目安の測り直し（n5）

EXP-0033の上限の目安を，微調整後のattention方策の行動を初期値にした探索で測り直し，今の設定で50%削減に届くかを改めて判定する．`oracle.py`に方策の行動を初期値にする機能を追加する．外部サーバで実行する．

```text
実行設定:  experiments/multivariate_pham/configs/exp-0028-n5-d22222-db2.yaml，exp-0028-n5-d33333-db2.yaml
実験記録:  docs/experiments/EXP-0035-policy-initialized-bound-n5/
生成物:    外部サーバのoutputs/EXP-0035/<run-id>/（実行スクリプトはoutputs/EXP-0035/scripts/）
```

### EXP-0036：n5-d33333の微調整の延長（2000反復→4000反復）

EXP-0034のattention方策の微調整を，同じ設定でさらに2000反復続け，試験500問題で2000反復の方策と比べる．外部サーバで実行する．

```text
実行設定:  experiments/multivariate_pham/configs/exp-0028-n5-d33333-db2.yaml
実験記録:  docs/experiments/EXP-0036-finetune-extension-n5-d33333/
生成物:    外部サーバのoutputs/EXP-0036/<run-id>/（実行スクリプトはoutputs/EXP-0036/scripts/）
```

### EXP-0037：方策＋探索による教師データの作り直し（第1周，n5）

n5-d22222とn5-d33333で，微調整後の方策の行動を初期値にした探索（予算100×3方策）で教師データを作り直し，教師あり学習と微調整をやり直して，これまでの方策と比べる．外部サーバで実行する．

```text
実行設定:  experiments/multivariate_pham/configs/exp-0028-n5-d22222-db2.yaml，exp-0028-n5-d33333-db2.yaml
実験記録:  docs/experiments/EXP-0037-policy-search-labels-n5/
生成物:    外部サーバのoutputs/EXP-0037/<run-id>/（実行スクリプトはoutputs/EXP-0037/scripts/）
```

### EXP-0038：方策＋探索による教師データの作り直し（第1周，n4-d3333）

EXP-0037と同じ手順（微調整後の方策を初期値にした探索，予算100×3方策）で，n4-d3333の教師データを作り直し，教師あり学習と微調整をやり直して，EXP-0023の方策と比べる．メインの環境で実行する．

```text
実行設定:  experiments/multivariate_pham/configs/exp-0018-n4-d3333-db2.yaml
実験記録:  docs/experiments/EXP-0038-policy-search-labels-n4-d3333/
生成物:    outputs/EXP-0038/<run-id>/（実行スクリプトはoutputs/EXP-0038/scripts/）
```

### EXP-0039：$d_b=2$での標準パイプライン（n2-d22，n2-d33，n3-d333）

報告の表をベジェ次数$d_b=2$でそろえるため，n2-d22，n2-d33，n3-d333で，$d_b=2$の標準パイプライン（ゼロからの探索の教師データ，attentionの教師あり学習，微調整）を実行する．メインの環境で実行する．

```text
実行設定:  experiments/multivariate_pham/configs/exp-0039-n2-d22-db2.yaml，exp-0039-n2-d33-db2.yaml，exp-0039-n3-d333-db2.yaml
実験記録:  docs/experiments/EXP-0039-db2-pipeline-n2-n3/
生成物:    outputs/EXP-0039/<run-id>/（実行スクリプトはoutputs/EXP-0039/scripts/）
```

### EXP-0040：追跡の計算時間（全8系の種類）

報告の表と同じネットワーク（seed 0）と線形パスについて，目的系1つ分の計算時間，ネットワークの計算時間，パス1本あたりの追跡時間，最も長いパスのステップ数を，全8系の種類で計測する．メインの環境で実行する．

```text
実行設定:  各ネットワークのrunの設定（EXP-0024，EXP-0032，EXP-0037，EXP-0038，EXP-0039）
実験記録:  docs/experiments/EXP-0040-runtime-all-systems/
生成物:    outputs/EXP-0040/（実行スクリプトはoutputs/EXP-0040/scripts/）
```

### EXP-0041：固定した目的系の集合での微調整（n4-d3333）

作り直し後の教師あり学習の方策を，学習データの32個の目的系だけで微調整し，微調整が伸びない原因（推定量・歩幅か，目的系間の一貫性か）を切り分ける診断．`finetune.py`に`--train-pool-size`を追加する．メインの環境で実行する．

```text
実行設定:  experiments/multivariate_pham/configs/exp-0018-n4-d3333-db2.yaml
実験記録:  docs/experiments/EXP-0041-fixed-pool-finetune-n4-d3333/
生成物:    outputs/EXP-0041/<run-id>/（実行スクリプトはoutputs/EXP-0041/scripts/）
```

### EXP-0042：教師データの一貫化（n4-d3333）

EXP-0038の3組の教師データの使い方を変える（重心を教師にする／出力に最も近い教師データとの誤差で学習する）．腕の選択は開発集合（9310000〜），最終確認は試験集合で1回だけ行う．メインの環境で実行する．

```text
実行設定:  experiments/multivariate_pham/configs/exp-0018-n4-d3333-db2.yaml
実験記録:  docs/experiments/EXP-0042-label-consistency-n4-d3333/
生成物:    outputs/EXP-0042/<run-id>/（実行スクリプトはoutputs/EXP-0042/scripts/）
```

### EXP-0043：微調整の推定量の改善（n4-d3333）

微調整の摂動対の数$P$，1反復の目的系数$K$，摂動幅$\sigma$（減衰を含む）を変えた腕を比べる．腕の選択は開発集合，最終確認は試験集合で1回だけ行う．メインの環境と外部サーバで実行する．

```text
実行設定:  experiments/multivariate_pham/configs/exp-0018-n4-d3333-db2.yaml
実験記録:  docs/experiments/EXP-0043-finetune-estimator-n4-d3333/
生成物:    outputs/EXP-0043/<run-id>/と外部サーバのoutputs/EXP-0043/（実行スクリプトは各outputs/EXP-0043/scripts/）
```

### EXP-0044：追跡の計算時間（標準の手順と，教師データ作り直し後）

EXP-0040と同じ計測を，全8系の種類の標準の手順のネットワーク（seed 0）と，教師データ作り直し後のネットワーク（n4-d3333，n5-d22222，n5-d33333）で行う．報告の結果1と結果2で，同じ設定のネットワークの計算時間を示すため．メインの環境で実行する．

```text
実行設定:  各ネットワークのrunの設定（EXP-0020，EXP-0021，EXP-0023，EXP-0031，EXP-0034，EXP-0037，EXP-0038，EXP-0039）
実験記録:  docs/experiments/EXP-0044-runtime-standard-and-relabel/
生成物:    outputs/EXP-0044/（実行スクリプトはoutputs/EXP-0044/scripts/）
```

### EXP-0045：複数の目的系をまとめたときのネットワークの計算時間

標準の手順のネットワーク（全8系の種類，seed 0）で，バッチの大きさを1，10，100，500と変えたときの，目的系1つあたりのネットワークの計算時間を測る．メインの環境で実行する．

```text
実行設定:  各ネットワークのrunの設定（EXP-0044の標準の手順のネットワークと同じ）
実験記録:  docs/experiments/EXP-0045-batched-network-time/
生成物:    outputs/EXP-0045/（実行スクリプトはoutputs/EXP-0045/scripts/）
```

### EXP-0046：複数の目的系をまとめたときのネットワークの計算時間（GPU）

EXP-0045と同じ計測を，ネットワークをGPU（RTX A5000）に置いて行う．バッチの大きさごとの1回あたりと目的系1つあたりの時間を測る．メインの環境で実行する．

```text
実行設定:  各ネットワークのrunの設定（EXP-0045と同じ）
実験記録:  docs/experiments/EXP-0046-batched-network-time-gpu/
生成物:    outputs/EXP-0046/（実行スクリプトはoutputs/EXP-0046/scripts/）
```


### EXP-0047：ベジェ次数$d_b=3$の標準の手順（n4-d3333）

n4-d3333で，$d_b=3$の標準の手順（ゼロからの探索の教師データ → attentionの教師あり学習 → 微調整）を，探索の予算500と1000の2つの腕で行い，$d_b=2$の標準の手順（EXP-0023）と比べる．判断は開発集合，報告用の値は試験集合で1回だけ求める．腕B500はメインの環境，腕B1000は外部サーバで実行する．

```text
実行設定:  experiments/multivariate_pham/configs/exp-0018-n4-d3333-db3.yaml
実験記録:  docs/experiments/EXP-0047-db3-pipeline-n4-d3333/
生成物:    outputs/EXP-0047/<run-id>/と外部サーバのoutputs/EXP-0047/（実行スクリプトは各outputs/EXP-0047/scripts/）
```

### EXP-0048：小さい系で学習したネットワークをそのままn6に使う（学習なしの転用）

n5とn4の標準の手順のネットワーク（各3つ）を，学習し直さずにn6-d222222とn6-d333333に転用し，テストデータでの削減率を求める．今後作るn6専用のネットワークと比べる基準にする．n6の目的系の集合（テスト・開発・学習用）もここで予約する．メインの環境で実行する．

```text
実行設定:  experiments/multivariate_pham/configs/exp-0048-n6-d222222-db2.yaml，exp-0048-n6-d333333-db2.yaml
実験記録:  docs/experiments/EXP-0048-zero-shot-transfer-n6/
生成物:    outputs/EXP-0048/<run-id>/（実行スクリプトはoutputs/EXP-0048/scripts/）
```

### EXP-0049：n5から転用したネットワークをn6-d333333で微調整する

EXP-0048でn5-d33333からn6-d333333へ転用したネットワーク（3つ）に，教師データを作らずに標準の手順の微調整だけを行い，転用からの伸びを測る．外部サーバ（seed 0，1）とメインの環境（seed 2）で実行する．

```text
実行設定:  experiments/multivariate_pham/configs/exp-0049-n6-d333333-db2.yaml
実験記録:  docs/experiments/EXP-0049-finetune-transferred-n6-d333333/
生成物:    outputs/EXP-0049/<run-id>/と外部サーバのoutputs/EXP-0049/（実行スクリプトは各outputs/EXP-0049/scripts/）
```

### EXP-0050：ベジェ次数$d_b=3$の標準の手順（n5-d22222，n5-d33333）

EXP-0047（n4-d3333）と同じ$d_b=3$の標準の手順（予算500）を，n5-d22222とn5-d33333で行い，$d_b=2$の標準の手順（EXP-0031，EXP-0034）と比べる．判断は開発集合，報告用の値はテストデータで1回だけ求める．メインの環境と外部サーバで実行する．

```text
実行設定:  experiments/multivariate_pham/configs/exp-0033-n5-d22222-db3.yaml，exp-0033-n5-d33333-db3.yaml
実験記録:  docs/experiments/EXP-0050-db3-pipeline-n5/
生成物:    outputs/EXP-0050/<run-id>/と外部サーバのoutputs/EXP-0050/（実行スクリプトは各outputs/EXP-0050/scripts/）
```

### EXP-0051：ベジェ次数$d_b=3$の標準の手順（残りの系の種類：n2-d22，n2-d33，n3-d222，n3-d333，n4-d2222）

EXP-0050の判断に従い，報告の残りの5つの系の種類にも$d_b=3$を広げる．n2-d22，n3-d222，n4-d2222では$d_b=3$の標準の手順を新たに行い，n2-d33とn3-d333は既存の$d_b=3$のネットワーク（EXP-0026，EXP-0021）を使う．判断は新たに予約する開発集合，報告用の値はテストデータで1回だけ求める．メインの環境で実行する．

```text
実行設定:  experiments/multivariate_pham/configs/exp-0051-n2-d22-db3.yaml，exp-0007-n3-d222-seed0.yaml，exp-0014-n4-d2222.yaml
実験記録:  docs/experiments/EXP-0051-db3-pipeline-remaining/
生成物:    outputs/EXP-0051/<run-id>/（実行スクリプトはoutputs/EXP-0051/scripts/）
```

### EXP-0052：追跡の最大ステップ幅を緩めたときの削減率（学習し直さない評価）

報告の8つの系の種類（$d_b=2$）と$d_b=3$の3つの微調整後のネットワークを，学習し直さずに，追跡の`max_step_size`を0.05から0.1，0.2，0.5，1.0に緩めてテストデータで評価し，線形パスとベジェ曲線のパスのステップ数，削減率，取りこぼした解の数を調べる．メインの環境で実行する．

```text
実行設定:  各runのconfig.yaml（tracker.max_step_sizeだけをスクリプトで差し替える）
実験記録:  docs/experiments/EXP-0052-max-step-size-test/
生成物:    outputs/EXP-0052/（実行スクリプトはoutputs/EXP-0052/scripts/）
```

### EXP-0053：追跡のステップ幅のパラメータ（`TrackerParameters`）を変えたときのステップ数（学習し直さない評価）

報告（2026-10-17）で，n4以上ではステップ幅がほぼ`β_ω_p`と`a`の入った項で決まっていたことを受け，学習し直さずに，HC.jlの`TrackerParameters`と`max_step_size`を変えて，8つの系の種類（$d_b=3$の微調整後）のステップ数，削減率，追跡の失敗，解の取りこぼし，時間を調べる．安全な設定の中から開発集合で選び，テストデータで1回だけ確かめる．`TrackerParameters`を渡せるようにするコードの変更を伴う．メインの環境で実行する．

```text
実行設定:  各runのconfig.yaml（trackerだけをスクリプトで差し替える）
実験記録:  docs/experiments/EXP-0053-tracker-parameters/
生成物:    outputs/EXP-0053/（実行スクリプトはoutputs/EXP-0053/scripts/）
```

### EXP-0054：予測子に高次の微分を渡したときのステップ数（学習し直さない評価）

追跡用のホモトピーのテイラー係数を3次まで正しく計算して，HC.jlのPadé予測子を本来どおり働かせ，学習し直さずに，ステップ数，削減率，追跡の失敗，解の取りこぼし，時間を調べる．EXP-0053と同じ方法で，安全な設定の中から開発集合で選び，テストデータで1回だけ確かめる．`ModelKit.taylor!`の修正と追跡の設定の追加を伴う．メインの環境で実行する．

```text
実行設定:  各runのconfig.yaml（trackerだけをスクリプトで差し替える）
実験記録:  docs/experiments/EXP-0054-higher-order-predictor/
生成物:    outputs/EXP-0054/（実行スクリプトはoutputs/EXP-0054/scripts/）
```

### EXP-0055：安全係数`β_ω_p`=0.8・上限なしの追跡で，$d_b=3$の標準の手順をやり直す（8つの系の種類）

EXP-0053で選んだ追跡の設定（`β_ω_p`=0.8，`max_step_size`=1.0）で，教師データ・教師あり学習・微調整をやり直す．先に，既存の教師データの係数パスと探索で，解の取りこぼしがないことを学習データの目的系で確かめる．開発集合で今の標準（今の追跡の設定）の削減率と比べて系の種類ごとに判断し，テストデータで1回だけ確かめる．各段階の時間も記録する．メインの環境で実行する（n5-d33333の教師データの一部は外部サーバで作ってよい）．

```text
実行設定:  experiments/multivariate_pham/configs/exp-0055-*-db3-bw08.yaml（8つ）
実験記録:  docs/experiments/EXP-0055-pipeline-beta-omega-0.8/
生成物:    outputs/EXP-0055/<run-id>/（実行スクリプトはoutputs/EXP-0055/scripts/）
```

### EXP-0056：安全係数`β_ω_p`=0.8・上限なしの追跡を，6変数の系の既存のネットワークで試す

EXP-0048で5変数から6変数へ転用したネットワーク（6変数・各2次と各3次）を出発点に，これまでの追跡の設定と新しい追跡の設定のそれぞれで微調整だけを行う（6変数・各3次のこれまでの設定の微調整はEXP-0049を使う）．教師データは作らない．EXP-0055で見えた「変数が増えるほど，新しい追跡の設定で微調整したときの削減率が保たれやすい」傾向が6変数でも続くかを，開発集合で比べ，テストデータで1回だけ確かめる．6変数・各3次の微調整は外部サーバ，それ以外はメインの環境で実行する（実際には，計算機によってステップ数が変わったため，すべてメインの環境で実行した．記録の7.1節）．

```text
実行設定:  experiments/multivariate_pham/configs/exp-0056-n6-*.yaml（3つ）
実験記録:  docs/experiments/EXP-0056-n6-beta-omega-0.8/
生成物:    outputs/EXP-0056/<run-id>/（実行スクリプトはoutputs/EXP-0056/scripts/）
```

### EXP-0057：追跡をさらに大胆にしたとき，線形パスと学習したパスのどちらが解を取りこぼしやすいか

新しい追跡の設定で微調整したネットワーク（2〜5変数はEXP-0055の「微調整だけをやり直した場合」，6変数はEXP-0056．10の系の種類）を学習し直さずに使い，安全係数`β_ω_p`を0.8，0.5，0.3，0.2，0.1と下げて，線形パスと学習したパスのステップ数，追跡の失敗，解の取りこぼしを数える．同じ設定での取りこぼしの割合の差と，それぞれが取りこぼさない範囲で最も大胆な設定どうしでのステップ数の比較を，開発集合で判断し，テストデータで1回だけ確かめる．メインの環境で実行する．

```text
実行設定:  各runのconfig.yaml（β_ω_pだけをスクリプトで差し替える）
実験記録:  docs/experiments/EXP-0057-beta-omega-safety/
生成物:    outputs/EXP-0057/（実行スクリプトはoutputs/EXP-0057/scripts/）
```

### EXP-0058：Newton法の収束の判定を緩めて追跡を大胆にしたとき，線形パスと学習したパスのどちらが解を取りこぼしやすいか

EXP-0057と同じネットワーク（10の系の種類）を学習し直さずに使い，`β_ω_p`（0.8，0.3）とNewton法の収束の判定の緩さ`a`（0.125〜0.6）を変えた12の設定で，線形パスと学習したパスのステップ数，追跡の失敗，解の取りこぼしを数える．同じ設定での取りこぼしの割合の差と，それぞれが取りこぼさない設定の中でステップ数が最も少ないものどうしの比較を，開発集合で判断し，テストデータで1回だけ確かめる．メインの環境で実行する．

```text
実行設定:  各runのconfig.yaml（β_ω_pとaだけをスクリプトで差し替える）
実験記録:  docs/experiments/EXP-0058-newton-a-safety/
生成物:    outputs/EXP-0058/（実行スクリプトはoutputs/EXP-0058/scripts/）
```
