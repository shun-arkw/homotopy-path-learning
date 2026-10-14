# 実装設計：ラベルの一貫化と微調整の改善に必要なコード変更

> 2026-10-15に確定した実装設計（設計の経緯は`docs/design/drafts/2026-10-15-label-consistency-finetune/`）．実装・実験は行っていない．本書は`docs/design/implementation/training_evaluation.md`の追補として書く．記号は本文中で定義する．

## 0．範囲と共通の方針

### 0.1．コード変更が必要な項目と，不要な項目

| 記号 | 内容 | コード変更 | 変更する主なファイル |
|---|---|---|---|
| D4 | 固定した目的系の集合だけで微調整する（診断） | **要** | `experiments/multivariate_pham/finetune.py` |
| A2 | 最近傍ラベル損失（目的系ごとに$K$個のラベル候補のうち，出力に最も近いものとの二乗誤差） | **要** | `src/homotopy_path_learning/rl/behavior_cloning.py`，`experiments/multivariate_pham/behavior_cloning.py` |
| B3 | 微調整の摂動幅$\sigma$の線形減衰 | **要** | `src/homotopy_path_learning/rl/cost_finetune.py`（補助関数），`experiments/multivariate_pham/finetune.py` |
| A1 | 複数の探索runの教師データの重心を教師にする | データ作成スクリプトのみ（学習・微調整は既存のCLI） | `experiments/multivariate_pham/centroid_labels.py`（新規） |
| B1，B2，B4，B5，D4-b，D4-c，D4-c′ | 微調整の$K$（1反復の目的系数），$P$（摂動対の数），$\sigma$，学習率の変更 | **不要**（既存の`--batch-problems`，`--pairs`，`--sigma`，`--learning-rate`で実行できる） | — |

したがって，コード変更の前に始められるのは A1（スクリプトを書いた後）と B1・B2・B4 である．D4-b・D4-cは`--train-pool-size`が要るのでD4の変更を待つ．

### 0.2．共通の方針

- 既存のCLI・checkpoint・設定ファイルの動作は，新しい引数を指定しない限り完全に同じとする（出力ファイルの列と数値も同じ）．これはテストに加えて，6節の「基準線のずれの確認」で実データでも確かめる．
- 不変条件（ゼロ行動＝線形パス，端点と先頭係数の固定）には触れない．行動の箱$[-1,1]^A$（$A$は行動の次元）も変えない．
- 乱数はNumPyの`Generator`か，seedを与えた`torch.Generator`だけを使う．
- `--workers`の値によらず，すべての出力の数値が一致することを保つ（`training_evaluation.md` 9.3節の規則を踏襲する）．
- 単変数の実装には触れない．

記号：目的系の係数ベクトル（観測）を$\boldsymbol{x}$，ネットワークの出力（tanh後の行動）を$\boldsymbol{\mu}(\boldsymbol{x})\in[-1,1]^A$，教師データの行動を$\boldsymbol{a}^*$とする．n4-d3333，$d_b=2$では$A=40$である．

---

## 1．D4：固定した目的系の集合での微調整

### 1.1．目的

微調整の推定量（対称な有限差分）と最適化が，目的系ごとの探索と同じ水準（学習データで$\Delta J\approx-35\%$）まで到達できるかを，汎化の問題と切り離して調べる．そのために，毎反復で新しい目的系を引くのではなく，固定した$N_{\rm pool}$個の目的系を繰り返し使う．

注意（解釈の非対称性）：固定プールでは，1目的系あたりの追跡回数が$2P\times(\text{反復数})$になる（$P=4$，2000反復で約16000回．第1周のラベル探索は1目的系あたり100回，3方策分で300回）．したがって「プールでもラベルに届かない」は推定量・最適化が律速であることの強い証拠だが，「届く」は推定量が十分であることの弱い証拠にすぎない．このため，1目的系あたりの追跡回数がラベルの予算と同程度になる反復でのプールの値も記録する（1.4節）．

### 1.2．CLIの仕様（`finetune.py`）

新しい引数：

- `--train-pool-size N`（整数，既定`None`）．指定すると，訓練の目的系をseed `train_seed, ..., train_seed+N-1`（`--train-seeds-file`があればその先頭$N$個）に固定する．
- 反復$k$（1始まり）の$b$番目（0始まり）の訓練目的系は，プールの$((k-1)K+b)\bmod N$番目とする．$K$は`--batch-problems`．$N\le K$なら，各反復で同じ$N$個が（重複を含めて）$K$回出る．$N=K$を基本とする．
- 制約：`N >= 1`．`--train-seeds-file`と併用する場合，ファイルの長さが$N$未満ならエラー．`--train-pool-size`を指定した場合は，ファイルに$\text{iterations}\times K$個の要素は要求しない（$N$個でよい）．
- `--train-pool-size`を指定しなければ，従来と完全に同じ．

実装は`train_problem_seed(iteration, b)`の分岐を1つ足すだけである：

```python
def train_problem_seed(iteration: int, b: int) -> int:
    index = (iteration - 1) * args.batch_problems + b
    if args.train_pool_size is not None:
        index = index % args.train_pool_size
    return train_seeds[index] if train_seeds is not None else args.train_seed + index
```

`read_seed_list`（重複を拒否する）は変更しない．重複はseedリストではなく，プールの添字の剰余で作る．これで他の呼び出し側（`oracle.py`，`evaluate.py`）の検証は弱まらない．

### 1.3．追跡の再利用（任意）

固定プールでは，各反復で`pool.reset_many`（または`env.reset`）により同じ目的系の文脈（線形パスの追跡を含む）を作り直す．線形パスの追跡1回は候補行動1回分と同じコストなので，$2P=8$評価あたり1回の無駄になる（約11%）．初版では再利用せず，単純さを優先する．再利用する場合は，プールの文脈をプール作成時に1度だけ作り，反復ではその参照を使う．再利用の有無で数値は変わらないこと（線形パスの追跡は決定的）をテストで確かめる．

### 1.4．評価と記録

- 検証は従来どおり`--val-seed`からの200目的系で行い，`val_metrics.csv`の形式は変えない．
- **プール自体での$\Delta J$**を，別ファイル`pool_metrics.csv`（`--train-pool-size`指定時のみ作る）に記録する．列は`iteration, evaluations_per_target, pool_delta_j`．`evaluations_per_target`$=2P\times\text{iteration}$（線形パスの追跡を除く）．
- プールの評価は，反復1〜50では毎反復，それ以降は`val_interval`ごとに行う（プールの評価は$N$回の追跡で，n4-d3333では32×25 ms≈1秒なので負担は小さい）．これにより，1目的系あたりの追跡回数がラベルの予算（100回，300回）と同程度の反復（$P=4$では反復13と38，$P=16$では反復4と10）の値が残る．また，EXP-0041の主統計量「反復1501〜2000の記録点の平均」が，50反復ごとの10点（1550〜2000）から計算できる．
- プールの評価は，その反復の訓練で既に追跡した候補とは別に，出力$\boldsymbol{\mu}$そのものを追跡する．
- `metadata.json`の`finetune`に`"train_pool_size": N`（指定しなければ`null`）と`"train_pool_seeds": [first, last]`を足す．
- 最良の重みの選択は従来どおり検証$\Delta J$で行う（プールでの値は診断用で，選択には使わない）．

### 1.5．テスト

- 単体：`train_problem_seed`相当の関数を切り出し（`finetune.py`内の局所関数をモジュール関数に），`pool_size=None`で従来の値，`pool_size=N`で剰余になること，`N<=K`で各反復が同じ集合になること，`--train-seeds-file`の長さ$N$未満でエラーになることを確かめる（`tests/evaluation/test_seeds_and_selection.py`）．
- 統合（既存の微調整の統合テストに1ケース追加，Docker）：小さい系（n2-d22）で`--train-pool-size 4 --batch-problems 4 --iterations 3`を`--workers 1`と`--workers 2`で実行し，`val_metrics.csv`・`pool_metrics.csv`・`finetune_metrics.csv`（経過時間を除く）・checkpointの重みが一致すること．`--train-pool-size`なしでは`pool_metrics.csv`が作られず，出力が変更前と一致すること（回帰）．

### 1.6．後方互換性

新しい引数の既定値は`None`で，既定では分岐を通らず，新しいファイルも作らない．

---

## 2．A1：重心ラベルのデータスクリプト（コード変更はスクリプトのみ）

### 2.1．目的

同じ目的系に対する複数の探索run（EXP-0038のk0, k1, k2）の行動の**重心**（算術平均）を教師にする教師データrunを作る．教師あり学習（二乗誤差）の出力が実測で重心に近い（距離0.51．ラベル本体までは0.74）ことから，ラベルのばらつきを除いて当てはめ誤差を減らす狙いである．学習と微調整は既存のCLIをそのまま使う．

### 2.2．置き場所と使い方

- `experiments/multivariate_pham/centroid_labels.py`（新規，リポジトリ管理）．`outputs/EXP-0038/scripts/best_of.py`と同じ役割だが，再利用するのでリポジトリに置く．
- 使い方：`python3 experiments/multivariate_pham/centroid_labels.py --out OUT --first 9000000 --num 10000 --run-dir RUN...`．入力runは変更しない．
- 出力`OUT/`は`behavior_cloning.py --label-run-dir`で読める形とする：`config.yaml`（先頭のrunのコピー），`oracle_observations.npy`（形状`(N, obs_dim)`，`float32`），`oracle_actions.npy`（形状`(N, A)`，`float64`），`oracle.csv`，`metadata.json`．seedは`first`から昇順．
- `oracle.csv`の各行は，その目的系の**最良**run（コスト最小）の行を基にする．`oracle_cost`と`delta_j_percent`は最良runの値のまま残し（重心のコストは未知．`metadata.json`に`"oracle_cost_is_best_run": true`と明記），`best_action_l2_norm`・`best_action_max_abs`は重心の値に置き換える．重心のコストを追跡で測るのは本スクリプトの役割ではない．

### 2.3．検証

次を満たさなければ`ValueError`で止める．

- すべてのrunの`config.yaml`が，`_compatibility_key`（`behavior_cloning.py`と同じ項目）で一致する．`latent_basis.npy`があれば全runで一致する．
- 目的系seedごとに，全runに1行ずつある（欠け・重複はエラー）．
- 同じseedの`oracle_observations`が全runで完全に一致し，`linear_cost`が一致する（`best_of.py`と同じ）．
- 重心の各成分が$[-1,1]$の箱に入る（平均なので自動的に入るが，確認する）．

### 2.4．記録（`metadata.json`）

`{"centroid_of": [run...], "n_runs": K, "target_seeds": [first, last], "mean_linear_cost", "mean_best_oracle_cost", "best_delta_j_percent", "mean_pairwise_label_distance", "mean_label_norm", "oracle_cost_is_best_run": true}`．`mean_pairwise_label_distance`（同じ目的系のラベル間のL2距離の平均）は，一貫化の指標として記録する．

### 2.5．テスト

新規`tests/experiments/test_centroid_labels.py`（または`tests/evaluation/test_oracle.py`）：合成した2つの小さいrunディレクトリ（`tmp_path`）から，重心が正しいこと，観測不一致・線形コスト不一致・seed欠落でエラーになること，出力が`load_labels`で読めること．

---

## 3．A2：最近傍ラベル損失

### 3.1．定義

目的系$i$に$K$個のラベル候補$\boldsymbol{a}^*_{i,1},\dots,\boldsymbol{a}^*_{i,K}$があるとき，損失を

$$
\ell_i=\min_{k}\ \frac{1}{A}\bigl\|\boldsymbol{\mu}(\boldsymbol{x}_i)-\boldsymbol{a}^*_{i,k}\bigr\|_2^2
$$

とする（$K=1$なら従来の二乗誤差と一致）．ミニバッチの損失はその平均．勾配は，選ばれた$k$の項だけに流れる（`min`はtorchの`min(dim=1)`で自動微分される）．

**非目標：** 目的系ごとに$K$が異なるデータは扱わない．全目的系で同じ$K$を要求する（EXP-0038は$K=3$で揃っている）．可変の$K$が必要になった場合は別の設計とする．

### 3.2．データの形

- `observations`：形状`(N, obs_dim)`，`float32`（従来どおり）．
- `actions`：形状`(N, K, A)`，`float32`．従来の`(N, A)`は`K=1`として`(N, 1, A)`に変換して扱う．
- `target_seeds`：長さ$N$．同じ目的系の$K$個は同じ行に並べる（別々の行にしない）．

### 3.3．ライブラリ側（`rl/behavior_cloning.py`）

`train_behavior_cloning`の`actions`引数を2次元または3次元で受け付ける：

```python
def train_behavior_cloning(
    agent, observations, actions, *, epochs, learning_rate, batch_size,
    val_fraction, seed, device="cpu", augmentations=None,
    label_reduction: str = "mean",   # "mean" | "min"
) -> tuple[BehaviorCloningResult, torch.optim.Optimizer]:
```

- `actions`が`(N, A)`なら`(N, 1, A)`に直す．`(N, K, A)`で`K >= 1`．それ以外は`ValueError`．
- `label_reduction="mean"`（既定）は$K=1$の従来用途に限る．$K>1$で`mean`はエラー（重心を学ぶならA1のデータスクリプトを使う．役割を分けて混乱を避ける）．
- `label_reduction="min"`：3.1の損失．
- 対称性による拡張：ミニバッチの各サンプルで群の要素$g$を1つ引き，観測に$M_{\rm obs}(g)$，**$K$個すべてのラベルに同じ$M_{\rm act}(g)$**をかける（`einsum("bij,bkj->bki")`）．クリップは従来どおり箱に収め，クリップした成分の割合は$K$個すべてで数える．
- 検証の分割は**目的系（行）単位**で行う（`permutation(n)`は行に対するものなので，$K$個は同じ側に入る）．検証損失も同じ`label_reduction`で計算する．
- `history`の`train_loss`・`val_loss`は`label_reduction`後の値．加えて，`min`のときは各エポックで「前エポックと選ばれた$k$が変わった訓練目的系の割合」を`history`の`switch_fraction`として記録する（観察用．判断基準には使わない）．`train_metrics.csv`にはこの列を`min`のときだけ出す．
- `BehaviorCloningResult`に`label_reduction: str = "mean"`と`n_labels_per_target: int = 1`を足す（既定値つきなので既存の呼び出しは壊れない）．

決定性：`min`の同点（2つの候補との距離が完全に一致）は`torch.min`の最初の添字で決まる．損失値と勾配は添字によらず同じなので，結果は変わらない．

### 3.4．CLI（`experiments/multivariate_pham/behavior_cloning.py`）

- 新しい引数`--label-group DIR1,DIR2,...`（`action="append"`）．1つのグループは「同じ目的系集合に対する$K$個の教師データrun」で，カンマ区切り．複数のグループを足すことができ（例：seed範囲の分割），すべてのグループで$K$が等しいこと．
- `--label-run-dir`（従来）は$K=1$のグループと同じ扱いにする．`--label-run-dir`と`--label-group`の併用は，$K$が異なるのでエラー（両方$K=1$の場合は許す）．
- 新しい引数`--label-reduction {mean,min}`（既定`mean`）．`--label-group`で$K>1$なら`min`を要求する（3.3）．
- グループの読み込み`load_label_group(base_config, dirs) -> (obs, actions(N,K,A), seeds)`：各runを従来の`load_labels`の検証（互換キー，長さ，ファイルの存在）にかけた上で，目的系seedの集合が全runで一致すること（順序が違えばseedで並べ替えてから一致を確かめる），観測が全runで完全に一致すること，線形コストが一致することを確かめる．不一致は`ValueError`．
- `--num-train`は行（目的系）単位で先頭から切る（従来どおり）．
- `metadata.json`の`behavior_cloning`に`"label_groups": [[dir...], ...]`，`"n_labels_per_target": K`，`"label_reduction"`を足す．`"label_run_dirs"`は従来どおり（`--label-run-dir`がなければ空リスト）．

### 3.5．テスト（`tests/rl/test_behavior_cloning.py`）

- `(N, A)`と`(N, 1, A)`で同じ結果（損失の推移・重み）になること．
- `K=2`で，一方の候補を遠くに置いたとき`min`の損失が近い方だけで決まること（手計算との一致）．
- `K=2`で，2候補を同一にしたとき`min`と$K=1$が一致すること．
- 対称性の拡張で，$K$個すべてに同じ要素がかかること（拡張後の候補間の差が，拡張前の差を$M_{\rm act}$で写したものに等しい）．
- 検証分割が行単位であること（`n_train + n_val == N`）．
- $K>1$で`mean`を指定するとエラー．
- CLIの`load_label_group`：観測不一致・seed不一致・$K$がグループ間で異なる場合にエラー，順序違いは並べ替えで通る．

### 3.6．後方互換性

引数を増やすだけで，既定では従来の関数と同一の計算になる．checkpointの形式は変えない（出力は1つのまま）．`train_metrics.csv`の列は`min`のときだけ増える．

---

## 4．B3：微調整の$\sigma$の減衰

### 4.1．定義

反復$k$（1始まり，総数$T$）の摂動幅を

$$
\sigma_k=\sigma_{\rm start}+(\sigma_{\rm end}-\sigma_{\rm start})\frac{k-1}{T-1}\quad(T>1)
$$

とする（線形減衰）．$T=1$なら$\sigma_1=\sigma_{\rm start}$．勾配の推定は$\sigma_k$で割る（候補の生成と推定の両方に同じ$\sigma_k$を使う）．

### 4.2．CLI（`finetune.py`）

- `--sigma`（従来）は$\sigma_{\rm start}$として使う．新しい引数`--sigma-end`（既定`None`）．`None`なら一定（従来）．`--sigma-end`は正で`--sigma`以下を要求（増加は許さない）．
- `--sigma-end`を**指定したときだけ**，`finetune_metrics.csv`に列`sigma`（各反復の$\sigma_k$）を足す．指定しない場合のファイルの列は従来と同じ．
- `metadata.json`の`finetune`に`"sigma_end": value|null`と`"sigma_schedule": "constant"|"linear"`を足す．

### 4.3．ライブラリ側

`rl/cost_finetune.py`に純粋関数を足す：

```python
def sigma_at(iteration: int, *, iterations: int, sigma_start: float, sigma_end: float | None) -> float:
    """Return the perturbation scale of ``iteration`` (1-based) under a linear schedule.

    ``sigma_end=None`` returns ``sigma_start``. Requires ``1 <= iteration <= iterations``,
    ``sigma_start > 0`` and, when given, ``0 < sigma_end <= sigma_start``.
    """
```

推定量（`antithetic_candidates`，`antithetic_gradients_from_costs`）は変更しない．呼び出し側が反復ごとに`sigma=sigma_at(...)`を渡す．

### 4.4．テスト

- `sigma_at`：端点，`None`，`iterations=1`，不正値でのエラー（`tests/rl/test_cost_finetune.py`）．
- 統合：`--sigma-end`指定で`finetune_metrics.csv`に`sigma`列があり線形に減ること．未指定では`sigma`列がなく，出力が従来と一致すること（回帰）．

---

## 5．実装順と工数の目安

| 件 | 変更量 | 工数の目安 | 依存 |
|---|---|---|---|
| A1 | スクリプト1本＋テスト | 半日 | なし（EXP-0042のA1腕の開始条件） |
| D4 | 数十行＋テスト | 半日 | なし（EXP-0041の開始条件） |
| B3 | 数十行＋テスト | 半日 | なし（EXP-0043のB3腕の開始条件．B1・B2・B4は待たない） |
| A2 | ライブラリの形状の一般化＋CLIのグループ読み込み＋テスト | 1〜2日 | なし（EXP-0042のA2腕の開始条件） |

4件は互いに独立である．いずれも実装後に`python3 -m pytest -q`と，Dockerでの統合テストを実行する．その後，6節の確認を行う．

---

## 6．コード変更後の「基準線のずれ」の確認（EXP-0042／Bの腕の実行前）

テストに加えて，実データで既存の手順の出力が変わっていないことを確かめる．D4・B3・A2の変更をすべて取り込んだcommitで行う．

- **(a) 微調整（厳密）：** EXP-0038の第1周BC seed 0のcheckpoint（`exp-0038-n4-d3333-db2-bc-attn-seed0/checkpoints/last.pt`）を初期値に，新しいコードで**新しい引数を使わずに**，元のrunと同じ設定（$K=32$，$P=4$，$\sigma=0.05$，学習率$10^{-4}$，$\lambda=0$，検証seed 9600000，訓練seed 9700000，検証200目的系，50反復ごと）で**200反復**だけ微調整する（`--iterations 200`）．`finetune_metrics.csv`（`elapsed`列を除く）と`val_metrics.csv`（反復0，50，100，150，200の5行）が，元のrun（`exp-0038-n4-d3333-db2-ft-attn-seed0`）の先頭200反復と**完全に一致**すること．CPUでの追跡と乱数の消費は決定的なので，一致しなければコード変更が既存の手順を変えたとみなし，**止めて原因を調べる**．ワーカー数は元のrunと同じ8でよいが，9.3節の規則により数値はワーカー数によらない．
  - 反復50，100，150，200の検証は，`iteration % val_interval == 0`で元のrunでも行われているので，行が対応する．
- **(b) 教師あり学習（緩い確認，GPU）：** 第1周BC seed 0を新しいコードで同じ設定（best-of-3ラベル，300エポック，GPU）で学習し直し，最終（300エポック）の検証損失が元の値0.01917に対して5%以内（0.0182〜0.0201）であること．GPUの計算はビット単位では再現しないので，参考として報告し，小さい差では止めない．5%を超える場合は原因を調べてから進む．
- **(c) 教師あり学習（厳密，CPU）：** A2は`train_behavior_cloning`そのものを変える（ラベルの形状の一般化）ので，(b)より強い確認を行う．第1周BC seed 0と同じ設定（best-of-3ラベル，attention，対称性による拡張，バッチ256，学習率$10^{-3}$，検証10%，seed 0）を，**CPUで3エポック**（`--epochs 3 --device cpu`）だけ，変更前のcommit（`git worktree`で別に取り出す）と新しいコード（新しい引数なし）の両方で学習する．`train_metrics.csv`（3行）と保存した重み（`checkpoints/last.pt`の`model_state_dict`）が**完全に一致**すること．CPUのtorchでこれらの演算（attention，`einsum`，Adam）がビット単位で決定的でないことが分かった場合は，重みの最大絶対差を記録し，$10^{-6}$未満であることを要求する（記録にその旨を書く）．一致しなければ止めて原因を調べる．
  - 評価（`evaluate_checkpoint`）は時間がかかるだけなので，この確認では`--num-instances 1`で済ませてよい．
- 結果（一致の有無，差）はEXP-0042・EXP-0043の記録の「実行前の確認」に書く．

---

## 付録A：温間開始の教師あり学習（予定なし，参考）

> EXP-0042の腕からは外した（初期値とラベル集合を同時に変えるため，効果の帰属ができない）．将来，独立に検証する場合の設計だけ残す．

- `behavior_cloning.py`に`--init-checkpoint PATH`（既定`None`）．指定すると`build_agent`で作ったagentにcheckpointの`model_state_dict`を読み込む．
- 要求：`validate_checkpoint`（観測の形，行動空間，系の仕様，形式版）を通ること．`actor_type`と`attention_kwargs`がCLIの指定と一致すること（不一致は`ValueError`で両方の値を出す）．`--latent-dim`は`full`に固定．
- `train_behavior_cloning`に`include_initial: bool = False`を足し，`True`のとき初期重みの検証損失を`best_val`の初期値とし，`best_state`を初期重みにする（現在は`best_val=inf`から始めるため，初期重みは候補に入らない）．既定`False`では従来と同一．
- `metadata.json`に`"init_checkpoint": {"path", "sha256", "run_id", "git_commit"}`．
- テスト：`include_initial=True`で学習が悪化する設定では`best_epoch=0`で重みが初期値と一致すること．`actor_type`不一致でエラー．

## 付録B：EXP-C向けの近接型の受理規則（優先度低，参考）

探索（`search_best_action`）の受理規則は「候補のコストが現在値以下なら移る」で，コストが等しい平坦部ではランダムウォークになる．これが探索ごとに異なるラベルを生む一因と考えられる．近接型では，候補のコストが現在値と**等しい**ときは，基準点$\boldsymbol{a}_0$（方策出力，`initial_action`）に近い場合だけ移る：

```python
if cost < incumbent_cost or (cost == incumbent_cost and np.linalg.norm(candidate - anchor) < np.linalg.norm(incumbent - anchor)):
    incumbent, incumbent_cost = candidate, cost
```

- 引数`proximal_anchor: np.ndarray | None = None`（形状`(dim,)`）を足す．`None`なら従来と同一．`oracle.py`には`--proximal`（`--init-checkpoint`と併用時のみ）を足し，`initial_action`を基準点にする．
- 乱数の消費順序は変えない（候補の生成は同じ）ので，`proximal_anchor=None`での結果は従来と完全に一致する．
- より強い案（目的に$\lambda\|\boldsymbol{a}-\boldsymbol{a}_0\|^2$を足す）は，`best_cost`が$J$でなくなり`oracle.csv`の意味が変わるので，採らない．
- 中間判定の指標：同じ目的系の3回の探索の移動ベクトルの相互コサイン（現状0.15）と，教師あり学習の検証MSE（現状0.018）．
