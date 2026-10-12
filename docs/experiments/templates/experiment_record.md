# EXP-XXXX：実験名

## 1．基本情報

- 実験ID：EXP-XXXX
- 状態：Planned / Running / Completed / Failed
- 実施日：YYYY-MM-DD
- Git branch：
- Git commit：
- 設定ファイル：`experiments/multivariate_pham/configs/exp-XXXX-<variant>[-seedK].yaml`
  （`experiment.id`がファイル名の`XXXX`と一致していることを確認済み：□）
- 出力先：`outputs/EXP-XXXX/<run-id>/`
- 実行環境：参照Docker`homotopy-continuation`
- CPU：
- GPU：
- Python：
- Julia：
- 学習seed：`experiment.seed`/`path.basis_seed` ∈ {0,1,2}（1seedのみの場合はその旨を明記）
- 評価seed：`evaluation.seed`＝，`random_action_seed`＝（比較対象間で共通）

## 2．目的

本実験で確認する内容を記載する．

## 3．仮説

期待する結果と，その根拠を記載する．

## 4．多項式系

- 変数数：$n=$
- 次数：$(d_1,\ldots,d_n)=$
- パス数：$N_{\mathrm{path}}=$
- 候補指数集合：
- 係数分布：複素一様分布，`bound`$=b=$
- 学習サンプルの生成方法：
- 評価サンプル数：
- 開始系 $G=\{g_1,\ldots,g_n\}$：
  $$g_1=\ldots,\qquad g_2=\ldots$$
- 目的系 $F=\{f_1,\ldots,f_n\}$：
  $$f_1=\ldots$$
  $$f_2=\ldots$$

## 5．係数パス

- パス種別：Bézier
- ベジェ次数：$d_b=$
- 潜在次元：$m=$
- 摂動スケール：
- 先頭係数固定：有効

## 6．追跡設定

- 最大ステップ数：
- 最小ステップ幅：
- 拡張精度：
- 成功判定：
- 失敗ペナルティ：
- 棄却ステップ重み：

## 7．PPO設定

- 総学習ステップ数：
- 学習率：
- ロールアウト長：
- バッチサイズ：
- 割引率：
- GAE係数：
- clipping係数：
- entropy係数：
- ネットワーク構造：

## 8．事前チェック

本番の学習前に，目的系を100件程度サンプリングし，線形パス（ゼロ行動）の
追跡成功率を確認した結果を記載する．

- 線形パス追跡成功率：
- 線形パスのコスト（平均・最大）：

## 9．比較対象

- 線形パス
- ランダムベジェパス
- 学習済みベジェパス

## 10．評価指標

- $\Delta J := (\overline{J}_{\text{method}}-\overline{J}_{\mathrm{lin}})/\overline{J}_{\mathrm{lin}}\times100$（%，負が改善）
- Positive rate（評価問題のうち，その手法のコストがLinearより小さかった問題の割合）
- 問題単位成功率・パス単位成功率
- 受理ステップ数・棄却ステップ数
- 残差ノルム
- 実行時間
- explained variance（価値関数の学習状況）

## 11．結果

### 集計

括弧内は$\Delta J$．

| Method | Success rate | Mean cost | Median cost | Std. | Positive rate |
|---|---:|---:|---:|---:|---:|
| Linear |  |  |  |  | — |
| Random Bézier |  | ( %) |  |  |  |
| Learned Bézier |  | ( %) |  |  |  |

### seed別（複数seedの場合）

| Seed | Linear | Learned Bézier | Positive rate |
|---:|---:|---:|---:|
| 0 |  | ( %) |  |
| 1 |  | ( %) |  |
| 2 |  | ( %) |  |
| 平均 |  | ( %) |  |
| 母標準偏差 | — | （$\Delta J$で pt） | — |

### 追跡失敗

| Method | Failed problems | Failed paths | Main failure reason |
|---|---:|---:|---|
| Linear |  |  |  |
| Random Bézier |  |  |  |
| Learned Bézier |  |  |  |

## 12．観察

結果から直接確認できる事実を記載する．

## 13．考察

仮説との整合性，改善または悪化の理由，限界を記載する．

## 14．実行中の問題

中断したrunがあれば，その事実と原因をここに残す．

- 

## 15．結論

本実験から得られた結論を記載する．1seedのみの結果に基づく場合はその旨を明記する．

## 16．次の実験

- 

<!-- 追加検証は，上の節を書き換えず「17．追加検証：…」のように末尾へ追記する． -->
