# EXP-XXXX：実験名

## 1．基本情報

- 実験ID：EXP-XXXX
- 状態：Planned / Running / Completed / Failed
- 実施日：YYYY-MM-DD
- Git branch：
- Git commit：
- 設定ファイル：`experiments/multivariate_pham/configs/exp-xxxx.yaml`
- 出力先：`outputs/EXP-XXXX/`
- 実行環境：
- CPU：
- GPU：
- Python：
- Julia：
- 乱数シード：

## 2．目的

本実験で確認する内容を記載する．

## 3．仮説

期待する結果と，その根拠を記載する．

## 4．多項式系

- 変数数：$n=$
- 次数：$(d_1,\ldots,d_n)=$
- パス数：$N_{\mathrm{path}}=$
- 候補指数集合：
- 係数分布：
- 学習サンプルの生成方法：
- 評価サンプル数：

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
- ネットワーク構造：

## 8．比較対象

- 線形パス
- ランダムベジェパス
- 学習済みベジェパス

## 9．評価指標

- 問題単位成功率
- パス単位成功率
- 平均追跡コスト
- 受理ステップ数
- 棄却ステップ数
- 最大パスコスト
- 残差ノルム
- 実行時間

## 10．結果

### 集計

| Method | Success rate | Mean cost | Median cost | Std. | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| Linear |  |  |  |  |  |  |
| Random Bézier |  |  |  |  |  |  |
| Learned Bézier |  |  |  |  |  |  |

### 追跡失敗

| Method | Failed problems | Failed paths | Main failure reason |
|---|---:|---:|---|
| Linear |  |  |  |
| Random Bézier |  |  |  |
| Learned Bézier |  |  |  |

## 11．観察

結果から直接確認できる事実を記載する．

## 12．考察

仮説との整合性，改善または悪化の理由，限界を記載する．

## 13．実行中の問題

- 

## 14．結論

本実験から得られた結論を記載する．

## 15．次の実験

- 
