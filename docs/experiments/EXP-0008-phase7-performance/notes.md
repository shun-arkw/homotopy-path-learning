# EXP-0008 Notes

## ボトルネック

baselineでは，固定100問題の`benchmark`全体が`track_same_control_points`単体より
大きく，評価器が同じtarget seedで線形追跡を繰り返す構造が明確な重複になっていた．

## 採用

- 評価時の線形追跡重複削減を採用した．
- Gymnasiumの`reset()`と`step()`の意味は変更せず，評価専用のepisode contextを追加した．
- contextはPython所有の配列コピーを返し，外部変更で環境内部状態が変わらないようにした．

## 採用しなかった候補

- 単項式評価計画，変数べき事前計算，融合system/Jacobian評価：
  今回の主要改善対象は評価器の重複追跡であり，Juliaカーネルを変更せず同値性を保つ方を優先した．
- 前進差分in-place更新，追跡内部バッファ再利用：
  小規模smokeで主要ボトルネックとして確認できず，API返却結果の独立性リスクに対して効果を確認できなかった．
- Tracker再利用：
  HomotopyContinuation.Trackerのmutable homotopy更新に対する安全性保証をPhase 7内で十分確認できなかったため採用しない．
- Python-Julia batch API：
  境界呼出しそのものより，評価器の線形追跡重複の削減で十分な改善が出たため採用しない．
- threaded mode：
  参照Dockerの`JULIA_NUM_THREADS`は未設定であり，4パスsmokeでは並列化オーバーヘッドが利益を上回る可能性が高いため採用しない．

## 数値同値性

固定100問題の詳細tracking recordsで，path success，tracking step，failure code，
endpoint，residualの差分は0だった．

## 注意

本結果はEXP-0008の小規模smoke測定であり，大規模問題全般の高速化を主張しない．
