# bezier_runtime_bench

## 本当のボトルネック確認（1パス追跡中の内訳）

- **profile_eval.jl**（上）… トラッカーなしで「1回あたりの評価コスト」を比較。**eval_coeffs0! が 5.75 倍**という結果で、ボトルネックが係数評価であることを示している。
- **Docker 内で実行するもの**（本番と同様 23VIF がロードされる環境）:
  - **profile_track.jl** … **Linear と Bezier を同一 ctrl で**それぞれ複数回（デフォルト 20 回）プロファイルし、両方の Top functions と **Total snapshots の比（Bezier/Linear）** を表示。どこで余分に時間がかかっているかの比較用。4番目の引数で回数指定可。
    ```bash
    julia -t 1 profile_track.jl 80 4 1 [n_profile]
    ```
  - **count_track_calls.jl** … 1回の track のあいだに **eval_coeffs0! / eval_coeffs_k! / poly_only_horner / poly_and_deriv_horner** がそれぞれ何回呼ばれたかをカウント。profile_eval の「1回あたり μs」と掛け合わせると、どこで時間が消えているかの目安になる。
    ```bash
    julia -t 1 count_track_calls.jl 80 4 1
    ```

## 学習済みモデルからの問題例（jsonl.gz）を直接解析

1. `eval_per_instance.jsonl.gz` から候補を 1 件抽出し、Julia ケースファイルを生成:
  ```bash
   python3 extract_problem_instance.py \
     --input ../../results/bezier_ppo/univar/degree40_bezier3_ep1/omega0.8_tau0.85_strict0.8/run_20260409_170630/eval_per_instance.jsonl.gz \
     --output /tmp/bezier_case.jl \
     --mode best
  ```
2. 生成したケースを解析:
  ```bash
   julia -t 1 scripts/bezier_runtime_bench/analyze_problem_instance.jl "tmp_case_best.jl" 20 100000 [warmup]
  ```
3. 出力で以下を確認:
  - `Repeated runs`: 各 `i` で `compute_newton_iters=true`（step attempts / Newton 記録）→ `false`（追跡時間・呼び出し回数・内訳タイム）のペア。`warmup` 回（デフォルト 1）は計測前に `false` のみ実行。表示は主に 1 実行あたり平均（Linear / Bezier / ratio）。
  - `time per Newton` 比（`sum(false の追跡時間) / sum(true の Newton)`）
  - `Bezier call counts`（どの API が多く呼ばれるか）
  - `Micro eval`（1 回あたりコスト差）



## 結論
今回抽出した問題例では、Bezier は Linear より **Newton反復回数（および step attempts）が少ない**にもかかわらず、実行時間が増加する原因は、  
**1 Newton あたりの処理時間が大きいこと**にある。

## 観測結果（要点）
- Newton反復回数比（Bezier/Linear）は **1未満**（= Bezierの方が少ない）
- 一方で `time per Newton` 比は **1より大きい**（= Bezierの方が1反復が重い）
- 実行時間比は、
  - `time per Newton の増加` × `反復回数の減少`
  の積でほぼ説明でき、実測値と整合した

## 何が重いか
分解計測から、Bezier側で特にコスト増が大きいのは次の処理:
- 係数更新（`eval_coeffs0!`）
- 予測子側（`taylor! k=1` / `eval_coeffs_k!`）

つまり、**反復回数は減らせているが、各反復の単価が高い**ため、総時間で不利になるケースが発生している。

## まとめ
原因は「収束回数」ではなく「1反復あたり計算コスト」の増加。  
したがって、今後の改善の主眼は **Newton反復数の維持/削減に加えて、1反復コストの削減**（特に係数更新系）に置くべきである。



```
Analyze extracted case: degree=40 bezier_degree=3 n_runs=20 warmup=1 threads=1

## Repeated runs (n_runs=20, warmup=1; each i: true=steps/Newton only, false=time+counts)
  Mean tracking time per run (s): Linear 0.001388  Bezier 0.002156  ratio 1.5533 (Bezier/Linear)
  Mean step attempts (true runs): Linear 1730.0  Bezier 1615.0  ratio 0.9335 (Bezier/Linear)
  Mean Newton iterations (true runs): Linear 6485.0  Bezier 6179.0  ratio 0.9528 (Bezier/Linear)
  Mean evaluate_and_jacobian calls per false run: Linear 6605.0  Bezier 6299.0  ratio 0.9537 (Bezier/Linear)
  time per Newton (μs): Linear 0.2101  Bezier 0.3423  ratio 1.6296

## Linear call counts (all solution paths tracking)
  tracking_time_sec = 0.001398
  evaluate_and_jacobian     6605

## Bezier call counts (all solution paths tracking)
  tracking_time_sec = 0.002175
  eval_coeffs0              6339
  evaluate_and_jacobian     6299
  poly_and_deriv_horner     6299
  eval_coeffs_k             3609
  poly_only_horner          40

## Micro eval (same extracted ctrl) n_evals=100000
  eval_coeffs0! only (μs):      Linear 0.0175  Bezier 0.0269  ratio 1.541
  evaluate_and_jacobian! (μs):  Linear 0.0912  Bezier 0.1017  ratio 1.115
  taylor! k=1 (μs):             Linear 0.0612  Bezier 0.1461  ratio 2.387
```