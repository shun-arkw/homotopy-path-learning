# bezier_runtime_bench

Bezier homotopy path tracking の **実行時間削減** を調査するための検証用ディレクトリです。

- **scripts/bezier_hc_ppo とは依存関係なし**（include や import で参照しない）
- **同一サンプル**で Linear（線形パス）と Bezier の tracking_time を比較し、どこで時間が増えているか切り分ける
- ここで `bezier_univar.jl` を改変して Bernstein / 係数和などの実装を変え、計測で効果を確認する
- 原因が分かり有効な変更が決まったら、**scripts/bezier_hc_ppo/bezier_univar.jl** に反映する

## 使い方

1. このディレクトリで Julia を起動（HomotopyContinuation が使える環境であること）
2. **比較用にはスレッド数 1 で実行**（同じサンプルで公平に比較するため）:
   ```bash
   julia -t 1 run_bench.jl [degree] [bezier_degree] [n_runs]
   ```
   例: `julia -t 1 run_bench.jl 160 4 20` … degree=160, bezier_degree=4 で 20 サンプル、Linear と Bezier の両方を計測

3. 出力で **Linear** と **Bezier** の tracking_time (mean/std/min/max) および **Bezier/Linear 比** を確認する
4. **Newton 反復 1 回あたりの時間**で比較する場合: `julia -t 1 bench_time_per_newton.jl [degree] [bezier_degree] [seed] [n_runs]`（反復回数の差を除いた公平な比が出る）
5. **bezier_univar.jl** を編集して再実行し、Bezier の時間がどう変わるか比較する

## ファイル

- **linear_univar.jl** … 線形パス用スタンドアロン版（Bezier と同一 TrackerOptions: automatic_differentiation=1）
- **bezier_univar.jl** … ベジェパス用スタンドアロン版（ここを編集して検証する）
- **run_bench.jl** … 同一 ctrl で Linear（start→target）と Bezier をそれぞれ track し、両方の tracking_time を表示
- **bench_time_per_newton.jl** … 同一 ctrl で Linear と Bezier をそれぞれ n_runs 回 track し、**evaluate_and_jacobian! 呼び出し回数**（≒ Newton 反復数）と総時間から **1 反復あたりの時間（μs）** を算出。Bezier/Linear の時間-per-Newton 比で「反復回数の差」を除いた公平な比較ができる。
- **profile_eval.jl** … トラッカーを使わず「ホモトピー評価だけ」のマイクロベンチ。同一インスタンスで Linear vs Bezier の **1回あたりの評価コスト** を比較し、遅延の原因を数値で示す

## 遅延の原因（profile_eval.jl で確認）

同じインスタンスでも Bezier が線形より遅い主因は、**1回の H(x,t) 評価の重さ**の差です。

- **eval_coeffs0!**（係数更新のみ）: Bezier は線形の **約 5～6 倍**（Bernstein de Casteljau O(d_b²) + 係数の重み付き和 O(d·d_b)。線形は (1-τ)*a+τ*b の O(d) のみ）。
- **evaluate! / evaluate_and_jacobian!**（+ Horner）: Bezier は線形の **約 1.5～1.7 倍**（Horner が共通なので差は縮まる）。
- トラッカーは 1 パスあたりこれらの評価を **数千回** 呼ぶため、1回あたりの差がそのまま全体の tracking_time の差になる。

確認: `julia profile_eval.jl 160 4 100000`

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
