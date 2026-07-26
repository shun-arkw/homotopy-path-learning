# ホモトピー追跡の計算量指標と実行時間（ベンチ・メモ）

本稿は `scripts/bezier_runtime_bench/` で行った検証の要約であり、強化学習の報酬設計や論文記述のたたき台とする。細かい実装は README や各スクリプトに委ね、ここでは概念・数式・観測結果を中心に書く。

---

## 1. 強化学習向けの計算量指標

### 1.1 ステップ数からニュートン作業量へ

ホモトピー連続法の追跡では、予測子–修正子法によりパラメータ $t$ に沿って $H(\mathbf{x},t)=\mathbf{0}$ の解曲線を追う。実装では **予測子–修正子の 1 サイクル（ステップ試行 1 回）** を数えることが多いが、学習で抑えたいのはしばしば **数値コアの作業量** に近い量である。

修正子は $t=t_{i+1}$ 固定のもとでのニュートン型反復であり（記法は JSAI 原稿 §3.2 に合わせる）、

$$
\mathbf{x}^{(0)}=\tilde{\mathbf{x}}_{i+1},\qquad
\mathbf{x}^{(j+1)}=\mathbf{x}^{(j)}-J_{H}\left(\mathbf{x}^{(j)},t_{i+1}\right)^{-1}H\left(\mathbf{x}^{(j)},t_{i+1}\right),
\quad J_H=\frac{\partial H}{\partial \mathbf{x}},
$$

を収束まで繰り返す。各修正子呼び出しでの反復回数を $k_i$ とし、**1 本の解路**に対するニュートン反復の総数を

$$
N_{\mathrm{Nw}} := \sum_i k_i
$$

と定義する（複数の開始解を追う場合は、路ごとの $N_{\mathrm{Nw}}$ を合算した総量を同様に扱う）。

**ステップ試行数** $N_{\mathrm{step}}$（受理・棄却の試行を含む）は、パス幾何とアルゴリズムの意思決定が絡み、**修正子あたりの反復負荷**を直接は表さない。一方 $N_{\mathrm{Nw}}$ は、残差とヤコビアンに基づく線形代数の回数と強く結びつくため、**経過時間（追跡に要した計測時間）** に対する単調性が期待しやすい。

### 1.2 ニュートン回数取得が重い理由（本リポジトリの実装）

HomotopyContinuation.jl 上で $N_{\mathrm{Nw}}$ を得るために、`track(...; debug=true)` のログから `iters → …` を集計する経路を取っている（`bezier_univar.jl` / `linear_univar.jl` の `compute_newton_iters=true` 分岐）。このとき次が重なる。

1. **`debug=true`** による詳細ログの **整形・出力**（高頻度の I/O と文字列処理）。
2. 標準出力・標準エラを **パイプにリダイレクト**し、非同期読み出しで **ログ全文を文字列化**してから正規表現で走査するオーバーヘッド。
3. ニュートン回数取得時は **開始解を順に逐次追跡**する一方、通常計測（`compute_newton_iters=false`）では **スレッド並列**で各路を追う。次数 $n$ に開始解が比例する設定では、並列度の差だけで大きな乖離が出る。

すなわち「ニュートン反復の数学的コストが変わる」というより、**ログ付き・逐次・パイプ経由の取得方式**が、学習ループに毎ステップ埋め込むには不向きである。

### 1.3 提案する代理指標：`ModelKit.evaluate_and_jacobian!` の呼び出し回数

HomotopyContinuation の **ModelKit** では、カスタムホモトピーに対し [カスタムホモトピーのガイド](https://juliahomotopycontinuation.org/guides/custom-homotopy) が示すように、追跡に

- `evaluate!` … $H(\mathbf{x},t)$ の値、
- `evaluate_and_jacobian!` … $H(\mathbf{x},t)$ と $\partial H/\partial \mathbf{x}$、
- `taylor!` … $t$ に関する展開（予測子側）

を実装する。`evaluate_and_jacobian!` は与えられた $(\mathbf{x},t)$ に対し、**残差ベクトル $\mathbf{u}=H(\mathbf{x},t)$** と **ヤコビ行列 $U=\partial H/\partial \mathbf{x}$** を（インプレースで）求める。修正子の各ニュートン反復ではこれが繰り返し必要になるため、**呼び出し回数 $N_{\mathrm{ej}}$** は $N_{\mathrm{Nw}}$ と強く相関する。予測子など他経路でも呼ばれ得るため、厳密等式 $N_{\mathrm{ej}}=N_{\mathrm{Nw}}$ は仮定せず、**実測で比例性を確認する**のがよい。

本ベンチでは `bezier_univar.jl` / `linear_univar.jl` 内の該当メソッド入口で `evaluate_and_jacobian` をカウントしている（`analyze_problem_instance.jl` が `compute_newton_iters=false` の走査で時間・カウントを取る）。

**利点:** デバッグ追跡を回さずに $N_{\mathrm{Nw}}$ に近いスカラーを報酬に使える可能性がある。

---

## 2. ニュートン反復は減っているのに Bézier の方が遅い理由

### 2.1 経過時間の分解

1 本あたりの追跡に要した経過時間を $T_{\mathrm{tr}}$、ニュートン反復総数を $N_{\mathrm{Nw}}$ とし、**反復 1 回あたりの平均経過時間**を

$$
\tau := \frac{T_{\mathrm{tr}}}{N_{\mathrm{Nw}}}
$$

とおく（$N_{\mathrm{Nw}}$ はデバッグ走査、$T_{\mathrm{tr}}$ は非デバッグ走査から `analyze_problem_instance.jl` が対に取る）。粗い積分解として

$$
\frac{T_{\mathrm{Béz}}}{T_{\mathrm{Lin}}}
\approx
\frac{N_{\mathrm{Nw}}^{\mathrm{Béz}}}{N_{\mathrm{Nw}}^{\mathrm{Lin}}}
\cdot
\frac{\tau_{\mathrm{Béz}}}{\tau_{\mathrm{Lin}}}
$$

が成り立つ。反復数比が 1 未満でも $\tau$ の比が大きければ、総時間は Bézier 側が劣る。

### 2.2 観測例（抽出 1 インスタンス）

`extract_problem_instance.py` で生成したケース（`tmp_case_best.jl`）に対し、`analyze_problem_instance.jl` を

`julia -t 1 scripts/bezier_runtime_bench/analyze_problem_instance.jl tmp_case_best.jl 20 100000 2`

で実行したときの **20 回平均**（warmup 2）の要約。列は **Bézier | Linear | 比（Bézier/Linear）**。

| 指標 | Bézier | Linear | 比（Bézier/Linear） |
|------|--------|--------|---------------------|
| ステップ試行数 $N_{\mathrm{step}}$ | 1615.0 | 1730.0 | 0.9335 |
| ニュートン反復総数 $N_{\mathrm{Nw}}$ | 6179.0 | 6485.0 | 0.9528 |
| `evaluate_and_jacobian` 呼び出し回数 $N_{\mathrm{ej}}$ | 6299.0 | 6605.0 | 0.9537 |
| 追跡の平均経過時間 $T_{\mathrm{tr}}$（s） | 0.003131 | 0.002333 | 1.3417 |
| $\tau = T_{\mathrm{tr}}/N_{\mathrm{Nw}}$（μs） | 0.5067 | 0.3598 | 1.4081 |

$N_{\mathrm{ej}}$ と $N_{\mathrm{Nw}}$ の比はほぼ一致しており、**代理指標としての $N_{\mathrm{ej}}$** の整合性が示唆される。一方、$\tau$ の比は 1 を大きく超えるため、**反復単価の増**が総時間の不利を説明する。

### 2.3 直感的な理由（Bézier が $\tau$ を押し上げる要因）

同一の追跡器設定のもとで、Bézier ホモトピーはパラメータ $t$（実装上は $\tau$）に応じて **係数ベクトルを都度更新**する。線形パスは端点の凸結合で済むのに対し、Bézier では Bernstein 重みによる結合など **係数更新（実装上 `eval_coeffs0!` 周り）** のコストが大きくなりやすい。予測子の `taylor!` でも係数の高階情報が絡み、**1 回の `evaluate_and_jacobian!` 内部**でもその前処理が乗る。マイクロベンチ（同一制御点で多数回評価）では、`eval_coeffs0!` 単体や `taylor!`（$k=1$）で Bézier/線形の比が開き、`evaluate_and_jacobian!` 全体では Horner 部分で比がやや縮まる、という内訳が `analyze_problem_instance.jl` の出力でも確認できる。

---

## 3. 今後の実験（任意）

- 複数シード・複数インスタンスで $(N_{\mathrm{Nw}}, N_{\mathrm{ej}})$ の散布図と相関係数を報告し、代理指標の妥当性を定量化する。
- デバッグ追跡の経過時間と、通常追跡＋$N_{\mathrm{ej}}$ 計測のコストを対比し、学習パイプライン上の実用性を補強する。

---

## 参照（リポジトリ内）

- `README.md` … スクリプト一覧と実行例
- `analyze_problem_instance.jl` … 抽出ケースに対する反復数・$N_{\mathrm{ej}}$・経過時間・マイクロ評価のまとめ
- `extract_problem_instance.py` … 評価ログからケース用 Julia ファイルを生成
- `bezier_univar.jl` / `linear_univar.jl` … ホモトピー定義、`evaluate_and_jacobian!`、ニュートン回数取得分岐
