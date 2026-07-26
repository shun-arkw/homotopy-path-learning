# テスト計画

## 1．方針

多変数化では，PPOよりも多項式表現，係数添字，ヤコビ行列，Taylor係数，Python–Julia間の配列変換にバグが入りやすい．学習を開始する前に，決定的な小規模テストを完了させる．

## 2．Python単体テスト

### 多項式系仕様

- Pham型サポートから`exponents`と`offsets`が正しく生成される．
- 各方程式の先頭指数と定数指数が正しい．
- 不正な次数，不正な指数，定数項欠落を拒否する．
- 先頭単項式と重複する指数を拒否する．

### 係数生成

- 開始係数が$x_i^{d_i}-1$を表す．
- 目的係数の先頭成分が常に$1$である．
- 同じseedで同じ係数を生成する．

### ベジェ制御点

- ゼロ行動で線形補間制御点になる．
- 端点が変化しない．
- 先頭係数が全制御点で$1$である．
- action shape不正を拒否する．

### Gym環境

- 観測shapeが`observation_space`と一致する．
- 初期観測は目的係数の自由係数を`[Re(c_free), Im(c_free)]`の順に並べた`float32`配列である．
- 行動shapeが`action_space`と一致する．
- 行動shapeは`((bezier_degree - 1) * latent_dim,)`であり，有限値かつ`action_space`の範囲内である．
- 1ステップ後に終了する．
- `reset()`時にゼロ行動の線形ベースラインを1回だけ追跡し，`step()`では再追跡しない．
- ゼロ行動の制御点が線形補間制御点と一致し，報酬が数値誤差の範囲で0となる．
- パス単位コストは成功時`accepted + reject_weight * rejected`，失敗時`failure_penalty`である．
- 平均コスト`J_lin`および`J_bez`から`reward_scale * (J_lin - J_bez)`を計算する．
- 数値的な追跡失敗は有限コストとして扱い，プログラム上の例外は握りつぶさない．
- `info`に必須フィールドが含まれる．
- Gymnasium checkerを`check_env(env, skip_render_check=True)`で実行する．
- 実Juliaバックエンドでランダム行動100エピソードのsmoke testを実行する．

## 3．Julia単体テスト

### 端点

任意の$\boldsymbol{x}$について，

$$
H(\boldsymbol{x},t_{\mathrm{HC}}=1)=G(\boldsymbol{x})
$$

および

$$
H(\boldsymbol{x},t_{\mathrm{HC}}=0)=F(\boldsymbol{x})
$$

を確認する．

### ヤコビ行列

ランダムな$\boldsymbol{x}$と$t$について，解析ヤコビ行列を複素有限差分または自動微分による参照値と比較する．

### Taylor係数

$t$方向の1階から必要階数までのTaylor係数を，有限差分または直接微分したベジェ係数と比較する．特に$\tau=1-t$による奇数階の符号を確認する．

### 開始解

- 開始解数が$\prod_i d_i$である．
- 各開始解の残差が許容誤差以下である．
- 重複する開始解がない．

### 追跡

小規模な既知問題について，全パスが目的解へ到達し，目的系残差が許容誤差以下である．

## 4．Python–Julia統合テスト

- Pythonで構成した制御点がJuliaで同じ端点を生成する．
- Juliaの返却配列shapeがPythonの`TrackingResult`と一致する．
- ゼロ行動の結果が，専用の直線ホモトピーまたは既存実装と一致する．
- 同じ入力を複数回評価した結果が決定的である．
- Juliaのウォームアップ前後で数値結果が変化しない．

## 5．Phase 6 PPO・実験CLIテスト

- YAML設定を厳密にdataclassへ読み込み，未知キー，必須キー欠落，不正型，不正batch設定を拒否する．
- 有界行動分布が常にaction space内の行動を生成し，log probabilityを再計算できる．
- action clipping，reward clipping，reward normalizationを使用しないことを確認する．
- actor–critic networkの出力shape，state dict round-trip，決定論的行動再現性を確認する．
- 1ステップterminal transitionのGAEを手計算と比較する．
- PPO updateが有限のlossと診断値を返し，パラメータを更新する．
- 偽の1ステップ連続行動環境でrollout収集とcheckpoint保存・読込を確認する．
- 固定評価seedからLinear，RandomBezier，LearnedBezierを同じ目的係数で評価する．
- 評価CSV，benchmark CSV，benchmark summary CSV，summary Markdownを生成する．
- 参照Docker内の実`BezierPhamEnv`でPhase 6 CLI smokeを実行する．

## 6．Phase 7性能・同値性テスト

- 性能計測値が有限かつ非負であることを確認する．
- 空の測定結果を拒否し，mean，median，母標準偏差，p50，p95，
  throughputが手計算と一致することを確認する．
- baselineとoptimizedの評価CSVを，elapsed timeとrun IDを除外して比較する．
- 評価用episode contextの配列コピーを外部で変更しても，環境内部状態が変化しないことを確認する．
- 同一target seedについて，Linear，RandomBezier，LearnedBezierが同じ`J_lin`を共有し，
  線形追跡が1回だけ実行されることを確認する．
- 固定100問題の詳細tracking recordsで，path success，accepted/rejected steps，
  failure code，endpoint，residualに差分がないことを確認する．
- Phase 7後もPhase 6のtrain/evaluate/benchmark/analyzeが成功することを確認する．

## 7．最初のテスト問題

### ケースA

$$
n=2,
\qquad
(d_1,d_2)=(2,2)
$$

$$
f_1=x_1^2+a_{11}x_1+a_{12}x_2+a_{10}
$$

$$
f_2=x_2^2+a_{21}x_1+a_{22}x_2+a_{20}
$$

開始パス数は$4$である．

### ケースB

目的系を開始系と同一にする．すべてのパスの終点は開始点と一致する必要がある．

### ケースC

中間制御点を線形補間上に置き，ベジェ評価が直線補間と一致することを確認する．

## 8．受入基準

PPO学習へ進む前に，次を満たす必要がある．

- Python単体テストがすべて成功する．
- Julia単体テストがすべて成功する．
- ケースAで全4パスを追跡できる．
- 終点残差が設定した許容誤差以下である．
- ゼロ行動と線形ホモトピーの追跡統計が許容範囲で一致する．
- 100個程度のランダム問題でクラッシュしない．
