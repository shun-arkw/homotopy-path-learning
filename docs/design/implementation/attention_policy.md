# attention方策（ADR-0007 第1段階）の実装設計

## 1．目的と範囲

- ADR-0007で提案した，置換に同変なTransformerエンコーダーの方策を，第1段階として実装する．
- 対象は，すべての方程式の自由な単項式が$\{x_1,\dots,x_n,1\}$（1次の項と定数項）であるサポートに限る（n3-d222，n3-d333，n4-d2222，n4-d3333など）．このとき，目的系の自由係数は，行列$A\in\mathbb{C}^{n\times n}$（$A_{ij}$は方程式$i$の$x_j$の係数）と，定数ベクトル$c\in\mathbb{C}^n$に整理できる．
- 一般のサポート（2次の項などを含む系）は第2段階とし，本段階では`NotImplementedError`で明示的に拒否する．
- 環境，追跡，報酬，行動空間（全次元の潜在行動，基底$U$）は変更しない．推論時の制約（方策の順伝播1回と追跡1回）も変わらない．

## 2．入力の解析

- 観測は自由係数の実ベクトル$[\mathrm{Re}\,c_{\mathrm{free}},\mathrm{Im}\,c_{\mathrm{free}}]$（`np.flatnonzero(spec.free_coefficient_mask)`の順）である．
- これを複素の自由係数ベクトルに戻し，`spec.exponents`から各自由係数が$(i,j)$（$x_j$の係数）か，$(i,\cdot)$（定数項）かを求めて，$A$と$c$に並べ替える．
  - 対応は，係数の並び順を仮定せずに，指数ベクトルから構成時に1回だけ計算し，添字のテンソルとして保持する．
  - サポートが条件を満たさない場合は，構成時に`NotImplementedError`を送出する．

## 3．トークン

| トークン | 個数 | 特徴量 |
|---|---:|---|
| 辺トークン$(i,j)$ | $n^2$ | $\mathrm{Re}A_{ij}$，$\mathrm{Im}A_{ij}$，対角かどうか（$i=j$），$d_i$，$d_j$ |
| 頂点トークン$i$ | $n$ | $\mathrm{Re}\,c_i$，$\mathrm{Im}\,c_i$，$d_i$ |

- 辺トークンと頂点トークンは，それぞれ別の線形層で$d_{\mathrm{model}}$次元に埋め込む．
- 変数の番号そのものは特徴量に入れない．位置埋め込みも使わない．
- 頂点$i$は方程式$i$と変数$i$の両方を表す．開始系$x_i^{d_i}-1$により，両者は対応しているためである．

## 4．ブロック（$L$層）

各層は，pre-LNの形で次の3つを順に行う．

1. **行attention**：方程式$i$ごとに，系列［頂点$i$，辺$(i,1),\dots,(i,n)$］に多頭attentionをかけ，残差で更新する．全行で重みを共有する．
2. **列attention**：変数$j$ごとに，系列［頂点$j$，辺$(1,j),\dots,(n,j)$］に多頭attentionをかけ，残差で更新する．全列で重みを共有する．
3. **FFN**：全トークンに，トークンごとのMLP（GELU）をかけ，残差で更新する．

- 変数と方程式の同時置換$\sigma$に対して，辺$(i,j)\to(\sigma(i),\sigma(j))$，頂点$i\to\sigma(i)$と並べ替えると，出力も同じように並べ替わる（置換に同変）．
- 既定値：$d_{\mathrm{model}}=128$，ヘッド数4，$L=4$，FFNの中間次元256，dropoutなし．

## 5．出力

- 辺の出力ヘッド（線形層）：辺$(i,j)$ごとに$2(d_b-1)$個の実数．中間制御点$k$での$\delta A_{ij}$の実部と虚部である．
- 頂点の出力ヘッド（線形層）：頂点$i$ごとに$2(d_b-1)$個の実数．$\delta c_i$の実部と虚部である．
- 出力ヘッドの重みは，MLPの方策と同じく標準偏差0.01で初期化する．初期の行動はほぼゼロ（線形パス）になる．
- 中間制御点$k$ごとに，$\delta A,\delta c$を自由係数の並び（指数ベクトルから求めた添字）に戻し，実ベクトル$\delta_k\in\mathbb{R}^{2D_{\mathrm{free}}}$にする．
- 潜在行動の平均（tanhをかける前）を$u_k=U^\top\delta_k$とし，$k$について連結して出力する．$U$は環境の基底（正方の直交行列，全次元の行動空間）で，モジュールのbufferとして保持する．
- 以降は既存のMLPの方策と同じく，`TanhDiagNormal`で$\tanh$をかけて行動の箱に収める．
- **注意：** 置換の同変性は$\delta$（係数空間）までは厳密である．最後の$\tanh$は回転した潜在座標でかかるため，同変性はわずかに崩れる．行動は箱の内側の小さい値が中心なので，影響は小さいと見込む．対称性による教師データの拡張は，MLPと同じく併用する．

## 6．既存コードへの組み込み

- `src/homotopy_path_learning/rl/attention.py`（新規）：`AttentionActor(nn.Module)`．`forward(obs)`は，既存の`actor_mean`と同じく，形が`(batch, action_dim)`のtanh前の平均を返す．
- `ActorCritic`（`rl/networks.py`）：
  - 引数`actor_type`（`"mlp"`が既定，または`"attention"`）と`attention_kwargs`を追加する．
  - `"attention"`のときは，`actor_mean`を`AttentionActor`にする．構成には`system_spec`，`basis`，`bezier_degree`が必要である．
  - 価値関数（critic）はMLPのままとする（教師あり学習と微調整では使わない）．
  - 既存の呼び出し（引数なし）は従来どおりMLPになる．
- 設定（`config.py`の`PPOSettings`）：任意の項目として，`actor_type`（既定`mlp`），`attention_d_model`，`attention_heads`，`attention_layers`，`attention_ff_dim`を追加する．既存のYAMLは変更なしで読めること．
- checkpoint（`rl/checkpoint.py`）：`actor_type`とattentionのハイパーパラメータを保存し，`build_agent_from_checkpoint`で復元する．既存のcheckpoint（項目なし）はMLPとして読む．
- CLI：
  - `behavior_cloning.py`に`--actor-type`とattentionのハイパーパラメータの引数を追加する．
  - `finetune.py`は，初期値のcheckpointからactorの種類を引き継ぐ．
  - `predicted_actions`，`train_behavior_cloning`，`finetune_step`は変更せずに動くこと．

## 7．テスト

- 同変性：n3-d222とn4-d2222のサポートで，ランダムな観測と，置換のみの対称変換（`systems/symmetry.py`）$g$について，$\delta(T_g\,\mathrm{obs})$と$T_g\,\delta(\mathrm{obs})$が一致すること（係数空間，許容誤差$10^{-5}$）．
- 形と初期値：出力の形が`(batch, (d_b-1)·2D_free)`であること．初期化直後の出力が小さいこと．
- 対象外のサポート：n2-d33-denseのサポートでは，構成時に`NotImplementedError`を送出すること．
- checkpoint：保存と読み込みで出力が一致すること．既存のMLPのcheckpointも読めること．
- 設定：既存のYAMLが従来どおり読めること．
- 統合：smoke設定で，attentionの方策による教師あり学習（数エポック）と微調整（数反復）を実行できること．

## 8．推論時間の計測（計算2）

- 新規CLI`experiments/multivariate_pham/timing.py`：学習済みのrun（MLP，attention）と線形パスについて，評価問題ごとに次の時間を測る．
  - 線形パス：制御点の構成と追跡
  - 方策：観測の構成，方策の順伝播（バッチ1，`torch.no_grad`），制御点の構成，追跡
- ウォームアップ後に各問題を複数回測り，中央値をとる．
- 出力は，問題ごとの時間と$J$（CSV），および要約（平均・中央値，線形パスとの時間比）．
- 他の計算がないときに，1プロセス・1スレッドで実行する．

## 9．評価の方針（敵対的検証の反映）

- **新しい試験問題集合：** 構成の比較には使っていない評価seedで，n4-d2222について500問題を用意する（評価seed 41000〜41499）．試験問題でも，oracle（全次元，予算$B=1000$）を求め，到達率の分母にする．
- **構成の選択：** 構成の選択（エポック，反復，ハイパーパラメータ）は，検証問題と既存の評価100問題で行う．試験問題は最終比較に1回だけ使う．
- **比較ツール：** 新規CLI`experiments/multivariate_pham/compare_runs.py`．同じ問題集合で評価した複数のrunについて，次を出す．
  - $\Delta J$と到達率（seedごと，および3seedの方策のコストを問題ごとに平均した値）
  - 手法間の差の，問題についての対応のあるブートストラップ区間（95%，2000回）
- **判断基準（EXP-0020，実行前に固定）：** 対象はn4-d2222，$d_b=2$とする．教師データ・エポック数・微調整の反復数は，EXP-0017と同一にする．
  - 主基準：微調整後の$\Delta J$の差（attention−MLP，試験500問題）が$-1.0$ポイント以下，かつ95%区間が0を含まない → attentionは有効．
  - 副基準：教師あり学習の段階の到達率の差が+0.05以上 → 教師あり学習の律速を緩める．
  - EXP-0019（MLP，教師データ3万件）の方策も，同じ試験問題で評価して並べる．

## 10．比較用の系列Transformer（EXP-0022）

attention方策の改善が「置換の対称性を構造で満たすこと」によるかを調べるため，置換の対称性を構造として持たないTransformerを比較用に実装する（ADR-0007の「比較の基準」）．

- **名前：** 系列Transformer．`actor_type="sequence"`．新規ファイル`src/homotopy_path_learning/rl/sequence.py`のクラス`SequenceActor(nn.Module)`．
- **トークン：** 自由係数1つにつき1トークン（$D_{\mathrm{free}}$個）．並びは自由係数の順（`np.flatnonzero(spec.free_coefficient_mask)`の順）．
  - 特徴量は係数の実部と虚部の2つ．線形層で$d_{\mathrm{model}}$次元に埋め込む．
  - 学習する位置埋め込み（$D_{\mathrm{free}}\times d_{\mathrm{model}}$，標準偏差0.02の正規分布で初期化）を足す．これにより，各トークンがどの方程式のどの単項式かを位置で区別する．
  - 次数などの特徴量は入れない．位置埋め込みがその情報を担う．
- **ブロック：** attention方策と同じ構成で，pre-LNの「attention副層2つ＋FFN」を$L$層重ねる．ただし，2つのattention副層はどちらも全トークンにかかる．attention方策の`_AttentionSublayer`を再利用してよい．パラメータ数は，同じ$d_{\mathrm{model}}$，ヘッド数，$L$，FFNの次元のattention方策とほぼ同じになる．
- **出力：** 最後にLayerNormをかけ，線形の出力ヘッド（重みは標準偏差0.01の直交初期化，バイアス0）で，トークンごとに$2(d_b-1)$個の実数（中間制御点$k$での$\delta$の実部と虚部）を出す．
  - これを自由係数の順の実ベクトル$\delta_k=[\mathrm{Re}\,\delta,\mathrm{Im}\,\delta]$に並べ，$u_k=U^\top\delta_k$として連結する（5節と同じ）．
- **サポート：** 任意のサポートで動く（1次の項と定数項に限らない）．
- **組み込み：**
  - `ActorCritic`，`build_agent`，checkpoint，`behavior_cloning.py`の`--actor-type`に`"sequence"`を追加する．
  - 大きさの引数は，attention方策と同じ`attention_d_model`，`attention_heads`，`attention_layers`，`attention_ff_dim`（CLIでは`--attention-*`）を使う．
  - `finetune.py`，`evaluate.py`，`timing.py`は，checkpointから種類を引き継いで動くこと．
  - 既存の`"mlp"`と`"attention"`の動作，既存のcheckpointの読み込みは変えない．
- **テスト：**
  - 出力の形が`(batch, (d_b-1)·2D_free)`であること．初期化直後の出力が小さいこと．
  - n2-d33-denseなど，2次の項を含むサポートでも構成と順伝播ができること．
  - n4-d2222で，パラメータ数が同じ大きさのattention方策の±5%以内であること．
  - checkpointの保存と読み込みで出力が一致すること．
  - smoke設定で，`--actor-type sequence`の教師あり学習（数エポック）と微調整（数反復）を実行できること．
