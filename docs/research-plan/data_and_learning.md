# データ生成と機械学習（Data and Learning）

[README](README.md) ｜ [研究仮説](research_ideas.md) ｜ [開始系](starting_system.md) ｜ [実験](experimental_design.md) ｜ [文献](related_work.md)

> **Version 0.2．** 多変数Bézier係数パスのMLは既存コードで実装・実験済み．以下の**実数解候補の学習・開始系再構成**は新規構想として未実験．対応仮説：N3，N4，N6．

## 0．既存の学習設定と新構想の切り分け

既存の `multivariate_system` は，$n\in\mathbb Z_{>0}$ に対する複素Pham型の目的多項式系 $F:\mathbb C^n\to\mathbb C^n$ を生成し，開始系 $G_i(\boldsymbol{x})=x_i^{d_i}-1$ を固定する．$d_i\in\mathbb Z_{>0}$ は第 $i$ 成分の次数である．Oracle探索，Behavior Cloning，PPOによる微調整，Attention方策は**係数パス**を最適化する．`EXP-0059`〜`EXP-0061` は，複素係数の低次元ファミリーと学習しやすさに関する既存研究である．

`ADR-0008` のファミリーは自由係数ベクトル $\boldsymbol{c}_{F,\mathrm{free}}\in\mathbb C^m$ に対し，$\boldsymbol{c}_{F,\mathrm{free}}=W\boldsymbol{u}$ として生成する．ここで $m=n(n+1)\in\mathbb Z_{>0}$，$W\in\mathbb C^{m\times k_f}$，$\boldsymbol{u}\in\mathbb C^{k_f}$，$k_f\in\{1,\ldots,m\}$ はファミリーの複素次元である．この $k_f$ は**候補点数** $k_c$ とは異なる．実応用の物理パラメータ化を自動的に表すものでもない．

新構想では，**別のデータ生成器・別の問題設定**として，少なくとも1つの正則な実数解を持つ $F:\mathbb R^n\to\mathbb R^n$ を作り，MLから候補点 $\hat{\boldsymbol{x}}\in\mathbb R^n$ または集合 $C$ を生成する．既存PPOモデルが，そのまま実数解候補を予測できるとは想定しない．

## 1．問題設定とデータの単位

パラメータ化した目的多項式系 $F:\mathbb R^n\times\mathcal P\to\mathbb R^n$（$\mathcal P\subseteq\mathbb R^q$，$q\in\mathbb Z_{>0}$），目的パラメータ $\boldsymbol{p}^*\in\mathcal P$，解集合 $\mathcal S_F=\{\boldsymbol{x}\in\mathbb R^n:F(\boldsymbol{x};\boldsymbol{p}^*)=\boldsymbol{0}_n\}$ を考える．学習対象としては，(a) 単一候補 $\hat{\boldsymbol{x}}\in\mathbb R^n$，(b) 候補集合 $C_\theta(F)$，(c) 開始系や経路の設計指標，(d) 追跡の成功確率・時間などの予測，を別々に比較する．

**重要：** 学習時に既知の `Planted Target Solution` と，推論時に未知の `Target Solution` を混同しない．データセットで植え込んだ解は，推論時の開始系構成に直接渡してはならない．評価時の真値検証には利用できる．

## 2．Root-first：解を先に作る合成データ

1. 実数点 $\boldsymbol{x}^*\sim q_X$（$q_X$ は $\mathbb R^n$ 上の確率分布） を生成する．
2. サポート $A_i$ または問題ファミリー $F(\boldsymbol{x};\boldsymbol{p})$ を決める．
3. $F(\boldsymbol{x}^*;\boldsymbol{p}^*)=\boldsymbol{0}_n$ を満たす目的パラメータ／係数を構成する．
4. $J_{\boldsymbol{x}}F(\boldsymbol{x}^*;\boldsymbol{p}^*)\in\mathbb R^{n\times n}$ の正則性と，必要な実行可能領域を検査する．
5. 目的系 $F$，パラメータ $\boldsymbol{p}^*\in\mathcal P$，既知の埋め込み解 $\boldsymbol{x}^*\in\mathbb R^n$，生成条件と乱数種を保存する．

**固定サポートでの単純な構成例：** 元のサポートに定数項を含むなら，任意に係数をサンプリングした $\widetilde F:\mathbb R^n\to\mathbb R^n$ に対し $F(\boldsymbol{x})=\widetilde F(\boldsymbol{x})-\widetilde F(\boldsymbol{x}^*)$ とすれば $F(\boldsymbol{x}^*)=\boldsymbol{0}_n$．定数項が許されない場合には，単項式評価ベクトル $\boldsymbol{\phi}_i(\boldsymbol{x}^*)\in\mathbb R^{s_i}$ の核に係数を射影するなど，可解性を確認した別の構成が必要である．どちらもJacobian正則性や一般の係数分布を自動的には保証しない．

**複数解を埋め込む場合：** $X^*=\{\boldsymbol{x}^{*(1)},\ldots,\boldsymbol{x}^{*(r)}\}\subseteq\mathbb R^n$ に対し $V_i(X^*)\boldsymbol{a}_i=\boldsymbol{0}_r$ を課す．任意の点集合を指定サポートの中に埋め込めるわけではない．$r$ とサポートの自由度，零空間次元，Bézout/BKK等の上界を検証する．`r`（植え込む目的解の数）と `k`（推論時の候補点数）は**別の変数**とする．

**問題：** Root-firstは $q_X$，条件付け，不可能なサンプルの棄却などにより目的系の分布が偏る．目標が特定の実応用分布である場合，合成データがその分布を代表するとは限らない．

## 3．Parameter-first：問題を先にサンプリングする

$\boldsymbol{p}^*\sim q_P$（$q_P$ は $\mathcal P$ 上の確率分布） から目的系を先に作り，数値計算・HC等で教師解を取得する．自然なパラメータ分布を表現しやすい一方，教師解の取得は高コストになり得る．実数解が0個の問題も発生するため，問題分布の定義や採否条件を明示する．

**推奨：** Root-first合成データとParameter-firstデータを無条件に同じものとして扱わず，E3で係数分布，ヤコビアン条件付け，実解数分布，数値解の難易度を比較する．可能なら実応用由来の問題も別ホールドアウトとする．

## 4．学習方法の選択肢（M10）

| 候補 | 出力 | 利点 | 主な懸念 |
|---|---|---|---|
| 回帰モデル（MLP / Transformer） | $\hat{\boldsymbol{x}}$ | 軽量な基準手法 | 多峰的な解分布で平均へ潰れる |
| $k$候補の集合予測 | $\{\hat x^{(j)}\}_{j=1}^k$ | 複数枝を探索できる | 候補の重複，順序対称性，再構成コスト |
| Conditional Flow Matching等 | 候補分布のサンプル | 複数モードを表現し得る | サンプリング費用，分布推定とHC性能のずれ |
| 成功確率・費用のランキング | 候補優先順位 | 最初の1解に合う | 追跡結果のラベル作成に費用がかかる |
| RL / サロゲート最適化 | 候補・経路方策 | 総コストを最適化し得る | 非微分な失敗指標，データ効率，計算費用 |

**用語上の注意：** ML出力は `Predicted Candidate` であり，$G(\hat{\boldsymbol{x}})=\boldsymbol{0}_n$ を確認するまでは `Starting Solution` と呼ばない．

### MSE回帰に関する反例

$F(x)=x^2-1$ に対し，教師ラベルが同じ確率で $x^*=\pm1$ なら，平方誤差の条件付き平均は $\hat x=0$ になる．これは目的系の解ではなく，$F'(0)=0$ でもある．単純な解の平均が望ましい候補とは限らない．集合予測・mode selection・生成モデルの必要性を検討する理由であるが，それらを使えば自動的にHCが高速になるわけではない．

## 5．Learning Objective（N3・N4）

### 解回帰・候補被覆のベースライン

単一解の場合には $\|\hat{\boldsymbol{x}}-\boldsymbol{x}^*\|_2^2$，複数ラベルと複数候補の場合には適切な集合マッチング，minimum-over-set損失などを候補とする．**埋め込み解だけに近いことと，任意の実数解へ到達しやすいことは同一ではない**．

### HCの性能を考慮する目標

$\mathcal R$ を開始系再構成演算，$\pi$ を経路・候補の選択方策として，

$$
\min_{\theta,\pi}\mathbb E_{F\sim\mathcal D}\big[T_{\mathrm{first}}(F,\mathcal R(F,C_\theta(F)),\pi)\big]
$$

を候補とする．ただし，解探索に失敗した試行を単純に平均から除外すると「難しい問題を捨てる」方法が有利になる．実際の目的関数は，時間上限 $B$ に対する失敗ペナルティ，成功率制約，認証の費用などを明記して決める．

推論時間の分解例は，

$$
T_{\mathrm{online}}=T_{\mathrm{inference}}+T_{\mathrm{reconstruction}}+T_{\mathrm{path\ design}}+T_{\mathrm{tracking}}+T_{\mathrm{verification}}+T_{\mathrm{retry}}.
$$

大規模事前学習を採用する場合には，オフライン学習費用 $T_{\mathrm{train}}$ を，想定する問題処理数 $Q$ に応じて $T_{\mathrm{train}}/Q$ として別途評価する．

### 低費用な代理損失（仮説）

候補残差，$\sigma_{\min}(J_G)$，再構成実現可能性，短い予備追跡のステップ数，過去の追跡失敗率等を使う可能性がある．ただし，Jacobian最小特異値やユークリッド距離は適切な正規化の下でのみ比較可能であり，経路全体の成功は保証しない．

## 6．情報漏洩と評価分割

1. **ターゲット解を推論入力にしない．** `Planted Target Solutions` は教師・評価用途とし，Starting Pairの構成は予測・ランダム候補のみを使う．
2. **同じ構造の重複系をまたがせない．** 単なる変数順序・スケーリング変換などで同一視できる問題を訓練・評価に無秩序に分けない．
3. **固定サポート／未知サポートを区別する．** 研究の一般化範囲を実験で測った範囲より広げない．
4. **外挿テストを設ける．** 変数数，次数，係数スケール，条件数，解の個数，問題ファミリー等を可能な範囲で変化させる．
5. **同じ計算予算で比較する．** 生成候補数，GPU / CPU，前処理，再試行回数を明示する．

## 7．実応用データの生成とベンチマーク分布

| 問題ファミリー | 学習・評価データの生成 | 最初の1解の成功条件に対する注意 |
|---|---|---|
| 一般の固定サポート合成系 | Root-firstおよびParameter-firstを別分布で生成 | 植え込み解の漏洩・同等問題の重複を防ぐ |
| P3P・Generalized Camera Resectioning | 姿勢・3D点と投影観測の整合性から生成，または公開データ | 正方系の最小問題とノイズ下の最小二乗PnPを混同しない．最初に得た実数解が真の姿勢とは限らない |
| Mass-Action化学反応ネットワーク | ネットワーク構造，正の速度定数，保存量等を指定 | 正の定常状態，保存量，必要なら安定性まで検査．多重定常性の証明には1解では不足 |
| Kuramoto等の力学系 | 結合グラフとパラメータから平衡点方程式を構成 | 位相固定と円周制約，必要なら安定性を評価 |

実応用のパラメータ集合は一般に非線形であり，任意の実数解を埋め込めるとは限らない．選定したファミリーごとに**Planted Root-firstが適用可能か**を先に調べる．公開ソルバー（Catalyst，専用PnPソルバー等）も比較候補に含める．具体的なベンチマーク構成は [experimental_design.md](experimental_design.md) を正本とする．

## 8．研究上の判断

最初に比較すべきは，(a) ランダム候補，(b) 単一候補回帰，(c) 複数候補生成，(d) 候補ランキング，である．再構成アルゴリズムとパスを一度固定することで，候補生成方法の効果を分離する．Flow MatchingとPPOは**可能性を残すが必須としない**．

N3が支持されずともStarting System設計を続けられる．逆にN1が否定されても，独立Starting PairについてN3・N4を研究できる．詳細な実験計画は [experimental_design.md](experimental_design.md) を参照．
