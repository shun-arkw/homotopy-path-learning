# Starting Systemの構成と数学的制約

[README](README.md) ｜ [研究仮説](research_ideas.md) ｜ [データと学習](data_and_learning.md) ｜ [経路](homotopy_paths.md) ｜ [実験](experimental_design.md)

> **Version 0.2．** 本文のM1〜M5は新構想の候補または既存手法のベースラインであり，既存の `multivariate_system` の固定Pham開始系とは別モードである．追跡による性能優位性は未検証．関連：N1，N2，N6．

## 1．基本問題

$n\in\mathbb Z_{>0}$ はスカラーの変数・方程式数とする．目的多項式系を $F=(f_1,\ldots,f_n):\mathbb R^n\to\mathbb R^n$，各成分を $f_i:\mathbb R^n\to\mathbb R$（スカラー値多項式），候補点を $\boldsymbol{y}\in\mathbb R^n$（ベクトル）とする．同型の開始多項式系 $G=(g_1,\ldots,g_n):\mathbb R^n\to\mathbb R^n$ を構成して，

$$
G(\boldsymbol{y})=\boldsymbol{0}_n,\qquad \det J_G(\boldsymbol{y})\ne0
$$

を満たした場合，$(G,\boldsymbol{y})$ を正則なStarting Pairとして扱う．ここで $J_G(\boldsymbol{y})\in\mathbb R^{n\times n}$ はJacobian行列である．複数点では $C=\{\boldsymbol{y}^{(1)},\ldots,\boldsymbol{y}^{(k)}\}\subset\mathbb R^n$（$k\in\mathbb Z_{>0}$）の全点に要求する場合と，一部のみへ要求する場合を分ける．

本研究で保存し得る性質は少なくとも次の3層である．

1. **Support-Preserving**：各 $g_i$ の単項式を目的系の指定集合 $A_i$ の中に制限する（$\operatorname{supp}(g_i)\subseteq A_i$）．
2. **Algebraic-Structure-Preserving**：対称性，多重次数，係数間の非自明な代数関係などを保つ．
3. **Problem-Family-Preserving**：$F(x;p)$ の許容パラメータ集合 $\mathcal P$ に属する系として $G(x)=F(x;p_0)$ を実現する．

「同じサポート」と「同じ問題ファミリー」は同義ではない．非零係数の完全一致も，一般のサポート包含とは異なる．

## 2．方法索引

| Method | 名称 | 概要 | 制約・リスク |
|---|---|---|---|
| **M1** | Individual Newton Homotopy | 各候補 $y$ に対して $G_y(x)=F(x)-F(y)$ | $J_F(y)$ が特異なら開始点も特異．定数項の変化を許す必要がある |
| **M2** | Single-Point Coefficient Projection | 指定サポート内の係数を最小変更 | 消失性のみで正則性を保証しない |
| **M3** | Multi-Point Constrained Projection | 全候補で消失する係数へ射影 | 零空間が小さくなり，零多項式しか得られない可能性 |
| **M4** | Vanishing-Ideal / ABM-Based Construction | 有限点集合から消失多項式・基底を構成 | 出力が正方・正則・サポート保存とは限らない |
| **M5** | Structure-Preserving Reconstruction | 許容ファミリー内で開始パラメータを選ぶ | 実現不能な候補点が存在する |

数式中で既存の単変数反例を記す場合は $x\in\mathbb R$ をスカラーとして使用する．それ以外の多変数系では $\boldsymbol{x},\boldsymbol{y}\in\mathbb R^n$ をベクトルとする．多変数系においてはベクトルを太字で表す．単変数反例に限り，$x,y\in\mathbb R$ をスカラーとして使用する．

線形係数経路など経路側の方法IDは [homotopy_paths.md](homotopy_paths.md) を参照．

## 3．M1：Individual Newton Homotopy

定数項の変更を許容する場合，任意の $\boldsymbol{y}\in\mathbb R^n$ に対して

$$
G_{\boldsymbol{y}}(\boldsymbol{x})=F(\boldsymbol{x})-F(\boldsymbol{y}),\quad G_{\boldsymbol{y}}(\boldsymbol{y})=\boldsymbol{0}_n,\quad J_{G_{\boldsymbol{y}}}(\boldsymbol{y})=J_F(\boldsymbol{y})
$$

となる．したがって，$J_F(\boldsymbol{y})$ が非特異であれば正則なStarting Pairが得られる．線形係数ホモトピーは

$$
H_{\boldsymbol{y}}(\boldsymbol{x},t)=F(\boldsymbol{x})-(1-t)F(\boldsymbol{y}),\qquad t\in[0,1]
$$

である．$t=0$ における解経路の接ベクトルは

$$
\frac{d\boldsymbol{x}}{dt}(0)=-J_F(\boldsymbol{y})^{-1}F(\boldsymbol{y})
$$

となり，初期方向はNewtonステップに一致する．ただし，**Newton HomotopyとNewton反復が同じアルゴリズムであることは意味しない**．経路追跡と直接Newton法を双方ベースラインにする．$F$ の定数項サポートが欠けている場合，元のサポートの外に項が現れ得る．関連：R13．

### M1を含む比較の基本的な反例

**Newton Homotopyも経路全体の正則性を保証しない．** 例えば，$F:\mathbb R^2\to\mathbb R^2$，$F(\boldsymbol{x})=(x_1^3-3x_1+3,\, x_2)^\top$，$\boldsymbol{y}=(2,0)^\top\in\mathbb R^2$ に対して，$J_F(\boldsymbol{y})=\operatorname{diag}(9,1)$ は正則である．しかし $H_{\mathrm{NH}}(\boldsymbol{x},t)=(x_1^3-3x_1-2+5t,\, x_2)^\top$ は $t=4/5$，$\boldsymbol{x}=(1,0)^\top$ でJacobianが特異となり，$\boldsymbol{y}$ からの実数枝がそこへ到達する．正則な開始系からのHCが必ず目的解へ到達する，という主張は誤りである．

## 4．M2：Single-Point Coefficient Projection

第 $i$ 成分（$i\in\{1,\ldots,n\}$）の許容サポートを $A_i\subset\mathbb Z_{\ge0}^n$，単項式数を $s_i=|A_i|\in\mathbb Z_{>0}$，単項式評価ベクトルを $\boldsymbol{\phi}_i:\mathbb R^n\to\mathbb R^{s_i}$，係数ベクトルを $\boldsymbol{a}_i,\boldsymbol{b}_i\in\mathbb R^{s_i}$ とする．$f_i(\boldsymbol{x})=\boldsymbol{a}_i^\top\boldsymbol{\phi}_i(\boldsymbol{x})$，$g_i(\boldsymbol{x})=\boldsymbol{b}_i^\top\boldsymbol{\phi}_i(\boldsymbol{x})$ と書く．$\boldsymbol{\phi}_i(\boldsymbol{y})\ne\boldsymbol{0}_{s_i}$ のもとで，

$$
\min_{\boldsymbol{b}_i\in\mathbb R^{s_i}}\|\boldsymbol{b}_i-\boldsymbol{a}_i\|_2^2\quad\mathrm{s.t.}\quad \boldsymbol{b}_i^\top\boldsymbol{\phi}_i(\boldsymbol{y})=0
$$

の解は，

$$
\boxed{\boldsymbol{b}_i=\boldsymbol{a}_i-\frac{f_i(\boldsymbol{y})}{\|\boldsymbol{\phi}_i(\boldsymbol{y})\|_2^2}\boldsymbol{\phi}_i(\boldsymbol{y})}
$$

である．分母が正でない場合（例えば許容項が $x^2$ だけで $y=0$），通常の公式は適用できない．その場合，制約が自動的に満たされることはあり得るが，Jacobianが非特異となる保証はない．

正定値行列 $W_i\in\mathbb R^{s_i\times s_i}$ による重み付き目的関数 $(\boldsymbol{b}_i-\boldsymbol{a}_i)^\top W_i(\boldsymbol{b}_i-\boldsymbol{a}_i)$ の場合，

$$
\boldsymbol{b}_i=\boldsymbol{a}_i-\frac{f_i(\boldsymbol{y})}{\boldsymbol{\phi}_i(\boldsymbol{y})^\top W_i^{-1}\boldsymbol{\phi}_i(\boldsymbol{y})}W_i^{-1}\boldsymbol{\phi}_i(\boldsymbol{y})
$$

となる（分母が正の場合）．「係数距離」は基底・重みの選択に依存するため，距離が小さいことが追跡の速さを示すとは限らない．

**追加の反例：** $F:\mathbb R^2\to\mathbb R^2$，$F(\boldsymbol{x})=(x_1^2-3x_1+1,\, x_2)^\top$，候補 $\boldsymbol{y}=(1,0)^\top$ は $J_F(\boldsymbol{y})=\operatorname{diag}(-1,1)$ が正則である．第1成分を単項式基底 $(1,x_1,x_1^2)$ で射影すると，開始系は $G(\boldsymbol{x})=((4/3)(x_1-1)^2,\, x_2)^\top$ となり，$J_G(\boldsymbol{y})$ は特異である．したがって **Coefficient Projectionの最小係数距離は正則性を保証しない**．

**実現可能だが正則とは限らない反例：** $F(x)=x^2-1$，$y=0$，サポート $A=\{1,x^2\}$ では $G(0)=0$ を満たす系は $G(x)=a x^2$ となるため $G'(0)=0$．どの係数でも正則な開始解を作れない．

## 5．M3：Multi-Point Constrained Projection

相異なる候補点集合 $C=\{\boldsymbol{y}^{(1)},\ldots,\boldsymbol{y}^{(k)}\}\subset\mathbb R^n$（$k\in\mathbb Z_{>0}$）に対し，各成分の単項式評価行列 $V_i(C)\in\mathbb R^{k\times s_i}$ を，

$$
V_i(C)=\begin{bmatrix}\boldsymbol{\phi}_i(\boldsymbol{y}^{(1)})^\top\\\vdots\\\boldsymbol{\phi}_i(\boldsymbol{y}^{(k)})^\top\end{bmatrix}
$$

とする．すべての点で $g_i=0$ となる非零係数ベクトル $\boldsymbol{b}_i\in\mathbb R^{s_i}$ の存在条件は，

$$
\boxed{\dim\ker V_i(C)=|A_i|-\operatorname{rank}V_i(C)>0}
$$

である．最小二乗係数射影として，

$$
\boxed{\boldsymbol{b}_i=(I_{s_i}-V_i(C)^+V_i(C))\boldsymbol{a}_i}
$$

を求められる．$V_i(C)^+\in\mathbb R^{s_i\times k}$ はMoore–Penrose擬似逆行列，$I_{s_i}\in\mathbb R^{s_i\times s_i}$ は単位行列である．この射影による $\boldsymbol{b}_i$ は零ベクトルとなる場合があり，非零性を保証しない．また，行ごとの非零性は $\det J_G(\boldsymbol{y}^{(j)})\ne0$ の十分条件ではない．

**係数距離の単調性（Established）：** 同じ係数基底・ノルムで比較し，単一点の消失制約を複数点制約に追加した場合，許容集合が縮小するので最適値は非減少である．これはHCの追跡コストが非減少であることを**意味しない**．

**Bézout型制約：** 例えば2変数の2次多項式2本の正則な孤立複素解は高々4個であり，8個の相異なる実点を同じ正則な正方系の解として埋め込めない．一般にはサポートと点配置による代数的制約を先に確認する．

### 正則性を制約として加える設計

$$
\min_{G}\sum_{i=1}^n\|\boldsymbol{b}_i-\boldsymbol{a}_i\|_2^2 \quad
\mathrm{s.t.}\quad G(\boldsymbol{y}^{(j)})=\boldsymbol{0}_n,\quad \sigma_{\min}(J_G(\boldsymbol{y}^{(j)}))\ge \varepsilon_J>0
$$

という探索問題を候補とする．$\sigma_{\min}$ の下界は変数スケーリングや方程式スケーリングに依存するため，正規化規則も明記する必要がある．非凸であり，常に可解とは限らない．この定式化自体は未検証のアルゴリズム案である．

## 6．M4：Vanishing Ideal，Buchberger–Möller，ABM

有限点集合 $C$ のVanishing Idealは，

$$
I(C)=\{g\in\mathbb R[x_1,\ldots,x_n]:g(y)=0\ \forall y\in C\}
$$

である．相異なる有限個の点に対する $I(C)$ は零次元の根基イデアルである．Buchberger–Möller法は有限点集合の正確な消失イデアルを計算する古典的手法であり（R15），Approximate Buchberger–Möller（ABM）／Approximate Vanishing Ideal（AVI）系の手法は近似的な点集合・浮動小数点データを扱う（R9，R10）．具体的な実験の実装候補としてMAVI（R11）を参照する．

**注意：** $I(C)$ の生成元数は一般に $n$ と一致しない．基底の生成元から任意に $n$ 個選んでも，その共通零点が正確に $C$ だけである保証はない．ABMの近似消失性は，HCの開始条件 $G(y)=0$ の厳密性を直ちに保証しない．ABM+GWNの勾配重み付き正規化による安定化（R10）も，$J_G$ の良条件性やパス正則性を意味しない．

### 存在を示す構成（古典的な補間原理）

任意の相異なる有限実点集合 $C\subset\mathbb R^n$ に対し，点を区別する実線形形式 $\ell(x)$ を選び，可逆な実線形変換によって $z_1=\ell(x)$ を第1座標とする．各点の第1座標を相異なる $t_j$ とする．補間多項式 $h_2,\ldots,h_n$ を $h_i(t_j)=z_i^{(j)}$ となるよう構成すると，

$$
G(z)=\begin{pmatrix}\prod_{j=1}^k(z_1-t_j)\z_2-h_2(z_1)\\\vdots\z_n-h_n(z_1)\end{pmatrix}
$$

は，複素数体上でも $C$ をちょうど零点集合とする正方系である．各点のJacobian行列式は $\prod_{\ell\ne j}(t_j-t_\ell)\ne0$ に比例するので正則である．ただし，**元のサポート・次数・問題ファミリーは通常保存しない**．「複数点から正則な正方系を作る」こと単独は新規な主張ではない．

### 共通開始系の実例

$$
C=\{(-1,0),(0,0),(1,0)\},\qquad G(x_1,x_2)=\begin{pmatrix}x_1^3-x_1\x_2\end{pmatrix}
$$

とすると，3点とも正則な開始解であり，それ以外の複素零点は存在しない．この例は構成可能性のみを示し，効率改善は示していない．

## 7．M5：Structure-Preserving Reconstruction

元の問題ファミリーを $F:\mathbb R^n\times\mathbb R^q\to\mathbb R^n$（$q\in\mathbb Z_{>0}$），許容パラメータ領域を $\mathcal P\subseteq\mathbb R^q$，目的パラメータを $\boldsymbol{p}^*\in\mathcal P$，候補点を $\boldsymbol{y}\in\mathbb R^n$ とする．開始パラメータ $\boldsymbol{p}_0\in\mathcal P$ を，

$$
\min_{\boldsymbol{p}_0\in\mathcal P}d(\boldsymbol{p}_0,\boldsymbol{p}^*)\quad
\mathrm{s.t.}\quad F(\boldsymbol{y};\boldsymbol{p}_0)=\boldsymbol{0}_n
$$

で選ぶ案を検討する．距離 $d$ を単純なユークリッド距離とする場合，座標変更に依存することに注意する．

パラメータにアフィンな $F(\boldsymbol{x};\boldsymbol{p})=A(\boldsymbol{x})\boldsymbol{p}+\boldsymbol{b}(\boldsymbol{x})$ で，$A(\boldsymbol{x})\in\mathbb R^{n\times q}$，$\boldsymbol{b}(\boldsymbol{x})\in\mathbb R^n$，$\mathcal P=\mathbb R^q$ の場合，$\boldsymbol{y}$ の実現可能性は

$$
\boxed{-\boldsymbol{b}(\boldsymbol{y})\in\operatorname{Im}A(\boldsymbol{y})}
$$

と同値である．ランク条件 $\operatorname{rank}A(\boldsymbol{y})=\operatorname{rank}[A(\boldsymbol{y})\mid-\boldsymbol{b}(\boldsymbol{y})]$ と表せる．$\mathcal P$ に追加制約がある場合はこれだけでは不十分である．$F(\boldsymbol{y};\boldsymbol{p}_0)=\boldsymbol{0}_n$ でも $\det J_xF(y;p_0)\ne0$ の保証はない．

**不可能性の反例：** $F(x;p)=x^2+p^2-1$，$p^*=0$，$y=2$ とすると，$F(2;p)=3+p^2>0$ なので許容実数パラメータ $p_0$ は存在しない．

**サポートとの違いの例：** $F(x;p)=(x-p)^2-1$，目的系 $F(x;0)=x^2-1$，候補 $y=2$ の場合，$p_0=1$ を選べば $G(x)=x^2-2x$ となる．同じファミリーに属するが，目的系と同じ非零サポートを持たない．

**複数点への拡張：** $F(y^{(j)};p_0)=0$ を全点で同時に満たす必要がある．通常の単一点より厳しい制約であり，幾何学的に自然な複数点が同一のパラメータ値で成立するとは限らない．

## 8．開始系とホモトピーの失敗例

### 実数枝の特異性

$$
G(x)=x^3-3x,\quad F(x)=x^3+x,\quad
H(x,t)=x(x^2-3+4t)
$$

において開始解は $-\sqrt3,0,\sqrt3$ で，$t=3/4$ において **3つの実数枝は同じ点で衝突してJacobianが特異になる**．中央の定数枝 $x(t)=0$ も $t=3/4$ でJacobianが零になるため，**正則な経路のまま到達できるとは言えない**．この例は「中央の枝だけは常に正則」という説明の反例でもある．

### 係数の退化と無限遠

$$
F(x)=x^3+x,\quad y=\sqrt3,\quad G(x)=-\tfrac15x^3+\tfrac35x
$$

では $G(y)=0$ かつ $G'(y)=-6/5\ne0$．係数線形補間 $H(x,t)=((6t-1)/5)x^3+((3+2t)/5)x$ では $y$ からの分枝 $x(t)=\sqrt{(3+2t)/(1-6t)}$ が $t\uparrow1/6$ で発散する．目的系は正則な実解 $x=0$ を持つが，選んだ開始経路では有限に到達できない．

### 共通開始系 vs 独立開始系

一つの $G_C$ に全候補点を埋め込むことは，複数の $(G_j,y^{(j)})$ を個別に作ることとは異なる．いずれも複数候補を使って「最初の1解」だけを狙える．**共通開始系が安価だとは予断しない**．実験E2の主比較とする．

## 9．既存の固定Pham開始系との境界

既存リポジトリのPham型 $G_i(\boldsymbol{x})=x_i^{d_i}-1$（$G:\mathbb C^n\to\mathbb C^n$，$d_i\in\mathbb Z_{>0}$）は**既存のBézier学習の前提**であり，M1〜M5で自動的に置換しない．Starting Pair構成を実装する場合には，任意系を受ける新たな構成モードを別途設計する．既存の全複素解追跡機能と，提案する1実数解探索機能の成功条件・時間計測は別にする．

## 10．検証すべき実装上の論点

- 候補点数 $k$ と許容サポート $A_i$ の関係，評価行列の数値ランク，非零係数の可解性．
- $G$ が零多項式や共通因子を含まないか，追加解・正の次元の解集合が現れないか．
- $\|F(y)\|$，$\sigma_{\min}(J_G(y))$，係数ノルム，スケール正規化．
- 厳密に既知の開始解か，浮動小数点近似をNewton補正した解か，認証まで行った解かを区別．
- ABMの計算費用とランク判定・SVDを使う単純な射影法との比較．
- 失敗時の再試行回数と経路追跡にかかる時間．

具体的なベースラインと指標は [experimental_design.md](experimental_design.md) に集約する．
