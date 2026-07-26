\title{
ホモトピー連続法における曲線パスの最適化
}

\section*{Curve－Path Optimization in Homotopy Continuation}

\author{
荒川 駿＊1 川本 一彦＊1 計良 宥志＊1＊2 Thibault Liard＊3 \\ Shun Arakawa Kazuhiko Kawamoto Hiroshi Kera \\ Olivier Ruatta ${ }^{* 3 * 4}$ Tristan Vaccon ${ }^{* 3}$ \\ ＊1千葉大学 $* 2$ 国立情報学研究所 \\ Chiba University National Institute of Informatics \\ ＊3Université de Limoges，XLIM，CNRS UMR 7252 \\ ＊4CANARI INRIA Bordeaux and University of Bordeaux
}

\begin{abstract}
Homotopy continuation solves a target polynomial system by constructing a homotopy from a start system with known solutions and tracking the resulting solution paths．Although widely used，path tracking can become numerically unstable near singularities（ill－conditioned Jacobian）or when solution paths closely approach each other．Designing coefficient－space paths that avoid such regions is therefore crucial for stable and efficient tracking．In this study，we focus on univariate polynomials with complex coefficients and represent the homotopy path as a low－degree Bézier curve．We optimize the intermediate control points using Proximal Policy Optimization（PPO）to reduce tracking cost．Experiments show that the learned Bézier paths consistently reduce both the mean and the standard deviation of the number of tracking steps compared with the linear path，while maintaining a $100 \%$ success rate．
\end{abstract}

\section*{1．はじめに}

ホモトピー連続法は，既知の解を持つ多項式系（開始系）から解を求めたい多項式系（目的系）へ連続変形す るホモトピーを構成し，その解の軌跡を追跡することで，目的系の解を数値的に得る手法である。多変数多項式系 の解法として，コンピュータビジョン［1］，化学［2］，ロ ボット運動学［3］など様々な分野で利用されてきた。実用上は開始系と目的系の係数を直線補間で結ぶ線形ホモ トピーが標準的であるが，解の追跡は常に安定とは限ら ない。特に，追跡途中でヤコビアンが特異に近づく場合 や，目的系が重解（あるいはそれに近い状況）を含む場合には，条件数が急激に増大する。その結果，ニュート ン修正子の収束域が狭くなり，ステップ幅が縮みやすい ［4，5］．さらに，異なる解の軌跡が互いに近接する状況 では，追跡中に別の軌跡へ移ってしまう path jumping も問題となる［5］．これらの現象はステップ数と計算時間を増大させ，追跡失敗の要因になり得る．そのような追跡困難性は condition length などの理論指標と関連付けて議論されているものの［6］，数値的に不安定な領域を避けるホモトピーパスを係数空間上で具体的に設計 することは，依然として課題である。

本研究では，$n$ 次一変数複素係数多項式を対象とし，係数空間 $\mathbb{C}^{n+1}$ 上で開始多項式から目的多項式へ至るホモ トピーパスを低次数のベジェ曲線で表現する。そして， ベジェ曲線の中間制御点を近接方策最適化（PPO）［7］ により最適化し，追跡コストの削減を目指す。ベジェ曲線によるパス設計自体は先行研究にも見られ，condition length の最小化に基づく議論がある［8］．これに対し本研究は，condition length を直接最小化するのではなく，追跡コストに基づく報酬を用いて，開始系と目的系の係

\footnotetext{
連絡先：計良 宥志，kera＠chiba－u．jp
}

数から制御点配置を決定する方策を学習し，各目的多項式ごとに追跡効率の良いパスを予測する。実験の結果，次数 $n=5,10,20,40,80$ の一変数多項式では，成功率は全条件で $100 \%$ を維持したまま，ステップ数の平均と標準偏差がベジェ曲線で一貫して改善した。特に，ベジェ曲線の次数 $d_{b}=4$ では，目的多項式の次数 $n=10-80$ において平均ステップ数がおおよそ $15-19 \%$ 減少し，標準偏差も同様に減少した（表 1）。以上より，本実験設定において，多項式ごとに有効なパスを予測する枠組み が，追跡効率の改善に有効である可能性を示した。

\section*{2．関連研究}

ホモトピー追跡の計算量を理論的に捉える枠組みとし て，condition metric と condition length が導入されて いる。前者は，各点での条件数を重みとしてパスの長さ を測るための計量であり，後者はその計量に基づいて定義 されるホモトピーパスの長さ（経路積分）である．Shub の研究［6］では，ホモトピーパスの condition length が，目的系の近似解を得るために必要なニュートン反復回数 の上界と結び付くことが示され，condition metric にお ける測地線（geodesic）を探索する動機付けが与えられ ている［9］．一方で，実際に測地線を求めることは一般 に難しく，数値的に扱いやすい形でパスを設計する方法論が求められる。

機械学習とホモトピー連続法の融合としては，開始 ペア（開始系とその解）の選択や，追跡規則の意思決定 を学習により補助する研究が進んでいる．Hruby らは， 3D ビジョンの最小問題（minimal problem）に対し，有望な開始ペアを学習して，全解を求める代わりに限られ た本数のパス追跡で目的解を得る枠組みを示した［10］． Zhang らは，回帰器とオンラインシミュレータにより，目的の問題と整合する開始ペアを生成し，少数のパス追

跡で効率と成功率を両立する手法を報告している［11］． さらに Mai らは，ステップ幅選択や反復停止といった予測子•修正子の意思決定を PPO により学習する Neural Predictor－Corrector を提案した［12］．

\section*{3．背景知識}

この節では，ホモトピー連続法の基本的な枠組みと，パ ス追跡に用いる標準的な予測子•修正子法（Predictor－ Corrector method）を整理する。記法は一般の多変数多項式系として与えるが，4．節以降の提案手法と実験は一変数多項式に限定する。以降では， $\mathbf{x}=\left(x_{1}, \ldots, x_{\nu}\right) \in \mathbb{C}^{\nu}$ とし，$F, G \in \mathbb{C}\left[x_{1}, \ldots, x_{\nu}\right]^{\nu}$ を $\nu$ 本の多項式からなる多項式系とする。

\section*{3.1 ホモトピー連続法}

目的系 $F(\mathbf{x})=\mathbf{0}$ の解を得るために，解が既知な開始系 $G(\mathbf{x})=\mathbf{0}$ から $F$ へ連続変形するホモトピー
$$
H(\mathbf{x}, t)=(1-p(t)) G(\mathbf{x})+p(t) F(\mathbf{x})
$$

を考える．ここで $t \in[0,1]$ はパラメータであり， $p:[0,1] \rightarrow \mathbb{R}$ は
$$
p(0)=0, \quad p(1)=1
$$

を満たす関数とする。典型的には $p(t)=t$ であり，こ の場合，上式は線形ホモトピーと呼ばれる。開始系の解 $\mathbf{x}(0)$ を初期値として，方程式
$$
H(\mathbf{x}(t), t)=\mathbf{0}
$$

を満たす解曲線 $\mathbf{x}(t)$ を $t=0$ から $t=1$ まで追跡し， $\mathbf{x}(1)$ を目的系の数値近似解として得る。

\section*{3.2 予測子•修正子法}

解曲線上では常に $H(\mathbf{x}(t), t)=\mathbf{0}$ が成り立つため，$t$ で微分して
$$
\frac{\partial H}{\partial \mathbf{x}}(\mathbf{x}(t), t) \frac{\mathrm{d} \mathbf{x}}{\mathrm{~d} t}+\frac{\partial H}{\partial t}(\mathbf{x}(t), t)=\mathbf{0}
$$

を得る．ここで
$$
J_{H}(\mathbf{x}, t):=\frac{\partial H}{\partial \mathbf{x}}(\mathbf{x}, t) \in \mathbb{C}^{\nu \times \nu}
$$

を $\mathbf{x}$ に関するヤコビ行列とする。 $J_{H}(\mathbf{x}(t), t)$ が正則で ある $t$ に対しては，
$$
\frac{\mathrm{d} \mathbf{x}}{\mathrm{~d} t}=-J_{H}(\mathbf{x}(t), t)^{-1} \frac{\partial H}{\partial t}(\mathbf{x}(t), t)
$$

により接ベクトルが与えられる。
予測子•修正子法では，既知の近似解 $\mathbf{x}_{i} \approx \mathbf{x}\left(t_{i}\right)$ か ら，刻み幅 $\Delta t>0$ を用いて予測子（例：オイラー法）
$$
\tilde{\mathbf{x}}_{i+1}=\mathbf{x}_{i}+\left.\Delta t \frac{\mathrm{~d} \mathbf{x}}{\mathrm{~d} t}\right|_{\mathbf{x}_{i}, t_{i}}, \quad t_{i+1}=t_{i}+\Delta t
$$

を計算する。続いて，$t=t_{i+1}$ を固定した方程式 $H\left(\mathbf{x}, t_{i+1}\right)=\mathbf{0}$ に対し，ニュートン法を修正子として適用して予測値を次の反復により補正する。
$$
\begin{aligned}
\mathbf{x}^{(0)} & =\tilde{\mathbf{x}}_{i+1} \\
\mathbf{x}^{(j+1)} & =\mathbf{x}^{(j)}-J_{H}\left(\mathbf{x}^{(j)}, t_{i+1}\right)^{-1} H\left(\mathbf{x}^{(j)}, t_{i+1}\right)
\end{aligned}
$$

この反復を収束するまで行い，得られた解を $\mathrm{x}_{i+1}$ とす る．一般に，追跡途中で $H(\cdot, t)$ が重解を持つ（あるい はそれに近い）状況では，$J_{H}$ が特異に近づき，条件数が増大しやすい。この場合，ニュートン法の収束域が狭く なり，刻み幅 $\Delta t$ の縮小や反復回数の増加を招くため，追跡が困難になることがある［ 4,5 ］．

\section*{4．提案手法}

この節では，$n$ 次一変数複素係数多項式に対するホモ トピー連続法を対象として，係数空間 $\mathbb{C}^{n+1}$ 上のホモト ピーパスを低次数のベジェ曲線で表現し，その中間制御点を強化学習により最適化する枠組みを述べる。目的は， パス追跡の成功率を維持しつつ，全ての解の追跡に要す るステップ数（受理ステップ数と棄却ステップ数）を削減することである。

\section*{4.1 問題設定（一変数複素係数多項式）}

一変数 $n$ 次複素係数多項式
$f(x)=c_{n} x^{n}+c_{n-1} x^{n-1}+\cdots+c_{0}, \quad c_{i} \in \mathbb{C}(i=0, \ldots, n)$ を目的多項式とし，係数ベクトルを
$$
\mathbf{c}_{f}:=\left(c_{0}, c_{1}, \ldots, c_{n}\right) \in \mathbb{C}^{n+1}
$$

と表す。開始多項式として
$$
g(x)=x^{n}-1
$$

を用い，その係数ベクトルを $\mathbf{c}_{g} \in \mathbb{C}^{n+1}$ とする。この とき，$g(x)=0$ は
$$
\omega_{k}=\exp \left(\frac{2 \pi i k}{n}\right), \quad k=0,1, \ldots, n-1
$$

で与えられる $n$ 個の相異なる解を持つ。ここで $i$ は虚数単位（ $i^{2}=-1$ ）である。各エピソードでは，開始多項式は $g$ で固定し，目的多項式 $f$ の係数をランダムにサン プリングする。具体的には各 $i=0, \ldots, n$ に対し，実部 と虚部をそれぞれ
$$
\Re\left(c_{i}\right), \Im\left(c_{i}\right) \stackrel{\text { i.i.d. }}{\sim} \mathcal{U}[-b, b]
$$

とし，$b>0$ は定数（例：$b=5$ ）とする。

\section*{4.2 ベジェ曲線と制御点のパラメータ化}

ベジェ次数を $d_{b} \geq 2$ とし，制御点 $P_{0}, P_{1}, \ldots, P_{d_{b}} \in \mathbb{C}^{n+1}$ を導入する。端点は
$$
P_{0}=\mathbf{c}_{g}, \quad P_{d_{b}}=\mathbf{c}_{f}
$$

と固定する．パラメータ $t \in[0,1]$ に対し，次数 $d_{b}$ のベ ジェ曲線
$$
B(t)=\sum_{k=0}^{d_{b}}\binom{d_{b}}{k} t^{k}(1-t)^{d_{b}-k} P_{k} \in \mathbb{C}^{n+1}
$$

を係数空間上のパスとして用いる．このとき $B(0)= P_{0}, B(1)=P_{d_{b}}$ が成り立つ。本研究では，中間制御点
$P_{1}, \ldots, P_{d_{b}-1}$ を最適化対象とする。各 $k=1, \ldots, d_{b}-1$ に対し，端点の線形補間
$$
\bar{P}_{k}:=\left(1-\frac{k}{d_{b}}\right) P_{0}+\frac{k}{d_{b}} P_{d_{b}}
$$

を基準点とし，摂動 $\boldsymbol{\delta}_{k} \in \mathbb{C}^{n+1}$ により
$$
P_{k}=\bar{P}_{k}+\boldsymbol{\delta}_{k}
$$

と定める。

\section*{$4.3 \quad \mathrm{PPO}$ による制御点最適化}

エピソード長は $T=1$ とする。すなわち， 1 エピソー ドで 1 つの問題 $(g, f)$ を固定し，エージェントは 1 回の行動出力により中間制御点 $P_{1}, \ldots, P_{d_{b}-1}$ を決定する。 その後，得られたベジェ曲線に対してパス追跡を実行 し，得られた追跡コストに基づき報酬を与えて終了する。状態，行動，追跡コストと報酬はそれぞれ以下の通りで ある。

状態。開始多項式と目的多項式の係数を状態 $\mathrm{s}:=$ （ $\mathbf{c}_{g}, \mathbf{c}_{f}$ ）とする。
行動．潜在次元 $m$ を導入し，行動を行列 $A \in \mathbb{R}^{m \times\left(d_{b}-1\right)}$ とする。ここで $m$ は中間制御点の摂動を表現する自由度であり，$m$ が大きいほど表現力は増す一方で，探索 は難しくなる。また $m$ を固定することで，ベジェ次数 $d_{b}$ に対して行動の次元を固定できる。 $A$ の第 $k$ 列を $\mathbf{a}_{k} \in \mathbb{R}^{m}\left(k=1, \ldots, d_{b}-1\right)$ とし，あらかじめ定めた行列 $W \in \mathbb{C}^{(n+1) \times m}$ により線形写像 $\Phi: \mathbb{R}^{m} \rightarrow \mathbb{C}^{n+1}$ を
$$
\Phi(\mathbf{a}):=W \mathbf{a}
$$

と定める．このとき
$$
\boldsymbol{\delta}_{k}=\Phi\left(\mathbf{a}_{k}\right), \quad k=1, \ldots, d_{b}-1
$$

として摂動を生成する。
追跡コストと報酬．全ての解の追跡を実行したとき，受理ステップ数を $N_{\mathrm{acc}}$ ，棄却ステップ数を $N_{\mathrm{rej}}$ とする。 ここで 1 ステップは，予測子•修正子法を 1 サイクル実行することに対応する。成功時の追跡コストを
$$
J_{\mathrm{bez}}:=N_{\mathrm{acc}}+\rho N_{\mathrm{rej}}, \quad \rho \geq 0
$$

と定義し，失敗時は定数ペナルティ $M$ を用いて $J_{\mathrm{bez}}:= M$ とする。同一の追跡設定で，線形パスを用いたとき のコストを $J_{\text {lin }}$ とおく。エピソード長が 1 のため，報酬 は終端でのみ与え，
$$
r:=\mu\left(J_{\mathrm{lin}}-J_{\mathrm{bez}}\right), \quad \mu>0
$$

とする。これにより，線形パスに比べてベジェ曲線が追跡コスト $J_{\mathrm{bez}}$ を減少させるほど正の報酬が得られる。方策最適化には PPO［7］を用いる。

\section*{5．実験}

実験設定。多項式次数は $n \in\{5,10,20,40,80\}$ ，ベジェ次数は $d_{b} \in\{2,3,4\}$ とする。各エピソードでは開始多

項式 $g$ を固定し，目的多項式 $f$ をサンプリングする。エ ピソード長は $T=1$ とし， 1 回の行動でベジェ曲線を決定した後にパス追跡を実行して終了する。追跡コストは $J=N_{\mathrm{acc}}+\rho N_{\mathrm{rej}}(\rho=1)$ とし，追跡失敗時は $J=M$ （ $M=3000$ ）のペナルティを与える。PPO の学習設定 は，総学習ステップ数 $5 \times 10^{6}$ ，ロールアウト長 2048 ，学習率 $3 \times 10^{-4}$ とした。

評価方法。学習済み方策が生成するベジェ曲線と線形 パスを，同一の追跡設定の下で比較する。評価用データ セットには，学習時と同一の分布から独立にサンプリン グした 1000 個の目的多項式を利用する。そして，評価指標として成功率とステップ数（受理ステップ数と棄却 ステップ数の合計）を用いる。

追跡結果の比較。成功率は全条件で $100 \%$ であったた め，以降では全ての解の追跡に要するステップ数に着目 して比較する。表1は，ステップ数の平均，中央値，標準偏差，最小値，最大値を示している．

全体として，ベジェ曲線は多くの次数でステップ数を減少させ，とくに $d_{b}=4$ が一貫して良好であった．平均および中央値の観点では，$n=10,20,40,80$ において $d_{b}=4$ が最小となり，線形パスと比べて概ね $15-19 \%$ の短縮が得られた。標準偏差についても，多くの次数で $d_{b}=4$ が最小となり，ばらつきの減少が確認できる。最小値も $n=10,20,40,80$ で $d_{b}=4$ が最小であり，最良 ケースでは線形パスと同等以上の効率が得られている。

一方で最大値を見ると，$n=10,20,80$ では $d_{b}=4$ が最大値まで改善しているのに対し，$n=5$ および $n=40$ では線形パスがわずかに小さかった（ $n=40$ の差は小 さい）。また $d_{b}=3$ は平均および中央値では改善する ものの，いくつかの次数で最大値が増加しており，外れ値が生じやすい可能性がある。以上より，本設定では自由度の大きい $d_{b}=4$ が，難所の回避を通じてステップ数の削減と安定化に寄与している可能性がある。

\section*{6．おわりに}

本研究では，一変数複素係数多項式に対するホモト ピー連続法において，係数空間上のホモトピーパスを低次数のベジェ曲線で表現し，中間制御点を PPO により最適化する枠組みを提案した。実験では成功率を $100 \%$ に保ったまま，特に $d_{b}=4$ において多くの次数でステッ プ数の平均および標準偏差が改善し，追跡の効率化と安定化の可能性を示した。今後の研究では，多変数多項式系へ拡張し，条件数の悪化や path jumping が顕在化す る設定においても，ホモトピーパスの最適化が有効であ るかを検証する。

\section*{謝辞}

本研究は JSPS 科研費 JP23K24914，JP23KK0208， JST BOOST（若手研究者支援）JPMJBY24C6，JST さきがけ JPMJPR24K4，および ANR－24－CE46－7295 の助成を受けたものです。

\begin{table}
\captionsetup{labelformat=empty}
\caption{表 1：線形パスとベジェ曲線における全解の追跡に要するステップ数の比較。ここで 1 ステップは，予測子•修正子を 1 サイクル実行することに対応する。評価データセットのサイズは 1000 であり，括弧内はステップ数の改善率を表す。}
\begin{tabular}{|l|l|l|l|l|l|l|}
\hline \multirow{2}{*}{degree} & \multirow{2}{*}{path} & \multicolumn{5}{|c|}{\＃steps ↓} \\
\hline & & mean & median & std & min & max \\
\hline \multirow{4}{*}{5} & Linear & 82.4 & 75.0 & 33.3 & 33.0 & 225.0 \\
\hline & Bézier（ $d_{b}=2$ ） & 73.8 （－10．4\％） & 67.5 （－10．0\％） & 24.9 （－25．1\％） & 36.0 （＋9．1\％） & 196.0 （－12．9\％） \\
\hline & Bézier（ $d_{b}=3$ ） & 65.2 （－20．8\％） & 60.0 （－20．0\％） & 25.6 （－23．1\％） & 30.0 （－9．1\％） & 267.0 （＋18．7\％） \\
\hline & Bézier（ $d_{b}=4$ ） & 69.4 （－15．8\％） & 64.0 （－14．7\％） & 23.2 （－30．2\％） & 29.0 （－12．1\％） & 226.0 （＋0．4\％） \\
\hline \multirow{4}{*}{10} & Linear & 190.6 & 181.0 & 57.1 & 97.0 & 470.0 \\
\hline & Bézier（ $d_{b}=2$ ） & 178.7 （－6．2\％） & 170.0 （－6．1\％） & 48.3 （－15．4\％） & 105.0 （＋8．2\％） & 429.0 （－8．7\％） \\
\hline & Bézier（ $d_{b}=3$ ） & 175.0 （－8．2\％） & 166.0 （－8．3\％） & 49.4 （－13．5\％） & 102.0 （＋5．2\％） & 573.0 （＋21．9\％） \\
\hline & Bézier（ $d_{b}=4$ ） & 161.5 （－15．7\％） & 152.0 （－16．3\％） & 43.5 （－23．9\％） & 92.0 （－5．2\％） & 368.0 （－21．5\％） \\
\hline \multirow{4}{*}{20} & Linear & 426.3 & 404.0 & 107.0 & 241.0 & 994.0 \\
\hline & Bézier（ $d_{b}=2$ ） & 407.7 （－4．4\％） & 389.0 （－3．7\％） & 90.0 （－15．9\％） & 261.0 （＋8．3\％） & 805.0 （－19．0\％） \\
\hline & Bézier（ $d_{b}=3$ ） & 371.5 （－12．9\％） & 356.0 （－11．9\％） & 80.3 （－25．0\％） & 237.0 （－1．7\％） & 784.0 （－21．1\％） \\
\hline & Bézier（ $d_{b}=4$ ） & 344.8 （－19．1\％） & 328.0 （－18．8\％） & 79.3 （－25．9\％） & 206.0 （－14．5\％） & 686.0 （－31．0\％） \\
\hline \multirow{4}{*}{40} & Linear & 914.5 & 883.0 & 172.1 & 610.0 & 1638.0 \\
\hline & Bézier（ $d_{b}=2$ ） & 892.1 （－2．4\％） & 861.0 （－2．5\％） & 157.6 （－8．4\％） & 620.0 （＋1．6\％） & 1583.0 （－3．4\％） \\
\hline & Bézier（ $d_{b}=3$ ） & 815.8 （－10．8\％） & 780.5 （－11．6\％） & 150.8 （－12．3\％） & 574.0 （－5．9\％） & 1766.0 （＋7．8\％） \\
\hline & Bézier（ $d_{b}=4$ ） & 756.1 （－17．3\％） & 726.0 （－17．8\％） & 138.4 （－19．6\％） & 527.0 （－13．6\％） & 1678.0 （＋2．4\％） \\
\hline \multirow{4}{*}{80} & Linear & 1942.5 & 1908.5 & 271.1 & 1427.0 & 3548.0 \\
\hline & Bézier（ $d_{b}=2$ ） & 1917.2 （－1．3\％） & 1881.5 （－1．4\％） & 248.3 （－8．4\％） & 1459.0 （＋2．2\％） & 2923.0 （－17．6\％） \\
\hline & Bézier（ $d_{b}=3$ ） & 1743.9 （－10．2\％） & 1705.0 （－10．7\％） & 248.8 （－8．2\％） & 1265.0 （－11．3\％） & 2941.0 （－17．1\％） \\
\hline & Bézier（ $d_{b}=4$ ） & 1637.9 （－15．7\％） & 1598.0 （－16．3\％） & 242.7 （－10．5\％） & 1213.0 （－15．0\％） & 2786.0 （－21．5\％） \\
\hline
\end{tabular}
\end{table}

\section*{参考文献}
［1］Chiang－Heng Chien，Hongyi Fan，Ahmad Ab－ delfattah，Elias Tsigaridas，Stanimire Tomov， and Benjamin Kimia．Gpu－based homotopy con－ tinuation for minimal problems in computer vi－ sion．In Proceedings of the IEEE／CVF Confer－ ence on Computer Vision and Pattern Recogni－ tion，pages 15765－15776， 2022.
［2］Elizabeth Gross，Brent Davis，Kenneth L Ho， Daniel J Bates，and Heather A Harrington．Nu－ merical algebraic geometry for model selection and its application to the life sciences．Journal of The Royal Society Interface，13（123）：20160256， 2016.
［3］Charles W．Wampler and Andrew J．Sommese． Numerical algebraic geometry and algebraic kinematics．Acta Numerica，20：469－567， 2011.
［4］Carlos Beltrán and Anton Leykin．Certified nu－ merical homotopy tracking．Experimental Math－ ematics，21（1）：69－83， 2012.
［5］Sascha Timme．Mixed precision path tracking for polynomial homotopy continuation．Advances in Computational Mathematics，47（5）：75， 2021.
［6］Michael Shub．Complexity of bézout＇s theorem vi：Geodesics in the condition（number）met－ ric．Foundations of Computational Mathematics， 9（2）：171－178， 2009.
［7］John Schulman，Filip Wolski，Prafulla Dhari－ wal，Alec Radford，and Oleg Klimov．Proximal
policy optimization algorithms．arXiv preprint arXiv：1707．06347， 2017.
［8］Bao Duy Tran．Optimal path homotopy for univariate polynomials．arXiv preprint arXiv：2008．01465， 2020.
［9］Carlos Beltrán and Michael Shub．Complexity of bézout＇s theorem vii：Distance estimates in the condition metric．Foundations of Computational Mathematics，9（2）：179－195， 2009.
［10］Petr Hruby，Timothy Duff，Anton Leykin，and Tomas Pajdla．Learning to solve hard minimal problems．In Proceedings of the IEEE／CVF Con－ ference on Computer Vision and Pattern Recog－ nition，pages 5532－5542， 2022.
［11］Xinyue Zhang，Zijia Dai，Wanting Xu，and Lau－ rent Kneip．Simulator hc：Regression－based online simulation of starting problem－solution pairs for homotopy continuation in geometric vi－ sion．In Proceedings of the Computer Vision and Pattern Recognition Conference，pages 27103－ 27112， 2025.
［12］Jiayao Mai，Bangyan Liao，Zhenjun Zhao，Ying－ ping Zeng，Haoang Li，Javier Civera，Tailin Wu， Yi Zhou，and Peidong Liu．Neural predictor－ corrector：Solving homotopy problems with re－ inforcement learning．In The Fourteenth Interna－ tional Conference on Learning Representations， 2026.