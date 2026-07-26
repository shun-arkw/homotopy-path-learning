## 1．問題設定

$n$変数$n$本の複素多項式系を対象とする．変数ベクトルを

$$
\boldsymbol{x}
=
(x_1,\ldots,x_n)
\in\mathbb{C}^n
$$

とする．

標準基底ベクトルを

$$
\boldsymbol{e}_1,\ldots,\boldsymbol{e}_n
\in\mathbb{Z}_{\geq0}^n
$$

とし，$\boldsymbol{e}_i$は第$i$成分のみが$1$であるベクトルとする．

目的系を

$$
F(\boldsymbol{x})
=
\begin{pmatrix}
f_1(\boldsymbol{x})\\
\vdots\\
f_n(\boldsymbol{x})
\end{pmatrix}
=
\boldsymbol{0}
$$

とする．

第$i$方程式の先頭単項式を$x_i^{d_i}$とする．ただし，
$d_i\in\mathbb{Z}_{>0}$である．先頭単項式以外に使用する単項式の指数集合を

$$
\mathcal{A}_i
=
\left\{
\boldsymbol{\alpha}_{i,1},
\ldots,
\boldsymbol{\alpha}_{i,m_i}
\right\}
\subset\mathbb{Z}_{\geq0}^n
$$

とする．ただし，$d_i\boldsymbol{e}_i\notin\mathcal{A}_i$とする．
多重指数$\boldsymbol{\alpha}=(\alpha_1,\ldots,\alpha_n)$に対して，

$$
\boldsymbol{x}^{\boldsymbol{\alpha}}
=
x_1^{\alpha_1}\cdots x_n^{\alpha_n}
$$

と定義する．

各目的多項式を

$$
f_i(\boldsymbol{x})
=
x_i^{d_i}
+
\sum_{\boldsymbol{\alpha}\in\mathcal{A}_i}
c_{i,\boldsymbol{\alpha}}
\boldsymbol{x}^{\boldsymbol{\alpha}},
\qquad
i=1,\ldots,n
$$

とする．

固定した単項式順序$\prec$に対して，

$$
\boldsymbol{x}^{\boldsymbol{\alpha}}
\prec
x_i^{d_i},
\qquad
\boldsymbol{\alpha}\in\mathcal{A}_i
$$

を仮定する．このとき，

$$
\operatorname{LM}(f_i)=x_i^{d_i}
$$

である．また，$i\neq j$に対して，

$$
\gcd\left(x_i^{d_i},x_j^{d_j}\right)=1
$$

が成り立つ．したがって，Buchbergerの判定法により
$\{f_1,\ldots,f_n\}$はGröbner基底であり，初期イデアルは

$$
\operatorname{in}\langle f_1,\ldots,f_n\rangle
=
\left\langle
x_1^{d_1},\ldots,x_n^{d_n}
\right\rangle
$$

である．本稿では，互いに素な純冪先頭単項式を保つ多項式系を，
Pham型サポートをもつ多項式系と呼ぶ．

次数適合的な単項式順序，すなわち
$|\boldsymbol{\alpha}|<|\boldsymbol{\alpha}'|$なら
$\boldsymbol{x}^{\boldsymbol{\alpha}}\prec\boldsymbol{x}^{\boldsymbol{\alpha}'}$
となる単項式順序を使用する場合は，十分条件として

$$
|\boldsymbol{\alpha}|
=
\sum_{j=1}^n\alpha_j
<
d_i
$$

を課すことができる．


## 2．固定する指数集合

第$i$方程式で使用する候補指数集合を

$$
\widetilde{\mathcal{A}}_i
=
\left\{
d_i\boldsymbol{e}_i
\right\}
\cup
\mathcal{A}_i
$$

とする．

実験中は$\widetilde{\mathcal{A}}_i$を固定し，
$\boldsymbol{\alpha}\in\mathcal{A}_i$に対応する自由係数のみを変更する．
各自由係数は，例えば

$$
\operatorname{Re}
\left(
c_{i,\boldsymbol{\alpha}}
\right),
\quad
\operatorname{Im}
\left(
c_{i,\boldsymbol{\alpha}}
\right)
\overset{\mathrm{i.i.d.}}{\sim}
\operatorname{Unif}[-b,b]
$$

により生成する．先頭単項式$x_i^{d_i}$の係数は常に$1$に固定する．


## 3．開始系

開始系を

$$
G(\boldsymbol{x})
=
\begin{pmatrix}
g_1(\boldsymbol{x})\\
\vdots\\
g_n(\boldsymbol{x})
\end{pmatrix}
=
\boldsymbol{0}
$$

とし，各開始多項式を

$$
g_i(\boldsymbol{x})
=
x_i^{d_i}-1,
\qquad
i=1,\ldots,n
$$

と定義する．すなわち，

$$
G(\boldsymbol{x})
=
\begin{pmatrix}
x_1^{d_1}-1\\
x_2^{d_2}-1\\
\vdots\\
x_n^{d_n}-1
\end{pmatrix}
$$

である．

各方程式$g_i(\boldsymbol{x})=0$の解は

$$
x_i
=
\exp\left(
\frac{2\pi\mathrm{i}k_i}{d_i}
\right),
\qquad
k_i\in\{0,\ldots,d_i-1\}
$$

である．添字集合を

$$
\mathcal{K}
=
\prod_{i=1}^n
\{0,\ldots,d_i-1\}
\subset\mathbb{Z}^n
$$

とする．$\boldsymbol{k}=(k_1,\ldots,k_n)\in\mathcal{K}$に対して，

$$
\boldsymbol{x}_{\boldsymbol{k}}(0)
=
\left(
\exp\left(\frac{2\pi\mathrm{i}k_1}{d_1}\right),
\ldots,
\exp\left(\frac{2\pi\mathrm{i}k_n}{d_n}\right)
\right)
$$

と定義する．開始解集合は

$$
\mathcal{S}_G
=
\left\{
\boldsymbol{x}_{\boldsymbol{k}}(0)
\mid
\boldsymbol{k}\in\mathcal{K}
\right\}
$$

で与えられる．

開始解の総数，すなわち追跡する解パスの本数は

$$
N_{\mathrm{path}}
=
\prod_{i=1}^n d_i
$$

である．

また，目的系$F$についても
$\operatorname{in}\langle f_1,\ldots,f_n\rangle
=\langle x_1^{d_1},\ldots,x_n^{d_n}\rangle$
であるため，剰余環の標準単項式は

$$
\boldsymbol{x}^{\boldsymbol{\alpha}},
\qquad
\boldsymbol{\alpha}\in\mathbb{Z}_{\geq0}^n,
\quad
0\leq \alpha_i<d_i
\quad
(i=1,\ldots,n)
$$

で与えられる．したがって，$F$の孤立解の個数は重複度込みで
$\prod_{i=1}^n d_i$である．


## 4．係数ベクトル

第$i$方程式の候補指数集合を

$$
\widetilde{\mathcal{A}}_i
=
\left\{
\boldsymbol{\alpha}_{i,0},
\boldsymbol{\alpha}_{i,1},
\ldots,
\boldsymbol{\alpha}_{i,m_i}
\right\}
$$

と並べる．ただし，

$$
\boldsymbol{\alpha}_{i,0}
=
d_i\boldsymbol{e}_i,
\qquad
\left\{
\boldsymbol{\alpha}_{i,1},
\ldots,
\boldsymbol{\alpha}_{i,m_i}
\right\}
=
\mathcal{A}_i
$$

とする．先頭係数を含む全係数数を

$$
M
=
\sum_{i=1}^n (m_i+1)
$$

とする．第$i$方程式の係数ベクトルを

$$
\boldsymbol{c}_{F,i}
=
\left(
c_{i,\boldsymbol{\alpha}_{i,0}},
c_{i,\boldsymbol{\alpha}_{i,1}},
\ldots,
c_{i,\boldsymbol{\alpha}_{i,m_i}}
\right)
\in\mathbb{C}^{m_i+1}
$$

とする．ここで，先頭係数は

$$
c_{i,\boldsymbol{\alpha}_{i,0}}
=
c_{i,d_i\boldsymbol{e}_i}
=
1
$$

に固定する．

全方程式の係数を連結した目的系の係数ベクトルを

$$
\boldsymbol{c}_F
=
\left(
\boldsymbol{c}_{F,1},
\ldots,
\boldsymbol{c}_{F,n}
\right)
\in\mathbb{C}^M
$$

と定義する．

各$\mathcal{A}_i$は定数項に対応する指数

$$
\boldsymbol{0}
=
(0,\ldots,0)
$$

を含むものとする．開始系の第$i$係数ベクトルを

$$
\boldsymbol{c}_{G,i}
=
\left(
c_{G,i,\boldsymbol{\alpha}_{i,0}},
c_{G,i,\boldsymbol{\alpha}_{i,1}},
\ldots,
c_{G,i,\boldsymbol{\alpha}_{i,m_i}}
\right)
\in\mathbb{C}^{m_i+1}
$$

とし，各成分を

$$
c_{G,i,\boldsymbol{\alpha}}
=
\begin{cases}
1,
&
\boldsymbol{\alpha}=d_i\boldsymbol{e}_i,
\\
-1,
&
\boldsymbol{\alpha}=\boldsymbol{0},
\\
0,
&
\text{otherwise}
\end{cases}
$$

と定義する．

開始系全体の係数ベクトルは

$$
\boldsymbol{c}_G
=
\left(
\boldsymbol{c}_{G,1},
\ldots,
\boldsymbol{c}_{G,n}
\right)
\in\mathbb{C}^M
$$

である．


## 5．ベジェ曲線による係数パス

ベジェ曲線の次数を$d_b$とし，制御点を

$$
P_0,P_1,\ldots,P_{d_b}
\in\mathbb{C}^M
$$

とする．端点を

$$
P_0=\boldsymbol{c}_G,
\qquad
P_{d_b}=\boldsymbol{c}_F
$$

に固定する．
また，すべての制御点$P_k$と任意の$i$について，第$i$ブロックの
$d_i\boldsymbol{e}_i$成分を$1$に固定する．これにより，
先頭係数はパス全体で変化しない．

係数空間上のベジェ曲線を

$$
\boldsymbol{c}(t)
=
\sum_{k=0}^{d_b}
\binom{d_b}{k}
t^k(1-t)^{d_b-k}P_k,
\qquad
t\in[0,1]
$$

と定義する．

このとき，

$$
\boldsymbol{c}(0)=\boldsymbol{c}_G,
\qquad
\boldsymbol{c}(1)=\boldsymbol{c}_F
$$

が成り立つ．

係数ベクトルを方程式ごとのブロックに分割し，

$$
\boldsymbol{c}(t)
=
\left(
\boldsymbol{c}_1(t),
\ldots,
\boldsymbol{c}_n(t)
\right)
$$

と表す．第$i$ブロックの$\boldsymbol{\alpha}\in\widetilde{\mathcal{A}}_i$
に対応する成分を$c_{i,\boldsymbol{\alpha}}(t)$とする．


## 6．ベジェホモトピー

係数パス$\boldsymbol{c}(t)$から，多項式ホモトピーを

$$
H(\boldsymbol{x},t)
=
\begin{pmatrix}
H_1(\boldsymbol{x},t)\\
\vdots\\
H_n(\boldsymbol{x},t)
\end{pmatrix}
=
\boldsymbol{0}
$$

と定義する．各方程式は

$$
H_i(\boldsymbol{x},t)
=
\sum_{\boldsymbol{\alpha}\in\widetilde{\mathcal{A}}_i}
c_{i,\boldsymbol{\alpha}}(t)
\boldsymbol{x}^{\boldsymbol{\alpha}},
\qquad
i=1,\ldots,n
$$

である．

端点では，

$$
H(\boldsymbol{x},0)=G(\boldsymbol{x}),
\qquad
H(\boldsymbol{x},1)=F(\boldsymbol{x})
$$

が成り立つ．

先頭単項式$x_i^{d_i}$の係数はパス全体で$1$に固定されているため，

$$
\operatorname{LM}\left(H_i(\cdot,t)\right)
=
x_i^{d_i},
\qquad
t\in[0,1]
$$

である．したがって，係数パスは固定したPham型サポートをもつ
多項式族の内部を移動する．

ただし，この性質だけでは追跡中の正則性は保証されない．
係数$\boldsymbol{c}$に対応する多項式系を$H_{\boldsymbol{c}}$と書き，
判別集合を

$$
\Sigma
=
\left\{
\boldsymbol{c}\in\mathbb{C}^M
\mid
\exists\boldsymbol{x}\in\mathbb{C}^n,\,
H_{\boldsymbol{c}}(\boldsymbol{x})=\boldsymbol{0},\,
\det J_{\boldsymbol{x}}H_{\boldsymbol{c}}(\boldsymbol{x})=0
\right\}
$$

とする．通常の予測子・修正子法で終点直前まで正則に追跡するためには，

$$
\boldsymbol{c}([0,1))\cap\Sigma=\emptyset
$$

を仮定する．さらに終点も非特異であることを要求する場合は，
$\boldsymbol{c}([0,1])\cap\Sigma=\emptyset$を仮定する．終点
$\boldsymbol{c}_F$が$\Sigma$に属する場合は，$t=1$で特異解に到達する
可能性があるため，終端処理を別に扱う必要がある．


## 7．ベジェ曲線の微分

係数パスの$t$に関する微分は

$$
\dot{\boldsymbol{c}}(t)
=
d_b
\sum_{k=0}^{d_b-1}
\binom{d_b-1}{k}
t^k(1-t)^{d_b-1-k}
\left(
P_{k+1}-P_k
\right)
$$

である．

したがって，各ホモトピー方程式の$t$に関する偏微分は

$$
\frac{\partial H_i}{\partial t}
(\boldsymbol{x},t)
=
\sum_{\boldsymbol{\alpha}\in\widetilde{\mathcal{A}}_i}
\dot{c}_{i,\boldsymbol{\alpha}}(t)
\boldsymbol{x}^{\boldsymbol{\alpha}}
$$

である．ただし，先頭係数は固定されているため
$\dot{c}_{i,d_i\boldsymbol{e}_i}(t)=0$である．


## 8．ヤコビ行列

$\boldsymbol{x}$に関するヤコビ行列を

$$
J_{\boldsymbol{x}}H(\boldsymbol{x},t)
=
\left[
\frac{\partial H_i}{\partial x_j}
(\boldsymbol{x},t)
\right]_{i,j=1}^n
\in\mathbb{C}^{n\times n}
$$

と定義する．

各成分は

$$
\frac{\partial H_i}{\partial x_j}
(\boldsymbol{x},t)
=
\sum_{\substack{
\boldsymbol{\alpha}\in\widetilde{\mathcal{A}}_i\\
\alpha_j\geq1
}}
c_{i,\boldsymbol{\alpha}}(t)
\alpha_j
\boldsymbol{x}^{\boldsymbol{\alpha}-\boldsymbol{e}_j}
$$

である．


## 9．解曲線の接ベクトル

解曲線$\boldsymbol{x}(t)$は

$$
H(\boldsymbol{x}(t),t)
=
\boldsymbol{0}
$$

を満たす．両辺を$t$で微分すると，

$$
J_{\boldsymbol{x}}H(\boldsymbol{x}(t),t)
\frac{d\boldsymbol{x}}{dt}
+
\frac{\partial H}{\partial t}
(\boldsymbol{x}(t),t)
=
\boldsymbol{0}
$$

を得る．

追跡点$\boldsymbol{x}(t)$において
$J_{\boldsymbol{x}}H(\boldsymbol{x}(t),t)$が正則である場合，接ベクトルは

$$
\frac{d\boldsymbol{x}}{dt}
=
-
J_{\boldsymbol{x}}H(\boldsymbol{x}(t),t)^{-1}
\frac{\partial H}{\partial t}
(\boldsymbol{x}(t),t)
$$

で与えられる．

実装上は逆行列を明示的に計算せず，

$$
J_{\boldsymbol{x}}H(\boldsymbol{x}(t),t)
\boldsymbol{v}(t)
=
-
\frac{\partial H}{\partial t}
(\boldsymbol{x}(t),t)
$$

を解き，

$$
\boldsymbol{v}(t)
=
\frac{d\boldsymbol{x}}{dt}
$$

とする．


## 10．予測子・修正子法

時刻$t_\ell$における近似解を$\boldsymbol{x}_\ell$とし，刻み幅を$\Delta t$とする．オイラー予測子は

$$
\widetilde{\boldsymbol{x}}_{\ell+1}
=
\boldsymbol{x}_\ell
+
\Delta t\,
\boldsymbol{v}(t_\ell)
$$

である．次の時刻は

$$
t_{\ell+1}
=
t_\ell+\Delta t
$$

である．

修正子では，$t=t_{\ell+1}$を固定し，ニュートン方程式

$$
J_{\boldsymbol{x}}H
\left(
\boldsymbol{x}^{(r)},t_{\ell+1}
\right)
\Delta\boldsymbol{x}^{(r)}
=
-
H
\left(
\boldsymbol{x}^{(r)},t_{\ell+1}
\right)
$$

を解き，

$$
\boldsymbol{x}^{(r+1)}
=
\boldsymbol{x}^{(r)}
+
\Delta\boldsymbol{x}^{(r)}
$$

と更新する．初期値は

$$
\boldsymbol{x}^{(0)}
=
\widetilde{\boldsymbol{x}}_{\ell+1}
$$

である．


## 11．中間制御点のパラメータ化

第$k$中間制御点の線形補間上の基準点を

$$
\overline{P}_k
=
\left(
1-\frac{k}{d_b}
\right)\boldsymbol{c}_G
+
\frac{k}{d_b}\boldsymbol{c}_F,
\qquad
k=1,\ldots,d_b-1
$$

とする．

中間制御点を

$$
P_k
=
\overline{P}_k+\boldsymbol{\delta}_k
$$

と定義する．ここで，$\boldsymbol{\delta}_k\in\mathbb{C}^M$はPPOにより決定する摂動である．
ただし，先頭係数固定制約を保つため，任意の$i,k$について

$$
\delta_{k,i,d_i\boldsymbol{e}_i}
=
0
$$

とする．

複素ベクトルを実数ベクトルへ変換する写像を

$$
\mathcal{R}:\mathbb{C}^M\rightarrow\mathbb{R}^{2M}
$$

とし，

$$
\mathcal{R}(\boldsymbol{z})
=
\begin{pmatrix}
\operatorname{Re}\boldsymbol{z}\\
\operatorname{Im}\boldsymbol{z}
\end{pmatrix}
$$

と定義する．

潜在次元を$m$とし，固定行列

$$
U\in\mathbb{R}^{2M\times m}
$$

を導入する．先頭係数に対応する実部・虚部の行は$0$に固定し，
$U\boldsymbol{z}_k$が先頭係数成分を変化させないようにする．
PPOが出力する潜在行動を

$$
\boldsymbol{z}_k\in\mathbb{R}^m
$$

とし，摂動を

$$
\mathcal{R}(\boldsymbol{\delta}_k)
=
U\boldsymbol{z}_k
$$

により生成する．すなわち，

$$
\boldsymbol{\delta}_k
=
\mathcal{R}^{-1}
\left(
U\boldsymbol{z}_k
\right)
$$

である．

全行動を

$$
\boldsymbol{a}
=
\begin{pmatrix}
\boldsymbol{z}_1\\
\vdots\\
\boldsymbol{z}_{d_b-1}
\end{pmatrix}
\in\mathbb{R}^{m(d_b-1)}
$$

とする．


## 12．PPOの状態

開始系$G$を固定する場合，状態を目的系の係数ベクトルにより

$$
\boldsymbol{s}
=
\mathcal{R}(\boldsymbol{c}_F)
=
\begin{pmatrix}
\operatorname{Re}\boldsymbol{c}_F\\
\operatorname{Im}\boldsymbol{c}_F
\end{pmatrix}
\in\mathbb{R}^{2M}
$$

と定義する．
ただし，先頭係数成分は常に固定されているため，
学習入力ではこれらの固定成分を除いてもよい．

初期の1ステップGymnasium環境では，固定成分を除いた自由係数のみを観測に用いる．
`free_coefficient_mask`で抽出した

$$
\boldsymbol{c}_{F,\mathrm{free}}
$$

に対して，観測を

$$
\boldsymbol{s}
=
\begin{pmatrix}
\operatorname{Re}\boldsymbol{c}_{F,\mathrm{free}}\\
\operatorname{Im}\boldsymbol{c}_{F,\mathrm{free}}
\end{pmatrix}
$$

とする．返却配列のdtypeは`float32`である．
潜在行動のshapeは

$$
\left((d_b-1)m\right)
$$

であり，$m$は潜在次元である．

複数回の行動により制御点を更新する場合は，時刻$\tau$における状態を

$$
\boldsymbol{s}_{\tau}
=
\begin{pmatrix}
\mathcal{R}(\boldsymbol{c}_F)\\
\boldsymbol{a}_{\tau}\\
p_{\tau}
\end{pmatrix}
$$

とすることができる．ここで，$p_{\tau}$はエピソード内の進行度を表す特徴量である．


## 13．追跡コスト

第$\ell$解パスにおける受理ステップ数を$N_{\mathrm{acc}}^{(\ell)}$，棄却ステップ数を$N_{\mathrm{rej}}^{(\ell)}$とする．

第$\ell$解パスの追跡コストを

$$
J_\ell
=
\begin{cases}
N_{\mathrm{acc}}^{(\ell)}
+
\rho N_{\mathrm{rej}}^{(\ell)},
&
\text{追跡成功時},
\\
M_{\mathrm{fail}},
&
\text{追跡失敗時}
\end{cases}
$$

と定義する．ここで，$\rho\geq0$は棄却ステップに対する重み，$M_{\mathrm{fail}}$は追跡失敗時のペナルティである．

全解パスに対する平均追跡コストを

$$
J_{\mathrm{bez}}
=
\frac{1}{N_{\mathrm{path}}}
\sum_{\ell=1}^{N_{\mathrm{path}}}
J_\ell
$$

と定義する．

線形係数パスを使用した場合の平均追跡コストを

$$
J_{\mathrm{lin}}
=
\frac{1}{N_{\mathrm{path}}}
\sum_{\ell=1}^{N_{\mathrm{path}}}
J_{\mathrm{lin},\ell}
$$

とする．

初期環境では，線形ベースラインの制御点をゼロ潜在行動から生成し，
`reset()`時に1回だけ追跡する．この結果から得た`J_lin`を同じエピソード内でキャッシュし，
`step()`で計算した`J_bez`と比較する．
追跡失敗は上式の`M_fail`によりパス単位で扱う．
Juliaランタイム障害や不正な配列shape，NaN/Inf，返却payload不正は数学上の追跡失敗ではなく，
実装上の例外として扱う．


## 14．報酬

ベジェパスと線形パスの追跡コストの差に基づき，報酬を

$$
r
=
\mu
\left(
J_{\mathrm{lin}}-J_{\mathrm{bez}}
\right)
$$

と定義する．ここで，$\mu>0$は報酬のスケーリング係数である．

$J_{\mathrm{bez}}<J_{\mathrm{lin}}$の場合に正の報酬が得られ，線形パスよりも追跡コストを削減する制御点配置が学習される．


## 15．最適化問題

多変数Pham型サポート系に対するパス最適化問題は

$$
\min_{P_1,\ldots,P_{d_b-1}}
J_{\mathrm{bez}}
$$

と表される．

端点制約は

$$
P_0=\boldsymbol{c}_G,
\qquad
P_{d_b}=\boldsymbol{c}_F
$$

である．
さらに，各$P_k$の先頭係数成分は$1$に固定する．

正則な係数パスのみを許容する理論的な定式化では，

$$
\boldsymbol{c}([0,1))\cap\Sigma=\emptyset
$$

を制約として課す．終点も非特異であることを要求する場合は，
$\boldsymbol{c}([0,1])\cap\Sigma=\emptyset$を課す．一方，数値実験では
追跡失敗時のペナルティ$M_{\mathrm{fail}}$により，判別集合に近づく
または交差する係数パスをコストで抑制する．

潜在変数によるパラメータ化を用いる場合は，

$$
\min_{
\boldsymbol{z}_1,\ldots,
\boldsymbol{z}_{d_b-1}
}
J_{\mathrm{bez}}
$$

と表される．ただし，

$$
P_k
=
\left(
1-\frac{k}{d_b}
\right)\boldsymbol{c}_G
+
\frac{k}{d_b}\boldsymbol{c}_F
+
\mathcal{R}^{-1}
\left(
U\boldsymbol{z}_k
\right)
$$

である．ここで，$U$は先頭係数成分を変化させないように選ぶ．
