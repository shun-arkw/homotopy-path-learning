# ホモトピー経路の設計・最適化

[README](README.md) ｜ [研究仮説](research_ideas.md) ｜ [開始系](starting_system.md) ｜ [実験](experimental_design.md) ｜ [文献](related_work.md)

> **Version 0.2．既存の複素Pham型Bézier係数パス最適化は実装・実験済み．** 本文に示す新規性候補は，開始系を可変にして実数解探索へ拡張する設計であり，性能は未検証．関連：N2，N5，N6．

## 1．基本的な定義と向き

$n\in\mathbb Z_{>0}$，$\mathbb K\in\{\mathbb R,\mathbb C\}$ とし，多項式系 $F,G:\mathbb K^n\to\mathbb K^n$ と既知の正則Starting Pair $(G,\boldsymbol{x}_0)$（$\boldsymbol{x}_0\in\mathbb K^n$）を考える．ホモトピー $H(\boldsymbol{x},t)$ は，

$$
H(\boldsymbol{x},0)=G(\boldsymbol{x}),\qquad H(\boldsymbol{x},1)=F(\boldsymbol{x}),\qquad H(\boldsymbol{x}_0,0)=\boldsymbol{0}_n
$$

を満たすよう構成する．解経路 $\boldsymbol{x}(t)\in\mathbb K^n$ が存在して $H(\boldsymbol{x}(t),t)=\boldsymbol{0}_n$ が成立し，$J_{\boldsymbol{x}}H(\boldsymbol{x}(t),t)$ が正則なら，局所的には

$$
\boxed{\frac{d\boldsymbol{x}}{dt}=-J_{\boldsymbol{x}}H(\boldsymbol{x}(t),t)^{-1}\frac{\partial H}{\partial t}(\boldsymbol{x}(t),t)}
$$

と書ける．正則性，有限値に収まること，$t=1$ までの到達は別途必要な仮定・検証対象である．複数開始解があっても，それぞれ別の解経路になる．

## 2．方法索引

| Method | 名称 | 定義・設計対象 | 留意点 |
|---|---|---|---|
| **M6** | Linear Coefficient Path | $H=(1-t)G+tF$ | 単純で比較に適するが，特異・発散経路がある |
| **M7** | Parameter Homotopy | $H(x,t)=F(x;p(t))$ | 同じ問題ファミリー内に保てるが，経路正則性は保証しない |
| **M8** | Bézier Coefficient / Parameter Path | 制御点で曲線を表し調整 | 表現能力と最適化費用のトレードオフ |
| **M12** | Path / Candidate Policy Optimization | 候補選択，経路，追跡コストの最適化 | 学習・探索費用を総時間に含める |

M1〜M5は [starting_system.md](starting_system.md)，M9〜M11は [experimental_design.md](experimental_design.md) に索引を置く．

## 3．M6：Linear Coefficient Path

$$
H_{\mathrm{lin}}(\boldsymbol{x},t)=(1-t)G(\boldsymbol{x})+tF(\boldsymbol{x})
$$

最初の実験の対照群とする．$G$ と $F$ のサポートが異なっても，両方のサポートの和集合上で定義できる．ただし，次数や最高次係数の退化によって解が無限遠へ向かう可能性がある．経路途中のJacobianの正則性は一般に保証されない．

**反例参照：** [starting_system.md §8](starting_system.md) に係数退化・特異経路の解析例を記載した．

## 4．M7：Parameter Homotopy

対象問題に意味のあるパラメータ $\boldsymbol{p}\in\mathcal P\subseteq\mathbb K^q$（$q\in\mathbb Z_{>0}$）がある場合，$G(\boldsymbol{x})=F(\boldsymbol{x};\boldsymbol{p}_0)$，目的系 $F(\boldsymbol{x};\boldsymbol{p}^*)$ に対して，

$$
H_{\mathrm{par}}(\boldsymbol{x},t)=F(\boldsymbol{x};\boldsymbol{p}(t)),\qquad \boldsymbol{p}(0)=\boldsymbol{p}_0,\quad \boldsymbol{p}(1)=\boldsymbol{p}^*
$$

を用いる．問題の係数が $p$ に非線形に依存するとき，

$$
F(\boldsymbol{x};(1-t)\boldsymbol{p}_0+t\boldsymbol{p}^*)\ne (1-t)F(\boldsymbol{x};\boldsymbol{p}_0)+tF(\boldsymbol{x};\boldsymbol{p}^*)
$$

が一般に成立する．この差を比較実験で意識する．問題ファミリーが持つパラメータ制約 $\boldsymbol{p}\in\mathcal P$ がある場合，経路全体が $\mathcal P$ 内にあることを確認する．$\mathcal P$ が非凸なら直線補間で外に出る可能性がある．

**「構造保存＝安全」ではない反例：** $F(x;p)=x^2-p$，開始 $(x_0,p_0)=(2,4)$，目標 $p^*=1$ とする．線形パラメータ経路 $p(t)=4-3t$ は $p>0$ を保つ．一方，$p(t)=(2-3t)^2$ は同じ端点を持つが $t=2/3$ で $p=0$ を通り，$x=0$ が特異解になる．経路形状が重要である．

関連：古典的なParameter Homotopy（R5），構造保存ホモトピー（R6），実判別集合の学習（R4）．

## 5．M8：Bézier Path

**既存実装との対応：** リポジトリ `paths/parameterization.py`，`julia/src/tracking.jl`，設計文書 `docs/design/implementation/bezier_path.md` に複素Pham型のBézier係数経路と追跡器がある．既存の固定開始系から全経路の平均追跡ステップ数を改善する実験（EXP-0004〜0060）は，新規のStarting Pair・最初の実数解研究とは別に扱う．

ベクトル空間の次元を $m\in\mathbb Z_{>0}$，Bézier次数を $d_b\in\mathbb Z_{>0}$，制御点を $\boldsymbol{P}_0,\ldots,\boldsymbol{P}_{d_b}\in\mathbb K^m$ とする．係数経路の場合は $m$ を採用する係数基底の次元，パラメータ経路の場合は許容パラメータの次元とする．

$$
\boldsymbol{P}(t)=\sum_{j=0}^{d_b}{d_b\choose j}(1-t)^{d_b-j}t^j\boldsymbol{P}_j,\qquad \boldsymbol{P}(0)=\boldsymbol{P}_0,\;\boldsymbol{P}(1)=\boldsymbol{P}_{d_b}
$$

とする．係数経路に用いる場合は $\boldsymbol{P}_j\in\mathbb K^m$ が係数ベクトルで，$H(\boldsymbol{x},t)=F_{\boldsymbol{P}(t)}(\boldsymbol{x})$ と書く．Parameter Homotopyとして用いる場合は $\boldsymbol{P}_j\in\mathbb K^m$ が問題パラメータで，$H(\boldsymbol{x},t)=F(\boldsymbol{x};\boldsymbol{P}(t))$ とする．後者では $\boldsymbol{P}(t)\in\mathcal P$ を確認する必要がある．

制御点を調整する方法として数値最適化，サロゲート最適化，強化学習等を検討する．**線形パスはBézierの特別な場合**だが，BézierにするだけでHCが高速になるわけではない．同じ始終点でも特異集合へ近づく例がある．

### 再パラメータ化への注意

正則な幾何学的軌跡が同じでも，$t$ の進み方によって数値ステップの分布や実装上の計算時間は変わり得る．したがって，単に制御点を増やす比較だけでなく，単調な再パラメータ化との比較も候補とする．

## 6．判別集合・条件付けと正則性

パラメータ化された正方系について，有限領域の判別的な集合を

$$
\Delta=\{p:\exists x\in\mathbb C^n,\ F(x;p)=0,\ \det J_xF(x;p)=0\}
$$

のように考える（正確な代数的定義には射影閉包や無限遠の扱いが必要な場合がある）．実係数パラメータと実数解だけに着目した実判別集合も別途考える．正則な実数枝の個数は，局所的な滑らかな範囲内では変化しないが，特異点・無限遠・境界条件等で変化し得る．

数値上は $\kappa(J_xH)$，$\sigma_{\min}(J_xH)$，Newton反復数，失敗ステップを記録する．しかし，**$\int\kappa(J_xH)\|x'(t)\|dt$ のような任意の代理量を，理論上のcondition lengthと同一視しない**．Shub（R12）の条件計量は適切に正規化された複素同次系の解多様体上に定義され，対象・仮定が異なる．Newton Homotopyの保証付き追跡についてはR13を参照．

### 実経路と複素経路

実係数・実パラメータだけを用いた場合，実数枝が特異点で衝突して複素枝に変わることがある．複素パラメータに迂回して判別集合を回避する案も候補だが，目的系で**実数解を得られることは自動的に保証されない**．実数解に到達する分枝選択・検証が必要である．

## 7．N5：Starting Pairと経路の共同最適化

開始系構成 $\mathcal R$，候補集合 $C$，経路の決定規則 $\pi$ に対して，

$$
\min_{C,\mathcal R,\pi}\;\mathbb E\left[T_{\mathrm{first}}\big(F,\mathcal R(F,C),\pi\big)\right]
$$

を理想的な目標とする．少なくとも次の2つの比較を明確に分離する．

- **開始系の効果：** 同じ候補点・同じ線形係数経路を使って，開始系再構成だけを変える．
- **経路の効果：** **同じStarting Pair**を使って，線形係数パス・Parameter Path・Bézier Pathを比較する（Parameter Pathは同じ問題ファミリーに属するStarting Pairの場合のみ定義可能）．

これらを基準としたうえで，Starting Pairとパスを共同最適化したときの**相互作用による追加利得**があるか測る．共同最適化が高価なら，学習済み方策により推論費用を償却する方法も検討する．

### 重大な交絡

`Coefficient Projection + Linear Path` と `Structure-Preserving Reconstruction + Parameter Path` だけを比較しても，高速化の理由を特定できない．また，構造保存ではない開始系にParameter Homotopyを適用できない場合，無理に完全な要因実験を作らず，適用可能なセルのみ評価する．

## 8．既存実験の帰結をどこまで利用できるか

- `EXP-0059`：一般の複素目的系を固定したまま行動だけランダム低次元へ制限しても，事前登録した探索短縮条件を満たさなかった．**任意の構造保存方式が無効であることを示さない**．
- `EXP-0060`：$n=4,5$，$d_i=3$（全成分）で，$k_f=4,8$ の目的系ファミリーの基底上のOracle探索が，事前登録した改善の保持と速さの基準をともに満たした．$k_f=12,16$ は保持のみ満たした（EXP-0060 §5・§7.4）．速さは必要な探索予算に基づく基準であり，総wall timeではない．この差は**探索による全経路の平均ステップ数**であり，学習済み方策の性能でも，実数解を1つ見つける総計算時間でもない．
- `EXP-0061`：同じファミリーで教師あり学習と微調整を評価する確認的な実験（事前登録済み．記録の状態は Planned，結果の節は未記入）．事前登録 §4.2 には計算機間の照合手順と不一致時の代替手順が定められている．2026-11-06 時点の監査・監視では，第0段の照合で計算機間に係数の末尾の差とステップ数のずれが生じたと報告されている（報告された観察であり，記録には未記載．今回の作業では元の出力による独立検証は行っていない）．原因の特定と学習性能の結論は記録を待つ．
- 提案：新しい実験では乱数種だけでなく，生成された**係数配列・基底行列・開始解・追跡器設定**をファイル化し，ハッシュとコードコミットを対応付ける案．正式に採用した記録は確認されていない．採用するかは事前登録で決める（`experimental_design.md` §0・§7）．

## 9．評価時の注意

- 解経路追跡時間だけでなく，経路設計の時間，サロゲート推論，制御点最適化時間を含む．
- 同一の最大ステップ数，許容残差，実数性判定，認証条件，タイムアウトで比較する．
- Accepted / Rejected steps，Newton iterations，wall time，成功率，条件付け指標を記録する．
- 開始系やパスのスケーリングを明示し，数値条件数の比較の公平性を確保する．
- Step-sizeなど既存Trackerの最適化機能やNeural Predictor-Corrector（R3）との関係を整理する．

実験プロトコルは [experimental_design.md](experimental_design.md) に一元化する．
