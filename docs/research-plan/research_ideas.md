# 研究仮説・新規性候補（Novelties）

[README](README.md) ｜ [開始系](starting_system.md) ｜ [学習](data_and_learning.md) ｜ [経路](homotopy_paths.md) ｜ [実験](experimental_design.md) ｜ [文献](related_work.md)

> **Version 0.2 / 2026-11-06．N1〜N6はいずれも新構想として Proposed / Hypothesis．** 既存の多変数Bézierパス学習には実験記録（EXP-0004〜0060）が存在する．その結果をN1〜N6の検証済み結果と読み替えない．

## 既存成果との接続と優先順位を決めない原則

既存リポジトリは，複素Pham型多項式系の**全経路**に対するBézier係数パス最適化を既に実装し，PPO・Behavior Cloning・Attentionを用いた実験を積み重ねている．`EXP-0059` では**一般の目的系に対するランダム低次元行動空間**でのOracle探索が期待した探索時間短縮を達成しなかった一方，`EXP-0060` では**低次元の目的系ファミリーと対応する行動基底**を組み合わせたOracle探索で有望な改善が記録された．`EXP-0061` は現在実行中であり，学習の容易さは未確定である．

したがって，**N5はまったく新しいパス学習の開始を意味しない**．既存の経路学習を開始系設計・最初の実数解取得と接続する場合の追加貢献仮説である．既存の `EXP` と新計画の `E` を混同しない．また，ランダムな線形係数部分空間が幾何学的・物理的な構造保存を示すわけではない．

## 仮説の一覧

| ID | 仮説・新規性の候補 | 主な方法 | 検証実験 | 関連文献 |
|---|---|---|---|---|
| **N1** | Multi-Solution Starting System Reconstruction：複数候補を共通開始系に埋め込み，最初の目的解への到達を効率化できるか | M3，M4 | E1，E2 | R8，R9，R10，R15 |
| **N2** | Structure-Preserving Reconstruction：サポートだけでなく問題ファミリーの構造制約を保つことに効果があるか | M2，M5 | E1，E2，E5 | R5，R6，R7，R8 |
| **N3** | Learning-Based Candidate Generation：学習した単一／複数候補がランダム点より有用か | M10 | E3，E4 | R1，R2 |
| **N4** | Tracking-Cost-Aware Learning：解予測誤差ではなくHCの総コストを目的にした学習が有用か | M10，M12 | E4，E6 | R1，R2，R3 |
| **N5** | Joint Starting Pair and Path Optimization：開始系と係数・パラメータ経路の最適な組み合わせに相互作用があるか | M2〜M8，M12 | E5，E6 | R2，R3，R5，R12，R13 |
| **N6** | Theory-Guided Reconstruction：再構成の可解性，正則性，条件付け，判別集合の情報を生成・選択に利用できるか | M3，M5，M12 | E1，E5，E6 | R4，R12，R13 |

## N1｜Multi-Solution Starting System Reconstruction

**Motivation：** 一つの共通開始系 $G$ に $k$ 個の既知開始解を持たせれば，複数の解経路を選択・再試行できる．点集合のVanishing Idealを利用できる可能性がある．

**仮説：** 同じ候補点集合 $C$ から複数の独立開始系 $G_j$ を作る方式よりも，適切に構成した共通開始系 $G_C$ の方が，最初の有効なTarget Solutionを低コストで得られる問題クラスが存在する．

**最も強い反論：** 個別に $G_j(x)=F(x)-F(y^{(j)})$ とすれば各点を既知の零点にできる（定数項の変更を許す場合）．共通開始系は全候補への消失制約を同時に満たすため，係数の自由度が減り，再構成が高価になる可能性がある．共通化の効果を示さずにABMを導入しても新規性は弱い．

**数学的制約：** 指定サポート $A_i$ の行列 $V_{A_i}(C)$ について，非零係数の必要十分条件は $\ker V_{A_i}(C)\ne\{0\}$．全開始解でのJacobian正則性は別条件である．詳細は [starting_system.md](starting_system.md)．

**Validation：** E1の構成可能性とE2の共通開始系 vs 独立開始系を同一候補点・同一時間予算で比較．**反証基準：** 全ての検証対象クラスで共通方式の成功率・総時間が不利なら，この方式を中心貢献に据えない．クラス依存性は保存する．

## N2｜Structure-Preserving Reconstruction

**Motivation：** 元の幾何・物理パラメータに由来する係数間の関係を保つことで，意味のない開始系を避けられる可能性がある．

**仮説：** 特定のファミリーにおいて，構造保存が追跡安定性や総計算時間の改善につながる．

**反論：** パラメータを全係数としてしまえば通常のCoefficient Projectionと変わらない．候補点を零点として実現するパラメータが存在しない例がある．パラメータ経路が判別集合を横断する場合，構造保存でも失敗する．問題構造を利用したParameter Homotopy自体は古典的な既存研究（R5〜R8）である．

**Validation：** 同じ候補・同じ線形係数経路でM2とM5を比較．M5の同じStarting Pairについて線形係数経路とParameter Pathを比較（E2，E5）．

## N3｜Learning-Based Candidate Generation

**Motivation：** 未知のTarget Systemを入力し，良いStarting Pairにつながる候補点を短時間で予測できる可能性がある．

**仮説：** ランダム生成や単純な数値初期化に比べ，学習した候補点が所定の時間予算内の成功率を高める．

**反論：** 単一解の回帰＋開始系Simulatorは既にR2に存在する．複数候補の生成自体も一般的なML技術である．Root-first合成データの分布バイアス，複数解平均の退化，真の解を使ってしまう情報漏洩，推論負荷に注意．

**Validation：** E3の分布監査，E4のsingle vs multi，ランダム点，Newton初期化，Zhang型1候補への比較．

## N4｜Tracking-Cost-Aware Learning

**Motivation：** 解とのユークリッド距離が小さい候補でも，Jacobianやパスが悪ければHCは遅い．予測誤差よりも「解へ到達するまでの費用」を直接重視する．

**仮説：** $\mathbb E[T_{\mathrm{first}}]$ または成功率制約付き代理損失に基づく学習が，解回帰だけの場合より高速になる．

**反論：** 評価にはHCを回す費用がかかり，計算時間は非滑らか・ノイズを含む．RLの適用自体はR3，学習による開始選択はR1に存在する．正確なラベルと学習コストの回収に注意．

**Validation：** E4，E6．残差予測，候補ランキング，成功確率／所要時間予測，サロゲート，PPO等を予算一致で比較．

## N5｜Joint Starting Pair and Path Optimization

**Motivation：** 開始系とパスの組合せによってJacobianの特異性への近さ，追跡ステップ数が変わり得る．

**仮説：** Starting Pairと経路の組を共同に設計した場合，それぞれを独立に最適化する場合以上の追加利得がある．

**反論：** 非線形パスを選べば必ず良いわけではない．共同最適化の計算コストが改善分を上回り得る．既存のParameter Homotopy・condition metricの議論との区別が必要．

**既存成果との違い：** 既存EXPは主として固定Pham開始系と複素全経路の平均ステップ数を対象とする．N5では新たに開始系・開始解・実数解到達成功率・Time to First Feasible Solutionを設計変数と評価対象に含める．EXP-0060のOracle改善をN5の共同最適化の証拠としない．

**Validation：** E5の2因子アブレーションとE6の共同最適化．**反証基準：** 独立最適化と統計的に区別できず，総時間でも改善しないなら，統合を論文の必須要素にしない．

## N6｜Theory-Guided Reconstruction

**Motivation：** 消失制約や正則性の条件，判別集合への接近などの数学的情報を学習・最適化に取り込む．

**仮説：** 制約を正しく実装すると，再構成失敗率・経路追跡失敗率が低下する問題クラスがある．

**反論：** 初等線形代数で得られるランク条件は新規な定理ではない．条件数の代理指標は追跡コストの保証ではない．正規化方法が変わると数値の意味も変わる．

**Validation：** E1の可解性と正則性，E5のパス正則性，E6のコスト・成功率を制約あり／なしで比較．

## 新規性調査の重点と競合候補

- **N1**：有限点のVanishing Idealだけでなく，**独立したStarting Pair**に対し共通の開始系を使うことの追加利点が必要．Möller–Buchberger，ABM，FRG（R8，R9，R15，R18）を確認する．
- **N2**：Parameter Homotopy（R5），Monodromyのファミリー上のseed pair（R20），SAGBI / Khovanskii型（R21，R25）と比較する．
- **N3**：Hruby（R1）およびZhang（R2）と「予測対象・再構成・複数候補・計算コスト」の差を示す．
- **N4**：Leykinの開始系最適化（R19）とMaiの学習型追跡器（R3）を正面から扱う．
- **N5**：既存の自身のBezier/PPO研究と，係数パスの非線形最適化・パラメータ経路研究との差を具体化する．
- **N6**：実判別集合の学習（R4），Real Polyhedral Homotopy（R22），条件計量（R12）を確認する．

**この対応表は文献レビューの結論ではない．** 未発見の直接競合研究が存在し得るため，網羅的なLiterature Reviewで更新する．

## 新規性に関する留保

- `N1`〜`N6` は互いに排他的ではない．全てを組み込む必要もない．
- 「MLを使った」「Vanishing Idealを計算した」「Parameter Pathを使った」という手段のみを新規性としない．
- 優れた総実行時間だけで十分か，別途理論的な貢献が必要かは投稿先・比較対象・一般化性能によって決まる．
- 既存研究との対応は [related_work.md](related_work.md) で更新する．
- 実験で否定された仮説は削除せず，条件と理由を追記し `Rejected` または `On Hold` にする．
