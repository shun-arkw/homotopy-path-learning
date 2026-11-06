# 関連研究・一次情報（Related Work）

[README](README.md) ｜ [研究仮説](research_ideas.md) ｜ [開始系](starting_system.md) ｜ [学習](data_and_learning.md) ｜ [経路](homotopy_paths.md) ｜ [実験](experimental_design.md)

> **文献調査更新：2026-11-06（Version 0.2）．** 原著本文・学会公式要旨・書誌確認・未確認を分けて記載する．ここは**網羅的調査の完成版ではない**．Passau大学のABM原典など，本文の完全な照合が済んでいない資料は明示する．

## 1．学習によるHC・実数解の扱い

### R1｜Hruby et al.（2022）

- **Petr Hruby, Timothy Duff, Anton Leykin, Tomas Pajdla.** *Learning To Solve Hard Minimal Problems*. CVPR 2022, pp. 5532–5542.
- **原著：** https://openaccess.thecvf.com/content/CVPR2022/html/Hruby_Learning_To_Solve_Hard_Minimal_Problems_CVPR_2022_paper.html
- **確認した内容：** RANSACの幾何学的minimal problemに対して，学習で適切な開始問題・開始解の組（anchors）を選び，目的解への単一経路追跡を行う．
- **今回との差分候補：** 固定anchorsの選択に限らず，未知系に依存して候補点や開始系を再構成する方法．ただし，候補選択だけを新規としない．
- **関係：** N3，N4．**確認状態：学会公式要旨・公開論文の検索可能本文を確認．**

### R2｜Zhang et al.（2025）

- **Xinyue Zhang, Zijia Dai, Wanting Xu, Laurent Kneip.** *Simulator HC: Regression-based Online Simulation of Starting Problem-Solution Pairs for Homotopy Continuation in Geometric Vision*. CVPR 2025. DOI: `10.1109/CVPR52734.2025.02524`.
- **原著：** https://doi.org/10.1109/CVPR52734.2025.02524
- **公開版：** https://arxiv.org/abs/2411.03745 （プレプリントの初期タイトルは異なる）
- **確認した内容：** 回帰ネットワークで近似解を推測し，幾何学的Simulatorから整合的な開始問題・解の組を構成し，単一の経路を追跡する．主な実験は幾何学的コンピュータビジョン問題に対するもの．
- **今回との差分候補：** 問題固有のSimulatorに限らない構造制約付き再構成，複数候補点，追跡費用を考慮した学習．ただしこれらの組合せが新規である保証はない．
- **関係：** N2，N3，N4，N5．**確認状態：IEEE公式要旨と公開論文の検索可能本文を確認．**

### R3｜Mai et al.（2026）

- **Jiayao Mai et al.** *Neural Predictor-Corrector: Solving Homotopy Problems with Reinforcement Learning*. ICLR 2026.
- **公式：** https://proceedings.iclr.cc/paper_files/paper/2026/hash/3104e1ab39875cf54fe1eb4473e7c5a1-Abstract-Conference.html
- **実装：** https://github.com/maijiayao1/NPC
- **確認した内容：** predictor-correctorのステップ幅や終了条件などをRLで選択する枠組み．多項式の根探索以外にも広いHomotopy問題を対象とする．
- **今回との差分候補：** 開始系・候補点・係数経路の生成と最適化は，追跡器内部の制御とは別の設計変数である．同論文と直接比較可能なタスクでは公平な評価が必要．
- **関係：** N4，N5．**確認状態：学会公式要旨と公式実装を確認．**

### R4｜Bernal et al.（2023）

- **Edgar A. Bernal, Jonathan D. Hauenstein, Dhagash Mehta, Margaret H. Regan, Tingting Tang.** *Machine learning the real discriminant locus*. Journal of Symbolic Computation 115 (2023), 409–426. DOI: `10.1016/j.jsc.2022.08.001`.
- **原著：** https://doi.org/10.1016/j.jsc.2022.08.001
- **確認した内容：** 実数解の数に応じたパラメータ空間の領域境界を，分類器などの機械学習で推定する．HCを用いて学習データを生成する．
- **今回との差分候補：** 判別集合の近似それ自体を新規とせず，Starting Pair再構成・経路選択に使う具体的方法を検討する．
- **関係：** N6，N5．**確認状態：出版社公式要旨を確認．**

## 2．Parameter Homotopyと構造保存

### R5｜Morgan & Sommese（1989）

- **Alexander P. Morgan, Andrew J. Sommese.** *Coefficient-parameter polynomial continuation*. Applied Mathematics and Computation 29(2) (1989), 123–160. DOI: `10.1016/0096-3003(89)90099-4`.
- **原著：** https://doi.org/10.1016/0096-3003(89)90099-4
- **確認した内容：** 多項式係数がパラメータから決まる状況を考え，パラメータ空間上のホモトピーによって構造を利用する理論．全幾何学的孤立解の計算が基本的対象．
- **今回への制約：** 同じファミリー内でパスを引くこと自体は古典的な既存手法である．
- **関係：** N2，N5．**確認状態：出版社公式要旨を確認．**

### R6｜Morgan & Sommese（1987）

- **Alexander P. Morgan, Andrew J. Sommese.** *A homotopy for solving general polynomial systems that respects m-homogeneous structures*. Applied Mathematics and Computation 24(2) (1987), 101–113. DOI: `10.1016/0096-3003(87)90063-4`.
- **原著：** https://doi.org/10.1016/0096-3003(87)90063-4
- **確認した内容：** 多重斉次構造を尊重するホモトピーにより必要な経路数を削減できることを示す．
- **今回への制約：** 「代数的構造を保存する」という概念だけでは新規性にならない．
- **関係：** N2．**確認状態：出版社公式要旨を確認．**

### R7｜Huber & Sturmfels（1995）

- **Birkett Huber, Bernd Sturmfels.** *A polyhedral method for solving sparse polynomial systems*. Mathematics of Computation 64(212) (1995), 1541–1555. DOI: `10.1090/S0025-5718-1995-1297471-4`.
- **原著：** https://doi.org/10.1090/S0025-5718-1995-1297471-4
- **確認した内容：** Newton polytopeとmixed subdivisionを利用する疎多項式のPolyhedral Homotopy．対象とする系と条件の下で追跡経路数を削減する．
- **今回への制約：** サポートの利用は既存であり，サポート保存だけで追跡の容易さは保証しない．
- **関係：** N2，N6．**確認状態：学術誌の書誌・要旨を確認．**

### R8｜Plecnik & Fearing（2016；関連誌版2017）

- **Mark M. Plecnik, Ronald S. Fearing.** *Finding Only Finite Roots to Large Kinematic Synthesis Systems*. ASME IDETC 2016．関連するJournal of Mechanisms and Robotics誌版は2017年，DOI: `10.1115/1.4035967`．
- **会議版原著：** https://people.eecs.berkeley.edu/~ronf/PAPERS/mplecnik-idetc16.pdf
- **確認した内容：** Finite Root Generation（FRG）で，機構の寸法をランダムに生成することで開始系・既知の有限開始解を作り，各開始系から1経路を追跡する．同研究の対象は主に機構合成における有限解の列挙・効率化である．
- **今回への制約：** 「既知解から開始系を逆構成して不必要な経路を減らす」という発想は既存．共通開始系・複数候補点・学習・追跡費用との違いを明示する．
- **関係：** N1，N2，N3．**確認状態：会議版の本文（特に方法の説明）を確認．**

## 3．Vanishing Ideal，Border Basis，ABM

### R9｜Heldt et al.（2009）

- **Daniel Heldt, Martin Kreuzer, Sebastian Pokutta, Hennie Poulisse.** *Approximate computation of zero-dimensional polynomial ideals*. Journal of Symbolic Computation 44(11) (2009), 1566–1591. DOI: `10.1016/j.jsc.2008.11.010`.
- **原著：** https://doi.org/10.1016/j.jsc.2008.11.010
- **確認した内容：** 近似的な点データから近似消失イデアルや近似Border Basisを構成するAVI系の数値的手法を扱う．
- **注意：** **HeldtらのAVIと，LimbeckのABM（R18）を同一のアルゴリズムだと扱わない**．
- **関係：** N1，N6．**確認状態：出版社公式要旨を確認．**

### R10｜Kera & Kehrein（2025）

- **Hiroshi Kera, Achim Kehrein.** *Gradient-Weighted, Data-Driven Normalization for Approximate Border Bases — Concept and Computation*. arXiv:2506.09529 (2025).
- **原著：** https://arxiv.org/abs/2506.09529
- **確認した内容：** 近似Border Basisの勾配重み付き正規化を提案し，データスケーリングへの不変性，ノイズへの頑健性等を研究．
- **今回への制約：** HCのStarting Systemが正則・正方・同サポート・追跡容易となることまでは保証しない．
- **関係：** N1，N6．**確認状態：arXiv要旨を確認．定理の詳細な仮定の全再検証は今後の作業．**

### R11｜MAVI（公式実装）

- **Hiroshi Kera et al.** *Monomial Agnostic Vanishing Ideal*（ライブラリ／GitHub）．
- **公式：** https://github.com/HiroshiKERA/monomial-agnostic-vanishing-ideal
- **確認した内容：** VCA，ABM，ABM-GWNなど，有限点集合から消失イデアル／近似基底を扱う実装．READMEではNumPy，JAX，PyTorchバックエンド間の対応差や浮動小数点での注意点を記載．
- **今回への制約：** ライブラリを使用することそのものは新規性にならない．ベースラインとしてSVDによる単純な射影法を用意する．
- **関係：** N1，N6．**確認状態：公式READMEを確認（2026-11-03）．**

### R15｜Möller & Buchberger（1982）

- **H. Michael Möller, Bruno Buchberger.** *The construction of multivariate polynomials with preassigned zeros*. In: Computer Algebra (Marseille, 1982), LNCS 144, 24–31. DOI: `10.1007/3-540-11607-9_3`.
- **原著：** https://doi.org/10.1007/3-540-11607-9_3
- **確認した内容：** 有限点集合上で消失する多変数多項式の基底構成と補間に関する古典的研究．
- **今回への制約：** 有限点集合の消失イデアル生成自体は新しくない．
- **関係：** N1．**確認状態：出版社側の書誌・概要を確認．**

### R18｜Limbeck（2014）

- **Jan Limbeck.** *Computation of Approximate Border Bases and Applications*. 博士論文，2014．
- **ユーザー提示リンク：** https://opus4.kobv.de/opus4-uni-passau/frontdoor/index/index/year/2014/docId/207
- **確認した内容：** 二次情報・著者公開の学位論文要旨により，ABMを含む近似Border Basisの研究であることを確認．
- **注意：** **指定URLの博士論文本文は今回の作業で全体を直接照合できていない**．細かな定理，アルゴリズム番号，実装内容を引用する際は原著本文を改めて確認する．
- **関係：** N1，N6．**確認状態：書誌・概要を間接確認，本文の精読は保留．**

## 4．HCの経路解析・保証・ソルバー

### R12｜Shub（2009）

- **Michael Shub.** *Complexity of Bézout's Theorem VI: Geodesics in the Condition (Number) Metric*. Foundations of Computational Mathematics 9 (2009), 171–178. DOI: `10.1007/s10208-007-9017-6`.
- **原著：** https://arxiv.org/abs/math/0702344
- **確認した内容：** 正規化された複素同次多項式系の問題・解の経路に対する条件計量と計算複雑性の理論．
- **注意：** 任意の実アフィン系で計算したJacobian条件数積分をそのまま正式なcondition lengthと呼ばない．
- **関係：** N5，N6．**確認状態：arXiv要旨，出版書誌を確認．**

### R13｜Hauenstein & Liddell（2016）

- **Jonathan D. Hauenstein, Alan C. Liddell Jr.** *Certified predictor–corrector tracking for Newton homotopies*. Journal of Symbolic Computation 74 (2016), 239–254. DOI: `10.1016/j.jsc.2015.07.001`.
- **原著：** https://doi.org/10.1016/j.jsc.2015.07.001
- **確認した内容：** 定数項のみを変化させるNewton Homotopyの認証付きPredictor-Corrector追跡を分析し，特定の条件計量による複雑性上界を与える．
- **関係：** N5，N6．**確認状態：出版社公式要旨を確認．**

### R14｜HomotopyContinuation.jl（公式ドキュメント）

- **公式：** https://www.juliahomotopycontinuation.org/HomotopyContinuation.jl/stable/
- **リポジトリ：** https://github.com/JuliaHomotopyContinuation/HomotopyContinuation.jl
- **確認した内容：** Start Systems（Total Degree，Polyhedral），Parameter Homotopy，Path Tracking，Certification等の機能がある．
- **今回への制約：** 既存ソルバーの機能・強力なベースラインを無視して「新しいHC」だと主張しない．比較時は具体的バージョンのAPIと数値設定を記録する．
- **関係：** N1〜N6の評価．**確認状態：公式ドキュメント・GitHubを確認（2026-11-03）．**

### R16｜Beltrán & Pardo（2011）

- **Carlos Beltrán, Luis Miguel Pardo.** *Fast Linear Homotopy to Find Approximate Zeros of Polynomial Systems*. Foundations of Computational Mathematics 11(1) (2011), 95–129. DOI: `10.1007/s10208-010-9078-9`.
- **原著：** https://doi.org/10.1007/s10208-010-9078-9
- **確認した内容：** 複素同次系の枠組みにおける既知の開始系・解の分布と線形ホモトピーの理論を扱う．
- **注意：** 実係数疎多項式系への同じ保証を無条件に持ち込まない．
- **関係：** N5，N6．**確認状態：書誌と概要を確認．**

### R17｜Breiding & Timme（2018）

- **Paul Breiding, Sascha Timme.** *HomotopyContinuation.jl: A Package for Homotopy Continuation in Julia*. ICMS 2018, 458–465. DOI: `10.1007/978-3-319-96418-8_54`.
- **原著：** https://doi.org/10.1007/978-3-319-96418-8_54
- **プレプリント：** https://arxiv.org/abs/1711.10911
- **確認した内容：** Juliaによる数値的HCソルバーの実装とアルゴリズム基盤．
- **関係：** N1〜N6の評価．**確認状態：出版書誌と公式リポジトリを確認．**

## 5．追加した直接競合・実応用文献（Version 0.2）

### R19｜Leykin（2011）— 開始系の平均計算量最適化

- **Anton Leykin.** *A Search for an Optimal Start System for Numerical Homotopy Continuation*（2011）．
- **原著：** https://arxiv.org/abs/1105.4324
- **確認した内容：** 正則な多項式系の**全解を求める平均追跡費用**を低減するための開始系探索．
- **今回への制約：** 「追跡費用を考慮してStarting Systemを最適化する」という目的だけでは新規といえない．**実数解1つ・学習による償却・構造制約**などの具体差分が必要．
- **関係：** N2，N4，N5．**確認状態：arXiv原著の書誌・要旨を確認，詳細定理は要精査．**

### R20｜Duff et al.（2019）— Parameterized SystemsとMonodromy

- **Timothy Duff, Cvetelina Hill, Anders Jensen, Kisun Lee, Anton Leykin, Jeff Sommars.** *Solving Polynomial Systems via Homotopy Continuation and Monodromy*. IMA Journal of Numerical Analysis 39(3) (2019), 1421–1446．DOI: `10.1093/imanum/dry017`．
- **原著：** https://doi.org/10.1093/imanum/dry017
- **確認した内容：** パラメータ化された多項式ファミリーを用いた既知開始解の取得・経路追跡・Monodromyのアルゴリズム．ファミリーと解を対応付ける正則領域・分岐集合の枠組みが明示される．
- **今回への制約：** 任意の点を零点にするパラメータを決める操作，単一のseed pairからファミリー上を追跡する操作は古典的な延長線上にある．
- **関係：** N1，N2，N5．**確認状態：出版元の本文（Introduction，Background，Implementation）を確認．**

### R21｜Betti & Borovik（2025）— SAGBI型の構造化Starting System

- **Barbara Betti, Viktoriia Borovik.** *The SagbiHomotopy.jl Package for Solving Polynomial Systems*（2025）．
- **原著：** https://arxiv.org/abs/2506.06264
- **確認した内容：** Horizontally parameterizedな系でSAGBI構造に基づく開始系とホモトピーを使い，対象の例でPolyhedral Homotopyより追跡経路数を削減する実装を示す．
- **今回への制約：** 「幾何学的・代数的構造を使って開始系を効率化」それ自体は既存研究．問題クラス・構造と追跡経路数 vs 1実数解の差が重要．
- **関係：** N2，N5，N6．**確認状態：arXiv要旨を確認，具体実験・理論条件は要精査．**

### R22｜Ergür & de Wolff（2023）— Real Polyhedral Homotopy

- **Alperen A. Ergür, Timo de Wolff.** *A Polyhedral Homotopy Algorithm for Real Zeros*. Foundations of Computational Mathematics（2023）．
- **原著：** https://arxiv.org/abs/1910.01957
- **確認した内容：** 疎多項式系の実数解に対し，パッチワーク手法に基づく実数経路追跡を行う．有効性は判別集合のAmoebaなどに関連する**限定された係数領域と条件**に依存する．
- **今回への制約：** 「実数解だけを優先するHC」という目標だけでは新規でない．
- **関係：** N5，N6．**確認状態：arXiv原著要旨を確認，完全な適用条件は要精査．**

### R23｜Chien et al.（CVPR 2022）— GPU-HCと幾何学的最小問題

- **Chiang-Heng Chien, Hongyi Fan, Ahmad Abdelfattah, Elias Tsigaridas, Stanimire Tomov, Benjamin Kimia.** *GPU-Based Homotopy Continuation for Minimal Problems in Computer Vision*. CVPR 2022．
- **原著：** https://openaccess.thecvf.com/content/CVPR2022/html/Chien_GPU-Based_Homotopy_Continuation_for_Minimal_Problems_in_Computer_Vision_CVPR_2022_paper.html
- **確認した内容：** 多項式系のGPU並列経路追跡を実装し，複数の幾何学的最小問題でベンチマークする．
- **関係：** N3，N5，実応用の比較．**確認状態：CVF公式要旨を確認．**

### R24｜Catalyst.jl — 化学反応ネットワークの定常状態HC

- **公式：** https://docs.sciml.ai/Catalyst/dev/steady_state_functionality/homotopy_continuation/
- **確認した内容：** Mass-Action反応ネットワークから定常状態を求める `hc_steady_states` の例，Wilhelmモデルの複数の正の定常状態と境界解，保存則の扱い．
- **今回への制約：** 既存の実用的ソルバーが存在し，**正の定常状態1つ・多重定常性・安定性**は異なる成功条件である．
- **関係：** E7とベースライン設計．**確認状態：公式ドキュメント本文を確認．**

### R25｜Burr, Sottile & Walker（2023）— Khovanskii Bases

- **Michael Burr, Frank Sottile, Elijah Walker.** *Numerical Homotopies from Khovanskii Bases*. Mathematics of Computation 92 (2023), 2333–2353．DOI: `10.1090/mcom/3689`．
- **原著：** https://doi.org/10.1090/mcom/3689
- **確認した内容：** 有限Khovanskii基底を持つ多様体上の求解を，トーリック退化等の構造を利用して行う．
- **関係：** N2，N5，N6．**確認状態：AMS公式要旨を確認．**

### R26｜Bates & Sottile（2011）— Khovanskii–Rolle Continuation

- **Daniel J. Bates, Frank Sottile.** *Khovanskii-Rolle Continuation for Real Solutions*．
- **原著：** https://arxiv.org/abs/0908.4579
- **確認した内容：** 変形ホモトピーとは異なる実曲線追跡を利用し，非退化実数解の探索を目的とする．
- **今回への制約：** 実数解だけを求めるための既存の数値的代数手法を比較対象候補から外さない．
- **関係：** N6，実数解探索．**確認状態：原著要旨・著者公開の紹介ページを確認．**

## 6．研究仮説との対応関係

| 研究仮説 | 特に重要な先行研究 | 新規性を主張する前に確認すべき差分 |
|---|---|---|
| N1：共通開始系 | R8，R9，R10，R11，R15，R18 | 既知の点集合の基底計算ではなく，HCの総費用に対する有効性があるか |
| N2：構造保存 | R5，R6，R7，R8，R20，R21，R25 | 問題固有の構造をどう保存し，従来のParameter Homotopyより何を改善するか |
| N3：学習による候補 | R1，R2 | 1候補の回帰／anchorsの選択を超えて何を生成するか |
| N4：追跡費用に基づく学習 | R1，R2，R3，R19 | 学習目的と推論コストにどんな具体的な違いがあるか |
| N5：開始系と経路の共同最適化 | R3，R5，R12，R13，R19，R20 | 単独最適化の組合せを超える利得・解析があるか |
| N6：数学に基づく制約 | R4，R10，R12，R13，R16，R22，R25，R26 | 単なるランク判定・条件数指標ではなく，何を新しく導くか |

## 7．網羅的調査に向けた追加確認事項

**第1優先：** R1〜R3，R19〜R22，R25は，Starting Pairの選択・再構成・経路最適化・実数解探索に直接関係するため，**原著のアルゴリズム，仮定，実験対象，学習対象，評価指標**を比較表にする．

**調査方法：** キーワード検索（Start System Optimization，Newton Homotopy，One-Path Continuation，Parameter Homotopy，Monodromy，Learned Continuation，Differentiable Homotopy，Sparse Interpolation，Real-Root Tracking，SAGBI，Khovanskii，Vanishing Ideal），引用追跡，2026年11月までの新着研究の確認を行う．完全網羅を保証せず，検索式・対象期間・確認できなかった原稿を記録する．

**最も厳しい検証：** 「MLによる候補生成」「共通Starting System」「Structure-Preserving」「経路との共同学習」が**それぞれ単体で既知か，統合済みの直接競合があるか**を調べる．存在しないと結論付けるためには，未発見であるという限定を明示する．

## 8．今後の文献確認事項

- [ ] R18（Limbeck博士論文）のABMの正確なアルゴリズム定義，適用条件，数値正規化，停止条件を原著本文で照合する．
- [ ] R10の定理番号・仮定・保証対象を原著本文で精査する．
- [ ] Starting Pair・非線形パスの共同学習に関する2026年以前の周辺文献を網羅的に追加調査する．
- [ ] 実応用ベンチマークを選定した段階で，その問題を扱う既存専用ソルバーを追加する．
- [ ] ベンチマークに利用するHomotopyContinuation.jlのバージョンと標準パラメータを固定する．

文献IDは永続的とし，今後の精査で説明を更新しても再利用しない．文献の掲載可否は研究仮説の採否とは独立して記録する．
