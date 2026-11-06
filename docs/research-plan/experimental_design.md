# 実験計画と評価プロトコル

[README](README.md) ｜ [研究仮説](research_ideas.md) ｜ [開始系](starting_system.md) ｜ [学習](data_and_learning.md) ｜ [経路](homotopy_paths.md) ｜ [文献](related_work.md)

> **Version 0.2：E0〜E7は未実施の新構想の検証計画である．** 既存の係数パス学習には `EXP-0004`〜`EXP-0060` の実験記録があり，`EXP-0061` は2026-11-06時点で実行中．`E` と `EXP-` は異なる命名空間である．

## 0．既存実験の現状と原則的な切り分け

既存の `multivariate_system` は $F,G:\mathbb C^n\to\mathbb C^n$（$n\in\mathbb Z_{>0}$）のPham型多項式系について，**既知の全開始解から全複素経路**を追跡し，Bézier経路に対する平均追跡ステップ数を最適化している．これは既に実験済みであり，今回の「実行可能な実数解を最初に1つ求める」新構想とは成功条件・計算対象が異なる．

| 既存実験 | 記録上の観察 | 過剰に解釈しない点 |
|---|---|---|
| `EXP-0059` | 行動だけをランダムな低次元部分空間へ制限しても探索時間短縮の事前基準を満たさなかった | 低次元目的系の無効性を示すものではない |
| `EXP-0060` | 複素低次元ファミリー上で，対応する行動基底によるOracleの改善量が全次元行動を上回る場合があった | 学習済み方策の性能・実数解探索・総時間の改善ではない |
| `EXP-0061` | 監査時はサーバで分母探索を再実行中．学習結果は未確認 | 文書の `Planned` は現行ジョブの実行状況を反映していない |

**再現性の懸念：** 係数生成に使うQR分解・行列積の環境差により，CPU間で係数の最下位桁とステップ数が異なる現象が報告されている．因果原因はまだ完全には切り分けられていない．今後，乱数種だけに頼らず，生成した問題インスタンス・基底行列・全設定を保存し，再実行が本当に同一入力であるかを比較する．既存の事前登録や結果の数値は上書きしない．

## 1．実験全体の原則

1. **主指標はTime to First Feasible Solution** とする．純粋な実数解探索では $\Omega=\mathbb R^n$ としてTime to First Real Solutionを測定する．
2. **失敗を除外しない．** 固定時間予算 $B$ 内に解を認証した割合と，成功・失敗双方の時間を保存する．平均時間は打ち切り扱いを明記し，成功例のみの平均は補助指標とする．
3. **全経路HCだけを基準にしない．** 最初の1解を得るためのMultistart Newton，独立Newton Homotopy，標準HCの経路追跡を公平に比較する．
4. **候補点集合を揃える．** 共通開始系 vs 独立開始系の比較では，同じ候補 $C$，同じ順序決定の情報予算，可能な限り同じTracker設定を使う．
5. **要因を分離する．** 再構成方法，候補生成，経路形状，追跡器を一度に変更しない．
6. **事前学習とオンライン推論を区別する．** リアルタイム性能だけでなく，学習・前処理の償却可能性を必要に応じて示す．
7. **再現性を確保する．** 乱数種，使用コード・ライブラリ版，問題生成規則，CPU/GPU，精度，タイムアウト，許容誤差を記録する．

## 2．方法IDの索引

| ID | Method | 主な役割 |
|---|---|---|
| M1 | Individual Newton Homotopy | 候補点ごとに $G_y=F-F(y)$ を構成 |
| M2 | Single-Point Coefficient Projection | 1候補への係数射影 |
| M3 | Multi-Point Constrained Projection | 共通開始系への係数射影 |
| M4 | Vanishing-Ideal / ABM-Based Construction | 共通開始系構成の一候補 |
| M5 | Structure-Preserving Reconstruction | 問題ファミリー内の再構成 |
| M6 | Linear Coefficient Path | 経路設計の基準 |
| M7 | Parameter Homotopy | パラメータ経路による追跡 |
| M8 | Bézier Path | 非線形の係数・パラメータ経路 |
| **M9** | Multistart Newton / LM / Trust Region | HCを使わない直接解法ベースライン |
| **M10** | Learned Candidate Generation / Ranking | 単一・複数候補生成・選択 |
| **M11** | Standard HomotopyContinuation.jl Baselines | Total Degree / Polyhedral / Parameter Homotopy等，該当対象で利用 |
| **M12** | Candidate / Path Policy Optimization | 予測・最適化・RL・サロゲート |
| **M13** | Linear Starting System | $G_{\mathrm L}(\boldsymbol{x})=\boldsymbol{x}-\boldsymbol{y}$，次数差とスケールの対照 |
| **M14** | Jacobian-Matched Linear Starting System | $G_{\mathrm{JM}}(\boldsymbol{x})=J_F(\boldsymbol{y})(\boldsymbol{x}-\boldsymbol{y})$，開始Jacobianの対照 |

詳細な数式はM1〜M5が [starting_system.md](starting_system.md)，M6〜M8・M12が [homotopy_paths.md](homotopy_paths.md)，M10が [data_and_learning.md](data_and_learning.md) にある．M11のソルバーの機能はR14を参照．

## 3．データセットと問題クラス

### 第1段階：小規模な合成系

- 正方系 $n\in\{1,2,3\}$ を基本とし，サポート，次数，係数スケーリングを制御する．
- 少なくとも1つの正則実数解が既知となるRoot-first生成を用意する．
- 複数Target Solutionsの埋め込みには，先に可解性を判定する．$r$（埋め込み解数）と $k$（推論時候補数）は独立に変える．
- 目的系をParameter-firstで生成する対照群を用意し，分布差を記録する．
- ベンチマークが簡単すぎるとNewton法で即座に解けるため，難易度別に整理する．

### 第2段階：実応用に由来する構造化多項式系

| ベンチマーク層 | 問題例 | 成功条件・評価時の注意 |
|---|---|---|
| 固定サポート合成系（診断） | Root-firstで既知の正則実数解を埋め込む正方系 | 開始点の品質・正則性・再構成可能性を切り分ける |
| Application-derived synthetic | P3P／Generalized Camera Resectioning | 追加観測に対する姿勢の整合性．ノイズ付き過剰決定PnPを正方の最小問題と混同しない |
| Application-derived synthetic | Mass-Action化学反応ネットワークの定常状態 | 正の濃度，反応速度定数の正値，保存則．安定性は別判定 |
| Application-derived synthetic | 位相を固定したKuramoto等の平衡点方程式 | 多項式化による円周制約と必要に応じた安定性 |
| Published / Real-world | 公開論文の具体例，観測由来のデータ | 設定・解の意味・既存専用ソルバーを再現可能に記録 |

**実応用は現時点で採用候補であり，学習用・評価用のファミリーを確定していない**．最初は固定サポート合成系に加え，化学反応ネットワークとCVから各1ファミリーの検証を候補とする．CRNの多重定常性検出やPnPの真の姿勢推定は，任意の実数解1つの探索より強いタスクであるため，成功条件を必ず別に定義する．原著・公式文書はR23，R24など．

### 分割と一般化

Training / Validation / Testを問題インスタンス単位で分割する．別の乱数種だけでなく，係数分布・条件付け・問題ファミリーの差を考慮した外挿評価も検討する．生成時の真の解はテスト時の開始系構成に使わない．

## 4．実験ID一覧

| ID | 研究上の問い | 比較 | 主な対応仮説 |
|---|---|---|---|
| **E0** | 同じ候補点からNewton法と複数の開始系HCを比較できるか | M9，M1，M2，M13，M14．低次元合成系，正則性反例 | N2，N4，N6 |
| **E1** | 複数点から正則な開始系を作れるか | $k$，サポート，評価行列ランク，M2〜M5 | N1，N2，N6 |
| **E2** | 共通開始系と独立開始系のどちらが有利か | M1/M2 vs M3/M4/M5，同一候補集合・同一M6 | N1，N2 |
| **E3** | Root-first合成データに偏りがあるか | Root-first vs Parameter-first | N3 |
| **E4** | 学習で候補を出す意味はあるか | Random，M9，単一M10，複数M10，ランキング | N3，N4 |
| **E5** | 経路形状の効果はどれくらいあるか | 同じStarting PairにM6，M7，M8 | N2，N5，N6 |
| **E6** | 共同最適化に独立最適化以上の利得があるか | 個別最適化 vs M12，推論コスト含む | N4，N5，N6 |
| **E7** | 実行可能解の実応用に有効か | 選定した物理問題クラスでM9/M11と比較 | N1〜N6の該当部分 |

### E0：Single-Candidate Diagnostic（最初の新予備実験候補）

**目的：** 同じ候補点でNewton法，damped/trust-region法，Newton Homotopy，Coefficient Projection，Linear Starting System，Jacobian-Matched Linear Starting Systemを比較する．新手法の新規性を主張する実験ではなく，開始系再構成の必要性を診断する．

$n=2$（変数数・方程式数），$d=3$（最大総次数）を初期例とする．$F:\mathbb R^2\to\mathbb R^2$ は目的多項式系，$f_i:\mathbb R^2\to\mathbb R$ はスカラー値成分多項式とする．既知の正則な埋め込み解 $\boldsymbol{x}^*\in[-1,1]^2$ に対し，候補点 $\boldsymbol{y}=\boldsymbol{x}^*+\sigma\boldsymbol\varepsilon$ を用意する．$\sigma\in\mathbb R_{\ge0}$ はスカラー，$\boldsymbol\varepsilon\sim\mathcal N(\boldsymbol{0}_2,I_2)$ は $\mathbb R^2$ 値ベクトル，$I_2\in\mathbb R^{2\times2}$ は単位行列である．$\sigma\in\{0.05,0.2,1.0\}$ は初期の例示値とする．

- **診断用のOracle候補**：$\boldsymbol{x}^*$ を使った摂動候補は，推論時の利用可能情報ではない．予備実験では候補品質感度分析として別枠で扱う．
- **実利用に近い候補**：目的解を参照せずに生成したランダム候補も別群で評価する．
- **比較**：M9（Newton／Trust Region），M1（Newton Homotopy），M2（Coefficient Projection），M13（Linear Starting System），M14（Jacobian-Matched）．M13/M14は次数差を含む比較なので，結果を係数再構成の効果と短絡しない．
- **開始正則性**：$J_G(\boldsymbol{y})\in\mathbb R^{2\times2}$ の最小特異値・非正則率を記録．M2が正則性を失う解析的な反例と，M1の経路が途中で特異となる解析的反例を小規模テストに含める．
- **公平性**：同じ問題・候補，精度，トラッカー・タイムアウトを揃え，開始系ごとの相対スケールも記録．候補生成コストと追跡コストを分けて計上する．

E0の結果が否定的でも，Structure-Preserving Reconstructionが有効な**特定の実応用ファミリー**での可能性まで否定しない．

### E1：実現可能性と正則性

候補数 $k\in\{1,2,4,8\}$ は**例示値**であり，Bézout/BKKとランク制約によって不可能な組を除外する．再構成が可能な割合，$\dim\ker V_i(C)$，$\sigma_{\min}(J_G(y))$，追加解の有無，生成時間を測る．不可能な点集合を「失敗」扱いする場合，その母数も明示する．

### E2：共通開始系 vs 独立開始系

同じ $C$ に対して，A：各点にM1/M2を構成，B：全点にM3/M4，可能ならC：M5を構成する．同じ線形係数パスM6を使い，1経路ずつ追跡し，最初の正則実数解を検証できた時点で停止する．**全候補を必ず追跡する実験とは区別する**．再構成不可能例は除外せず内訳を報告する．

### E3：データ生成監査

Root-firstとParameter-firstで係数・サポート・実解数・$J_F$ 条件付け・Newton法成功率・HC追跡費用の分布を比較する．同じ母集団を生成していると仮定しない．

### E4：学習による候補生成

単一回帰，複数候補生成，ランダム候補，残差等の簡単なランキングを比較する．候補数，同一データ量，同一推論予算を揃え，共通開始系を利用する場合と独立開始系の場合を分ける．まず固定M2/M6で候補の効果を分離する．

### E5：パス最適化の効果

同一$(G,x_0)$ にM6を適用した基準結果を作り，適用可能な場合のみM7，M8を比較する．Starting Pairを同時に変更してはならない．Bézierの制御点探索・モデル推論の費用も含む．再パラメータ化だけによる改善にも注意する．

### E6：共同最適化の追加利得

(1) 開始系のみ最適化，(2) パスのみ最適化，(3) 両者を独立に最適化，(4) 両者を共同最適化，を比較する．同一の候補生成・学習データ・計算予算に揃え，改善が相互作用に由来するか検証する．

### E7：実応用

物理制約 $x\in\Omega$ を具体化し，任意の実数解と実行可能解を区別する．モデル・サポート，条件の正確な引用，インスタンス分布，制約違反率，最初の実行可能解の認証費用を報告する．

## 5．評価指標

| 種類 | 指標 | 留意点 |
|---|---|---|
| **主指標** | Time to First Feasible Solution | 予測・再構成・経路設計・追跡・認証・再試行を含む |
| **主指標** | Success Rate @ Time Budget $B$ | 失敗の定義と時間予算を固定する |
| 計算量 | Wall time，CPU time，GPU time | 異種ハードウェアでは比較条件を明記 |
| Tracker | Accepted/Rejected steps，Newton iterations | 経路追跡の実装設定を揃える |
| 再構成 | Construction time，可解率，正則率 | 複数候補を構成できない場合も集計 |
| 数値安定性 | Jacobian condition estimates，residual | 適切な正規化を定義し，保証とは区別する |
| 解の性質 | Real / Feasible / Certified | 近似実数性と厳密認証を区別する |
| 学習 | 推論時間，$T_{\rm train}/Q$，メモリ | OnlineとAmortizedを分ける |

**実数解の検証：** 数値上 $\|F(x)\|\le\tau$，$\|\Im x\|\le\tau_{\rm real}$ とする暫定的判定と，区間演算等の認証を分ける．後者の候補はHomotopyContinuation.jlのCertification機能（R14）．正則な実数解の認証に失敗した場合を成功扱いしない設定を原則とするが，研究上の暫定評価では双方を明示して報告する．

**統計：** 複数乱数種，複数問題セット，信頼区間，対比較，難易度別集計を用いる．成功率が異なる手法について，成功例のみの平均時間の比較で性能を結論付けない．固定時間での成功確率と，打ち切りを扱う経過時間比較を併記する．

## 6．Go / No-Go判断（現時点の提案）

- E1で共通開始系がほとんど構成不能なら，N1の対象サポート／点集合の制約を見直す．
- E2で共通開始系が個別Starting Pairを上回らない場合，Vanishing Idealを中核とする必要性を再考する．
- E3で分布差が大きい場合，Root-first合成データだけで汎化性能を主張しない．
- E4で複数候補学習が単一回帰に勝たなければ，$k=1$ を維持する選択肢を残す．
- E5で非線形経路が改善しない場合，M6を採用してN3/N4に集中する．
- E6で共同最適化の利得がなく計算費用だけが増えるなら，N5を保留にする．
- HC全体がM9より高コストの場合，HCが有利になる問題分布や成功率条件を見直す．

実験結果が出た時点で判断の閾値・対象問題を再定義してよいが，変更した理由は記録する．

## 7．既存実験とのID・出力の管理

- 既存の実験ログは `docs/experiments/EXP-xxxx-*/`，学習・評価の成果物は主に `outputs/EXP-xxxx/` にある．**元の実験記録は数値・結論を書き換えない**．変更を加えるときは訂正・追記として履歴を残す．
- この計画のE0〜E7は**研究の問いに付けたID**であり，実行時の `EXP-xxxx` は既存リポジトリの実験ID管理規則に従って新たに割り当てる．同じEについて複数の `EXP-xxxx` があってよい．
- Git管理外の `outputs/` 等にチェックポイントが保存されており，成果物の保全とコードコミット・設定・インスタンスハッシュとの対応付けが必要である．2026-11-06のローカル監査では `outputs/` が約5.8 GB，チェックポイント約1514件と報告されたが，**この計画は個々のモデルの再現性を検証したものではない**．
- EXP-0061で見つかったCPU間の差は，実験条件を統一するだけでは消えない場合がある．**係数配列と基底行列を明示的に保存してインスタンスを固定する方針**を新実験の事前登録候補とする．現行ジョブを変更する指示ではない．

