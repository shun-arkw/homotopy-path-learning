# 研究計画：ホモトピー連続法による実数解探索の高速化

> **Version 0.2 / 2026-11-06 / 研究計画・作業メモ（暫定）**  
> **状態：既存の係数パス学習は実験済み．Starting Pair再構成と単一実数解探索の新構想は未実験．**  
> 本ディレクトリは，実装の確定仕様ではなく，複数の研究仮説・反例・文献・比較計画を管理する．既存コードや過去の実験記録を書き換える文書ではない．

## 0．既存研究との関係と現在の進捗

この計画は，既存の [homotopy-path-learning の `multivariate_system` ブランチ](https://github.com/shun-arkw/homotopy-path-learning/tree/multivariate_system) を継承・拡張するための**候補研究**である．現在の実装はすでに多変数・複素係数のPham型多項式系について，固定開始系から**全複素解経路**を追跡し，Bézier係数パスをPPO，Behavior Cloning，Oracle探索，Attention方策などで改善している．新しい構想は，「少なくとも1つの実数・実行可能解」を優先するため，問題分布・開始系・追跡対象・評価関数を変更するものである．**両者の結果は直接比較可能とは限らない**．

| 根拠 | 確認した内容 | 限界・状態 |
|---|---|---|
| リポジトリ `AGENTS.md`，`docs/design/mathematics/multivariate_system.md` | 既存Pham型の先頭係数，固定開始系，Bézierの始終点固定などの仕様 | 既存実装の仕様であり，新構想の必須条件ではない |
| `docs/design/decisions/ADR-0008-subspace-target-family.md` | 自由係数を複素低次元部分空間で生成する系の族を追加 | **代数的・物理的な実応用ファミリーとは異なる** |
| `docs/experiments/EXP-0059-*/README.md` | 一般の目的系で，行動をランダム部分空間に制限したOracle探索では探索時間短縮の条件を満たさなかった | 特定の系と探索設定に限る |
| `docs/experiments/EXP-0060-*/README.md` | 部分空間の目的系について，ファミリー基底を利用した探索で全次元行動より大きなステップ数削減が記録された | **学習済み方策の効果や総実行時間ではない** |
| `docs/experiments/EXP-0061-*/README.md` ＋ 2026-11-06のローカル監査 | 記録は `Planned` のままだが，サーバ側で実行中．Stage 0で計算機間の不一致があり，同一サーバで分母を再計算する代替手順へ進んでいる | 最終結果は未確認．手元の不要な追加探索は停止済み，サーバ側は継続中 |

**既存成果の正本はリポジトリの実験記録と出力データである．** 研究計画中では必要な概要のみを引用し，確認したZIPの時点（2026-11-06）以降の実験状況は自動更新されない．

### 保存場所・権限の境界

研究計画の配置候補は既存リポジトリの `docs/research-plan/` である．既存の `docs/design/` は実装が従う仕様，`docs/experiments/` は実験記録であり，本計画はそれらを上書きしない．既存 `AGENTS.md` の固定Pham型開始系の規則を変更する必要が生じたら，既存モードを壊さない**別モード・別仕様**として提案し，明示的に設計変更を承認してから実装する．

## 1．研究目的

パラメータ化された多項式系，あるいは指定サポートを持つ多項式系について，**少なくとも1つの実数解，必要に応じて実行可能解を求めるまでの総計算時間**を短縮する．その手段として，開始系・開始解の組（Starting Pair）の設計，複数候補点の活用，機械学習による候補生成，ホモトピー経路の設計・最適化を検討する．

基本的な**新構想の**対象は，$n\in\mathbb Z_{>0}$（スカラー）に対する正方多項式系 $F:\mathbb R^n\to\mathbb R^n$ であり，$\boldsymbol{x}\in\mathbb R^n$，$F(\boldsymbol{x})=\boldsymbol{0}_n$ と記述する．対象分布では，少なくとも1つの孤立した正則実数解が存在すると仮定する．応用問題では実行可能領域 $\Omega\subseteq\mathbb R^n$ を導入し，$F(\boldsymbol{x})=\boldsymbol{0}_n$ かつ $\boldsymbol{x}\in\Omega$ を満たす点を探す．実数解が存在しない問題，特異解のみの問題，正の次元の解集合を持つ問題は初期の保証範囲に含めない．

目的は「既知の教師解と同一の解を必ず回収すること」でも「全解の列挙」でもない．ただし，特定の応用において意味のある解が限定されるなら，その制約を明示する．

## 2．現時点の共通パイプライン

```text
学習・準備時：問題分布の設定 → Target System / Target Solutions のデータ生成・収集
                                  ↓
推論時：未知の Target System F
          ↓
       Candidate Point または Candidate Set の生成
          ↓
       Starting System Reconstruction（複数の候補方法）
          ↓
       正則性検査 → Starting Pair / Starting Solution Set
          ↓
       Homotopy Path Design（線形・Parameter・Bézier等）
          ↓
       Path Tracking / 必要に応じた再試行
          ↓
       実数性・残差・実行可能性の検証 → 最初の有効な Target Solution
```

上の各工程は**必須の実装仕様ではない**．例えば，学習を使わない場合，候補点を1個だけ用いる場合，Newton法で直接解ける場合も比較対象とする．共通開始系と独立開始系の優劣も未定である．

### 現段階で固定しない設計項目

- 開始候補点の個数 $k$：1点，複数点，適応的決定．
- 開始系の構成：Coefficient Projection，Structure-Preserving Reconstruction，Vanishing Ideal / ABM，個別Newton Homotopy等．
- 経路：Linear Coefficient Path，Parameter Path，Bézier Path，複素空間への迂回等．
- 学習：回帰，候補点集合予測，Flow Matching，ランキング学習，強化学習等．
- 主要な貢献：再構成アルゴリズム，理論，学習目的，経路設計，総実行時間の短縮など．

## 3．文書の構成

| ファイル | 責務 |
|---|---|
| [research_ideas.md](research_ideas.md) | 研究仮説・新規性候補・反論・採否判断・関連IDの索引 |
| [starting_system.md](starting_system.md) | 単一／複数開始解，Coefficient Projection，構造保存，Vanishing Ideal，成立条件と反例 |
| [data_and_learning.md](data_and_learning.md) | 教師データ生成，情報漏洩対策，機械学習の候補と損失設計 |
| [homotopy_paths.md](homotopy_paths.md) | 経路の数学的設計，正則性，条件数，Bézier，共同最適化 |
| [experimental_design.md](experimental_design.md) | 実験ID，公平な比較，ベースライン，評価，Go/No-Go条件 |
| [related_work.md](related_work.md) | 文献ID，先行研究の実際の貢献と限界，URL，文献確認状況 |

`README.md` は研究全体の入口と共通ルールを扱う．具体的な導出は各専門ファイルに記載し，ファイル間で長い説明をコピーせずリンクする．

## 4．ID Naming Conventions

| Prefix | Meaning | 対象 | 正本の置き場所 |
|---|---|---|---|
| `N` | Novelty / Research Hypothesis | 学術的新規性の候補・検証可能な仮説 | `research_ideas.md` |
| `M` | Method | 提案手法または比較手法 | 各技術ファイル・`experimental_design.md` の方法索引 |
| `E` | Experiment | 実験計画・結果への参照 | `experimental_design.md` |
| `R` | Reference | 原著論文・公式実装・文献 | `related_work.md` |

運用方針：

1. IDは**一意で不変**とし，一度割り当てたIDは再利用しない．項目の名称・説明・優先度・状態は更新可能である．
2. 仮説が本質的に別の問いに変わった場合は新IDを発行する．軽微な名称変更・日本語訳変更では同じIDを使う．
3. 他ファイルで紹介するときは，原則としてIDを併記する．例：`N1`，`M3`，`E2`，`R8`．
4. 新しい研究アイデアをすべて実験する必要はない．未検証・保留・棄却も履歴として残す．
5. `R` は文献で確認された内容を指し，研究の新規性そのものを証明するものではない．
6. 本計画の `E0`，`E1` などは**未実施の研究仮説検証計画**であり，既存リポジトリの `EXP-0004`〜`EXP-0061` と異なる名前空間である．`E` を実験実行済みと読まない．
7. 文献ID・研究IDは一度割り当てたら再利用しない．変更した場合には理由と旧IDとの関係を残す．

### Research Status（仮説・手法の進捗）

`Proposed`：提案のみ．`Under Investigation`：理論検討／実験中．`Supported`：限定した仮定・データで根拠あり．`Rejected`：特定の条件下で反証された．`On Hold`：保留．

### Evidence Level（記述の根拠）

`Established`：明記した仮定の下で数学的に導出できる事実，または原著で確認できる主張．`Hypothesis`：未検証の研究仮説．`Preliminary`：限定条件で確認した実験結果．`Open Question`：未解決事項．**既存実験の結果**を `Preliminary` と記載してよいが，**未実施のStarting Pair研究に流用してはならない**．

進捗と根拠の区分は独立である．例えば `Supported` でも学術的新規性が確定したことにはならない．

## 5．Terminology and Notation（暫定標準）

**用語と記号は将来変更可能である．** この表は現時点の同義語の混在を避けるための暫定的な標準であり，数学的定義を変更する場合は関連ファイルを同時に確認する．

| 英語標準表記 | 日本語 | 記号・注意 |
|---|---|---|
| Target System | 目的系 | $F:\mathbb K^n\to\mathbb K^n$，パラメータ系では $F(\boldsymbol{x};\boldsymbol{p}^*)$ |
| Target Solution | 目的系の解 | $\boldsymbol{x}^*\in\mathbb K^n$，$F(\boldsymbol{x}^*)=\boldsymbol{0}_n$ |
| Planted Target Solution | 埋め込み済み目的系の解 | データ生成時に既知．推論時には使用禁止 |
| Starting System | 開始系 | $G:\mathbb K^n\to\mathbb K^n$ |
| Starting Solution | 開始解 | $\boldsymbol{x}_0\in\mathbb K^n$，$G(\boldsymbol{x}_0)=\boldsymbol{0}_n$ |
| Starting Pair | 開始系・開始解の組 | $(G,\boldsymbol{x}_0)$；複数の既知解がある場合も1解ごとに1組 |
| Candidate Point | 候補点 | $\boldsymbol{y}\in\mathbb K^n$．目的系・開始系の解とは限らない |
| Predicted Candidate | 予測候補点 | $\hat{\boldsymbol{x}}\in\mathbb K^n$．ML生成点の一般名 |
| Candidate Set | 候補点集合 | $C=\{\boldsymbol{y}^{(1)},\ldots,\boldsymbol{y}^{(k)}\}\subset\mathbb K^n$ |
| Starting Solution Set | 開始解集合 | $X_0=\{\boldsymbol{x}_0^{(1)},\ldots,\boldsymbol{x}_0^{(k)}\}\subset\mathbb K^n$ |
| Homotopy | ホモトピー | $H(\boldsymbol{x},t)=\boldsymbol{0}_n$，$t\in[0,1]$ |
| Coefficient Path | 係数経路 | $\boldsymbol{c}(t)\in\mathbb K^m$ |
| Parameter Path | パラメータ経路 | $\boldsymbol{p}(t)\in\mathcal P\subseteq\mathbb K^q$ |
| Solution Path | 解経路 | $H(\boldsymbol{x}(t),t)=\boldsymbol{0}_n$ を満たす $\boldsymbol{x}(t)\in\mathbb K^n$ |
| Path Tracking | 解経路追跡 | Predictor-Corrector等の数値追跡 |
| Feasible Solution | 実行可能解 | $F(\boldsymbol{x})=\boldsymbol{0}_n$ と $\boldsymbol{x}\in\Omega\subseteq\mathbb R^n$ をともに満たす解 |
| Vanishing Ideal | 消失イデアル | $I(X_0)$，有限点集合の零点集合に対応 |
| Multi-Solution Reconstruction | 複数解再構成 | 複数候補を共有する開始系の構成．名称は暫定 |

**`Root` は一般的な多項式・論文固有の用語で使用してよいが，プロジェクト内の目的解・開始解は原則 `Solution` に統一する．** 文献の正式タイトルや技法名（Finite Root Generation等）は書き換えない．

**重要な区別：** $\boldsymbol{y}$ は候補点であり，$G(\boldsymbol{y})=\boldsymbol{0}_n$ が確かめられて初めて開始解となる．`Support-Preserving`（指定単項式の集合内に係数を制限すること）と `Structure-Preserving`（追加の代数・物理制約を保つこと）は同義ではない．また，$\operatorname{supp}(G)\subseteq A$ と厳密なサポート一致は異なる．

### 型・次元の原則

初出の数式では，関数の定義域と値域，またはスカラー・ベクトル・行列の次元を必ず示す．ベクトルは `\boldsymbol{}`，スカラーは通常の斜体，行列は $J_F,A,V,W$ 等とし，各箇所で形状を記載する．

| 記号 | 型・次元 | 意味 |
|---|---|---|
| $n,d,k,r,m,q$ | $\mathbb Z_{>0}$ のスカラー | 変数数，次数，候補数，埋め込み解数，自由係数数，パラメータ次元（文脈で区別） |
| $\boldsymbol{x},\boldsymbol{x}^*,\boldsymbol{x}_0,\boldsymbol{y}$ | $\mathbb R^n$ または $\mathbb C^n$ のベクトル | 変数，目的系の解，開始解，候補点（体は各実験で明記） |
| $F,G$ | $\mathbb K^n\to\mathbb K^n$ の多項式系，$\mathbb K\in\{\mathbb R,\mathbb C\}$ | 目的系・開始系 |
| $f_i,g_i$ | $\mathbb K^n\to\mathbb K$ のスカラー値多項式，$1\le i\le n$ | 多項式系の各成分 |
| $J_F(\boldsymbol{x}),J_G(\boldsymbol{x})$ | $\mathbb K^{n\times n}$ の行列 | 各多項式系のJacobian |
| $\boldsymbol{p},\boldsymbol{p}_0,\boldsymbol{p}^*$ | $\mathbb R^q$ または $\mathbb C^q$ のベクトル | 問題のパラメータ |
| $\boldsymbol{c}_F,\boldsymbol{c}_G$ | $\mathbb K^m$ のベクトル | 固定した係数基底における係数 |
| $\boldsymbol{\phi}_i(\boldsymbol{x}),\boldsymbol{a}_i,\boldsymbol{b}_i$ | $\mathbb K^{s_i}$ のベクトル，$s_i=|A_i|$ | 単項式評価・係数 |
| $V_i(C)$ | $\mathbb K^{k\times s_i}$ の行列 | $k$ 候補点での単項式評価行列 |
| $t$ | $[0,1]\subset\mathbb R$ のスカラー | ホモトピーのパラメータ |
| $H$ | $\mathbb K^n\times[0,1]\to\mathbb K^n$ の多項式系の連続族 | ホモトピー |

**系と成分を混同しない**：`多項式系 $F$` と `スカラー値の多項式 $f_i$` は別物である．実数解用の記法と既存の複素Pham系用の記法を混同しない．実験計画 `E` と既存実験 `EXP-` も区別する．

## 6．文書作成と変更のルール

- 日本語を基本とし，「だ・である」体，句読点は「，」「．」とする．英語タイトル・URL・コードは原綴りを維持する．
- 数式はLaTeXを使用し，正則性，ランク，実現可能性，対象とする体（$\mathbb R$ / $\mathbb C$）などの仮定を省略しない．
- 数学的な反例は反例であると明示する．一般論や高速化保証を無条件に主張しない．
- 技法の名称，根拠の強さ，文献への対応は `related_work.md` と照合する．原著を読めていない部分は「未検証」と書く．
- `research_ideas.md` では新規性を「候補」として扱い，実証前に確定しない．
- 数値実験では成功した問題のみの平均時間を主要指標としない．失敗率と再試行時間を含める．
- 用語変更は可能だが，同じ意味の表記を一括更新する．定義が変わる場合は変更理由・関係IDを記録する．
- Version 0.2は，既存実験の**限定的な確認済み結果**と，新構想の**未検証の仮説**を区分する．既存実験の元データをこの文書の記述で置き換えない．

## 7．現時点で明確なリスク

1. 任意の候補点集合を指定サポートや問題ファミリーに埋め込めるわけではない．
2. 既知の複数開始解と，目的系の複数実数解の存在は別概念である．
3. 正則なStarting Pairが構成できても，解経路の正則性・実数での到達・速さは保証されない．
4. 複数開始解の共通開始系は，独立開始系より良いとは限らない．
5. 学習で候補生成するコスト，再構成コスト，認証・再試行コストまで含めて比較する必要がある．
6. 「機械学習を使うこと」「Parameter Homotopyを使うこと」「Vanishing Idealを計算すること」はそれぞれ単独では学術的新規性とならない．

詳細： [研究仮説](research_ideas.md)，[開始系の成立条件](starting_system.md)，[検証計画](experimental_design.md)．
