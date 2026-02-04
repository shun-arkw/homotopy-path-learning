# 仕様書：Bézierホモトピー中間制御点のPPO最適化（一変数版）

## 1. 目的とスコープ

### 1.1 目的
- 入力：多項式ペア$(G,F)$と$\gamma$（$|\gamma|=1$）  
- 出力：Bézierパスの中間制御点$P_1,\dots,P_{d-1}$  
- 評価：Julia HCソルバーで追跡し，成功率と計算コスト（`total_step_attempts`，`total_accepted_steps`，`total_rejected_steps`，`success_flag`）を改善する．ただし，`total_step_attempts` = `total_accepted_steps` + `total_rejected_steps` である．

### 1.2 スコープ（今回やること／やらないこと）
- やる：一変数，サポート固定（次数$0..n$），Bézier degree $d\in\{2,3\}$，連続action（潜在変数方式）  
- やらない：可変サポート（離散action），多変数拡張（将来拡張点として設計に含めるのみ）  

---

## 2. 用語と記号
- 次数上限：$n$，係数数：$D=n+1$  
- 実ベクトル化：$\tilde c(h)=[\Re c(h);\Im c(h)]\in\mathbb{R}^{2D}$  
- Bézier degree：$d$，制御点：$P_0,\dots,P_d$  
- 端点：$P_0=\gamma G$，$P_d=F$  
- 基準点：$\bar P_k=(1-k/d)P_0+(k/d)P_d$  
- 潜在次元：$m$，基底：$U\in\mathbb{R}^{2D\times m}$  
- 潜在変数：$z_k\in\mathbb{R}^m$，摂動：$Uz_k$  

---

## 3. 入出力データ仕様

### 3.1 多項式表現
- 形式：$h(x)=\sum_{j=0}^{n} c_j x^j$，$c_j\in\mathbb{C}$  
- 内部表現（Python側推奨）：`np.ndarray` complex shape `(D,)`  
- 実ベクトル化：`float32/float64` shape `(2D,)`  

### 3.2 $\gamma$ の表現
- 形式：$\gamma=\exp(i\theta)$  
- 保存：`theta`（float）または `gamma_re,gamma_im`  
- 制約：$|\gamma|=1$（生成時に正規化する）  

### 3.3 制御点の表現
- 端点$P_0,P_d$は多項式（係数ベクトル）として保持  
- 中間$P_k$は$\tilde c(P_k)\in\mathbb{R}^{2D}$を標準とする（Juliaに渡す直前に複素へ復元）  

---

## 4. Bézierパス生成仕様

### 4.1 中間制御点の生成（潜在変数方式）
- 入力：$\tilde c(\bar P_k)$，$U$，$z_k$  
- 生成：$\tilde c(P_k)=\tilde c(\bar P_k)+Uz_k$  

### 4.2 暴走防止（必須要件）
次のいずれか，または併用を必須とする．
- 潜在ノルム制限：$\|z_k\|\le\alpha$  
- 摂動ノルム制限：$\|Uz_k\|\le\beta$  
- 実装要件：clip後のノルムをログに出し，違反が無いことをテストで保証する．  

---

## 5. 強化学習環境（Gym）仕様

### 5.1 環境名とI/F
- 環境名：`BezierHomotopy1D-v0`  
- `reset(seed) -> obs, info`  
- `step(action) -> obs, reward, terminated, truncated, info`  

### 5.2 観測（obs）
obsは連結ベクトル（`float32`）とする．
- $\tilde c(F)\in\mathbb{R}^{2D}$  
- $\tilde c(G)\in\mathbb{R}^{2D}$  
- $(\Re\gamma,\Im\gamma)\in\mathbb{R}^2$  
- 追加（任意だが推奨）：$\log\|\tilde c(F)\|$，$\log\|\tilde c(G)\|$  

観測正規化：
- running mean/var による標準化を採用可能  
- 正規化の対象範囲（係数部のみ等）を明記し，再現性のため統計を保存する．  

### 5.3 行動（action）
- one-step（最初の実装で推奨）
  - $d=2$：`action shape = (m,)`  
  - $d=3$：`action shape = (2m,)`（$z_1,z_2$ を連結）  
- multi-step（将来拡張）
  - `action = delta z`，内部状態として$z$を環境が保持する．  

### 5.4 エピソード設計
- one-step：`step` 1回で`terminated=True`  
- multi-step：$T$回まで反復し，`truncated` で打ち切り可能  

---

## 6. 報酬仕様

### 6.1 HC評価結果の定義
Julia HCから以下を取得する．
- `success_flag ∈ {True, Flase}`  
- `total_accepted_steps`  
- `total_rejected_steps`  

### 6.2 コスト関数
$$
J=
\begin{cases}
\text{total\_accepted\_steps}+\rho \cdot \text{total\_rejected\_steps} & \text{success\_flag=True} \\
M & \text{otherwise}
\end{cases}
$$
- 仕様パラメータ：$\rho>1$，$M$はかなり大きい値とする．  

### 6.3 報酬
- one-step：`reward = -J`  
- multi-step：差分報酬$r_i = J_i - J_{i+1}$（最後の扱いを明記）  

---

## 7. Julia HC評価器（純関数API）仕様

### 7.1 Python→Julia呼び出しI/F
関数名例：`track_bezier_paths_univar`  
- 入力：
  - `degree`: `int`
  - `bezier_degree`: `int`
  - `control_points: complex[(d-1), D]` または `real[(d-1), 2D]`  
  - `solver_config`（許容誤差，最大steps，最大Newtonなど）  
- 出力：
  - `success_flag`
  - `total_step_attempts`
  - `total_accepted_steps`
  - `total_rejected_steps` 
  - `total_newton_iterations`（任意）  

### 7.2 同一条件比較の要件
- 線形パスとBézierパスで`solver_config`を同一にする．  
- 評価前ウォームアップを仕様として要求する（JIT影響回避）．  

---

## 8. データセットと再現性仕様

### 8.1 評価用インスタンス保存形式
1インスタンス$i$ごとに以下を保存する．
- `coeff_F_i`  
- `coeff_G_i`（生成規則が決定的なら省略可だが，保存を推奨）  
- `gamma_i` または `theta_i`  
- `seed_i`  
- メタ：`n, d, m`，生成分布パラメータ  

### 8.2 乱数と決定性
- Python, NumPy, PyTorch, Julia のseed管理方針を明記する．  
- $U$ の生成seed固定と保存（行列そのもの，またはseed＋生成手順）．  

---

## 9. ロギング仕様
- 1エピソードごとの記録：
  - `success, J, steps, newton_iters, wall_time`  
  - `||z||, ||Uz||, ||tilde c(Pk)||`（暴走検知）  
  - `gamma, instance_id, seed`  
- 集計指標：
  - 成功率  
  - 平均，中央値  
  - 分位点（p90,p95）  
  - 失敗理由別カウント（`status_code`がある場合）  

---

## 10. テスト仕様（最小必須）
- 変換テスト：複素↔実ベクトル化が可逆（許容誤差内）  
- clipテスト：$\|z\|\le\alpha$ または $\|Uz\|\le\beta$ が常に成立  
- 回帰テスト：固定インスタンス集合で，線形パスの結果が固定（成功率，平均stepsが再現）  
- APIテスト：Julia評価器が例外を投げずに`status_code`を返す（失敗時も含む）  

---

## 11. 受入基準（Definition of Done）
- ベースライン（線形＋$\gamma$）の評価が再現可能  
- PPOをつないで学習ループが回り，ログが欠損しない  
- 評価セットで「成功率」または「$J$の分位点」に改善が観測される  
- 設定と結果が全て保存され，別環境で再評価できる  
