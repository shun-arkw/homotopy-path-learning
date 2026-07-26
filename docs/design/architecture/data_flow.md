# データフロー

## 1．初期化

```text
YAML設定
  ↓
PolynomialSystemSpecの生成
  ↓
サポート，次数，係数添字の検証
  ↓
開始係数ベクトルの生成
  ↓
開始解集合の生成
  ↓
潜在基底Uの生成
  ↓
Juliaバックエンドの初期化
  ↓
Gymnasium環境とPPOの初期化
```

## 2．学習エピソード

初期の`BezierPhamEnv`は1ステップ環境である．
1回の`reset()`で目的係数を1つサンプリングし，1回の`step(action)`で
ベジェ制御点を評価して`terminated=True`を返す．

```text
目的係数のサンプリング
  ↓
完全な目的係数ベクトルc_Fの生成
  ↓
自由係数のみから状態sを生成
  ↓
ゼロ行動の線形制御点を追跡
  ↓
線形コストJ_linをエピソード内へキャッシュ
  ↓
PPOが潜在行動aを出力
  ↓
中間制御点摂動δ_kを生成
  ↓
ベジェ制御点P_0,...,P_d_bを構成
  ↓
先頭係数固定制約を検証
  ↓
Juliaへcontrol_pointsを送信
  ↓
全開始解から全パスを追跡
  ↓
TrackingResultを取得
  ↓
ベジェコストJ_bezを計算
  ↓
線形コストJ_linとの差から報酬を計算
  ↓
1ステップのエピソードとして終了
```

将来の複数ステップ化では，以下のようにPPO更新用データを継続的に保存する可能性がある．

```text
状態sの生成
  ↓
PPOが潜在行動aを出力
  ↓
中間制御点摂動δ_kを生成
  ↓
ベジェ制御点P_0,...,P_d_bを構成
  ↓
先頭係数固定制約を検証
  ↓
Juliaへcontrol_pointsを送信
  ↓
全開始解から全パスを追跡
  ↓
TrackingResultを取得
  ↓
ベジェコストJ_bezを計算
  ↓
線形コストJ_linとの差から報酬を計算
  ↓
PPO更新用データを保存
```

## 3．線形ベースライン

中間制御点を端点間の線形補間上に配置すると，ベジェ曲線は厳密な直線となる．したがって，ゼロ摂動

$$
\boldsymbol{\delta}_k=\boldsymbol{0}
$$

を線形ベースラインとして用いる．同じカスタムホモトピーと同じTracker設定を利用することで，実装差による比較の偏りを減らす．

学習中に同一目的系の$J_{\mathrm{lin}}$を何度も評価しないよう，エピソード開始時に1回計算し，エピソード内でキャッシュする．評価データセットについては，線形コストを事前計算して保存してもよい．

追跡結果では，個々のパスの数値的失敗は失敗ペナルティを持つ有限コストとして扱う．
一方，Juliaランタイム初期化失敗，Python–Julia通信失敗，返却payload不正，shape不正，NaN/Infは
プログラム上の例外として扱い，報酬計算へ変換しない．

## 4．評価

```text
固定評価データセットを読み込む
  ↓
各目的系に対して線形パスを追跡
  ↓
各目的系に対して学習済み方策を実行
  ↓
成功率，平均，中央値，標準偏差，最小，最大を集計
  ↓
パス単位と問題単位の失敗を集計
  ↓
CSV，JSON，Markdown表を出力
```

Phase 6の固定評価データは，目的係数そのものではなく
`evaluation.seed + instance_index`で生成されるseed列として保存する．
各seedについて，Linear，RandomBezier，LearnedBezierは同じ`env.reset(seed=...)`により
同一目的係数を使用する．

学習時のPPOは単一環境を逐次的に使用する．rolloutには観測，有界行動，pre-tanh行動，
old log probability，報酬，terminated/truncated，value，環境診断情報を保存する．
`terminated=True`ではGAEのbootstrapを停止する．
行動は`tanh`変換付き対角正規分布から生成し，action clippingは使用しない．
報酬clip，報酬正規化，観測正規化も使用しない．

出力ディレクトリは次の構造である．

```text
outputs/EXP-0004/<run-id>/
├── config.yaml
├── metadata.json
├── latent_basis.npy
├── train_metrics.csv
├── evaluation_seeds.json
├── checkpoints/
│   ├── last.pt
│   ├── best.pt
│   └── step_XXXXXXXX.pt
├── evaluation/evaluation.csv
├── benchmark/benchmark.csv
├── benchmark/benchmark_summary.csv
└── analysis/summary.md
```

## 5．再現性情報

各実験で，次を保存する．

- 実験ID
- Git commit hash
- 設定ファイルのコピー
- Python，Julia，主要パッケージのバージョン
- 乱数シード
- 固定サポートの指数配列
- 潜在基底$U$
- 学習済みモデル
- 評価データセットまたは生成シード
- 生の評価結果
