# ADR-0002：PythonとJuliaの責務を分離する

## 状態

Accepted.

## 背景

PPOとGymnasiumはPythonで実装し，ホモトピー追跡にはHomotopyContinuation.jlを使用する．既存一変数実装でもPythonとJuliaを連携しているが，環境クラスとバックエンドの責務が密結合している．

## 決定

問題生成，係数パス構築，強化学習，報酬計算をPythonで実装する．多項式評価，ヤコビ行列評価，Taylor係数評価，全パス追跡をJuliaで実装する．両者の境界は数値配列と型付き結果に限定する．

## 理由

- 各言語の強みを利用できる．
- 数値追跡をHomotopyContinuation.jlへ集約できる．
- Gym環境からJulia実装の詳細を分離できる．
- 将来的に別バックエンドを追加できる．

## 影響

- 配列shape，dtype，添字順序を文書化し，統合テストを用意する必要がある．
- Julia側の公開APIを`api.jl`へ限定する．
