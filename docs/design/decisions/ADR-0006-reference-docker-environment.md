# ADR-0006：参照実行環境をDockerコンテナとする

## 状態

Accepted．

## 背景

Phase 3以降では，Python，Julia，HomotopyContinuation.jl，
juliacallの互換性が実行環境に依存する．

Codexの実行環境ではJulia 1.5.2が使用され，
Julia 1.11系で生成された`julia/Manifest.toml`を読み込めなかった．

一方，プロジェクトのDockerコンテナ
`homotopy-continuation`では，Julia 1.11.1，
Python 3.10.12およびJulia depot `/opt/julia`を使用しており，
PythonテストとJuliaテストの双方が成功している．

## 決定

本プロジェクトの正式な参照実行環境を，
Dockerコンテナ`homotopy-continuation`とする．

Phase 3以降のJulia単体テストおよびPython–Julia統合テストは，
このDockerコンテナ内で実行する．

Codexからは，`docker exec`を使用してコンテナ内のテストと
実装確認を行う．

## 理由

- Python，JuliaおよびJuliaパッケージのバージョンを統一できる．
- `julia/Manifest.toml`と実行時Juliaの不一致を防げる．
- Python–Julia統合テストの再現性を確保できる．
- Codexの実行環境に依存しない受入判定ができる．

## 影響

- Codex環境やホスト環境のみで実行したJuliaテストは，
  正式な受入テストとして扱わない．
- Julia依存関係の生成および更新は参照Docker内で行う．
- `julia/Manifest.toml`を別のJuliaバージョンで更新しない．
- Docker環境を変更する場合は，Dockerfile，Project.toml，
  Manifest.tomlおよび統合テストを同時に確認する．
