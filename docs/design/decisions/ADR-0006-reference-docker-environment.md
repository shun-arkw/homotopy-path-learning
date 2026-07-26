# ADR-0006：参照実行環境をDockerコンテナとする

## 状態

Accepted.

## 背景

Phase 3以降では，Python，Julia，HomotopyContinuation.jl，
juliacallの互換性が実行環境に依存する．

Codex CloudではJulia 1.5.2が使用され，
Julia 1.11.1で生成された`julia/Manifest.toml`を読み込めなかった．

一方，本リポジトリの`docker/Dockerfile.sage-julia`は，
Julia 1.11.1を使用する環境として構成されている．

## 決定

本プロジェクトの参照実行環境を，
`docker/Dockerfile.sage-julia`から構築したDockerコンテナとする．

Phase 3以降のJulia単体テストおよびPython–Julia統合テストは，
参照Dockerコンテナ内で実行する．

以下を参照環境の一部として扱う．

- Julia 1.11.1
- Python 3.10
- Python仮想環境`/opt/pyenv`
- Julia depot`/opt/julia`
- リポジトリルート`/app`
- `julia/Manifest.toml`に記録されたJulia依存関係

Codex Cloudなど，参照Dockerと異なる環境での結果は補助的な確認とし，
正式な受入テスト結果とはしない．

## 影響

- Julia依存関係の生成と更新は参照Docker内で行う．
- `julia/Manifest.toml`を別のJuliaバージョンで更新しない．
- CodexによるPhase 4以降の実装とテストは，原則として
  参照Docker内で実行する．
- Docker環境を変更する場合は，Dockerfile，Project.toml，
  Manifest.tomlおよび統合テストを同時に確認する．