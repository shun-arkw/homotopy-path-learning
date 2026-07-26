# Documentation

本ディレクトリでは，設計仕様と実験記録を分離して管理する．

```text
docs/
├── design/
│   ├── mathematics/
│   ├── architecture/
│   ├── implementation/
│   └── decisions/
└── experiments/
    └── templates/
```

## `design/`

現在の実装が準拠すべき仕様を記載する．数学的定式化，システム構成，具体的な実装方針，重要な設計判断を含む．仕様を変更した場合は，該当文書を更新する．

## `experiments/`

実験ごとに新しいディレクトリを追加し，目的，仮説，設定，結果，考察，次の課題を記録する．過去の実験記録は原則として上書きしない．
