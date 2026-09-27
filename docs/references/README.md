# References

本ディレクトリは，設計仕様そのものではなく，設計の背景となった外部資料
（ドラフトノート，申請書など）を保管する．`design/`（現在の実装が準拠すべき
仕様）とは異なり，ここに置く資料は未確定の検討事項や，実装スコープ外の
プロジェクト全体像を含みうる．内容が `design/` と食い違う場合は，`design/`
側を正とする．

## ファイル一覧

- `pham_systems_homotopy_rl_notes.pdf`（旧`PhamSystem-1.pdf`）：Pham系
  （$f_i = x_i^{d_i} + h_i$の形の多項式系）の定義・性質，条件数，強化学習に
  よるホモトピーパス最適化の動機についての英語ドラフトノート．
  [`docs/design/mathematics/multivariate_system.md`](../design/mathematics/multivariate_system.md)
  の数学的背景に相当するが，"Condition number?" などの未確定事項を含む
  ドラフト段階の資料であり，仕様としては扱わない．
- `phc_sakura_2027_proposal.pdf`（旧`PhamSystem-2.pdf`）：PHC Sakura
  （日仏共同研究プログラム）への申請書．プロジェクトの背景，チーム構成，予算，
  Task 1（一変数）／Task 2（多変数Pham系・一般系）のロードマップを含む，
  プロジェクト全体の背景資料である．本リポジトリの実装フェーズ計画は
  [`docs/design/implementation/codex_implementation_plan.md`](../design/implementation/codex_implementation_plan.md)
  を正とする．
