# EXP-0004 Notes

- Phase 6では，action clippingを使わず，tanh-squashed Gaussianで常に行動空間内の行動を生成する．
- 報酬clip，報酬正規化，観測正規化は導入しない．
- 参照Dockerの既存PyYAMLは`5.4.1`であるため，実装は`yaml.safe_load`と`yaml.safe_dump`のみを使用する．
- smoke学習はパイプライン接続確認であり，PPOの有効性評価ではない．
