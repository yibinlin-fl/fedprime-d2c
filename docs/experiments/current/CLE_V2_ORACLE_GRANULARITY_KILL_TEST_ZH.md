# CLE-v2 Oracle Family / Operator / Random 粒度 Kill Test

Updated: 2026-09-11

状态：Formal已完成并独立复算；判定`NO_GO_OPERATOR_GRANULARITY_GAP`，层次化PEW训练不授权。
此前13项聚焦测试及三臂本地CUDA smoke通过；中止的本地benchmark残留仍不得引用。

## 2026-09-11 Formal结果

```text
DSA: OF 0.017232, OO 0.015783, RO 0.067895
RO-OO: 0.052112, CI95 [0.051231, 0.052983]  -> G1 PASS
OF-OO: 0.001449, CI95 [0.001086, 0.001790]  -> G2 FAIL (<0.02)
grid Avg: OF 21.9333, OO 21.5200, RO 18.7817 -> L0 FAIL (RO<20)
last-5 Avg: OF 21.0960, OO 20.0273          -> G3 FAIL by 0.0687pp beyond tolerance
I0: 48 paired traces matched                -> PASS
```

即使忽略很窄的G3失败，决定性G2仍只达到冻结绝对门槛的约7.24%，结论不会改变。真实operator
关联相对随机分组有效，但operator细分相对family没有实质增益，且按客户端并不一致。因此不得
实现operator-level/hierarchical PEW，不得通过调权重、补seed或改门槛复活。

完整报告：

```text
deliverables/cle_v2_oracle_granularity_formal_20260911/RESULT_SUMMARY_ZH.md
```

## 问题与边界

本实验只判断：在固定CLE-v2强shortcut场景中，真实operator分组是否相对真实family分组提供可观测
且有任务效用的额外收益。Oracle读取私有真实operator metadata，仅作为不可部署上界；任何结果都
不能把Oracle包装成训练方法，也不能直接证明层次化PEW有效。

```text
h9_of = oracle family + hard BER + AugMix/JSD/DCL + strict AsymHFL-val
h9_oo = oracle operator + hard BER + AugMix/JSD/DCL + strict AsymHFL-val
h9_ro = class-wise shuffled oracle operator + same hard BER/base
```

`h9_ro`在每个客户端、每个类别内部打乱operator标签，严格保持每个`class x operator`计数，仅破坏
样本与operator的对应关系。三臂固定相同初始化、fit/audit划分、batch/AugMix轨迹、通信、优化器、
BER参数和报告协议。CDep与learned PEW均禁止。

## 冻结配置

```text
scenario: seed0_split0 gamma0.9
training seed: 0
models: ResNet10, ResNet12, ShuffleNet, MobileNetV2
formal: 12 rounds, 16 local batches/client/round, batch 64
BER: support_gamma 0.5, count_cap 32, min_group_count 2
family groups: 6-slot PEW-compatible axis (实际使用noise/blur/weather/digital)
operator groups: 15
paired DSA: 1000 sources x 15 operators x 4 clients, severity 3
bootstrap: source-paired 2000 samples
```

## 冻结门槛

- `I0`：三臂local batch/AugMix trace完全一致。
- `L0`：三臂operator-grid pooled accuracy均至少20%。
- `G1` operator association：`DSA(RO)-DSA(OO)>=0.02`且source-bootstrap CI95下界大于0。
- `G2` granularity gap：`DSA(OF)-DSA(OO)>=0.02`且source-bootstrap CI95下界大于0。
- `G3` utility noninferiority：OO的operator-grid pooled accuracy和last-5 Avg分别不低于OF超过1pp；
  OO四客户端零召回类别总数不高于OF。

Formal四门全过才记为`GO_OPERATOR_GRANULARITY_GAP`并允许讨论层次化PEW；任一失败即
`NO_GO_OPERATOR_GRANULARITY_GAP`。不得通过修改BER参数、删除客户端/类别、补seed或改窗口翻案。

## 分阶段预算

```text
smoke:     1 round x 1 local batch/client，执行链验证，无科学结论
benchmark: 1 round x 8 local batches/client，成本估计，无科学结论
formal:    12 rounds x 16 local batches/client，当前锁定
```

smoke只证明三种分组方式、训练、检查点和分析链路可执行，其数值没有科学意义。正式实验实际
耗时`12426.45 s`（3.4518 V100 GPU-hours），其中训练`12365.30 s`、分析`61.15 s`。

入口：

```text
scripts/run_cle_v2_oracle_granularity.py
scripts/analyze_cle_v2_oracle_granularity.py
scripts/openi_cle_v2_oracle_granularity_entry.py
tests/test_cle_v2_oracle_granularity.py
```
