# CLE-v2 FedDF-fidelity × PEW+BER 插件验证

Updated: 2026-09-11

状态：原matched-robust设计已作废；修正后的native-CE协议冻结，22项回归测试、本地CUDA smoke
和OpenI V100 benchmark通过。Formal未授权。

## 2026-09-11 OpenI benchmark

```text
1 round x 8 local batches/client
training 88.62 s, analysis 8.29 s, total 96.91 s
peak CUDA memory 2224.13 MB
4 paired standard-batch traces matched
verdict: BENCHMARK_ONLY_NO_SCIENTIFIC_DECISION
```

输入、配置哈希、FedDF诊断、checkpoint推理和20-source分析均完整且有限。两臂operator-grid
accuracy约10%，未达到学习下限；benchmark DSA/准确率及分析器门槛真假全部禁止作为方法证据。
按Formal 12轮、双倍local batch和完整评估保守预估0.5--1.0 V100 GPU-hour。Formal仍需用户
另行明确批准。

报告：`deliverables/cle_v2_feddf_plugin_benchmark_20260911/RESULT_SUMMARY_ZH.md`。

## 2026-09-11 本地验证

修正后的两臂均完成1个local batch/client、一次post-local FedDF server update、四模型checkpoint
保存与20-source paired分析；4条standard single-view batch trace完全匹配。FedDF teacher entropy、teacher
disagreement和server update诊断均有限，峰值显存约1.53GB。入口`--help`与Formal双锁通过。

```text
verdict: SMOKE_ONLY_NO_SCIENTIFIC_DECISION
```

单批次smoke的模型指标未达到学习下限；截断报告batch与balanced operator-grid还给出不同方向。
这不能用于判断方法有效或失败，但要求下一步先用OpenI `benchmark`检查8个local batches下的
数值/成本稳定性；禁止直接据此调参，也不得将smoke升级为论文证据。此前包含AugMix/JSD/DCL的
本地输出目录仅保留为作废审计，不得运行、引用或上传。

## 研究问题

检验coarse PEW+BER的CLE抑制是否能从AsymHFL通信迁移到第二种模型异构联邦通信。FedDF采用
仓库中已审计的`feddf_fidelity`核心机制适配，不使用历史`feddf`适配器，也不声称逐行复现官方
完整recipe。

```text
fd_b = standard CE + FedDF-fidelity communication
fd_p = PEW-grouped hard BER-weighted CE + identical FedDF-fidelity communication
```

两臂固定相同CLE-v2 `seed0_split0/gamma0.9`、training seed 0、四个异构模型、初始化、fit/audit、
私有single-view batch轨迹、公共batch、FedDF温度/优化器/步数与评价协议。两臂均禁用
AugMix、JSD、DCL与CDep。唯一处理差异是：candidate用冻结PEW family给标准逐样本CE分组，再以
BER替代CE均值；因此这是原生CE型FedDF适配器上的真正本地插件归因。仍应称
`protocol-matched FedDF-fidelity`，不能声称未经修改地复现官方完整训练recipe。

## 预算与冻结门槛

```text
smoke:     1 round x 1 local batch/client，无科学结论
benchmark: 1 round x 8 local batches/client，无科学结论
formal:    12 rounds x 16 local batches/client，未授权
```

- I0：两臂local batch/AugMix trace完全一致。
- L0：两臂operator-grid pooled accuracy均至少20%。
- P1：`DSA(fd_b)-DSA(fd_p)>=0.02`，source-bootstrap CI95下界大于0，且4/4客户端同向。
- P2：candidate-minus-base last-5 Avg不低于0，且candidate operator-grid pooled accuracy不比
  base低超过1pp。

Formal四门全过才是`GO_PEW_BER_FEDDF_PLUGIN_SEED0`。Worst/WCCA/CFG和逐客户端结果必须完整
报告，但不用于覆盖本实验限定的平均效用门；本实验不能建立架构一致效用、跨场景或通用HFL结论。
不得根据smoke/benchmark或Formal结果修改门槛、BER/PEW/FedDF参数或补seed翻案。

## 入口

```text
scripts/run_cle_v2_feddf_plugin.py
scripts/analyze_cle_v2_feddf_plugin.py
scripts/openi_cle_v2_feddf_plugin_entry.py
tests/test_cle_v2_feddf_plugin.py
```
