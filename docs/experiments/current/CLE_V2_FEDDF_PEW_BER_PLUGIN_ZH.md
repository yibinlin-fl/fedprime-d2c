# CLE-v2 FedDF-fidelity × PEW+BER 插件验证

Updated: 2026-09-11

状态：协议冻结，12项聚焦测试和两臂本地CUDA smoke通过；OpenI benchmark与Formal未授权。

## 2026-09-11 本地验证

两臂均完成1个local batch/client、一次post-local FedDF server update、四模型checkpoint保存与
20-source paired分析；4条local batch/AugMix trace完全匹配。FedDF teacher entropy、teacher
disagreement和server update诊断均有限，峰值显存约1.53GB。入口`--help`与Formal双锁通过。

```text
verdict: SMOKE_ONLY_NO_SCIENTIFIC_DECISION
```

单批次smoke的模型指标未达到学习下限，且插件臂早期准确率很低。这不能用于判断方法有效或失败，
但要求下一步先用OpenI `benchmark`检查8个local batches下的数值/成本稳定性；禁止直接据此调参，
也不得将smoke升级为论文证据。

## 研究问题

检验coarse PEW+BER的CLE抑制是否能从AsymHFL通信迁移到第二种模型异构联邦通信。FedDF采用
仓库中已审计的`feddf_fidelity`核心机制适配，不使用历史`feddf`适配器，也不声称逐行复现官方
完整recipe。

```text
fd_b = AugMix/JSD/DCL + FedDF-fidelity communication
fd_p = fd_b + frozen coarse PEW + hard BER
```

两臂固定相同CLE-v2 `seed0_split0/gamma0.9`、training seed 0、四个异构模型、初始化、fit/audit、
私有batch/AugMix轨迹、公共batch、FedDF温度/优化器/步数与评价协议。CDep禁止。唯一处理差异是
PEW+BER开关，因此本实验支持的是“在matched robust local backbone下迁移到FedDF通信”，不是
“未经修改的官方FedDF+插件”。

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
