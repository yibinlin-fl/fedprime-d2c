# CLE-HFL V0.7实验表矩阵

Updated: 2026-09-25

| 论文位置 | 表格文件 | 当前证据 | 待补实验 | 允许支持的结论 |
|---|---|---|---|---|
| Main Table 1 | `protocol_information_boundary.tex` | 协议与代码审计 | 无 | 训练/评价信息隔离与协议差异 |
| Main Table 2 | `local_first_multiseed.tex` | seed 0 | training seeds 1/2 | local-first在训练随机性下是否复现 |
| Main Table 3 | `map2_method_comparison.tex` | 四臂seeds 0/1/2 Formal完成 | O1可选 | BER与ERM/CVaR/共享PEW-GroupDRO的matched比较及训练随机性稳定性 |
| Main Table 4 | `cross_setting_dataset.tex` | original/map1/FedDF seed 0；map2三seed | second private dataset + FedMD four-objective replication | 跨binding、通信底座和第二private task边界 |
| Main Table 5 | `hfl_context_baselines.tex` | AsymHFL两行已有；十个HFL行待跑 | held-out-map2 Formal | CLE在代表性HFL协议与机制家族中的文献坐标；FedTGP为protocol-matched core adapter |
| Appendix | `oracle_granularity.tex` | 已完成 | 无 | correspondence重要、operator粒度增益有限 |
| Appendix | `appendix/taxonomy_stress.tex` | 无 | bounded taxonomy stress | 一个预注册未见operator下的退化边界 |
| Appendix | `appendix/pew_audit.tex` | 已完成distribution audit | 可补PEW confusion | BER改变有效支持结构，不保证模型性能 |
| Appendix | `appendix/per_client_architecture.tex` | M1三seed已完成 | 无 | pooled结论是否掩盖客户端差异 |
| Appendix | `appendix/seed_uncertainty.tex` | M1三seed + checkpoint source CI | M2/M3 | 区分训练随机性与评价source不确定性 |
| Appendix | `appendix/computation_cost.tex` | 无统一计时 | benchmark manifests | PEW一次性成本、JTT双阶段成本、通信不变 |

## 填表顺序

```text
held-out map2三seed -> 已完成：Main Table 3/4 + per-client + seed table
local-first seeds 1/2 -> Main Table 2 + seed table
second private dataset -> Main Table 4 + seed table
HFL context Formal -> Main Table 5
optional JTT Formal -> Main Table 3 JTT row（仅在需要正式JTT主张时）
bounded taxonomy stress -> taxonomy stress appendix
benchmarks -> computation cost appendix
```

任何表都不得使用smoke或benchmark accuracy作为科学结果。
