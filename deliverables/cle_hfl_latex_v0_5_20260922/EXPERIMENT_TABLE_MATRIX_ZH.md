# CLE-HFL V0.5实验表矩阵

Updated: 2026-09-22

| 论文位置 | 表格文件 | 当前证据 | 待补实验 | 允许支持的结论 |
|---|---|---|---|---|
| Main Table 1 | `protocol_information_boundary.tex` | 协议与代码审计 | 无 | 训练/评价信息隔离与协议差异 |
| Main Table 2 | `local_first_multiseed.tex` | seed 0 | M2 seeds 1/2 | local-first在训练随机性下是否复现 |
| Main Table 3 | `map2_method_comparison.tex` | 四臂seed 0 | M1 seeds 1/2；O1可选 | BER与ERM/CVaR/共享PEW-GroupDRO的matched比较 |
| Main Table 4 | `cross_setting_dataset.tex` | original/map1/map2/FedDF seed 0 | M1与M3 | 跨binding、通信底座和第二private task边界 |
| Main Table 5 | `hfl_context_baselines.tex` | 无正式数字 | S2 | CLE在代表性HFL协议中的文献坐标 |
| Appendix | `oracle_granularity.tex` | 已完成 | 无 | correspondence重要、operator粒度增益有限 |
| Appendix | `appendix/taxonomy_stress.tex` | 无 | S1 | 一个预注册未见operator下的退化边界 |
| Appendix | `appendix/pew_audit.tex` | 已完成distribution audit | 可补PEW confusion | BER改变有效支持结构，不保证模型性能 |
| Appendix | `appendix/per_client_architecture.tex` | 部分缓存存在 | M1审计 | pooled结论是否掩盖客户端差异 |
| Appendix | `appendix/seed_uncertainty.tex` | seed 0 source CI | M1/M2/M3 | 区分训练随机性与评价source不确定性 |
| Appendix | `appendix/computation_cost.tex` | 无统一计时 | benchmark manifests | PEW一次性成本、JTT双阶段成本、通信不变 |

## 填表顺序

```text
M1 -> Main Table 3/4 + per-client + seed table
M2 -> Main Table 2 + seed table
M3 -> Main Table 4 + seed table
S2 -> Main Table 5
O1 -> Main Table 3 JTT row（仅在需要正式JTT主张时）
S1 -> taxonomy stress appendix
benchmarks -> computation cost appendix
```

任何表都不得使用smoke或benchmark accuracy作为科学结果。
