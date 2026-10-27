# CLE-HFL V0.8实验表矩阵

Updated: 2026-10-14

| 论文位置 | 表格文件 | 当前证据 | 待补实验 | 允许支持的结论 |
|---|---|---|---|---|
| Main Table 1 | `protocol_information_boundary.tex` | 协议与代码审计 | 无 | 训练/评价信息隔离与协议差异 |
| Main Table 2 | `local_first_multiseed.tex` | seeds 0/1/2完成；2 GO、1 undertrained | 无 | local-first在训练随机性下方向稳定；完整门槛状态透明报告 |
| Main Table 3 | `map2_method_comparison.tex` | 固定预算四臂seeds 0/1/2 Formal完成 | O1 JTT可选 | BER与ERM/CVaR/共享PEW-GroupDRO的matched机制比较；准确披露本地batch预算 |
| Main Table 4 | `cross_setting_dataset.tex` | original/map1/FedDF seed 0；map2三seed固定预算 | second private dataset + FedMD four-objective replication | 跨binding与通信底座的机制迁移；不能声称统一utility获益 |
| New Main/Appendix | `moderate_dose_response.tex` | 主评价协议gamma 0/0.5/0.9 ERM与gamma0.9 PEW+BER seed-0 Formal完成 | 无；多seed属于增强项 | 在固定partition/mask下验证剂量响应，并报告shortcut--utility trade-off |
| Main Table 5 | `hfl_context_baselines.tex` | 主评价协议Local/FedMD/FedProto/AsymHFL-ERM/AsymHFL+PEW+BER seed-0 Formal完成 | FedDF/AugHFL/RAHFL | CLE在代表性HFL与corruption-robust机制家族中的位置；八行完成前不做领域排名 |
| Appendix | `oracle_granularity.tex` | 已完成 | 无 | correspondence重要、operator粒度增益有限 |
| Appendix | `appendix/taxonomy_stress.tex` | 无 | bounded taxonomy stress | 一个预注册未见operator下的退化边界 |
| Appendix | `appendix/pew_audit.tex` | 已完成distribution audit | 可补PEW confusion | BER改变有效支持结构，不保证模型性能 |
| Appendix | `appendix/per_client_architecture.tex` | M1三seed已完成 | 无 | pooled结论是否掩盖客户端差异 |
| Appendix | `appendix/seed_uncertainty.tex` | M1三seed + checkpoint source CI | M2/M3 | 区分训练随机性与评价source不确定性 |
| Appendix | `appendix/computation_cost.tex` | 无统一计时 | benchmark manifests | PEW一次性成本、JTT双阶段成本、通信不变 |

## 填表顺序

```text
strong-stress held-out map2固定预算三seed -> 已完成机制证据；不再默认要求昂贵submission-budget复跑
主评价协议dose response -> 已完成并回填；seed-0 DSA随gamma单调，PEW+BER显著降低gamma=.9 DSA
主评价协议八行HFL context -> 五行已完成；FedDF/AugHFL/RAHFL三行待跑
local-first seeds 1/2 -> 已回填Main Table 2 + seed table
second private dataset -> Main Table 4 + seed table
旧strong Local/FedMD -> 仅内部工程记录，不进入论文表
optional JTT Formal -> Main Table 3 JTT row（仅在需要正式JTT主张时）
bounded taxonomy stress -> taxonomy stress appendix
benchmarks -> computation cost appendix
```

任何表都不得使用smoke或benchmark accuracy作为科学结果。
