# CLE-HFL有用结果留存审计（2026-10-25）

## 结论

当前论文主线依赖的核心正式证据均已有可定位的解析报告；V0.8及网页端交接使用的是这些报告，
不是聊天记忆。已核实存在的稳定结果目录包括：

| 证据 | 稳定目录 |
|---|---|
| strong CLE与local-first seed-0 | `deliverables/cle_v2_mechanism_stage1_20260910/` |
| held-out map2训练seed稳定性 | `deliverables/cle_hfl_map2_multiseed_20260922/` |
| map2四目标Formal | `deliverables/cle_v2_spurious_final_map2_20260913/` |
| Oracle粒度边界 | `deliverables/cle_v2_oracle_granularity_formal_20260911/` |
| FedDF跨通信证据 | `deliverables/cle_v2_feddf_plugin_formal_20260911/` |
| moderate剂量响应与Axis-II seed-0 | `deliverables/cle_v2_moderate_axis2_dose_formal_seed0_20261025/` |

上述目录至少包含正式结果摘要；moderate结果还包含论文表格和原始产物清单。理论证明、验证缓存、
论文快照和当前实验入口分别由`docs/research/status/`、`deliverables/`、Git历史和当前handoff索引。

## 必须保留的边界

`outputs/`与`local_runs/`通常不进入Git。Git能保护解析报告、代码、配置和论文，但不能代替原始
OpenI下载包、checkpoint和数据集压缩包的备份。因此“已保存”分为两层：

1. 论文可复核层：解析报告、表格、配置、哈希和结论已保留；
2. 原始重分析层：仍依赖本机`outputs/openi_downloads/`中的压缩包与recovery包。

不要删除上述原始目录。提交论文前应把关键Formal原始包再复制到一个独立磁盘或云盘；这属于备份，
不应把数百MB产物强行提交进Git。

