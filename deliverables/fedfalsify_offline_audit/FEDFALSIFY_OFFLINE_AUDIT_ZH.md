# FedFalsify v0.1 离线审计摘要

> 这里使用独立 `test_same` 标签做离线 Go/No-Go 审计，绝不能把这些标签接入正式训练路由。

## 核心结果

| gamma | 外来环境生存差距 | min=5 预计样本激活率 | FRA+TAU 组合激活率 | 直接 KD 相对 CE 净变化 | CMT 相对 CE 净变化 | TAU 精度/召回 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.0016 | 0.1156 | 0.2381 | -0.0421 | 0.0019 | 0.9028/1.0000 |
| 0.6 | 0.0692 | 0.0823 | 0.1905 | -0.0489 | 0.0019 | 0.9041/1.0000 |
| 0.9 | 0.1559 | 0.0450 | 0.1429 | -0.0724 | 0.0017 | 0.8571/1.0000 |

## 冻结判断

1. **问题信号成立。** 随着纠缠增强，模型对自己环境的相对优势显著扩大，外来客户端知识更容易失效。
2. **直接 peer KD 明显不安全。** 它相对本地 CE 的净影响为负，且随 gamma 增大进一步恶化。
3. **CMT 方向正确但效应很小。** 三档 gamma 都只有约 0.0017-0.0019 的平均审计损失净改善。
4. **TAU 有筛选价值，但计算代价高。** 一步审计中精度约 0.86-0.90、召回率为 1.00。
5. **FRA 是当前瓶颈。** 强纠缠时完整组合只覆盖约 14% 的 source-receiver-class，投影到训练样本约 4.5%。

因此，FedFalsify v0.1 **不应直接进入昂贵的 40 轮训练**。

## v0.2 Source Ranking Audit

| gamma | policy | coverage | positive precision | selected mean increment | policy mean including abstention |
|---:|---|---:|---:|---:|---:|
| 0.0 | foreign_accuracy_top1 | 1.0000 | 0.9143 | 0.003474 | 0.003474 |
| 0.0 | fra_top1 | 0.4571 | 0.8750 | 0.004387 | 0.002005 |
| 0.0 | tau_top1 | 1.0000 | 0.9143 | 0.003537 | 0.003537 |
| 0.0 | head_tau_top1 (agree=0.8000) | 1.0000 | 0.9143 | 0.003477 | 0.003477 |
| 0.0 | oracle_increment_top1 | 1.0000 | 0.9143 | 0.003629 | 0.003629 |
| 0.6 | foreign_accuracy_top1 | 1.0000 | 0.9143 | 0.003612 | 0.003612 |
| 0.6 | fra_top1 | 0.4000 | 0.8571 | 0.004622 | 0.001849 |
| 0.6 | tau_top1 | 1.0000 | 0.9429 | 0.003668 | 0.003668 |
| 0.6 | head_tau_top1 (agree=0.6286) | 1.0000 | 0.9143 | 0.003171 | 0.003171 |
| 0.6 | oracle_increment_top1 | 1.0000 | 0.9429 | 0.003752 | 0.003752 |
| 0.9 | foreign_accuracy_top1 | 1.0000 | 0.8286 | 0.002804 | 0.002804 |
| 0.9 | fra_top1 | 0.3143 | 0.8182 | 0.004263 | 0.001340 |
| 0.9 | tau_top1 | 1.0000 | 0.8571 | 0.003197 | 0.003197 |
| 0.9 | head_tau_top1 (agree=0.7429) | 1.0000 | 0.8571 | 0.002809 | 0.002809 |
| 0.9 | oracle_increment_top1 | 1.0000 | 0.8857 | 0.003456 | 0.003456 |

`TAU Top-1` 在三档 gamma 都覆盖全部可审计 receiver-class，并且接近离线 oracle。FRA Top-1 的单次增益较高，但覆盖率随纠缠增强降至 31.4%。因此 v0.2 应使用 TAU 作为安全门和来源排序依据，FRA 只作软先验或平局处理。

Head-only TAU 的来源一致率为 80.0% / 62.9% / 74.3%，正净增益精度为 91.4% / 91.4% / 85.7%。它略逊于 full TAU，但分类头参数只占各模型约 0.067%-0.224%，适合作为 12 轮 probe 的默认低成本版本。

这仍然只是一步更新审计，不等价于 12 轮收益证明。

## 复现实验

```powershell
conda run -n pytorch python -u scripts/audit_fedfalsify_foreign_tensor.py --batch-size 512
conda run -n pytorch python -u scripts/audit_fedfalsify_gate_coverage.py --bootstrap-repeats 200 --min-audit-per-class 5 --output-dir outputs/fedfalsify_audit/gate_coverage_min5
conda run -n pytorch python -u scripts/audit_fedfalsify_one_step.py --gamma 00
conda run -n pytorch python -u scripts/audit_fedfalsify_one_step.py --gamma 06
conda run -n pytorch python -u scripts/audit_fedfalsify_one_step.py --gamma 09
conda run -n pytorch python -u scripts/audit_fedfalsify_source_ranking.py
conda run -n pytorch python -u scripts/summarize_fedfalsify_audit.py
```
