# 负结果代码移除与证据归档索引

Updated: 2026-08-09

## 目的

本索引用于在精简活动代码的同时，永久保留已经探索过的方法、否定结论和证据位置。
代码移除不等于抹除研究历史，也不改变既有实验结论。原始结果、解析产物和历史文档不因本次整理而删除或重写。

统一的长期时间线与精确结论仍以以下文件为准：

```text
docs/project/CURRENT_PROJECT_MEMORY.md
```

## 第一阶段：与正式训练入口隔离的离线审计代码

第一阶段只处理从未接入当前 `run_experiment.py` 主线、且仅被自身审计脚本和单元测试引用的模块。
移除前已通过静态引用检查确认：当前 PEW+BER、AsymHFL、RAHFL 和外部基线运行路径不依赖这些文件。

状态：已于 2026-08-09 完成。移除后再次检查 `fedprime/`、`scripts/` 和 `tests/`，未发现对上述模块的悬空导入；当前通信、外部基线、忠实度和 PCCD 兼容路径的聚焦回归为 `30 passed`。

### FedCIS-v0

结论：离线可辨识性假设未获支持，冻结为 `NO-GO`；不进入 12/40 轮训练。

历史证据：

```text
docs/archive/methods/FEDCIS_FRAMEWORK_AND_OFFLINE_AUDIT_ZH.md
docs/archive/methods/FEDCISA_FRAMEWORK_REVIEW_AND_MODULE_SPEC_ZH.md
docs/project/CURRENT_PROJECT_MEMORY.md  -> FedCIS Offline Audit Result
```

移除的活动代码：

```text
fedprime/analysis/fedcis.py
scripts/audit_fedcis_sensitivity.py
tests/test_fedcis_audit.py
```

### Taxonomy-free continuous witness

结论：手工连续 nuisance witness 的离线审计未通过，冻结为 `NO-GO`。

历史证据：

```text
docs/archive/methods/CONTINUOUS_WITNESS_OFFLINE_AUDIT_ZH.md
docs/project/CURRENT_PROJECT_MEMORY.md  -> Continuous Taxonomy-Free Witness Audit
```

移除的活动代码：

```text
fedprime/methods/continuous_nuisance.py
scripts/audit_continuous_nuisance.py
tests/test_continuous_nuisance.py
```

### FedCFSA

结论：四客户端 CLE-HFL v2 下覆盖审计为 `NO-GO`；K=8 冗余分析只保留为历史可行性观察，未升级为当前方法。

历史证据：

```text
docs/project/CURRENT_PROJECT_MEMORY.md  -> Historical: FedCFSA Multi-Client Redundancy Audit
docs/project/CURRENT_PROJECT_MEMORY.md  -> Historical: FedCFSA Coverage Audit
deliverables/fedcfsa_coverage_audit_20260727/
deliverables/fedcfsa_source_redundancy_audit_20260727/
```

移除的活动代码：

```text
fedprime/analysis/fedcfsa_coverage.py
scripts/audit_fedcfsa_source_redundancy.py
tests/test_fedcfsa_coverage.py
```

### FedRIFT / robust frontier audit

结论：robust frontier 可作为可靠性诊断，但未达到替代当前协作机制所需的精度门槛；冻结为历史探索。

历史证据：

```text
docs/project/CURRENT_PROJECT_MEMORY.md  -> robust frontier / FedRIFT 相关记录
deliverables/robust_frontier_audit_20260726/
```

移除的活动代码：

```text
fedprime/analysis/robust_frontier.py
scripts/audit_robust_frontier.py
scripts/audit_robust_frontier_one_step.py
tests/test_robust_frontier_audit.py
```

## 暂不移除的高耦合负结果

下列方法虽已冻结，但实现仍与统一 runner、数据划分或当前本地训练路径共享代码。必须先完成惰性加载、模块拆分和黄金回归，才可逐文件移除：

```text
D2C / Oracle D2C
FedPRIME-PAIR / CPAD
PRAC-HFL communication
FedCARA v1 communication
FedCLEAR v0.1 / PCCD
EBST / EBST-v2
FedFalsify v0.2/v0.3
CDep v1/v2
```

其中 `fedprime/data/fedfalsify.py` 仍承载当前 strict fit/audit 划分逻辑；
`local_rahfl.py`、`local_fedease.py` 等也仍被活动方法复用，因此不能按名称直接删除。

## 第二阶段：独立旧 runner

### FedPRIME-PAIR / CPAD

状态：已于 2026-08-09 从统一入口解除注册并移除活动实现。

结论：CPAD 未超过匹配的 LogitAvg；类别对 public logits 没有形成足以改善 missing/tail class 的有效互补信号，冻结为负结果。

历史证据：

```text
docs/project/CURRENT_PROJECT_MEMORY.md  -> FedPRIME-PAIR / CPAD route
docs/research/status/CURRENT_RESEARCH_STATUS_FOR_EXTERNAL_AI_ZH.md  -> 4.2 FedPRIME-PAIR / CPAD
analysis_fedprime_pair_results/
```

移除的活动代码与配置：

```text
fedprime/methods/cpad.py
fedprime/methods/fedprime_pair.py
configs/debug_fedprime_pair_cifar10c.yaml
configs/kaggle_t4_fedprime_pair_full.yaml
scripts/run_kaggle_pair.sh
scripts/analyze_pair_expertise.py
```

解析产物 `analysis_fedprime_pair_results/` 保留，不参与训练。

### D2C / Oracle D2C / LogitAvg-PRIME runner

状态：已于 2026-08-09 从统一入口和旧 engine 导出中解除注册，并移除专属实现、测试及配置族。

结论：跨域 public logits 估计私有类别先验没有改善 missing/tail class；即使使用真实 private prior，Oracle D2C 仍未优于普通 LogitAvg，冻结为 `NO-GO`。

历史证据：

```text
docs/archive/methods/C3LP_D2C_LITERATURE_AND_METHOD_ZH.md
docs/archive/methods/FRAMEWORK_REVIEW_C3LP_D2CCR_ZH.md
docs/project/CURRENT_PROJECT_MEMORY.md  -> D2C / Oracle D2C 结果
docs/research/status/CURRENT_RESEARCH_STATUS_FOR_EXTERNAL_AI_ZH.md
```

移除范围：

```text
fedprime/methods/d2c.py
fedprime/methods/fedprime_d2c.py
tests/test_d2c_diagnostics.py
configs/*fedprime_d2c*.yaml
configs/*logitavg_prime*.yaml
configs/ablations/fedprime_d2c_*.yaml
```

`local_prime.py` 和 PRIME augmentation vendor 没有随 D2C 删除，因为其他历史/活动本地训练路径仍可能复用这些通用组件。

### PRAC-HFL communication runner

状态：已于 2026-08-09 从统一入口解除注册，专属通信 runner、运行配置和脚本已移除。

结论：PRAC-HFL 的 public-logit 教师选择/安全门控没有形成可推广收益，通信路线冻结。曾借用该 runner 的 AugMix-DCL、NIR-DCL 和 SARA local-only 配置不是随 PRAC 结论一起删除；这些配置已迁移到当前 `rahfl` runner，并显式设置 `communication: none`，从而保留历史本地消融的可复现入口。

历史证据：

```text
docs/project/CURRENT_PROJECT_MEMORY.md
docs/research/status/CURRENT_RESEARCH_STATUS_RAHFL_AND_COMMUNICATION_REVIEW_20260727_ZH.md
deliverables/prac_hfl_public4_analysis/
deliverables/prac_hfl_safe_analysis/
deliverables/prac_vs_rahfl_analysis/
```

移除范围：

```text
fedprime/methods/prac_hfl.py
configs/debug_prac_hfl_cifar10c.yaml
configs/kaggle_t4_prac_hfl.yaml
configs/kaggle_t4_prac_hfl_public1_lite.yaml
scripts/run_kaggle_prac.sh
```

### FedFalsify v0.2/v0.3 router

状态：已于 2026-08-09 移除专属 router、transfer、evidence、experiment、审计脚本、测试和配置族。

结论：source ranking、覆盖率与 one-step 审计不足以支持可靠路由；v0.2/v0.3 均冻结为 `NO-GO`，不得重新作为活动方法。

当前正式实验仍需要的分层 fit/audit 划分已独立迁移为：

```text
fedprime/data/strict_fit_audit.py
```

迁移只改变模块/符号和日志名称，不改变索引生成、持久化 split、loader 或 final-test 隔离语义；`tests/test_strict_asymhfl.py` 继续验证该协议。

历史证据：

```text
docs/experiments/archive/FEDFALSIFY_AUDIT_GUIDE_ZH.md
docs/experiments/archive/FEDFALSIFY_OPENI_RUN_GUIDE_ZH.md
docs/archive/methods/FEDFALSIFY_LATEST_EXTERNAL_AI_DISCUSSION_BRIEF_ZH.md
docs/archive/methods/CLE_HFL_V2_FEDFALSIFY_FRAMEWORK_ZH.md
docs/project/CURRENT_PROJECT_MEMORY.md
deliverables/fedfalsify_probe_analysis_20260723/
deliverables/fedfalsify_v03_probe_analysis_20260724/
```

移除范围：

```text
fedprime/methods/fedfalsify_experiment.py
fedprime/methods/fedfalsify/
scripts/*fedfalsify*.py
tests/test_fedfalsify_audit.py
configs/*fedfalsify*.yaml
```

### FedCLEAR / FedCLEAR-PCCD 独立 runner

状态：已于 2026-08-09 从统一入口和 AsymHFL runner 完整拆除；独立 experiment、CCRE 本地路径、IRD/PCCD 通信方法体、OpenI/分析脚本、测试和专属配置均已移除。

结论：FedCLEAR v0.1、IRD 和 PCCD 路线未通过既定审计，冻结为负结果。

历史证据：

```text
docs/archive/methods/FEDCLEAR_LATEST_THEORY_FRAMEWORK_ZH.md
docs/archive/methods/FEDCLEAR_METHOD_DESIGN_REVIEW_ZH.md
docs/archive/methods/FEDCLEAR_V2_REVISION_PLAN_ZH.md
docs/archive/methods/FEDCLEAR_EXTERNAL_AI_REVIEW_BRIEF_ZH.md
docs/archive/methods/FEDCLEAR_CLE_HFL_PROPOSAL_ZH.md
docs/project/CURRENT_PROJECT_MEMORY.md
```

本批移除范围：

```text
fedprime/methods/fedclear.py
fedprime/methods/fedclear_pccd.py
fedprime/methods/ccre.py
fedprime/methods/local_fedclear.py
fedprime/methods/ird.py
fedprime/methods/pccd.py
configs/*fedclear*.yaml
scripts/openi_fedclear_entry.py
scripts/openi_fedclear_pccd_entry.py
scripts/analyze_fedclear_probe.py
scripts/analyze_pccd_probe.py
```

### FedCARA v1

状态：已于 2026-08-09 移除 `fedcara` 入口、CARA-C 类别加权通信分支、`cara_l` 本地别名和两份专属配置。通用 NIR-DCL/SARA/AsymHFL 实现保留。

结论：FedCARA v1 没有形成稳定优于 RAHFL 的协作收益，冻结为负结果。

历史证据：

```text
docs/archive/methods/FEDCARA_CURRENT_FRAMEWORK_EXPERIMENTS_AND_NEXT_PLAN_ZH.md
docs/project/CURRENT_PROJECT_MEMORY.md
deliverables/fedcara_analysis/
```

### CDep v1/v2 与 EBST/EBST-v2/SCP

状态：已于 2026-08-09 从当前 `local_fedease` 和统一 runner 拆除。当前本地方法代码只保留 `PEW 环境注释 + BER + AugMix/JSD + DCL`；当前 CLE-HFL 基准配置不再声明 CDep、EBST 或 SCP。

结论：CDep-v1 多个 lambda 未显示增益，CDep-v2 的共享-PEW配对实验四个预注册门槛全部失败；EBST/EBST-v2 归因实验同样为负结果。它们不再作为论文当前方法组件。

历史证据：

```text
docs/experiments/archive/CLE_CDEP_V2_SINGLE_ARM_OPENI_RUN_ZH.md
docs/experiments/archive/CLE_CDEP_V2_PAIRED_OPENI_RUN_ZH.md
docs/archive/methods/FEDEASE_V2_1_FRAMEWORK_AND_IMPLEMENTATION_ZH.md
docs/experiments/archive/FEDEASE_EBST_V2_OPENI_RUN_ZH.md
docs/experiments/archive/FEDEASE_PEW_EBST_V2_OPENI_RUN_ZH.md
docs/project/CURRENT_PROJECT_MEMORY.md
deliverables/fedease_ebst_attribution_20260722/
deliverables/fedease_pew_ebst_v2_analysis_20260721/
```

移除的核心模块：

```text
fedprime/methods/conditional_dependence.py
fedprime/methods/environment_structural_transfer.py
fedprime/methods/safe_communication_projection.py
```

已完成的旧 12/40 轮结果与配置语义仍由历史文档、结果压缩包和 Git 历史保存；不得把清理后的 PEW+BER 配置误称为对旧 PEW+BER+CDep 运行的逐字节复现。

## 保留与恢复边界

- 保留 `docs/archive/`、`docs/project/CURRENT_PROJECT_MEMORY.md` 和相关 `deliverables/`。
- 保留 `outputs/` 中用户下载的原始实验压缩包，不进行清理。
- 不把已否定的方法重新列入活动 TODO，也不通过调权重重新包装为新结论。
- 已被 Git 跟踪的后续移除项可从 Git 历史恢复；未跟踪的离线审计源码以本索引、历史文档和产物作为研究存底。
- 每一阶段清理后都必须运行当前主线与基线的聚焦回归测试。
