# FedPRIME-D2C 文档总索引

Updated: 2026-08-09

当前待运行的 Multi-label PEW + Soft-BER 配对筛选：

```text
docs/experiments/current/CLE_MULTILABEL_PEW_SOFTBER_OPENI_RUN_ZH.md
scripts/openi_cle_multilabel_softber_entry.py
```

## 2026-08-09 基线公平性与忠实度修复

```text
docs/research/baselines/BASELINE_FIDELITY_REPAIR_ZH.md
docs/experiments/current/CLE_BASELINE_FIDELITY_OPENI_RUN_ZH.md
deliverables/baseline_fairness_audit_20260809/BASELINE_FAIRNESS_AUDIT_ZH.md
```

## 新会话最小读取顺序

```text
AGENTS.md                  执行约束、科研纪律和文档放置规则
docs/handoffs/latest.md    当前目标、正式实验、待决策事项与下一入口
docs/README_ZH.md          按任务定位其余文档（本文件）
```

默认到此停止。只有当前任务需要精确历史、实现位置或旧实验依据时，才继续读取对应文件；禁止为了“全面理解项目”一次性加载全部长文档。

## 事实来源优先级

```text
相关代码/配置与当前 Git diff
> docs/handoffs/latest.md
> 当前实验指南
> docs/project/ 下的长期日志
> docs/archive/ 与历史 deliverables
```

## 最新正式实验

当前论文证据补全总入口：

```text
docs/experiments/current/CLE_HFL_PAPER_EVIDENCE_OPENI_RUN_ZH.md
docs/experiments/archive/CLE_PEW_LOO_OPENI_RUN_ZH.md               已完成 Strict PEW operator-LOO（GO）
docs/experiments/archive/CLE_REMAINING_BASELINES_OPENI_RUN_ZH.md   已完成 FedDF/KT-pFL/FCCL 匹配筛选
docs/experiments/archive/CLE_CDEP_V2_PAIRED_OPENI_RUN_ZH.md      已完成的 CDep-v2 共享PEW配对实验（NO-GO）
docs/experiments/archive/CLE_CDEP_V2_SINGLE_ARM_OPENI_RUN_ZH.md   已完成但归因不匹配的单臂实验
```

```text
docs/experiments/archive/STRICT_PEW_ASYMHFL_VAL_OPENI_RUN_ZH.md  已完成的 seed-0 strict A/B 说明（GO）
docs/experiments/archive/STRICT_PEW_ASYMHFL_VAL_MULTISEED_OPENI_RUN_ZH.md  已完成的 training-seed 0/1/2 复验（GO）
docs/experiments/current/STRICT_PEW_ASYMHFL_VAL_40ROUND_OPENI_RUN_ZH.md  当前 40 轮 training-seed 1/2 复验说明
docs/experiments/guides/CLE_HFL_V2_OPENI_RUN_GUIDE_ZH.md        CLE-HFL v2 通用平台说明
```

## 长期记忆与代码地图

这些文件很大，只在需要追溯精确历史时按需读取：

```text
docs/project/CURRENT_PROJECT_MEMORY.md  按时间记录的重要实验决策与结果
docs/project/PROJECT_STATE.md           已实现代码和实验状态的长日志
docs/project/TODO_NEXT.md               历史与当前待办日志
docs/project/ARCHITECTURE.md            代码模块和 runner 架构
docs/experiments/guides/EXPERIMENT_GUIDE_ZH.md      配置、指标与实验运行指南
docs/project/RESEARCH_CODE_PRACTICES_ZH.md  科研代码实践
```

## 基线理解

```text
docs/research/baselines/RAHFL_IMPLEMENTATION_READING_ZH.md  RAHFL 源码与训练流程精读
docs/archive/legacy/AGENT.md                            早期 D2C 约束，已经过时，禁止作为当前指令
docs/archive/legacy/experiment_plan.md                  最早期 D2C 实验计划，仅作历史背景
docs/archive/legacy/ARCHITECTURE_PRE_CLEANUP_2026_08_09.md  清理前的完整历史架构快照
```

## CLE-HFL 与当前研究问题

```text
docs/archive/methods/FEDCLEAR_CLE_HFL_PROPOSAL_ZH.md
docs/archive/methods/CLE_HFL_V2_FEDFALSIFY_FRAMEWORK_ZH.md
docs/research/status/CURRENT_RESEARCH_STATUS_RAHFL_AND_COMMUNICATION_REVIEW_20260727_ZH.md
docs/research/status/CURRENT_RESEARCH_STATUS_FOR_EXTERNAL_AI_ZH.md
```

## 已归档的候选方法与负结果

```text
docs/archive/methods/NEGATIVE_CODE_REMOVAL_INDEX_ZH.md
docs/archive/methods/CONTINUOUS_WITNESS_OFFLINE_AUDIT_ZH.md
docs/archive/methods/FEDCIS_FRAMEWORK_AND_OFFLINE_AUDIT_ZH.md
docs/archive/methods/FEDCISA_FRAMEWORK_REVIEW_AND_MODULE_SPEC_ZH.md
docs/experiments/archive/FEDFALSIFY_AUDIT_GUIDE_ZH.md
docs/experiments/archive/FEDFALSIFY_OPENI_RUN_GUIDE_ZH.md
docs/archive/methods/FEDFALSIFY_LATEST_EXTERNAL_AI_DISCUSSION_BRIEF_ZH.md
docs/archive/methods/FEDCLEAR_LATEST_THEORY_FRAMEWORK_ZH.md
docs/archive/methods/FEDCLEAR_METHOD_DESIGN_REVIEW_ZH.md
docs/archive/methods/FEDCLEAR_V2_REVISION_PLAN_ZH.md
docs/archive/methods/FEDCLEAR_EXTERNAL_AI_REVIEW_BRIEF_ZH.md
docs/archive/methods/FEDCARA_CURRENT_FRAMEWORK_EXPERIMENTS_AND_NEXT_PLAN_ZH.md
docs/experiments/archive/FEDSARA_CS_SCENARIO_OPENI_GUIDE_ZH.md
docs/archive/methods/FRAMEWORK_REVIEW_C3LP_D2CCR_ZH.md
docs/archive/methods/PROTOGRAPH_LITERATURE_REVIEW_AND_CTPG_FRAMEWORK_ZH.md
docs/archive/methods/C3LP_D2C_LITERATURE_AND_METHOD_ZH.md
```

这些文件用于保留研究证据，不代表仍应继续实现其中的方法。

## 历史 FedEASE 运行资料

```text
docs/archive/methods/FEDEASE_V2_1_FRAMEWORK_AND_IMPLEMENTATION_ZH.md
docs/experiments/archive/FEDEASE_OPENI_RUN_GUIDE_ZH.md
docs/experiments/archive/FEDEASE_CALIBRATED_PEW_LOCAL_OPENI_RUN_ZH.md
docs/experiments/archive/FEDEASE_EBST_V2_OPENI_RUN_ZH.md
docs/experiments/archive/FEDEASE_PEW_EBST_V2_OPENI_RUN_ZH.md
```

当前只复用其中已经验证过的 calibrated PEW + BER+CDep 本地机制；EBST
与 EBST-v2 已经冻结为负结果。

## 汇报与外部讨论稿

```text
docs/research/status/RECENT_PROGRESS_REPORT_2026_07_20_ZH.md
docs/research/status/EXTERNAL_AI_RESEARCH_REVIEW_BRIEF_2026_07_22_ZH.md
docs/research/status/CURRENT_RESEARCH_STATUS_FOR_EXTERNAL_AI_ZH.md
docs/archive/methods/FEDCLEAR_EXTERNAL_AI_REVIEW_BRIEF_ZH.md
docs/archive/methods/FEDFALSIFY_LATEST_EXTERNAL_AI_DISCUSSION_BRIEF_ZH.md
```

## 结果产物的位置

```text
outputs/       原始压缩包和运行输出，默认不提交 Git
deliverables/  已解析的表格、图和报告
local_runs/    本地数据和临时运行，默认不提交 Git
```

紧凑的 RAHFL CLE 基线轮次表位于：

```text
deliverables/baselines/rahfl_cle_alpha05_gamma09_seed0_round00_11.csv
deliverables/strict_pew_asymhfl_val_probe_20260804/RESULT_SUMMARY_ZH.md
deliverables/strict_pew_asymhfl_val_multiseed_20260804/RESULT_SUMMARY_ZH.md
deliverables/strict_pew_asymhfl_val_40round_seed0_20260805/RESULT_SUMMARY_ZH.md
```

## 文档目录职责

```text
docs/handoffs/             当前交接，只保留 latest.md
docs/project/              架构、长期记忆、实现状态、TODO、科研代码规范
docs/experiments/current/  正在运行或等待正式结果的实验
docs/experiments/guides/   可复用的基准与平台指南
docs/experiments/archive/  已完成、已替代或已冻结实验的运行资料
docs/research/status/      当前研究综述与带日期的阶段报告
docs/research/baselines/   基线实现精读
docs/archive/methods/      失败、冻结或被替代的方法证据
docs/archive/legacy/       仅保留来源的过时说明
```

## 后续维护规则

1. 根目录只保留 `README.md` 和 `AGENTS.md`，不再新增研究 Markdown。
2. 新正式结果先更新 `docs/handoffs/latest.md`，再按需追加长期记忆或产出分析报告。
3. 新建、移动、归档文档时同步更新本索引和有效交叉引用。
4. 失败方法保留证据并移入归档，不把它重新写成活动 TODO。
5. 原始日志、压缩包、检查点和数据不放入 `docs/`；分别使用 `outputs/`、`deliverables/`、`local_runs/`。
6. 历史 `outputs/` 与 `deliverables/` 中的快照不因目录整理而重写。
