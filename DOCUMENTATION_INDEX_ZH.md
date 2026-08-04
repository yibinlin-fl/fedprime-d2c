# FedPRIME-D2C 文档总索引

Updated: 2026-08-04

## 日常只读这两份

```text
AGENTS.md            Codex 的轻量入口、规则和当前实验
SESSION_HANDOFF.md   当前目标、正在运行的实验、下一步判据
```

新窗口通常不需要读取其他大文档。

## 当前正式实验

```text
STRICT_PEW_ASYMHFL_VAL_OPENI_RUN_ZH.md  当前 OpenI A/B 启动说明
CLE_HFL_V2_OPENI_RUN_GUIDE_ZH.md        CLE-HFL v2 通用平台说明
```

## 长期记忆与代码地图

这些文件很大，只在需要追溯精确历史时按需读取：

```text
CURRENT_PROJECT_MEMORY.md  按时间记录的重要实验决策与结果
PROJECT_STATE.md           已实现代码和实验状态的长日志
TODO_NEXT.md               历史与当前待办日志
ARCHITECTURE.md            代码模块和 runner 架构
EXPERIMENT_GUIDE_ZH.md      配置、指标与实验运行指南
RESEARCH_CODE_PRACTICES_ZH.md  科研代码实践
```

## 基线理解

```text
RAHFL_IMPLEMENTATION_READING_ZH.md  RAHFL 源码与训练流程精读
AGENT.md                            早期 D2C 约束，已经过时，禁止作为当前指令
```

## CLE-HFL 与当前研究问题

```text
FEDCLEAR_CLE_HFL_PROPOSAL_ZH.md
CLE_HFL_V2_FEDFALSIFY_FRAMEWORK_ZH.md
CURRENT_RESEARCH_STATUS_RAHFL_AND_COMMUNICATION_REVIEW_20260727_ZH.md
CURRENT_RESEARCH_STATUS_FOR_EXTERNAL_AI_ZH.md
```

## 已归档的候选方法与负结果

```text
CONTINUOUS_WITNESS_OFFLINE_AUDIT_ZH.md
FEDCIS_FRAMEWORK_AND_OFFLINE_AUDIT_ZH.md
FEDCISA_FRAMEWORK_REVIEW_AND_MODULE_SPEC_ZH.md
FEDFALSIFY_AUDIT_GUIDE_ZH.md
FEDFALSIFY_OPENI_RUN_GUIDE_ZH.md
FEDFALSIFY_LATEST_EXTERNAL_AI_DISCUSSION_BRIEF_ZH.md
FEDCLEAR_LATEST_THEORY_FRAMEWORK_ZH.md
FEDCLEAR_METHOD_DESIGN_REVIEW_ZH.md
FEDCLEAR_V2_REVISION_PLAN_ZH.md
FEDCLEAR_EXTERNAL_AI_REVIEW_BRIEF_ZH.md
FEDCARA_CURRENT_FRAMEWORK_EXPERIMENTS_AND_NEXT_PLAN_ZH.md
FEDSARA_CS_SCENARIO_OPENI_GUIDE_ZH.md
FRAMEWORK_REVIEW_C3LP_D2CCR_ZH.md
PROTOGRAPH_LITERATURE_REVIEW_AND_CTPG_FRAMEWORK_ZH.md
docs/C3LP_D2C_LITERATURE_AND_METHOD_ZH.md
```

这些文件用于保留研究证据，不代表仍应继续实现其中的方法。

## 历史 FedEASE 运行资料

```text
FEDEASE_V2_1_FRAMEWORK_AND_IMPLEMENTATION_ZH.md
FEDEASE_OPENI_RUN_GUIDE_ZH.md
FEDEASE_CALIBRATED_PEW_LOCAL_OPENI_RUN_ZH.md
FEDEASE_EBST_V2_OPENI_RUN_ZH.md
FEDEASE_PEW_EBST_V2_OPENI_RUN_ZH.md
```

当前只复用其中已经验证过的 calibrated PEW + BER+CDep 本地机制；EBST
与 EBST-v2 已经冻结为负结果。

## 汇报与外部讨论稿

```text
RECENT_PROGRESS_REPORT_2026_07_20_ZH.md
EXTERNAL_AI_RESEARCH_REVIEW_BRIEF_2026_07_22_ZH.md
CURRENT_RESEARCH_STATUS_FOR_EXTERNAL_AI_ZH.md
FEDCLEAR_EXTERNAL_AI_REVIEW_BRIEF_ZH.md
FEDFALSIFY_LATEST_EXTERNAL_AI_DISCUSSION_BRIEF_ZH.md
```

## 结果产物的位置

```text
outputs/       原始压缩包和运行输出，默认不提交 Git
deliverables/  已解析的表格、图和报告
local_runs/    本地数据和临时运行，默认不提交 Git
```

## 后续整理规则

1. 根目录只新增“当前交接页”或“当前正式运行指南”。
2. 新的实验结果首先更新 `SESSION_HANDOFF.md`，再按需追加长期日志。
3. 失败方法保留证据，但不再写入 `AGENTS.md` 详细展开。
4. 外部 AI 讨论稿统一以日期命名，后续逐步迁入 `docs/archive/`。
5. 在所有交叉引用修正前，不批量移动旧文件，避免链接失效。
