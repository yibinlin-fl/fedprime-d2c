# CLE-HFL V0.8 内部写作包

这是由V0.7-writing派生的下一版论文快照。V0.7及V0.7-writing均未覆盖；V0.8只回填已经审计的
moderate seed-0 Formal，并同步论断边界，不伪造仍待运行的结果。

`Paper Review → Academic Writing Skills → Scientific Writing → Defensive Writing Auditor → Humanizer → Final Fact Audit`

## 编译

- `main.tex` 默认 `\draftskeletontrue`：显示 `[EVIDENCE NEEDED]` 和内部 pending 表，适合继续研究/写作。
- 若只查看已有证据，可临时改成 `\draftskeletonfalse`；不要因此删除 pending 状态。
- bibliography 使用 `natbib` + `plainnat`。

## 目录

- `sections/`：正文与附录；
- `tables/`：主表与附录表；
- `figures/`：五张论文图、PDF/SVG/PNG导出及可复现生成脚本；
- `references.bib`：参考文献；
- `EXPERIMENT_TABLE_MATRIX_ZH.md`：最新实验表矩阵；
- `CITATION_AUDIT_ZH.md`：引用边界；
- `CLAIM_EVIDENCE_AUDIT_ZH.md`：逐项论断—证据—禁用表述；
- `FIGURE_MANIFEST.md`：五张图的论断、证据来源和正文位置；
- `WRITING_STYLE_AUDIT_ZH.md`：去模板化、去重复和科学边界保护记录；
- `SOURCE_AUDIT_ZH.md`：源码静态完整性检查结果；
- `REVISION_LOG_ZH.md`：本轮修改范围与后续工作。

## 当前最重要 pending

- local-first training seeds 1/2；
- second controlled private dataset；
- bounded taxonomy stress；
- FedMD four-objective cross-communication replication；
- moderate Axis II剩余FedDF/AugHFL/RAHFL三行Formal（可按预算收缩）。

## 2026-10-13 双难度协议

- strong stress：CIFAR-10、`alpha=0.5`、`rho=1.0`、`gamma=0.9`，用于问题存在、DSA、
  local-first和Axis I shortcut-objective attribution；绝对准确率不得冒充普通CIFAR-10性能。
- moderate utility：CIFAR-10、`alpha=1.0`、`rho=0.5`、`gamma={0,0.5,0.9}`；seed-0
  dose-response及AsymHFL成对方法锚点已经完成，Axis II当前完成5/8行。
- moderate协议不是第二private dataset，不能关闭外部有效性缺口。
- V0.6/V0.7不可变；本版本已回填的唯一新Formal来源是
  `deliverables/cle_v2_moderate_axis2_dose_formal_seed0_20261025/RESULT_SUMMARY_ZH.md`。

benchmark/smoke 结果不得填写为 Formal 科学证据。

## 图表与实验骨架

- 正文已加入 RQ1--RQ7 表，分别对应存在性、local-first归因、匹配目标比较、跨通信、剂量响应、HFL机制家族和外部边界。
- 五张图均由 `figures/generate_paper_figures.py` 生成；没有使用生成式插画，也没有把pending结果画成完成证据。
- `figures/latex_includes.tex` 仅是图块参考片段，未被 `main.tex` 引用；正文中的五个图块已经分别放入相应章节。
