# CLE-HFL V0.6-SKILL Overleaf package

这是由 V0.5 经过真实上传 Skill 工作流生成的内部会议稿：

`Paper Review → Academic Writing Skills → Scientific Writing → Defensive Writing Auditor → Humanizer → Final Fact Audit`

## 编译

- `main.tex` 默认 `\draftskeletontrue`：显示 `[EVIDENCE NEEDED]` 和内部 pending 表，适合继续研究/写作。
- 若只查看已有证据，可临时改成 `\draftskeletonfalse`；不要因此删除 pending 状态。
- bibliography 使用 `natbib` + `plainnat`。

## 目录

- `sections/`：正文与附录；
- `tables/`：主表与附录表；
- `figures/`：两张论文图；
- `references.bib`：参考文献；
- `EXPERIMENT_TABLE_MATRIX_ZH.md`：最新实验表矩阵；
- `CITATION_AUDIT_ZH.md`：引用边界；
- `audit/`：V0.6-SKILL 执行日志、修改日志、事实审计和机器辅助审计结果。

## 当前最重要 pending

- M2 local-first seeds 1/2；
- M3 second private dataset：dataset/protocol pending；
- S1 bounded taxonomy stress：protocol pending；
- S2 Axis II Formal scientific metrics pending。

benchmark/smoke 结果不得填写为 Formal 科学证据。
