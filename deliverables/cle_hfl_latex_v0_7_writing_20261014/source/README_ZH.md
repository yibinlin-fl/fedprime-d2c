# CLE-HFL V0.7-writing 工作包

这是由冻结V0.7快照派生的独立写作版。冻结版没有被覆盖；本目录用于论证、语言和证据边界优化，未伪造待运行结果。

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
- `CLAIM_EVIDENCE_AUDIT_ZH.md`：逐项论断—证据—禁用表述；
- `REVISION_LOG_ZH.md`：本轮修改范围与后续工作。

## 当前最重要 pending

- local-first training seeds 1/2；
- second controlled private dataset；
- bounded taxonomy stress；
- FedMD four-objective cross-communication replication；
- submission-budget Axis I逐臂Formal恢复；
- moderate Axis II剩余五行Formal。

## 2026-10-13 双难度协议

- strong stress：CIFAR-10、`alpha=0.5`、`rho=1.0`、`gamma=0.9`，用于问题存在、DSA、
  local-first和Axis I shortcut-objective attribution；绝对准确率不得冒充普通CIFAR-10性能。
- moderate utility：CIFAR-10、`alpha=1.0`、`rho=0.5`、`gamma={0,0.5,0.9}`，用于待补
  dose-response与Axis II完整HFL机制家族表。
- moderate协议不是第二private dataset，不能关闭外部有效性缺口。
- V0.6不可变；本骨架只同步实验职责，未回填任何不存在的结果。

benchmark/smoke 结果不得填写为 Formal 科学证据。
