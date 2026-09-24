# CLE-HFL V0.7 Overleaf package

这是由V0.6不可变快照派生的内部会议稿。V0.7只同步最终论文定位与实验骨架，未伪造待运行结果。

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

## 当前最重要 pending

- local-first training seeds 1/2；
- second controlled private dataset；
- bounded taxonomy stress；
- FedMD four-objective cross-communication replication；
- held-out-map2 HFL mechanism-family Formal table。

benchmark/smoke 结果不得填写为 Formal 科学证据。
