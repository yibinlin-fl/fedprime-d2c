# V0.11网页审稿包

日期：2026-10-31。当前任务：等待积分期间，审查现有论文的表达和论证。
本包不是新论文版本，没有改写V0.11源码，也没有新增实验或接入新的配图。

## 第一轮最少上传四份材料

1. `CLE_HFL_LATEX_V0_11_FIGURES_20261028.pdf`：当前正文与表格；仍包含上一批插图。
2. `CLE_HFL_FULL_PAPER_WEB_HANDOFF_ZH.md`：更新到今天的完整证据交接，不是短增量。
3. `CURRENT_EVIDENCE_AND_REVIEW_LOCKS_ZH.md`：当前完成状态、科学边界、审稿标准。
4. `WEB_REVIEW_PROMPTS_ZH.md`：将第一阶段提示词复制进聊天框，不急于使用第二阶段。

精确修改源码时再上传`CLE_HFL_LATEX_V0_11_FIGURES_OVERLEAF_20261028.zip`。
讨论文献时再上传`references.bib`、`CITATION_AUDIT_ZH.md`；引用审计继承日期为2026-10-14，
本次没有重新核验全部外部文献。
讨论图片时再上传`figure_candidates/`中的五张PNG。它们是10月30日修订候选，尚未进入PDF。

`CLE_HFL_WEB_REVIEW_V0_11_20261031.zip`用于一次性传递/备份。本地解压后按上述顺序上传；
若网页端无法展开ZIP，直接上传独立PDF/MD，不假设它已经读到了ZIP内所有材料。
源码ZIP中的旧版本审计说明保留其历史日期，事实状态优先读本包最新事实锁与完整交接0.0节。

## 网页端输出回来后

已提前创建返回目录：

```text
C:\Users\asus\Desktop\FedPRIME-D2C\local_runs\paper_web_review_returns\v0_11_20261031
```

把完整意见保存为MD放入该目录，再让Codex核验“必须接受/可选/不适用建议”。
不用先自行识别每个英语写作问题；优先确认模型有没有说错研究对象、指标、图意或实验状态。
确认后再选择两个小节修改，最后核对事实和编译，避免整篇一键改写。

## 打包与保护

`build_review_package.py`只复制显式列出的文件，记录大小和SHA256，检查PDF/源码ZIP仍与原版一致，
并核验两组CSV内容。所有旧包均保留。`PACKAGE_MANIFEST.json`为包内文件清单；
`PACKAGE_AUDIT.json`记录最终ZIP哈希。本次不推送、不启动OpenI、不新增付费任务。
