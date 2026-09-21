# CLE-HFL LaTeX V0.4

Updated: 2026-09-21

V0.4是V0.3的模块化版本。此版本只拆分文件，不改写论文内容、公式、实验数字、引用键、证据占位符
或科学边界。将所有`\input`递归展开并忽略空白后，V0.4与V0.3内容完全一致，校验结果为
`CONTENT_EQUIVALENCE=PASS`。

## 目录职责

```text
main.tex                         模板、宏包、标题、模块装配和参考文献入口
sections/abstract.tex            摘要
sections/introduction.tex        引言
sections/related_work.tex        相关工作
sections/method.tex              问题定义、DSA、形成分析、PEW与BER
sections/experiments.tex         实验正文
sections/discussion.tex          讨论与局限
sections/conclusion.tex          结论
sections/appendix.tex            附录
tables/heldout_map2.tex          map2四臂主表
tables/cross_setting.tex         跨设置DSA表
tables/oracle_granularity.tex    Oracle粒度表
figures/                         当前两张论文图的PDF与SVG
references.bib                   BibTeX数据库
WEB_REVISION_AND_FIGURE_GUIDE_ZH.md  网页端去AI味与绘图改造指南
```

## 编辑规则

1. `main.tex`只负责模板和文件装配，正文改在`sections/`中。
2. 数值表改在`tables/`中，不要在正文复制第二份数字。
3. 图文件继续放在`figures/`；图中结论必须能追溯到冻结结果。
4. 确定投稿会议后，优先替换`main.tex`中的document class、宏包和参考文献样式。
5. 网页端与仓库不得同时编辑。网页端完成一轮后，下载完整Source ZIP交回仓库做diff和事实审计。
6. 所有`[EVIDENCE NEEDED]`必须保留，直到相应Formal结果完成并通过审计。

当前source of truth是本目录。V0.3继续作为拆分前的不可变对照。
