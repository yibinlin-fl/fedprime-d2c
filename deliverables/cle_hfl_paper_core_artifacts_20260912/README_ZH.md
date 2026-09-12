# CLE-HFL论文核心表图包

日期：2026-09-12

本目录把已经冻结、独立复算的Formal证据整理为可直接交给网页端GPT或论文排版使用的核心材料：

```text
PAPER_MAIN_TABLE_ZH.md                 正文主表、表注和放置建议
PAPER_MAIN_TABLE.csv                   可导入Excel/LaTeX的机器可读数值
PAPER_MECHANISM_EVIDENCE_CHAIN.png     汇报与网页端预览图
PAPER_MECHANISM_EVIDENCE_CHAIN.pdf     论文排版矢量图
PAPER_MECHANISM_EVIDENCE_CHAIN.svg     可编辑矢量源图
plot_paper_mechanism_figure.py         确定性生成脚本
```

推荐论文顺序：

```text
CLE场景定义
-> paired DSA诊断与识别理论
-> HFL-vs-Local local-first归因
-> PEW+BER局部干预
-> 主表：原map、新map1、第二通信底座
-> Oracle粒度与BER理论边界
-> limitations
```

图和表中的“GO”均按对应冻结estimand解释，不得用cross-map GO覆盖原Stage-2或FedDF的overall
NO-GO，也不得把两张binding map写成跨真实领域泛化。
