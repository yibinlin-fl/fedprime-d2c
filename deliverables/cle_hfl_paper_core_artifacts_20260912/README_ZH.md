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

## 2026-09-13待回填项

原binding map五臂12轮screen已完成，晋级`ERM/CVaR-DRO/PEW+GroupDRO/PEW+BER`。用于最终
比较的held-out map2四臂40轮Formal已启动，结果尚未返回。本目录现有主表和机制图不提前写入
screen数值，也不预设map2结论；结果通过完整性审计和冻结门槛后再统一重生成表图。

```text
docs/experiments/current/CLE_V2_SPURIOUS_BASELINE_SCREEN_ZH.md
docs/experiments/current/CLE_V2_SPURIOUS_FINAL_MAP2_ZH.md
```
