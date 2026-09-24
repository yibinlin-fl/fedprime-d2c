# CLE-HFL LaTeX V0.7 内部骨架

本目录从 V0.6 不可变快照派生。V0.7 完成的是论文定位与实验骨架同步，不是最终投稿稿，也没有
把待运行实验写成已完成结果。

主要变化：

1. 统一定位为 CLE-HFL 的 diagnosis--attribution--mitigation study；
2. PEW+BER 写成 taxonomy-assisted、communication-agnostic local mitigation module；
3. 明确六段证据链：存在、归因、方法、跨通信、领域比较、外部边界；
4. 增加 FedMD 四目标跨通信复现槽位；
5. HFL 领域表固定到 held-out map2，并增加 AsymHFL--ERM 与 AsymHFL+PEW+BER 两行；
6. 面向论文的文字不再使用 M1/M2/S1/S2 等内部任务代号。

主源码：`source/main.tex`。

Overleaf上传包：`CLE_HFL_LATEX_V0_7_CLEAN_OVERLEAF_20260925.zip`。该包排除了V0.6的旧审计快照，
只保留编译与当前讨论所需文件。本机没有安装LaTeX编译器，
因此本轮只完成源码结构、引用路径和回归检查，尚未生成V0.7 PDF；请在Overleaf编译后回传PDF。
