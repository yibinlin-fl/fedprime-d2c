# CLE-HFL V0.11：五图接入与版面审查

日期：2026-10-28。由V0.10不可变快照派生。V0.10与所有旧图/中间图均保留。
这是含待补实验占位的内部会议稿，不是已完成页数压缩的投稿定稿。

## 文件入口

- `source/main.tex`：分章节LaTeX源码。
- `CLE_HFL_LATEX_V0_11_FIGURES_20261028.pdf`：已编译并逐页目视检查的16页PDF。
- `CLE_HFL_LATEX_V0_11_FIGURES_OVERLEAF_20261028.zip`：根目录为main.tex的精简上传包。
- `PDF_QA_ZH.md`：图内文字、实际分页、图注和版面检查。
- `RELEASE_AUDIT.json`：保护性内容检查、五图SHA256、PDF/ZIP大小与SHA256。
- `audit_and_package.py`：重复核验与打包脚本，不训练模型、不修改科学结果。

## 本轮实际改动

1. 图1改为CLE场景与paired DSA：补充损坏条件C=1、合法target pair，保留训练/评价信息边界。
2. 图2改为PEW→BER执行图：与实现一致展示clean/noise/blur/weather/digital/unknown六种
   粗伪环境；支持矩阵和图像处理仅为示意，真实私有环境标签不进入训练。
3. 图3展示三seed的Local/HFL效应及有符号通信差分；右轴明确放大尺度。
4. 图4展示四目标×三seed的DSA–accuracy位置，误差条为训练seed样本标准差。
5. 图5展示三个实测ERM剂量点、gamma=.9的单个BER点，以及包括下降项在内的六种效用差分。
6. local-first原有正文和结果表移入Experiments，便于图序1→5；修改相应引用和图注。
7. 修正附录uncertainty表的超宽，只变列布局，不变单元格内容。

Abstract/Introduction/Related Work/Discussion/Conclusion/Appendix文本和references.bib与
V0.10逐字节相同；所有结果表内容、Method+Experiments数学表达式集合和citation keys核验不变。
未新增实验、未填充pending结果、未重新执行网页端写作技能链。

## 版本与后续

绘图工作区：`../cle_hfl_figures_v1_20261028/`，其版本日志记录全部图件历史。
本版先供作者确认页内视觉效果；后续选择目标会议模板，再集中压缩篇幅、处理浮动表位置，
以及按实际完成的实验替换pending占位。不要因为图件接入就声称投稿实验已经齐全。
本轮仅本地提交，未推送。
