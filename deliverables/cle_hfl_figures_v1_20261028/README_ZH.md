# CLE-HFL五图候选总入口

更新：2026-10-30。V0.11保留原插图；当前五张独立修订候选尚未接入论文。

图3—5修订已完成：图3用清晰配对标记，图4保留十二个种子点与SD，图5保留全部效用差分。
三图统一Latin Modern字体及蓝灰配色，移除大标题、强调环和重复摘要条。
数据、坐标范围及旧版本已逐项核验，原尺寸/通栏效果已实际查看。
见[图3—5修订审查、复现及图注](REFINED_DATA_FIGURES_REVIEW_20261030_ZH.md)。

| 图 | 最新独立修订候选（figures/） | 当前V0.11仍使用 |
|---|---|---|
| 1 | `tikz_cle_paired_dsa_v7` | `fig_cle_paired_dsa_v4` |
| 2 | `tikz_pew_ber_execution_v7` | `fig_pew_ber_execution_v3` |
| 3 | `fig_local_first_multiseed_v2` | v1 |
| 4 | `fig_objective_tradeoff_v3` | v2 |
| 5 | `fig_dose_response_v4` | v3 |

2026-10-30在重新审查之后，已直接对照RAHFL/AugHFL/FCCL原图，用TikZ重绘图1/2：
`figures/tikz_cle_paired_dsa_v7`和`figures/tikz_pew_ber_execution_v7`为新的独立候选。
见[原文参考与重绘审查](REFERENCE_REDRAW_REVIEW_20261030_ZH.md)；全部嵌图显示比例=1.000，
像素保持不变，图2改为正确的损失/分组汇合与模型更新。V0.11仍采用下表旧图，尚未覆盖。
之前发现的问题仍保存在[重新审查报告](VISUAL_REAUDIT_20261030_ZH.md)和几何JSON中。

| 论文图 | V0.11已接入版本（历史选择） | 作用 |
|---|---|---|
| 1 | `figures/fig_cle_paired_dsa_v4` | 客户端CLE场景与同源paired DSA |
| 2 | `figures/fig_pew_ber_execution_v3` | 公共PEW、私有BER与保持不变的通信 |
| 3 | `figures/fig_local_first_multiseed_v1` | 三训练seed的Local/HFL effect与通信差分 |
| 4 | `figures/fig_objective_tradeoff_v2` | 四目标×三seed的DSA–utility位置与波动 |
| 5 | `figures/fig_dose_response_v3` | 主评价协议剂量响应及完整效用变化 |

每个stem对应`.pdf/.svg/.png`，PDF用于LaTeX，SVG可编辑，PNG用于预览。
全部Python生成器在`figures/`；数据图另附CSV和来源/环境/hash JSON。
图4/5每一版还附同名`.py`源码快照，便于对应版本。

环境：`D:\ResearchTools\envs\paper-figures\Scripts\python.exe`。
注意图2最初使用pytorch环境生成，具体版本保持在原记录；其他新图使用专用绘图环境。

- [整体规划](PAPER_PLAN.md)
- [版本历史](FIGURE_VERSION_LOG.md)
- [图3数据与审查](FIGURE_3_REVIEW_ZH.md)
- [图4/5数据与审查、复现和建议图注](FIGURES_4_5_REVIEW_ZH.md)

旧论文与中间版本全部保留。V0.11位于`../cle_hfl_latex_v0_11_20261028/`，其
`PDF_QA_ZH.md`记录实际分页、目视审查和待处理的会议页数问题。
本轮没有新增实验，不改变冻结科学结论。
