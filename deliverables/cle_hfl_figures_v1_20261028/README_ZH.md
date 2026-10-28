# CLE-HFL五图候选总入口

更新：2026-10-28。五张图已接入独立的V0.11排版审查稿；V0.10保持不变。

| 论文图 | 当前候选 | 作用 |
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
