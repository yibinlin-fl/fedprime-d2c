# CLE-HFL LaTeX V0.5

Updated: 2026-09-22

V0.5在保留V0.4全部已核验正文、公式、引用和seed-0数字的前提下，建立投稿版实验表骨架。
V0.4保持不可变，V0.5成为当前LaTeX source of truth。

## 本轮变化

1. 新增实验协议与信息边界表；
2. 将local-first表从正文抽离为独立文件，并明确M2 seeds 1/2；
3. map2四臂表已填入M1 seeds 0/1/2 mean$\pm$std，并为条件性JTT Formal保留行；
4. 跨设置表已填入held-out map2三seed结果，并为CIFAR-100第二private task保留行；
5. 新增S2-v2十臂HFL context基线表，覆盖FedMD/FedProto/FedTGP/FedDF/KT-pFL/FCCL/RHFL/AugHFL/RAHFL与Local；
6. 新增taxonomy stress、PEW审计、逐客户端、seed不确定性、计算成本等附录表；
7. 所有未完成单元均显示红色`pending`，不得当作实验结果。

## 目录职责

```text
main.tex                                  模板、宏、draft skeleton开关和模块装配
sections/                                 正文与附录
tables/protocol_information_boundary.tex  主表1：协议和信息边界
tables/local_first_multiseed.tex           主表2：M2 local-first
tables/map2_method_comparison.tex          主表3：M1方法比较与O1 JTT槽位
tables/cross_setting_dataset.tex           主表4：map/base/dataset复现
tables/hfl_context_baselines.tex           主表5：S2 HFL背景基线
tables/oracle_granularity.tex              已完成Oracle边界表
tables/appendix/                           S1、PEW、per-client、seed和成本表
figures/                                   当前概念图；结果图等待M1/M2
references.bib                             BibTeX数据库
EXPERIMENT_TABLE_MATRIX_ZH.md              表格—实验—科学结论映射
```

## Draft skeleton开关

`main.tex`当前设置：

```tex
\draftskeletontrue
```

用于显示所有待证据表。准备纯证据版本时改为：

```tex
\draftskeletonfalse
```

这会隐藏完全pending的表，但不会隐藏已经含正式数字的部分完成表。

## 科学边界

- `pending`不是零、不是缺失值估计，也不是预期结果；
- smoke/benchmark数字不得填入论文表；
- M1已提供training-seed稳定性；M2完成前仍不得把local-first的seed-0 source bootstrap解释为训练稳定性；
- S2-v2是统一40轮protocol-matched context，不是所有原论文official-recipe leaderboard；FedTGP只称core adapter；
- M3只能支持第二个受控图像任务，不能支持真实医院或汽车部署；
- JTT只有完成O1 Formal后才能进入经验胜负结论。

## 编辑规则

1. 结果回来后只修改对应`tables/*.tex`，同时更新正文结论；
2. 不在正文复制第二份实验数字；
3. 网页端与仓库不得同时编辑；
4. 网页端返回完整Source ZIP后，由仓库diff核验数字、引用和边界；
5. V0.4继续保留用于版本比较，不回写。
