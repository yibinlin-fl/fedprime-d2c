# 图4与图5：冻结数据、绘制与审查

日期：2026-10-28。状态：独立候选图已生成并通过实际目视检查；未接入V0.10。
当前选择：图4 `fig_objective_tradeoff_v2`；图5 `fig_dose_response_v3`。
v1及中间版本均保留；每版有PDF、SVG、300dpi PNG、绘图CSV、JSON来源清单及生成时Python快照。

## 图4：Axis-I shortcut–utility

从三个已提取的Formal `analysis/RESULT_SUMMARY.json`读取全部4臂×3训练seed的完整精度结果：

```text
outputs/openi_downloads/cle_v2_spurious_final_map2_formal/extracted/
outputs/openi_downloads/m1_map2_multiseed/extracted_seed1/
outputs/openi_downloads/m1_map2_multiseed/extracted_seed2/
```

逐seed核对正式状态、40轮、本地轨迹配对和已有seed标识。随后和
`deliverables/cle_hfl_map2_multiseed_20260922/MULTISEED_SUMMARY.json`的逐seed差分、
`RESULT_SUMMARY_ZH.md`的均值/样本SD交叉核对；计算与冻结结果一致。

横轴为operator-grid accuracy百分数，纵轴为DSA概率尺度。12个小标记表示独立training seed，
4个大标记表示各方法均值；横/竖误差棒分别为3个training seed的±1样本SD（ddof=1）。
误差棒不是source-bootstrap CI，也不是二维联合置信区域。不同形状与低饱和色区分方法，
BER以锈红外环强调；方法名直接标在点簇旁边。

| 方法 | DSA mean ± SD | Grid accuracy mean ± SD |
|---|---:|---:|
| ERM | 0.250530 ± 0.008670 | 20.4950 ± 0.3718 |
| CVaR-DRO | 0.086654 ± 0.005514 | 19.2206 ± 1.1055 |
| PEW+GroupDRO | 0.205417 ± 0.008045 | 21.4517 ± 0.5783 |
| PEW+BER | 0.071470 ± 0.011288 | 25.1772 ± 1.2522 |

解释：在当前固定partition/map2和相同AsymHFL通信下，BER呈现更好的pooled DSA–utility组合。
该图不声称各客户端均优于CVaR，也不跨数据集或协议外推。误差棒重叠与否不是paired方法差分
的显著性检验；正式差分仍以冻结报告为准。

模板：查看`academic.error_analysis`实际预览后，保留其散点、白边点、直接标注和重点外环。
由于没有参数量或效率测量，移除对应气泡大小/效率线；四个离散方法不支持连续可达前沿，
因此不画Pareto曲线。补入真实seed点和样本SD，必要适配已在源码说明。

v1实际目视发现BER标签靠近右侧seed点；v2仅移动该标签，全部点、误差棒与统计值不变。
v2已实际查看整图，方法标签、点簇、刻度和图例无遮挡。

## 图5：剂量响应与效用边界

唯一数值来源：
`deliverables/cle_v2_moderate_axis2_dose_formal_seed0_20261025/PAPER_TABLE.csv`。
历史路径保留moderate工程名称，图中统一使用main evaluation protocol。
该图来自training seed 0的40轮×64 local batches协议，不与图4机制协议混成排行榜。

左侧展示三个实测ERM剂量点：

```text
gamma 0.0: DSA -0.0007302367
gamma 0.5: DSA  0.0396793816
gamma 0.9: DSA  0.1845142617
gamma 0.9, PEW+BER: DSA 0.0900242753
```

点间直线仅引导读取实测剂量顺序，不是拟合的连续响应规律。BER仅在gamma=.9有结果，
因此只画一个方形点，不补造其他gamma下的BER轨迹或不确定性带。
匹配干预使DSA降低约51.21%，以同一gamma下的竖括号标示。

右侧展示全部六种正式checkpoint效用端点，计算方式均为candidate minus matched ERM，
单位为百分点（pp）；包括负向clean/mixed结果：

| 端点 | candidate − control（pp，舍入） |
|---|---:|
| Clean | -4.3500 |
| Mixed | -1.7925 |
| Operator grid | +0.9667 |
| Balanced corrupted | +0.7650 |
| Seen | +0.9205 |
| Unseen | +0.3375 |

该结果支持seed-0受控剂量响应与显著CLE抑制，效用变化并不一致。
不使用legacy logger的last-10指标替换这些统一checkpoint端点，也不声称无损效用。

模板：查看`basic.line`实际预览，保留实测点、折线、点形与白色点边。没有逐剂量多seed区间，
因此去掉模板随机误差带和装饰性极值星号。右侧新增零中心横向条形图以完整展示效用代价。
v1存在两处点标签靠近折线；v2移开；v3进一步缩短零剂量标签到点的距离。v3已实际查看，
数值、折线、图例、括号、全部六种效用端点及底部协议说明均完整可读。

## 环境、留存与复现

环境：`D:\ResearchTools\envs\paper-figures\Scripts\python.exe`。
Python 3.12.13、Matplotlib 3.11.2、NumPy 2.5.4；运行依赖仅标准库及后两者。
`figure_common.py`提供统一风格、版本防覆盖、CSV/JSON导出和源码快照；其hash记录于每版JSON。

每个`.py`同名快照保留该版生成时源码；通用生成器持续维护，Git留存历史。
如原始下载目录不在新机器，可以把已留存的同名CSV作为`--data-csv`传入；
图4仍会对照仓库里的冻结多seed汇总验证数值。

未来新修订示例（先进入本工作区）：

```powershell
& 'D:\ResearchTools\envs\paper-figures\Scripts\python.exe' `
  'figures\generate_objective_tradeoff.py' --revision v3 `
  --data-csv 'figures\fig_objective_tradeoff_v2.csv'
& 'D:\ResearchTools\envs\paper-figures\Scripts\python.exe' `
  'figures\generate_dose_response.py' --revision v4 `
  --data-csv 'figures\fig_dose_response_v3.csv'
```

图4原生7.15×3.75英寸、2145×1125 PNG；图5原生7.15×3.60英寸、2145×1080 PNG。
预计6.7英寸论文双栏引用，对应约0.94倍缩放，8–10pt正文图字号缩放后约7.5–9.4pt。
SVG可解析，保留文本；产物hash与来源清单一致。实测重复运行同一revision会拒绝覆盖，
原产物hash保持不变。两张图均尚未接受论文页内布局检验。
工作区局部`.gitattributes`对图4/5产物及源码保留字节，避免Windows自动换行转换破坏SHA256。

## 建议英文图注

Figure 4:

> Shortcut–utility comparison under the fixed held-out map2 mechanism protocol.
> Small markers denote individual training seeds; large markers show means
> across three seeds. Horizontal and vertical bars show one sample SD for
> operator-grid accuracy and DSA, respectively. All objectives use the same
> AsymHFL communication and matched 40-round, 16-local-batch budgets.

Figure 5:

> Binding-strength response and task-utility changes under the main evaluation
> protocol, with training seed 0. (a) ERM DSA increases across the three measured
> binding strengths. The matched PEW+BER intervention at gamma=0.9 reduces DSA
> by 51.2%. Lines connect measured ERM settings; no BER response curve is
> inferred. (b) Accuracy changes relative to matched ERM include improvements
> on corrupted evaluations and declines on clean and mixed evaluations.
