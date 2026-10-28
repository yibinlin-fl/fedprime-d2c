# 图3：Local-first 多训练种子归因图

日期：2026-10-28。候选：`figures/fig_local_first_multiseed_v1.*`。
状态：独立图件生成与目视检查通过；尚未接入V0.10，未做论文页内排版验收。

## 选择与数据

研究问题：HFL与Local的CLE effect是否接近，通信差分是否跨training seed稳定为正？
已实际查看`advanced.dumbbell`与`advanced.paired_dot`预览，选择前者：每行一个matched
training seed，两个端点与连线对应Local/HFL effect。右面板单独展示精确差分，单位相同、
刻度放大且已明确注明。保留模板端点、白边、连线和箭头，删除无数据依据的显著性星号；
沿用本项目青色/锈红/紫色及方形/圆形/菱形，兼顾灰度区分。

唯一输入是`deliverables/cle_v2_local_first_multiseed_20261026/PAPER_TABLE.csv`。
源CSV哈希、实际绘图行、产物哈希及环境版本分别保存于同名JSON/CSV。

| Seed | Delta Local | Delta HFL | Delta Comm |
|---:|---:|---:|---:|
| 0 | 0.107960 | 0.119889 | +0.011929 |
| 1 | 0.127287 | 0.123717 | -0.003570 |
| 2 | 0.131210 | 0.124638 | -0.006573 |

Delta为DSA(gamma=.9)减DSA(gamma=0)的冻结contrast。脚本逐seed检查
`Delta_Comm = Delta_HFL - Delta_Local`，并检查均值与样本SD和原CSV一致。
点标签4位小数、底部摘要5位，完整精度保存在CSV。

底部为3个training seeds的mean ± sample SD（ddof=1），不是source-bootstrap区间，也不是
均值95%CI。图中不添加误差线、p值或显著性星号。

## 解释边界

Local effect在三个seed中均较大且接近HFL effect；通信差分跨seed改变符号，均值约0.00060。
这支持当前matched factorial下的local-first解释；不构成通信等价性检验，不把均值接近零
写成通信没有影响，也不作样本级中介比例解释。

按论文DSA主终点绘出全部三个seed。内部审计仍保留seed 2因辅助20%学习下限失败而得到的
`INCONCLUSIVE_UNDERTRAINED`原始完整verdict，已在同名CSV/JSON中保留，不改写历史状态。

## 环境、复现与版本

实际执行环境：`D:\ResearchTools\envs\paper-figures\Scripts\python.exe`；
Python 3.12.13、Matplotlib 3.11.2、NumPy 2.5.4。
生成脚本独立运行，不依赖未提交的`_utils`。

```powershell
& 'D:\ResearchTools\envs\paper-figures\Scripts\python.exe' `
  'C:\Users\asus\Desktop\FedPRIME-D2C\deliverables\cle_hfl_figures_v1_20261028\figures\generate_local_first_multiseed.py' `
  --revision v2
```

已有版本拒绝覆盖；后续使用新的vN，旧版保留。每版保存PDF/SVG/300dpi PNG和CSV/JSON。
Python源码及模板来源底稿由Git留存。

## 检查结果

原生7.15 × 3.38英寸，预计6.7英寸双栏引用，对应0.94倍缩放；8–10pt字号缩放后约7.5–9.4pt。
PNG为2145 × 1014，SVG为514.8 × 243.36pt且含38个text元素，PDF文件头有效。
已实际打开整图：全部seed、数值、刻度、图例、底部SD摘要完整可读，无遮挡/裁切。
右侧零线和放大刻度说明清楚。第一轮即通过，无需视觉修复。

建议英文图注：

> Local-first formation across three matched training seeds. (a) The CLE effect
> is the DSA contrast between gamma=0.9 and gamma=0 under HFL or Local training.
> Paired endpoints correspond to the same training seed. (b) The communication
> difference is Delta_HFL minus Delta_Local, shown on a magnified zero-centred
> axis. Bottom summaries report the mean and sample SD across training seeds.
> Results use the fixed 12-round, 16-local-batch mechanism protocol.

V0.10、其旧图与图1/2均未被本轮绘图修改。
