# 图3—5修订候选：数据、视觉与版本审查

日期：2026-10-30。状态：候选图已生成并目视审查，等待作者确认。
本次没有替换V0.10/V0.11源码或插图，没有运行训练，也没有改变实验结论。

## 当前候选与改动

| 图 | 新候选stem（均在figures/） | 对应保留旧版 | 本次修订 |
|---|---|---|---|
| 3 | `fig_local_first_multiseed_v2` | v1 | Local/HFL上下分隔标记、灰色配对线代替箭头、精简标题、均值/SD移入随图记录 |
| 4 | `fig_objective_tradeoff_v3` | v1/v2 | 空心种子点、实心均值、保留双向SD误差棒；基线按形状区分，BER用单一蓝色；取消强调环和图内大标题 |
| 5 | `fig_dose_response_v4` | v1/v2/v3 | 缩短面板标题、统一蓝灰配色、细化折线与柱形；保留三剂量、一个BER点及全部六项效用差分 |

三张图均提供PDF/SVG/300-DPI PNG/CSV/JSON/同stem Python源码快照。
新图宽度7.06英寸，高度依次2.55/3.05/2.85英寸；字体采用TeX Latin Modern，
与图1/2的TikZ候选保持一致。图内没有渐变、阴影、装饰卡片、绩效徽章或海报式大标题。
这是一组可检查、可复现的科研数据图；最终视觉偏好仍由作者确认。

## 数据与解释边界

### 图3：配对Local/HFL与通信差分

唯一科学输入：`../cle_v2_local_first_multiseed_20261026/PAPER_TABLE.csv`。
图中的9个横坐标与冻结表逐项一致：

| training seed | Local effect | HFL effect | HFL−Local |
|---|---:|---:|---:|
| 0 | 0.107960 | 0.119889 | +0.011929 |
| 1 | 0.1272867853 | 0.1237172825 | −0.0035695028 |
| 2 | 0.1312102392 | 0.1246375956 | −0.0065726436 |

均值±样本SD仍为Local `0.122152±0.012446`、HFL `0.122748±0.002518`、
Comm `0.000596±0.009929`；精确数值保存在JSON，不用大摘要条重复结果表。
左横轴仍为[0,0.16]、右横轴仍为[−0.0205,0.023]，右面板明确标注放大尺度。
同一seed内标记的上下间隔只是分类布局；横坐标没有jitter。无方向箭头，避免暗示
“Local先发生、HFL后发生”的时间轨迹。该图是匹配factorial对比，不是中介分解或通信等效检验。
12轮×16 local batches、batch64及seed2原始内部verdict均保留，不修改历史判定。

### 图4：四目标×三training seeds

原三个Formal JSON由维护生成器载入，并与冻结的
`../cle_hfl_map2_multiseed_20260922/MULTISEED_SUMMARY.json`及正式报告交叉校验。
导出CSV的12行与旧v2完全相同。全部种子点、四组均值、四组横纵样本SD均保留；
SD采用ddof=1，不称联合置信区域或source-bootstrap CI。
横轴仍为[17.2,29.4]%、纵轴仍为[0,0.29]；未截断低效用结果或移动数据点。
蓝色仅帮助找到BER，全部方法名称使用同等字号的黑色，不用大环和粗体预设胜负。
协议仍为held-out map2、同一AsymHFL、40轮×16 local batches、固定partition。

### 图5：剂量响应与效用代价

唯一数值输入：`../cle_v2_moderate_axis2_dose_formal_seed0_20261025/PAPER_TABLE.csv`。
导出CSV与旧v3完全相同。ERM的DSA为`−0.0007302367/0.0396793816/0.1845142617`；
BER只在gamma=.9有观测，DSA为`0.0900242753`。差分`0.0944899864`、相对下降约51.21%。
没有补画BER剂量曲线，也没有虚构误差带或多seed不确定性。

六项BER−ERM准确率变化全部保留（pp）：Clean −4.350；Mixed −1.7925；
Operator grid +0.966667；Corrupted +0.765；Seen +0.920455；Unseen +0.3375。
原坐标范围全部保留。负值与正值同色、同标准展示，未隐藏clean/mixed下降。
协议仍为主评价、seed0、alpha1/rho.5、40轮×64 local batches；与图4的预算不同。

## 源码来源与复现

维护脚本：

```text
figures/generate_local_first_multiseed_refined.py
figures/generate_objective_tradeoff_refined.py
figures/generate_dose_response_refined.py
figures/refined_data_style.py
```

三者由原实际生成器复制后局部修改，保持dumbbell/scatter/line的既定结构。
图4/5沿用只读的`figure_common.py`导出；新style不会改变旧生成器的样式。
图3补存v1源码快照，旧图文件未变。每次生成新后缀，禁止覆盖已存在版本。
快照源码导入同目录辅助模块；独立复现时须同时保留上述style与figure_common.py。

环境：`D:\ResearchTools\envs\paper-figures\Scripts\python.exe`，
Python3.12.13、NumPy2.5.4、Matplotlib3.11.2、`D:\texlive\2026\bin\windows`。
源码比较记录保存在`.vivid/*-refined-source-review.json`，分别有32/25/33条advisory。
它们反映字体、图层和画布等适配，不等同于数据错误或视觉通过。

从仓库根目录复现到下一新后缀（当前v2/v3/v4保持不变）：

```powershell
& 'D:\ResearchTools\envs\paper-figures\Scripts\python.exe' deliverables/cle_hfl_figures_v1_20261028/figures/generate_local_first_multiseed_refined.py --revision v3
& 'D:\ResearchTools\envs\paper-figures\Scripts\python.exe' deliverables/cle_hfl_figures_v1_20261028/figures/generate_objective_tradeoff_refined.py --revision v4
& 'D:\ResearchTools\envs\paper-figures\Scripts\python.exe' deliverables/cle_hfl_figures_v1_20261028/figures/generate_dose_response_refined.py --revision v5
```

## 已完成检查

- `verify_refined_local_first.py`：9个坐标、左右尺度、均值/SD、CSV、输出哈希、v1原图完整性通过。
- `verify_refined_tradeoff_dose.py`：12个种子点、均值/SD、四个剂量/方法点、六项效用差分、
  全部坐标范围、PDF单页矢量文字、来源/产物哈希，以及图4/5五个旧版本完整性通过。
- 三张原尺寸PNG已实际打开；图3及图4/5的7.06英寸通栏PDF校样也均已渲染并实际查看，
  未发现标注重叠、文字裁切或缺失。校样独立保存在`local_runs/tikz_review_20261030/`，
  没有编译/改写论文。三页校样编译无Overfull、Undefined或Error。

## 建议接入时使用的图注要点

3：matched training seeds；gamma.9−0；右轴放大；上下间隔为布局；保留12轮预算说明。

4：同一AsymHFL、held-out map2；空心为seed、实心为均值；横纵棒为各维样本SD；
低DSA/高operator-grid accuracy更好；不能将SD写成CI。

5：主评价seed0；只连接实测ERM点；BER仅gamma.9；右侧为BER−ERM的六项准确率差分；
保留负效用变化，不宣称无损兼容。

下一步由作者并排查看五张独立候选，确认后再决定接入新的论文快照。本轮停在候选交付。
