# CLE-HFL 网页端去AI味写作与科研绘图改造指南

Updated: 2026-09-22

> V0.5补充：网页端不得把红色`pending`单元格补成估计数字，也不得删除完全pending的表格骨架。
> 实际投稿前可在`main.tex`设置`\draftskeletonfalse`隐藏尚无Formal证据的整张表。

## 1. 本轮目标与禁止事项

目标不是“AI降重”，而是形成作者自己的论证节奏、判断和语言。网页端可以提出候选改写，但不得：

- 新增实验结果、引用、定理、数据集或实现细节；
- 删除或弱化`[EVIDENCE NEEDED]`；
- 把seed-0、固定partition或受控CLE结果扩大为普遍结论；
- 把PEW写成taxonomy-free，或把Oracle operator写成部署方法；
- 把FedDF-fidelity、CVaR-DRO或FCCL adapter描述成官方逐行复现；
- 宣称首次研究联邦spurious correlation、首次环境推断或首次反事实shortcut评价；
- 为了语言流畅而改变数字、公式、citation key、表格或结论方向。

## 2. 当前“AI味”最可能来自哪里

逐节检查以下模式，而不是全篇一次性重写：

1. 每段都以背景句开头，再用`However/Therefore/Thus`机械转折。
2. 反复使用`not merely ... but ...`、`rather than ...`、`This is important because ...`。
3. 连续多句采用三项或四项并列，句长和语气过度整齐。
4. 同一个边界在Abstract、Introduction、Experiments和Discussion中重复完整解释。
5. 使用`comprehensive/novel/robust/significant`等评价词，却没有紧邻具体证据。
6. Related Work像逐篇摘要，没有说清这些工作为何不能回答本文的estimand。
7. 段落结尾总是总结上一段，而不是推进下一个科学问题。
8. 过量加粗、冒号、破折号和“evidence chain/complete framework”等包装语言。

## 3. 推荐的人工化改写原则

- 每段先写一句作者真正想让审稿人记住的判断，再保留支撑它所需的最少背景。
- 用具体对象替代泛称，例如写`same-source cross-operator response`，少写`a comprehensive diagnostic`。
- 数字只在它改变判断时出现；不要把结果表逐行复述成正文。
- 把大部分限制集中到Discussion；正文只保留影响当前解释的局部限定。
- 允许句式不完全对称。四条贡献可以平行，但普通段落不必像模板。
- Related Work按“它测什么—本文测什么—二者为何不等价”组织，而非按作者名单组织。
- 对负结果和折中直接陈述，不使用宣传式缓冲句。
- 优先删句，其次合并句，最后才替换同义词。

## 4. 分节改写职责

```text
Abstract       只保留问题、DSA、local-first、PEW+BER和最重要边界；避免贡献清单语气。
Introduction   形成一个连续问题链：corruption nuisance -> directional cue -> diagnosis -> formation -> mitigation。
Related Work   强调estimand、信息边界和优化对象的差异，不逐篇复述摘要。
Method         以定义和推导为主，减少“we emphasize/we note/we clarify”元话语。
Experiments    先写研究问题和protocol，再写决定性contrast；不把每列数字改写成句子。
Discussion     集中处理taxonomy、utility、seed/dataset和现实外部有效性。
Conclusion     不重复四条贡献，只回答论文最终知道了什么、仍不知道什么。
```

## 5. 网页端逐节改写提示词

将V0.4源码ZIP、本指南和论文证据交接文档上传后使用：

```text
你是这篇机器学习会议论文的严格英文合作者。请先阅读全部材料，但本轮只修改我指定的一个
sections/*.tex文件。目标是减少模板化AI文风，形成克制、具体、由科学问题推动的作者语言。

硬约束：
1. 不得新增、删除或修改任何实验数字、公式、citation key、表格、定理条件和
   [EVIDENCE NEEDED]标记；
2. 不得扩大seed、partition、dataset、taxonomy或现实部署范围；
3. PEW始终是coarse family-level、taxonomy-assisted；operator信息仅用于CLE构造、DSA评价和
   Oracle审计；
4. FedDF-fidelity utility gate失败、CVaR存在逐客户端例外、JTT只有screen证据等边界必须保留；
5. 不要以规避AI检测器为目标，不要做同义词替换式降重；
6. 优先删除重复元话语，打破机械平行句式，让每段只有一个中心判断；
7. 输出三部分：A. 主要文风问题；B. 修改后的完整LaTeX文件；C. 逐项说明哪些句子被删改以及为何
   没有改变科学含义。

本轮只修改：<填写一个文件，例如 sections/introduction.tex>。
不要修改其他文件。若你认为事实需要变化，只在C中提出问题，不要擅自改正文。
```

网页端每次只处理一个section。推荐顺序：Introduction、Related Work、Abstract、Experiments、
Discussion、Conclusion，最后才检查Method。每一轮返回后必须由仓库版本做diff和事实审计。

## 6. 为什么当前图片容易有“AI味”

生成式论文图常见问题不是清晰度，而是缺少科学信息层级：渐变色、圆角卡片、发光箭头、图标拼贴、
过多短句和“problem/solution/result”流程框会让图像像产品宣传页。科研图应让读者在数秒内读出变量、
干预、比较单位和结论，装饰只服务于区分对象。

因此不要让网页端“自由生成漂亮论文图”。先给它明确的数据对象、布局草图、颜色语义和禁止元素，
最终以SVG/PDF矢量方式绘制，并从论文真实表格或结果文件取数。

## 7. 建议的四张核心图

### Figure 1：CLE-HFL问题设定

左到右三栏：不同客户端的class--operator binding、异构本地模型、公共预测通信。只用两个具体类别
和两个operator示意，突出不同客户端binding不同。图底部单独标注train-time可见信息与sealed
evaluation-only信息。不要塞入全部证据链。

### Figure 2：Paired DSA estimand

固定一张source image，在同一水平轴展示多个operator view；上方画预测概率变化，下方突出目标
operator对应的bound-class mass与其他operator平均值之差。该图应让读者不读公式也理解DSA，
不使用大段说明文字。

### Figure 3：Local-first formation

使用带置信信息的点/区间或紧凑分解图展示`delta_Local`、`delta_HFL`和`delta_Comm`，避免把90.05%
画成严格因果饼图。seeds 1/2完成前只能使用seed-0并明确标注。

### Figure 4：Shortcut--utility trade-off

横轴DSA（越低越好），纵轴operator-grid accuracy或Avg；用一致符号表示ERM、CVaR、PEW+GroupDRO
和PEW+BER。多seed完成后用点和区间；完成前不得伪造误差条。该图比装饰性机制总览更能传递论文
核心结果。

## 8. 视觉规范

- 白底、无渐变、无阴影、无3D、无卡通机器人或大面积圆角卡片。
- 全文最多一套主色加一套对照色；同一方法在所有图中保持同色。
- 建议：ERM灰、CVaR蓝、PEW+GroupDRO橙、PEW+BER深红；色盲友好并配合点形/线型。
- 字体与最终会议正文一致；缩到单栏宽度后仍能读取。
- 箭头只表示明确的数据流或因果干预，不表示模糊的“促进”。
- caption写清图回答的问题、统计单位和边界，不写宣传语。
- 优先用Matplotlib/TikZ/SVG代码生成可复现图，不采用不可编辑的AI位图作为最终论文图。

## 9. 图稿网页端提示词

```text
请不要直接生成图片。先为CLE-HFL论文的<Figure编号>输出一份科研图设计规格：
1. 该图唯一要回答的科学问题；
2. 每个panel的变量、统计单位和数据来源；
3. 布局线框；
4. 颜色、点形和线型映射；
5. caption草稿；
6. 哪些元素会造成AI宣传图风格并必须禁止；
7. 实现为Matplotlib/TikZ/SVG所需的精确输入表。

不得虚构实验数据、误差条、置信区间或模型结构。不要使用渐变、阴影、3D、发光、卡通图标、
装饰性大箭头或大段流程框文字。待设计规格经作者确认后，再生成可复现绘图代码。
```

## 10. 网页端上传材料

```text
deliverables/CLE_HFL_LATEX_V0_4_MODULAR_OVERLEAF_20260921.zip
deliverables/cle_hfl_latex_v0_4_20260921/WEB_REVISION_AND_FIGURE_GUIDE_ZH.md
deliverables/cle_hfl_full_paper_web_handoff_20260911/CLE_HFL_FULL_PAPER_WEB_HANDOFF_ZH.md
```

若只做单节语言修改，优先上传对应`sections/*.tex`与本指南，避免网页端一次性重写全篇。
