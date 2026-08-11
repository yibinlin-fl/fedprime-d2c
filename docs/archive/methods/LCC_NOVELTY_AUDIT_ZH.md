# LCC 数学定义与新颖性审计

日期：2026-08-11
状态：`THEORY NO-GO`，未实现、未接 runner、未启动实验

## 1. 研究目标

LCC（Latent Correction Conflict）原本试图用同类别样本的优化方向代替 PEW 的
`clean/noise/blur/weather/digital/unknown` 环境标签，再用兼容更新保留 hard BER 对弱
`class x environment` 单元的收益。

本审计只回答两个问题：这个对象是否数学成立，以及它是否足够新颖、值得进入实现。

## 2. 冻结数学定义

对本地模型的倒数第二层表示 `h_i`、预测分布 `p_i` 和真实类别 `c`，最后分类层的逐样本梯度为

\[
G_i=(p_i-e_c)h_i^\top,
\qquad
u_i=\frac{\operatorname{vec}(G_i)}{\|G_i\|_2+\epsilon}.
\]

LCC-v1 在每个类别内部对 `u_i` 做密度聚类，得到不带语义名称的潜在修正集合
`Q_c={q}`。每个集合的平均分类梯度为

\[
g_{c,q}=\mathbb E[\nabla_\theta\ell_i\mid y_i=c,q_i=q].
\]

候选更新不是简单优化最大组风险，而是求最小范数凸组合

\[
\alpha_c^*=\arg\min_{\alpha\in\Delta^{|Q_c|}}
\left\|\sum_q\alpha_q g_{c,q}\right\|_2^2,
\qquad
d_c=-\sum_q\alpha_{c,q}^*g_{c,q}.
\]

若 `g_{c,q}^T d_c < 0` 对所有有效 `q` 成立，则一阶近似下同一步会同时降低所有潜在集合风险；
若不存在严格公共下降方向，则最小范数组合给出 Pareto-stationary 证书，而不是强行牺牲一个集合。

LCC-v2 曾考虑取消离散聚类：在类别内以 `u_i` 构造 KNN 图，令局部邻域风险为

\[
R_{c,i}^{\rm nbr}=\sum_j A_{ij}\ell_j,
\]

再对图邻域的梯度做同一公共下降优化。这能避免固定组数，却没有改变方法的核心信息来源。

## 3. 为什么它不需要五类环境标签

LCC 只读取 `(x,y)`、模型表示、预测和参数梯度；不读取 operator ID、family、severity、PEW
预测、公共配对干预或 final-test 标签。潜在集合是优化几何对象，不被解释成 noise/blur 等具体
环境。因此“无五类标签”在形式上成立。

若同一隐藏环境内的样本梯度集中、不同环境的修正方向可分，且每个环境具有足够支持，梯度集合可
近似恢复环境风险方向。公共下降随后有机会像 BER 一样避免多数环境的梯度掩盖少数环境。但这是
可识别性假设，不是由标签缺失自动保证的事实。

## 4. 与项目内方法的逐项区别

| 方法 | 数学对象 | LCC 的区别 |
|---|---|---|
| hard PEW + BER | 显式 `class x predicted-family` 组风险 | LCC 不预测 family，按梯度几何形成潜在修正集合 |
| PIE/MPIE | 公共配对干预学习的 latent/severity 表征 | LCC 不使用公共配对干预或环境 encoder |
| C3R | AugMix 前后 CE 的反事实退化遗憾 | LCC 使用逐样本参数梯度，不使用退化差值 |
| GroupDRO | 已知组上的最大风险 | LCC 先从梯度推断伪组，再求多目标公共下降 |
| CVaR | 按损失分位数选择尾部样本 | LCC 按方向而非损失大小组织样本 |
| CCAD/IRD | 公共多视图上的一致蒸馏/残差蒸馏 | LCC 是纯本地私有样本优化，不构造公共 teacher |
| CRSR | 类条件预测残差协方差的最大特征值 | LCC 使用逐样本参数梯度集合和多目标优化 |

这些区别说明 LCC 不是项目内冻结方法的同名复活，但不能证明它相对外部文献具有新颖性。

## 5. 外部文献碰撞

### 5.1 LCC-v1 的直接碰撞

[GRASP](https://openreview.net/forum?id=0Xemsc4Nlp) 已经在无组标签场景中使用类别条件的逐样本
参数梯度、在梯度空间用 DBSCAN 推断潜在组，并以这些组进行鲁棒训练。LCC-v1 的“类别内梯度
聚类得到潜在组”不是新的核心对象。

[MGDA](https://proceedings.neurips.cc/paper_files/paper/2018/hash/432aca3a1e345e339f35a30c8f65edce-Abstract.html)
及 [CAGrad](https://proceedings.neurips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html)
已经建立最小范数凸组合与共同下降/Pareto 冲突处理。把它接在梯度伪组之后属于合理组合，不能单独
支撑论文主创新。

### 5.2 LCC-v2 的更近碰撞

[Graph of Gradients (GoG), KDD 2025](https://arxiv.org/abs/2412.03706) 直接使用最后一层梯度
构造 KNN 梯度图，把图当作无 demographic 标签的软分组结构，并进行对抗样本加权。它覆盖了
LCC-v2 想用“梯度图邻域替代硬聚类”解决的主要新颖性空间。

邻近工作还包括无组标签的特征聚类再 GroupDRO（
[GEORGE](https://proceedings.neurips.cc/paper_files/paper/2020/hash/e0688d13958a19e087e123148555e4b4-Abstract.html)）
以及环境推断（[EIIL](https://proceedings.mlr.press/v139/creager21a.html)）。因此不能把“没有人工
环境名称”本身作为 LCC 的新颖性。

## 6. 判定

```text
数学合理性：     中等；依赖梯度集合与真实弱环境对齐
不需要五类标签： 是
相对项目旧方法： 可清楚区分
外部新颖性：     不通过
实现成本：       中高；逐样本梯度、图/聚类和多目标求解
实验归因：       差；伪组发现与公共下降两个机制耦合
论文价值：       不足以作为主方法
```

正式判定：`THEORY NO-GO`。LCC 不进入最小实现，不运行本地 3050 或 OpenI，不允许通过更换
聚类器、KNN 邻居数、梯度层、损失权重或 MGDA 求解器复活。该结论是新颖性否决，不是实验失败。
