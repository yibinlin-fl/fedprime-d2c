# FedPRIME-CTPG：原型关系路线排重与正式框架

**版本：** 研究设计 v0.1  
**日期：** 2026-06-25  
**状态：** 未编码、未验证；用于决定是否启动下一条研究路线。  
**拟议名称：** FedPRIME-CTPG  
**全称：** PRIME-guided Corruption Transport Prototype Geometry for Heterogeneous Federated Learning。

---

## 0. 结论摘要

原始的 ProtoGraph 想法是：

> 不传 raw prototype，传客户端内类别 prototype 的静态关系图。

经过专门排重后，**这个原始版本不应直接实现**。原因是：

- FedTGP 已在模型与数据异构下研究 trainable global prototypes、prototype margin shrink 和 prototype contrastive learning；
- MP-FedCL 已研究 class multi-prototype 与 federated contrastive learning；
- FedSC 已提出 relational prototypes 与 prototype-wise semantic collaboration；
- FedDAP 已研究多 prototype、相似度加权聚合与 prototype contrastive alignment；
- FedPLC 已研究 label-wise community adaptation 与 prototype-anchored representation learning。

因此，“原型关系”“多原型”“可靠聚合”“对比学习”任意一个单独出现，都不能作为新的核心贡献。

仍值得进一步研究的候选创新是：

> **不是交流静态 prototype 或静态 prototype graph，而是交流 PRIME 已知扰动下，类别原型几何从基准视图到频域、颜色域、空间域视图的变化轨迹，即 corruption transport。**

拟议方法为：

\[
\boxed{
\text{FedPRIME-CTPG}
=
\text{PRIME}
+
\text{MPTL}
+
\text{CTGC}
}
\]

- **MPTL**：Multi-view Prototype Transport Learning，多视图原型传输学习；
- **CTGC**：Corruption Transport Geometry Consensus，损坏传输几何共识通信。

该方向仍是“改进 FedProto”的路线：使用类别原型作为通信语义单位；但它不聚合 raw prototype，也不要求异构模型的特征坐标对齐。

---

## 1. 研究问题

### 1.1 当前实验场景

项目同时面对：

1. **模型异构：** ResNet10、ResNet12、ShuffleNet、MobileNetV2 的参数和 feature dimension 不同；
2. **数据异构：** Dirichlet label skew，主设置 \(\alpha=0.5\)，部分客户端拥有很少甚至零个类别样本；
3. **数据损坏：** 本地训练和独立测试使用 RAHFL-style random corruption；
4. **鲁棒增强：** PRIME 在频域、颜色域和空间域生成增强视图。

当前 RAHFL 强基线为 AugMix + DCL + AsymHFL，在固定 seed-0 设置下达到 avg_acc 56.41%、worst_acc 44.72%。旧 D2C 最高约 52.31%，Oracle D2C 也没有改善，因此不继续以 public-logit prior 去偏作为主路线。

### 1.2 FedProto 在本场景中的问题

标准 FedProto 对本地类 prototype 做均值：

\[
p_{k,c}=\frac{1}{N_{k,c}}\sum_{i:y_i=c}h_{k,i},
\]

并在服务器直接聚合：

\[
\bar p_c=\sum_k \omega_{k,c}p_{k,c}.
\]

它在当前任务至少有四个未充分处理的假设：

1. 不同模型的 \(h_{k,i}\) 处在可直接比较的共同坐标系；
2. 每类一个均值 prototype 足以表达受损 Non-IID 数据；
3. 各客户端上传的 prototype 同等可信；
4. 静态 prototype 足以描述在 corruption 前后的语义稳定性。

前两个假设在模型异构和 label skew 下脆弱；后两个假设在 corruption 与强增强下尤其脆弱。

---

## 2. 专门文献排重

### 2.1 核心近邻工作

| 工作 | 已有机制 | 对本路线的影响 | 结论 |
|---|---|---|---|
| [FedProto, AAAI 2022](https://arxiv.org/abs/2105.00243) | class mean prototype 通信和原型距离正则 | 基础 baseline | 必须直接比较 |
| [FedTGP, AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/29617) | trainable global prototype 与 adaptive-margin contrastive learning，面向模型和数据异构 | 覆盖简单加权 prototype、prototype margin、prototype contrastive | 不可把 margin 或可训练 prototype 当创新 |
| [MP-FedCL, 2023](https://arxiv.org/abs/2304.01950) | 每类 k-means multi-prototype 和 federated contrastive learning | 覆盖多 prototype 处理 feature / label skew | 不可把 multi-prototype 当创新 |
| [FedSC, 2025](https://arxiv.org/abs/2506.21012) | relational prototype 与 prototype-wise semantic collaboration | 与静态 prototype relation 最接近 | 原始静态 ProtoGraph 高度重叠，应放弃 |
| FedDAP, 2026 preprint | domain-specific prototype、相似度加权融合、跨 domain prototype contrastive learning | 证明“单 prototype 语义稀释”已被讨论 | 不使用已知 domain 标签或 domain prototype bank |
| FedPLC, Sensors 2026 | prototype anchoring、label-wise client community | 证明 label-level 细粒度协作已被讨论 | 不复制动态 client clustering 或 classifier head 重组 |
| F2DC, 2026 preprint | domain feature decoupling 和 calibration | feature 分解与校正已有直接近邻 | 不采用 feature decoupling + calibration 的叙事 |

补充说明：

- FedDAP、F2DC 的实验主线是 domain shift，且使用共享模型或参数聚合；不能直接证明其适配模型架构异构；
- FedPLC 的主问题是时间 concept drift，不是 common corruption；
- FedTGP 是必须正视的强基线，因为它已经明确瞄准 heterogeneous prototype FL；
- FedSC 使“relational prototype”这个表述本身不再新颖。

### 2.2 对原始 ProtoGraph 的裁决

| 原始想法 | 排重结果 | 裁决 |
|---|---|---|
| 静态类 prototype relation graph | 被 FedSC 等 relational prototype 工作强烈覆盖 | 放弃作为创新 |
| 多 prototype bank | 被 MP-FedCL、FedDAP 等覆盖 | 不作为主创新 |
| 可靠加权 prototype 聚合 | 被 FedTGP、FedDAP、FedPAD 等邻近工作覆盖 | 不作为主创新 |
| class-wise client community | 被 FedPLC 类方法覆盖 | 不作为主创新 |
| 参数/feature decoupling 与 calibration | 与 F2DC 叙事过近，且不适合强模型异构 | 不采用 |

### 2.3 尚未被上述工作直接覆盖的候选空位

在本次阅读与定向检索的范围内，未发现下列组合的直接等价方案：

1. 使用 PRIME 的频域、颜色域、空间域原语构造可解释的多 view corruption channels；
2. 用每个 channel 前后**类别 prototype geometry 的变化量**，而不是静态 prototype 或静态 relation，作为通信语义；
3. 在不共享模型参数、raw feature、public image 或已知 domain label 的条件下，对该变化量做类别边级可靠聚合；
4. 让特征维度不同的异构模型通过 \(C\times C\) 的几何变化矩阵协作。

这不是“没有文献”的证明。投稿前仍必须继续检索 corruption-aware prototype FL、prototype transport、geometry distillation 与 domain-generalization FL；但它比原始 ProtoGraph 的空位清晰得多。

---

## 3. 新框架的核心假设

### 3.1 关键观察

同一类别的静态 prototype 并不能充分表征模型鲁棒性。

设 \(p_{k,c}^{(0)}\) 是客户端 \(k\) 在基准输入视图上类别 \(c\) 的 prototype，\(p_{k,c}^{(a)}\) 是施加 PRIME 原语 \(a\) 后的 prototype，其中：

\[
a\in\{\mathrm{freq},\mathrm{color},\mathrm{spatial}\}.
\]

一个模型即使在基准图像上分类正确，也可能在频域滤波、颜色扰动或空间形变后使类别间边界迅速塌缩。静态 \(p_{k,c}^{(0)}\) 看不到这种现象。

因此，应该学习：

\[
\Delta G_{k}^{(a)}
=
G_k^{(a)}-G_k^{(0)},
\]

而不是只学习 \(G_k^{(0)}\) 或 \(\bar p_c\)。

### 3.2 研究假设

> 在受损 Non-IID 异构联邦学习中，可靠客户端在 PRIME 已知扰动下具有更稳定且可泛化的类别几何传输模式。将这种传输模式按类别边可靠聚合，可以帮助本地模型避免 corruption-induced class collapse，并改善已观测 tail 类的边界。

这是一条可被实验否定的假设：若全框架不优于静态 FedProto、FedTGP 或 PRIME + LogitAvg，就不能保留该主张。

---

## 4. 模块一：PRIME Channel Views

对本地带标签样本 \((x_i,y_i)\)，保留一个基准视图：

\[
x_i^{(0)}=x_i.
\]

并复用 PRIME 的三种已存在原语生成受控视图：

\[
x_i^{(F)}=\operatorname{PRIME}_{freq}(x_i),\quad
x_i^{(C)}=\operatorname{PRIME}_{color}(x_i),\quad
x_i^{(S)}=\operatorname{PRIME}_{spatial}(x_i).
\]

这里不是重新发明或修改 PRIME。它只是把 PRIME 原有的频域、颜色域、空间域 primitive 单独暴露，供后续鲁棒几何估计使用。

客户端 \(k\) 的异构模型仅在自己内部得到归一化表示：

\[
h_{k,i}^{(a)}
=
\frac{f_k(x_i^{(a)})}{\|f_k(x_i^{(a)})\|_2}.
\]

不同客户端的 \(h_{k,i}^{(a)}\) 可以有不同维度。它们不会上传。

---

## 5. 模块二：MPTL 多视图原型传输学习

### 5.1 本地类别 prototype

为降低单 batch 过小带来的方差，客户端在本地对每个已观测类别维护 EMA prototype：

\[
p_{k,c}^{(a)}
=
\operatorname{norm}
\left[
\operatorname{EMA}
\left(
\frac{1}{|\mathcal B_{k,c}|}
\sum_{i\in\mathcal B_{k,c}}h_{k,i}^{(a)}
\right)
\right].
\]

若 \(|\mathcal B_{k,c}|=0\)，不更新该类别。若客户端从未见过 \(c\)，不定义它的 prototype。

### 5.2 类别 prototype geometry

每个客户端在自己的 feature space 内计算类间余弦关系：

\[
G_k^{(a)}(c,c')
=
\cos\left(p_{k,c}^{(a)},p_{k,c'}^{(a)}\right).
\]

\[
G_k^{(a)}\in\mathbb R^{C\times C}.
\]

虽然本地 feature dimension 可不同，但最终 \(G_k^{(a)}\) 的尺寸仅由公共标签数 \(C\) 决定，因此可跨模型比较。

### 5.3 Corruption transport graph

对于每个 PRIME channel：

\[
\Delta G_k^{(a)}(c,c')
=
G_k^{(a)}(c,c')-G_k^{(0)}(c,c').
\]

直观例子：

- 若频域扰动后“汽车-卡车”的 cosine similarity 急剧上升，说明两类在该客户端的边界开始混淆；
- 若“汽车-飞机”关系几乎不变，说明该边较稳定；
- \(\Delta G\) 记录的是类别几何受损坏的方向与幅度，而不是静态类别特征。

### 5.4 Local transport regularization

客户端先在本地保持自己的类别几何在 PRIME view 下不过度塌缩：

\[
\mathcal L_{local-transport}
=
\frac{1}{|\mathcal A||\mathcal E_k|}
\sum_{a\in\mathcal A}
\sum_{(c,c')\in\mathcal E_k}
\left(\Delta G_k^{(a)}(c,c')\right)^2.
\]

\(\mathcal E_k\) 是客户端中两个类别都出现的边集合。该项不是强迫所有类别完全不变，而是对明显的几何崩塌提供基础约束。

---

## 6. 模块三：CTGC 损坏传输几何共识

### 6.1 类别可靠性

客户端对每个类别 \(c\) 和 channel \(a\) 计算可靠性：

\[
r_{k,c}^{(a)}
=
\underbrace{\frac{n_{k,c}}{n_{k,c}+\gamma}}_{\text{support}}
\cdot
\underbrace{\exp\left[
-\frac{1-\cos(p_{k,c}^{(0)},p_{k,c}^{(a)})}{\tau_r}
\right]}_{\text{prototype stability}}
\cdot
\underbrace{\exp\left[-\frac{v_{k,c}^{(a)}}{\tau_v}\right]}_{\text{intra-class compactness}}.
\]

其中 \(v_{k,c}^{(a)}\) 是类别内样本围绕 prototype 的平均角距离。

这三个因子分别防止：

- 极少量 tail 样本以偶然均值支配通信；
- PRIME 下不稳定的类别污染教师；
- 类内过散、可能含严重损坏样本的类别产生错误关系。

### 6.2 服务器的边级聚合

对类别边 \((c,c')\) 和 channel \(a\)，服务器只聚合同时拥有这两个类别的客户端：

\[
\overline{\Delta G}^{(a)}(c,c')
=
\frac{
\sum_{k\in\mathcal K_{c,c'}}
r_{k,c}^{(a)}r_{k,c'}^{(a)}
\Delta G_k^{(a)}(c,c')
}{
\sum_{k\in\mathcal K_{c,c'}}
r_{k,c}^{(a)}r_{k,c'}^{(a)}
\epsilon
}.
\]

输出是：

\[
\left\{
\overline{\Delta G}^{(F)},
\overline{\Delta G}^{(C)},
\overline{\Delta G}^{(S)}
\right\}
\in\mathbb R^{3\times C\times C}.
\]

CIFAR-10 时核心通信教师仅为 \(3\times10\times10=300\) 个标量，不随模型参数量或 feature dimension 增长。

### 6.3 客户端的全局传输校准

收到服务器的 transport graph 后，客户端仅在自己具备的类别边上对齐：

\[
\mathcal L_{CTGC}
=
\frac{1}{|\mathcal A||\mathcal E_k|}
\sum_{a\in\mathcal A}
\sum_{(c,c')\in\mathcal E_k}
w_{k,c,c'}^{(a)}
\left(
\Delta G_k^{(a)}(c,c')
-\operatorname{sg}\left[
\overline{\Delta G}^{(a)}(c,c')
\right]
\right)^2.
\]

\(w_{k,c,c'}^{(a)}\) 可使用本地可靠性乘积。stop-gradient 保证服务器几何教师不会在本地反向传播中被修改。

该损失的含义不是让所有客户端静态 prototype 相同，而是让它们在同一种已知 corruption channel 下，维持可信客户端共有的类别边界响应。

---

## 7. 完整本地目标与训练流程

### 7.1 总损失

\[
\mathcal L_k=
\mathcal L_{CE}
+
\lambda_{jsd}\mathcal L_{JSD}
+
\lambda_{lt}\mathcal L_{local-transport}
+
\lambda_{ct}\mathcal L_{CTGC}.
\]

- \(\mathcal L_{CE}\)：正常分类监督；
- \(\mathcal L_{JSD}\)：PRIME view 的预测一致性；
- \(\mathcal L_{local-transport}\)：本地 corruption geometry stability；
- \(\mathcal L_{CTGC}\)：跨客户端的可靠传输共识。

### 7.2 一次通信轮

1. 客户端从私有训练数据取 batch；
2. 用 PRIME 原语生成基准、频域、颜色域、空间域四个 view；
3. 每个异构模型本地前向，更新 CE、JSD 和本地 prototype；
4. warmup 后计算 local transport loss；
5. 客户端上传三份 \(\Delta G_k^{(a)}\) 及类别可靠性，不上传图片、参数或 raw feature；
6. 服务器按类别边、按 channel 做可靠聚合；
7. 服务器下发三张 global transport graph；
8. 下一轮客户端用 CTGC 对齐其局部 corruption transport。

### 7.3 Warmup

前 \(W\) 轮只训练：

\[
\mathcal L_{CE}+\lambda_{jsd}\mathcal L_{JSD}.
\]

原因：随机初始化下 prototype 和 transport graph 没有可靠语义。warmup 后依次启用：

1. 本地 prototype EMA；
2. local transport loss；
3. 客户端上传 transport graph；
4. CTGC 共识损失，且从较小权重线性增长。

---

## 8. 三类挑战如何对应

| 挑战 | 失效原因 | CTPG 的对应机制 | 不能声称的内容 |
|---|---|---|---|
| 模型异构 | raw feature coordinate 与维度不同 | 只通信 \(C\times C\) 的 prototype geometry transport | 不传模型参数，不强制 feature 对齐 |
| 数据异构 | tail prototype 方差大，head 类支配聚合 | 类别支持度、类内紧凑度和边级可靠性 | 不生成完全 missing 类别知识 |
| 数据损坏 | prototype / 类边界随频域、颜色、空间扰动漂移 | PRIME channel views 和 corruption transport graph | 不保证每种增强绝对语义保持 |

---

## 9. 与 RAHFL 的区别

| RAHFL | FedPRIME-CTPG |
|---|---|
| AugMix 产生鲁棒增强 | PRIME 提供频域、颜色、空间的可解释原语视图 |
| DCL 做本地对比学习 | MPTL 直接约束类别 prototype geometry 的 corruption transport |
| AsymHFL 在 public data 上以客户端可靠性进行异步协作 | CTGC 在私有数据统计上按类别边、按损坏 channel 交流鲁棒几何响应 |
| 主要关注哪些客户端可以更可靠地教别人 | 主要关注哪些类别关系在何种损坏下仍可安全共享 |

这是一种不同的通信对象和不同的鲁棒性定义，不是把 DCL 或 AsymHFL 换名。

---

## 10. 必要基线与最小实验链

如果走原型路线，不能只对比 RAHFL；至少需要：

| 编号 | 方法 | 目的 |
|---|---|---|
| B0 | RAHFL | 强鲁棒异构基线 |
| B1 | PRIME + LogitAvg | 已有基础，验证纯 public-logit 通信 |
| B2 | PRIME + FedProto | 验证原始 prototype 通信在当前四模型上是否可行 |
| B3 | PRIME + FedTGP | 强 prototype-HtFL 基线，必须纳入或至少说明无法复现原因 |
| B4 | PRIME + MPTL | 验证本地 corruption transport 是否有效 |
| B5 | PRIME + MPTL + CTGC | 验证新通信是否有独立收益 |

所有实验必须使用相同：

- RAHFL-style CIFAR-10 corrupted train/test；
- 固定 Dirichlet partition；
- 四种异构模型；
- 通信轮、本地 epoch、优化器、batch size、随机种子；
- avg_acc、worst_acc、tail_acc、missing_acc 和 corruption group 指标。

只有同时满足：

1. B4 优于 B2；
2. B5 优于 B4；
3. B5 在 avg_acc、worst_acc 或 tail_acc 至少一个核心异构指标上稳定优于 B0；

CTPG 才有继续投入多 seed 和论文写作的价值。

---

## 11. 风险清单

1. **新颖性风险：** 若发现已有工作已交流 transformation-conditioned prototype relation 或 prototype transport graph，则需重新设计或放弃；
2. **有效性风险：** JSD 已在本地约束 view 一致性，CTGC 可能只带来很小额外收益；
3. **tail 风险：** 极端 tail 类 prototype 统计不稳定，需要支持度门槛与 EMA；
4. **计算风险：** 四 view 前向比当前三 view 更慢，需先用 debug 和小轮数验证显存、运行时间；
5. **隐私风险：** relation graph 与 support 仍会泄露部分标签结构，论文不能声称严格隐私保证；
6. **公平性风险：** RAHFL 使用 CIFAR-100 public data，而 CTPG 不使用。应透明说明方法使用更少外部数据，必要时补充资源等价实验；
7. **命名风险：** 不宜正式使用 ProtoGraph 作为论文名称，因为已有 Federated Prototype Graph Learning 主要针对图数据。CTPG 暂定名更准确，但投稿前仍需检索名称冲突。

---

## 12. 立项建议

**建议继续探索 CTPG，但不要立刻做完整工程。**

推荐按下列门槛推进：

1. 先完成 B2：PRIME + FedProto 的短轮 smoke test，确认 prototype 通信在四个真实异构模型中技术可行；
2. 实现一个只计算、不参与反向传播的 transport graph diagnostic，确认 PRIME 三 channel 的 prototype geometry 确实存在可区分的稳定/不稳定模式；
3. 若 diagnostic 没有明显 pattern，停止该方向；
4. 若 pattern 明显，再实现 B4 的 local transport loss；
5. B4 有独立收益后，才实现服务器 CTGC；
6. B5 超过 B4 后，再投入资源与 RAHFL、FedTGP 跑完整对照。

这条路线的真正创新不是“使用原型”，而是：

> **将 PRIME 诱导的 class geometry deformation 视为可通信的鲁棒知识，并以模型无关、类别边级的方式在异构客户端之间进行可靠共识。**

---

## 13. 与现有项目文档的关系

- 当前可运行旧框架记录于 docs/project/ARCHITECTURE.md、docs/project/PROJECT_STATE.md、docs/project/TODO_NEXT.md；
- C3L-P 与 D2C-CR 设计稿记录于 docs/archive/methods/FRAMEWORK_REVIEW_C3LP_D2CCR_ZH.md；
- 本文档是新的 prototype communication 候选路线，不覆盖旧代码，也不代表已审批方案；
- 任何代码实现前，需由项目负责人明确确认本路线，并更新 docs/archive/legacy/AGENT.md 中“禁止 prototype communication”的旧限制。
