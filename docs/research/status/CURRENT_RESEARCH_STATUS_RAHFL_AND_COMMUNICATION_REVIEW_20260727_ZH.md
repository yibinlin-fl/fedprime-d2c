# 当前研究现状、RAHFL 基线与通信路线复盘

更新时间：2026-07-27

用途：将本文档直接交给其他 AI 或研究人员，用于讨论下一步方法设计。本文只记录已经实现、已经验证或已经失败的内容，不把尚未跑通的想法描述成结果。

---

## 1. 研究目标

目标是在图像分类联邦学习中同时处理：

1. **模型异构**：不同客户端使用不同网络，如 ResNet10、ResNet12、ShuffleNet、MobileNetV2。
2. **数据异构**：客户端类别分布服从 Dirichlet label skew，部分客户端某些类别极少甚至完全缺失。
3. **数据损坏**：训练和测试图片包含 noise、blur、weather、digital 等 CIFAR-C 风格损坏。
4. **损坏与标签纠缠**：同一客户端内部，某些类别长期与特定损坏 operator 绑定，模型可能把损坏模式当作标签捷径。

希望最终形成一套具有清晰问题动机、独立通信机制和可复现实验协议的方法，并在严格公平条件下超过 RAHFL。

---

## 2. RAHFL 基线概述

RAHFL 是当前项目的主要强基线。其整体结构可以写为：

\[
\boxed{
\text{RAHFL}
=
\text{AugMix}
+\text{JSD}
+\text{DCL}
+\text{AsymHFL}
}
\]

### 2.1 AugMix 与 JSD

每张本地训练图片生成：

- 一个基础视图；
- 两个独立 AugMix 强增强视图；
- 一个 weak view，供 DCL 使用。

基础分类部分对三个 JSD 视图进行预测，并约束预测分布一致：

\[
\mathcal L_{\mathrm{local}}
=
\mathcal L_{\mathrm{CE}}
+\lambda_{\mathrm{JSD}}\mathcal L_{\mathrm{JSD}}.
\]

JSD 的作用是避免模型对不同增强视图给出完全不同的类别分布，从而提高损坏鲁棒性。

### 2.2 DCL

RAHFL 不是简单照搬普通监督对比学习，而是使用其自己的 DCLLoss，将：

- clean feature；
- weak feature；
- strong/AugMix feature；

按照标签关系组织对比约束。

它的目标是让同类不同增强视图在特征空间靠近，同时保持不同类别可分。DCL 是 RAHFL 本地鲁棒表征学习的一部分，但历史实验说明，RAHFL 的高性能主要来自完整的 `AugMix + JSD + DCL` 组合，不能把最终收益全部归因于 DCL。

### 2.3 AsymHFL

由于客户端模型结构不同，不能直接 FedAvg 参数。RAHFL 使用一批公共 CIFAR-100 图片：

1. 所有异构客户端对公共图片输出 CIFAR-10 logits；
2. 服务端根据客户端准确率确定较强模型；
3. 较强模型通过公共样本 logits 非对称地指导较弱模型；
4. 不直接聚合网络参数或异构特征。

这使 RAHFL 支持模型异构，但存在两个重要问题：

1. 原实现每轮使用最终测试集准确率选择教师，存在测试信息参与训练决策的问题；
2. 公共数据是 CIFAR-100，而私有任务是 CIFAR-10，属于跨域 public logits。

### 2.4 RAHFL 的实验实现状态

本项目统一 runner 已经直接复用了 RAHFL 的：

- AugMix 实现；
- JSD 训练方式；
- DCLLoss；
- 异构模型定义；
- AsymHFL 主要通信逻辑。

经典 CIFAR-10-C、Dirichlet `alpha=0.5`、无独立 40 epoch 预训练的统一配置结果：

```text
RAHFL final Avg/Worst = 56.41 / 44.72
AugMix+DCL local-only = 56.11 / 44.23
```

在这套资源受限配置中，AsymHFL 相对 local-only 的最终增量只有：

```text
Avg   +0.30
Worst +0.49
```

这说明 RAHFL 的主要性能来源很可能是强本地鲁棒训练；但不能仅凭一个 seed 宣称 AsymHFL 无效。

---

## 3. 为什么提出 CLE-HFL 新场景

普通 corruption-skew 只表示不同客户端整体遭受不同损坏，容易被质疑：

> 如果客户端部署环境固定，它只需要学好自己的损坏环境，为什么必须跨客户端协作？

为形成更明确的联邦失败模式，项目提出：

\[
\boxed{
\text{CLE-HFL}
=
\text{Corruption-Label Entanglement in Heterogeneous FL}
}
\]

### 3.1 场景定义

先按 Dirichlet 参数 \(\alpha\) 划分类别数据，再为每个客户端 \(k\) 和类别 \(c\) 指定主要损坏 operator：

\[
\phi_k(c)=\text{客户端 }k\text{ 中类别 }c\text{ 的主要损坏}.
\]

训练样本的损坏条件分布为：

\[
P_k(g\mid y=c)
=
\gamma\mathbf 1[g=\phi_k(c)]
+(1-\gamma)P_{\mathrm{background}}(g).
\]

- \(\gamma=0\)：损坏与标签基本独立；
- \(\gamma\) 越大：类别与特定损坏绑定越强；
- \(\gamma=0.9\)：模型极易学习 corruption shortcut。

例如客户端 0 的猫主要带 motion blur，而客户端 1 的猫主要带 defocus blur。每个客户端可能形成不同的“损坏就是类别”的错误规律。

### 3.2 CLE-HFL v2

为了避免方法依赖四个固定 corruption family，v2 协议改为 operator 级评价：

```text
15 个具体 CIFAR-C 风格 operator
11 个 seen operator：可以出现在客户端训练中
4 个 unseen operator：不出现在任何客户端训练中
训练：client/class 与具体 seen operator 纠缠
测试：clean + seen + unseen + 全部 operator
```

operator ID、名称、family、seen/unseen 标志和 severity 都只能用于数据生成与事后评价，不能传给训练方法。

### 3.3 新场景是否有实验依据

旧版 CLE-HFL 中，固定其他条件，仅增加纠缠强度：

```text
gamma=0.0:
  Avg 52.17, Worst 44.17, WCCA 35.35, CFG 2.54

gamma=0.6:
  Avg 50.82, Worst 42.83, WCCA 25.88, CFG 5.91

gamma=0.9:
  Avg 46.72, Worst 38.16, WCCA 19.32, CFG 10.91
```

从 `gamma=0.0` 到 `0.9`：

```text
Avg   -5.45
Worst -6.02
WCCA  -16.02
CFG   +8.37
```

因此，随着 corruption-label entanglement 增强，RAHFL 的平均性能、最弱客户端表现和最差类别-损坏单元明显恶化，反事实差距明显增大。新问题具有可测的失败模式。

---

## 4. CLE-HFL 关键指标

### 4.1 Avg Accuracy

四个异构客户端在完整测试集上的平均准确率。

### 4.2 Worst Accuracy

最弱客户端的准确率，用于判断通信是否只帮助强客户端。

### 4.3 WCCA

Worst Class-Corruption Accuracy：

\[
\mathrm{WCCA}
=
\min_{k,c,g}
\operatorname{Acc}(k,c,g).
\]

它检查最差的 `client × class × operator` 单元。WCCA 很低说明平均准确率可能由 head class 掩盖，某些类别在某些损坏下几乎完全失效。

### 4.4 CFG

Counterfactual Gap。它衡量类别在训练中绑定的主要损坏与反事实损坏之间的性能差距。CFG 越大，说明模型越依赖 corruption shortcut；越小越好。

---

## 5. 已探索方法与真实结论

### 5.1 PRIME + LogitAvg / D2C

PRIME 本地增强能够正常训练，但 D2C 使用跨域 CIFAR-100 平均预测估计 private class prior：

\[
\hat\pi_{k,c}
=
\mathbb E_{u\sim D_{\mathrm{pub}}}
[p_k(c\mid u)].
\]

然后同时用该 prior 控制 prior debias、class-balanced aggregation 和 complementary KD。

结果：

```text
PRIME + LogitAvg final Avg ≈ 52.10
FedPRIME-D2C final Avg      ≈ 52.31
Oracle D2C final Avg        ≈ 51.74
```

主要理论问题：

- CIFAR-100 public prediction tendency 不等于 CIFAR-10 private label prior；
- 去偏只能移动 logits，不能创造客户端完全缺失的类别语义；
- 错误 prior 会同时污染去偏、聚合和互补蒸馏；
- Oracle prior 也没有改善结果，说明瓶颈不只是 prior 估计误差。

结论：D2C 路线失败，不再作为主方法。

### 5.2 FedPRIME-PAIR / CPAD

尝试将知识通信细化为类别对边界，并设计来源 expertise、leave-one-out teacher 和 pairwise BCE。

结果：

```text
FedPRIME-PAIR final Avg ≈ 50.15
best Avg ≈ 51.10
```

问题是类别对边界仍从跨域 public logits 构造，教师能力估计与真正的 counterfactual robustness 不一致。

结论：归档。

### 5.3 PRAC-HFL

使用 receiver 侧虚拟分类头更新，检查某教师是否降低本地风险，再决定是否接受通信。

结果：

```text
public1 final Avg/Worst = 54.63/41.88
public4 final Avg/Worst = 52.96/43.27
local-only              = 56.11/44.23
RAHFL                    = 56.41/44.72
```

PRAC 有非零通信行为，但通信没有超过相同本地训练的 local-only。虚拟更新成本较高，且风险估计与最终 counterfactual 指标不完全一致。

结论：归档。

### 5.4 NIR-DCL / SARA

对 DCL 进行 label-skew-aware 的类别平衡和关系校正。

关键结果：

```text
NIR-DCL local-only       = 53.30/36.01
NIR-DCL + AsymHFL       = 57.36/46.23

SARA local-only         = 54.10/32.06
SARA + AsymHFL          = 57.83/46.59
RAHFL                   = 56.41/44.72
```

SARA + AsymHFL 在 `alpha=0.5, seed=0` 下比 RAHFL 高：

```text
Avg   +1.42
Worst +1.87
```

但 SARA local-only 较弱，说明它不是独立更强的本地学习模块。更合理的解释是：

> SARA 改变了 label-skew 下的局部表征几何，使其与 AsymHFL 通信更兼容。

该结果是当前经典场景下最好的正结果，但创新主要集中在 DCL 改动，通信仍复用 AsymHFL，论文贡献偏薄。

### 5.5 FedCARA

尝试用自定义 CARA-C 替代 AsymHFL：

```text
FedCARA final Avg/Worst  = 55.88/45.93
CARA-L + AsymHFL        = 57.36/46.23
RAHFL                    = 56.41/44.72
```

自定义通信提高了 Worst，但降低 Avg，并弱于原 AsymHFL。说明新的通信方式没有正确保留 AsymHFL 的有效知识载荷。

### 5.6 FedEASE

围绕 CLE-HFL 使用显式环境建模：

- PEW：环境 witness；
- BER+CDep：环境均衡风险与条件依赖约束；
- EBST：环境结构通信；
- SCP：通信安全投影。

Oracle 环境信息下，本地模块产生明显收益：

```text
control final Avg/Worst/WCCA/CFG = 37.58/30.11/13.70/10.855
Oracle BER+CDep                  = 41.62/35.52/14.00/6.155
```

学习式 PEW 也保留大部分收益：

```text
learned PEW BER+CDep = 40.37/35.42/13.925/6.370
```

但 PEW 使用预定义环境类别，容易被质疑依赖固定 corruption taxonomy。EBST/EBST-v2 通信没有稳定超过匹配 local-only：

```text
calibrated local final = 42.85/36.23/WCCA 19.775/CFG 6.573
EBST-v2 final          = 42.63/35.30/WCCA 20.675/CFG 7.290
```

结论：

- 环境条件本地学习具有真实正信号；
- 硬 taxonomy 不适合作为最终主方法；
- EBST 通信失败。

### 5.7 FedFalsify v0.2/v0.3

使用固定 `D_fit/D_audit`，在接收方私有 audit 数据上验证教师是否有益：

- TAU：分类头梯度兼容性；
- paired correctness advantage；
- non-inferiority UCB；
- CMT：保守 margin transfer。

严格 control 与 v0.3：

```text
strict control final  = 37.7788/31.8025/WCCA 9.550/CFG 9.4625
FedFalsify v0.3 final = 39.0631/32.1475/WCCA 12.750/CFG 9.1125

final delta:
  Avg +1.2844
  Worst +0.3450
  WCCA +3.200
  CFG -0.3500
```

但最后五轮 CFG 变差；迁移到 operator-level CLE-HFL v2 后也失败：

```text
Strict control final = 30.7550/24.9800/WCCA 0.250/CFG 30.225
FedFalsify v0.3      = 31.0733/24.5733/WCCA 0.500/CFG 31.825
```

相对 control：

```text
Avg +0.3183
Worst -0.4067
WCCA +0.250
CFG +1.600
```

结论：教师筛选信号存在，但不足以控制 counterfactual negative transfer。

---

## 6. 最新 robust frontier 审计

为了彻底移除固定 corruption taxonomy，定义每个客户端类别对的多视图鲁棒边界：

\[
q_{k,c,j}
=
Q_{0.2}
\left[
\min_v
\left(
\bar z_{k,c}^{(v)}(x)
-
\bar z_{k,j}^{(v)}(x)
\right)
\mid y=c
\right].
\]

其中 \(\bar z\) 是每个样本在类别维做 z-score 后的 logits，用于消除 ResNet、ShuffleNet、MobileNet 之间的 logit 尺度差异。

该统计不使用 operator ID、名称、family、severity 或 seen/unseen 信息。

### 6.1 可辨识性结果

```text
本地边界 vs seen 最差 operator Spearman       = 0.434
本地边界 vs unseen 最差 operator Spearman     = 0.559
来源边界优势 vs seen 实际优势 Spearman         = 0.319
来源边界优势 vs unseen 实际优势 Spearman       = 0.548
```

说明 robust frontier 能诊断类别脆弱性。

但只要 `q_source > q_receiver` 就通信时，教师排序精度为：

```text
seen   52.94%
unseen 52.94%
```

无法安全路由。

只保留 all-view 边界优势最大的 25% 后，三个增强种子的排序精度为：

```text
seen:   77.78% / 88.89% / 88.89%
unseen: 88.89% / 100.00% / 88.89%
```

每次保留 9 条路线，三个种子有 7 条完全相同。

### 6.2 一步分类头更新审计

为了判断边界矩阵能否直接作为知识载荷，对 7 条稳定路线进行了匹配因果对照：

```text
control   = 相同 fit batch + 一次 CE head update
candidate = 相同设置 + robust-frontier tail-margin loss
```

Candidate 减 Control：

```text
目标类 audit accuracy 平均增量 = 0.0000
目标类 audit loss：7/7 微幅下降
seen mean accuracy 平均增量    = 0.0000
unseen mean accuracy 平均增量  = +0.0357
整体 audit accuracy 平均增量   = -0.0095
```

结论：

> Robust frontier 能表示“谁更可靠”，但一个 \(C\times C\) 边界矩阵没有足够的样本级语义内容，不能单独承担知识传递。

因此直接 boundary transfer 为 NO-GO。

---

## 7. 所有失败是否都是 public logits 导致的

答案是：

\[
\boxed{\text{不是，但 public logits 是当前通信路线的核心瓶颈之一。}}
\]

### 7.1 public logits 的确存在根本限制

#### 跨域语义不足

私有任务是 CIFAR-10，公共数据是 CIFAR-100。公共图片并不直接覆盖 CIFAR-10 的相同类别语义。

#### 不能创造缺失类知识

如果客户端从未见过汽车，单纯校准其 CIFAR-100 输出或减去 prior，无法让它学会汽车视觉特征。

#### 教师可靠性随样本变化

一个客户端可能擅长猫狗边界，但不擅长汽车卡车边界。客户端级总体准确率不能代表每个类别和公共样本上的可靠性。

#### 公共域上的优势不等于 counterfactual robustness

教师在 CIFAR-100 上输出稳定，不代表它在 CIFAR-10 unseen corruption 上更鲁棒。

#### 蒸馏载荷上限

只有 logits 时，能传递的是有限维决策结果，无法完整传递异构模型内部的鲁棒视觉表征。

### 7.2 但不能把所有失败都归因于 public logits

#### 本地模块自身可能失败

SARA、NIR-DCL 的 local-only 均弱于 AugMix+DCL local-only。这与 public logits 无关。

#### 教师选择可能失败

即使知识载荷本身可用，如果来源选择错误，也会发生负迁移。FedFalsify 和 PRAC 主要暴露的是路由/接受机制问题。

#### 硬 corruption taxonomy 会限制泛化

FedEASE 本地模块有正收益，但 PEW 依赖预定义环境类别。这不是 public logits 导致的，而是问题信息假设过强。

#### 过度压缩的非 logit 载荷也会失败

最新 \(C\times C\) robust frontier 完全不依赖 public logits，仍然无法直接教会模型，因为它缺少样本级语义。

#### 优化问题

PRAC 曾出现梯度 NaN；部分方法的 loss 权重、虚拟学习率和梯度尺度也会影响结果。这属于数值优化，不是通信媒介本身。

#### 公平性与测试泄漏

RAHFL 原始路由使用最终测试准确率；strict control 使用 fit/audit 划分。数据预算和选择信息不一致会混淆结果。

### 7.3 更准确的总判断

当前失败来自三个层次：

```text
第一层：可靠性
  能否判断谁在 receiver/class/sample 上真的更强？

第二层：知识载荷
  确认来源后，到底传什么才能让异构接收模型学到视觉知识？

第三层：安全注入
  如何在不损害本地已学知识的情况下吸收该载荷？
```

历史方法往往只解决其中一层：

- D2C：试图修正 logits，但 prior 错误；
- PRAC/FedFalsify：强化接受判断，但载荷仍弱；
- robust frontier：可靠性诊断较好，但载荷过度压缩；
- EBST：结构载荷存在，但来源和接收安全不足；
- SARA：改善本地表示与现有通信兼容性，但没有自己的通信。

所以真正的瓶颈不能简化成“public logits 一定错误”，而是：

> 当前尚未找到同时满足可靠来源、充分语义载荷和安全接收的异构通信机制。

---

## 8. 当前最清晰的候选结构

如果继续使用现有工程基础，逻辑上最完整的候选是：

\[
\boxed{
\text{Local Robust Base}
+
\text{Taxonomy-Free Reliability Gate}
+
\text{Sample-Level Semantic Payload}
+
\text{Receiver-Side Safety}
}
\]

对应含义：

1. `AugMix + JSD + DCL`：保留已验证的本地鲁棒基座；
2. robust frontier：只判断某来源是否有稳定且足够大的鲁棒优势；
3. sample-level payload：必须比单个 \(C\times C\) 摘要包含更多输入条件语义；
4. abstention/SCP：证据不足或更新伤害本地 audit 时拒绝通信。

目前第 1、2 层已有证据，第 3 层仍未解决。

若使用 public logits 作为第 3 层，应先回答：

1. 如何降低 CIFAR-100 与 CIFAR-10 的跨域偏差？
2. 如何让传递按 receiver/class/sample 条件化，而不是全局平均？
3. 如何证明它在 CLE-HFL unseen operator 上传递的是语义，而不是另一个 shortcut？
4. 如何在不使用最终 test 标签的条件下验证来源与更新？

若放弃 public logits，则新的载荷也必须满足：

- 不要求同构特征维度；
- 不直接共享私有图片；
- 不依赖固定 corruption taxonomy；
- 比 \(C\times C\) 统计保留更多可学习语义；
- 能在一步匹配对照中产生可测正收益。

---

## 9. 当前不应该做什么

1. 不直接运行 40 轮 robust-frontier transfer。
2. 不继续只调 `lambda`、margin、quantile 来碰运气。
3. 不把一次 seed0 的 SARA 提升直接写成稳定结论。
4. 不把 RAHFL 的测试准确率路由当作严格公平实现。
5. 不重新引入固定 noise/blur/weather/digital 标签作为最终方法输入。
6. 不宣称 public logits 已经被理论证明完全不可用。
7. 不再提出只有可靠性权重、但没有明确知识载荷的新通信模块。

---

## 10. 希望外部讨论重点回答的问题

请基于以上已验证事实回答，而不是重新建议已经失败的 prior debias、简单 teacher weighting 或类别原型平均。

### 问题一

在模型异构、label skew、corruption-label entanglement 下，除了跨域 public logits 和同维特征/普通 FedProto，还有什么可实现的 sample-level semantic knowledge payload？

### 问题二

能否将 taxonomy-free robust frontier 仅作为可靠性 gate，并设计一种理论上充分的知识载荷，使其真正改善 unseen corruption，而不是只改变分类头 margin？

### 问题三

如果继续 public logits 路线，怎样从信息论、蒸馏或域适应角度解决 CIFAR-100 public data 与 CIFAR-10 private task 的语义不匹配？

### 问题四

是否应该把论文贡献集中为：

```text
新问题 CLE-HFL v2
+ operator-level benchmark
+ taxonomy-free negative-transfer-safe communication
```

而不再强调 PRIME 或频域？

### 问题五

在现有结果下，什么最小方法改动既有真实理论动机，又足以形成 CCF-B 级别的小论文贡献，而不是 RAHFL 的轻量改版？

---

## 11. 相关项目文件

项目记忆：

```text
AGENTS.md
docs/project/CURRENT_PROJECT_MEMORY.md
docs/project/PROJECT_STATE.md
docs/project/TODO_NEXT.md
```

RAHFL 阅读：

```text
docs/research/baselines/RAHFL_IMPLEMENTATION_READING_ZH.md
RAHFL-master/
```

CLE-HFL v2：

```text
docs/archive/methods/CLE_HFL_V2_FEDFALSIFY_FRAMEWORK_ZH.md
docs/experiments/guides/CLE_HFL_V2_OPENI_RUN_GUIDE_ZH.md
```

最新边界审计：

```text
deliverables/robust_frontier_audit_20260726/ROBUST_FRONTIER_AUDIT_ZH.md
outputs/robust_frontier_audit_20260726/
outputs/robust_frontier_one_step_audit_20260727/
```

