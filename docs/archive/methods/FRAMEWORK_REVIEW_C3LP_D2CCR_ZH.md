# FedPRIME 新版完整框架审阅稿

**版本：** 设计版本 v0.1  
**日期：** 2026-06-24  
**状态：** 仅供方案审核，尚未编码、尚未产生新版实验结果。  
**拟议名称：** FedPRIME-C3L-P-D2C-CR  

---

## 0. 审阅前必须知道的事实

这份文档描述的是拟议的新版框架：

\[
\boxed{
\text{FedPRIME}=
\text{PRIME}
+
\textbf{C}^{3}\textbf{L-P}
+
\textbf{D}^{2}\textbf{C-CR}
}
\]

它与仓库当前可运行的 FedPRIME-D2C 不是同一版本。

| 内容 | 当前状态 |
|---|---|
| PRIME 本地多视图鲁棒训练 | 已实现并完成稳定性修复 |
| PRIME + LogitAvg | 已跑完，最终 avg_acc 为 52.10% |
| PRIME + 旧 D2C | 已跑完，最终 avg_acc 为 52.31% |
| Oracle D2C | 已跑完，最终 avg_acc 为 51.74% |
| RAHFL：AugMix + DCL + AsymHFL | 已跑完，最终 avg_acc 为 56.41% |
| C3L-P | 尚未实现 |
| D2C-CR | 尚未实现 |
| 新版全框架 | 尚未实现、尚未验证 |

旧 D2C 的“由跨域 CIFAR-100 public logits 推测 private prior，再做 logit 去偏”的理论假设已被 Oracle D2C 实验削弱。因此，旧 D2C 不再建议作为主创新继续调参。

**重要兼容性提醒：** 当前 docs/archive/legacy/AGENT.md 仍规定 D2C 必须使用 public logits、prior debiasing、class-balanced aggregation 和 complementary KD。D2C-CR 是对该旧通信定义的研究性替换，不是小改动。只有本审阅稿通过后，才应更新 docs/archive/legacy/AGENT.md 和训练代码。

---

## 1. 研究目标与问题边界

### 1.1 总目标

在以下三种困难同时存在时，构建不依赖模型参数聚合的鲁棒联邦学习方法：

1. **模型异构：** 客户端使用不同网络结构，参数和特征维度无法直接平均；
2. **数据异构：** Dirichlet label skew 使不同客户端类别比例显著不同；
3. **数据损坏：** 训练和测试图像受到 common corruption。

第一阶段任务为 CIFAR-10 十分类。当前客户端模型是 ResNet10、ResNet12、ShuffleNet、MobileNetV2，数据分区主设置为 Dirichlet \(\alpha=0.5\)，训练和测试都使用 RAHFL-style random corruption。

### 1.2 新框架真正要解决什么

新版不再声称解决“客户端没有该类任何样本时，仍能学会该类视觉语义”。

它的科学问题改为：

> 在受损的 Non-IID 客户端中，如何让每个客户端对其已经出现、但数量稀少的类别学习到更稳的表征和更合理的决策边界；并让结构不同的模型只通过共享类别输出空间进行协作。

因此，主指标是 avg_acc、worst_acc 和 tail_acc。missing_acc 必须保留报告，但作为方法边界诊断，而非主要成功标准。

### 1.3 不做什么

- 不做 FedAvg 参数聚合；
- 不上传或聚合 feature prototype；
- 不做跨客户端 feature alignment；
- 不从 CIFAR-100 输出推断 CIFAR-10 私有类别先验；
- 不使用测试集标签选择教师、路由客户端或调整训练超参数；
- 不制造 missing class 的虚假正样本。

---

## 2. 整体结构

\[
\text{Private corrupted data}
\rightarrow
\text{PRIME multi-view generation}
\rightarrow
\text{C3L-P local robust representation learning}
\rightarrow
\text{D2C-CR class-relation communication}
\rightarrow
\text{personalized local update}.
\]

可按功能分为三层：

| 层 | 模块 | 输入 | 输出 | 主要解决的问题 |
|---|---|---|---|---|
| 本地增强层 | PRIME | 私有损坏图像 | clean、PRIME-view1、PRIME-view2 | 抗数据损坏 |
| 本地表征层 | C3L-P | 三视图、标签、私有模型表示 | 类别均衡且损坏稳定的表征 | Non-IID 下 head/tail 失衡、强增强正对不可靠 |
| 通信层 | D2C-CR | 已标注私有验证样本的类别关系 | 类别条件软教师矩阵 | 模型异构下的决策边界协作 |

---

## 3. 输入、数据划分与符号

### 3.1 数据

设有 \(K\) 个客户端，每个客户端 \(k\) 有私有训练集：

\[
\mathcal D_k=\{(x_i,y_i)\}_{i=1}^{N_k},\qquad y_i\in\{1,\ldots,C\}.
\]

每个 \(\mathcal D_k\) 再按固定随机种子分成：

\[
\mathcal D_k=\mathcal D_k^{train}\cup\mathcal V_k.
\]

- \(\mathcal D_k^{train}\)：本地训练；
- \(\mathcal V_k\)：从训练集切出的私有小验证集，只用于关系可靠性估计；
- \(\mathcal T\)：独立测试集，只用于记录指标，绝不参与训练、教师选择或权重计算。

划分应分层进行：每个本地非零类别尽量在 \(\mathcal V_k\) 留出少量样本；样本过少的类别不上传关系行。

### 3.2 模型

客户端 \(k\) 自己持有：

\[
f_k: \text{image}\rightarrow\text{feature},\qquad
g_k: \text{feature}\rightarrow\mathbb R^{d_k},\qquad
a_k: \text{feature}\rightarrow\mathbb R^C.
\]

- \(f_k\)：异构分类骨干；
- \(g_k\)：仅用于本地对比学习的 projection head；
- \(a_k\)：输出所有客户端共同拥有的 \(C\) 类 logits。

注意 \(d_k\) 可以不同，任何 \(f_k\) 或 \(g_k\) 的参数和表示均不离开客户端。

### 3.3 PRIME 三视图

对私有训练样本 \((x_i,y_i)\)，产生：

\[
x_i^{(0)}=x_i^{clean},
\quad
x_i^{(1)}=\operatorname{PRIME}_1(x_i),
\quad
x_i^{(2)}=\operatorname{PRIME}_2(x_i).
\]

PRIME 的频域随机滤波、颜色变换与微分同胚形变共同提供多样的语义保持扰动。它是本地增强器，不承担通信功能。

对每个视图：

\[
h_i^{(v)}=
\frac{g_k(f_k(x_i^{(v)}))}
{\|g_k(f_k(x_i^{(v)}))\|_2},
\qquad
p_i^{(v)}=
\operatorname{softmax}\left(a_k(f_k(x_i^{(v)}))/T_d\right).
\]

---

## 4. 模块一：PRIME 本地鲁棒学习

PRIME 的作用是把同一私有样本暴露于频域、颜色域和空间域的合理扰动。客户端应同时看到 clean 与两个 PRIME 视图。

基础分类与一致性目标为：

\[
\mathcal L_{base}
=
\mathcal L_{CE}(p_i^{(0)},y_i)
+
\lambda_{jsd}\mathcal L_{JSD}
\left(p_i^{(0)},p_i^{(1)},p_i^{(2)}\right).
\]

其中 JSD 约束三视图在类别输出上保持一致；CE 保证这种一致性不脱离真实标签。

PRIME 已经是可运行模块。新版不会改造 PRIME 的基本原语，也不会把 PRIME 当作通信探针。

---

## 5. 模块二：C3L-P 本地监督对比学习

### 5.1 模块名称与直觉

**C3L-P = Corruption-Consistent, Class-Balanced, Probabilistic Contrastive Learning。**

它专门处理一个耦合问题：

> Non-IID 客户端中的 tail 类样本本来就少；一个 batch 中有效同类正对更少；而 PRIME 的强视图中又可能有部分对当前模型并不可靠。普通 SupCon 会让 head 类主导梯度，也可能把不可靠强视图硬拉近。

C3L-P 的三个部分是：

1. **Corruption-Consistent：** clean 与 PRIME 多视图属于同一语义，应形成同类正对；
2. **Class-Balanced：** 类别而非样本数决定对比损失的总体贡献；
3. **Probabilistic：** 为已出现但样本过少的 tail 类补足本地统计意义上的正对。

### 5.2 视图可靠性

先计算三视图预测的分歧：

\[
\bar p_i=\frac{1}{3}\sum_{v=0}^{2}p_i^{(v)},
\qquad
d_i=\frac{1}{3}\sum_{v=0}^{2}
\operatorname{KL}(p_i^{(v)}\Vert\bar p_i).
\]

warmup 结束后，定义可靠性：

\[
r_i=
\operatorname{clip}
\left(
\exp[-\operatorname{sg}(d_i)/\rho],
r_{min},1
\right).
\]

\(\operatorname{sg}\) 是 stop-gradient。它不是要求“增强前后预测必须完全一样”，而是当强视图与 clean 明显冲突时，避免让这一个不可靠正对产生过大的对比梯度。warmup 前固定 \(r_i=1\)，防止随机初始化造成错误过滤。

### 5.3 类别均衡监督对比

设 \(n_{k,c}\) 是客户端 \(k\) 中类别 \(c\) 的样本数。对出现的类别使用 effective-number 权重：

\[
w_{k,c}=\frac{1-\beta}{1-\beta^{n_{k,c}}}.
\]

对 anchor \(i\)，正对集合 \(\mathcal P_i\) 包含同标签的其他样本和同一样本的其他 PRIME 视图。候选集合为 \(\mathcal A_i\)。类别均衡对比项为：

\[
\ell_i^{bal}
=
-\frac{1}{|\mathcal P_i|}
\sum_{p\in\mathcal P_i}
\log
\frac{\exp(h_i^\top h_p/\tau)}
{\sum_{a\in\mathcal A_i}\widetilde w_{k,y_a}
\exp(h_i^\top h_a/\tau)}.
\]

最终损失先对每个类别平均，再对该类样本平均：

\[
\mathcal L_{bal}
=
\frac{1}{|\mathcal C_B|}
\sum_{c\in\mathcal C_B}
\frac{1}{|\mathcal I_c|}
\sum_{i\in\mathcal I_c}r_i\ell_i^{bal}.
\]

因此，大类不会只因样本数量多而压过小类；同时小类也不会因一个极端样本获得无限大权重。

### 5.4 概率式 tail 正对补偿

普通 SupCon 的问题是：如果 tail 类在当前 batch 中只有一个真实样本，就只有同一样本的两个增强视图，跨样本正对不足。

对已出现的 tail 类，在本地维护可靠 clean-view 表示的一阶统计：

\[
\mu_{k,c}
=
\operatorname{norm}
\left(
\operatorname{EMA}
\left[
\frac{\sum_{i:y_i=c}r_i h_i^{(0)}}
{\sum_{i:y_i=c}r_i+\epsilon}
\right]
\right).
\]

以 \(\mu_{k,c}\) 和分布集中度 \(\kappa_{k,c}\) 建立只存在客户端内存中的球面分布：

\[
q_{k,c}(h)
=
\operatorname{vMF}(h;\mu_{k,c},\kappa_{k,c}).
\]

只有当 \(n_{min}\le n_{k,c}<n_k^{ref}\) 时，才从该分布补充受上限限制的虚拟正对。它们：

- 不是合成图片；
- 不是跨客户端 prototype；
- 不上传、不聚合；
- 不更新其梯度；
- 对 \(n_{k,c}=0\) 的 missing 类别不产生任何内容。

此设计的研究假设是：对于有少量真实观测的 tail 类，稳定历史表征比“仅依赖当前 batch 的两三个正对”更可靠。

### 5.5 C3L-P 的边界

C3L-P 不保证解决 missing class。它只改善：

- 本地已经拥有但数量较少的类别；
- 同一类别在 PRIME 损坏前后的表征稳定性；
- head/tail 的表征梯度失衡。

---

## 6. 模块三：D2C-CR 通信

### 6.1 名称与核心变化

**D2C-CR = Distribution-aware Decision Consensus via Class Relations。**

它取代旧 D2C 的 prior 估计与 logit 去偏。通信不再回答“客户端的真实类别先验是什么”，而是回答：

> 对一个客户端实际拥有的类别，各模型对该类别的决策边界和混淆关系中，哪些部分可信、可以共享？

### 6.2 客户端本地类别关系

在私有验证集 \(\mathcal V_k\) 上，对本地存在类别 \(c\)，计算三视图平均软输出：

\[
R_{k,c}
=
\frac{1}{Z_{k,c}}
\sum_{(x_i,y_i=c)\in\mathcal V_k}
r_i
\cdot
\frac{p_i^{(0)}+p_i^{(1)}+p_i^{(2)}}{3}.
\]

\[
R_{k,c}\in\mathbb R^C.
\]

例如 \(R_{k,automobile}\) 的第 truck 维度较高，表示该客户端模型常把汽车与卡车混淆。这个向量描述类别条件下的决策关系，而非图像、特征或模型权重。

客户端额外上传：

- 该行支持样本数；
- 私有验证准确率；
- PRIME 三视图平均一致性。

客户端只上传样本足够、可靠性达标的类别行。

### 6.3 服务端可靠聚合

对每个类别 \(c\)，服务器仅从拥有该类的客户端集合 \(\mathcal K_c\) 聚合：

\[
R_{g,c}
=
\frac{\sum_{k\in\mathcal K_c}a_{k,c}R_{k,c}}
{\sum_{k\in\mathcal K_c}a_{k,c}+\epsilon}.
\]

其中权重：

\[
a_{k,c}
=
n^{val}_{k,c}
\cdot
\operatorname{Acc}^{val}_{k,c}
\cdot
\operatorname{Consistency}_{k,c}.
\]

实际实现中应先把三项归一化，再使用乘积或几何平均，避免大客户端单独垄断教师。关键是所有权重来自私有训练划分，绝不读取全局测试标签。

服务器回传：

\[
\mathcal R_g=
\{R_{g,1},\ldots,R_{g,C}\}\in\mathbb R^{C\times C}.
\]

对于 CIFAR-10，核心通信内容是 \(10\times10=100\) 个浮点关系值与少量元数据，远小于公共图像批次上的大量 logits。

### 6.4 本地关系蒸馏

客户端只在自己拥有真实标签 \(c=y_i\) 的样本上蒸馏：

\[
\widetilde R_{g,c}
=
(1-\delta)e_c
+
\delta R_{g,c}.
\]

\[
\mathcal L_{D2C-CR}
=
\frac{1}{|B|}
\sum_{i\in B}
\mathbb 1[\widetilde R_{g,y_i}\text{ exists}]
\operatorname{KL}
\left(
\operatorname{sg}(\widetilde R_{g,y_i})
\Vert p_i^{(0)}
\right).
\]

其中 \(e_c\) 是真实类 one-hot 向量。有界 teacher 使真实标签始终占主要地位，跨客户端软关系只承担边界校准作用。

### 6.5 为什么它支持模型异构

所有模型虽然内部结构不同，却共享任务输出空间：

\[
\mathbb R^{C}.
\]

因此，D2C-CR 只要求：

- 所有客户端有相同标签集合；
- 每个模型能输出 \(C\) 类概率；
- 客户端能在私有验证集上统计类别关系。

它不要求模型参数相同、特征维度相同、层数相同或共享视觉表示。

### 6.6 为什么它缓解数据异构

对一个本地稀缺但非零的类别，例如客户端 A 的汽车样本少：

1. C3L-P 先确保汽车类不会被本地大量 head 类淹没；
2. 客户端 B、C 若也有汽车类且验证表现更好，可向服务器提供更可靠的汽车决策关系；
3. 服务器聚合形成 \(R_{g,automobile}\)；
4. 客户端 A 在自己的汽车样本上学习该软边界，而不是从无关 CIFAR-100 图片上猜“汽车先验”。

这不是缺失类知识生成，而是对已观测 tail 类的跨模型边界校准。

### 6.7 风险与限制

D2C-CR 和 relational distillation / class relation KD 存在文献邻近性，不能直接宣称“首次关系蒸馏”。它是否是论文主创新取决于：

1. 定向文献排重后，能否证实它与已有 FL relation KD 在通信对象与条件聚合上不同；
2. E5 是否显著优于无 D2C-CR 的 E4；
3. 该增益是否在多个随机种子和更严格 RAHFL-val 下保持。

此外，类别关系和样本支持数会泄露一部分标签分布信息。正式论文应说明威胁边界，并研究 secure aggregation 或裁剪/噪声作为可选隐私增强。

---

## 7. 完整训练流程

### 7.1 初始化

1. 固定随机种子；
2. 生成并保存 Dirichlet partition；
3. 每客户端建立异构骨干、私有 projection head、私有分类器；
4. 从每个私有训练集分层划出 \(\mathcal V_k\)；
5. 初始化 C3L-P 的本地 EMA 统计；
6. 初始化服务器关系矩阵 \(\mathcal R_g\) 为空或均匀先验；
7. 设定 warmup 通信轮数。

### 7.2 第 \(t\) 轮本地阶段

每个客户端重复本地 epoch：

1. 从 \(\mathcal D_k^{train}\) 取一个带标签 batch；
2. 生成 clean、PRIME-view1、PRIME-view2；
3. 三份视图经各自异构模型前向；
4. 得到 CE、JSD、C3L-P 和可用的 D2C-CR 损失；
5. 反向传播，只更新本客户端的 \(f_k,g_k,a_k\)；
6. 更新 C3L-P 的本地可靠 tail 分布统计。

本地总损失：

\[
\mathcal L_k
=
\mathcal L_{CE}
+
\lambda_{jsd}\mathcal L_{JSD}
+
\lambda_c\mathcal L_{C3L-P}
+
\lambda_d\mathcal L_{D2C-CR}.
\]

### 7.3 第 \(t\) 轮通信阶段

1. 客户端在 \(\mathcal V_k\) 上计算每个可用类别的 \(R_{k,c}\) 和可靠性元数据；
2. 客户端上传这些 \(C\) 维类别关系行，不上传参数、图像或 feature；
3. 服务器按类别可靠聚合为 \(\mathcal R_g\)；
4. 服务器向所有客户端广播 \(\mathcal R_g\)；
5. 下一轮客户端把该矩阵作为关系蒸馏教师使用。

### 7.4 Warmup 规则

前 \(W\) 轮：

- 保持 PRIME + CE + JSD；
- 可启用类别均衡 SupCon 的基础版本；
- 不使用可靠性 gate；
- 不启用概率 tail 正对；
- 不使用 D2C-CR 蒸馏。

原因是早期预测不稳定。若一开始就按模型置信度过滤或构造教师，会把随机偏差固化。

从第 \(W+1\) 轮起：

- 启用 view reliability；
- 启用满足样本量门槛的概率 tail 正对；
- 上传并聚合类别关系；
- 启用较小权重的 D2C-CR，随后可线性升至目标 \(\lambda_d\)。

---

## 8. 三类异构的对应关系

| 挑战 | 根源 | 新框架中的处理 | 不做的夸张承诺 |
|---|---|---|---|
| 模型异构 | 模型参数、特征维度不同 | 只交换共同的 \(C\) 维类别关系矩阵 | 不平均模型参数，不要求 feature 对齐 |
| 数据异构 | 本地类别数量与比例不同 | 类别均衡对比、tail 概率正对、按类可靠关系聚合 | 不保证补全零样本 missing class |
| 数据损坏 | 图像频域、颜色、空间扰动 | PRIME 三视图、JSD、一致性加权的 C3L-P | 不假设所有增强都永远语义保持 |

---

## 9. 评估与公平比较

### 9.1 固定条件

新版与 RAHFL 的所有核心比较应固定：

- CIFAR-10-C RAHFL-style 训练数据和测试数据；
- 相同 Dirichlet partition 文件；
- 相同四个异构模型；
- 相同通信轮数、本地 epoch、优化器、学习率、batch size；
- 相同随机种子；
- 相同测试指标与 checkpoint 策略。

### 9.2 公共数据的公平性

RAHFL 与旧 LogitAvg 使用 CIFAR-100 public data。D2C-CR 的核心通信不依赖公共图像，因此外部数据依赖更弱，而不是额外占用资源。

论文中必须透明说明：

1. RAHFL 使用其原有 CIFAR-100 public batches；
2. FedPRIME-C3L-P-D2C-CR 不使用公共图像作为通信输入；
3. 这是“更少外部信息”的设置，不应被表述为与 RAHFL 完全相同的通信资源；
4. 若审稿人要求严格资源等价，可额外添加“双方均不使用 public data”或“双方使用同一同语义 public data”的补充实验。

### 9.3 指标

- avg_acc：所有客户端平均准确率；
- worst_acc：最弱客户端准确率；
- tail_acc：本地非零、但样本较少类别的准确率；
- missing_acc：本地零样本类别准确率，只作边界诊断；
- corruption group accuracy：Noise、Blur、Weather、Digital；
- 多 seed 均值和标准差；
- 训练时间、通信量和稳定性。

### 9.4 最小因果实验链

| 编号 | 方法 | 回答的问题 |
|---|---|---|
| E0 | RAHFL：AugMix + DCL + AsymHFL | 强基线 |
| E1 | PRIME + LogitAvg | PRIME 加简单通信的基础性能，已有 52.10% |
| E2 | PRIME + vanilla SupCon + LogitAvg | 是否只是普通 SupCon 带来收益 |
| E3 | PRIME + C3L + LogitAvg | 损坏一致与类别均衡是否有效 |
| E4 | PRIME + C3L-P + LogitAvg | 概率 tail 补偿是否必要 |
| E5 | PRIME + C3L-P + D2C-CR | 新通信是否有独立增益 |

每一步都要同时报告 avg_acc、worst_acc、tail_acc。只有 E5 优于 E4，D2C-CR 才能算通信创新；只有 E4 优于 E3，概率补偿才值得留下。

---

## 10. 可写入论文的贡献边界

当前可以研究、但不能预先宣称的贡献包括：

1. 设计 PRIME 驱动的损坏一致类别均衡概率式监督对比学习；
2. 设计不依赖参数平均、feature prototype 或 public prior 估计的异构通信形式；
3. 在模型异构、Non-IID 和数据损坏共存时验证 tail 类和最弱客户端的提升。

当前不能写入论文结论的内容包括：

1. “C3L-P 已证明有效”；它还未编码；
2. “D2C-CR 是首次关系蒸馏”；仍存在 FedRAD 等邻近工作；
3. “解决 missing class”；现有理论不支持；
4. “超过 RAHFL”；当前新版尚未运行。

---

## 11. 审核清单

在开始编码前，需要明确确认下列设计决策：

1. 是否接受将旧 D2C 的 public prior debiasing 正式替换为 D2C-CR；
2. 是否接受 C3L-P 使用仅本地的 vMF tail 分布统计；
3. 是否接受论文目标改为提升 Non-IID 的 overall、worst 和 tail 表现，而不承诺 missing class transfer；
4. 是否接受新版主方法不把 CIFAR-100 public logits 作为核心通信资源；
5. 是否先只实现 E2 和 E3，以低风险验证 C3L，再决定是否做 ProCo 式概率补偿和 D2C-CR；
6. 是否在实现前更新 docs/archive/legacy/AGENT.md 中旧 D2C 的约束和项目长期记忆文档。

---

## 12. 建议的编码顺序

审核通过后，推荐严格按风险由低到高推进：

1. 建立 projection head 和 PRIME + vanilla SupCon；
2. 加入 C3L 的类别均衡 anchor / negative 机制；
3. 做 E2、E3 的短轮 debug 和完整 40 轮筛选；
4. 若 C3L 有明确增益，再加入本地 vMF 概率 tail 正对；
5. 若 C3L-P 继续增益，再实现 D2C-CR 的私有验证划分、关系上传和服务器聚合；
6. 最后才跑完整对比、多 seed、RAHFL-val 与论文消融。

任何一步没有独立增益，就停止堆叠后续模块并回到该步骤诊断。

---

## 13. 相关设计稿

- 本方案的文献排重、公式细节和来源见 docs/archive/methods/C3LP_D2C_LITERATURE_AND_METHOD_ZH.md；
- 当前可运行代码的长期结构见 docs/project/ARCHITECTURE.md；
- 已完成实验与结果见 docs/project/PROJECT_STATE.md、docs/project/TODO_NEXT.md、docs/experiments/guides/EXPERIMENT_GUIDE_ZH.md；
- 当前旧 D2C 的历史总结见 deliverables/FedPRIME-D2C_阶段总结与D2C框架说明_2026-06-23.md。

本审阅稿优先级高于旧 D2C 设计，但只有在审核明确通过后才成为编码规范。
