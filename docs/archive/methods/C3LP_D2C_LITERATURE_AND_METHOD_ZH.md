# FedPRIME：C3L-P 与 D2C 重构的文献排重和正式方法定义

**状态：** 研究设计稿，尚未实现、尚未验证。  
**日期：** 2026-06-23  
**目的：** 为“PRIME + 新监督对比模块 + 新通信模块”提供可证伪的理论定义、文献边界与最小实验路线。

---

## 1. 结论先行

当前 D2C-v1 使用跨域 CIFAR-100 public logits 预测 CIFAR-10 私有类别先验，并据此调整 logits。已有结果表明：即便换成真实 private prior，Oracle D2C 仍未优于普通 LogitAvg。因此，**D2C-v1 不再作为主创新继续调参。**

拟议的新框架为：

\[
\text{FedPRIME} = \text{PRIME} + \textbf{C}^{3}\textbf{L-P} + \textbf{D}^{2}\textbf{C-CR}.
\]

其中：

- **C3L-P**：Corruption-Consistent, Class-Balanced, Probabilistic Contrastive Learning，损坏一致、类别均衡、概率式监督对比学习。
- **D2C-CR**：Distribution-aware Decision Consensus via Class Relations，基于类别决策关系的分布感知共识通信；这是待验证的候选通信机制，而非已确立贡献。

本方法不主张凭空补全客户端完全没有样本的类别。它的可验证目标是：在模型异构、Non-IID 标签偏斜和训练/测试损坏同时存在时，提高整体、最弱客户端和已出现 tail 类的性能。

---

## 2. 问题设定与研究边界

设有 \(K\) 个客户端，均解决 \(C\) 类分类任务。客户端 \(k\) 的私有训练集为：

\[
\mathcal D_k=\{(x_i,y_i)\}_{i=1}^{N_k}, \qquad y_i\in\{1,\ldots,C\}.
\]

不同客户端可以采用不同骨干网络，例如 ResNet10、ResNet12、ShuffleNet 和 MobileNetV2。因此参数、层数、特征维度均不要求一致，不能直接 FedAvg 参数。

当前实验包含：

- **数据损坏：** RAHFL-style random corruption CIFAR-10；
- **数据异构：** Dirichlet label skew，主设置 \(\alpha=0.5\)；
- **模型异构：** 四种不同 CNN；
- **固定分区：** 所有方法读取同一 partition 文件；
- **公共数据：** CIFAR-100 只保留为 RAHFL / LogitAvg 的公平对照或诊断资源，不再用于推断 private prior。

### 2.1 不做的主张

下列命题不作为本文方法的承诺：

1. 通过其他客户端的 soft logits 恢复本地从未出现类别的视觉语义；
2. 在没有任何目标类样本、生成样本或同语义公共样本时，让 missing-class accuracy 必然非零；
3. 因为传输了 logits 就自动得到类别知识迁移。

若客户端没有类别 \(c\) 的任何本地样本，C3L-P 不构造该类的本地统计量，D2C-CR 也不会对该类施加伪监督。missing accuracy 仍应报告，但它是边界诊断而不是 headline 指标。

---

## 3. 文献排重

本节是面向方法设计的初步排重，不等同于投稿前的穷尽式综述或专利检索。结论重点是：哪些机制可借鉴，哪些会与已有路线重合，哪些不应使用。

| 工作 | 已解决的问题 | 与 C3L-P / D2C-CR 的关系 | 处理结论 |
|---|---|---|---|
| [SupCon, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html) | 多正样本监督对比，类内拉近、类间分离 | C3L-P 的基础形式 | 必须作为直接 baseline，而非创新本身 |
| [PaCo, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Cui_Parametric_Contrastive_Learning_ICCV_2021_paper.html) | 长尾下 SupCon 偏向高频类；使用可学习类中心 | 说明长尾对比需要重平衡 | 不采用 class center，避免 prototype 路线 |
| [BCL, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Zhu_Balanced_Contrastive_Learning_for_Long-Tailed_Visual_Recognition_CVPR_2022_paper.html) | class averaging 平衡负类梯度；class complement 覆盖所有类 | C3L-P 借鉴类别均衡梯度 | 采用 class averaging 思想；不采用全局类别中心补全 |
| [TSC, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Targeted_Supervised_Contrastive_Learning_for_Long-Tailed_Recognition_CVPR_2022_paper.html) | 通过均匀 target 保持类别几何均匀 | 证明 tail 类特征空间会被 head 类破坏 | 不使用预设 target / center |
| [SBCL, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Hou_Subclass-balancing_Contrastive_Learning_for_Long-tailed_Recognition_ICCV_2023_paper.html) | 将 head 类划成子类，保持内部语义结构 | 说明简单重加权会损失 head 类子结构 | 不作为 MVP；可作为后续增强，不在首版引入动态聚类 |
| [ProCo, 2024](https://arxiv.org/abs/2403.06726) | 用在线 vMF 类分布构造期望对比对，缓解小 batch / tail 正对缺乏 | C3L-P 的概率式 tail 正对来源 | 仅使用本地、临时、不可通信的类分布统计；需要单独验证收益 |
| [FedProc, 2021](https://arxiv.org/abs/2109.12273) | 全局 class prototype 监督对比，处理 Non-IID | 与跨客户端共享特征中心相近 | 明确排除：C3L-P 不上传或聚合 prototype |
| [FedMD, 2019](https://arxiv.org/abs/1910.03581) | 在公共数据上聚合异构模型预测 | 当前 LogitAvg / 旧 D2C 的通信祖先 | 作为异构 public-logit baseline |
| [FedRAD, 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10385861/) | 使用 relational KD 与熵自适应权重缓解异构 FL 遗忘 | 与关系蒸馏存在直接邻近 | D2C-CR 新颖性风险高，必须避免写成泛化 relation KD |

### 3.1 C3L-P 的新颖性边界

C3L-P 不应被宣称为“首次使用监督对比学习处理长尾”或“首次使用概率对比学习”。这些问题已有 BCL、TSC、SBCL、ProCo 等工作。

可验证的差异在于它针对以下**耦合稀缺问题**：

> 在受损 Non-IID 客户端中，tail 类本来就缺少同类正对；强 PRIME 视图中又有一部分不可靠。普通 SupCon 会同时遭遇类别梯度失衡与可靠正对不足。

对应地，C3L-P 同时包含：

1. PRIME 多视图的损坏一致性；
2. 按本地类别覆盖均衡的优化；
3. 仅对出现但稀少的类别进行本地概率正对补偿；
4. 基于多视图分歧的 stop-gradient 可靠性加权。

这是一种有意义的组合创新，但能否称为论文创新取决于后续实验是否证明三个部分缺一不可，以及是否不存在已有 corruption-aware probabilistic balanced SupCon for heterogeneous FL 的等价方案。

### 3.2 D2C-CR 的新颖性风险

D2C-CR 交换类别条件下的输出关系，避免共享模型参数、特征或 prototype；这一点适合模型异构。但 relation distillation 本身已有 FedRAD 等先例。

因此当前策略是：

- 将 D2C-CR 定义为候选机制；
- 后续必须检索 class-conditional confusion / posterior relation communication 是否已有等价 FL 方案；
- 只有在它同时满足“无公共图像、无特征交换、类别条件可靠聚合、对 PRIME 视图一致性进行约束”且实验显著有效时，才将它升级为主创新；
- 若排重失败，则把论文重点放在 C3L-P，本通信层只作为框架适配实现。

---

## 4. C3L-P 的正式定义

### 4.1 PRIME 多视图与表示

对私有样本 \((x_i,y_i)\)，PRIME 产生三个视图：

\[
x_i^{(0)}=x_i^{clean}, \qquad x_i^{(1)}=\operatorname{PRIME}_1(x_i), \qquad x_i^{(2)}=\operatorname{PRIME}_2(x_i).
\]

客户端 \(k\) 的模型由私有骨干 \(f_k\)、私有 projection head \(g_k\) 和分类器 \(a_k\) 构成：

\[
h_i^{(v)}=\frac{g_k(f_k(x_i^{(v)}))}{\|g_k(f_k(x_i^{(v)}))\|_2}, \qquad
p_i^{(v)}=\operatorname{softmax}\left(a_k(f_k(x_i^{(v)}))/T_d\right).
\]

\(g_k\) 仅用于本地训练和训练期统计，测试时可丢弃；不要求不同客户端的 \(h_i^{(v)}\) 同维，也绝不上传它。

### 4.2 损坏视图可靠性

定义三视图预测的 Jensen-Shannon 分歧：

\[
\bar p_i=\frac{1}{3}\sum_{v=0}^{2}p_i^{(v)}, \qquad
d_i=\frac{1}{3}\sum_{v=0}^{2}\operatorname{KL}\left(p_i^{(v)}\Vert\bar p_i\right).
\]

在 warmup 结束后，给样本赋予停止梯度的可靠性权重：

\[
r_i=\operatorname{clip}\left(\exp\left[-\operatorname{sg}(d_i)/\rho\right],r_{min},1\right).
\]

这里 \(\operatorname{sg}\) 表示 stop-gradient。它不鼓励模型通过直接操纵 \(r_i\) 来降低损失；它只在本次更新中决定“这个 PRIME 强视图作为正对有多可信”。warmup 前固定 \(r_i=1\)，避免随机初始化阶段的错误预测过早过滤数据。

### 4.3 类别均衡的监督对比

令 \(n_{k,c}\) 是客户端 \(k\) 内类别 \(c\) 的样本数。对出现过的类别使用 effective-number 权重：

\[
w_{k,c}=\frac{1-\beta}{1-\beta^{n_{k,c}}}, \qquad n_{k,c}>0.
\]

其目的是让 tail 类的 anchor 产生更高损失权重，而不是对不存在的类别制造样本。对一个 anchor \(i\)，令 \(\mathcal P_i\) 是同标签、不同样本或不同视图的正对集合，\(\mathcal A_i\) 是其余候选表示。定义类别均衡的对比项：

\[
\ell_i^{bal}=-\frac{1}{|\mathcal P_i|}
\sum_{p\in\mathcal P_i}
\log
\frac{\exp(h_i^\top h_p/\tau)}
{\sum_{a\in\mathcal A_i}\widetilde w_{k,y_a}\exp(h_i^\top h_a/\tau)}.
\]

\(\widetilde w\) 是在当前 batch 内归一化后的类别权重。最终按“先类平均、再样本平均”的方式计算，以阻止 head 类仅因样本数量多而主导梯度：

\[
\mathcal L_{bal}=
\frac{1}{|\mathcal C_B|}
\sum_{c\in\mathcal C_B}
\frac{1}{|\mathcal I_c|}
\sum_{i\in\mathcal I_c}r_i\ell_i^{bal}.
\]

### 4.4 概率式 tail 正对补偿

普通 SupCon 在 tail 类每个 batch 只有一个样本时几乎没有额外同类正对。为此，只在客户端本地、仅对已出现的 tail 类 \(c\) 维护可靠 clean-view 表示的指数滑动统计：

\[
\mu_{k,c}=\operatorname{norm}\left(
\operatorname{EMA}\left[
\frac{\sum_{i:y_i=c}r_i h_i^{(0)}}{\sum_{i:y_i=c}r_i+\epsilon}
\right]\right).
\]

再以其平均方向与离散程度定义球面上的本地 von Mises-Fisher 分布：

\[
q_{k,c}(h)=\operatorname{vMF}(h;\mu_{k,c},\kappa_{k,c}).
\]

对样本不足、但 \(n_{k,c}\ge n_{min}\) 的类别，从 \(q_{k,c}\) 构造数量受上限控制的虚拟正对 \(\widetilde{\mathcal P}_{k,c}\)。这些向量：

- 是本地 projection space 内的统计量，不是图片；
- 不上传、不跨客户端聚合；
- 在反向传播时停止梯度；
- 仅补充已出现 tail 类，绝不为 \(n_{k,c}=0\) 的 missing class 生成知识。

将 \(\widetilde{\mathcal P}_{k,c}\) 加入 \(\mathcal P_i\)，得到 \(\mathcal L_{C3L-P}\)。概率补偿应在 warmup 后才启用；若其不优于无概率补偿的 C3L，则它应被删除，而不是为了“模块完整”保留。

### 4.5 本地总目标

\[
\mathcal L_k=
\mathcal L_{CE}
+\lambda_{jsd}\mathcal L_{JSD}
+\lambda_c\mathcal L_{C3L-P}
+\lambda_d\mathcal L_{D2C-CR}.
\]

首个验证版本应设置 \(\lambda_d=0\)，先隔离检验 C3L-P 是否超过 PRIME + LogitAvg。

---

## 5. D2C-CR 的候选正式定义

### 5.1 核心原则

D2C-CR 不使用跨域 public logits 推断 private prior。它只在共享的 \(C\) 维类别输出空间内传输“已观测类别的决策关系”。该空间对所有异构模型相同，即使其内部特征和参数完全不同。

### 5.2 本地类别条件关系

客户端将训练集划出固定的小验证子集 \(\mathcal V_k\)，不得使用全局测试集。对于本地存在类别 \(c\)，计算：

\[
R_{k,c}=
\frac{1}{Z_{k,c}}
\sum_{(x_i,y_i=c)\in\mathcal V_k}
r_i\cdot\frac{p_i^{(0)}+p_i^{(1)}+p_i^{(2)}}{3}.
\]

\(R_{k,c}\in\mathbb R^C\) 描述“真实为 \(c\) 时，此模型对各类别的平均软决策关系”。只上传有充分验证样本且可靠性达到阈值的行，以及该行的支持数和可靠性标量。

### 5.3 服务器可靠聚合

令 \(\mathcal K_c\) 是上传类别 \(c\) 行的客户端集合，\(a_{k,c}\) 是支持数、私有验证准确率和视图一致性的组合权重。服务器计算：

\[
R_{g,c}=
\frac{\sum_{k\in\mathcal K_c}a_{k,c}R_{k,c}}
{\sum_{k\in\mathcal K_c}a_{k,c}+\epsilon}.
\]

服务器回传 \(\{R_{g,c}\}_{c=1}^{C}\)。通信量为 \(C\times C\) 个浮点数及 \(C\) 个标量；CIFAR-10 中仅为 100 个关系值，且不随骨干模型尺寸增长。

### 5.4 本地关系蒸馏

对于本地样本 \((x_i,y_i)\)，仅当 \(R_{g,y_i}\) 存在时使用：

\[
\mathcal L_{D2C-CR}
=\frac{1}{|B|}\sum_{i\in B}
\mathbb 1[R_{g,y_i}\ \text{available}]
\operatorname{KL}\left(
\operatorname{sg}(R_{g,y_i})\Vert p_i^{(0)}
\right).
\]

为防止软关系覆盖真实标签，可使用有界 teacher：

\[
\widetilde R_{g,c}=(1-\delta)e_c+\delta R_{g,c}, \qquad 0<\delta<1.
\]

### 5.5 已知限制与新颖性门槛

D2C-CR 只应被解释为“对已有类别的跨模型决策边界校准”。它并不会，也不应被声称能，恢复完全缺失类别的视觉特征。

此外，FedRAD 等方法已使用 relational KD。因此投稿前必须证实：D2C-CR 与其不同之处不是名字，而是类别条件输出关系、无公共图像、无特征/参数共享、无全局教师模型和 PRIME 可靠性驱动的行聚合。若无法证明差异或实验增益很小，应放弃 D2C-CR，保留 C3L-P 作为主要研究贡献。

---

## 6. 一次通信轮的概念流程

1. 客户端用 PRIME 生成 clean、view-1、view-2。
2. 各异构模型在本地计算 CE、JSD 和 C3L-P；projection head 不离开客户端。
3. warmup 后，对出现的 tail 类启用本地概率正对补偿。
4. 客户端在私有验证子集估计类别关系 \(R_{k,c}\)，上传关系行与可靠性标量。
5. 服务器按类别可靠聚合，回传 \(R_{g,c}\)。
6. 客户端仅在自己拥有的类别上使用 D2C-CR 蒸馏，不使用全局测试标签，不伪造 missing-class target。

---

## 7. 最小实验路线与判定门槛

不应一开始就对所有超参数做大网格搜索。先完成以下因果链：

| 编号 | 方法 | 目的 |
|---|---|---|
| E0 | RAHFL：AugMix + DCL + AsymHFL | 强基线，已有 seed-0 结果 56.41% |
| E1 | PRIME + LogitAvg | 已有通信底座，结果 52.10% |
| E2 | PRIME + vanilla SupCon + LogitAvg | 证明收益并非随便加 SupCon |
| E3 | PRIME + C3L（无概率补偿）+ LogitAvg | 验证损坏一致与类别均衡是否有效 |
| E4 | PRIME + C3L-P + LogitAvg | 验证概率 tail 补偿是否有额外价值 |
| E5 | PRIME + C3L-P + D2C-CR | 检验新通信层是否产生独立增益 |

所有 E1--E5 必须使用与 E0 相同的数据、固定 partition、模型集合、轮数、优化器和评估流程。

主要指标：

- avg_acc：总体异构客户端平均准确率；
- worst_acc：最弱客户端准确率，防止只牺牲小模型换取平均值；
- tail_acc：本地样本较少但非零类别的准确率；
- missing_acc：仅作边界诊断；
- 每轮稳定性：local loss、JSD、对比损失、关系蒸馏损失及非有限值检查。

筛选原则：

1. E3 必须同时优于 E1 和 E2，C3L 才有保留价值；
2. E4 必须优于 E3，概率补偿才有保留价值；
3. E5 必须优于 E4，D2C-CR 才能称为通信贡献；
4. 达到单 seed 明显超过 RAHFL 的信号后，再跑 seeds 0/1/2 和严格无测试泄漏的 RAHFL-val；
5. 任何只提高 avg_acc、却显著损伤 worst_acc 或 tail 指标的方案，不应作为最终主方法。

---

## 8. 目前的研究表述

可以向导师准确表述为：

> 已完成 PRIME、RAHFL 与 D2C 的统一对照，发现基于跨域 public logits 的 prior 去偏无法带来有效增益，Oracle prior 也不能修复该问题。下一步不再把目标设为“补全缺失类别”，而是基于长尾监督对比学习的后续工作，设计 PRIME 下的损坏一致、类别均衡和概率式 tail 正对学习；通信端拟由 prior 预测改为类别条件决策关系共识，并先做严格排重和分阶段验证。

这是一条可被实验否定、也能形成论文叙事的路线，而不是把多个现成模块机械拼接。
