# FCNT / FPER / FRT 数学设计与新颖性审计

日期：2026-08-17

状态：纯理论与文献审计；未修改训练代码，未启动本地或 OpenI 实验

## 1. 结论先行

本报告审计三个试图替代 hard PEW + hard BER 的候选：

```text
FCNT：连续 nuisance 坐标上的联邦类条件传输
FPER：成对恢复干预上的联邦退化效应风险
FRT：公共多视图响应张量分解通信
```

最终判决：

```text
FCNT 作为当前无额外信息协议的核心方法：       NO-GO
FCNT 作为显式提供可信连续元数据的新问题路线：   CONDITIONAL REFRAME ONLY
FPER 作为 observed-only 可部署核心方法：         NO-GO
FPER 作为 paired/clean-source 机制上界：          ORACLE ONLY
FRT 作为下一通信创新：                           NO-GO
三个候选进入最小实现或训练：                     NONE
```

三个对象都能写出数学目标，但没有一个同时满足：

1. 在当前协议中具有可观察且可验证的额外信息；
2. 能保住 BER 弱 `class x environment` 单元收益的非循环保证；
3. 与已有集中式/联邦方法有足够清晰的新颖性间隔；
4. 不复活项目内冻结负结果；
5. 能在一到两个月内形成干净实验归因。

这不是实验失败，而是实现前的论文级否决。hard PEW + hard BER 继续保留为强监督
reference，不因本审计删除或改写其正结果。

## 2. 统一问题与 BER 目标

客户端为 `K`，类别为 `Y=c`，未观察退化变量为 `E`；`E` 可以是连续、离散或复合
退化。客户端 `k` 的类别条件分布为：

\[
P_k(X,E\mid Y=c)
=P(X\mid Y=c,E)\,\pi_{k,c}(E).
\]

类别与退化纠缠来自 `pi_{k,c}` 随客户端和类别变化。对决策对象 `f` 定义潜在单元风险：

\[
R_{k,c,e}(f)
=\mathbb E[\ell(f(X),c)\mid K=k,Y=c,E=e].
\]

hard BER 通过 PEW 伪环境标签显式构造 `class x predicted-environment` 单元，再进行类均匀、
类内环境重加权。忽略其 support shrinkage 细节，BER 的关键能力是：

```text
同一类别内部仍能区分多个退化单元
-> 给弱单元非零训练质量
-> 防止主导退化单元完全掩盖少数单元
```

删除 PEW 后，新候选不能只说“我不需要五类标签”。它必须回答：训练观察到了什么额外量，
这个量为什么能区分至少一部分潜在 `E`，以及它如何给弱单元提供质量下界。

## 3. 候选一：FCNT

### 3.1 名称与额外信息

FCNT 暂称 Federated Class-conditional Nuisance Transport。它假设存在一个冻结映射：

\[
Q=\phi(X)\in\mathbb R^d,
\]

其中 `Q` 是连续 nuisance 坐标。候选来源包括：

```text
设备采集元数据：ISO、曝光、压缩率、估计 SNR 等
冻结的外部图像质量/退化编码器
```

`phi` 不能用 CLE operator/family 或私有类别监督训练，否则只是把 hard PEW 换成 continuous
PEW。

### 3.2 最小成立假设

FCNT 至少需要四个假设：

**F1 nuisance relevance**：`Q` 对不同退化状态具有区分力。形式上，若 `e != e'` 对风险有不同
影响，则 `P(Q|c,e)` 与 `P(Q|c,e')` 不能完全相同。

**F2 semantic exclusion**：`Q` 主要描述 nuisance，而不是直接编码类别语义。一个理想化条件为：

\[
Q\perp S\mid E,
\]

其中 `S` 是语义内容。否则对齐 `Q` 可能同时抹去类别信息。

**F3 overlap**：目标参考分布 `nu_c` 必须被每个有效客户端的 `P_k(Q|c)` 覆盖：

\[
\nu_c\ll P_k(Q\mid c).
\]

没有 overlap 时，重要性权重不存在，通信不能创造本地从未出现的退化支持。

**F4 risk smoothness**：类别条件风险对 `Q` 的变化可控，例如：

\[
|r_{k,c}(q;f)-r_{k,c}(q';f)|
\le L\|q-q'\|,
\]

其中 `r_{k,c}(q;f)=E[ell(f(X),c)|k,c,Q=q]`。

这些是假设，不会因 `phi` 是预训练模型而自动成立。已有工作还表明 foundation model 能预测
spurious attribute，不等于它天然提供无偏语义。

### 3.3 数学对象

客户端的类条件 nuisance 分布为：

\[
\mu_{k,c}=P_k(Q\mid Y=c).
\]

一个自然的联邦参考是 Wasserstein 重心：

\[
\bar\nu_c
=\arg\min_{\nu}
\sum_{k\in\mathcal K_c}a_{k,c}W_2^2(\mu_{k,c},\nu).
\]

若密度比存在，客户端可定义：

\[
w_{k,c}(q)=\frac{d\bar\nu_c}{d\mu_{k,c}}(q),
\]

并优化：

\[
L_{FCNT}
=\frac1{|\mathcal C|}
\sum_c\frac1{|\mathcal K_c|}
\sum_{k\in\mathcal K_c}
\mathbb E_{P_k(X\mid c)}
[w_{k,c}(\phi(X))\ell(f_k(X),c)].
\]

在模型异构场景中，只交换 `mu_{k,c}` 的低维摘要或重心参数；不交换模型参数和主干特征。

### 3.4 它能证明什么

在 F4 下，Wasserstein 距离可以控制同一决策对象在两个 `Q` 分布上的风险差：

\[
|R_{\mu}(f)-R_{\nu}(f)|\le L W_1(\mu,\nu).
\]

因此 FCNT 有条件地控制“可由 `Q` 表示的分布移动”。这是一条标准 OT/domain adaptation
链条，不是新的 BER 保证。

### 3.5 为什么它不能自动保住 BER 收益

假设 `Q` 完美区分潜在单元 `A_e`，且参考分布满足：

\[
\nu_c(A_e)\ge\beta>0\quad\forall e,
\]

则非负损失下有粗界：

\[
\max_e R_{c,e}(f)
\le\frac{R_{\nu_c}(f)}{\beta}.
\]

这说明要保住 BER 的弱单元作用，真正需要的是每个潜在单元的质量下界 `beta`。普通
Wasserstein 重心只代表客户端分布的几何中心，并不保证给稀有区域非零下界；多数客户端都缺少
某个退化时，重心也可以忽略它。

但如果不知道 `E` 或其区域 `A_e`，算法又无法施加 `nu_c(A_e)>=beta`。因此：

```text
连续坐标解决“组合数量很多”
!= 自动解决“弱退化单元有训练质量”
```

这使 FCNT 无法在当前无额外 nuisance 保障的协议中给出 BER 式理论链条。

### 3.6 外部碰撞

1. [Class-Conditional Distribution Balancing](https://arxiv.org/abs/2504.17314) 已把
   spurious correlation 表述为类条件分布失配，并通过样本重加权实现平衡；
2. [FG-CCDB, ICLR 2026](https://openreview.net/forum?id=NEFldJX4zb) 已进一步处理多模态
   类条件分布和细粒度重加权；
3. [Class-conditioned Domain Generalization via Wasserstein DRO](https://arxiv.org/abs/2109.03676)
   已在类条件分布的 Wasserstein 重心周围构造最坏分布风险；
4. [FedWaD, ICLR 2024](https://openreview.net/pdf?id=rsg1mvUahT) 已给出联邦 Wasserstein
   距离计算；
5. [FedDaDiL](https://arxiv.org/abs/2309.07670) 已在联邦域适应中学习 Wasserstein
   分布字典和重心；
6. [SLOT-Align, 2026](https://arxiv.org/abs/2606.16655) 与 FCNT 结构最接近：共享冻结编码器、
   紧凑特征统计、Bures-Wasserstein 重心和客户端本地传输对齐；
7. [Group Robust Classification Without Any Group Information](https://arxiv.org/abs/2310.18555)
   已使用预训练自监督模型提取缺失 bias 信息并进行 robust learning；
8. [Prompting for Robustness, ICML 2024](https://openreview.net/forum?id=fdroxYsgzQ) 已用
   foundation model 预测 spurious attribute 后做组平衡。

FCNT 的剩余差异只是“连续 nuisance 坐标 + class-conditional + iterative model-HFL”。这是合理
组合，但不足以让当前公式成为新的核心方法对象。

### 3.7 与项目冻结方法的边界

- 相比 hard PEW：从人工六类分类改成连续 `Q`，但仍依赖外部 witness；
- 相比 continuous witness：外部冻结 `phi` 和 OT 替代手工22维统计与协方差惩罚，但信息来源
  仍是连续 nuisance proxy；
- 相比 CDep/CIRCE：FCNT 是分布重加权，不是条件独立正则；但不能因此获得新颖性；
- 相比纯 client-mixture contrast：FCNT 增加了 `Q`，避免只把 client 当 environment；
- 不得通过调 OT 正则、重心权重或 `Q` 维度复活为当前主线。

### 3.8 FCNT 判决

```text
数学成立性：       条件成立
额外信息透明度：   差；当前协议没有可信 q 来源
BER 收益链条：     不成立；无潜在单元质量下界
外部新颖性：       低；CCDB + class-WDRO + federated OT/SLOT-Align
实现成本：         中高；encoder、分布摘要、OT、权重稳定性
实验归因：         差；encoder 质量与 OT 目标耦合
论文价值：         当前低
决策：             CORE NO-GO；仅允许作为改变问题设定后的路线
```

若未来有真实设备连续元数据，FCNT 可以成为一个新应用问题的基线，但那是显式修改任务假设，
不是当前 taxonomy-free 方法的自然延续。

## 4. 候选二：FPER

### 4.1 名称与额外信息

FPER 暂称 Federated Paired Effect Risk。它假设每个观察图像 `X` 有一个恢复或参考版本：

\[
X^- = g(X),
\]

或直接有同内容的 paired capture。`g` 不能只在原损坏上继续叠加新损坏，否则会回到 C3R/CCRE
已经暴露的问题。

### 4.2 最小成立假设

**P1 label preservation**：

\[
Y(X^-)=Y(X).
\]

**P2 nuisance contraction**：`g` 确实减少原 nuisance，而不是产生另一种 artifact。

**P3 semantic distortion bound**：若潜在干净图为 `X^0`，需要：

\[
d(X^-,X^0)\le\epsilon_g.
\]

**P4 coverage**：上述性质必须对所有重要退化和复合退化成立，而不是只对恢复模型训练过的类型
成立。

当前 observed-only 协议无法在训练时验证 P1--P4。使用生成器保存的 clean parent 可以验证，
但会向方法额外开放当前基线没有的信息。

### 4.3 数学对象

定义从观察图到恢复图的路径：

\[
X(t)=(1-t)X+tX^-,\qquad t\in[0,1].
\]

成对效应可以写成：

\[
\Delta_f(X)
=\ell(f(X),Y)-\ell(f(X^-),Y),
\]

或路径敏感度：

\[
I_f(X)
=\int_0^1
\left\|
\frac{d\,p_f(X(t))}{dt}
\right\|_2dt.
\]

一个完整本地目标可写成：

\[
L_{FPER}
=E\left[
\ell(f(X),Y)
+\lambda_{pair}D(p_f(X),p_f(X^-))
+\lambda_{path}\sup_{t\in[0,1]}\ell(f(X(t)),Y)
\right].
\]

联邦版本只能额外聚合每个类别的 `Delta/I` 分布摘要，形成客户端归一化系数；主模型仍使用原
AsymHFL 或无通信本地训练。

线性路径 `X(t)` 还可能离开自然图像流形；将它替换成 diffusion/restoration trajectory 会降低
该问题，却进一步增加生成模型假设和计算成本。

### 4.4 理论链条与其局限

由三角不等式可写出：

\[
\ell(f(X),Y)
\le\ell(f(X^-),Y)
+|\ell(f(X),Y)-\ell(f(X^-),Y)|.
\]

若 `ell o f` 对输入是 `L_f`-Lipschitz，且 P3 成立：

\[
\ell(f(X^-),Y)
\le\ell(f(X^0),Y)+L_f\epsilon_g.
\]

因此 paired consistency 加有效 restoration 可以控制 corrupted risk。但这个界的关键完全是
`epsilon_g`；如果恢复改变语义或留下原 shortcut，目标只会强迫模型对恢复 artifact 一致。

此外，逐样本一致性不等价于 BER 的弱单元平衡。若少数退化样本本来就少，它们在期望中仍可能
被多数样本掩盖。再加入 loss-tail、伪组或 worst effect，又分别回到 CVaR/JTT、C3R 或 GroupDRO。

### 4.5 外部与内部碰撞

1. [Counterfactual Invariance, NeurIPS 2021](https://arxiv.org/abs/2106.00545) 已形式化对
   nuisance 干预保持预测不变，并连接到域外风险；
2. [Counterfactual and Invariant Data Generation, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Chang_Towards_Robust_Classification_Model_by_Counterfactual_and_Invariant_Data_Generation_CVPR_2021_paper.html)
   已通过修改非因果特征生成标签不变图像并训练鲁棒分类器；
3. [Counterfactual Alignment, TMLR 2025](https://openreview.net/forum?id=Utjw2z1ale) 已用
   classifier response to counterfactual images 检测和量化 spurious correlation；
4. [Debiasing Counterfactuals in the Presence of Spurious Correlations](https://arxiv.org/abs/2308.10984)
   已结合 counterfactual image generation 与 robust optimization；
5. NURD 已定义 nuisance-randomized distribution，并给出对变化 nuisance-label 关系鲁棒的
   表征目标：[Puli et al.](https://arxiv.org/abs/2107.00520)；
6. 项目内 C3R 已直接审计 AugMix 前后类条件退化遗憾并失败；
7. 项目内 FedCISA/CCR 已明确指出“对已损坏图像继续组合不等于替换 base corruption”，
   clean rerender 只能作为 oracle；
8. 普通 paired consistency、UDA 和路径平滑均是成熟组件。

FPER 的 `g` 若是 blind restorer，主要新增的是外部恢复器；若 `g` 使用 clean parent，则主要新增的
是成对数据。两者都不是当前 HFL 数学对象本身的创新；客户端之间聚合效应摘要也不是解决该问题
所必需，因此其 FL-specific contribution 很弱。

### 4.6 FPER 判决

```text
数学成立性：       依赖无法在 observed-only 中验证的 restoration oracle
不需要五类标签：   是，但换成更强 paired/restoration 假设
BER 收益链条：     不成立；一致性不保证弱单元质量
外部新颖性：       低；counterfactual invariance/generation/alignment 已成熟
实现成本：         高；恢复器推理、路径视图、缓存与训练
实验归因：         很差；恢复质量、额外视图和目标耦合
论文价值：         当前低
决策：             DEPLOYABLE NO-GO；paired/clean-source ORACLE ONLY
```

不得把 clean parent 开放给候选、却不给 matched control，然后声称超过 RAHFL。若未来研究
burst/paired acquisition，应将其作为新的数据可用性设定重新立题。

## 5. 候选三：FRT

### 5.1 名称与观测张量

FRT 暂称 Federated Response Tensor Factorization。对公共图像 `U_n` 和随机视图 `a_t`，客户端
上传公共输出空间中的响应变化：

\[
D_{k,t,n,c}
=\widetilde z_{k,c}(a_t(U_n))
-\widetilde z_{k,c}(U_n),
\]

其中 `tilde z` 必须先做模型内温度或行标准化，避免不同 backbone 的 logit 尺度直接混入。

### 5.2 假设分解

候选希望写成：

\[
\mathcal D=\mathcal S+\mathcal B+\mathcal N,
\]

其中：

```text
S：跨客户端共享、低秩、被解释为语义稳定响应
B：客户端特有或稀疏、被解释为 shortcut 响应
N：随机噪声
```

一个典型优化是：

\[
\min_{\mathcal S,\mathcal B}
\frac12\|\mathcal D-\mathcal S-\mathcal B\|_F^2
+\lambda_*\|\mathcal S\|_*
+\lambda_B\|\mathcal B\|_{2,1}.
\]

服务器只将 `S` 生成的 teacher 返回给客户端。

### 5.3 可识别性要求

低秩加稀疏分解不是自动唯一。Robust PCA 需要低秩子空间与稀疏支持不相干，并限制秩与稀疏度：
[Candès et al.](https://arxiv.org/abs/0912.3599)。CP/tensor 分解同样需要 Kruskal-rank 等条件：
[Bhaskara et al., COLT 2014](https://proceedings.mlr.press/v35/bhaskara14a.html)。

当前 CLE-HFL 恰好不支持这些解释：

1. 每个客户端都存在 shortcut，`B` 在 client mode 不是稀疏异常；
2. 相同架构不成立，共享语义响应会随容量、校准和训练状态变化，`S` 不必 client-invariant；
3. 某些类别的主导 family 在客户端间相同，shortcut 也可能落入共享低秩部分；
4. 随机 AugMix view 不保证只改变 nuisance；
5. 公共 CIFAR-100 与私有 CIFAR-10 跨域，公共响应分解不保证对应私有弱 cell；
6. 只有四个客户端，client mode 的观测秩和统计稳定性很有限。

更根本地，对任意张量 `H` 都有：

\[
\mathcal D=(\mathcal S+\mathcal H)+(\mathcal B-\mathcal H).
\]

若没有独立的低秩、稀疏或非高斯多视图条件，`S/B` 的“语义/shortcut”命名没有可识别内容。

### 5.4 为什么不能保证保住 BER

即使 FRT 完美去除了公共张量中的 client-specific component，也只说明公共 teacher 更稳定；它
没有给私有训练集中少数 `class x environment` 单元增加权重。BER 的主要正增益已经由本地
重加权归因，而非通信。

因此链条中缺少关键一步：

```text
公共响应分解正确
  -/-> 私有单样本环境可区分
  -/-> 私有弱单元获得训练质量
  -/-> BER 的 WCCA/CFG 收益被保留
```

### 5.5 外部与内部碰撞

1. [FCCL+](https://arxiv.org/abs/2309.16286) 已在异构 FL 中用跨客户端公共数据的
   cross-correlation matrix 和 instance-similarity distribution 对齐 logits/features；
2. FedMD/FedDF/FedHPL 等大量 HFL 方法已交换或聚合公共 logits；
3. 低秩/稀疏分解、tensor nuclear norm 和多视图共识本身均是成熟工具；
4. 项目内 CCAD 已用 clean/augmented public views 做 corruption-consistent distillation；
5. IRD/PCCD 已做公共 residual/relation distillation；
6. FedCIS 已尝试跨客户端恢复共享类别敏感子空间，离线可识别性失败；
7. EBST 已传递环境平衡结构，未形成稳定通信增益；
8. PRAC/FedCARA 已表明复杂 teacher 选择或类别路由不稳定优于 AsymHFL。

把上述公共多视图响应堆成张量并不会消除这些证据。

### 5.6 FRT 判决

```text
数学成立性：       分解可写，但语义解释不可识别
不需要五类标签：   是
BER 收益链条：     缺失；只改 teacher，不处理本地弱单元
外部新颖性：       低到中；具体张量实例可能不同，核心工具与通信结构拥挤
实现成本：         高；多视图 logits、分解求解、温度校准
实验归因：         极差；视图、分解、teacher 和架构混淆
论文价值：         低
决策：             NO-GO；不做离线或训练实现
```

## 6. 三候选横向评分

评分为 1（最差）到 5（最好）；成本为 1（最低）到 5（最高）。

| 候选 | 新颖性 | 理论合理性 | 信息可验证性 | BER链条 | 实现成本 | 实验归因 | 论文价值 | 决策 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| FCNT | 1 | 3（条件式） | 1 | 1 | 3 | 2 | 1 | `CORE NO-GO` |
| FPER | 1 | 3（oracle式） | 1 | 1 | 4 | 1 | 1 | `DEPLOYABLE NO-GO` |
| FRT | 2 | 1 | 1 | 1 | 4 | 1 | 1 | `NO-GO` |

若只问“哪一个最接近可继续讨论”：

```text
P1 FCNT：仅当愿意把可信连续设备元数据写成新的任务假设
P2 FPER：仅作 paired/clean-source oracle 机制研究
P3 FRT：直接停止
```

这不是实现优先级。当前实现优先级为：

```text
FCNT 0
FPER 0
FRT  0
```

## 7. 逐项区别于要求中的方法

| 方法 | 核心对象 | 与三个候选的关系 |
|---|---|---|
| PEW+BER | 人工 taxonomy 预测组 + 类组重加权 | FCNT换连续 proxy；FPER/FRT不显式分组，但均无 BER 质量下界 |
| PIE/MPIE | 公共合成干预学习 latent/severity | FRT仍依赖公共多视图；FPER改成恢复干预但需要更强 oracle |
| C3R | AugMix 前后损失遗憾 | FPER若只用 paired loss difference 就直接退化为其近邻 |
| CRSR | 类条件预测残差谱 | FRT用公共多视图张量，但共享/偏置分解同样缺少 cell relevance |
| LCC/GRASP/GoG | 梯度伪组/梯度图 | 三候选不使用逐样本梯度聚类 |
| GroupDRO | 已知组最大风险 | FCNT的OT平均和FPER一致性均不是最大组风险，但“不相同”不等于新颖 |
| CVaR/JTT | 高损失尾部 | 三候选不得靠加入tail mining修复弱单元，否则回到成熟路线 |
| CCDB/FG-CCDB | 类条件分布匹配与重加权 | FCNT直接相邻，是其最主要新颖性碰撞 |
| CCAD/IRD | 公共多视图/残差蒸馏 | FRT直接相邻；项目内已有负证据 |
| FedCIS | 跨客户端共享敏感子空间 | FRT重复“共享部分即语义”的不可识别风险 |

## 8. 为什么三个候选一起失败

三者分别尝试引入：

```text
FCNT：外部连续 nuisance 表征
FPER：成对 nuisance 干预
FRT：跨客户端多视图响应
```

但各自的额外信息都没有免费获得：

- FCNT 必须先证明 `phi` 是 nuisance-sufficient 且不携带语义；
- FPER 必须先证明 restoration 是 label-preserving 且真正移除 base nuisance；
- FRT 必须先证明共享/偏置张量满足唯一分解条件并与私有弱 cell 相关。

一旦把这些条件显式写出，就会发现：

1. 条件本身比“五类PEW”更强或更难验证；
2. 即使成立，也已有非常接近的成熟方法；
3. 它们控制平均分布或预测一致性，却没有 BER 的弱单元质量下界。

所以不能用“支持复合退化”作为放行理由。能表示无限组合与能识别弱组合是两件不同的事。

## 9. 最终研究决策

### 9.1 当前保留

```text
CLE-HFL：受控 model-HFL benchmark extension
hard PEW + hard BER：taxonomy-assisted strong reference
AsymHFL-val：采用的稳定通信骨架
```

### 9.2 当前停止

```text
FCNT 的 encoder/OT 实现或调研式跑分
FPER 的 restoration/clean-pair runner
FRT 的 public-response tensor audit 或通信 runner
用风险尾部、梯度聚类、公共多视图或 client variance 修补三者
```

### 9.3 论文层面的含义

三候选审计没有找到可进入实现的新核心方法。继续在当前信息协议中追求“完全 taxonomy-free 且
保留 BER 弱组收益”，已受到可识别性和新颖性的双重约束。下一决策应是战略选择，而不是第四次
盲目造模块：

1. 明确接受一种现实可获取的额外信息，重新定义任务；或
2. 保留 PEW+BER，转向较保守的 empirical/benchmark 论文并补强证据；或
3. 停止 CLE 方法主线，转回另一条更成熟且期限可控的投稿路线。

在用户选择上述路线前，不实现任何本报告候选，也不启动新实验。
