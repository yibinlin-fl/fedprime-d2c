# 模型异构联邦安全持续测试时适应：可识别性与新颖性审计

日期：2026-08-17

状态：纯理论与原始论文审计；未修改训练代码，未启动本地或 OpenI 实验

## 1. 结论先行

本报告审计以下候选新课题：

```text
Safe Collaborative Continual Test-Time Adaptation
under Model-Heterogeneous Federated Learning

模型异构联邦学习中的安全协作式持续测试时适应
```

候选希望在客户端模型结构不同、部署流无标签、损坏类型和组合随时间变化时，判断：

```text
接受其他客户端知识
只做本地适应
冻结当前模型
```

并保证协作不会提高目标客户端风险。

最终判决：

```text
完全无标签目标流下，协作收益符号可识别：          NO
source-audit / 公共数据 / 多模型输出自动解决该问题： NO
当前“安全且无标签”的强问题表述：                  THEORY NO-GO
把 AETTA/风险监控接到 FedCTTA/AsymHFL 上：          NOVELTY NO-GO
仅以 model heterogeneity 扩展现有 FTTA：             CORE-METHOD NO-GO
稀疏、无偏、延迟目标标签版本：                      CONDITIONAL REFRAME ONLY
当前进入实现或实验：                                NONE
```

关键原因不是缺少一个更好的相似度函数，而是目标风险包含不可观察的
`P_t(Y|X)`。在目标标签完全不可见时，可以构造两个具有完全相同源数据、目标图像流、
模型输出、公共响应和通信消息的世界，使同一次协作在一个世界有益、在另一个世界有害。

即使放弃严格保证，方法空间也已被 FedTHE、ATP、FedICON、FedTSA、CoLA、AETTA、
FedCTTA、Latte 和 TTA risk monitoring 从“联邦测试时适应、动态协作、无标签性能估计、
风险报警”多个方向包围。只增加四个异构 backbone 不足以形成新学习原理。

因此本候选不进入实现。若用户愿意把问题改为“具有稀疏且无偏的延迟任务标签”，应作为
一个全新课题重新查重，不能把它称为当前 fully-unsupervised FTTA 的实现细节。

## 2. 精确问题定义

客户端集合为 `k in {1,...,K}`，其模型结构可以不同。客户端 `k` 在部署时刻 `t` 收到
目标批次：

\[
X_{k,t}\sim P_{k,t}(X),
\]

但算法不观察对应的 `Y_{k,t}`。`P_{k,t}` 可以随时间变化，且图像可能承受未知、复合、
非平稳的退化。算法不观察 corruption 类型、严重度、组合或 clean counterpart。

客户端有三个候选动作：

```text
a = 0: 冻结 source/current 模型
a = L: 只用本地无标签流适应
a = j: 接收客户端 j 的消息后协作适应
```

记执行动作 `a` 后的客户端模型为 `f^a_{k,t}`，其真实目标风险为：

\[
R_{k,t}(a)
=\mathbb E_{(X,Y)\sim P_{k,t}}
  [\ell(f^a_{k,t}(X),Y)].
\]

相对于本地适应，接收客户端 `j` 知识的协作伤害定义为：

\[
H_{k\leftarrow j,t}
=R_{k,t}(j)-R_{k,t}(L).
\]

```text
H < 0: 协作有益
H = 0: 风险相同
H > 0: 协作有害
```

所谓“安全协作证书”至少要根据算法可观察信息 `O_{k,t}`，正确判断 `H` 的符号，或者在
不能判断时拒绝通信。当前候选允许 `O_{k,t}` 包括：

- 源训练集和 client-private source audit；
- 当前及历史无标签目标图像；
- 所有客户端在公共数据或目标数据上的 logits、entropy、features 或随机视图响应；
- 模型结构描述、历史通信消息和适应轨迹；
- 算法自身的随机数。

但 `O_{k,t}` 不包括当前目标任务标签。

## 3. 不可识别性反例

### 3.1 两世界反例

考虑二分类，输入空间包含一个目标点 `x*`。目标边缘分布在两个世界中完全相同：

\[
P_T(X=x^*)=1.
\]

两个候选模型在 `x*` 上给出相反预测：

\[
f^L(x^*)=0,\qquad f^j(x^*)=1.
\]

定义两个可能世界：

\[
\mathcal W_0:\quad P_T(Y=0\mid X=x^*)=1,
\]

\[
\mathcal W_1:\quad P_T(Y=1\mid X=x^*)=1.
\]

对于 0-1 loss：

\[
H(\mathcal W_0)=1-0=+1,
\]

\[
H(\mathcal W_1)=0-1=-1.
\]

但两个世界的下列观测完全一致：

```text
目标图像 X=x*
本地和协作模型的全部 logits / features / entropy
任意 dropout、多视图或增强响应
任意客户端之间的公共数据响应
所有模型结构和通信历史
```

因为这些量都是输入、固定模型和算法随机数的函数，不读取目标 `Y`。所以任何决策规则
`delta(O)` 在两个世界中必须做出相同动作，却至少在一个世界中判断错误。

### 3.2 保持 source 数据和 audit 完全相同

令 `x*` 不在 source 支持内。两个世界共享完全相同的：

```text
source P_S(X,Y)
训练集与 client-private audit
四个异构 source models
公共 CIFAR-100 数据和所有公共 logits
目标 P_T(X)
```

它们只在不可观察的 `P_T(Y|X=x*)` 上不同。因此：

> source-audit accuracy 可以约束 source risk，却不能识别超出 source 支持的目标协作伤害。

这直接否定把当前 `AsymHFL-val` 的 source-audit teacher 排序原样搬到部署阶段，作为
target no-harm 证书。

### 3.3 一般命题

**命题 1（无标签协作收益符号不可识别）**

设两个候选动作产生不同预测，且允许目标条件分布 `P_T(Y|X)` 在不改变可观察量
`O` 的情况下变化。则不存在只依赖 `O` 的决策规则，能对所有与 `O` 相容的目标联合分布
正确识别：

\[
\operatorname{sign}\{R_T(j)-R_T(L)\}.
\]

证明由两世界构造直接给出。

一个始终拒绝协作的规则当然可以形式上保证“不因通信受害”，但它不会在任何世界获得
协作收益。命题排除的是同时满足以下两点的非平凡规则：

```text
在有害世界可靠拒绝通信
在有益世界以非零概率接受通信
```

当两个世界的可观察量相同时，这两个目标不能同时保证。

### 3.4 推论

下列操作本身不会消除不可识别性：

1. 增加更多客户端；
2. 使用更多不同模型结构；
3. 比较 entropy、confidence、margin 或预测一致性；
4. 对输入加入 AugMix、随机视图或 dropout；
5. 在无标签公共数据上计算 logits 相似度；
6. 把历史 source-audit accuracy 作为路由分数；
7. 将以上信号放入更复杂的图、张量或神经网络。

它们可能提供经验相关性，但不会仅凭形式变化获得对目标真实风险差的分布无关保证。

该结论与 Ben-David 等人的 domain-adaptation impossibility 一致：无目标标签时，输入上
不可区分的任务可以对应相反的迁移结论；Gulrajani 和 Hashimoto 也把存在多个保持边缘
分布但不保持标签的映射称为 underspecified domain mapping。

## 4. “最弱成立假设”分层

这里必须区分：数学上足够、现实中可检查、以及是否仍属于原问题。

### 4.1 直接风险代理假设：形式最弱，但近乎循环

若存在可观察 gap proxy `G_{k<-j,t}` 满足：

\[
|G_{k\leftarrow j,t}-H_{k\leftarrow j,t}|\le \epsilon_H,
\]

则当：

\[
|G_{k\leftarrow j,t}|>\epsilon_H
\]

时可以识别 `H` 的符号。

等价地，若分别存在两个风险估计并满足：

\[
|\widehat R(a)-R(a)|\le\epsilon_R,
\qquad a\in\{L,j\},
\]

则：

\[
|\widehat H-H|\le 2\epsilon_R,
\]

只有在 `|widehat H|>2 epsilon_R` 时才能安全判断。

问题是：在目标标签不可见时，`epsilon_H` 或 `epsilon_R` 无法从目标数据验证。把“proxy
在所有动态复合退化下仍与真实损失一致”写成假设，本质上已假设了需要证明的安全性。
AETTA 使用 dropout disagreement 估计 TTA accuracy；NeurIPS 2025 的 TTA risk monitoring
则显式要求 loss proxy 在低/高损失之间保持可分性，并且这种可分性跨时间分布保持。它们
是有价值的条件式工具，但不是分布无关地绕过反例。

判决：

```text
数学充分性：有
可验证性：  无（在 fully unlabeled target 上）
新颖性：    低，与 AETTA / TTA risk monitoring 直接碰撞
```

### 4.2 Covariate shift + overlap：标准充分条件，但不适合任意损坏

假设对客户端 `k` 和时刻 `t`：

\[
P_{k,t}(Y\mid X)=P_{k,S}(Y\mid X),
\]

以及：

\[
P_{k,t}(X)\ll P_{k,S}(X),
\qquad
w_{k,t}(X)=\frac{dP_{k,t}(X)}{dP_{k,S}(X)}<\infty.
\]

若候选模型在评估样本之前冻结，或采用独立 sample split，则：

\[
H_{k\leftarrow j,t}
=\mathbb E_{P_{k,S}}
  [w_{k,t}(X)
  \{\ell(f^j(X),Y)-\ell(f^L(X),Y)\}].
\]

这使 source labeled audit 与 target unlabeled density-ratio estimation 在理论上足以估计 `H`。

但对严重 blur、noise、compression 或复合退化：

- target 图像可能位于 source 支持之外，违反 overlap；
- 退化造成信息丢失后，`P(Y|X)` 可以改变；
- 高维图像 density ratio 难以稳定估计；
- `P_T(Y|X)=P_S(Y|X)` 无法只用无标签目标流检验；
- 使用同一批次适应并评估会破坏普通独立估计，需要额外 sample splitting。

判决：

```text
数学充分性：有
对任意复合损坏的合理性：低
无标签可检验性：低
论文空间：已属于成熟 UDA / shift-estimation 假设体系
```

### 4.3 条件独立的多模型 ensemble：理论可行，项目中不可信

另一条无目标标签路线是假设多个模型的错误在给定真实类别后条件独立，并且其 confusion
matrices 可识别、方向由 source labels 锚定、各模型优于随机猜测。此时可以把模型预测视为
多个 noisy annotators，估计潜在标签和模型准确率。

当前项目不满足其可信前提：

- 客户端使用相同任务数据生成过程和多轮知识蒸馏，错误强相关；
- corruption 会让多个模型共同依赖同一脆弱特征；
- 只有四个客户端，且每个架构只有一个实例；
- 条件独立性和 target confusion matrix 在无标签流上不可验证；
- 一旦直接用预测一致性代替该统计模型，就回到 AETTA、FedCTTA、FRT/CCAD/FedCIS 类信号。

判决：理论条件式可行，但不作为当前项目的最弱现实假设。

### 4.4 稀疏、无偏、延迟目标任务标签：最低可操作方案

设每个目标样本的任务标签以概率 `q>0` 在延迟 `D` 后到达，且标签是否到达在给定已记录
信息后与候选动作损失独立（missing at random）。再假设决策窗口内 `H` 静态或总漂移有界。
此外，客户端必须保留本地与协作两个 shadow candidates，使二者在相同历史和同一样本上均可
评估；若只执行其中一个动作，则必须随机化动作并满足 positivity，再用 inverse-propensity
估计，否则延迟标签仍不能恢复未执行动作的反事实轨迹风险。

对获得标签的样本定义 paired loss difference：

\[
Z_i=\ell(f^j(X_i),Y_i)-\ell(f^L(X_i),Y_i).
\]

若 `ell in [0,1]`，则 `Z_i in [-1,1]` 且：

\[
\mathbb E[Z_i]=H.
\]

用 `n` 个无偏反馈样本，Hoeffding 给出：

\[
P(|\bar Z-H|\ge\epsilon)
\le 2\exp(-n\epsilon^2/2).
\]

因此：

\[
n\ge \frac{2}{\epsilon^2}\log\frac{2}{\delta}
\]

足以在误差 `epsilon`、置信度 `1-delta` 下估计协作伤害；原始目标样本量约为 `n/q`。
动态流应使用 time-uniform confidence sequence，并显式处理延迟和漂移。

它不需要 corruption 标签，只需要少量最终任务标签，因此比 PEW 的人工退化 taxonomy 更
现实且可验证。但它已经改变原设定：

```text
fully unsupervised TTA
-> online / continual learning with delayed supervision
```

NeurIPS 2024 已系统研究 online continual learning 的 label delay，并发现许多 SSL/TTA
方法甚至不如直接利用延迟监督的朴素基线。若转向这一设定，必须与 delayed-feedback、
online experts、federated online model selection 重新查重，不能把延迟标签只当工程补丁。

判决：

```text
数学充分性：高
现实可验证性：高于前三类
是否仍是原问题：否
当前许可：CONDITIONAL REFRAME ONLY
```

### 4.5 最弱假设总结

| 信息/假设 | 是否足以识别 `sign(H)` | 是否可在当前协议验证 | 是否保持 fully unlabeled | 判决 |
|---|---:|---:|---:|---|
| 目标 `X` + logits/features/entropy | 否 | 是 | 是 | `NO-GO` |
| source audit | 否，除非再加 shift 假设 | 是 | 是 | `NO-GO` |
| 公共数据或随机视图响应 | 否 | 是 | 是 | `NO-GO` |
| calibrated loss-gap proxy | 条件式是 | 否 | 是 | 循环/碰撞 |
| covariate shift + overlap | 是 | 关键条件不可检验 | 是 | 理论路线，不适配任意损坏 |
| 条件独立多模型 ensemble | 条件式是 | 否 | 是 | 项目中不可信 |
| 稀疏无偏延迟任务标签 | 是 | 是 | 否 | `CONDITIONAL REFRAME` |
| 当前目标完整标签 | 是 | 是 | 否 | oracle |

## 5. 现有方法碰撞矩阵

| 工作 | 已覆盖对象 | 与本候选的碰撞 | 尚存差异 | 判决影响 |
|---|---|---|---|---|
| RAHFL / AsymHFL | 模型异构、损坏客户端、选择性单向知识转移 | “向谁学习、避免低质量教师”已是核心 | RAHFL 在训练期且使用监督质量信号 | 不能把 AsymHFL 平移到测试期当创新 |
| FedTHE/FedTHE+ (ICLR 2023) | 联邦部署期分布偏移、无标签 head ensemble 与 TTA | 冻结/本地/组合模型的 test-time 决策相邻 | 非多客户端实时协作；非任意异构共存 | 新场景已非空白 |
| ATP (NeurIPS 2023) | test-time personalized FL；多种 shift；自适应模块适应率 | 无标签 FL 测试时适应和 corruption 已覆盖 | 主要从共享 FL 模型出发 | “FTTA+corruption”不能作为新贡献 |
| FedICON (NeurIPS 2023) | 用 inter-client heterogeneity 处理 intra-client test shift | “利用客户端差异帮助测试偏移”直接碰撞 | 训练期表示学习，不是持续通信安全 | 必须提出超出 invariant representation 的对象 |
| FedTSA (KDD 2024) | 动态环境中的 collaborative TTA；时空关系和客户端相似性 | 动态协作关系已覆盖 | 共享特征/同构聚合，不是任意模型结构 | 只换相似度不足 |
| CoLA (NeurIPS 2024) | cross-device collaborative lifelong TTA；共享 domain knowledge vectors | 持续、多设备、资源异构协作已覆盖 | system heterogeneity 不等于 model architecture heterogeneity | “cross-device continual”已拥挤 |
| AETTA (CVPR 2024) | 无标签 TTA accuracy estimation；failure recovery | 直接覆盖无标签安全信号和恢复 | 非联邦、非 pairwise collaboration gain | 把 disagreement 用作路由新颖性低 |
| UDA model-selection audit (NeurIPS 2024) | 无标签目标上的模型选择可靠性 | 说明多种无标签选择准则不能避免最坏选择 | 非联邦且不专门研究 TTA | 支持本报告的保守判决 |
| FedCTTA (2025) | continual FTTA；随机噪声输出相似度路由；时空 shift；CIFAR-C | 几乎覆盖动态 corruption 协作与 output-based routing | 通过模型聚合，通常要求参数兼容 | “改成公共 logits 支持异构模型”仍像组合扩展 |
| TTA Risk Monitoring (NeurIPS 2025) | 无标签动态 TTA 的 sequential risk alarm；uncertainty proxy | 直接覆盖适应伤害监控及报警 | 非联邦、不是 pairwise teacher routing | 安全证书核心对象已高度相邻 |
| Latte (ICCV 2025) | 联邦协作 TTA；corruption benchmark；跨客户端原型与 OOD robustness | 协作、个性化、corruption、安全过滤均相邻 | 共享 VLM embedding，不支持任意 backbone | model heterogeneity 是剩余差异，但不足单独成核 |
| MORPHEUS (2026) | 用无标签几何预测 TTA 方法和适应后准确率 | 冻结/选择适应策略的目标相邻 | 非联邦 | 进一步压缩 method-selection 新颖空间 |
| Label Delay in OCL (NeurIPS 2024) | 延迟标签流、SSL/TTA 与监督基线 | 覆盖最现实的可识别补救信息 | 非联邦、非模型异构 | delayed-label 版本需重新立题而非直接 GO |

### 5.1 剩余差异为什么不够

截至本次原始论文初筛，`arbitrary model architectures coexist during collaborative CTTA` 仍是
相对清晰的交叉差异。但仅有交叉差异不等于论文贡献：

```text
FedCTTA 的动态协作与 output similarity
+ 当前 public-logit HFL 的异构模型兼容层
+ AETTA / TTA risk monitoring 的无标签风险 proxy
= 一个自然组合，而非新的可识别学习对象
```

若不解决命题 1，异构模型只增加实现难度；若用一个经验 proxy 绕过命题 1，又直接进入
AETTA、risk monitoring 和当前冻结 FRT/CCAD/FedCIS 的拥挤区域。

## 6. 与项目冻结路线的冲突

当前项目不能通过以下方式修补候选：

```text
公共多视图 response tensor / subspace -> FRT、CCAD、IRD、FedCIS 冲突
source-audit teacher ranking            -> 只是复用 AsymHFL-val
客户端输出图或相似度路由               -> PRAC-HFL / FedCARA / FedCTTA 邻近
高 entropy / 高 loss 样本保护           -> CVaR、tail mining 邻近
连续 nuisance witness                   -> 已冻结 continuous witness
对 corruption 做隐式聚类                -> 又回到 PEW/PIE/LCC 的识别问题
```

特别是当前 `fedprime/communication/public_logits.py` 已经提供：

```text
public logits
异构 backbone
source-audit accuracy routing
KL knowledge transfer
```

把运行时无标签 entropy 或 disagreement 替换 source-audit accuracy，只会形成一个容易实现、
但理论不可验证且外部碰撞严重的通信启发式。

## 7. 评分与论文判决

评分为 1（最差）到 5（最好）；实现成本为 1（最低）到 5（最高）。

| 版本 | 新颖性 | 理论可识别性 | 信息可验证性 | FL-specific | 复用性 | 实现成本 | 实验归因 | 论文价值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fully-unlabeled safe MH-CTTA | 2 | 1 | 1 | 4 | 3 | 4 | 2 | 1 |
| empirical proxy-gated MH-CTTA | 2 | 2 | 2 | 4 | 3 | 4 | 2 | 2 |
| sparse delayed-label MH online adaptation | 3（待重查） | 4 | 4 | 4 | 2 | 5 | 3 | 3（待审计） |

### 7.1 预注册式纸面门槛

| 门槛 | 结果 |
|---|---|
| 不需要 corruption taxonomy | `PASS` |
| 信息在当前 fully-unlabeled 协议中可观察 | `PASS` |
| 协作伤害符号在该信息下可识别 | `FAIL` |
| 最弱充分假设可由当前协议验证 | `FAIL` |
| 模型异构构成新的数学对象而非兼容层 | `FAIL` |
| 与 FedCTTA/Latte/AETTA/risk monitoring 有核心间隔 | `FAIL` |
| 不复活项目冻结通信/公共响应路线 | `FAIL` |
| 一到两个月内能形成干净归因 | `FAIL` |

结果：`2/8 PASS`。

### 7.2 最终判决

```text
当前 fully-unlabeled safe collaborative MH-CTTA： PAPER NO-GO
最小实现：                                      NO
本地 smoke：                                    NO
OpenI：                                         NO
```

这不是说 FTTA 或 model heterogeneity 没有研究价值，而是当前提出的强主张同时受到：

1. 无标签目标风险不可识别；
2. 无标签 TTA 性能估计与风险监控已有直接工作；
3. 协作式、持续式联邦 TTA 已有多条成熟路线；
4. “支持异构 backbone”本身不足以抵消前三点。

## 8. 唯一保留的条件式路线

若愿意显式接受：

```text
客户端会收到稀疏、无偏、延迟的最终任务标签
```

则可以研究：

```text
model-heterogeneous federated online adaptation
with delayed feedback under evolving corruptions
```

其潜在数学对象不再是无标签 confidence，而是：

- pairwise collaboration loss-difference confidence sequence；
- delay-aware、drift-aware teacher/expert selection regret；
- 对异构模型只能传递输出/决策而不能平均参数的在线协作约束。

但该路线当前只有 `CONDITIONAL REFRAME`，原因是：

- 问题从 FTTA 改成 delayed-supervision online/continual learning；
- 需要新的数据流协议和评价指标；
- 与 Label Delay in OCL、online experts、federated online model selection 尚未完成专项查重；
- 当前 CLE runner 是离线多轮训练，不是部署反馈流，工程改动较大。

在用户明确接受该任务变化前，不设计其方法，不修改代码，不跑实验。

## 9. 原始论文证据

不可识别性与风险估计：

- Ben-David et al., [*Impossibility Theorems for Domain Adaptation*](https://proceedings.mlr.press/v9/david10a.html), AISTATS 2010.
- Gulrajani and Hashimoto, [*Identifiability Conditions for Domain Adaptation*](https://proceedings.mlr.press/v162/gulrajani22a.html), ICML 2022.
- Lee et al., [*AETTA: Label-Free Accuracy Estimation for Test-Time Adaptation*](https://openaccess.thecvf.com/content/CVPR2024/html/Lee_AETTA_Label-Free_Accuracy_Estimation_for_Test-Time_Adaptation_CVPR_2024_paper.html), CVPR 2024.
- Hu et al., [*Towards Reliable Model Selection for Unsupervised Domain Adaptation*](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f50cebc22663df45ce619645bfabb3b3-Abstract-Datasets_and_Benchmarks_Track.html), NeurIPS 2024.
- Schirmer et al., [*Monitoring Risks in Test-Time Adaptation*](https://proceedings.neurips.cc/paper_files/paper/2025/file/746960ad49ddb47248970a0e1404230c-Paper-Conference.pdf), NeurIPS 2025.

联邦与协作式测试时适应：

- Jiang and Lin, [*Test-Time Robust Personalization for Federated Learning*](https://arxiv.org/abs/2205.10920), ICLR 2023.
- Bao et al., [*Adaptive Test-Time Personalization for Federated Learning*](https://papers.nips.cc/paper_files/paper/2023/hash/f555b62384279b98732204cb1a670a23-Abstract-Conference.html), NeurIPS 2023.
- Tan et al., [*Is Heterogeneity Notorious? Taming Heterogeneity to Handle Test-Time Shift in Federated Learning*](https://openreview.net/forum?id=qJJmu4qsLO), NeurIPS 2023.
- Zhang et al., [*Enabling Collaborative Test-Time Adaptation in Dynamic Environment via Federated Learning*](https://doi.org/10.1145/3637528.3671908), KDD 2024.
- Chen et al., [*Cross-Device Collaborative Test-Time Adaptation*](https://proceedings.neurips.cc/paper_files/paper/2024/file/de0e668df3fe63ec89e5a7e68f3d350f-Paper-Conference.pdf), NeurIPS 2024.
- Rajib et al., [*FedCTTA: A Collaborative Approach to Continual Test-Time Adaptation in Federated Learning*](https://arxiv.org/abs/2505.13643), 2025.
- Bao et al., [*Latte: Collaborative Test-Time Adaptation of Vision-Language Models in Federated Learning*](https://openaccess.thecvf.com/content/ICCV2025/papers/Bao_Latte_Collaborative_Test-Time_Adaptation_of_Vision-Language_Models_in_Federated_Learning_ICCV_2025_paper.pdf), ICCV 2025.

延迟反馈：

- Csaba et al., [*Label Delay in Online Continual Learning*](https://openreview.net/forum?id=m5CAnUui0Z), NeurIPS 2024.

## 10. 下一决策

本轮已经完成用户要求的：

```text
不可识别性反例
最弱成立假设
现有方法碰撞矩阵
纸面 GO/NO-GO
```

当前不得进入实现。下一步只有一个需要用户决定的问题：是否接受将课题明确改为
“具有稀疏无偏延迟任务标签的模型异构联邦在线适应”。若不接受，则停止本候选并继续寻找
一个不以隐藏目标风险符号为核心的新课题。
