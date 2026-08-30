# CLE Local-First Directional Shortcut：定义、可识别性与方法碰撞审计

Updated: 2026-08-30

## 0. 结论先行

本审计只做纸面定义、反例和文献碰撞，不改代码、不运行训练。

```text
CLE 现实动机：                         GO
当前 CIFAR-10 CLE 协议作为受控压力测试： GO
“首次发现类别—损坏纠缠”主张：           NO-GO
DSA 作为项目定制的方向性诊断：           CONDITIONAL CONTRIBUTION
只靠 (损坏图像, y, i.i.d. AugMix views)
识别真实 directional shortcut：          IMPOSSIBLE
普通 view-consistency / wrong-class drift：NO-GO AS NEW CORE
加入最弱主动干预桥后的 PIDR：             CONDITIONAL GO FOR ZERO-TRAINING GATE
立即实现方法或启动训练：                  NO
```

最重要的边界是：Phase-A0/A1a 已经证明模型的预测移动与**真实训练绑定图**方向一致，
而不只是准确率下降；但是，训练阶段若既不知道退化环境，又没有能真正改写退化因素的配对
干预，就不能从普通 AugMix 视图中识别同一个因果量。

## 1. 当前数据包到底是什么

Phase-A1a OpenI 输入包不是一个外部下载的现成数据集，而是由项目从 CIFAR-10 构造出的两
个严格配对世界：

```text
private task:       CIFAR-10, 4 clients, Dirichlet alpha=0.5
client models:      ResNet10 / ResNet12 / ShuffleNet / MobileNetV2
gamma=0:            每个 family 概率均为 0.25
gamma=0.9:          dominant family 概率 0.925，其余各 0.025
families:           noise / blur / weather / digital
operators/family:   4，总计16个 operator
base severity:      每张训练图独立取 1--5
fit/audit:          每客户端固定分层约 85/15
public data:        CIFAR-100 tar，仅供 HFL public-logit communication
evaluation source:  1,000 张类别均衡 clean CIFAR-10 图像
```

客户端 `i`、类别 `c` 的 dominant family 是固定的循环图：

\[
d(i,c)=g_{(i+c)\bmod 4}.
\]

对一条本地训练样本，先按

\[
P(G=g\mid Y=c,I=i)
=\frac{1-\gamma}{4}+\gamma\mathbf 1\{g=d(i,c)\}
\]

抽 family，再在该 family 的四个 operator 中抽一个，并抽 severity。`gamma=0.9` 因而不是
“100% 固定”，而是 92.5% dominant、7.5% counter-binding。

两个世界保持 source index、真实标签、severity、fit/audit 角色和模型初始化一致；允许改变的
只有 family/operator 抽样。上传包包含两套 prepared private arrays、固定 split、1,000 张
clean evaluation source、CIFAR-100 public tar 和完整哈希。四个共享初始 checkpoint 不是
预先塞进数据包的，而是 OpenI 入口用固定随机种子在运行时生成并由四臂逐字节复用。

每张 prepared private image 只有一个**基础 corruption operator**。本地训练的 AugMix 又
从这张已经损坏的图像生成多个随机增强视图，所以训练时确实可能出现多操作叠加；但当前
CLE 的受控绑定变量仍是基础 family，而不是对任意复合退化组合建模。

## 2. 我们为什么有资格说模型学到了 shortcut

### 2.1 不能只靠准确率、WCCA 或 CFG

`gamma=0.9` 下 Avg/WCCA 下降、CFG 上升只能说明模型更脆弱，不能区分：

```text
operator 本身更难
模型一般性退化
模型真的把特定 corruption 当成类别证据
```

因此正式判断使用 Directional Shortcut Alignment（DSA），不是“掉点即 shortcut”。

### 2.2 DSA 的因果方向

对客户端 `i` 和 family `g`，令

\[
\mathcal B_{i,g}=\{c:d(i,c)=g\}
\]

为训练中主要与 `g` 绑定的类别集合，令

\[
Q_{i,g}(z)=\sum_{c\in\mathcal B_{i,g}}p_i(c\mid z).
\]

对真实标签不属于 `\mathcal B_{i,g}` 的同一 clean source，只替换 corruption family，定义

\[
\operatorname{DSA}_{i,g}
=\mathbb E\left[
\mathbb E_{o\in\mathcal O_g}Q_{i,g}(T_o(X))
-\mathbb E_{h\ne g,o\in\mathcal O_h}Q_{i,g}(T_o(X))
\right].
\]

它问的是：**当语义和 severity 不变、只施加 family `g` 时，预测概率是否定向流向训练中
与 `g` 绑定的错误类别。** 为避免仅仅测到某些类别集合天然更容易被预测，项目还保持每个
family 的绑定类别数量不变，随机打乱 class--family map 1,000 次作为零假设。

所以“shortcut 已观察到”需要同时满足：

```text
同一 source 的配对 corruption 干预
gamma0.9 - gamma0 的 DSA 明显为正
客户端方向一致
真实绑定显著超过保持组大小的随机绑定
```

Phase-A0 的 `Delta DSA=0.201623`、CI95 `[0.196412,0.207219]`、4/4 客户端同向且
随机绑定 `p=0.000999` 满足这些条件。Phase-A1a 又得到 HFL/Local 的 CLE effect 分别约
`0.2027/0.2044`，而通信差分中的差分接近0。因此当前证据支持：

\[
\boxed{\text{CLE}\rightarrow\text{local-first directional shortcut}}
\]

不支持“坏教师传播”或“通信放大”。

## 3. 协议是不是创新

需要把三层贡献分开：

| 层次 | 当前判断 | 可以说什么 | 不能说什么 |
|---|---|---|---|
| 现实问题 | 已有大量先例 | 不同站点的采集/质量线索会与标签相关并形成 shortcut | 首次发现 class--corruption spurious correlation |
| CLE benchmark | 有项目特异性 | 在模型异构 FL 中同时控制 client、class、corruption 与 gamma 的配对压力测试 | 它完整代表真实世界退化 |
| DSA 诊断 | 有潜在贡献 | 用同源干预和训练绑定方向判断“利用 corruption”而非仅“承受 corruption” | 仅凭当前一组 CIFAR 实验就是独立论文核心 |

因此，协议和 DSA **可以成为论文的一项评价/诊断贡献**，但目前不能单独承担整篇 CCF-B
方法论文。尤其 cyclic map、四个 family、`gamma=0.9` 都是为了可控归因，不是自然规律。
最终论文至少需要一个新方法，并在中等 gamma、不同绑定图或半自然/自然 source--quality
场景上复验，才能降低“人为 toy benchmark”攻击。

## 4. 现实性与审稿风险

### 4.1 现实问题存在

真实世界不需要满足“一类永远只有一种损坏”，只需满足：

\[
P_i(G\mid Y=c)\ne P_i(G)
\]

且部署时该相关性会变化。医院、设备、采集协议、患者选择和疾病患病率共同形成这种路径。
2024 年 npj Digital Medicine 在 13 个数据集、5 种模态上报告隐藏采集偏差导致传统内部评价
平均高估外部性能约20%，并明确讨论跨医院 scanner variation 与针对疑似病例采用特定成像
协议的问题。2024 年 Scientific Reports 的案例也总结了体位、皮肤标记、胸管以及图像
采集流程等 shortcut。它们证明“采集/质量线索与任务标签耦合”不是虚构问题。

### 4.2 当前描绘最容易被攻击的地方

```text
R1  四个 family 和循环绑定是人为设计；
R2  gamma=0.9 很强，不能直接代表常见真实强度；
R3  每张 prepared image 只有一个基础 operator，不能声称覆盖任意复合退化；
R4  CIFAR-10 语义和 32x32 合成 corruption 距离真实设备数据较远；
R5  class--corruption shortcut 与一般 spurious correlation / domain generalization 已有直接文献；
R6  DSA 使用 clean-source counterfactual grid，属于评价 oracle，不能假装训练时可见；
R7  seed0 只证明当前构造，尚不证明训练随机性和跨场景稳定性。
```

### 4.3 稳妥的问题表述

推荐写成：

> 多站点/多设备学习中，采集退化的分布可能同时依赖客户端和语义类别。模型在原分布上的
> corruption accuracy 可能来自利用这种 client-conditional acquisition cue，而非学习对
> 退化不变的语义。我们用可控的 CLE 压力测试分离这两种行为。

不要写成：

> 现实中每个客户端的每个类别都固定对应某一种 corruption；我们首次发现该问题。

## 5. 真正的 local-first directional shortcut 风险

令 `S` 为语义，`Y=h(S)`，`G` 为隐藏退化状态，观测图像为 `X=R(S,G)`。客户端 `i` 的
class-conditional dominant target set 记为 `\mathcal B_i(g)`。定义总体真风险：

\[
\mathcal R^{\mathrm{LFDS}}_i(\theta)
=\mathbb E_{S,Y}\mathbb E_g
\mathbf 1\{Y\notin\mathcal B_i(g)\}
\left[
p_\theta(\mathcal B_i(g)\mid R(S,g))
-\mathbb E_{h\ne g}p_\theta(\mathcal B_i(g)\mid R(S,h))
\right].
\]

它是 DSA 的 population 版本。“local-first”不是额外加一个通信项，而是指该风险在只做
本地训练时已经产生；Phase-A1a 表明 HFL 相对 Local 没有持久放大它。

该对象的优点是定义精确、直接对应已有证据；缺点是它需要隐藏 `G`、同一 `S` 的多退化
counterfactual 和绑定集合，因此是**评价风险，不是当前训练信息下的可计算损失**。

## 6. 不可识别性反例

### 6.1 允许的信息

严格本地方案只允许：

```text
一张已损坏训练图 X
任务标签 Y
模型 logits/features
由同一 X 生成的 i.i.d. AugMix views
客户端本地数据
```

不允许真实 corruption/environment label、clean original、真实 counterfactual swap、持久化
source ID 或 final-test 信息。

### 6.2 两世界反例

令 `Y~Bernoulli(1/2)`，训练中观测到的三个像素/特征恒为

\[
X=(U,V,W)=(Y,Y,0),
\]

固定分类器只读取 `V`：`f(X)=V`。普通 AugMix kernel `A` 只改变 `W`，因此两个世界中的
`(X,Y,A(X),f(X),f(A(X)))` 联合分布完全相同。

```text
World A: U 是语义，V 是与 Y 绑定的退化线索。
         保持 U、干预 V 后，f 随 V 改变；LFDS > 0。

World B: V 是语义，U 是与 Y 绑定但模型未使用的退化线索。
         保持 V、干预 U 后，f 不变；LFDS = 0。
```

任何只读取上述允许信息的统计量，在两个世界中分布完全一致，却必须给出不同的真实 LFDS。
因此不存在一致识别器。加入更多相同分布的 AugMix views 也不改变结论；如果基础 shortcut
在全部 views 中被保留，`view-to-view` residual 甚至可以严格为0。

这同时否定了一个看似自然的候选：

\[
\max_{c\ne y}\left[
\max_j p_c(A_j(X))-\operatorname{median}_j p_c(A_j(X))
\right]_+.
\]

它只能测“AugMix 改动造成的预测不稳定”，不能保证测到训练图中已存在且被所有 views
继承的退化捷径。

## 7. 最弱可操作修复假设

纯观察不可识别后，最少必须增加一个**主动干预桥**，而不是再发明一个 residual 名字。

设有可区分的 probe kernels `K_1,...,K_J`，但它们不需要知道真实样本的 corruption label。
需要：

```text
A1  semantic preservation:
    K_j 不改变任务标签，或标签改变概率有可控上界 eta。

A2  degradation overwrite / exchangeability:
    给定语义 S，K_j(X) 的主要退化由 probe j 决定，而不继续完全继承原 G；
    与理想 do(G=g_j) 的分布距离至多 epsilon。

A3  coverage:
    每个会导致 DSA 的退化方向至少被一个 probe 以概率 kappa>0 覆盖。

A4  paired observability:
    训练时知道多个 views 来自同一 X，并知道 probe index j；不要求持久化数据 source ID。

A5  no test leakage:
    probe、阈值和权重不能由 final CLE test 调参。
```

其中 A2 是不可再删除的核心。如果 probe 只是在原退化上叠加无关变化，就回到反例。
A4 中的 `probe index` 是生成器的查询身份，不是给真实数据人工标注五类环境；但审稿人仍
可能认为它把 augmentation taxonomy 以另一种形式带回来了，因此必须单独做敏感性审计。

## 8. 条件候选：Probe-Indexed Directional Promotion Risk（PIDR）

在满足 A1--A5 时，对客户端 `i`、probe `j`、错误类别 `c` 定义：

\[
M_{i,j,c}(\theta)=
\mathbb E_{(X,Y):Y\ne c}
\left[
p_\theta(c\mid K_j(X))
-\sum_{\ell\ne j}\omega_{\ell\mid j}p_\theta(c\mid K_\ell(X))
\right].
\]

定义无需 class--corruption map、无需预设 target-set 大小的方向风险：

\[
\boxed{
\mathcal R_i^{\mathrm{PIDR}}(\theta)
=\sum_j\omega_j\left\|[M_{i,j,:}(\theta)]_+\right\|_1
}.
\]

它惩罚的是：某个主动退化 probe 是否在许多本地样本上**一致地把概率推向特定错误类**。
这与 JSD 的“所有 view 输出都相同”不同，也与单样本 CE 最大化不同。

若某个真实 family `g` 的绑定集合为 `B_i(g)`，并由 probe `j` 忠实覆盖，则

\[
\left[\sum_{c\in B_i(g)}M_{i,j,c}\right]_+
\le \sum_c[M_{i,j,c}]_+.
\]

在 A1--A3 下，对有界概率可得到形式上的桥接界：

\[
\mathcal R_i^{\mathrm{LFDS}}(\theta)
\le \kappa^{-1}\mathcal R_i^{\mathrm{PIDR}}(\theta)
+O((\epsilon+\eta)/\kappa).
\]

这解释了 PIDR 为什么**可能**压低真实 DSA：它对未知绑定集合的正向概率流给出一个
taxonomy-free 上界。但该界只有在 probe 真能改写相关退化且保持语义时才有意义。

### 8.1 仍未解决的问题

```text
P1 当前 AugMix 的多个 slot 同分布；若 K_j 同分布，期望 M_{i,j,c}=0，PIDR 失效。
P2 固定不同 K_j 会重新引入人工 augmentation bank，需证明不是 PEW 式 taxonomy。
P3 强 corruption 可能破坏语义，使 PIDR 把合理的不确定性当 shortcut。
P4 minibatch class imbalance 会令 M 方差很大。
P5 PIDR 可能压制有用的、真实类条件纹理，损害 Avg/BER 收益。
P6 它目前只是风险上界候选，不是已证明新颖的方法。
```

## 9. 2024--2026 方法碰撞矩阵

| 方法 | 年份/场景 | 核心信息或机制 | 与 CLE/PIDR 的碰撞 | 结论 |
|---|---|---|---|---|
| FedPIN | ICML 2024，个性化 FL | 以 global invariant feature 为锚，信息论约束区分 personalized 与 spurious feature | “联邦 + shortcut-averse invariant learning”主叙事已被直接占据 | 强碰撞；不能泛称首个 federated shortcut suppression |
| GIC | ICML 2024 | 利用 spurious attribute 与标签相关、且跨不同 group distribution 变化来推断 group | 若 PIDR 变成先推断 latent corruption group 再做 DRO，直接落入该路线 | 强邻接 |
| Self-supervised low-rank debiasing | CVPR 2024 | 用低秩偏置 encoder 发现并上权 bias-conflicting samples | 无 bias label 的表征/样本发现已有成熟方案 | 中强碰撞；不得把“无标签发现 shortcut”作为唯一贡献 |
| FedCD | 2024 federated DG | 本地 self-supervised feature intervener + 全局 risk-extrapolation aggregation | “本地生成 intervention 抑制 spurious feature”与 PIDR 极近 | 强碰撞；必须在 directional probability object 与 corruption-specific failure 上区分 |
| FedCIFL | AAAI 2025 | sample reweighting + federated causal-effect feature selection | 若方案转为找 causal feature 或普通 reweighting，会直接碰撞 | 中强碰撞 |
| ShortcutProbe | IJCAI 2025 | 用 diverse probe set 在 latent space 找 prediction shortcut，再以 invariance objective 重训 | probe + 无 group label + shortcut retraining 与当前最弱修复高度相邻 | **最危险的中心化碰撞** |
| FedDiverse | 2025 | 定义 client/global attribute、class、spurious-correlation heterogeneity 指标，并用互补分布做 client selection | CLE 作为 federated spurious benchmark 与 client complementarity 已非空白 | benchmark/通信叙事强邻接 |
| FedCAug | 2025 federated OOD | 定位前景/背景，生成 client-local causal counterfactual augmentation | 若 PIDR 依赖显式因果图或背景替换，直接碰撞 | 强碰撞，但它针对 context/background，不是 degradation direction |
| FedDDL | IJCAI 2025 | intra-client 前景/背景 counterfactual + inter-client causal prototype debiasing | 已覆盖联邦内外双层去混杂与异构 representation bridge | 强碰撞；不能再讲 local+global causal debiasing 套件 |
| Let Samples Speak / NSF | CVPR 2025 | 由 feature-space clusterness 找 bias minority 并中和 spurious feature，无 bias label | 若改成聚类 corruption shortcut，再做 feature neutralization，直接邻接 | 强碰撞 |
| SilverLining | WACV 2026 | attention-based spatial/spectral shortcut identification 与 confounder-free preprocessing | corruption-like spatial/spectral shortcut 的数据侧处理已出现 | 中强碰撞；PIDR 若变成频域/空间遮罩会被覆盖 |

另外，现有通用家族已经覆盖以下退化形式：

```text
AugMix/JSD:       对随机增强输出做对称一致性；不识别 class-directed promotion。
AugMax:           对增强混合做 worst-case CE；不识别 population-level target direction。
IRM/VREx:         需要已知或推断 environment，优化跨环境不变风险。
GroupDRO/CVaR:    优化已知组或高损失尾部，不保证尾部就是 corruption shortcut。
EIIL/GIC/聚类法:  从 ERM 表征或误差推断环境/组；PIDR 若先聚类再不变学习就会归入此类。
counterfactual augmentation:
                  若可生成可信反事实，本身就是标准消除 spurious correlation 路线。
```

项目内部还必须遵守冻结负结果：不能把 PIDR 改名后重新实现 C3R、CRSR、PIE、CCAD/IRD、
C3R、连续 witness 或 PEW/BER。尤其“无 probe identity 的 batch residual/谱集中”会重新落回
CRSR/C3R 已否定的局部可观测信号。

## 10. 综合评分

| 候选 | 新颖性 | 理论合理性 | 实现成本 | 归因清晰度 | 论文价值 |
|---|---:|---:|---:|---:|---:|
| CLE 协议 + DSA 诊断 | 3/5 | 4/5 | 已完成 | 5/5 | 3/5，适合作为一项贡献而非唯一核心 |
| 仅 i.i.d. AugMix residual | 1/5 | 1/5 | 1/5 | 2/5 | NO-GO |
| latent clustering/group inference | 1/5 | 3/5 | 3/5 | 2/5 | NO-GO，直接拥挤 |
| PIDR + 可区分 intervention bridge | 2.5/5 | 3.5/5（有假设时） | 2.5/5 | 4/5 | CONDITIONAL；ShortcutProbe/FedCD 是主要威胁 |
| PEW/BER 继续作为核心 | 1/5 | 2/5 | 已有 | 3/5 | NO-GO AS CORE，只保留强 baseline |

## 11. 纸面 GO/NO-GO 与下一门槛

当前不是“CLE 全部失败”，而是结论收缩为：

```text
现象与诊断：GO
通信放大：NO-GO
纯观察的本地新风险：NO-GO（不可识别）
带最弱干预桥的方向风险：CONDITIONAL GO
```

在任何训练实现之前，只允许一个零训练门槛：复用现有 Phase-A0/A1a prediction cache，
**隐藏 family map**，只给分析器16个 probe identity，检查由 `M_{i,j,c}` 恢复出的正向类别
是否显著命中真实绑定类别，并相对 `gamma=0` 明显增强。

这一步必须同时报告：

```text
binding retrieval precision/recall（真实 map 仅用于最终评分）
PIDR gamma0.9 - gamma0
per-client / per-operator consistency
随机 class map 与随机 probe identity 零假设
与 DSA 的相关性，但不能只用相关性自证
```

若连使用 clean-source、已知 probe identity 的 oracle 条件都无法恢复绑定，方法 `NO-GO`；
若通过，也只说明方向对象可观测，仍需单独解决“训练时 probe 是否真正 overwrite 已有退化”
和 ShortcutProbe/FedCD 的方法新颖性，才允许12轮最小 A/B。

## 12. 已完成的零训练隐藏绑定恢复门槛

2026-08-30 使用 Phase-A1a 已返回的 round12/round40 softmax cache 完成正式门槛。没有重新
训练，也没有重新推理。估计阶段只读取：

```text
probabilities
task labels
16个 probe 在 tensor 中的身份/位置
```

`binding` 和 `operator_family_ids` 在四个 arm 的 promotion matrix 全部生成后才打开，且只
用于最终评分和置换零假设。正式 round40 结果：

| arm | PIDR | mAP | AUC | positive precision | positive recall | class-to-family hit |
|---|---:|---:|---:|---:|---:|---:|
| H0 | 0.024559 | 0.441855 | 0.510677 | 0.246516 | 0.468750 | 0.225 |
| H9 | 0.175479 | 0.844847 | 0.923906 | 0.569105 | 0.906250 | 0.850 |
| L0 | 0.025401 | 0.430622 | 0.507083 | 0.244765 | 0.431250 | 0.275 |
| L9 | 0.174728 | 0.865557 | 0.933177 | 0.587786 | 0.925000 | 0.875 |

预先固定的门槛为 `gamma9 mAP>=0.60`、`mAP delta>=0.20`、至少3/4客户端同向、
class-to-family hit `>=0.70`，以及 class-map/probe-identity 两类置换 `p<=0.01`。结果：

```text
HFL mAP delta:       +0.402993
Local mAP delta:     +0.434935
positive clients:    4/4, 4/4
H9 class/probe p:    0.000999 / 0.000999
L9 class/probe p:    0.000999 / 0.000999
verdict:              GO_TO_INTERVENTION_BRIDGE_DESIGN
```

H9/L9 的 AUC 为 `0.9239/0.9332`，而 H0/L0 为 `0.5107/0.5071`，说明 gamma0 的较高
mAP 部分来自有限正例与 tie ranking，并没有真实方向区分能力。round12 已出现同样模式：
H9/L9 mAP `0.8135/0.8360`，hit `0.80/0.875`。

该结果证明：在 clean paired source、可区分 probe 的 oracle 条件下，即使估计器不知道
family 或 class--family map，operator-induced probability direction 仍能恢复隐藏绑定。
它把 PIDR 从“只有公式的想法”推进到**主动干预桥设计阶段**。

它仍然不证明：

```text
普通 i.i.d. AugMix views 能 overwrite 已存在的基础退化；
训练阶段无 clean source 时 PIDR 仍可估计；
PIDR 已经与 ShortcutProbe/FedCD 形成方法新颖性；
加入 PIDR loss 后会降低 DSA/CFG 且不损害 Avg/WCCA；
```

实现和产物：

```text
fedprime/engine/cle_probe_directional_promotion.py
scripts/analyze_cle_pidr_zero_training_gate.py
tests/test_cle_probe_directional_promotion.py
deliverables/cle_pidr_zero_training_gate_20260830/
focused verification: 10 passed
```

下一步不应直接跑12轮训练。先在纸面设计满足 A1--A4 的最小 probe bridge，并完成它与
普通 AugMix、固定 corruption bank、ShortcutProbe 和 FedCD 的逐项机制差异；只有 bridge
能够真实改写基础退化且不依赖真实 corruption label，才进入一轮 smoke 和12轮 A/B。

## 13. 主要外部证据

- [Shortcut learning in medical AI hinders generalization（npj Digital Medicine 2024）](https://www.nature.com/articles/s41746-024-01118-4)
- [The risk of shortcutting in deep learning algorithms for medical imaging research（Scientific Reports 2024）](https://www.nature.com/articles/s41598-024-79838-6)
- [FedPIN: Causally Motivated Personalized Federated Invariant Learning（ICML 2024）](https://proceedings.mlr.press/v235/tang24a.html)
- [Improving Group Robustness Requires Preciser Group Inference（ICML 2024）](https://proceedings.mlr.press/v235/han24g.html)
- [Self-supervised Debiasing Using Low Rank Regularization（CVPR 2024）](https://openaccess.thecvf.com/content/CVPR2024/html/Park_Self-supervised_Debiasing_Using_Low_Rank_Regularization_CVPR_2024_paper.html)
- [Reducing Spurious Correlation for Federated Domain Generalization / FedCD（2024）](https://arxiv.org/abs/2407.19174)
- [Federated Causally Invariant Feature Learning（AAAI 2025）](https://ojs.aaai.org/index.php/AAAI/article/view/33866)
- [ShortcutProbe（IJCAI 2025）](https://www.ijcai.org/proceedings/2025/795)
- [FedDDL（IJCAI 2025）](https://www.ijcai.org/proceedings/2025/677)
- [Diversity-Driven Learning / FedDiverse（2025）](https://arxiv.org/abs/2504.11216)
- [Federated Causal Augmentation / FedCAug（2025）](https://arxiv.org/abs/2504.19882)
- [Let Samples Speak（CVPR 2025）](https://openaccess.thecvf.com/content/CVPR2025/html/Li_Let_Samples_Speak_Mitigating_Spurious_Correlation_by_Exploiting_the_Clusterness_CVPR_2025_paper.html)
- [SilverLining（WACV 2026）](https://openaccess.thecvf.com/content/WACV2026/html/Unnikrishnan_SilverLining_Data-First_Mitigation_of_Spatial_and_Spectral_Shortcuts_Without_Introducing_WACV_2026_paper.html)
