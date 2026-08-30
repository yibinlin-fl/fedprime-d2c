# CLE Public Canonicalization + Directional Withdrawal：干预桥与训练目标设计

Updated: 2026-08-30

## 0. 阶段结论

本文承接已经通过的 PIDR oracle gate，只做纸面方法设计、理论边界和方法碰撞审计；没有
修改训练代码，也没有启动实验。

```text
直接在私有坏图上再叠加固定 corruption bank：       NO-GO
纯 i.i.d. AugMix / view consistency：                 NO-GO
从私有单图无假设地恢复 clean image：                 IMPOSSIBLE
公共无标签 canonicalizer 作为 nuisance bridge：       CONDITIONAL GO
Signed Class-Directional Withdrawal risk：             CONDITIONAL GO
立即实现完整训练方法：                                 NO
下一步：                                                bridge-only Kill Test
```

当前最值得保留的组合是：

1. 用 HFL 本来就拥有的公共无标签图像训练一个退化中和器；
2. 中和器不预测 corruption type，也不读取私有类别或环境标签；
3. 客户端只比较同一私有图像在“原始退化”与“退化中和”状态下的错误类别概率；
4. 只惩罚跨样本稳定、统计上可信、朝某个错误类别的单向概率撤回；
5. 保留现有 AugMix/JSD/DCL 与 AsymHFL，不新增通信机制。

本文暂称干预桥为 **Public Nuisance Canonicalization Bridge (PNCB)**，风险对象为
**Signed Class-Directional Withdrawal (SCDW)**。名称是工作名，不是最终论文命名。

## 1. 为什么不能直接把 PIDR loss 塞进训练

已经通过的 PIDR gate 使用了 clean source 上的16个可区分 probe：

\[
M_{i,j,c}=\mathbb E_{Y\ne c}
\left[p_i(c\mid K_j(X))-\frac1{J-1}\sum_{\ell\ne j}p_i(c\mid K_\ell(X))\right].
\]

它证明“如果有可信干预，方向矩阵能恢复隐藏绑定”，但训练集只有已经损坏的私有图像。
若直接对坏图再调用原有 corruption operator，得到的是

\[
K_j(X)=T_j(R(S,G)),
\]

而不是理想的

\[
K_j(X)\approx R(S,G_j).
\]

前者通常同时保留旧退化 `G` 并叠加新退化 `j`。因此它无法排除：

```text
旧 shortcut 仍存在；
新操作只增加一般难度；
复合退化破坏语义；
所谓 probe identity 只是人工 corruption taxonomy 的改名。
```

当前 `build_corruption_skew_augmix_loaders()` 也确实先读取 prepared corrupted image，再由
RAHFL AugMix 生成视图；普通 AugMix 没有先恢复 clean/private semantic anchor。故不能把
已有 AugMix views 当作满足 overwrite 假设的 bridge。

## 2. 最小新增信息：公共退化中和器

### 2.1 为什么使用公共数据

当前异构联邦框架已经使用 CIFAR-100 public data 做 logit communication。PNCB 不要求新增
私有 clean counterpart，而是复用这份公共输入空间来学习一个所有异构客户端都能调用的
图像空间桥。这样避免在 ResNet、ShuffleNet、MobileNet 的不可比 feature space 中对齐
“shortcut direction”。

公共图像的类别标签不参与 canonicalizer 训练。对公共图像 `U`，从一个连续、复合、与类别
无关的随机退化分布 `Q` 抽 recipe `A`：

\[
\widetilde U=A(U),\qquad A\sim Q.
\]

训练中和器 `C_phi`：

\[
\min_\phi\;
\mathbb E_{U,A}
\left[
\|C_\phi(\widetilde U)-U\|_1
+\lambda_s(1-\operatorname{SSIM})
+\lambda_f\|\Phi(C_\phi(\widetilde U))-\Phi(U)\|_2^2
\right].
\]

首个 Kill Test 可只用 `L1 + SSIM`，不必立即引入大型 perceptual backbone。`Q` 使用随机链、
随机强度和随机混合权重，不输出 family/type 标签，也不学习 `type -> prompt` 分类器。

### 2.2 私有训练时的 paired path

对私有坏图 `X`，冻结 `C_phi`，构造

\[
\bar X=C_\phi(X),\qquad
X^{(t)}=(1-t)X+t\bar X,\quad t\in[0,1].
\]

`t=0` 是原图，`t=1` 是中和端点。中间轨迹只用于 bridge 审计与可选单调性分析；最小训练
实现只需原图和端点，避免三倍以上前向成本。

这里“canonical”不等于恢复真实 clean ground truth。它只要求输出对原退化更不敏感，同时
保留任务语义。模糊、压缩等退化不可逆，因此任何无条件 clean-recovery 主张都是错误的。

## 3. Bridge 的最弱成立假设

设 `X=R(S,G)`，任务标签 `Y=h(S)`。理想中和分布从与标签独立的参考退化
`G^0 ~ q(G)` 生成 `X^0=R(S,G^0)`。

PNCB 需要以下条件：

```text
B1  semantic preservation:
    P[h(C_phi(X)) != Y] <= eta，或语义漂移有可审计上界。

B2  original-nuisance contraction:
    给定 S，C_phi(X) 对原 G 的条件分布差异至多 epsilon；
    它不必精确恢复 clean，但必须显著削弱旧 G 的可辨识性。

B3  label-independent reference:
    公共 recipe Q、canonicalizer 训练和私有调用均不依赖私有 Y、client 或 binding map。

B4  paired observability:
    客户端知道 X 与 C_phi(X) 来自同一样本；不需要持久 source ID。

B5  one-sided dominance / limited cancellation:
    有害 degradation-to-class promotion 在对参考退化平均时不会被等大的反向效应完全抵消。

B6  no final-test tuning:
    bridge 架构、Q、阈值和 loss 权重不能通过 CLE final test 选择。
```

`B2` 是不可删除的核心，`B5` 是 SCDW 与 family-specific DSA 之间的桥。若 B2 失败，方案
退化为普通增强；若 B5 失败，平均 withdrawal 可能漏掉方向相反但都很强的多个 shortcut。

## 4. 新数学对象：SCDW

### 4.1 Population risk

对客户端 `i` 和错误类别 `c`，定义 canonical withdrawal：

\[
\Delta_{i,c}(\theta)
=\mathbb E_{(X,Y)\sim P_i:Y\ne c}
\left[
p_\theta(c\mid X)-p_\theta(c\mid C_\phi(X))
\right].
\]

定义单向风险：

\[
\boxed{
\mathcal R_i^{\mathrm{SCDW}}(\theta)
=\frac1C\sum_{c=1}^{C}[\Delta_{i,c}(\theta)]_+^2
}.
\]

它问：**中和旧退化后，是否有某个错误类别的概率在大量异类样本上系统性下降？** 若是，
原退化向该类别提供了稳定证据；如果只是随机预测波动，带符号的总体均值会抵消。

它不需要知道退化 family、绑定集合、target-set 大小、私有 clean image 或客户端环境标签。

### 4.2 Confidence-calibrated minibatch estimator

对 minibatch `B`：

\[
d_{b,c}=p_\theta(c\mid X_b)
-\operatorname{sg}\left[p_\theta(c\mid C_\phi(X_b))\right],
\quad Y_b\ne c.
\]

令 `mu_hat_c`、`s_hat_c` 和 `n_c` 分别是这些差值的均值、标准差和有效样本数。使用单侧
置信下界：

\[
\widehat{\mathcal L}_{\mathrm{SCDW}}
=\frac1C\sum_c
\left[
\widehat\mu_c
-\operatorname{sg}\left(z_\alpha\frac{\widehat s_c}{\sqrt{n_c}}\right)
\right]_+^2.
\]

首版固定 `z_alpha=1.645`（单侧95%），不把它当调参旋钮。对标准误 stop-gradient，避免模型
通过操纵方差改变门槛；对 canonical probability stop-gradient，避免 SCDW 通过抬高中和图
上的错误类概率获得假性下降。

### 4.3 最小本地目标

\[
\mathcal L_i=
\mathcal L_i^{\mathrm{RAHFL-local}}
+\lambda_a\operatorname{CE}(f_\theta(C_\phi(X)),Y)
+\lambda_d\widehat{\mathcal L}_{\mathrm{SCDW}}.
\]

其中原始 `AugMix/JSD/DCL` 完全保留。`CE(C_phi(X),Y)` 是必要的 bridge supervision，确保
中和视图仍学习真实标签；它必须作为独立 `bridge-only` 对照出现，否则无法判断收益来自
restoration augmentation 还是方向风险。

## 5. 理论链条与不能夸大的地方

定义理想的 label-independent neutral risk：

\[
\mathcal R_i^{0}(\theta)
=\frac1C\sum_c
\left[
\mathbb E_{Y\ne c}
\{p_\theta(c\mid R(S,G))-p_\theta(c\mid R(S,G^0))\}
\right]_+^2.
\]

因为概率在 `[0,1]`，若 B1--B3 成立，`C_phi(X)` 与理想 `R(S,G^0)` 的条件分布误差受
`epsilon + eta` 控制，则存在只依赖类别数的常数 `K`，使

\[
|\mathcal R_i^{\mathrm{SCDW}}-\mathcal R_i^0|
\le K(\epsilon+\eta).
\]

在 B5 的 one-sided dominance 下，family-specific DSA 的正向概率流不会在对 `G` 平均时
完全抵消，因此降低 `R_i^0` 会降低 DSA 的一个覆盖部分，并留下由 cancellation、bridge
误差和语义误差组成的余项。

这条理论链解释“为什么可能降低 CFG/DSA”：

```text
公共 paired reconstruction
 -> 原退化信息收缩
 -> 原图相对中和图的 class-directed evidence 可观测
 -> 只惩罚统计可信的错误类正向 evidence
 -> 减少 corruption -> wrong-class shortcut
 -> 保留 AugMix/JSD 对一般 corruption robustness 的收益
```

不能声称 SCDW 无条件上界所有 DSA。没有 B2/B5 时，多种方向可能抵消，canonicalizer 也
可能产生新 artifact。该边界必须写在论文定理与实验限制中。

## 6. 为什么它不是已有模块的换名

| 方法 | 核心机制 | 与 PNCB-SCDW 的关键差异 |
|---|---|---|
| AugMix/JSD | 同一样本随机增强输出的对称一致性 | SCDW 是跨样本、类别条件、单向错误类证据；bridge 先中和旧退化 |
| 固定 corruption bank | 对人工列举的操作逐一训练/打标签 | PNCB 不输出 type/family；公共 `Q` 是随机复合 recipe，私有风险只有原图/中和图 |
| image restoration | 提高重建质量或直接预处理分类输入 | restoration 只是 bridge；论文核心必须由 `bridge-only` 对照证明 SCDW 的额外作用 |
| ShortcutProbe (IJCAI 2025) | held-out probe set 上学习低秩 latent shortcut subspace，post-hoc 重训最后层 | SCDW 不学习 shortcut detector、不需要 held-out probe set、不投影 latent feature，端到端作用于任意异构 backbone 的概率输出 |
| FedCD (2024) | feature mask/intervener、梯度不变性和 risk-extrapolation aggregation | SCDW 不对齐 feature/gradient，不改 aggregation；对象是 corruption withdrawal 的 class-directed probability flow |
| BiaSwap (ICCV 2021) | 用 biased classifier 排序 bias-guiding/contrary 样本并做 style swap | PNCB 不识别 easy bias 样本、不做跨类 style donor swap，只对公共合成退化学习无标签中和 |
| Nuisances via Negativa (TMLR 2024) | 故意破坏 semantic 来暴露 nuisance | PNCB 的必要条件恰好是 semantic preservation，并撤回 nuisance |
| DDB (ICCV 2025) | 文本反演、分割和 diffusion 生成 minority groups 以平衡组 | SCDW 无 group/minority generation、无文本提示、无样本重平衡 |
| counterfactual shortcut analysis (MIDL 2026) | 已知 protected attribute，训练因果生成模型并在 latent 中移除属性做 post-hoc 评价 | SCDW 面向未知退化、无需属性标签、用于本地训练；该工作仍是强邻接，不能泛称首个 counterfactual shortcut metric |
| SilverLining (WACV 2026) | 定位并消除 spatial/spectral shortcut，同时防止 removal artifact | SCDW 不定位空间/频率 mask；但 canonicalizer artifact 风险相同，必须有 clean-null 与 bridge-only 对照 |

因此，论文不能把“使用 restoration”“无 group label”“counterfactual debiasing”单独写成
创新。可能成立的新核心必须是以下组合：

```text
HFL 公共输入空间生成可共享的 nuisance-neutral bridge
+ hidden corruption 下的 class-directed withdrawal estimand
+ confidence-calibrated one-sided training objective
+ CLE/DSA 对“robust but right for the wrong reason”的配对评价
```

## 7. Bridge-only Kill Test：先杀 bridge，再谈训练

下一步不应立即跑12轮分类训练。先只训练一个小型 public canonicalizer，并复用已有
H0/H9/L0/L9 checkpoint 做推理审计。

### 7.1 三个 bridge 对照

```text
I  Identity:              C(X)=X，负对照；
A  AugMix overlay:        直接在坏图上继续增强；
P  Public canonicalizer:  公共 paired reconstruction 后的中和端点。
```

### 7.2 只在最终审计打开的 oracle 信息

训练 canonicalizer 和计算 SCDW 时不读取 private family/binding。clean source、family 和
binding 只用于 bridge quality 与 retrieval 的最终评分。

### 7.3 必须报告

```text
Q1 semantic preservation:
   中和前后 task-label accuracy、clean-source feature distance、明显语义破坏率。

Q2 nuisance contraction:
   同一 clean source 跨 family 的表示距离是否收缩；
   原 family 从 canonical output 中的可辨识度是否显著下降。

Q3 directional observability:
   H9/L9 的 observable SCDW 是否高于 H0/L0；
   M 矩阵生成后才按 family 分层，做 binding retrieval mAP/hit 的 oracle 归因。

Q4 overlay comparison:
   Public canonicalizer 必须比 AugMix overlay 更强地收缩旧 family 信息。

Q5 artifact null:
   clean source 相对 canonical-clean 的 max-arm SCDW 不应超过0.03。

Q6 client consistency:
   至少3/4客户端同向；结果不能只由一个 architecture 驱动。
```

### 7.4 Draft promotion gates

门槛在实现前冻结，建议起点为：

```text
semantic accuracy drop <= 1.0pp
within-source family-distance contraction >= 25%
family separability reduction >= 30% relative
H9 and L9 binding mAP >= 0.65
gamma9 - gamma0 mAP delta >= 0.20
class-to-family hit >= 0.70
positive clients >= 3/4
clean-vs-canonical-clean max-arm SCDW <= 0.03
```

若 PNCB 不能同时通过 semantic preservation 和 nuisance contraction，整条方法 NO-GO。
不能通过调 SCDW 权重救 bridge。

## 8. 通过 bridge gate 后的最小训练归因

只有 bridge gate 通过，才允许一个短程 matched A/B/C：

```text
B0  原 RAHFL local objective
B1  B0 + PNCB canonical-view CE            （bridge-only）
B2  B1 + SCDW                              （完整候选）
```

所有 arm 匹配初始化、partition、fit/audit、AugMix RNG、public batches、路由与评价。首轮只跑
`gamma=0.9, seed=0, 12 rounds`；若完整候选不能明显优于 bridge-only，SCDW 方法核心 NO-GO。

首轮重点不是仅看 Avg，而是：

```text
DSA / CFG:        完整候选是否真的减少 directional shortcut；
WCCA / Worst:     counter-binding 组是否恢复；
Avg:              是否出现不可接受的语义损伤；
SCDW:             训练对象是否下降；
B2 - B1:          新风险对象是否有独立贡献；
B1 - B0:          restoration augmentation 本身贡献多少。
```

建议的 paper promotion 条件是：`B2` 相对 `B1` 同时降低 DSA/CFG、改善 WCCA，且 Avg 不下降
超过1pp。若 `B1` 已包含全部收益，论文只能得到“public restoration preprocessing”，不应把
SCDW 写成方法贡献。

## 9. 实现与成本判断

### 9.1 最小 canonicalizer

首个 gate 不需要 PromptIR、diffusion 或大型 foundation model。优先使用一个小型 U-Net/
residual autoencoder，在 public CIFAR-100 上在线生成 compound recipes 做 paired reconstruction。
它的目的不是刷新 image restoration SOTA，而是验证 B1/B2 是否可能同时成立。

如果小模型失败，不立即换大型扩散模型救场。只有失败明确来自容量不足、而非不可逆信息与
语义--退化纠缠，才讨论 PromptIR/one-step diffusion；否则工程复杂度会吞掉论文主线。

### 9.2 预计额外成本

```text
public canonicalizer pretrain:  一次性，独立于四个 client backbone；
private preprocessing:         可离线缓存 C_phi(X)，不进入梯度；
classifier forward:            bridge-only 约增加1个 endpoint forward；
communication:                 0新增；
private metadata:              0新增；
```

因此 OpenI 比本地 RTX 3050 更合适；但当前阶段尚未批准任何运行。

## 10. 论文价值与主要风险

| 维度 | 当前评分 | 理由 |
|---|---:|---|
| 新颖性 | 3.0/5 | 单个模块均有先例，组合与 SCDW 对象有差异，但 ShortcutProbe/FedCD/MIDL-2026 邻接强 |
| 理论合理性 | 3.8/5 | 不可识别性逼出了必要 bridge，风险可在 B1--B5 下解释；无条件保证不存在 |
| 实现成本 | 3.0/5 | 小 canonicalizer + 一个额外 forward 可控；大型 restoration 会迅速失控 |
| 实验归因 | 4.5/5 | Identity/AugMix/bridge-only/full 四层对照能清楚分离原因 |
| HFL 适配 | 4.0/5 | 输入/概率空间与 backbone 无关，不需要共享参数或 feature |
| FL-specific novelty | 2.5/5 | failure 是 local-first，方法也可用于 centralized；不能强行包装通信贡献 |
| CCF-B 潜力 | CONDITIONAL | bridge gate 和 B2-B1 独立增益都通过后才有资格进入论文主线 |

最大风险不是代码，而是 PNCB 能否在32x32、severity 1--5、复合 AugMix 情况下同时做到：

```text
削弱旧退化信息；
保住类别语义；
不制造新的 class-directed restoration artifact。
```

这是下一步必须优先回答的科学问题。

## 11. 当前决策

```text
PNCB-SCDW paper design:       CONDITIONAL GO
training implementation:     NOT YET
communication modification:  NO
PEW/BER revival:              NO
next action:                  freeze bridge-only Phase-B0 spec
```

只有用户确认上述对象和 Kill Test 后，才实现 canonicalizer harness、离线缓存和零训练分析器。

## 12. 主要外部依据

- [ShortcutProbe（IJCAI 2025）](https://www.ijcai.org/proceedings/2025/795)
- [Reducing Spurious Correlation for Federated Domain Generalization / FedCD（2024）](https://arxiv.org/abs/2407.19174)
- [BiaSwap（ICCV 2021）](https://arxiv.org/abs/2108.10008)
- [Nuisances via Negativa（TMLR 2024）](https://arxiv.org/abs/2210.01302)
- [DDB: Diffusion Driven Balancing（ICCV 2025）](https://openaccess.thecvf.com/content/ICCV2025/html/Parast_DDB_Diffusion_Driven_Balancing_to_Address_Spurious_Correlations_ICCV_2025_paper.html)
- [Evaluating Shortcut Utilization through Counterfactual Analysis（MIDL 2026）](https://proceedings.mlr.press/v301/vigneshwaran26a.html)
- [SilverLining（WACV 2026）](https://openaccess.thecvf.com/content/WACV2026/html/Unnikrishnan_SilverLining_Data-First_Mitigation_of_Spatial_and_Spectral_Shortcuts_Without_Introducing_WACV_2026_paper.html)
- [PromptIR（NeurIPS 2023）](https://proceedings.neurips.cc/paper_files/paper/2023/hash/e187897ed7780a579a0d76fd4a35d107-Abstract-Conference.html)
- [Adaptive Blind All-in-One Image Restoration（2024）](https://arxiv.org/abs/2411.18412)
