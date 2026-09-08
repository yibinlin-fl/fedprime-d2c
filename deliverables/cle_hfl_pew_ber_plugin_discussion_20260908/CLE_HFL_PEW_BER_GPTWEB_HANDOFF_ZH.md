# CLE-HFL → DSA → Local-first → PEW+BER：网页端 GPT 插件扩展讨论稿

日期：2026-09-08

## 0. 这份文档的用途

本文件用于和网页端 GPT 讨论：在不篡改既有负结果、不重复已有模块的前提下，PEW+BER
这个本地插件还能否增加一个足以支撑 B 类会议投稿的研究贡献。

当前只允许进行研究设计和论证，不授权修改代码、训练模型、运行 OpenI、补 seed 或启动
Formal。任何新候选都必须先给出数学对象、可识别信息、最弱假设、与既有方法的区别和低成本
Kill Test，再由用户确认是否实现。

---

## 1. 整体研究链条

```text
① CLE-HFL 场景
   类别与 corruption 在不同客户端形成不同映射
                         ↓
② Paired counterfactual + DSA
   证明模型确实利用 corruption 定向预测绑定类别
                         ↓
③ HFL-vs-Local 归因
   证明 shortcut 在本地训练中已经形成，通信并非唯一成因
                         ↓
④ PEW+BER 本地插件
   识别粗环境，并在每个类别内部平衡预测环境风险
```

这四步构成：

```text
定义问题 → 证明问题 → 定位干预位置 → 验证一个针对性缓解方案
```

必须避免把第四步单独包装成全新 HFL 框架。PEW+BER 的算法创新有限，论文价值需要由完整
问题—诊断—机制—干预链条以及更充分的跨场景/跨通信证据共同承担。

---

## 2. ① CLE-HFL 场景

### 2.1 普通 corruption 与 CLE 的区别

普通 corruption-robust HFL 通常令损坏类型与任务类别近似独立：

\[
P(E=e\mid Y=c) \approx P(E=e).
\]

损坏主要降低图像质量，但不稳定地指示类别。

CLE 则显式令客户端 `k` 内的类别 `c` 与损坏环境 `e` 强相关：

\[
P_k(E=e\mid Y=c) \neq P_k(E=e),
\]

并且不同客户端具有不同映射：

\[
d(k,c) \neq d(k',c).
\]

于是 corruption 不只是 nuisance，也成为预测类别的捷径。例如：

```text
client 0: airplane → blur,  car → weather, bird → digital
client 1: airplane → weather, car → digital, bird → noise
```

模型可能学习 `blur ⇒ airplane`，而非飞机的语义结构。

### 2.2 CLE-v1 的受控生成

Phase-A0/A1a 使用四个 corruption family：

```text
noise:    gaussian_noise, shot_noise, impulse_noise, speckle_noise
blur:     defocus_blur, glass_blur, motion_blur, zoom_blur
weather:  snow, frost, fog, spatter
digital:  contrast, brightness, jpeg_compression, pixelate
```

客户端 `k`、类别 `c` 的 dominant family 为：

\[
d(k,c)=g_{(c+k)\bmod 4}.
\]

family 采样概率：

\[
P(E=g\mid k,c)
=
\frac{1-\gamma}{4}
+
\gamma\mathbf 1\{g=d(k,c)\}.
\]

- `gamma=0`：四个 family 均为 `0.25`，类别与 family 独立；
- `gamma=0.9`：dominant family 概率 `0.925`，其他 family 各 `0.025`；
- family 内再采样 concrete operator；
- 每张训练图独立采样 severity 1--5；
- 4 个客户端，CIFAR-10，Dirichlet `alpha=0.5`；
- 模型分别为 ResNet10、ResNet12、ShuffleNet、MobileNetV2。

`gamma=0.9` 不是 100% 固定绑定，仍保留 7.5% counter-binding 样本。

### 2.3 CLE-HFL v2

PEW+BER 的正式主表使用 `cle_hfl_v2/alpha05_gamma09_seed0_split0`。v2 将绑定细化为
client-specific、class-specific concrete operator map，并将 operator 划分为 seen/unseen，支持
更细粒度的 operator-cell、WCCA、CFG 和 operator-level LOO 评估。

训练方法看不到 operator ID、family、severity 或 seen/unseen 标记；这些信息只用于数据生成、
完整性检查和最终报告。

### 2.4 场景的新颖性边界

可以讨论的新点：

- 在模型异构 FL 中构造客户端特定的 class-corruption mappings；
- 同时控制 client、class、family/operator、CLE 强度与模型架构；
- 用配对反事实而非单纯掉点验证 shortcut；
- 提供 operator-cell 与 directional harm 评估。

不能声称：

- 首次发现类别与 corruption 会形成虚假相关；
- 首次研究模型异构 FL 的数据损坏；
- 当前合成场景完整代表真实世界退化。

更稳妥的表述：

> To the best of our knowledge, we present the first systematic study of client-specific
> class-corruption entanglement in model-heterogeneous federated learning.

该表述仍需在投稿前完成正式文献检索。

---

## 3. ② Paired counterfactual + DSA

### 3.1 为什么准确率、WCCA、CFG 不足以证明 shortcut

corrupted accuracy 下降可能来自：

1. corruption 破坏了真正的语义信息；
2. 模型具有普通 corruption fragility；
3. 模型把 corruption 当成类别线索。

只有第三项是 CLE directional shortcut。Avg/Worst/WCCA/CFG 能说明性能损害，却不能单独区分
上述三种原因。

### 3.2 冻结配对反事实网格

Phase-A0 使用：

```text
clean sources:        1,000（每个 CIFAR-10 类别 100 张）
operators:            16（四个 family 各四个）
severity:             固定为 3
generation seed:      hash(source_id, operator_id, 20260830)
evaluation images:    1,000 × 16 = 16,000
gamma conditions:     0 与 0.9
clients/models:       4
```

同一个 source image 在 16 个 operator 下被配对评估。语义标签、source identity 和 severity
保持不变，只有 corruption operator 改变。

### 3.3 DSA 数学定义

令客户端 `k` 中与 family `g` 绑定的类别集合为：

\[
\mathcal C_{k,g}=\{c:d(k,c)=g\}.
\]

模型在输入 `z` 上分给这些绑定类别的总概率为：

\[
Q_{a,k,g}(z)
=
\sum_{c\in\mathcal C_{k,g}}p_{a,k}(c\mid z),
\]

其中 `a` 表示训练 CLE 强度条件。

对真实标签不属于 `C_{k,g}` 的 source，比较施加 family `g` 与施加其他 family 后，预测概率
是否定向流向 `g` 所绑定的类别：

\[
\operatorname{DSA}_{a,k,g}
=
\mathbb E_{(x,y):y\notin\mathcal C_{k,g}}
\left[
\frac{1}{|\mathcal O_g|}\sum_{o\in\mathcal O_g}Q_{a,k,g}(T_o(x))
-
\frac{1}{3}\sum_{h\neq g}
\frac{1}{|\mathcal O_h|}\sum_{o\in\mathcal O_h}Q_{a,k,g}(T_o(x))
\right].
\]

客户端 DSA 对四个 family 等权平均，总体 DSA 再对四个客户端等权平均。主对比为：

\[
\Delta\operatorname{DSA}
=
\operatorname{DSA}_{\gamma=0.9}
-
\operatorname{DSA}_{\gamma=0}.
\]

直观上，它问：

> 当语义不变、只施加 family `g` 时，模型是否把概率系统性地推向训练中与 `g` 绑定的类别？

### 3.4 随机绑定对照

保持每个 family 绑定的类别数量不变，随机置换训练绑定图，并在同一预测缓存上重算 DSA。
如果真实绑定图的 DSA 超过随机图的 95th percentile，才说明方向与训练绑定特异一致，而不是
模型天然偏向某些类别。

### 3.5 Phase-A0 正式结果

```text
gamma0 pooled DSA:   -0.0003018894
gamma0.9 pooled DSA:  0.2013210658
delta DSA:            0.2016229552
paired CI95:          [0.1964123272, 0.2072188988]
positive clients:     4/4
shuffled-map p:       0.000999001
```

辅助性能同时恶化：

```text
Avg:       52.2422 → 46.8250
Worst:     43.4188 → 37.9375
WCCA:      36.0000 → 19.3125
CFG:        2.6500 → 11.8813
flip rate:  0.1483 → 0.3534
```

允许结论：strong CLE 模型确实形成从 corruption family 到绑定类别的 directional shortcut。

不允许结论：Phase-A0 本身不能说明 shortcut 是通信、教师路由、模型异构或本地训练导致。

---

## 4. ③ HFL-vs-Local 归因

### 4.1 需要回答的问题

发现 DSA 后，需要区分：

- shortcut 是由客户端本地 class-corruption distribution 直接产生；
- 还是主要由 HFL 通信传播或放大。

“local-first”不表示联邦学习没有价值；它表示 shortcut 在通信前已经存在，因此只修改教师选择
或通信路由不足以消除根因，干预至少必须覆盖本地训练。

### 4.2 四臂 matched factorial

```text
H0: HFL        gamma=0.0
H9: HFL        gamma=0.9
L0: Local-only gamma=0.0
L9: Local-only gamma=0.9
```

四臂共同固定：

```text
CIFAR-10 CLE-v1, alpha=0.5, train seed=0
4 clients / 4 heterogeneous models
40 rounds, local epoch=1, batch=64
Adam, lr=0.001, weight_decay=0
AugMix/JSD + DCL
persisted strict fit/audit split
same initialization and local augmentation traces
```

唯一 H/L 差异：

- H0/H9：启用 strict AsymHFL-val public-logit communication；
- L0/L9：通信为 exact no-op。

所有本地梯度只使用 fit；HFL teacher routing 只使用 client-private audit；final test 只报告。

### 4.3 差分中的差分

令：

```text
S[m,g,k] = client k 在训练 arm (m,g) 下的 DSA
E[m,k]   = S[m,0.9,k] - S[m,0,k]
A[k]     = E[H,k] - E[L,k]
A_pool   = mean_k A[k]
```

`E[m,k]` 是 CLE effect；`A[k]` 才是 communication-induced amplification。

### 4.4 正式结果

```text
H0 DSA:           0.0015228341
H9 DSA:           0.2042704937
L0 DSA:           0.0008234010
L9 DSA:           0.2051892788

HFL CLE effect:   0.2027476596
Local CLE effect: 0.2043658778
A_pool:          -0.0016182182
CI95:            [-0.0033365882, 0.0001283891]
positive clients: 1/4
```

正式 verdict：`NO_GO_FL_SPECIFIC_AMPLIFICATION`。

准确解释：

> CLE 在 HFL 和 Local-only 中都产生约 0.204 的强 directional shortcut。strict AsymHFL
> 没有进一步稳定放大它，也没有在 40 轮时持续消除它。因此 shortcut 在本地训练阶段已经
> 形成，通信并非唯一成因；缓解方法必须作用于本地优化，但这不表示联邦协作不再需要。

不应再说“坏教师传播是主要故事”或“通信显著放大 CLE”。

### 4.5 为什么仍然需要联邦

任务仍然具有：

- 数据不可集中；
- 客户端类别覆盖和样本量不同；
- 客户端模型结构不同；
- 客户端仍希望利用跨客户端语义知识。

本地偏差的存在并不使联邦任务消失。正确问题是：

> 能否先抑制本地 shortcut，再保留异构联邦协作可能提供的语义收益？

当前缺失一个关键 factorial：`Local + PEW+BER`。因此尚不能严格拆出 PEW+BER 的纯本地增益
与 communication-after-mitigation 的额外增益。

---

## 5. ④ PEW+BER 本地插件

### 5.1 与 RAHFL 的继承关系

```text
RAHFL 底座
├── AugMix 图像增强
├── JSD 预测一致性
├── DCL 特征对比学习
├── strict AsymHFL-val 通信
└── 新增：PEW + BER 本地风险模块
```

PEW+BER 不是从零设计的新 HFL 框架。它保持通信不变，主要将普通 clean CE 替换为按
PEW pseudo-environment 构造的 BER。

### 5.2 PEW：Public Environment Witness

PEW 使用公共 CIFAR-100 carrier，按人工 corruption taxonomy 合成六类监督：

```text
clean, noise, blur, weather, digital, unknown
```

- 四个基础 family 从各自 operator pool 采样；
- unknown 是两个不同基础 family 的顺序复合；
- severity 为 1--5；
- 小型 CNN 输出六类环境、五类 severity 和 32 维 embedding；
- hard PEW 优化环境 CE + `0.25 × severity CE`；
- Adam，lr `1e-3`，5 epochs；
- 公共 validation environment accuracy 选择最佳 checkpoint；
- unknown threshold 只在公共 validation 上校准；正式 seed0 为 `0.0`；
- PEW 冻结后给每个私有训练样本产生 hard pseudo-environment `e_hat`。

训练阶段不读取私有 operator/family metadata。私有真值只用于实验后 PEW group accuracy 和
oracle 消融。

必须承认：PEW 使用人工公共 taxonomy，不是 taxonomy-free environment discovery。PEW 网络
本身属于低创新的小型环境分类器，其主要作用是为 BER 提供 side information。

### 5.3 BER：Balanced Environment Risk

对客户端 `k`、任务类别 `c`、PEW 环境 `e`，只在 strict fit subset 统计：

\[
n_{k,c,e}=\#\{i:y_i=c,\hat e_i=e\}.
\]

有效组要求：

\[
n_{k,c,e}\ge 2.
\]

类内环境权重：

\[
a_{k,c,e}
=
\frac{\min(n_{k,c,e},32)^{0.5}}
{\sum_{e'}\min(n_{k,c,e'},32)^{0.5}}.
\]

组风险：

\[
R_{k,c,e}
=
\frac{1}{n_{k,c,e}}
\sum_{i:y_i=c,\hat e_i=e}\ell_i.
\]

BER 对有效类别均匀、类别内按上述 support-shrunk 权重平均：

\[
L_{BER,k}
=
\frac{1}{|\mathcal C_k^{valid}|}
\sum_{c\in\mathcal C_k^{valid}}
\sum_e a_{k,c,e}R_{k,c,e}.
\]

它不是让“不同类别都预测对”，而是防止同一类别中样本最多的 corruption 环境垄断梯度。
它也不是 GroupDRO/CVaR，因为没有最大化最坏组风险。

### 5.4 与 AugMix/JSD/DCL 的区别

| 模块 | 操作单位 | 约束对象 | 没有直接解决什么 |
|---|---|---|---|
| AugMix | 单张图像 | 构造随机增强视图 | 类别内部环境支持失衡 |
| JSD | 同一样本的多个预测 | clean/strong/strong 预测一致 | 大组仍可按样本数主导训练 |
| DCL | 同一样本的多个特征 | clean/strong/weak 表示一致与类别判别 | corruption 仍可能成为类别聚类线索 |
| PEW | 每个私有样本 | 推测粗环境 | 本身不改变任务模型风险 |
| BER | 类别×预测环境组 | 改变不同组的风险贡献 | 依赖 PEW taxonomy 与伪标签质量 |

例子：若 `cat+blur=900`、`cat+noise=20`，JSD 可以让每张 `cat+blur` 的增强预测稳定，但
900 个样本仍然主导 ERM。BER 会分别计算 `cat+blur` 与 `cat+noise` 的组风险，再压缩大组的
支持优势。

### 5.5 完整本地目标

RAHFL-like baseline：

\[
L_{base}=L_{CE}+12L_{JSD}+L_{DCL}.
\]

PEW+BER：

\[
L_{local}=L_{BER}(\hat e_{PEW})+12L_{JSD}+L_{DCL}.
\]

通信仍为 strict AsymHFL-val。

### 5.6 exact PEW+BER 主结果

固定 CLE-HFL v2 `seed0_split0`、训练 seed0、12 轮 last-five：

| 方法 | Avg | Worst | WCCA | CFG |
|---|---:|---:|---:|---:|
| RAHFL | 30.0853 | 25.0427 | 0.8500 | 30.4400 |
| PEW+BER | 34.6320 | 29.4280 | 7.2500 | 24.6400 |

差值：

```text
Avg   +4.5467 pp
Worst +4.3853 pp
WCCA  +6.4000 pp
CFG   -5.8000 pp
```

PEW 诊断：

```text
private group accuracy:  62.21%
public val env accuracy: 57.40%
unknown AUROC:            0.8167
ECE:                      0.0341
NLL:                      1.0825
```

### 5.7 matched 消融

| Arm | 正确解释 | Avg | Worst | WCCA | CFG |
|---|---|---:|---:|---:|---:|
| A0 | RAHFL | 30.0853 | 25.0427 | 0.8500 | 30.4400 |
| A1 | calibrated hard PEW + hard BER | 34.6320 | 29.4280 | 7.2500 | 24.6400 |
| A2 | CDep only | 30.4070 | 24.7707 | 1.1500 | 30.7750 |
| A3 | PEW+BER+CDep | 34.0230 | 28.9467 | 5.9000 | 24.1200 |
| A4 | fixed threshold 0.55 PEW+BER | 33.5820 | 28.6040 | 5.0500 | 27.0700 |
| A5 | shuffled PEW labels + BER | 31.5437 | 26.0147 | 2.5500 | 37.3750 |
| A6 | oracle family + BER | 35.1200 | 30.7253 | 7.7000 | 20.6900 |

历史报告称 A1 为 `BER-only`，这是错误命名；A1 仍使用 learned PEW，只是关闭 CDep。

Oracle 更好说明 PEW 误差是瓶颈，但不能把 oracle metadata 用于可部署训练。

### 5.8 operator-level LOO

从公共 PEW train/validation 和私有 fit 中同时留出：

```text
impulse_noise, zoom_blur, fog, pixelate
```

Strict-LOO PEW+BER 相对 RAHFL：

```text
Avg   +4.9027 pp
Worst +6.2547 pp
WCCA  +4.6000 pp
CFG   -6.1100 pp
```

只允许解释为已知四个 family 内的 operator-level generalization，不能外推到未见 family、
任意复合 corruption 或现实域偏移。

### 5.9 当前证据归属限制

- exact PEW+BER 只有固定场景 training seed0 的 12 轮主证据；
- 历史 3-seed 与 40-round positive package 含 CDep，不是 exact 当前方法；
- 历史 A0/A1a DSA/local-first 来自 CLE-v1，而 PEW+BER 主表来自 CLE-HFL v2；
- 因此不能把机制与方法结果写成同一批端到端实验；
- 尚无 `Local + PEW+BER`，不能严格证明通信在 mitigation 后提供额外收益；
- 尚未验证 PEW+BER 跨多种通信规则稳定有效。

---

## 6. “插件”主张需要怎样验证

结构上不修改通信，不等于实验上已经证明可迁移。最低限度应考虑以下 factorial：

| 通信条件 | Base local objective | + PEW+BER |
|---|---|---|
| No communication | Local base | Local + PEW+BER |
| AugHFL communication | AugHFL base | AugHFL + PEW+BER |
| strict AsymHFL-val | RAHFL | RAHFL + PEW+BER |

每一对必须匹配数据、初始化、训练 seed、fit/audit、AugMix/JSD/DCL、轮数和评估。报告每种通信
条件内部的：

\[
\Delta_m
=
Metric(m+PEW/BER)-Metric(m).
\]

需要观察 Avg、Worst、WCCA、CFG、DSA 和不同架构方向。

可能结论：

- 三种条件都改善：支持 `communication-compatible local plugin`；
- 只有 AsymHFL 改善：只能称 RAHFL-specific extension；
- Local 改善、HFL 无额外收益：联邦必要性受到质疑；
- Local/HFL 均改善且 HFL+plugin 优于 Local+plugin：最支持“本地纠偏 + 联邦语义协作互补”。

这个 factorial 能加强插件证据，但本身主要是泛化验证，不自动产生新的算法创新。

---

## 7. 已冻结负结果：不得换名字复活

网页端 GPT 提出的扩展不得只是以下路线的改名、调权重、换阈值或补 seed：

### 7.1 PEW/BER 直接变体

- CDep：matched 消融无稳定独立贡献，已从最终方法移除；
- multi-label PEW + Soft-BER：正式 `NO-GO`，0/4 last-five gates；
- WEC-BER：public pooled Q 满秩，但 operator cross-fit error transfer 失败；
  median/p75 L1=`0.3093/0.5267`，冻结门槛 `0.25/0.35`，不得通过删 operator 或按 family
  选择性保留来救；
- 固定阈值和 shuffled PEW 已有消融，不应作为新方法。

### 7.2 Taxonomy-free public-response 路线

- K0-B generic probe detector：只保留为离线审计，不进入训练；
- CVRS：MobileNetV2 上 proxy 下降，但真实 DSA 比 Public-JSD 更高；
  `DSA_JSD-DSA_CVRS=-0.013 < +0.02`；
- CRSF/K1-C-Minimal、P2/P3/P4 targeting、SDMN、CDR-SNR、PNCB/SCDW 等已有冻结负结果；
- generic public-response proxy 的下降不能保证真实 CLE harm 同步下降。

### 7.3 其他永久冻结路线

```text
D2C / Oracle D2C
FedPRIME-PAIR / CPAD
PRAC-HFL communication
FedCARA v1 communication
FedCLEAR / PCCD
EBST / EBST-v2
FedFalsify
FedCIS
handcrafted taxonomy-free continuous witness
C3R / CRSF-style local proxy
```

新方案如果只是在这些对象上修改 rank、threshold、loss weight、temperature 或名称，应直接
判为重复路线，而不是重新训练。

---

## 8. 新插件组件必须回答的问题

任何拟加入 PEW+BER 的组件必须先完整回答：

1. **数学对象与目标**：新增变量、损失或估计量是什么？写出公式。
2. **最弱成立假设**：需要 taxonomy、paired view、设备元数据、恢复器还是公共标签？
3. **可识别信息**：什么训练时可观测量能识别真正有害的 class-corruption routing？
4. **与现有模块区别**：为何不是 JSD、DCL、BER、Public-JSD、CDep、Soft-BER、WEC、CVRS、
   CRSF 或 GroupDRO 的重述？
5. **理论合理性**：为什么优化该对象应降低 DSA，而不只是降低一个相关 proxy？
6. **反例防御**：如何面对“proxy 下降但 MobileNetV2 DSA 更差”的 CVRS 反例？
7. **最小 Kill Test**：能否先用 existing predictions/checkpoints 做零训练或单步测试？
8. **归因风险**：提升是否可能只是更多增强、更大计算量或重新调参？
9. **实现成本**：需不需要新数据、checkpoint、GPU、OpenI 或完整 HFL？
10. **论文价值**：是新方法核心、插件泛化证据，还是只能成为额外消融？

### 一票否决条件

- 训练阶段读取 private operator/family/binding 或 final-test labels；
- 用 DSA oracle 调参数；
- 只因为 Avg 提升就宣称降低 shortcut；
- 只在 ResNet10 有效、MobileNetV2 失败却用 pooled mean 掩盖；
- 没有比 Public-JSD 更强的 matched 对照；
- 依赖未说明的强 taxonomy 或 clean counterpart；
- 只有更复杂的损失，没有新的可观测信息或可识别对象。

---

## 9. 当前最重要的研究分岔

网页端 GPT 应明确区分两种工作，不要混为一个“新模块”：

### 路线 A：验证插件可迁移性

做 `NoComm/AugHFL/AsymHFL × Base/PEW+BER` factorial。价值是强化插件主张、回答为什么仍然
需要联邦；它不是高算法创新。

### 路线 B：增加真正的新方法对象

必须引入能更接近有害 pairwise class routing 的可观测信息，或明确增加一个现实可获得的
side-information 假设。不能再依赖未经验证的 generic response scalar，也不能把 PEW 的
operator-unstable confusion 当作可迁移通道。

如果路线 B 找不到通过可识别性和查重的对象，就应把论文定位为 CLE-HFL benchmark +
mechanism study + taxonomy-assisted empirical baseline，而不是继续堆模块。

---

## 10. 请网页端 GPT 完成的任务

将本文件上传网页端 GPT 后，可以直接发送以下提示词：

```text
请把自己当作严格的 B 类会议审稿人和方法设计合作者，审查这份 CLE-HFL → DSA →
Local-first → PEW+BER 项目交接。

目标不是随便给 PEW+BER 再加一个 loss，而是判断是否存在一个有论文价值、可识别、且没有被
项目冻结负结果覆盖的新插件组件。

请依次输出：
1. 对当前四段研究链条最强的五个审稿攻击；
2. PEW+BER 相对 AugMix/JSD/DCL 的真实非冗余部分与仍然重叠的部分；
3. “communication-compatible plugin”是否需要 NoComm/AugHFL/AsymHFL 三组 factorial；
4. 最多三个候选扩展。每个候选必须给出数学对象和优化目标、最弱假设、训练时可观测信息、
   为什么能识别真正有害的 CLE routing、与 Public-JSD/PEW/BER/CDep/Soft-BER/WEC/CVRS/
   CRSF 的区别、理论合理性、零训练或低成本 Kill Test、实现成本和论文价值；
5. 若没有候选能通过，请明确回答“不要再加模块”，并给出 benchmark/mechanism paper 的最小
   补实验清单；
6. 单独评估 CLE-v1 的 DSA/local-first 证据与 CLE-HFL v2 的 PEW+BER 证据之间如何建立严谨
   bridge，不能把两者误写成同一次实验；
7. 不得复活文档中的冻结路线，不得建议只调权重、阈值、temperature、rank 或补 seed 来救。

当前只做研究设计，不写代码、不启动训练。对无法证明的新颖性必须标记为待正式检索，禁止使用
“首次”“SOTA”“taxonomy-free”等未经支持的表述。
```

---

## 11. 当前项目状态

```text
CLE directional shortcut:                  ESTABLISHED
FL-specific communication amplification:   NO-GO
supported mechanism:                       LOCAL-FIRST
exact PEW+BER on fixed CLE-v2 seed0:        POSITIVE
PEW+BER as sole B-class method novelty:     INSUFFICIENT
PEW+BER as taxonomy-assisted plugin:        SUPPORTED ON ASYMHFL ONLY
cross-communication plugin claim:           NOT YET TESTED
exact PEW+BER multi-seed / 40-round:         NOT AVAILABLE
active training or OpenI experiment:        NONE
```

下一步必须先完成网页端研究审查。未经用户明确同意，不实现候选、不运行付费/长时/Formal/
多种子/完整 HFL 实验。
