# CLE-HFL 完整论文证据与网页端 GPT 初稿交接文档

创建：2026-09-11；最新更新：2026-09-12
用途：交给网页端 GPT 进行论文结构审查、创新性讨论和初稿生成
当前阶段：DSA与BER机制理论均已升级；cross-binding-map map1 Formal已四门全过

---

## 0. 给网页端 GPT 的重要说明

请把这项工作理解为一篇以**问题、诊断、机制归因和实证缓解**为主的 CLE-HFL 论文，而不是
一篇以全新网络结构为中心的算法论文。

本文已经完成的核心链条是：

```text
模型异构联邦学习中的类别—损坏环境绑定（CLE-HFL）
  ↓
paired counterfactual + operator-level DSA
  ↓
证明模型沿训练绑定方向利用corruption猜类别
  ↓
HFL-vs-Local matched factorial
  ↓
证明shortcut主要在客户端本地训练阶段形成
  ↓
taxonomy-assisted PEW+BER本地缓解
  ↓
在AsymHFL与FedDF-fidelity两个通信底座上均显著降低DSA
  ↓
Oracle family/operator/random消融界定环境对应与粒度边界
  ↓
准确率收益并非跨底座一致：shortcut suppression与utility必须分开报告
  ↓
固定其余因素、只更换binding map的cross-scenario复现（S2完成，尚无科学结果）
```

必须保留以下区分：

```text
shortcut mitigation efficacy:      GO
cross-base DSA suppression:         GO on fixed CLE-v2 seed0 scenario
architecture-uniform utility:       NOT ESTABLISHED
universal lossless plugin claim:    NOT ESTABLISHED / frozen overall NO-GO
taxonomy-free claim:                FALSE
cross-scenario generalization:      NOT ESTABLISHED
```

不要把 Formal 的联合 `NO-GO` 误写成“PEW+BER 没有效果”；也不要因为 DSA 很强就把它改写成
“通用无损插件 GO”。联合 gate 和主科学终点必须同时、透明地报告。

---

## 1. 研究问题与动机

### 1.1 普通 corruption robustness 没有覆盖的问题

普通 corruption-robust 学习通常默认损坏环境 `E` 与任务类别 `Y` 近似独立：

\[
P(E=e\mid Y=c)\approx P(E=e).
\]

此时 blur、noise、weather 或 digital corruption 主要被视为应当抵抗的 nuisance。

CLE-HFL 考虑不同情况：在客户端 `k` 内，类别 `c` 与损坏环境 `e` 被定向绑定：

\[
P_k(E=e\mid Y=c)\neq P_k(E=e),
\]

而不同客户端的绑定又不同：

\[
d(k,c)\neq d(k',c).
\]

因此 corruption 不再只是降低图像质量，还可能成为类别捷径。例如某客户端训练集中多数猫图像
带 blur，多数狗图像带 noise，模型可能形成 `blur -> cat`，而不是学习稳定的猫语义。

### 1.2 为什么这是 HFL 问题

项目使用四个异构客户端模型：

```text
client0: ResNet10
client1: ResNet12
client2: ShuffleNet
client3: MobileNetV2
```

私有任务为 CIFAR-10，客户端数据采用 Dirichlet `alpha=0.5` 的非IID划分。类别分布、
class-corruption绑定、模型架构与通信共同作用。研究问题不是泛化地声称“联邦通信制造了捷径”，
而是：

1. 模型异构FL中能否形成与客户端训练绑定方向一致的corruption shortcut？
2. 该shortcut主要在本地学习形成，还是由通信制造/放大？
3. 能否在不读取私有真实corruption标签的前提下缓解它？
4. 缓解能否跨通信底座迁移，代价是什么？

### 1.3 新颖性边界

可以主张并由实验支持的差异化组合：

- 模型异构FL中的客户端特定 class-corruption directional binding；
- 保持source语义不变的operator-level paired counterfactual；
- 直接测量概率质量是否沿训练绑定方向移动的DSA；
- matched HFL-vs-Local机制归因；
- paired cancellation、binding specificity与source-level推断理论；
- 一个不改变通信协议的taxonomy-assisted本地缓解模块；
- 真实环境对应、环境粒度及跨通信底座的边界验证。

不能在未完成文献检索前声称：

- 首次研究联邦学习中的spurious correlation；
- 首次发现corruption与类别会产生虚假相关；
- PEW或BER的基础思想本身全新；
- 当前合成CLE完整代表真实世界损坏。

---

## 2. CLE-HFL 场景

### 2.1 CLE-v1：先导机制场景

早期 Phase-A0/A1a 使用四个 corruption family：

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

family 的受控采样为：

\[
P(E=g\mid k,c)
=\frac{1-\gamma}{4}
+\gamma\mathbf 1\{g=d(k,c)\}.
\]

- `gamma=0`：类别与family独立，每个family概率为0.25；
- `gamma=0.9`：dominant family概率为0.925，其余各0.025；
- family内采样concrete operator；severity为1--5；
- strong CLE仍保留7.5%的counter-binding样本，并非100%固定绑定。

Phase-A0 的 paired counterfactual 与 DSA 首先证明 strong directional shortcut 存在。该先导
发现不能因为后续某个训练方法失败而被否定。

### 2.2 CLE-HFL v2：当前正式闭环场景

当前正式机制与插件实验使用固定：

```text
scenario: cle_hfl_v2_paired_factorial_seed0_split0_v1
private task: CIFAR-10
clients: 4
private samples available: 40,000
Dirichlet alpha: 0.5
strong CLE: gamma=0.9
control: gamma=0
paired evaluation: 1,000 source × 15 operator × 4 client
evaluation severity: 3
```

CLE-v2 将绑定细化为 client-specific、class-specific concrete operator map，并区分seen/unseen
operator，支持operator-cell、WCCA、CFG、LOO及operator-level DSA。40,000是可用私有数据规模，
不代表每个模型在有限训练轮次中完整遍历40,000个不同样本。

训练方法看不到private operator ID、真实family、severity、seen/unseen标记或最终测试标签。
这些字段只用于数据生成、完整性审计、paired DSA和事后报告。

### 2.3 数据角色隔离

```text
private fit:   产生客户端本地梯度；PEW+BER的组计数只读fit
private audit: strict AsymHFL通信路由使用，不产生本地训练梯度
final test:    仅报告Avg/Worst/WCCA/CFG，不用于训练、路由、选择或调参
paired grid:   仅做最终operator-level counterfactual评价
public data:   训练PEW或支持既有公共通信，不含private task真值
```

该隔离是论文可信度的一部分。不得将旧实现中可能读取final-test标签的做法与当前strict协议混用。

---

## 3. Paired counterfactual 与 DSA

### 3.1 为什么准确率不足以证明shortcut

准确率下降可能来自普通corruption难度，也可能来自类别不平衡或模型容量。即便一个模型在强CLE
条件下准确率降低，也不能证明它把blur当成某个类别。

因此评价必须固定source image、语义标签和severity，只改变corruption operator：

\[
x\longrightarrow \{T_o(x):o\in\mathcal O\}.
\]

如果同一source换成operator `o` 后，模型概率系统性流向训练中与 `o` 绑定的类别，这才是
binding-specific directional shortcut。

### 3.2 operator-level DSA

对客户端 `k`、operator `o`，记训练中与其绑定的类别集合为 `B_{k,o}`。排除真实标签本身属于
该集合的source后，定义：

\[
\operatorname{DSA}_{k,o}(p)
=\mathbb E_x\left[
\sum_{c\in B_{k,o}}p(c\mid T_o(x))
-\frac{1}{|\mathcal O|-1}
\sum_{o'\neq o}\sum_{c\in B_{k,o}}p(c\mid T_{o'}(x))
\right].
\]

总体DSA对有效operator和客户端等权平均。它回答：

> 同一语义source只改变corruption operator时，模型是否把概率质量定向推向训练中与该
> operator绑定的类别？

DSA位于概率尺度。`0.12`表示约12个百分点的定向概率质量变化，不是0.12个百分点准确率。

### 3.3 shuffled-binding 对照

保持每个operator绑定类别数量不变，随机置换绑定图，在同一预测缓存上重新计算DSA。如果真实
绑定的DSA远高于随机绑定null，才能排除“某些corruption普遍推高某些类别”的替代解释。

---

## 4. Strong CLE directional shortcut：正式机制证据

### 4.1 四臂 matched factorial

固定CLE-v2场景、training seed 0、12 rounds、每客户端每轮最多16个local batches：

```text
h0_b: HFL,   gamma=0.0
h9_b: HFL,   gamma=0.9
l0_b: Local, gamma=0.0
l9_b: Local, gamma=0.9
```

四臂均使用同一 AugMix/JSD/DCL baseline；本阶段不加载或训练PEW，不启用BER/CDep。

### 4.2 正式结果

| Arm | Scope | Gamma | operator-grid Acc (%) | pooled DSA |
|---|---|---:|---:|---:|
| h0_b | HFL | 0.0 | 24.9300 | -0.000245 |
| h9_b | HFL | 0.9 | 21.4367 | 0.119644 |
| l0_b | Local | 0.0 | 25.5683 | -0.001914 |
| l9_b | Local | 0.9 | 22.4233 | 0.106046 |

| Estimand | Estimate | 95% source-bootstrap CI |
|---|---:|---:|
| HFL CLE effect | 0.119889 | [0.118050, 0.121562] |
| Local CLE effect | 0.107960 | [0.106145, 0.109678] |
| Communication add-on | 0.011929 | [0.011115, 0.012712] |

真实binding下的h9 HFL DSA为`0.119644`；shuffled-binding null p95为`0.029559`，置换检验
`p=0.000999`。所有预注册机制门槛通过：

```text
verdict: GO_CLE_V2_MECHANISM_STAGE1
```

允许结论：固定强CLE-v2场景中存在binding-specific directional shortcut，并伴随operator-grid
accuracy从24.93%降至21.44%。

限制：本结果是training seed 0、固定scenario、12轮正式机制证据；source-bootstrap只覆盖固定
训练结果下的source不确定性，不覆盖训练随机性或新CLE mapping。

---

## 5. HFL-vs-Local：shortcut 主要是 local-first

将两种通信条件下的CLE effect分别写为：

\[
\Delta_{HFL}=DSA(h9_b)-DSA(h0_b)=0.119889,
\]

\[
\Delta_{Local}=DSA(l9_b)-DSA(l0_b)=0.107960.
\]

pooled通信附加量为：

\[
\Delta_{comm}=\Delta_{HFL}-\Delta_{Local}=0.011929.
\]

描述性比例为：

\[
\frac{\Delta_{Local}}{\Delta_{HFL}}=90.05\%.
\]

这支持：HFL中观察到的大部分CLE效应，在没有通信的本地训练中已经形成；通信在pooled平均上
只增加较小附加量。因此最合理的干预位置是客户端本地目标，而不是首先重新设计通信。

必须避免两种误读：

1. 90.05%不是对每个样本的精确因果分解，而是两个matched contrast的描述性比值；
2. local-first不等于联邦学习不重要。联邦场景决定了多客户端、模型异构、非IID数据和共享知识
   交互；通信仍可能改变个别客户端表现，只是不是shortcut的主要起源。

---

## 6. DSA 的识别理论与五个可检验性质

### 6.0 DSA究竟识别什么

令`B_{k,o}`为客户端`k`中与operator `o`预先绑定的类别集合，并排除真实标签落入该集合的
source。定义：

\[
m_{k,o}(z,o')=\sum_{c\in B_{k,o}}p_k(c\mid T_{o'}(x)).
\]

若绑定质量可分解为operator-invariant语义基线和operator response：

\[
m_{k,o}(z,o')=s_{k,o}(z)+r_{k,o}(z,o'),
\]

则paired contrast中的`s`精确相消，DSA识别：

\[
E_z\left[r_{k,o}(z,o)-\frac{1}{|\mathcal O|-1}\sum_{o'\ne o}r_{k,o}(z,o')\right].
\]

也就是说，DSA识别的是**同一source跨operator时沿预先固定训练binding方向的概率响应差**。
它不是无条件恢复模型内部因果机制；成立依赖paired变换语义保持、binding预注册、source作为
配对统计单位，以及评价标签/operator/binding不进入训练或选择。

### 6.1 命题一：operator条件交换时DSA为零

若对同一source，模型预测对operator条件交换，即：

\[
p(\cdot\mid T_o(x))=p(\cdot\mid T_{o'}(x)),\quad \forall o,o',
\]

则DSA中的两项相消：

\[
\operatorname{DSA}(p)=0.
\]

缓存验证：exchangeable projection DSA为`-2.50e-20`；gamma0条件下经验最大绝对DSA为
`0.001914`。

### 6.2 命题二：DSA对预测概率混合线性

对：

\[
p_\lambda=(1-\lambda)p_0+\lambda p_1,
\]

因为DSA是预测概率的线性泛函：

\[
DSA(p_\lambda)=(1-\lambda)DSA(p_0)+\lambda DSA(p_1).
\]

当 `p_0` 是无shortcut预测器、`p_1` 沿绑定方向移动概率质量时，DSA随shortcut混合强度
单调增加。HFL和Local的11点缓存混合曲线均严格单调，最大仿射误差`2.78e-17`。

### 6.3 命题三：JSD低不推出DSA低

同一operator附近的三个增强视图可以满足：

\[
p(x_o)=p(a_1(x_o))=p(a_2(x_o)),
\]

从而JSD为0；但不同operator之间仍可沿不同绑定类别移动，因此DSA大于0。缓存反例中，三个
复制预测视图最大JSD为0，而h9 HFL DSA仍为`0.119644`。

这解释了为什么AugMix/JSD与PEW+BER不等价：JSD约束同一样本附近的增强一致性，DSA测量跨
operator的绑定方向。JSD可以保留一个稳定但错误的corruption-to-class shortcut。

### 6.4 命题四：paired cancellation与已知方向响应恢复

在概率单纯形中注入强度已知的binding-aligned response，11点曲线的最大恢复误差为
`6.66e-16`。向同一source的所有operator加入相同概率位移后，DSA变化仅`8.33e-17`，验证
operator-invariant项被paired contrast消除。同一受控响应的真实binding DSA为`0.241071`，
shuffled-binding null p95为`0.026786`，`p=0.000999`。

### 6.5 命题五：统计单位必须是source

同一source的15个operator图像不是15个独立样本。先在source内部形成directional contrast，
再以source为单位bootstrap。若source effect位于`[-1,1]`，`n=1000, delta=.05`的保守
Hoeffding半径为`0.085894`。主文bootstrap CI覆盖source抽样不确定性，不覆盖训练seed、
binding map或partition不确定性。

这些结果验证识别代数、受控恢复、经验零点和反例，不等于DSA已经被证明为跨所有场景的因果
充分统计量。其完整定理、证明边界和推断说明见仓库理论文档。

---

## 7. PEW+BER：taxonomy-assisted 本地缓解

### 7.1 与 RAHFL-like 底座的继承关系

```text
RAHFL-like底座
├── AugMix图像增强
├── JSD预测一致性
├── DCL特征对比学习
├── strict AsymHFL-val通信
└── 本文加入：PEW + BER本地风险模块
```

PEW+BER不修改服务器通信。在AsymHFL实验中，它主要以按伪环境分组的BER替换普通CE；JSD与
DCL继续承担样本级增强一致性和表征学习。

### 7.2 PEW：Public Environment Witness

PEW只使用公共CIFAR-100 carrier，按人工corruption taxonomy合成六类环境监督：

```text
clean, noise, blur, weather, digital, unknown
```

其中unknown由两个不同基础family顺序复合；severity为1--5。小型CNN输出六类环境、五类
severity和32维embedding，以环境CE加`0.25 × severity CE`训练；Adam、学习率`1e-3`、5 epochs。
最佳checkpoint及unknown threshold只由公共validation选择。PEW冻结后，为每个private fit样本
产生hard pseudo-environment `\hat e_i`。

PEW不读取私有真实operator/family metadata或最终测试标签。必须承认它依赖人工family taxonomy，
不是taxonomy-free discovery；PEW分类器本身的结构创新有限，主要提供BER所需side information。

### 7.3 BER：Balanced Environment Risk

对客户端 `k`、任务类别 `c`、PEW环境 `e`，只在strict fit subset统计：

\[
n_{k,c,e}=\#\{i:y_i=c,\hat e_i=e\}.
\]

有效组要求 `n_{k,c,e}\ge2`。类内环境权重为：

\[
a_{k,c,e}=
\frac{\min(n_{k,c,e},32)^{0.5}}
{\sum_{e'}\min(n_{k,c,e'},32)^{0.5}}.
\]

组风险为：

\[
R_{k,c,e}=\frac{1}{n_{k,c,e}}
\sum_{i:y_i=c,\hat e_i=e}\ell_i.
\]

最终：

\[
L_{BER,k}=
\frac{1}{|\mathcal C_k^{valid}|}
\sum_{c\in\mathcal C_k^{valid}}
\sum_e a_{k,c,e}R_{k,c,e}.
\]

BER不是“让所有类别都预测正确”，也不是直接优化最坏组的GroupDRO/CVaR。它压缩同一类别中
大环境组的样本数优势，使少数伪环境组不会完全被多数环境组淹没。

### 7.4 BER的有效分布与CLE失衡压缩定理

BER等价于在有效分布`Q_gamma`上训练，其中：

\[
Q_\gamma(\hat E=e\mid Y=c)=a_{c,e}^{(\gamma)}.
\]

对两个有效环境组：

\[
\frac{Q_\gamma(e_1\mid c)}{Q_\gamma(e_2\mid c)}=
\left(\frac{\min(n_{c,e_1},K)}{\min(n_{c,e_2},K)}\right)^\gamma.
\]

所以无cap区域的环境log-odds被压缩为原来的`gamma`倍。当前`gamma=.5,K=32,m=2`使有效
伪环境组最大质量比不超过4。若`gamma=0`且所有类别共享同一环境支持，则
`Q_0(\hat E|Y)=Q_0(\hat E)`，从而`Y`与伪环境独立，纯环境预测器不再获得类别优势。

固定strict-fit CPU审计验证了生产实现与有效分布公式（最大误差`7.32e-16`）。四客户端伪环境
TV依赖全部下降，pooled从`0.487505`降至`0.199744`（`-59.03%`）；只作报告的真实family TV
也在4/4客户端下降，pooled从`0.634914`降至`0.433513`（`-31.72%`）；真实family-only Bayes
advantage平均下降`23.25%`。

PEW误差下只有条件边界：

\[
TV(Q_{Y,E},Q_YQ_E)\le
TV(Q_{Y,\hat E},Q_YQ_{\hat E})+2P_Q(E\ne\hat E).
\]

当前BER有效分布中的PEW误差为`0.507--0.575`，故该上界为平凡的`1.0`，不能宣称真实环境
去相关的非平凡理论保证。分布压缩也不自动推出DSA为零或准确率提升，最终行为仍由Formal
paired DSA检验。

### 7.5 与 AugMix/JSD/DCL 的区别

| 模块 | 操作单位 | 直接解决的问题 | 未直接解决的问题 |
|---|---|---|---|
| AugMix | 单样本 | 构造随机增强视图 | 数据集级类别×环境支持失衡 |
| JSD | 同一样本多预测 | 增强前后预测一致 | 一致的shortcut仍可保留 |
| DCL | 同一样本多特征 | 表征一致与类别判别 | corruption仍可成为类别聚类线索 |
| PEW | 每个private样本 | 推测粗伪环境 | 本身不改变任务风险 |
| BER | 类别×伪环境组 | 平衡组风险贡献 | 依赖PEW taxonomy及伪标签质量 |

AsymHFL下的本地目标：

\[
L_{base}=L_{CE}+12L_{JSD}+L_{DCL},
\]

\[
L_{plugin}=L_{BER}(\hat e_{PEW})+12L_{JSD}+L_{DCL}.
\]

---

## 8. PEW+BER 在 AsymHFL 上的正式结果

### 8.1 协议

固定CLE-v2 `seed0_split0/gamma0.9`、training seed 0、12 rounds、每客户端每轮16 batches：

```text
h9_b = AugMix/JSD/DCL + strict AsymHFL-val
h9_p = h9_b + frozen public PEW + hard BER
CDep = disabled
```

两臂48条local batch/AugMix trace完全匹配。真实operator/binding只在最终DSA报告中使用。

### 8.2 结果

```text
pooled DSA: 0.119644 -> 0.041252
absolute reduction: 0.078392
relative reduction: 65.52%
source-bootstrap CI95: [0.077191, 0.079601]
client reductions: [0.111312, 0.011317, 0.115762, 0.075178]

last-5 plugin-minus-base:
Avg   +1.7323 pp
Worst -0.3760 pp
WCCA  +0.3500 pp
CFG   -9.2600 pp
```

冻结gate：

```text
I0 PASS
L0 PASS
P1 shortcut mitigation PASS
P2 full utility FAIL（仅Worst未达到预注册+1.0门槛）
formal verdict: NO_GO_PEW_BER_STAGE2_SEED0
```

科学解释应分层：

- shortcut mitigation efficacy：GO；
- Avg、WCCA、CFG方向积极；
- architecture/client-uniform utility：未建立；
- 不得事后删除Worst门槛并把冻结总判定改为GO。

零训练归因显示client2/ShuffleNet的operator-grid accuracy下降5.1733pp，但DSA仍下降0.115762。
该现象是类别选择性决策重分配，而不是所有类别均匀退化。由于架构与非IID客户端数据绑定，不能
将其因果归结为“小模型容量”。

### 8.3 新 binding map 的正式复现

为排除结论只依赖一张偶然 class-operator mapping，在固定partition seed0、evaluation seed、
training seed0、初始权重、公共数据、评价grid和同一冻结PEW的条件下，只更换客户端特定binding
map。map1 Formal结果为：

```text
Base DSA: 0.113761
PEW+BER DSA: 0.049531
reduction: 0.064230 (56.46%)
source-bootstrap CI95: [0.063105, 0.065328]
client reductions: [0.064195, 0.047682, 0.085614, 0.059428]
operator-grid pooled: 20.9450% -> 21.4583%
last-5 delta: Avg +0.7040, Worst +1.0760, WCCA +0.3500, CFG -9.8550
I0/L0/C1/C2: all PASS
verdict: GO_PEW_BER_CROSS_MAP1
```

因此可以主张shortcut形成和PEW+BER缓解已跨两张不同binding direction复现；不能写成跨新
partition、训练seed、corruption库、severity、数据集或真实场景泛化。该GO也不覆盖上节原
Stage-2的冻结整体NO-GO。

---

## 9. PEW+BER 在 native-CE FedDF-fidelity 上的跨底座结果

### 9.1 为什么要做这一实验

若只在RAHFL-like底座中加入PEW+BER，审稿人可能认为效果依赖AugMix/JSD/DCL或AsymHFL通信。
因此第二底座采用真正的原生CE两臂：

```text
fd_b = standard single-view CE + feddf_fidelity
fd_p = PEW-grouped hard BER-weighted CE + identical feddf_fidelity
```

两臂均禁用AugMix/JSD/DCL/CDep，FedDF公共蒸馏、初始化、私有batch及预算完全相同。唯一处理
差异是PEW分组与BER重加权。实现只能称protocol-matched `feddf_fidelity` adapter，不能称官方
完整FedDF recipe逐行复现。

### 9.2 正式结果

```text
pooled DSA: 0.136989 -> 0.029012
absolute reduction: 0.107977
relative reduction: 78.82%
source-bootstrap CI95: [0.106498, 0.109417]
client reductions: [0.150575, 0.074155, 0.053119, 0.154059]

operator-grid pooled: 18.4917 -> 18.4517, delta -0.0400 pp
last-5 delta:
Avg   -0.6660 pp
Worst +0.2147 pp
WCCA  -0.1000 pp
CFG   -13.2350 pp
```

冻结gate：

```text
I0 PASS
L0 FAIL（两臂operator-grid均低于冻结20%学习下限）
P1 shortcut mitigation PASS
P2 FAIL（operator-grid非劣子条件通过；last-5 Avg下降0.6660pp）
formal verdict: NO_GO_PEW_BER_FEDDF_PLUGIN_SEED0
```

该结果的正确解释：

1. PEW+BER的DSA抑制并不依赖AugMix/JSD/DCL，在第二通信底座上仍然很强；
2. pooled operator-grid accuracy只下降0.04pp，几乎持平；
3. last-5 Avg轻微下降，且逐客户端效用方向不一致；
4. 因此跨底座shortcut suppression成立，但“普适无损性能插件”不成立；
5. 去除shortcut后原绑定分布准确率轻微下降在概念上并不矛盾，因为shortcut本身在原分布中
   具有预测价值；但不能以此为理由隐瞒冻结utility gate。

---

## 10. Oracle family/operator/random 粒度边界

### 10.1 三臂定义

固定同一强CLE-v2场景，比较：

```text
Oracle family:   用private真实family做BER分组
Oracle operator: 用private真实operator做BER分组
Random operator: 在类别内打乱operator对应后做BER
```

Oracle只用于不可部署的机制上界。它在训练中读取真实corruption metadata，因此绝不能包装成
可部署方法。

### 10.2 结果

| Grouping | pooled DSA | operator-grid Avg | last-5 Avg | last-5 Worst |
|---|---:|---:|---:|---:|
| Oracle family | 0.017232 | 21.9333 | 21.0960 | 17.8400 |
| Oracle operator | 0.015783 | 21.5200 | 20.0273 | 15.8827 |
| Random operator | 0.067895 | 18.7817 | 18.3933 | 14.7867 |

```text
DSA(Random)-DSA(Oracle operator) = 0.052112
CI95 [0.051231, 0.052983]

DSA(Oracle family)-DSA(Oracle operator) = 0.001449
CI95 [0.001086, 0.001790]
```

正式结论：

- 真实环境对应关系明显优于随机分组，说明不是任意重加权都能降低DSA；
- operator相对family只额外降低`0.001449` DSA，远低于冻结`0.02`门槛；
- 粗family已捕获几乎全部可用粒度收益；operator细化不是当前瓶颈；
- 禁止继续训练层次化/operator PEW来“救”方法。

```text
verdict: NO_GO_OPERATOR_GRANULARITY_GAP
hierarchical_pew_training_authorized: false
```

历史固定场景上下文：baseline DSA `0.119644`，learned coarse PEW+BER `0.041252`，Oracle
family `0.017232`。这提示learned PEW与Oracle仍有差距，但不是本次预注册的因果对比，也不授权
复活已冻结的PEW纠错路线。

---

## 11. 统一证据表

| 论文问题 | 实验/理论对象 | 结果 | 当前结论 |
|---|---|---|---|
| strong CLE是否产生定向shortcut | gamma0 vs gamma0.9 paired DSA | HFL CLE effect 0.119889 | GO |
| DSA是否沿真实binding而非随机方向 | shuffled-binding null | 0.119644 > null p95 0.029559 | GO |
| shortcut主要来自哪里 | HFL-vs-Local factorial | Local/HFL描述性share 90.05% | local-first |
| DSA是否有可解释零点 | exchangeable projection | -2.50e-20 | PASS |
| DSA是否对应shortcut混合强度 | probability mixture | 严格单调，误差2.78e-17 | PASS |
| JSD能否替代DSA | JSD=0反例 | JSD 0而DSA 0.119644 | 不能 |
| paired DSA是否消除operator-invariant位移 | 受控概率注入 | 不变误差8.33e-17 | PASS |
| DSA能否恢复已知binding方向强度 | 11点受控注入 | 最大误差6.66e-16 | PASS |
| 推断单位是否明确 | source-level界与bootstrap | n=1000半径0.085894 | PASS |
| BER是否对应明确有效分布 | 生产公式/CPU审计 | 最大误差7.32e-16 | PASS |
| BER是否压缩类别—伪环境依赖 | strict-fit TV | 0.487505→0.199744，-59.03% | PASS |
| 压缩是否传递到真实family | oracle reporting-only TV | 0.634914→0.433513，4/4下降 | PASS |
| PEW+BER能否抑制AsymHFL中的CLE | matched Stage-2 | DSA降低65.52% | mitigation GO |
| 结论是否依赖唯一binding map | cross-binding-map map1 | DSA降低56.46%，4/4客户端同向 | replication GO |
| PEW+BER能否跨第二底座抑制CLE | native-CE FedDF | DSA降低78.82% | cross-base mitigation GO |
| 效用是否跨客户端/底座一致 | Worst/Avg/逐客户端 | mixed | 未建立 |
| 环境对应是否重要 | Oracle operator vs random | DSA差0.052112 | GO |
| 是否需要operator级PEW | Oracle family vs operator | 仅差0.001449 | NO-GO |

---

## 12. 建议的论文核心贡献

可考虑写成四项贡献：

1. **场景贡献**：提出受控CLE-HFL评估问题，在模型异构、非IID客户端中构造client-specific
   class-corruption directional bindings。
2. **诊断贡献**：提出paired counterfactual operator grid与DSA，直接识别预测概率是否沿训练
   binding方向移动；给出paired cancellation识别定理、零基准、混合线性、binding specificity、
   source-level推断和JSD不充分性反例。
3. **机制贡献**：通过matched HFL-vs-Local factorial发现shortcut主要local-first，通信只在
   pooled平均上增加较小附加效应。
4. **干预与边界贡献**：给出taxonomy-assisted PEW+BER本地缓解，在AsymHFL和FedDF-fidelity
   两个底座上分别降低DSA 65.52%和78.82%，并在新binding map上再次降低56.46%；
   Oracle/random消融证明真实环境对应重要、粗family足够；有效分布定理与CPU审计进一步说明
   BER如何压缩CLE的统计来源，同时揭示PEW误差上界可能平凡、shortcut抑制与任务效用可能解耦。

不建议把贡献写成“提出一个全新的PEW网络”或“提出普适无损HFL插件”。更稳妥的总定位是：

\[
\boxed{
\text{CLE-HFL受控问题}
+\text{paired DSA诊断}
+\text{local-first机制归因}
+\text{跨两种通信底座的shortcut缓解与效用边界}
}
\]

---

## 13. 当前证据允许和禁止的摘要表述

### 13.1 允许

- fixed strong CLE-v2场景中，类别—operator绑定导致显著directional shortcut；
- shortcut主要在本地训练阶段形成；
- PEW+BER在不读取private真实corruption标签的情况下显著降低DSA；
- DSA抑制在AsymHFL和native-CE FedDF-fidelity中均复现；
- pooled FedDF operator-grid accuracy近似保持（`-0.04pp`），但last-5 Avg轻微下降；
- 真实环境对应优于随机分组，family细化到operator收益很小；
- BER在固定strict-fit分布上使伪环境TV依赖下降59.03%、真实family TV下降31.72%；
- taxonomy-assisted方法有效，但效用收益依赖底座/客户端。

### 13.2 禁止

- “PEW+BER在所有指标、所有架构上都提升”；
- “PEW+BER已被证明为通用无损插件”；
- “PEW是taxonomy-free环境发现”；
- “Oracle分组是可部署训练方法”；
- “FedDF实验是官方完整FedDF recipe复现”；
- “90.05%的每个样本shortcut都由本地因果产生”；
- “12轮等价于达到40轮最终性能”；
- “source-bootstrap覆盖训练seed或cross-scenario不确定性”；
- “BER已经无条件保证真实环境与类别独立”或“BER理论保证DSA必为零”；
- 用smoke/benchmark准确率作为论文证据；
- 混用早期CLE-v1与当前CLE-v2数字而不标注协议差异。

---

## 14. 当前不足与投稿前待审计项

核心链完成不等于论文实验已经完整。网页端GPT应重点审查：

1. 纯PEW+BER正式证据目前主要是固定CLE-v2 scenario、training seed 0；历史三seed正结果包含
   CDep，不能冒充纯PEW+BER-only多seed。
2. 新class-operator binding map的map1 Formal已经完成：Base DSA `0.113761`，PEW+BER DSA
   `0.049531`，下降`0.064230`（56.46%），CI95 `[0.063105,0.065328]`，4/4客户端同向，
   I0/L0/C1/C2全部通过。可宣称跨binding-map复现，但不能扩写成跨partition、训练seed、
   corruption库、数据集或真实场景泛化；map2尚未授权。
3. 架构与客户端数据划分绑定，不能把client2现象单独归因于ShuffleNet容量。
4. FedDF两臂低于预注册20%学习下限；可作为支持性跨底座机制证据，但主表定位需谨慎。
5. 应审计主表是否还缺标准ERM/Local/RAHFL/FedDF等必要对照；不要为了“工作量”添加不能回答
   审稿问题的实验。
6. 合成CLE的现实性需要在limitations中承认，并讨论现实传感器、地域、设备或采集管线形成
   class-corruption correlation的可能对应。
7. novelty仍需要最新文献检索，尤其是federated spurious correlation、group reweighting、
   pseudo-group discovery、corruption robustness与model-heterogeneous FL。

下一步应先把cross-map1 Formal加入主表、机制图和初稿，再审计投稿前剩余证据缺口。不得把追加
实验用于事后翻转已经冻结的NO-GO门槛；map2不是自动续跑项。

---

## 15. 建议论文结构

### 1. Introduction

- corruption在CLE-HFL中从nuisance变成class shortcut；
- 普通accuracy/JSD不能识别binding-specific harm；
- 概述DSA、local-first结论、PEW+BER及边界；
- 列出四项贡献。

### 2. Related Work

- heterogeneous federated learning；
- corruption robustness与AugMix/JSD；
- spurious correlation、group robustness和pseudo-group discovery；
- paired counterfactual diagnostics；
- 明确本文不是首次提出group reweighting。

### 3. Problem Setup: CLE-HFL

- 客户端、异构模型、数据划分；
- class-corruption binding与gamma；
- fit/audit/test/paired-grid角色隔离；
- threat model和不可见信息。

### 4. Paired Diagnostic and Mechanism Analysis

- paired operator counterfactual；
- DSA定义；
- 识别定理与五个可检验性质；
- shuffled-binding；
- HFL-vs-Local factorial与local-first。

### 5. Taxonomy-Assisted Local Mitigation

- PEW公共训练；
- BER数学目标；
- BER有效分布等价、log-odds压缩、理想独立性与PEW误差条件边界；
- 与AugMix/JSD/DCL、GroupDRO的区别；
- 可部署信息边界。

### 6. Experiments

- CLE-v2协议和指标；
- strong CLE机制表；
- AsymHFL PEW+BER主结果；
- FedDF跨底座结果；
- Oracle family/operator/random粒度消融；
- 诚实报告所有冻结gate和负结果。

### 7. Discussion and Limitations

- shortcut suppression与in-distribution utility的权衡；
- taxonomy依赖；
- 固定scenario/seed限制；
- 架构与客户端数据混杂；
- synthetic CLE到现实场景的外推边界。

### 8. Conclusion

- 总结CLE-HFL诊断、local-first机制和可行动缓解；
- 不声称普适无损或taxonomy-free。

---

## 16. 建议的主表与图

以下主表和机制图已经生成，可直接作为初稿素材：

```text
deliverables/cle_hfl_paper_core_artifacts_20260912/PAPER_MAIN_TABLE_ZH.md
deliverables/cle_hfl_paper_core_artifacts_20260912/PAPER_MAIN_TABLE.csv
deliverables/cle_hfl_paper_core_artifacts_20260912/PAPER_MECHANISM_EVIDENCE_CHAIN.png
deliverables/cle_hfl_paper_core_artifacts_20260912/PAPER_MECHANISM_EVIDENCE_CHAIN.pdf
deliverables/cle_hfl_paper_core_artifacts_20260912/PAPER_MECHANISM_EVIDENCE_CHAIN.svg
```

### 表1：CLE机制与local-first归因

使用第4节四臂表及三个estimand，包含DSA、operator-grid accuracy、CI和shuffled-null。

### 表2：PEW+BER跨底座结果

| Base | DSA Base | DSA +PEW/BER | Relative reduction | Utility delta | Gate interpretation |
|---|---:|---:|---:|---|---|
| AsymHFL | 0.119644 | 0.041252 | 65.52% | Avg +1.7323; Worst -0.3760 | mitigation GO; uniform utility未建立 |
| AsymHFL / new binding map | 0.113761 | 0.049531 | 56.46% | Avg +0.7040; Worst +1.0760 | cross-map I0/L0/C1/C2全部PASS |
| FedDF-fidelity | 0.136989 | 0.029012 | 78.82% | grid -0.04; last-5 Avg -0.6660 | mitigation GO; overall frozen NO-GO |

### 表3：环境粒度消融

使用Oracle family/operator/random结果，突出`Random-Operator=0.052112`和
`Family-Operator=0.001449`。

### 图1：场景与完整证据链

```text
client-specific class↔operator binding
  -> paired operator interventions
  -> DSA directional mass shift
  -> HFL/Local attribution
  -> PEW pseudo-environment
  -> BER class-conditional environment balancing
  -> original-map / new-map / cross-base Formal validation
```

正式图将这条链拆为“问题—诊断—归因—干预—受控验证”五段，并把原map、新map1和FedDF的DSA
结果放在同一证据带中；底部边界框明确taxonomy-assisted、效用非一致及外推范围。

### 图2：DSA可解释性

- gamma0与gamma0.9；
- true-binding与shuffled-binding；
- HFL与Local；
- 可以用四组柱状图或paired probability-flow示意。

### 图3：跨底座shortcut—utility二维图

横轴为DSA reduction，纵轴为utility delta。AsymHFL与FedDF均位于高DSA reduction区域，但纵轴
方向不同，用于直观表达“机制迁移、效用不一致”。

---

## 17. 已冻结路线与研究纪律

不得通过简单调权重、改门槛、补seed或换名字直接复活：

```text
SDMN
CDR-SNR
CRSF / K1-C-Minimal
P2/P3/P4 targeting
CVRS
WEC-BER
multi-label PEW + Soft-BER
hierarchical/operator PEW（Oracle粒度门已失败）
以及项目AGENTS.md列出的其他冻结负结果
```

特别是CVRS已经显示：generic public-response proxy下降不保证真实DSA下降。任何未来方法必须说明
什么可观测信息能够识别真正有害的CLE routing，不能再把相关proxy当作真实harm替代物。

---

## 18. 请网页端 GPT 完成的任务

请把自己当作严格的CCF-B类会议审稿人、联邦学习研究者和论文合作者，基于本文件完成以下任务：

1. 判断上述四项贡献是否足以形成一篇CCF-B会议论文，并指出最可能的三条拒稿理由。
2. 在不夸大PEW结构创新、不隐瞒Formal NO-GO的前提下，给出最强且诚实的论文定位。
3. 审查当前证据矩阵，区分投稿前“必补实验、最好补实验、无需补实验”；尤其判断是否必须补
   新CLE mapping、纯PEW+BER多seed或更标准的基线。
4. 设计一套不把FedDF写成失败、也不把它写成无损成功的结果叙事。
5. 检查DSA识别定理、五个性质、BER有效分布/失衡压缩定理、PEW误差条件边界和JSD反例是否
   还存在逻辑缺口；不得把当前平凡PEW误差界写成强保证。
6. 进行最新相关工作检索，核查CLE-HFL、federated spurious correlation、group reweighting、
   pseudo-group discovery与paired counterfactual diagnostics的创新性边界。
7. 给出论文标题、摘要、Introduction、Method、Experiments、Limitations的详细提纲。
8. 在完成上述审查后，生成一份中文论文初稿；所有无法由本文件支持的句子标注`[待证据]`，
   所有需引用的事实标注`[待引用]`，不得自行编造实验数字或SOTA结论。

建议网页端GPT首先回答“还缺哪些必做实验”，确认后再写全文，以免初稿建立在过度主张上。

---

## 19. 仓库内证据来源

```text
deliverables/cle_v2_mechanism_stage1_20260910/RESULT_SUMMARY_ZH.md
docs/experiments/current/CLE_DSA_THEORY_CACHE_VALIDATION_ZH.md
docs/research/status/CLE_DSA_IDENTIFICATION_THEORY_2026_09_11_ZH.md
docs/research/status/CLE_BER_MECHANISM_THEORY_2026_09_11_ZH.md
docs/experiments/current/CLE_V2_CROSS_SCENARIO_BINDING_MAP_ZH.md
docs/experiments/current/CLE_V2_PEW_BER_STAGE2_OPENI_ZH.md
deliverables/cle_v2_oracle_granularity_formal_20260911/RESULT_SUMMARY_ZH.md
deliverables/cle_v2_feddf_plugin_formal_20260911/RESULT_SUMMARY_ZH.md
docs/research/status/CLE_HFL_PAPER_CLOSURE_2026_09_11_ZH.md
```

原始输出位于`outputs/`，解析报告位于`deliverables/`。Smoke与benchmark只验证执行或成本，不是
科学证据。本文所有正式数字均应回到以上正式报告及其冻结协议核验。
