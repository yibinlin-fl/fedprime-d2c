# CLE-HFL 完整论文证据与网页端 GPT 初稿交接文档

创建：2026-09-11；最新更新：2026-10-31
用途：交给网页端进行当前V0.11结构、论证与语言审稿；本轮先审查，不直接重写全文
当前阶段：DSA/BER理论与proxy不可识别闭环完成；机制协议三seed结果完成；
主评价协议seed-0剂量响应与AsymHFL方法对照Formal完成，local-first三训练种子完成，Axis II完成5/8行

---

## 0. 给网页端 GPT 的重要说明

### 0.0 2026-10-31本轮材料与任务（最高优先级）

当前完整正文是`cle_hfl_latex_v0_11_20261028/source/`，不是一个月前的V0.6。
V0.11的PDF/Overleaf包在本审稿包中按原字节复制；本轮没有修改论文源码或科学结果。
V0.6是网页端六Skill历史润色版；V0.8/0.9已回填后续正式结果；V0.10统一协议命名；
V0.11接入五图及调整相关图注、位置。版本号递增不等于重新执行了那六个Skill。

2026-10-30又完成五张独立修订候选：图1/2为TikZ v7，图3为local-first v2，
图4为objective-tradeoff v3，图5为dose-response v4。它们尚未替换V0.11中的旧版插图。
审稿包`figure_candidates/`保存新图PNG，仅用于视觉比较；不可声称PDF已接入这些候选。
三张数据图冻结数值和坐标范围不变；图2明确BER是损失聚合/优化，不是像素预处理。

本轮先按`WEB_REVIEW_PROMPTS_ZH.md`第一阶段审查，并按`CURRENT_EVIDENCE_AND_REVIEW_LOCKS_ZH.md`
保护事实。输出须逐项引用原句、文件/小节、问题理由和最小修改建议。不得输出AI生成概率，
不得仅凭某个词、句式工整或语法流畅就判为“AI写作”。可以判定没有问题，不强行找茬。
原稿若存在事实措辞歧义，应单独列科学修正建议，不为“数字锁定”而保留已识别的错误语义。

已完成：机制Axis I四目标40轮三training-seed；local-first三training-seed；map1/map2；
FedDF-fidelity成对验证；Oracle粒度消融；主评价seed0剂量响应和AsymHFL成对方法结果。
尚未完成：第二private dataset benchmark/Formal；bounded taxonomy stress；主评价Axis II
FedDF/AugHFL/RAHFL三行Formal。不得重新把已完成的种子实验列为待补。
引用审计继承自2026-10-14；本次只同步材料，不冒称重新核查了全部外部文献。

### 0.1 科学状态摘要（继承2026-10-27，版本入口已更新）

本文件是当前交给网页端GPT的**完整事实交接**。若本文件中较早日期段落与本节冲突，以本节、
第14--16节、第20节和第22--24节的已完成结果为准；本轮审稿范围以0.0节为准。`20260925`目录下的短交接只是历史增量
说明，不能脱离本文件单独用于重写论文。

论文当前采用“双协议、不同职责、禁止混表”的实验结构。论文正文冻结以下中性命名；历史文件名、
脚本名和结果目录中的`strong`/`moderate`只作为工程标识保留，不应在新正文中反复出现：

```text
mechanism protocol:
  CIFAR-10, held-out map2, alpha=0.5, rho=1.0, gamma=0.9
  -> existence / paired DSA / local-first / Axis I objective attribution

main evaluation protocol:
  CIFAR-10, held-out map2, alpha=1.0, rho=0.5, gamma={0,0.5,0.9}
  -> dose response / clean-corrupted utility / Axis II HFL context
```

机制协议用于最大化shortcut可识别性；主评价协议用于在固定clean--corrupted mixture下同时报告
DSA与任务效用。主评价协议不是第二private dataset，也不是对机制协议的替换。机制协议的主要
科学终点是binding-directed DSA；其准确率必须诚实报告，但不能冒充普通CIFAR-10性能或HFL领域
排行榜。两套协议的`alpha/rho/gamma`差异只在实验设置中集中说明一次，后文统一简称
`mechanism protocol`与`main evaluation protocol`。旧高暴露Local/FedMD结果仅保留为内部工程
记录，不进入正文或附录科学表。

当前论文版本关系：

```text
V0.6 = 网页端多Skill润色后的不可变polished snapshot
V0.7 = 已保留的上一版内部事实与实验骨架
V0.8 = 已回填主评价协议Formal的历史内部写作快照
V0.9 = 从V0.8派生、回填local-first三训练种子结果的历史快照
V0.10 = 从V0.9派生、仅收敛协议命名与叙事的历史快照
V0.11 = 从V0.10派生、接入五图及相关图注/布局的当前稿；后续五图修订候选尚未接入
```

当前V0.11源码：

```text
deliverables/cle_hfl_latex_v0_11_20261028/source/main.tex
```

本轮新Formal的唯一汇总依据：

```text
deliverables/cle_v2_moderate_axis2_dose_formal_seed0_20261025/RESULT_SUMMARY_ZH.md
deliverables/cle_v2_moderate_axis2_dose_formal_seed0_20261025/PAPER_TABLE.csv
deliverables/cle_v2_moderate_axis2_dose_formal_seed0_20261025/ARTIFACT_MANIFEST.csv
deliverables/cle_v2_local_first_multiseed_20261026/RESULT_SUMMARY_ZH.md
deliverables/cle_v2_local_first_multiseed_20261026/PAPER_TABLE.csv
deliverables/cle_v2_local_first_multiseed_20261026/ARTIFACT_MANIFEST.csv
```

四个任务实测训练加分析合计`51.8363` V100小时；按6积分/小时约`311.02`积分。该成本数字只用于
预算与复现说明，不是科学结果。

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
固定其余因素、只更换binding map的map1/map2复现与三training-seed验证
  ↓
主评价协议gamma=0/0.5/0.9剂量响应成立；AsymHFL下PEW+BER将DSA降低51.21%
  ↓
local-first三training seeds完成：DSA机制门槛3/3复现；辅助学习下限只保留在内部审计
  ↓
Axis II已有5/8行；待补第二private dataset与taxonomy stress
```

必须保留以下区分：

```text
shortcut mitigation efficacy:      GO
cross-base DSA suppression:         GO on fixed CLE-v2 seed0 scenario
architecture-uniform utility:       NOT ESTABLISHED
universal lossless plugin claim:    NOT ESTABLISHED / frozen overall NO-GO
taxonomy-free claim:                FALSE
cross-binding-map replication:      ESTABLISHED on controlled CIFAR-10
main-evaluation dose response:       ESTABLISHED at training seed 0
main-evaluation AsymHFL suppression: ESTABLISHED at training seed 0
local-first training-seed direction: ESTABLISHED on seeds 0/1/2 by the DSA mechanism gates
cross-private-dataset generalization: NOT ESTABLISHED
```

不要把 Formal 的联合 `NO-GO` 误写成“PEW+BER 没有效果”；也不要因为 DSA 很强就把它改写成
“通用无损插件 GO”。联合 gate 和主科学终点必须同时、透明地报告。

### 0.1 2026-10-26冻结的剩余实验清单

网页端后续不得把所有历史TODO重新包装成投稿必做项。当前优先级只有三层：

```text
投稿优先：
  P1 CIFAR-100第二private task
     当前只完成输入/入口/审计；benchmark和Formal未跑
     必须先seed-0 benchmark，再决定是否授权Formal
  P2 bounded taxonomy stress
     默认先做复用checkpoint的evaluation-only coverage stress

条件性：
  主评价协议Axis II若保留完整八行表，需补FedDF-fidelity/AugHFL/RAHFL三行Formal
  优先级为RAHFL > AugHFL > FedDF
  若不补，最终论文删除pending行，不声称完整八方法排名

可选且不阻塞成稿：
  FedMD四目标复现、JTT Formal、EIIL联邦适配、额外partition/map/seed
```

明确不再要求：local-first更多seed或40轮扩展、operator-level PEW、新BER模块、昂贵strong
full-budget Axis I复验。当前写作工作转入科研配图；不得以“实验仍不完整”为由擅自扩方法。

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

### 2.2 CLE-HFL v2：机制协议与主评价协议

已完成的正式机制与Axis I实验使用机制协议：

```text
scenario: cle_hfl_v2_paired_factorial_seed0_split0_v1
private task: CIFAR-10
clients: 4
private samples available: 40,000
Dirichlet alpha: 0.5
strong-binding condition: gamma=0.9
control: gamma=0
private corruption rate rho: 1.0
paired evaluation: 1,000 source × 15 operator × 4 client
evaluation severity: 3
```

CLE-v2 将绑定细化为 client-specific、class-specific concrete operator map，并区分seen/unseen
operator，支持operator-cell、WCCA、CFG、LOO及operator-level DSA。40,000是可用私有数据规模，
不代表每个模型在有限训练轮次中完整遍历40,000个不同样本。

主评价协议专门用于剂量响应、任务效用与HFL领域比较：

```text
private task: CIFAR-10
held-out binding map: map2
Dirichlet alpha: 1.0
private corruption rate rho: 0.5
gamma: 0.0 / 0.5 / 0.9
partition seed: 0
binding-map seed: 2
```

三个gamma条件共享source、label、clean/corrupted mask、severity、fit/audit split、初始化和paired
operator grid。`alpha`描述客户端标签支持，`rho`描述损坏暴露比例，`gamma`描述被损坏样本中
class--operator绑定浓度。由于两套协议同时改变alpha和rho，协议之间不是单因素因果消融；真正的
gamma剂量响应只在主评价协议内部做配对比较。主评价协议仍然是CIFAR-10受控合成场景，不能代替
第二private dataset。

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

固定CLE-v2场景、training seeds 0/1/2、12 rounds、每客户端每轮最多16个local batches：

```text
h0_b: HFL,   gamma=0.0
h9_b: HFL,   gamma=0.9
l0_b: Local, gamma=0.0
l9_b: Local, gamma=0.9
```

四臂均使用同一 AugMix/JSD/DCL baseline；本阶段不加载或训练PEW，不启用BER/CDep。

### 4.2 Seed 0正式结果与三种子扩展

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

Seed 1/2随后在完全相同的四臂协议下完成。三个seed的`Delta_HFL`分别为
`0.119889/0.123717/0.124638`，均通过directional shortcut与binding-specificity门槛。
Seed 0/1通过完整预注册门槛；Seed 2的strong-CLE HFL grid accuracy为`19.0683%`，低于20%学习
下限，因此总verdict为`INCONCLUSIVE_UNDERTRAINED`，但其三个机制门槛均通过。source-bootstrap
仍只覆盖固定checkpoint下的source不确定性；三seed离散度才反映本次训练随机性。

---

## 5. HFL-vs-Local：shortcut 主要是 local-first

对每个训练种子，将两种通信条件下的CLE effect分别写为：

\[
\Delta_{HFL}=DSA(h9_b)-DSA(h0_b),
\]

\[
\Delta_{Local}=DSA(l9_b)-DSA(l0_b).
\]

pooled通信附加量为：

\[
\Delta_{comm}=\Delta_{HFL}-\Delta_{Local}.
\]

三seed的`Delta_Local`为`0.107960/0.127287/0.131210`；`Delta_comm`为
`0.011929/-0.003570/-0.006573`。均值加样本标准差为：

```text
Delta_HFL   = 0.122748 +/- 0.002518
Delta_Local = 0.122152 +/- 0.012446
Delta_comm  = 0.000596 +/- 0.009929
```

这支持：HFL中观察到的大部分CLE效应，在没有通信的本地训练中已经形成；通信差值均值很小且
跨seed符号不稳定。因此最合理的干预位置是客户端本地目标，而不是首先重新设计通信。

必须避免两种误读：

1. 每seed的Local/HFL比值不是对每个样本的精确因果分解；Seed 1/2超过100%只表示该seed的
   Local contrast略大于HFL contrast；
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

## 10.1 Held-out Map2 40轮四臂Formal：机制对照闭环

五臂12轮screen完成选择后，map2保持未见，并在结果前冻结40轮四臂协议：ERM、CVaR-DRO、
PEW+GroupDRO、PEW+BER。四臂共享partition、training/evaluation seed、initial states、private
batch轨迹和strict AsymHFL-val通信；single-view训练，AugMix/JSD/DCL/CDep全部禁用。

| Arm | DSA↓ | Operator-grid Acc↑ | Last-10 Avg↑ | Last-10 Worst↑ | WCCA↑ | CFG↓ |
|---|---:|---:|---:|---:|---:|---:|
| ERM | 0.260308 | 20.3067 | 19.8578 | 16.3007 | 0.000 | 32.8325 |
| CVaR-DRO | 0.088982 | 20.4567 | 18.6700 | 14.0380 | 0.125 | 27.7775 |
| PEW+GroupDRO | 0.214575 | 20.7883 | 20.9562 | 17.8140 | 0.025 | 31.5275 |
| **PEW+BER** | **0.077741** | **23.7433** | **24.4518** | **20.4727** | **2.650** | **22.8025** |

5000次source-paired bootstrap：

```text
ERM-BER DSA:       0.182567; CI95 [0.180596,0.184407]
GroupDRO-BER DSA:  0.136834; CI95 [0.134952,0.138706]
BER-CVaR DSA:     -0.011240; CI95 [-0.012355,-0.010092]
```

全部冻结门槛I0/S0/P1--P5通过，verdict=`GO_FOUR_ARM_HELDOUT_MAP2`。这使方法归因明显增强：
共享相同PEW分组的GroupDRO仍有高DSA，因而收益不能仅归于PEW side information；BER针对
类别内部环境支持失衡的目标与CLE结构更匹配。在本次held-out pooled Formal中BER也取得低于
CVaR的DSA并明显更高的任务效用。

逐客户端上，BER相对ERM和GroupDRO均4/4降低DSA；相对CVaR只在c2/c3更低，c0/c1仍由CVaR
更低。BER的operator-grid accuracy在4/4客户端均高于其余三臂。因此允许写pooled胜出，禁止写
所有架构无条件支配。screen中CVaR曾比BER低`0.002909`，而map2 Formal中BER低`0.011240`；
总体应写“BER的DSA抑制与CVaR具有竞争力，并在held-out Formal中胜出，同时持续取得更好效用”。

---

## 10.2 Proxy non-identifiability：补齐BER到真实CLE行为的理论桥

BER直接控制的是伪环境`Z=E_hat`下的类别内支持结构，而论文目标是潜在真实corruption环境`E`
及最终模型行为。仅从`P(Y,Z)`不能无条件推出`P(Y,E)`：对相同的可观测`P(Y,Z)`，可以构造
两个潜在世界，一个令`E`与`Y`独立，另一个令`E=g(Y)`。后者的真实依赖TV为：

\[
\mathcal D(Y,E)=1-\sum_eP(E=e)^2,
\]

而二者所有proxy-only观测完全相同。因此不存在只依赖`P(Y,Z)`的无条件非平凡真实环境证书。
现有条件桥仍然成立：

\[
\mathcal D_Q(Y,E)
\le \mathcal D_Q(Y,Z)+2P_Q(Z\ne E),
\]

但当前BER有效分布PEW误差使其截断为平凡的1.0；WEC-BER又已正式否定简单稳定误差通道假设。
这不是需要隐藏的失败，而是说明必须保留target-aligned行为评价。

冻结缓存与Formal数值验证全部通过：

```text
same observed P(Y,Z):                  true
latent world A/B true TV:              0.000000 / 0.800000
identical-view JSD / cached DSA:        0 / 0.119644
CVRS-vs-JSD proxy / DSA change:        -0.088854 / +0.013000
N0/N1/N2/N3/N4:                        all PASS
```

由此得到论文协议中的必要性推论：若方法只优化不读取真实binding的proxy目标，则proxy下降不能
证明CLE directional harm下降；提出CLE缓解主张必须额外报告读取冻结evaluation binding、基于
同源跨operator响应的target-aligned estimand。本文采用paired DSA，但不声称DSA是所有shortcut
问题唯一合法指标。

完整理论链为：

```text
BER压缩伪环境支持
-> 有PEW误差条件时可传递到真实环境
-> 无识别条件时proxy-only原则上不可识别
-> paired DSA直接验证真实binding-directed行为
-> 多组Formal检验实际缓解
```

完整证明与图：

```text
docs/research/status/CLE_PROXY_NONIDENTIFIABILITY_THEORY_2026_09_13_ZH.md
deliverables/cle_proxy_nonidentifiability_20260913/RESULT_SUMMARY_ZH.md
deliverables/cle_proxy_nonidentifiability_20260913/PROXY_NONIDENTIFIABILITY_THEORY.png
```

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
| 通用困难样本方法能否解释PEW+BER收益 | 12轮五臂selection screen | CVaR DSA最低；BER效用与综合折中更强 | 仅筛选，不是Formal |
| BER是否只是标准GroupDRO | map2相同PEW分组Formal | BER DSA低0.136834、grid高2.9550pp | 机制对照GO |
| 四臂结论能否在筛选未见map复现 | held-out map2，40轮 | I0/S0/P1--P5全部通过 | GO |
| BER能否对抗taxonomy-free CVaR | map2 40轮Formal | pooled DSA低0.011240、grid高3.2867pp | 本协议GO；非所有客户端DSA支配 |
| proxy下降能否证明真实CLE下降 | 不可识别定理、Stage-1缓存、CVRS Formal | 同观测TV可为0/0.8；CVRS proxy降而DSA升 | 不能；DSA是本文必要评价终点 |
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
   source-level推断和JSD不充分性反例；进一步证明proxy-only不可识别性，由此说明target-aligned
   DSA在本文证据协议中为何是必要而非可选。
3. **机制贡献**：通过matched HFL-vs-Local factorial发现shortcut主要local-first，通信只在
   pooled平均上增加较小附加效应。
4. **干预与边界贡献**：给出由CLE支持结构导出的taxonomy-assisted PEW+BER本地缓解，在AsymHFL和FedDF-fidelity
   两个底座上分别降低DSA 65.52%和78.82%，并在新binding map上再次降低56.46%；
   Oracle/random消融证明真实环境对应重要、粗family足够；有效分布定理与CPU审计进一步说明
   BER如何压缩CLE的统计来源；held-out map2 40轮Formal进一步证明BER不仅优于ERM，而且优于
   相同PEW分组的GroupDRO，并在pooled DSA与效用上同时优于CVaR。仍保留PEW误差上界可能平凡、
   shortcut抑制与任务效用可能解耦的边界。

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
- held-out map2 40轮Formal中，BER pooled DSA低于CVaR `0.011240`，且operator-grid accuracy
  高`3.2867pp`；相同PEW信息下BER DSA低于GroupDRO `0.136834`。

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

1. held-out map2的ERM/CVaR-DRO/PEW+GroupDRO/PEW+BER已经完成training seeds 0/1/2并通过
   冻结稳定性门槛；但它仍是固定CIFAR-10、partition seed0和strong协议，不能冒充dataset或
   partition层面的稳定性。
2. 新class-operator binding map的map1 Formal已经完成：Base DSA `0.113761`，PEW+BER DSA
   `0.049531`，下降`0.064230`（56.46%），CI95 `[0.063105,0.065328]`，4/4客户端同向，
   I0/L0/C1/C2全部通过。可宣称跨binding-map复现，但不能扩写成跨partition、训练seed、
   corruption库、数据集或真实场景泛化。用于四臂最终比较的map2在五臂筛选时保持未见；其40轮
   Formal已完成并通过门槛，随后又完成training seeds 1/2；仍固定partition与private dataset。
3. 架构与客户端数据划分绑定，不能把client2现象单独归因于ShuffleNet容量。
4. FedDF两臂低于预注册20%学习下限；可作为支持性跨底座机制证据，但主表定位需谨慎。
5. Axis II已冻结为主评价协议八行：Local、FedMD、FedProto、FedDF、AugHFL、RAHFL、
   AsymHFL--ERM、AsymHFL+PEW+BER。旧机制协议Local/FedMD不进入论文。FedTGP、KT-pFL、FCCL、
   RHFL是已实现但非必跑扩展，禁止看结果后选择性加入。当前Local/FedMD/FedProto及两条
   AsymHFL行已完成，FedDF/AugHFL/RAHFL三行仍待补；未完成八行前不得写完整领域排名。
6. 合成CLE的现实性需要在limitations中承认，并讨论现实传感器、地域、设备或采集管线形成
   class-corruption correlation的可能对应。
7. novelty仍需要最新文献检索，尤其是federated spurious correlation、group reweighting、
   pseudo-group discovery、corruption robustness与model-heterogeneous FL。

主评价协议ERM的gamma0/0.5/0.9及gamma0.9 PEW+BER已完成，不再重复seed 0。local-first训练
种子1/2也已完成；当前优先级转为真正第二private dataset和bounded taxonomy stress；FedDF/RAHFL是有
余力再补的领域锚点。不得根据新结果调参，也不得用追加实验事后翻转冻结的旧NO-GO门槛。
短预算、smoke与benchmark均不是论文证据。

---

## 15. 建议论文结构

### 15.0 总叙事与暂定标题

论文不能以“提出一个PEW+BER插件”为唯一叙事。建议总链条固定为：

```text
定义CLE-HFL
-> paired反事实识别
-> DSA量化与不可识别性边界
-> HFL-vs-Local local-first归因
-> PEW+BER本地干预
-> 机制协议跨binding map / 训练随机性 / 通信底座验证
-> 主评价协议剂量响应与HFL机制家族定位
-> taxonomy-assisted与外部有效性边界
```

暂定英文标题：

```text
When Corruption Becomes a Label: Diagnosing and Mitigating
Class-Corruption Shortcuts in Model-Heterogeneous Federated Learning
```

摘要按五句话组织：HFL背景；CLE问题；现有accuracy/JSD/proxy诊断缺口；paired DSA、local-first
和PEW+BER；受控验证结果与taxonomy-assisted边界。

### 1. Introduction

按五段写作：

1. 模型异构FL通过公共数据、logit或蒸馏协作，但现有重点主要是精度、通信和普通鲁棒性。
2. 用“cat+blur / dog+noise”的客户端定向绑定例子说明corruption从nuisance变成class code；
   CLE-HFL关注的是沿binding方向的概率迁移，不只是损坏后的accuracy下降。
3. 说明普通accuracy、同一样本增强一致性的JSD以及generic public proxy为何不能识别
   binding-specific harm，引出paired operator counterfactual与DSA。
4. 概述strong CLE证据、local-first归因、PEW+BER本地干预和跨map/底座验证。
5. 冻结四项贡献：具体CLE-HFL问题；paired DSA及识别/不可识别理论；local-first机制结论；
   由支持结构理论导出的taxonomy-assisted PEW+BER及受控验证。不得声称首次研究联邦虚假相关。

### 2. Related Work

分为四组，并让每组最后一句明确本文缺口：

1. `Model-Heterogeneous Federated Learning`：FedMD/FedDF、异构公共知识蒸馏、RAHFL-like
   鲁棒底座；缺少客户端特定class-corruption directional shortcut诊断。
2. `Corruption Robustness and Consistency Learning`：corruption benchmark、AugMix、JSD、
   DCL；强调“同operator附近一致”不推出“跨operator不利用binding”。
3. `Spurious Correlation and Group Robustness`：IRM、GroupDRO、JTT、CVaR-DRO、EIIL和
   pseudo-group discovery；承认伪组与重加权不是首次提出，再区分CLE结构、模型异构和DSA。
4. `Federated Spurious Learning and Counterfactual Diagnosis`：FedCD、个性化联邦不变学习、
   联邦域泛化及其他领域counterfactual shortcut evaluation；安全表述是本文把这些思想特化为
   HFL中的class-corruption定向绑定，并给出paired协议、local-first归因和可行动干预。

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

按研究问题而不是按运行时间组织：

- RQ1：CLE directional shortcut是否存在？
- RQ2：shortcut主要在本地优化还是通信阶段形成？
- RQ3：PEW+BER能否在保留任务效用时降低DSA？
- RQ4：收益是否只是困难样本重加权或标准group robustness？
- RQ5：结论能否跨binding map、training seed和HFL底座复现？
- RQ6：PEW taxonomy粒度、误差和未知/复合corruption带来什么边界？
- RQ7：在主评价协议下，DSA是否呈gamma剂量响应，普通与corruption-robust
  HFL方法处于什么位置？

正文依次放机制协议CLE/local-first表、held-out-map2四臂机制表、主评价协议剂量响应、
主评价协议八行HFL领域表、FedDF支持性跨底座结果和Oracle粒度消融。12轮screen只解释方法选择，
不得作为最终主表证据。机制协议与主评价协议结果禁止混成同一无条件排行榜；所有冻结gate和负结果
必须保持原判定。

### 7. Discussion and Limitations

- shortcut suppression与in-distribution utility的权衡；
- PEW依赖预定义corruption taxonomy，但训练不读取私有真实环境标签；
- 固定scenario/seed限制，以及source-bootstrap不覆盖训练与数据集不确定性；
- 架构与客户端数据混杂；
- synthetic CLE到现实场景的外推边界；
- 未知、复合、连续和自然形成corruption尚未全面验证。任意开放世界corruption可作为未来工作，
  但投稿前最好增加一个有界的unseen/compound stress test，避免外部有效性完全没有实验证据。

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
deliverables/cle_proxy_nonidentifiability_20260913/PROXY_NONIDENTIFIABILITY_THEORY.png
deliverables/cle_proxy_nonidentifiability_20260913/PROXY_NONIDENTIFIABILITY_THEORY.pdf
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

### 表4：通用spurious基线与BER机制对照

12轮screen只作为附录式选择依据。最终表使用held-out map2 40轮Formal，只保留
`ERM/CVaR-DRO/PEW+GroupDRO/PEW+BER`，并同时报告DSA和utility。

screen完整数字仅用于设计/审计最终表：

| Arm | DSA | Grid Acc | Last-5 Avg | Last-5 Worst | WCCA | CFG |
|---|---:|---:|---:|---:|---:|---:|
| ERM | 0.214812 | 18.4183 | 17.8240 | 15.5213 | 0.0000 | 35.5800 |
| JTT | 0.046833 | 15.2617 | 16.0470 | 12.2240 | 0.4000 | 19.3500 |
| CVaR-DRO | **0.034980** | 16.5917 | 16.6053 | 12.9427 | 0.1000 | 25.6350 |
| PEW+GroupDRO | 0.177739 | 19.5450 | 19.2727 | 14.9640 | 0.6000 | 32.1150 |
| PEW+BER | 0.037889 | **21.1267** | **19.3430** | **16.4133** | **0.7000** | **19.5450** |

5000次source-paired bootstrap：`ERM-BER=[0.174556,0.179224]`、
`GroupDRO-BER=[0.137643,0.142017]`、`BER-CVaR=[0.002163,0.003661]`。最后一项为正，故
screen中CVaR的DSA确实略低；但BER相对CVaR的grid/Avg/Worst分别高`4.5350/2.7377/3.4707pp`。
这些区间不覆盖training seed或场景不确定性，不能将screen升级为最终论文证据。

最终Formal表见本文件第10.1节及：

```text
deliverables/cle_v2_spurious_final_map2_20260913/RESULT_SUMMARY_ZH.md
deliverables/cle_hfl_paper_core_artifacts_20260912/PAPER_FOUR_ARM_FINAL_TABLE.csv
```

### 表5：主评价协议CLE剂量响应（seed-0 Formal已完成）

固定map2、partition、mask、severity和初始化，报告：

```text
ERM gamma=0.0
ERM gamma=0.5
ERM gamma=0.9
PEW+BER gamma=0.9
```

正式结果：

| gamma | Objective | DSA | Grid | Clean | Corrupt | Mixed |
|---:|---|---:|---:|---:|---:|---:|
| 0.0 | ERM | -0.000730 | 48.480 | 52.325 | 47.853 | 50.089 |
| 0.5 | ERM | 0.039679 | 48.395 | 54.350 | 47.930 | 51.140 |
| 0.9 | ERM | 0.184514 | 37.193 | 49.700 | 37.008 | 43.354 |
| 0.9 | PEW+BER | 0.090024 | 38.160 | 45.350 | 37.773 | 41.562 |

ERM DSA随gamma严格升高；相邻增量的5,000次source-paired bootstrap CI95均大于零。gamma=0
未超过shuffled null，gamma=.5/.9均显著。gamma=.9下PEW+BER相对AsymHFL--ERM将DSA下降
`0.094490`（51.21%），4/4客户端同向，CI95 `[0.091877,0.097081]`。Grid/Corrupt提升
`0.967/0.765pp`，Clean/Mixed下降`4.350/1.793pp`。这支持显著shortcut抑制与非均匀utility
trade-off，不支持无损或完全消除。

### 表6：主评价协议Axis II八行HFL领域表（5/8行Formal完成）

正文预先冻结八行：

```text
Local/ERM
FedMD
FedProto
FedDF
AugHFL
RAHFL
AsymHFL--ERM
AsymHFL+PEW+BER
```

所有行使用`alpha=1.0/rho=0.5/gamma=0.9`、training seed0、相同初始化、训练预算和paired grid。
该表同时报告DSA与clean/corrupted/overall utility。当前已完成Local/ERM、FedMD、FedProto、
AsymHFL--ERM、AsymHFL+PEW+BER；FedDF、AugHFL、RAHFL仍为pending。旧strong Local/FedMD
完全不进入论文表，当前五行不得包装成八方法完整排名。

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

## 18. 模拟审稿攻击、当前回答与证据缺口

| 可能攻击 | 当前最强且诚实的回答 | 投稿前还需什么 |
|---|---|---|
| PEW只是公共corruption分类器，BER只是重加权，创新不足 | 单个组件并非全新；贡献主体是CLE-HFL问题、paired DSA、local-first归因、proxy不可识别性和结构匹配干预的闭环 | 用相关工作矩阵逐项对比；不要把网络结构创新写成主贡献 |
| 场景是人为设计的，现实中是否存在CLE未知 | 受控binding用于隔离“corruption被当作class code”的因果机制，对应设备、地域、采集管线与类别共现的现实风险 | 最好补第二数据集或更现实CLE；至少增加unseen/compound stress test |
| PEW依赖人工taxonomy，不是开放世界方法 | 正确，本文是taxonomy-assisted而非taxonomy-free；公共程序化标签训练PEW，私有样本只用伪环境，不需要私有真实corruption标签 | 摘要、假设和limitations同时披露；加入CVaR/JTT/EIIL等无taxonomy概念对照讨论 |
| DSA使用真实binding，是否发生测试标签泄露 | binding/operator metadata只在训练封存后构造paired evaluation和报告DSA；fit/audit/test角色隔离，训练与路由不读取最终test标签 | 在算法框图和伪代码中明确seal point，并报告输入审计 |
| DSA是不是为本方法量身定做 | DSA在PEW+BER之前定义；gamma0近零、shuffled-binding null、强度恢复和source级推断均独立验证；CVRS反例说明generic proxy不能替代target harm | 报告accuracy、Worst/WCCA/CFG和DSA，不用DSA单指标挑赢家 |
| JSD已经实现corruption invariance，为什么还需要BER | JSD约束同一operator附近的增强视图；它不平衡数据集级类别×环境支持，也不推出跨operator的binding方向不变 | 保留`JSD=0 but DSA>0`构造与缓存反例 |
| shortcut主要local-first，那为什么还要研究联邦学习 | local-first确定了干预位置；HFL系统仍决定异构客户端、共享知识、公共通信和最终模型行为，通信有较小附加效应但不是主因 | 保留HFL-vs-Local factorial，不宣称通信是主要放大器 |
| PEW误差高，BER理论上界为1，理论没有作用 | 这恰好说明proxy-only传递原则在无附加假设时不可识别；BER理论解释伪环境有效分布上的支持压缩，真实CLE改善必须由paired DSA验证 | 不伪造PEW必准假设；明确“机制解释+target-aligned验证”而非无条件性能定理 |
| 为什么不用GroupDRO、CVaR、JTT或EIIL | 已有matched四臂/五臂比较；held-out map2中BER同时优于ERM、matched PEW+GroupDRO和pooled CVaR，说明不是单纯困难样本或worst-group追逐 | EIIL若未实现只能列相关工作，不能声称实证击败；保留公平协议说明 |
| 结果只有一个数据集、固定partition或少量seed | 当前有跨binding map、map2四臂三training-seed和两种通信底座，但仍是固定partition与单private dataset | 真正第二private dataset与必要的partition边界仍是重要缺口；source-bootstrap不能替代它们 |
| FedDF上效用下降，不能叫通用无损插件 | 同意；FedDF只支持shortcut抑制机制迁移，不支持architecture-uniform utility或无损插件 | 主表显式保留FedDF NO-GO边界；方法定位改为local intervention而非universal no-harm plug-in |
| CVaR在部分客户端DSA更低，BER并非全面最好 | held-out map2 pooled BER DSA更低且4/4客户端grid accuracy最高，但c0/c1 DSA由CVaR更低；结论是整体shortcut-utility优势而非逐客户端支配 | 报告逐客户端结果和Pareto图，禁止只报pooled赢家 |
| source-bootstrap区间很窄，是否夸大稳定性 | 区间只覆盖固定checkpoint下evaluation-source抽样，不覆盖训练随机性、partition、场景与数据集不确定性 | 多训练seed必须单独报告；不能把source CI叫作全流程置信区间 |
| 架构差异是否导致某个小模型受损 | 架构与非IID客户端身份固定绑定，当前不能把client2现象因果归于模型容量 | 删除“小模型受害”因果措辞；若要回答必须做架构×数据交叉实验 |
| 未知、复合corruption一来方法可能失效 | 当前声明范围是closed-set taxonomy-assisted CLE；开放世界泛化不是已证明结论 | 一个有界stress test适合正文/附录；任意开放世界泛化保留为limitations |
| 超参数、cap和alpha是否事后调到最好 | 当前Formal使用冻结配置和预注册门槛，不能因结果继续调参翻案 | 补有限灵敏度或明确选择来源；保持held-out map2未用于screen的事实 |

对AI辅助审稿的额外防线：逐项核对定理前提、表格数字、verdict、数据角色、引用和图中主张，确保
摘要中的每个数字能回溯到正式结果。AISTATS 2027的会议侧AI审查只检查事实正确性，不替代人类
审稿；其公开政策还明确禁止人类reviewer用LLM生成评审。因此不要把“AI审稿”理解成可以只为机器
优化文字，真正目标是让事实、公式、配置和结论完全一致。

---

## 19. 当前投稿候选与时间边界（截至2026-09-13）

```text
AISTATS 2027（CCF-C）: abstract 2026-09-29 AoE; full 2026-10-06 AoE
IJCNN 2027（CCF-C）:   regular paper 2027-01-31
ECAI 2027（CCF-B）:    full paper 2027-04-14
```

AISTATS是激进冲刺候选，不是已经确定的投稿目标：理论、统计与可信ML主题适配，但只剩约三周，
当前纯BER多seed和第二数据集仍是证据缺口。若在2026-09-20前不能形成完整英文主文骨架、经核验的
相关工作表和冻结补实验清单，应停止赶AISTATS，转向IJCNN或ECAI。CCF-A通常可口语称为该领域
顶会层级，但CCF类别不是国际影响力或录用难度的绝对刻度；AISTATS虽为CCF-C，仍是严格的主流
ML会议，不能按“普通C会”估计难度。

ECML-PKDD 2027（CCF-B）、ICDM 2027（CCF-B）、ICME 2027（CCF-B）和PAKDD 2027（CCF-C）
应持续关注；截至本日期，其2027 regular/full paper官方截止日期尚未全部公布，禁止写入猜测日期。

---

## 20. 请网页端 GPT 完成的任务

当前网页端不得再以V0.2/V0.5/V0.6作为改稿底本。polished历史快照是V0.6；
当前完整稿是V0.11，结果状态以本文件0.0节和冻结汇总为准：

```text
deliverables/cle_hfl_latex_v0_11_20261028/source/main.tex
```

全部旧版本不可变；网页端下一轮只基于V0.11与本完整交接做审稿/修改建议。当前稿已回填审计过的
主评价协议dose与AsymHFL行、local-first三seed，但仍不得删除其他`\evidenceneeded{}`。其中继续冻结：CLE-v2和DSA为
operator-level，PEW+BER为coarse family-level；Oracle operator只是不进入训练的边界消融，
不是operator-level PEW候选。

在开始英文初稿前，必须先读取并使用最新的相关工作发表格局：

```text
docs/research/status/CLE_HFL_RELATED_WORK_PUBLICATION_LANDSCAPE_2026_09_14_ZH.md
```

该文档已经区分正式主会/期刊、workshop与仅预印本，并冻结与RAHFL、FedPIN、个性化FL
spurious features、FedCD、EIIL、GroupDRO、JTT和counterfactual shortcut诊断的创新边界。
若网页端检索得到更新的正式发表信息，应附官方来源并标明核验日期，不得静默改写本文贡献。

请把自己当作严格的CCF-B类会议审稿人、联邦学习研究者和论文合作者，基于本文件完成以下任务：

1. 判断上述四项贡献是否足以形成一篇CCF-B会议论文，并指出最可能的三条拒稿理由。
2. 在不夸大PEW结构创新、不隐瞒FedDF效用边界和单private task/固定partition限制的前提下，给出最强且诚实的论文定位；主评价为seed0，机制Axis I和local-first均已三seed。
3. 审查当前证据矩阵，区分投稿前“必补实验、最好补实验、无需补实验”；不得重新要求已完成的
   map2三training-seed，也不得把主评价协议误写成第二private dataset。
4. 设计一套不把FedDF写成失败、也不把它写成无损成功的结果叙事。
5. 检查DSA识别定理、五个性质、BER有效分布/失衡压缩定理、proxy不可识别定理、PEW误差
   条件边界和JSD/CVRS反例是否还存在逻辑缺口；不得把当前平凡PEW误差界写成强保证。
6. 进行最新相关工作检索，核查CLE-HFL、federated spurious correlation、group reweighting、
   pseudo-group discovery与paired counterfactual diagnostics的创新性边界。
7. 给出论文标题、摘要、Introduction、Method、Experiments、Limitations的详细提纲。
8. 对V0.11做结构与语言审查；主评价协议剂量响应、两条AsymHFL行和local-first seeds 1/2已经完成，
   禁止改动其数字。FedDF/AugHFL/RAHFL、第二private dataset和taxonomy stress仍须标注
   `[待证据]`。不得自行编造实验数字或SOTA结论。

本节是完整任务目录，不要求本轮全部重做。本轮执行0.0节的有界写作审稿，不自动扩充方法、重启实验或重写全文。

---

## 21. 仓库内证据来源

```text
deliverables/cle_v2_mechanism_stage1_20260910/RESULT_SUMMARY_ZH.md
docs/experiments/current/CLE_DSA_THEORY_CACHE_VALIDATION_ZH.md
docs/research/status/CLE_DSA_IDENTIFICATION_THEORY_2026_09_11_ZH.md
docs/research/status/CLE_BER_MECHANISM_THEORY_2026_09_11_ZH.md
docs/research/status/CLE_PROXY_NONIDENTIFIABILITY_THEORY_2026_09_13_ZH.md
docs/research/status/CLE_HFL_RELATED_WORK_PUBLICATION_LANDSCAPE_2026_09_14_ZH.md
docs/experiments/current/CLE_V2_CROSS_SCENARIO_BINDING_MAP_ZH.md
docs/experiments/current/CLE_V2_PEW_BER_STAGE2_OPENI_ZH.md
docs/experiments/current/CLE_V2_SPURIOUS_BASELINE_SCREEN_ZH.md
docs/experiments/current/CLE_V2_SPURIOUS_FINAL_MAP2_ZH.md
deliverables/cle_v2_spurious_final_map2_20260913/RESULT_SUMMARY_ZH.md
deliverables/cle_v2_oracle_granularity_formal_20260911/RESULT_SUMMARY_ZH.md
deliverables/cle_v2_feddf_plugin_formal_20260911/RESULT_SUMMARY_ZH.md
docs/research/status/CLE_HFL_PAPER_CLOSURE_2026_09_11_ZH.md
docs/experiments/current/CLE_HFL_SUBMISSION_EXPERIMENT_MATRIX_2026_09_15_ZH.md
docs/experiments/current/CLE_V2_MODERATE_ALPHA10_RHO05_DATASET_2026_10_13_ZH.md
deliverables/cle_hfl_latex_v0_8_20261025/source/main.tex
deliverables/cle_hfl_latex_v0_9_20261026/source/main.tex
deliverables/cle_v2_local_first_multiseed_20261026/RESULT_SUMMARY_ZH.md
```

原始输出位于`outputs/`，解析报告位于`deliverables/`。Smoke与benchmark只验证执行或成本，不是
科学证据。本文所有正式数字均应回到以上正式报告及其冻结协议核验。

---

## 22. 2026-09-22双轴实验结构更新

> 2026-10-25覆盖说明：Axis I继续使用strong协议并以DSA为主要科学终点；Axis II不再使用旧
> strong Local/FedMD结果，也不再要求十臂全部进入正文。下面Axis II列表以本节新版本为准。

论文的基线比较必须分成两条轴，二者回答不同问题，禁止混成一张无条件排行榜。

### Axis I：通用shortcut与group-risk目标

在完全匹配的AsymHFL协议下比较`ERM / CVaR-DRO / PEW+GroupDRO / PEW+BER`。held-out map2
已经完成40轮、training seeds 0/1/2 Formal；它回答BER收益是否只是通用高损失样本优化或标准
GroupDRO。三seed pooled DSA分别为`0.250530 / 0.086654 / 0.205417 / 0.071470`。安全结论是
PEW+BER在这四个正式方法中取得最佳pooled shortcut--utility trade-off；不能写成对所有shortcut
方法的普遍最优。JTT仍是12轮screen-only，EIIL未做Formal迁移。

### Axis II：HFL机制家族背景比较

最终正文统一在主评价协议held-out-map2、`alpha=1.0/rho=0.5/gamma=0.9`下比较八行：

```text
Local/ERM / FedMD / FedProto / FedDF
AugHFL / RAHFL / AsymHFL--ERM / AsymHFL+PEW+BER
```

它回答普通模型异构通信和corruption-robust HFL是否会自然消除CLE，并把DSA与可用任务效用放在
同一协议下。当前5/8行已完成：Local/FedMD/FedProto及两条AsymHFL行；FedDF/AugHFL/RAHFL
仍是`[待证据]`。当前不做单独AugHFL/RAHFL官方40+40表；Axis II不能
声称逐行复现各论文完整recipe，也不承担BER因果归因。FedTGP、KT-pFL、FCCL、RHFL保留为
已实现的条件性扩展，不进入预冻结正文八行。

V0.10已经包含Axis I表、已完成主评价协议剂量响应和Axis II五条正式行。网页端后续改稿应把
Experiments明确组织为：

```text
Axis I: matched shortcut-objective attribution
Axis II: protocol-matched HFL mechanism context
Shared validation: local-first / cross-map / M3 genuine second dataset / S1 taxonomy stress
```

最新S2-v2工程与忠实度边界：

```text
docs/research/baselines/CLE_HFL_S2_V2_AND_FEDTGP_KILL_TEST_2026_09_22_ZH.md
deliverables/cle_hfl_map2_multiseed_20260922/RESULT_SUMMARY_ZH.md
```

---

## 23. 2026-10-31网页端实际使用方式

下一次交给网页端GPT时，最小上传集合应为：

```text
1. 本完整交接文档
2. V0.11 LaTeX源码包（需要精确源码修改时上传）
3. V0.11已编译PDF（本包为原字节复制，尚未接入10月30日五图候选）
4. references.bib与citation audit
5. CURRENT_EVIDENCE_AND_REVIEW_LOCKS_ZH.md和WEB_REVIEW_PROMPTS_ZH.md
```

不要上传`docs/handoffs/latest.md`让网页端自行理解论文；该文件是Codex工程接力日志，包含大量
历史协议和失败路线。不要再单独使用`20260925`短交接，也不要让网页端根据旧V0.2/V0.5重新生成
全文。当前应基于V0.11先审稿再选择性修改；不得把仍pending的三条Axis II、第二private dataset
或taxonomy stress写成已完成。local-first多seed已经完成，论文主文报告3/3 DSA机制门槛复现；
辅助学习下限仅属于内部审计边界，不应抢占摘要、引言、主表或结论的叙事位置。

---

## 24. 2026-10-26证据状态与成本边界

### 24.1 local-first training seeds 1/2（已完成）

该实验不是再次证明PEW+BER有效，而是检验seed-0的机制归因能否跨训练随机性保持。每个training
seed都需要四个matched arms：`HFL gamma0/HFL gamma.9/Local gamma0/Local gamma.9`，并计算：

```text
Delta_HFL   = DSA(HFL gamma.9) - DSA(HFL gamma0)
Delta_Local = DSA(Local gamma.9) - DSA(Local gamma0)
Delta_comm  = Delta_HFL - Delta_Local
```

结果已经回传并完成审计。三个seed的`Delta_HFL`为`0.119889/0.123717/0.124638`，
`Delta_Local`为`0.107960/0.127287/0.131210`，`Delta_comm`为
`0.011929/-0.003570/-0.006573`。方向性、binding-specificity和local-first门槛3/3复现；
完整预注册verdict为Seed 0/1 GO，Seed 2因strong-CLE HFL grid accuracy `19.0683% < 20%`
判为undertrained。不得写成“三seed完整门槛全过”，也不得丢弃Seed 2后只报告成功seed。

### 24.2 第二private dataset

机制协议与主评价协议都以CIFAR-10为private semantic task；它们增强的是同一任务内的内部有效性，
不是跨数据集外部有效性。第二private dataset用于检验CLE、DSA和缓解结果是否只是CIFAR-10类别、
纹理和合成corruption组合的特例。它不是所有会议论文的硬性规则，也不意味着RAHFL等工作必须采用
大量数据集；但对本文这种“新受控failure mode + taxonomy-assisted mitigation”的主张，它能直接
降低`synthetic CIFAR-10 artifact`这一审稿攻击。若版面和算力有限，可把它作为CCF-B增强项并诚实
保留单数据集限制，不能把公共蒸馏数据集误写成第二private task。

### 24.3 bounded taxonomy stress

PEW依赖预先定义的粗corruption family。bounded stress只预注册一个有限错配，例如冻结模型后评估
held-out operator或固定的双corruption组合，并报告PEW混淆、DSA与效用如何退化。它的目标是刻画
taxonomy不完全覆盖时的方法边界，不是证明taxonomy-free或开放世界泛化。优先采用evaluation-only
低成本版本；若改动训练分布，则必须另立协议，不能与现有Formal混表。

### 24.4 本轮与后续预算

本轮四个主评价协议Formal实测训练加分析为`51.8363` V100小时，按OpenI `6积分/小时`约
`311.02`积分；未计依赖安装、解包和上传的小额平台开销。单项约`76.61--78.35`积分。

Axis II若继续补表，当前保守预算是：FedDF单行至少预留约`100`积分；RAHFL单行按已有robust
benchmark预留约`120`积分；两者分账号运行建议总计至少`220`积分。AugHFL再增加约`102--120`
积分。以上是启动预算而非承诺时长；必须复核当前入口的recovery/fail-closed保护，且不因补齐表格
而改变已冻结的40轮、64 local batches、seed 0和主评价协议map2设置。
