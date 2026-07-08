# FedCLEAR / CLE-HFL 方案草案

更新时间：2026-07-08

本文档用于整理一个新的研究方向：

```text
Corruption-Label Entanglement in Heterogeneous Federated Learning
损坏-类别纠缠的异构联邦学习
```

目标是摆脱“只是 RAHFL 小改”的感觉，把问题从普通 corrupted clients 提升为：

```text
模型异构 + 数据异构 + 数据损坏 + 损坏-类别虚假相关
```

这份文档是方案草案，不是最终论文版本。后续需要结合实验信号继续收敛。

---

## 1. 新问题：Corruption-Label Entanglement in HFL

### 1.1 问题动机

现有鲁棒异构联邦学习方法，例如 RAHFL，主要考虑：

```text
不同客户端模型结构不同；
不同客户端数据可能被损坏；
需要通过鲁棒本地训练和异构通信提升整体性能。
```

但是它通常隐含一个假设：

```text
corruption 是与 label 相对独立的 nuisance factor。
```

也就是说，损坏只是让图像变差，但不会系统性地和某些类别绑定。

真实场景中，这个假设可能不成立。由于设备、环境、采集策略、类别稀缺性等因素，某些类别更容易和某些损坏共同出现：

```text
自动驾驶：
  夜晚/雨雪场景中行人、车灯、交通标志样本更多；
  白天晴天场景中普通车辆、道路样本更多。

医疗影像：
  某些少见病种可能主要来自低质量设备或老旧扫描仪；
  不同医院的类别分布和成像质量同时变化。

工业视觉：
  某类缺陷只在高速产线出现，因此同时伴随 motion blur；
  某类产品来自旧摄像头，因此同时伴随 noise/compression。

移动端视觉：
  某些用户常在夜间拍宠物，另一些用户常在运动场景拍车辆；
  类别偏好和成像退化同时发生。
```

这会导致模型学习错误捷径：

```text
训练集中：
  猫大多清晰；
  狗大多模糊。

模型可能学到：
  清晰 -> 猫；
  模糊 -> 狗。
```

这种 shortcut 在训练分布上可能有效，但在反事实测试中会崩：

```text
模糊猫
清晰狗
```

因此，新问题不是简单问：

```text
数据损坏了怎么办？
```

而是问：

```text
当损坏模式和类别标签在客户端本地数据中纠缠时，
异构联邦模型如何避免学习 corruption-label shortcut？
```

### 1.2 与 RAHFL 的区别

RAHFL 主要研究：

```text
Corrupted clients
```

其核心思路是：

```text
1. 用 AugMix/JSD/DCL 提升本地鲁棒表征；
2. 用 AsymHFL 避免弱客户端污染强客户端。
```

但在 CLE-HFL 中，困难不是“某个客户端整体质量差”这么简单，而是：

```text
同一个客户端内部，不同类别和不同损坏模式发生绑定；
本地模型可能把损坏当作类别线索；
通信时，客户端 public logits 也可能携带这种 shortcut bias。
```

因此，CLE-HFL 要研究的是：

```text
corruption-label shortcut learning
under model-heterogeneous federated learning
```

这比普通 corrupted-client setting 更细，也更贴合数据异构和数据损坏同时存在的情况。

---

## 2. 新协议：CLE-HFL Benchmark Protocol

CLE-HFL 是一个可控实验协议，用来系统构造：

```text
label-skew + corruption-label dependency + model heterogeneity
```

它不是一个新数据集名字，而是一套生成和评估规则。

### 2.1 符号

```text
K: 客户端数量
C: 类别数
G: corruption context/operator pool
alpha: label-skew Dirichlet 参数
rho 或 gamma: corruption-label entanglement strength
```

其中：

```text
G = {g_1, g_2, ..., g_M}
```

可以实例化为：

```text
noise / blur / weather / digital
```

也可以换成：

```text
不同 severity
不同设备退化
不同压缩链路
不同采集条件
```

重点是：算法不应依赖固定四类名字，四类只是 CIFAR-style benchmark 的一种实例化。

### 2.2 Step 1：先做 label-skew

对 CIFAR-10 训练集，先用 Dirichlet 分布划分到客户端：

```text
n_{k,c} ~ Dirichlet(alpha)
```

例如 alpha=0.5 时，一个客户端可能主要拥有：

```text
client 0:
  class 4, 6, 7 很多；
  class 1, 8, 9 很少或没有。
```

这一步制造普通数据异构。

### 2.3 Step 2：再做 corruption-label dependency

对每个客户端 k，构造一个类别到损坏上下文的映射：

```text
phi_k(c) -> g
```

意思是：

```text
在 client k 上，类别 c 更容易伴随 corruption g 出现。
```

然后定义：

```text
P_k(g | y=c)
= gamma * 1[g = phi_k(c)]
  + (1 - gamma) * Uniform(G)
```

其中：

```text
gamma = 0 表示 corruption 和 label 独立；
gamma 越大，corruption-label 纠缠越强；
gamma = 1 表示每个类别几乎绑定到固定 corruption。
```

举例：

```text
client 0:
  class 4 -> blur
  class 6 -> noise
  class 7 -> clean/mild

client 1:
  class 2 -> digital
  class 3 -> weather
  class 8 -> blur
```

于是本地数据里会出现：

```text
class 4 大多是 blur；
class 6 大多是 noise；
class 7 大多是 mild/clean。
```

模型如果不加约束，就可能学到：

```text
blur -> class 4
noise -> class 6
```

这就是 corruption-label shortcut。

### 2.4 Step 3：生成训练图像

对每个样本：

```text
(x, y) in client k
```

先采样：

```text
g ~ P_k(g | y)
s ~ severity distribution
```

然后生成：

```text
\tilde{x} = T_{g,s}(x)
```

训练集中保存：

```text
image
label y
corruption context g
severity s
```

### 2.5 Step 4：反事实测试集

CLE-HFL 的核心不是测试原始偏置分布，而是测试模型有没有学 shortcut。

因此要构造 counterfactual test：

```text
同一类别 y，在不同 corruption context g 下都出现。
```

例如：

```text
cat-clean
cat-noise
cat-blur
cat-weather
cat-digital

dog-clean
dog-noise
dog-blur
dog-weather
dog-digital
```

这不是说真实部署分布一定均匀，而是一个反事实压力测试：

```text
如果模型真的学到了类别语义，
那么同一类别不应该因为 corruption 改变而严重掉点。
```

### 2.6 Optional：unseen corruption method test

为了避免审稿人说“训练和测试都用同几类 corruption”，可以做组内留出：

```text
train noise:
  gaussian_noise, shot_noise

test noise:
  impulse_noise, speckle_noise
```

或者：

```text
train severity: 1-3
test severity: 4-5
```

这样 CLE-HFL 就不仅是记住 corruption 组名，而是检验更真实的泛化能力。

---

## 3. 新方法：FedCLEAR

暂定名字：

```text
FedCLEAR
Federated Corruption-Label Entanglement-Aware Robust Learning
```

中文：

```text
损坏-类别纠缠感知的鲁棒异构联邦学习
```

目标：

```text
1. 本地打破 corruption-label shortcut；
2. 通信时减少 shortcut knowledge 的传播；
3. 在模型异构下仍然能协作。
```

### 3.1 总体框架

每个客户端 k 有模型：

```text
f_k = h_k o b_k
```

其中：

```text
b_k: backbone，可以异构；
h_k: classifier head，输出 C 类 logits。
```

由于模型异构，不直接聚合参数。

FedCLEAR 仍然采用 output-space / public-data based communication：

```text
客户端上传 public views 上的 logits 或统计量；
服务端构造 teacher；
客户端本地蒸馏。
```

但它不直接相信单个 public logits，而是关注：

```text
跨 counterfactual views 的稳定语义部分。
```

### 3.2 模块一：Counterfactual Corruption Augmentation

对本地样本：

```text
(x, y)
```

不管它原始来自什么 corruption，都额外生成多个反事实视图：

```text
x^{(1)} = A_1(x)
x^{(2)} = A_2(x)
...
x^{(M)} = A_M(x)
```

这些 augmentation 可以来自：

```text
AugMix-style random corruption
PRIME-style perturbation
CIFAR-C style corruption pool
severity ladder
```

核心不是具体是哪四类，而是：

```text
同一类别要暴露在不同 corruption/context 下。
```

这一步的目的：

```text
破坏训练集中 y 和 g 的固定绑定关系。
```

例如原始训练里：

```text
dog 大多 blur
```

反事实增强后：

```text
dog-clean
dog-noise
dog-digital
dog-weather
```

模型就不能简单依赖：

```text
blur -> dog
```

### 3.3 模块二：Class-Balanced Invariant Alignment

本地特征：

```text
q_k(x) = normalized feature from backbone
```

对同一类别 y 的不同 counterfactual views，要求特征靠近：

```text
q_k(x^{(a)}) ≈ q_k(x^{(b)})
```

但要避免 head class 支配训练，因此做 class-balanced weighting。

一个可能的损失：

```text
L_inv =
1 / |C_k| * sum_{c in C_k}
  1 / |I_c| * sum_{i in I_c}
    SupCon(q_i, positives=same label across corruptions)
```

目标：

```text
同类不同 corruption 靠近；
不同类即使 corruption 相同也分开。
```

这和 RAHFL DCL 的区别：

```text
RAHFL DCL:
  强/弱增强特征对齐，增强鲁棒表示。

FedCLEAR invariant alignment:
  明确针对 corruption-label shortcut，
  让类别语义跨 counterfactual corruption 保持不变。
```

### 3.4 模块三：Shortcut-Suppressed Heterogeneous Distillation

服务器拿 public images：

```text
u
```

构造多个随机 counterfactual views：

```text
u^{(m)} = A_m(u)
```

每个客户端输出：

```text
z_k^{(m)}(u)
p_k^{(m)}(u) = softmax(z_k^{(m)}(u))
```

定义稳定语义预测：

```text
\bar{p}_k(u) = 1/M * sum_m p_k^{(m)}(u)
```

定义 shortcut sensitivity：

```text
S_k(u) = 1/M * sum_m JSD(p_k^{(m)}(u), \bar{p}_k(u))
```

如果 S_k(u) 大，说明客户端对 corruption/context 很敏感，可能依赖 shortcut。

教师权重：

```text
w_k(u) = exp(-S_k(u) / tau)
```

全局 teacher：

```text
p_T(u) =
sum_k w_k(u) * \bar{p}_k(u)
/ sum_k w_k(u)
```

学生蒸馏：

```text
L_distill =
KL(p_T(u) || \bar{p}_i(u))
+ lambda_cons * sum_m JSD(p_i^{(m)}(u), \bar{p}_i(u))
```

含义：

```text
1. 学全局稳定语义预测；
2. 不学习对 corruption 很敏感的教师；
3. 进一步压低自身 shortcut sensitivity。
```

### 3.5 总损失

客户端 k 的本地目标：

```text
L_k =
L_CE
+ lambda_jsd * L_JSD
+ lambda_inv * L_inv
+ lambda_distill * L_distill
```

其中：

```text
L_CE: 基础分类；
L_JSD: 本地多视图预测一致性；
L_inv: 类别-损坏反事实不变对齐；
L_distill: shortcut-suppressed heterogeneous distillation。
```

---

## 4. 实验设计

### 4.1 必跑基线

因为我们仍然在鲁棒异构联邦学习方向，需要继续和 RAHFL 比。

主基线：

```text
RAHFL:
  AugMix + DCL + AsymHFL
```

本地强基线：

```text
AugMix + DCL local-only
```

当前已有强方法：

```text
SARA + AsymHFL
```

建议还加入：

```text
RAHFL + FedCLEAR local objective only
FedCLEAR full
```

如果计算允许，可加：

```text
FedMD/FedDF style public-logit average
FedProto-style method
```

但由于模型异构，普通 FedAvg 不一定适合作主基线。

### 4.2 实验变量

label-skew：

```text
alpha = 1.0, 0.5, 0.3, 0.1
```

entanglement strength：

```text
gamma = 0.0, 0.3, 0.6, 0.9
```

其中：

```text
gamma = 0.0:
  corruption 与 label 独立，退化为普通 corruption robustness。

gamma = 0.9:
  corruption 与 label 强绑定，shortcut 最严重。
```

多 seed：

```text
seed = 0, 1, 2
```

模型异构：

```text
ResNet10
ResNet12
ShuffleNet
MobileNetV2
```

### 4.3 实验表格建议

主表：

```text
alpha=0.5, gamma=0.6 或 0.9
比较 RAHFL / local-only / SARA+AsymHFL / FedCLEAR
```

鲁棒性表：

```text
不同 gamma 下的 avg_acc / worst_acc / worst-class-corruption acc
```

消融表：

```text
FedCLEAR full
- counterfactual local alignment
- shortcut-suppressed teacher weighting
- class-balanced weighting
```

泛化表：

```text
seen corruption methods
unseen corruption methods
seen severity
unseen high severity
```

---

## 5. 指标设计

### 5.1 Average Accuracy

```text
avg_acc = 平均客户端准确率
```

用于看整体性能。

### 5.2 Worst Client Accuracy

```text
worst_acc = min_k Acc_k
```

用于看最弱客户端是否被保护。

### 5.3 Worst Class-Corruption Accuracy

对测试集按：

```text
(class c, corruption context g)
```

分组。

定义：

```text
Acc(c,g) = accuracy on class c under corruption g
```

然后：

```text
WCCA = min_{c,g} Acc(c,g)
```

意义：

```text
模型是否在某个“类别-损坏组合”上崩溃。
```

这对 CLE-HFL 很重要，因为 shortcut 往往在反事实组合上暴露。

### 5.4 Counterfactual Gap

对每个类别 c：

```text
Gap(c) = max_g Acc(c,g) - min_g Acc(c,g)
```

整体：

```text
CFG = 1/C * sum_c Gap(c)
```

越小越好。

含义：

```text
同一个类别在不同 corruption 下表现差异越小，
说明模型越不依赖 corruption shortcut。
```

### 5.5 Shortcut Sensitivity

对同一张测试图像 x，生成多个 counterfactual corruption views：

```text
x^{(1)}, ..., x^{(M)}
```

模型预测：

```text
p^{(m)}
```

定义：

```text
SS(x) = 1/M * sum_m JSD(p^{(m)}, mean_m p^{(m)})
```

越小越好。

意义：

```text
模型面对语义不变的 corruption 变化时，预测是否稳定。
```

---

## 6. 初步文献排查结论

不能直接说“没人做”。更准确的说法是：

```text
已有工作研究过 FL 中的数据异构、spurious correlation、group robustness，
也已有 RAHFL 研究模型异构 + corrupted clients。

但目前没有看到一个直接聚焦：
  corruption-label entanglement
  + model-heterogeneous FL
  + corrupted robust learning
的完整设定。
```

相关方向包括：

```text
RAHFL:
  研究 model heterogeneous + corrupted clients，
  但没有显式建模 corruption-label shortcut。

FedDiverse / FL spurious correlation:
  研究 federated learning 中 spurious correlations 和数据异构，
  但不是专门针对 corruption-label dependency 和异构模型鲁棒通信。

JTT / group robustness:
  中心化场景下处理 spurious correlations 和 worst-group accuracy，
  可作为动机支撑，但不是 HFL。
```

因此，CLE-HFL 的潜在贡献可以写成：

```text
We identify and benchmark corruption-label entanglement as a neglected failure
mode in robust heterogeneous federated learning.
```

但正式写论文前仍需做更系统的 related work 排重。

---

## 7. 这条路线的优点与风险

### 7.1 优点

相比 corruption-skew，CLE-HFL 更稳：

```text
1. 不需要声称部署时每个客户端会遇到均衡 corruption；
2. balanced/counterfactual test 是为了检测 shortcut，不是模拟真实平均分布；
3. 问题动机来自 spurious correlation / shortcut learning，机器学习社区更容易理解；
4. 数据异构和数据损坏不再是机械叠加，而是发生纠缠；
5. 更容易解释为什么本地训练不够，因为本地训练会强化 shortcut。
```

### 7.2 风险

```text
1. 协议是人造的，需要证明合理性；
2. 如果 RAHFL 的 AugMix/JSD 已经很好地打破 shortcut，FedCLEAR 增益可能不大；
3. 如果 public data 跨域太强，shortcut-suppressed distillation 信号可能弱；
4. 方法不能堆太多模块，否则会被认为是复杂拼接。
```

### 7.3 最小可行实验路线

建议先做最小版本：

```text
1. 生成 CLE-HFL 数据协议：
   alpha=0.5, gamma=0.6 或 0.9, seed=0

2. 跑 RAHFL：
   看 RAHFL 在 counterfactual gap / WCCA 上是否暴露短板。

3. 跑 SARA + AsymHFL：
   看已有方法是否已经足够强。

4. 再决定是否实现完整 FedCLEAR。
```

如果 RAHFL 在 CLE-HFL 上没有明显短板，这条路线就不值得继续烧算力。

如果 RAHFL 平均准确率还行，但：

```text
counterfactual gap 大
worst-class-corruption acc 低
```

那 FedCLEAR 就有明确改进空间。

---

## 8. 当前推荐结论

相比之前的 corruption-skew 场景，CLE-HFL 更适合作为论文主场景。

一句话定位：

```text
RAHFL 解决 corrupted clients；
FedCLEAR 解决 corruption-label shortcut under heterogeneous FL。
```

如果要冲 CCF-B，小论文故事可以是：

```text
1. 提出 CLE-HFL 问题；
2. 构造 CLE-HFL benchmark protocol；
3. 提出 FedCLEAR；
4. 在 avg_acc、worst_acc、counterfactual gap、worst-class-corruption acc 上系统验证。
```

