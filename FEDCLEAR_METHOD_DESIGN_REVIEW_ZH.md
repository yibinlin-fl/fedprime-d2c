# FedCLEAR 方法设计审核稿

更新时间：2026-07-10
文档状态：方法定义 v0.1 已编码并通过本地 smoke test，等待 OpenI probe 验证
适用问题：CLE-HFL（Corruption-Label Entanglement in Heterogeneous Federated Learning）

---

## 0. 本文档的目的

本文档定义一个当前可以进入实现阶段的完整方法，而不是再记录一个松散想法。

待解决的问题同时包含：

```text
1. 数据损坏：训练和部署图像可能受到多种 corruption；
2. 数据异构：不同客户端类别分布不同，且类别与 corruption 存在客户端特有的虚假关联；
3. 模型异构：不同客户端使用 ResNet、ShuffleNet、MobileNet 等不同架构；
4. 知识传播：服务器不能直接平均不同结构模型的参数。
```

暂定方法名：

```text
FedCLEAR
Federated Corruption-Label Entanglement-Aware Robust Learning
```

核心结构：

```text
FedCLEAR = CCRE 本地反事实风险学习 + IRD 不变残差蒸馏
```

其中：

```text
CCRE = Class-Conditional Counterfactual Risk Equalization
IRD  = Invariant-Residual Distillation
```

这两个模块分别对应 CLE-HFL 的两个关键失败环节：

```text
CCRE：阻止客户端在私有训练中形成“corruption -> label”捷径；
IRD ：阻止这种客户端特有捷径通过联邦通信传播给其他客户端。
```

---

## 1. CLE-HFL 问题定义

### 1.1 变量与因果关系

记：

```text
Y：类别标签；
S：决定类别的语义信息；
G：corruption context，例如某种噪声、模糊或数字失真；
X：最终观测到的图像；
K：客户端编号。
```

理想的数据生成关系为：

```text
Y -> S -> X
G ------> X
```

语义 `S` 和环境 `G` 都会改变图像 `X`，但预测目标应主要依赖 `S`。

CLE-HFL 中，客户端的数据采集或筛选过程额外导致：

```text
P_k(G | Y) != P_k(G)
```

例如客户端 0 的猫大多带 noise，狗大多带 blur；客户端 1 的映射关系又不同。模型因此可能学到：

```text
noise -> cat
blur  -> dog
```

而不是学习猫和狗的真实语义。

### 1.2 现实动机

CLE-HFL 并不是假设现实数据会被人为平均分成四种 corruption，而是描述一种常见的数据采集偏差：标签和采集条件在客户端内部并非独立。

例如：

```text
自动驾驶：某城市客户端的行人样本主要来自夜间雨天，卡车样本主要来自高速运动模糊；
医疗影像：某医院的特定病种主要由某台扫描仪采集，因此病种和扫描伪影发生关联；
工业检测：某类缺陷集中出现在特定产线，缺陷标签和相机、光照或压缩模式发生关联。
```

只要部署时采集流程、设备或环境发生变化，原先的 `label-corruption` 组合就会重新排列。模型若依赖采集条件而不是语义，就会在新的组合上失败。

因此论文故事不是“测试分布为什么必须均衡”，而是：

```text
用反事实 class-corruption 组合系统审计模型是否依赖采集 shortcut，
并提高模型面对组合重排时的最坏情况性能。
```

### 1.3 Gamma 的严格含义

当前协议有四个 corruption family。对客户端 `k` 的类别 `c`，先指定主导组 `phi_k(c)`，然后按下式采样损坏组：

$$
P_k(g\mid y=c)
=
\gamma\,\mathbf 1[g=\phi_k(c)]
+
\frac{1-\gamma}{4}.
$$

因此：

| gamma | 主导组概率 | 每个非主导组概率 |
|---:|---:|---:|
| 0.0 | 25.0% | 25.0% |
| 0.6 | 70.0% | 10.0% |
| 0.9 | 92.5% | 2.5% |

不同 gamma 数据集保持以下条件完全一致：

```text
原始 CIFAR-10 样本；
Dirichlet alpha；
partition seed；
客户端数量；
每个客户端的样本索引；
模型结构；
训练和通信预算。
```

只改变 `P_k(G | Y)` 的纠缠强度。因此 gamma 增大后指标单调恶化，可以归因于 corruption-label entanglement 增强，而不是 label partition 改变。

### 1.4 四个是损坏家族，不是四种损坏

当前协议使用 4 个损坏家族、16 个具体算子和 5 级 severity：

| 家族 | 具体算子 |
|---|---|
| noise | gaussian noise、shot noise、impulse noise、speckle noise |
| blur | defocus blur、glass blur、motion blur、zoom blur |
| weather | snow、frost、fog、spatter |
| digital | contrast、brightness、JPEG compression、pixelate |

四个家族只承担两个作用：

```text
1. 构造可控的 Y-G 相关性；
2. 统计 class-corruption 指标。
```

FedCLEAR 的训练目标不得硬编码这四个家族。训练只接收一个可扩展、与标签独立的 intervention bank，因此之后增加新的 corruption 算子时，不需要改模型和损失公式。

### 1.5 当前证据说明了什么

在 `alpha=0.5, seed=0` 下，RAHFL 结果为：

| gamma | avg_acc | worst_acc | WCCA | CFG |
|---:|---:|---:|---:|---:|
| 0.0 | 52.17 | 44.17 | 35.35 | 2.54 |
| 0.6 | 50.82 | 42.83 | 25.88 | 5.91 |
| 0.9 | 46.72 | 38.16 | 19.32 | 10.91 |

从 gamma 0.0 到 0.9：

```text
avg_acc   下降 5.45；
worst_acc 下降 6.02；
WCCA      下降 16.02；
CFG       上升 8.37。
```

这证明 RAHFL 在当前协议中会随着 entanglement 增强而系统退化，尤其是最差类别-损坏组合明显崩溃。

这仍属于“问题成立的初步证据”，不是论文最终证明。正式协议还需要多 seed、官方 corruption 实现、共享测试集哈希和未见 corruption 测试。

---

## 2. 为什么 RAHFL 仍会失败

### 2.1 AugMix + JSD 解决的是样本级预测一致性

RAHFL 对一张图像生成多个增强视图，并约束预测分布接近。它可以抑制一般的数据损坏敏感性，但不能直接保证：

```text
对每一个类别，模型在不同 corruption context 下都具有相近且正确的风险。
```

三个视图可能得到相近预测，但也可能一致地预测错误。JSD 只约束“彼此接近”，没有使用真实标签判断这种一致是否正确。

### 2.2 DCL 解决的是强弱视图特征关系

DCL 强化 clean、weak、strong view 之间的局部表征一致性，但没有显式优化：

$$
\max_g R(c,g),
$$

也没有直接对应 WCCA 或 CFG。严重 label skew 下，head class 仍可能主导本地表示。

### 2.3 AsymHFL 可能传播客户端特有 shortcut

原始 AsymHFL 在 public data 上进行客户端级教师选择。一个客户端可以总体准确率较高，但对某些 `class-corruption` 组合存在严重 shortcut。整体路由无法分辨这种细粒度风险。

此外，当前统一 runner 中的原始 RAHFL 路由曾使用测试准确率选择教师。FedCLEAR 不使用测试标签，也不使用测试准确率参与训练决策。

---

## 3. FedCLEAR 总览

FedCLEAR 的目标不是让所有特征在所有环境中完全一样，而是学习：

$$
Z \perp G \mid Y,
$$

即类别确定后，表示和预测尽量不依赖 corruption context。

整体流程：

```text
私有样本
  -> 标签无关的多环境反事实视图
  -> CCRE 优化每个类别的最差环境风险
  -> 得到本地抗 shortcut 模型

公共无标签样本
  -> 多环境反事实视图
  -> 客户端提取跨视图不变 logit anchor
  -> 服务器鲁棒聚合 anchor
  -> IRD 对客户端最差公共视图进行蒸馏
```

模块分工：

| 问题 | FedCLEAR 处理方式 |
|---|---|
| 数据损坏 | 多环境干预与最坏环境风险优化 |
| label-skew | 先按类别计算风险，再对本地出现类别等权平均 |
| corruption-label shortcut | 直接优化同类跨环境正确性，不允许仅靠 context 猜标签 |
| 模型异构 | 只交换固定 `C` 维 logits anchor，不聚合参数或 feature |
| shortcut 传播 | 通信前分离跨视图 anchor 与环境 residual，只聚合 anchor |
| 测试泄漏 | 不使用 test label、test accuracy 或 oracle prior |

---

## 4. 模块一：CCRE 本地反事实风险学习

### 4.1 反事实视图生成

对客户端 `k` 的私有样本 `(x_i, y_i)`，从与标签无关的 intervention distribution `Q_A` 中采样 `M` 个干预：

$$
a_m \sim Q_A,\qquad
x_i^{(m)} = a_m(x_i),\quad m=1,\ldots,M.
$$

关键约束：

```text
Q_A 不读取 y；
Q_A 不读取客户端的 class-corruption map；
Q_A 不固定依赖四个 family 名称；
干预必须尽量保持类别语义；
训练和 baseline 使用相同的基础图像与同等视图预算。
```

首版建议 `M=3`，与 RAHFL 的多视图预算保持接近，避免仅靠更多前向计算获得提升。

### 4.2 类别-环境风险矩阵

在一个 batch 中，类别 `c`、视图 `m` 的分类风险为：

$$
R_{k,c,m}
=
\frac{1}{|B_c|}
\sum_{i\in B_c}
\operatorname{CE}
\left(f_k(x_i^{(m)}),y_i\right).
$$

这里得到一个局部的：

```text
本地出现类别数 x M 个反事实环境
```

风险矩阵。

### 4.3 平滑最坏环境目标

对类别 `c`，使用 log-sum-exp 近似最坏环境风险：

$$
\widetilde R_{k,c}
=
\tau_r
\log\left[
\sum_{m=1}^{M}
\exp\left(
\frac{R_{k,c,m}}{\tau_r}
\right)
\right].
$$

当 `tau_r` 较小时，它接近：

$$
\max_m R_{k,c,m}.
$$

并满足：

$$
\max_m R_{k,c,m}
\le
\widetilde R_{k,c}
\le
\max_m R_{k,c,m}+\tau_r\log M.
$$

因此 CCRE 是最坏环境风险的可微上界，而不是普通的平均 CE。

仅在 batch 内等权仍不够，因为极少类进入 batch 的次数更少。客户端在本地根据类别计数估计类别 `c` 进入大小为 `B` 的 batch 的概率：

$$
\rho_{k,c}^{(B)}
=
1-\left(1-\frac{n_{k,c}}{N_k}\right)^B.
$$

使用无额外可调参数的出现概率校正：

$$
w_{k,c}=\frac{1}{\rho_{k,c}^{(B)}+\epsilon}.
$$

再对当前 batch 出现的类别加权平均：

$$
\mathcal L_{\mathrm{CCRE}}^k
=
\frac{
\sum_{c\in\mathcal C_B}w_{k,c}\widetilde R_{k,c}
}{
\sum_{c\in\mathcal C_B}w_{k,c}
}.
$$

类别计数只在客户端内存中使用，不上传到服务器。这个校正使极少类虽然出现在较少 batch 中，但每次出现时获得更高权重，从而让整个 epoch 的期望贡献更接近类别等权。

这条损失同时解决两个问题：

```text
类别均衡：校正类别进入 batch 的概率，减少 head class 支配；
环境鲁棒：每个类别中当前风险最高的反事实环境得到最大梯度。
```

### 4.4 为什么它比单独 JSD 更贴近 CLE-HFL

假设一张猫图在三个视图下得到：

```text
view 1 -> cat 0.80
view 2 -> cat 0.75
view 3 -> dog 0.70
```

JSD 会要求三个分布接近，但它本身没有指定应该向哪个正确方向接近。

CCRE 使用真实标签 `cat` 计算每个视图的 CE，`view 3` 会形成最大的类别-环境风险，因此获得最大修正梯度。模型不能通过“三个视图一致地预测 dog”来降低 CCRE。

### 4.5 本地总损失

FedCLEAR 首版保留多视图 JSD 作为通用 corruption robustness 基础，但使用显式、标签无关的 counterfactual operator bank 生成视图，不调用 RAHFL DCL：

$$
\mathcal L_{\mathrm{private}}^k
=
\mathcal L_{\mathrm{CE}}
+
\lambda_{js}\mathcal L_{\mathrm{JSD}}
+
\lambda_{cr}\mathcal L_{\mathrm{CCRE}}^k.
$$

其中 CCRE 使用同一批多视图前向结果，不重复运行 backbone。

保留 JSD 的原因是它负责样本级平滑；加入 CCRE 的原因是它负责类别条件下的最坏环境正确性。两者目标不同。RAHFL AugMix+DCL 仍作为不改动的强基线保留。

---

## 5. 模块二：IRD 不变残差蒸馏

### 5.1 公共数据的定位

公共数据不用于：

```text
估计 private class prior；
猜测客户端缺失了哪些类别；
生成真实 CIFAR-10 标签；
计算客户端测试准确率。
```

公共图片只作为统一 probe，测量不同异构模型面对同一组环境干预时的响应。

这规避了 D2C 的核心问题：跨域 CIFAR-100 平均预测不是 CIFAR-10 private prior。

### 5.2 多环境公共响应

对公共图片 `u`，各客户端使用由服务器种子确定的相同 `M` 个干预：

$$
u^{(m)}=a_m(u).
$$

客户端 `k` 输出：

$$
z_{k,m}(u)=f_k(u^{(m)})\in\mathbb R^C.
$$

### 5.3 异构 logit 尺度标准化

不同模型的 logit 尺度不同。对每个样本和视图进行类别维标准化：

$$
\widehat z_{k,m}(u)
=
\frac{
z_{k,m}(u)-\mu_c(z_{k,m}(u))
}{
\sigma_c(z_{k,m}(u))+\epsilon
}.
$$

这里的均值和标准差都只沿类别维计算，不使用数据集标签。

### 5.4 不变 anchor 与 shortcut residual

定义客户端的跨视图不变 anchor：

$$
a_k(u)
=
\frac{1}{M}
\sum_{m=1}^{M}
\widehat z_{k,m}(u).
$$

定义环境残差：

$$
r_{k,m}(u)
=
\widehat z_{k,m}(u)-a_k(u).
$$

直觉上：

```text
a_k(u)：该客户端在不同 corruption intervention 下反复出现的响应；
r_km(u)：只在某个环境视图中出现的变化，更可能包含 context shortcut。
```

客户端只上传 `a_k(u)`，不上传 `M` 份完整 logits。因此通信量仍为：

$$
O(B_{pub}C),
$$

与普通 public-logit 通信同阶，而不是 `M` 倍。

### 5.5 Leave-one-out 鲁棒教师

为接收客户端 `i` 构造教师时排除自身：

$$
a_T^{-i}(u)
=
\operatorname{Median}_{k\neq i}
a_k(u).
$$

首版使用类别坐标上的中位数聚合，而不是使用测试准确率进行客户端排序。

教师分布为：

$$
q_T^{-i}(u)
=
\operatorname{softmax}
\left(
\frac{a_T^{-i}(u)}{T}
\right).
$$

中位数聚合的目的：

```text
压低单个 shortcut 客户端的离群响应；
不上传 private class count；
不引入测试标签泄漏；
不依赖所有模型具有相同 feature dimension。
```

### 5.6 最坏公共视图蒸馏

接收客户端 `i` 对每个公共视图产生：

$$
p_{i,m}(u)
=
\operatorname{softmax}
\left(
\frac{\widehat z_{i,m}(u)}{T}
\right).
$$

逐视图蒸馏风险：

$$
D_{i,m}(u)
=
\operatorname{KL}
\left(
q_T^{-i}(u)\,\|\,p_{i,m}(u)
\right).
$$

IRD 同样优化平滑最坏视图：

$$
\mathcal L_{\mathrm{IRD}}^i
=
\frac{1}{|B_{pub}|}
\sum_u
\tau_d
\log\left[
\sum_m
\exp\left(
\frac{D_{i,m}(u)}{\tau_d}
\right)
\right].
$$

这样不是只让客户端的“平均公共预测”靠近教师，而是优先修正与不变教师差距最大的环境视图。

---

## 6. 完整优化目标与训练时序

客户端 `k` 的完整目标为：

$$
\boxed{
\mathcal L_k
=
\mathcal L_{\mathrm{CE}}
+
\lambda_{js}\mathcal L_{\mathrm{JSD}}
+
\lambda_{cr}\mathcal L_{\mathrm{CCRE}}^k
+
\lambda_{kd}(t)\mathcal L_{\mathrm{IRD}}^k
}
$$

通信权重采用 warmup schedule：

$$
\lambda_{kd}(t)=0,\qquad t<t_w,
$$

$$
\lambda_{kd}(t)>0,\qquad t\ge t_w.
$$

首版建议：

```text
private views M = 3；
public views M = 3；
warmup rounds = 3；
不增加新的 projection head；
不读取 train_corruption_ids；
不上传 class count；
不使用测试集路由；
checkpoint 只保留 latest/best，避免磁盘膨胀。
```

需要调节的核心超参数只有：

```text
lambda_cr：CCRE 强度；
lambda_kd：IRD 强度；
tau_r / tau_d：平滑最大值温度；
T：蒸馏温度。
```

为避免参数过多，首版可以令 `tau_r=tau_d`，并沿用 RAHFL 的蒸馏温度 `T`。

---

## 7. 理论解释

### 7.1 本地目标对应条件分布鲁棒优化

对类别 `c`，理想目标是：

$$
\min_f
\max_{q(g)\in\mathcal Q}
\mathbb E_{g\sim q(g)}
\mathbb E_{x\sim P_k(x\mid y=c,g)}
\ell(f(x),c).
$$

CCRE 使用标签无关干预构造有限个反事实环境，并以 log-sum-exp 近似环境上的最大风险。再对类别等权平均，使优化目标更接近 WCCA，而不是普通样本平均准确率。

这不是严格证明模型获得了因果表示；更准确的表述是：

```text
在干预保持标签语义、且覆盖主要 nuisance variation 的条件下，
降低每类跨干预的最坏风险，会降低模型利用单一 corruption-label shortcut 的收益。
```

### 7.2 通信目标对应响应分解

假设公共 probe 上的标准化 logits 可以近似分解为：

$$
\widehat z_{k,m}(u)
=
s(u)+b_k(u)+r_{k,m}(u)+\varepsilon,
$$

其中：

```text
s(u)：跨客户端共享的语义响应；
b_k(u)：客户端或模型特有的偏置；
r_km(u)：环境干预导致的 shortcut residual；
epsilon：随机误差。
```

跨视图平均会压低均值接近零的 `r_km`，跨客户端中位数进一步抑制离群 `b_k`。IRD 再让客户端的最差环境视图逼近聚合 anchor，从而同时抑制本地残差和通信中的 shortcut 传播。

这里必须保留一个理论边界：`a_k(u)` 只是候选不变响应，不自动等于真实语义。如果原始 corruption shortcut 在所有在线视图中都没有被改变，它会作为稳定偏差残留在 anchor 中。FedCLEAR 依靠标签无关干预覆盖、不同客户端 shortcut 映射差异和中位数聚合共同削弱它，而不是声称一次平均就能严格完成因果分解。

### 7.3 三个必要假设

FedCLEAR 的理论故事依赖以下可检验假设：

```text
A1. 干预算子大体保持类别语义；
A2. intervention bank 覆盖了与部署 corruption 相关的主要 nuisance direction；
A3. 不同客户端的 shortcut 不完全同向，因此鲁棒聚合可以保留共享响应并削弱个体偏置。
```

论文中不能写成“必然恢复真实因果特征”，而应写成“通过反事实风险和跨客户端不变响应，降低 corruption-label shortcut 的可利用性”。

### 7.4 与相关理论的关系

FedCLEAR 的理论来源可以清楚归位：

```text
数据增强作为环境干预：为反事实视图提供因果解释；
Group DRO / worst-group learning：为优化最差 class-context 风险提供依据；
Risk Extrapolation：为跨环境风险一致与 OOD 泛化提供依据；
federated invariant learning：说明联邦数据差异可以用于识别共享稳定关系。
```

但论文不能宣称首次研究“联邦学习中的虚假相关”。准确创新边界应是：

```text
首次或较早系统研究 corruption-label entanglement
在 model-heterogeneous federated learning 中的形成、传播和评价，
并提出同时作用于本地风险与异构通信响应的对应方法。
```

正式投稿前必须继续做专门文献排重。

---

## 8. 一次真实训练轮的完整流程

下面以第 5 轮为例，此时三轮 warmup 已结束。

### Step 1：读取客户端私有 batch

客户端 0 使用 ResNet10，取 64 张本地样本。由于 `alpha=0.5, gamma=0.9`：

```text
类别数量严重不均衡；
cat 样本大多带 noise；
dog 样本大多带 blur；
其他客户端的 class-corruption 映射不同。
```

### Step 2：生成三个标签无关反事实视图

对一张原本带 gaussian noise 的 cat 图像，在线产生：

```text
view 1：基础/轻增强视图；
view 2：一次随机模糊或空间干预；
view 3：一次随机数字或颜色干预。
```

实际训练不需要知道这些操作属于哪个 family，只记录视图编号用于计算环境风险。

### Step 3：一次前向得到三份 logits

假设 cat 的三个 CE 为：

```text
view 1: 0.22
view 2: 0.91
view 3: 1.86
```

dog 的三个 CE 为：

```text
view 1: 0.35
view 2: 0.43
view 3: 0.51
```

### Step 4：按类别计算 CCRE

cat 的平滑最坏风险主要由 `1.86` 决定；dog 的风险较均匀。

即使 batch 中 dog 样本比 cat 多很多，最终仍然先分别得到：

```text
R_cat
R_dog
...
```

再对本地出现类别等权平均。cat 不会因为样本少而在损失中被忽略。

### Step 5：完成本地更新

计算：

```text
CE + JSD + CCRE
```

执行反向传播。CCRE 产生的最大梯度指向 cat 的最差反事实视图，迫使模型利用在多个环境中都成立的猫语义，而不是只利用 noise shortcut。

### Step 6：客户端处理公共 batch

服务器提供 128 张公共 CIFAR-100 图片及本轮固定 augmentation seed。

每个客户端对同一张公共图片 `u` 生成三个视图并输出标准化 logits。例如某些类别坐标简化为：

```text
client 0 anchors: [ 1.20, -0.20, -0.40, ...]
client 1 anchors: [ 0.85, -0.10, -0.35, ...]
client 2 anchors: [ 1.05, -0.30, -0.25, ...]
client 3 anchors: [-0.50,  1.30, -0.10, ...]
```

client 3 在该 probe 上明显离群。服务器对接收客户端构造 leave-one-out 中位数教师，离群值不会像普通平均那样强烈改变 teacher。

### Step 7：下发不变教师

服务器只下发每张公共图片的一份 `C` 维 teacher anchor。它并不声称公共图像属于 CIFAR-10 的某个类别，而是把公共图像作为统一输入坐标，传递模型间共享的决策响应。

### Step 8：执行最差公共视图蒸馏

客户端 0 对某个 teacher 类别坐标的三个视图置信度可能为：

```text
view 1: 0.64
view 2: 0.57
view 3: 0.24
teacher: 0.61
```

普通平均蒸馏可能被前两个较好视图掩盖；IRD 的 log-sum-exp 主要修正 `view 3`，使该客户端在环境变化下仍保持接近全局共享响应。

### Step 9：评价和保存

本轮结束后记录：

```text
avg_acc / worst_acc；
WCCA / CFG；
每类反事实最坏风险；
public residual norm；
IRD worst-view KL；
non-finite batch 数量；
本地和通信阶段耗时。
```

这些诊断可以区分：

```text
本地仍在学习 shortcut；
教师 anchor 本身混乱；
教师有效但学生没有吸收；
方法提高 WCCA 却损害平均准确率。
```

### 8.1 训练伪代码

```text
Input:
  heterogeneous models {f_k}
  private loaders {D_k}
  unlabeled public loader D_pub
  label-independent intervention bank Q_A

for round t = 0 ... R-1:
    # Private phase
    for client k:
        for private batch (x, y):
            sample M interventions from Q_A
            build M semantic-preserving views
            forward all views once
            compute CE and JSD
            compute class-view risk matrix R[k, c, m]
            compute CCRE smooth worst-context risk
            update all local model parameters

    # Communication phase
    if t >= warmup_rounds:
        server samples public batch u and shared intervention seeds

        for client k:
            build the same M public views
            compute normalized logits
            average views into invariant anchor a_k(u)
            upload only a_k(u)

        server builds leave-one-out median teacher for each receiver i

        for client i:
            regenerate the M public views
            compute per-view KL to teacher
            optimize IRD smooth worst-view KL

    evaluate without using evaluation labels for routing
    write heartbeat, metrics and bounded checkpoints
```

---

## 9. 指标与方法目标的对应关系

### 9.1 WCCA

$$
\operatorname{WCCA}
=
\min_{c,g}\operatorname{Acc}(c,g).
$$

CCRE 直接优化每个类别的最坏反事实环境，因此它主要应该提高 WCCA。

### 9.2 CFG

$$
\operatorname{CFG}
=
\frac{1}{C}
\sum_c
\left[
\max_g\operatorname{Acc}(c,g)
-
\min_g\operatorname{Acc}(c,g)
\right].
$$

CCRE 和 IRD 都在压低跨环境最差风险，因此应降低 CFG。

### 9.3 Average / Worst Client Accuracy

WCCA 和 CFG 改善不能以整体分类性能崩溃为代价，所以仍要求：

```text
avg_acc 不低于 RAHFL；
worst_acc 不低于 RAHFL；
或者至少在 gamma=0.9 下显著恢复 RAHFL 的退化。
```

---

## 10. 实验设计与成功标准

### 10.1 第一阶段：只验证完整方法

首个正式运行：

```text
alpha=0.5
gamma=0.9
seed=0
4 heterogeneous clients
与 RAHFL 完全相同的本地 epoch、通信轮、public batch 和 batch size
```

已有 RAHFL 参照：

```text
avg_acc=46.72
worst_acc=38.16
WCCA=19.32
CFG=10.91
```

第一阶段正面信号：

```text
WCCA 至少提升 4~5 个点；
CFG 至少下降 2~3 个点；
avg_acc 达到约 49% 或更高；
worst_acc 达到约 40% 或更高；
全程无 NaN。
```

这些不是论文录用门槛，而是判断方法是否真的击中 CLE failure mode 的工程门槛。

### 10.2 第二阶段：普通场景不得明显退化

如果 gamma 0.9 有正面结果，再运行：

```text
gamma=0.6：验证中等纠缠；
gamma=0.0：验证没有 shortcut 时不会明显伤害普通 robustness。
```

### 10.3 第三阶段：论文级验证

最终需要：

```text
seed 0/1/2；
alpha 1.0/0.5/0.3；
gamma 0.0/0.6/0.9；
官方 CIFAR-C corruption；
seen method / unseen method；
seen severity / unseen severity；
同一 counterfactual test set 哈希；
通信量和计算量对比。
```

### 10.4 必需基线

```text
RAHFL；
AugMix + JSD + DCL local-only；
SARA + AsymHFL；
FedCLEAR local-only（只开 CCRE）；
FedCLEAR full（CCRE + IRD）；
LogitAvg 或普通 public-logit distillation；
使用真实 corruption group 的 GroupDRO 作为 oracle/诊断上界。
```

Oracle GroupDRO 不作为公平主方法，而用于回答：如果直接知道真实 group，WCCA 最多能恢复多少。

---

## 11. 形式化协议需要补强的地方

### 11.1 当前 corruption 是轻量近似实现

现有 `fedprime/data/corruptions.py` 明确用于生成可复现的轻量协议，并非精确复现 CIFAR-C 内部实现。

因此：

```text
当前结果可以支持“问题值得继续”；
正式论文不能只依赖当前轻量 corruption 得出普适结论。
```

正式版本应切换到官方 corruption 实现或经过验证的标准实现。

### 11.2 固定测试集

不同 gamma 和不同方法必须读取完全相同的 counterfactual test 文件，并保存 SHA256。不能依赖“相同 seed 大概率生成相同测试集”。

### 11.3 未见损坏测试

建议每个 family 至少留出一个训练未见算子，或者增加官方 CIFAR-C 中当前未覆盖的 corruption。算法训练时不读取 family 名称，才能证明方法不是针对四组规则写死。

### 11.4 方法不读取 corruption ID

主方法默认不读取：

```text
train_corruption_ids.npy
train_corruption_method_ids.npy
```

这些只用于协议审计、指标统计和 oracle baseline。这样可以避免“方法依赖人工 corruption 标签”的攻击。

---

## 12. 与现有工作的区别和创新边界

### 12.1 不能宣称的内容

```text
不能宣称首次研究 federated spurious correlation；
不能宣称数据增强必然等价于真实 causal intervention；
不能宣称四个 corruption family 覆盖现实中的全部损坏；
不能宣称 public logits 能凭空传递客户端完全缺失的类别知识；
不能在单 seed、单数据集结果上声称全面超过 RAHFL。
```

### 12.2 可以形成的贡献链

```text
贡献 1：提出 CLE-HFL 问题，刻画模型异构 FL 中 corruption 与 label 的客户端特有纠缠；
贡献 2：提出可控 gamma 协议和 WCCA/CFG 反事实指标，揭示 RAHFL 的细粒度失败；
贡献 3：提出 CCRE，从类别条件的最坏反事实风险层面阻止 shortcut 形成；
贡献 4：提出 IRD，在 output-space 中分离不变 anchor 和环境 residual，阻止 shortcut 跨异构模型传播；
贡献 5：在 seen/unseen corruption、不同 alpha/gamma 和多 seed 下验证。
```

方法层面的关键区别不是“再加一个稳定性权重”，而是：

```text
训练目标从样本平均风险改成类别条件最坏环境风险；
通信对象从单视图/客户端整体 logits 改成跨干预不变响应；
蒸馏目标从平均视图对齐改成最坏视图对齐。
```

---

## 13. 主要风险与止损条件

### 13.1 Intervention 不保持语义

过强 corruption 会破坏类别本身，此时最坏风险训练会强迫模型拟合不可识别样本。需要限制在线干预 severity，并记录每种算子的语义保持检查。

### 13.2 跨域 public data 响应无意义

CIFAR-100 public images 与 CIFAR-10 标签空间不同。如果所有客户端 anchor 都高度混乱，中位数教师也不会自动产生知识。需要记录：

```text
跨客户端 anchor disagreement；
teacher entropy；
IRD 前后的 worst-view KL。
```

若这些指标表明 teacher 无信息，则应更换为同域无标签 public subset 或 data-free carrier，而不是继续调 IRD 权重。

### 13.3 稳定 shortcut 残留在 anchor

如果在线 intervention 只是在原图上叠加轻微变化，而原始 corruption cue 在所有视图中始终存在，那么它不会进入零均值 residual，而会残留在 anchor。应通过以下实验检查：

```text
增加 held-out intervention 后 anchor 是否仍保持一致；
不同客户端 anchor 的共享部分是否对应更低 WCCA gap；
去掉跨客户端鲁棒聚合后性能是否明显下降。
```

如果 anchor 对未见 corruption 仍高度敏感，则 IRD 的分解假设不成立。

### 13.4 所有客户端形成同向 shortcut

如果所有客户端对同一类别都绑定同一 corruption，跨客户端聚合不能自动抵消该 shortcut。CLE-HFL 当前使用 client-specific mapping，研究对象是不同客户端偏差互补的场景。

### 13.5 止损条件

在 `gamma=0.9, seed=0` 的完整运行中，如果：

```text
WCCA 提升小于 2 点；
CFG 几乎不下降；
avg_acc 继续低于 RAHFL；
public anchor disagreement 始终很高；
```

则不能仅通过调参宣称方向有效，应重新判断 public carrier 是否适合这条通信路线。

---

## 14. 当前代码落点

已经新增或修改：

```text
fedprime/methods/ccre.py
  - 类别-视图风险矩阵
  - smooth worst-context loss

fedprime/methods/ird.py
  - logit 标准化
  - invariant anchor
  - leave-one-out median teacher
  - worst-view distillation

fedprime/methods/local_fedclear.py
  - 标签无关反事实视图上的 CE、JSD 与 CCRE 本地更新

fedprime/augmentations/counterfactual.py
  - 可扩展的标签无关 counterfactual operator bank

fedprime/methods/fedclear.py
fedprime/methods/rahfl_asymhfl.py
  - FedCLEAR 统一 runner 编排
  - warmup、日志、WCCA/CFG、checkpoint

configs/debug_fedclear_cle_gamma09.yaml
configs/openi_v100_fedclear_cle_gamma09_probe.yaml
configs/openi_v100_fedclear_cle_gamma09_full.yaml

scripts/openi_fedclear_entry.py
  - 启智非交互式入口
  - heartbeat
  - c2net prepare/output 回传
```

所有新模块必须可配置关闭，以支持：

```text
CCRE local-only；
IRD-only communication control；
FedCLEAR full；
RAHFL unchanged baseline。
```

---

## 15. 参考理论来源

1. Hendrycks et al., AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty, ICLR 2020: <https://openreview.net/pdf?id=S1gmrxHFvB>
2. Ilse et al., Selecting Data Augmentation for Simulating Interventions, ICML 2021: <https://proceedings.mlr.press/v139/ilse21a.html>
3. Krueger et al., Out-of-Distribution Generalization via Risk Extrapolation, ICML 2021: <https://proceedings.mlr.press/v139/krueger21a.html>
4. Tang et al., Causally Motivated Personalized Federated Invariant Learning with Shortcut-Averse Information-Theoretic Regularization, ICML 2024: <https://icml.cc/virtual/2024/poster/34335>
5. Ma et al., Reducing Spurious Correlation for Federated Domain Generalization, 2024: <https://arxiv.org/abs/2407.19174>

这些工作提供相关理论背景，但不能替代后续对 CCRE、IRD 和 CLE-HFL 的专门文献排重。

---

## 16. 审核清单

在进入编码前，需要确认以下问题：

```text
[x] FedCLEAR = CCRE + IRD 的核心结构已实现。
[x] 主方法不使用 corruption ID，只把它用于评估。
[x] 使用显式 counterfactual views + JSD，不调用 DCL。
[x] public data 只作为响应 probe，而非 prior estimator。
[x] 首发只跑 alpha=0.5, gamma=0.9, seed=0。
[x] WCCA/CFG 是首轮机制验证的优先指标。
[ ] 是否需要把 IRD 的中位数聚合改成其他聚合形式？
```

当前下一步：

```text
1. 在 OpenI 运行 12 轮 probe；
2. 检查 WCCA、CFG、avg_acc、worst_acc 和 IRD 诊断；
3. probe 为正时再运行 40 轮 full；
4. 根据完整结果决定是否调整 IRD 聚合。
```
