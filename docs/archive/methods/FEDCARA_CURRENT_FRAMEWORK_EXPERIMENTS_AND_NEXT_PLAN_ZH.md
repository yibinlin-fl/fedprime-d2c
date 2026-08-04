# FedCARA 当前框架、实验结果与下一步改进方案咨询稿

更新时间：2026-07-01

本文档用于向其他 AI、导师或同学咨询当前研究方案。内容包括：当前任务目标、现有框架设计、已经完成的主要实验结果、当前问题判断，以及下一步准备改进的通信模块方案。

## 1. 研究目标与实验场景

本项目目标是在以下三种困难同时存在的联邦学习场景下，构造一个能够正面对比并尽量超过 RAHFL 的方法：

```text
1. 模型异构：不同客户端使用不同网络结构。
2. 数据异构：客户端私有数据服从 Dirichlet label-skew Non-IID 划分。
3. 数据损坏：训练与测试图像均使用 RAHFL-style 随机损坏 CIFAR-10。
```

当前最重要的基线是：

```text
RAHFL = AugMix + JSD + DCL + AsymHFL
```

当前主线方法经历了多次迭代，最新有效方向是：

```text
FedCARA = AugMix/JSD + CARA-L + CARA-C
```

其中：

```text
CARA-L：Non-IID-aware robust local contrastive learning，本地鲁棒对比学习模块。
CARA-C：Class-aware reliable communication，类别自适应可靠通信模块。
```

## 2. 当前统一实验设置

目前主要实验均使用相同的轻量 Kaggle T4 统一 runner 设置：

```text
dataset private train: RAHFL-style corrupted CIFAR-10 train
dataset test:          RAHFL-style corrupted CIFAR-10 test
public dataset:        CIFAR-100
clients:               4
models:                ResNet10, ResNet12, ShuffleNet, MobileNetV2
partition:             Dirichlet alpha=0.5
private samples/client: 10000
private corrupt rate:  1
test corrupt rate:     1
rounds:                40
local epochs/round:    1
batch size:            64
public batch size:     128
public batches/round:  4
pretraining:           no independent 40-epoch local pretraining
```

固定划分文件：

```text
outputs/partitions/cifar10c_alpha05_seed0_clients4_samples10000.npz
```

这个设置不是 RAHFL 论文的完整复现。RAHFL 原文更强：通常包含 40 epoch 本地预训练、更大的 batch size 和更完整的 public communication budget。当前结果主要用于同一代码、同一数据、同一预算下的公平比较。

## 3. 当前数据划分情况

当前 alpha=0.5 的固定划分已经明显 Non-IID。典型客户端类别分布如下：

```text
client 0: class_0=3420, class_1=254,  class_2=851,  class_3=1832,
          class_4=526,  class_5=229,  class_6=65,   class_7=1880,
          class_8=564,  class_9=379

client 1: class_0=48,   class_1=743,  class_2=264,  class_3=261,
          class_4=551,  class_5=256,  class_6=2081, class_7=310,
          class_8=1629, class_9=3857

client 2: class_0=21,   class_1=2643, class_2=1783, class_3=45,
          class_4=274,  class_5=3038, class_6=344,  class_7=1852,
          class_8=0,    class_9=0

client 3: class_0=1050, class_1=26,   class_2=978,  class_3=2099,
          class_4=2542, class_5=5,    class_6=1456, class_7=17,
          class_8=1827, class_9=0
```

可以看到 client 2 完全缺失 class 8/9，client 3 完全缺失 class 9。这也是后续判断 missing/tail 知识迁移的重要依据。

## 4. 当前已完成实验总表

| 方法 | 本地模块 | 通信模块 | final avg_acc | final worst_acc | best avg_acc | best worst_acc | 当前判断 |
|---|---|---|---:|---:|---:|---:|---|
| RAHFL | AugMix + JSD + DCL | AsymHFL | 56.41 | 44.72 | 56.41 | 44.72 | 当前统一 runner 强基线 |
| AugMix+DCL local-only | AugMix + JSD + DCL | 无通信 | 56.11 | 44.23 | 56.94 | 44.23 | 说明本地鲁棒学习贡献很大 |
| PRIME + LogitAvg | PRIME + JSD | LogitAvg | 52.10 | 39.72 | 52.19 | 39.98 | PRIME+普通 public logits 不足 |
| FedPRIME-D2C | PRIME + JSD | D2C | 52.31 | 39.78 | 52.83 | 39.78 左右 | D2C 仅略高于 LogitAvg |
| Oracle D2C | PRIME + JSD | 使用真实 prior 的 D2C | 51.74 | 39.13 | 52.65 | 39.89 | 真实 prior 也未改善 D2C |
| FedPRIME-PAIR | PRIME + CBCL | CPAD pairwise KD | 约 50.15 | 约 39-40 | 约 51.10 | - | pairwise public-logit 路线失败 |
| PRAC-HFL public1 | AugMix + JSD + DCL | PRAC receiver-safe KD | 54.63 | 41.88 | 55.53 | 43.43 | 稳定但不如 RAHFL |
| PRAC-HFL public4 | AugMix + JSD + DCL | PRAC receiver-safe KD | 52.96 | 43.27 | 52.96 | 43.27 | public4 下平均精度下降 |
| CARA-L local-only | AugMix + JSD + CARA-L | 无通信 | 53.30 | 36.01 | 54.74 | 37.37 | CARA-L 单独本地训练不成立 |
| CARA-L + AsymHFL | AugMix + JSD + CARA-L | 原始 AsymHFL | 57.36 | 46.23 | 57.89 | 46.33 | 当前最强结果 |
| FedCARA v1 | AugMix + JSD + CARA-L | CARA-C | 55.88 | 45.93 | 56.86 | 45.93 | worst_acc 超 RAHFL，但 avg_acc 未超 |

## 5. 对已完成实验的核心判断

### 5.1 RAHFL 的优势主要来自强本地学习

`AugMix + DCL local-only` 已经达到：

```text
final avg_acc   = 56.11
final worst_acc = 44.23
```

而 RAHFL 是：

```text
final avg_acc   = 56.41
final worst_acc = 44.72
```

这说明当前统一 runner 下，RAHFL 的大部分性能来自：

```text
AugMix + JSD + DCL 本地鲁棒训练
```

AsymHFL 通信确实有贡献，但在当前设置下不是唯一主要来源。

### 5.2 D2C 路线没有成立

D2C 目标原本是通过估计客户端 prior、做 prior debias、class-balanced aggregation 和 complementary KD 来解决 Non-IID public logits 偏差。

但实验显示：

```text
PRIME + LogitAvg: final avg_acc = 52.10
FedPRIME-D2C:     final avg_acc = 52.31
Oracle D2C:       final avg_acc = 51.74
```

这说明：

```text
1. D2C 只比普通 LogitAvg 高约 0.21 个点，基本没有实质提升。
2. 即使用真实 private prior，Oracle D2C 也没有提升。
3. 问题不只是 predicted prior 估计不准，而是 D2C 的 prior debias/complementary KD 机制本身不适合当前跨域 public CIFAR-100 logits。
```

同时，RAHFL 和 D2C 在 missing class 上都没有真正解决完全缺失类迁移：

```text
RAHFL:
  client 2 missing_acc = 0.00
  client 3 missing_acc = 0.00

Oracle D2C:
  client 2 missing_acc = 0.00
  client 3 missing_acc = 0.00
```

因此，“用跨域 CIFAR-100 public logits 补全完全缺失 CIFAR-10 类别”这个目标在当前框架下没有被验证成功。

### 5.3 CARA-L + AsymHFL 是当前最强结果，但创新性不足

CARA-L + AsymHFL 达到：

```text
final avg_acc   = 57.36
final worst_acc = 46.23
```

相对 RAHFL：

```text
avg_acc   +0.95
worst_acc +1.51
```

这个结果是目前最好的，但它仍然使用原始 AsymHFL 通信，因此如果只把论文写成：

```text
改进 DCL + 继续使用 AsymHFL
```

创新性会比较弱。

### 5.4 FedCARA v1 的问题

FedCARA v1 使用：

```text
CARA-L + CARA-C
```

结果：

```text
FedCARA v1 final avg_acc   = 55.88
FedCARA v1 final worst_acc = 45.93
```

相对 RAHFL：

```text
avg_acc   -0.53
worst_acc +1.21
```

相对 CARA-L + AsymHFL：

```text
avg_acc   -1.48
worst_acc -0.30
```

这说明 CARA-C v1 有一定公平性收益，能提升 worst-client，但平均精度不足。最近画出的曲线显示 FedCARA 和 CARA-L + AsymHFL 的趋势几乎一样，这说明：

```text
1. 本地 CARA-L / AugMix / JSD 是主要驱动力。
2. CARA-C v1 只是对相同 public logits 做类别重加权，没有引入足够新的学习信号。
3. 当前 CARA-C v1 太像 AsymHFL 的 reweight 版本，创新性和性能都不够强。
```

## 6. 当前框架设计：CARA-L

CARA-L 是本地 Non-IID-aware robust contrastive learning 模块，替代原始 DCL。

每个客户端本地输入图像 `x` 经过 AugMix 得到 clean/weak/strong 多视图：

```text
x^0 = clean view
x^1 = weak/aug view
x^2 = strong/aug view
```

模型输出：

```text
z_i^v = logits
h_i^v = embedding feature
```

本地总损失为：

```text
L_local = L_CE + lambda_jsd * L_JSD + lambda_cara * L_CARA-L
```

其中：

```text
L_CE = CE(f(x^0), y)
```

JSD 保持多视图预测一致：

```text
p_m = (p^0 + p^1 + p^2) / 3

L_JSD =
  KL(p_m || p^0) +
  KL(p_m || p^1) +
  KL(p_m || p^2)
```

CARA-L 的主要改进包括：

```text
1. class-balanced DCL：
   先对每个类别内部平均，再对 batch 内出现的类别平均，避免 head class 支配对比损失。

2. client-local feature queue：
   每个客户端维护本地类别特征队列，缓解 Non-IID batch 中 tail class 正样本不足的问题。

3. strong-view reliability gate：
   如果强增强视图已经破坏语义，则降低该视图在对比对齐中的权重。

4. stable relation alignment：
   使用更稳定的 softmax(sim / T) 关系对齐，而不是数值更激进的 exp(sim)/T 形式。
```

一个抽象形式为：

```text
L_CARA-L =
  1 / |C_B| * sum_{c in C_B}
  1 / |I_c| * sum_{i in I_c}
  r_i * ell_i
```

其中：

```text
C_B：当前 batch 中出现的类别集合
I_c：batch 中属于类别 c 的样本集合
r_i：增强视图可靠性权重
ell_i：样本 i 的对比/关系对齐损失
```

增强视图可靠性可以用 true-class margin 表示：

```text
r_i = sigmoid((z_{i,y_i}^{strong} - max_{c != y_i} z_{i,c}^{strong}) / tau_m)
```

直觉：

```text
如果 strong view 仍然能正确支持真实类别，则说明增强没有破坏语义，r_i 较大。
如果 strong view 已经让模型明显偏向错误类别，则该视图可能是有害增强，r_i 较小。
```

## 7. 当前框架设计：CARA-C v1

CARA-C v1 试图把 RAHFL 的客户端级通信改成类别级通信。

RAHFL / AsymHFL 的思想是：

```text
如果 teacher 客户端整体准确率高于 student，
则 student 在 public data 上学习 teacher 的完整 softmax 分布。
```

问题是：

```text
Non-IID 下，一个客户端整体准确率高，不代表它每个类别都强。
一个客户端整体弱，也可能在少数类别上是专家。
```

CARA-C v1 因此对 receiver i、teacher j、class c 定义：

```text
w_{i,j,c} = acc_{j,c} * (1 - acc_{i,c})
```

其中：

```text
acc_{j,c}：teacher j 在类别 c 上的可靠性。
1 - acc_{i,c}：receiver i 对类别 c 的需求程度。
```

还加入安全门：

```text
only use class c if acc_{j,c} > acc_{i,c} + margin
```

在 public data 上，teacher/student 概率为：

```text
p_j(u) = softmax(z_j(u) / T)
p_i(u) = softmax(z_i(u) / T)
```

class-weighted KL 为：

```text
L_CARA-C^{i,j}
= sum_{u in D_pub} sum_c
  w_{i,j,c} * p_{j,c}(u) *
  log(p_{j,c}(u) / p_{i,c}(u))
```

这个设计能让弱类别获得更多通信关注，因此 worst_acc 有提升；但它直接替换了原 AsymHFL 的 full-distribution KD，导致平均精度下降。

## 8. 当前最大问题

当前最大问题不是代码能不能跑，而是研究定位：

```text
1. 如果只用 CARA-L + AsymHFL，结果最好，但创新性偏弱。
2. 如果用 FedCARA v1 完全替换 AsymHFL，创新性稍强，但 avg_acc 没有超过 RAHFL。
3. 如果做 Hybrid：L = AsymHFL + lambda * CARA-C，性能可能变好，但创新性会更像 RAHFL 补丁。
```

因此下一步不应只做 AsymHFL hybrid，而应该设计一个更独立的通信模块。

## 9. 准备进行的改进：类别专家拼接式通信

下一步建议将 FedCARA 的通信部分从“teacher 客户端选择”改为“类别专家拼接”。

暂定名称：

```text
Class-Expert Mosaic Distillation, CEMD
```

中文可称为：

```text
类别专家拼接式蒸馏
```

新的主框架：

```text
FedCARA-v2 = CARA-L + CEMD
```

核心动机：

```text
RAHFL/AsymHFL 以客户端为最小知识单元。
但在 Non-IID 下，客户端通常是偏科的。
一个客户端整体强，不代表每个类别都强。
一个客户端整体弱，也可能在某些类别上是专家。

因此通信不应选择“哪个客户端整体当老师”，而应选择“每个类别由哪个客户端教”。
```

## 10. CEMD 具体公式设计

### 10.1 客户端类别可靠性

对客户端 k、类别 c，定义类别专家可靠性：

```text
R_{k,c} = Acc_{k,c}^{val} * sqrt(n_{k,c} / (n_{k,c} + eta))
```

其中：

```text
Acc_{k,c}^{val}：客户端 k 在本地 held-out validation split 上对类别 c 的准确率。
n_{k,c}：客户端 k 本地类别 c 的训练样本数。
eta：平滑系数，避免极少样本类别被过度相信。
```

注意：这里应使用从训练集划出的 held-out validation，而不是最终 test set，以避免测试集泄漏。

### 10.2 public logits 尺度校准

由于模型异构，不同客户端 logit 尺度可能不同，因此需要做简单校准：

```text
bar_z_k(u) = (z_k(u) - mean_c z_{k,c}(u)) / (std_c z_{k,c}(u) + epsilon)
```

或使用温度校准：

```text
p_k(u) = softmax(z_k(u) / T_k)
```

第一版建议先使用 per-sample logit standardization，减少 ResNet/MobileNet/ShuffleNet 输出尺度差异。

### 10.3 按类别选择专家

对 receiver i，类别 c 的候选教师为所有其他客户端：

```text
k != i
```

专家权重为：

```text
alpha_{i,k,c}
= softmax_{k != i}(R_{k,c} / tau_r)
```

如果要更安全，可以加入 only-better gate：

```text
alpha_{i,k,c} = 0, if R_{k,c} <= R_{i,c} + margin
```

### 10.4 拼接 mosaic teacher logits

对 public sample u，receiver i 的 mosaic teacher 第 c 类 logit 为：

```text
z_{i,c}^{mosaic}(u)
= sum_{k != i} alpha_{i,k,c} * bar_z_{k,c}(u)
```

也就是说：

```text
airplane logit 可能来自 client 0
car logit      可能来自 client 1
cat logit      可能来自 client 2
horse logit    可能来自 client 3
```

最终拼成一个新的 teacher：

```text
q_i(u) = softmax(z_i^{mosaic}(u) / T)
```

这个 teacher 不是任何一个客户端的完整输出，而是由全网类别专家拼接出来的“组合教师”。

### 10.5 receiver demand gate

客户端 i 只在自己弱的类别上强烈学习：

```text
g_{i,c} = max(0, max_{k != i} R_{k,c} - R_{i,c})
```

归一化后得到：

```text
tilde_g_{i,c} = g_{i,c} / (mean_c g_{i,c} + epsilon)
```

如果别人确实比我更懂类别 c，则该类别蒸馏权重大；如果我自己已经很强，则该类别蒸馏权重小。

### 10.6 CEMD 通信损失

客户端 i 在 public data 上的通信损失为：

```text
L_CEMD^i
= sum_{u in D_pub} sum_c
   tilde_g_{i,c} * q_{i,c}(u)
   * log(q_{i,c}(u) / p_{i,c}(u))
```

其中：

```text
p_i(u) = softmax(z_i(u) / T)
```

整体训练目标：

```text
L_i = L_CE + lambda_jsd * L_JSD
      + lambda_cara * L_CARA-L
      + lambda_cemd * L_CEMD
```

## 11. CEMD 与 RAHFL/AsymHFL 的区别

| 维度 | RAHFL / AsymHFL | FedCARA-v2 / CEMD |
|---|---|---|
| 教师粒度 | 客户端级 | 类别专家级 |
| teacher 来源 | 某个整体更强客户端 | 每个类别由不同专家客户端拼接 |
| Non-IID 适配 | 依赖整体准确率 | 显式使用 per-class reliability |
| 模型异构 | 通过 public logits 支持 | 通过 public logits 支持 |
| 数据损坏 | AugMix + JSD + DCL | AugMix + JSD + CARA-L |
| 风险 | 粗粒度 teacher 可能只强在 head class | 类别级拼接可能存在 logit 校准问题 |

核心创新表述：

```text
RAHFL asks: which client should teach?
FedCARA-v2 asks: which client should teach each class?
```

## 12. CEMD 的潜在优势

该方案理论上更符合当前场景：

```text
1. 模型异构：
   不传参数和特征，只传 public logits，因此不要求模型结构或 embedding 维度一致。

2. 数据异构：
   不再用整体客户端准确率判断教师，而是用 per-class reliability，
   更适合 label-skew Non-IID。

3. 数据损坏：
   本地仍使用 AugMix/JSD/CARA-L 学习鲁棒表示；
   类别可靠性也可以从增强/损坏验证视图上估计。

4. 创新性：
   不再是 AsymHFL 的简单加权补丁，而是构造了一个 class-wise mosaic teacher。
```

## 13. CEMD 的主要风险

需要重点请其他 AI 或导师评估以下风险：

```text
1. 不同类别 logit 来自不同客户端，拼接成一个 softmax teacher 是否理论上合理？
2. 异构模型 logit 尺度不同，per-sample standardization 是否足够？
3. per-class validation accuracy 在 tail class 样本少时是否可靠？
4. 如果 public CIFAR-100 与 private CIFAR-10 语义域不同，类别专家拼接是否仍然有效？
5. 是否需要同域少量 public data，或者是否可以继续使用 CIFAR-100？
6. CEMD 是否会像 CARA-C v1 一样提高 worst_acc 但牺牲 avg_acc？
```

## 14. 建议下一步实现计划

建议不要继续盲跑 FedCARA v1，也不要马上做 AsymHFL hybrid 作为最终方案。更合理的下一步：

```text
Step 1: 保留 CARA-L 本地模块。
Step 2: 新增 CEMD 通信模块，不直接复用 AsymHFL teacher routing。
Step 3: 使用 held-out validation split 计算 per-class reliability。
Step 4: 在 public logits 上构造 class-wise mosaic teacher。
Step 5: 跑 debug smoke，确认 loss/accuracy/logging 正常。
Step 6: 跑 40-round Kaggle T4 full experiment。
```

第一版只需要比较：

```text
RAHFL:              56.41 / 44.72
CARA-L + AsymHFL:   57.36 / 46.23
FedCARA v1:         55.88 / 45.93
FedCARA-v2/CEMD:    待跑
```

成功标准：

```text
最低目标：
  final avg_acc > 56.41
  final worst_acc > 44.72

较强目标：
  final avg_acc >= 57.36
  final worst_acc >= 46.23

如果 avg_acc 略低但 worst_acc 明显更高：
  可以作为 fairness / weak-client robustness 方向，但论文主张需要调整。
```

## 15. 当前最需要咨询的问题

建议把以下问题直接拿去问其他 AI：

```text
1. 在模型异构 + label-skew Non-IID + corruption robust FL 中，
   用 class-wise expert mosaic teacher 替代 client-level teacher selection 是否合理？

2. CEMD 中每个类别 logit 来自不同客户端，最后拼成一个 softmax teacher，
   这个 teacher distribution 是否存在概率语义问题？如何修正？

3. 对异构模型 public logits 做 per-sample standardization 是否足够？
   是否需要 temperature calibration、rank-based logits 或 probability-level fusion？

4. held-out validation split 会减少本地训练数据。
   在每客户端只有 10000 张、且 tail class 很少的情况下，如何稳定估计 per-class reliability？

5. 如果 public data 是 CIFAR-100，而 private task 是 CIFAR-10，
   public logits 是否足以承载类别专家知识？
   是否需要引入少量同域 public CIFAR-10-like data？

6. 相比 RAHFL 的 AsymHFL，CEMD 的理论创新应该如何表述，才能不被认为只是 reweighting？

7. 如果实验上 CEMD 只提升 worst_acc，不提升 avg_acc，
   论文是否还能以 weak-client robustness / fairness 作为主要贡献？
```

## 16. 当前结论

当前项目已经明确排除了几个方向：

```text
D2C prior-debias route：基本失败。
Oracle D2C：没有改善。
FedPRIME-PAIR / CPAD：没有超过 LogitAvg。
PRAC-HFL：有行为，但没有稳定正收益。
FedCARA v1：worst_acc 有收益，但 avg_acc 不够。
```

当前最有价值的事实是：

```text
CARA-L + AsymHFL 已经超过 RAHFL。
```

但它创新性不足，因此下一步应该围绕：

```text
如何把通信从 client-level teacher selection
升级为 class-level expert composition
```

来形成更独立的新框架。

当前推荐主线：

```text
FedCARA-v2 = CARA-L + CEMD
```

不要继续把最终论文建立在 D2C、PAIR 或 FedCARA v1 上。
