# 当前研究现状、实验结果与核心困难总结

本文用于向其他 AI、导师或同学咨询新的研究框架。目标是让对方不需要阅读完整代码和聊天记录，也能理解当前项目的真实处境、已尝试路线、实验结果、瓶颈，以及为什么目前方案还不足以支撑一篇小论文。

## 1. 我的研究目标

我正在做一个联邦学习方向的小论文，希望同时覆盖三个挑战：

```text
1. 数据损坏鲁棒性：客户端图像存在 corruption，例如噪声、模糊、天气、压缩等。
2. 模型异构：不同客户端使用不同模型结构，例如 ResNet10、ResNet12、ShuffleNet、MobileNetV2。
3. 数据异构：客户端数据是 label-skew Non-IID，不同客户端类别分布明显不同。
```

当前最强基线是 RAHFL：

```text
RAHFL = AugMix + JSD + DCL + AsymHFL
```

我的目标不是只做普通复现，而是提出一个自己的框架，在相同的：

```text
corrupted CIFAR-10-C private data
+ CIFAR-100 public data
+ heterogeneous client models
+ Dirichlet label-skew Non-IID
```

设定下超过 RAHFL，最好能在 average accuracy 和 worst-client accuracy 上都明显提升。

## 2. 当前实验数据和基本设定

目前使用的是 RAHFL-style 数据。

私有训练/测试数据：

```text
RAHFL-master/Dataset/cifar_10_c/train/random_corrupt_1.npy
RAHFL-master/Dataset/cifar_10_c/train/labels.npy
RAHFL-master/Dataset/cifar_10_c/test/random_corrupt_1.npy
RAHFL-master/Dataset/cifar_10_c/test/labels.npy
```

含义：

```text
random_corrupt_1.npy 是 RAHFL 预生成的 corrupted CIFAR-10。
corrupt_rate=1 表示训练和测试图像都经过随机 corruption。
```

公共数据：

```text
CIFAR-100
```

公共数据主要用于 public logits 蒸馏通信，即客户端在 CIFAR-100 public images 上输出 CIFAR-10 logits，服务器/其他客户端利用这些 logits 进行知识迁移。

模型异构客户端：

```text
client 0: ResNet10
client 1: ResNet12
client 2: ShuffleNet
client 3: MobileNetV2
```

主要 Non-IID 划分：

```text
Dirichlet alpha = 0.5, 0.3, 0.1
```

alpha 越小，类别分布越偏斜。

当前统一 runner 的 RAHFL 不是论文最强完整复现，因为：

```text
1. 没有独立 40 epoch local pretraining。
2. public batches per round 比原论文少。
3. batch size 更小。
```

但是所有方法在统一 runner 下使用相同训练预算，所以可以作为当前公平对比。

## 3. 我重新精读 RAHFL 源码后的理解

RAHFL 不是简单套用原始 AugMix。源码中真实本地训练是四视图结构：

```text
images[0] = clean/base crop
images[1] = AugMix view 1
images[2] = AugMix view 2
images[3] = extra transformed crop
```

本地损失为：

```text
L_local = CE(clean)
        + 12 * JSD(clean, aug1, aug2)
        + DCL(clean_feature, weak_feature, strong_feature)
```

DCL 的关键不是简单把所有增强视图拉近，而是：

```text
clean + weak 做 supervised contrastive；
weak view 作为关系教师；
strong AugMix view 作为学生；
strong view 模仿 weak view 对 clean/weak 特征库的关系分布。
```

所以 RAHFL 本地模块真正强的地方是：

```text
1. AugMix/JSD 约束 prediction consistency。
2. DCL 约束 feature relation consistency。
3. strong augmentation 不直接作为普通正样本，而是通过关系蒸馏被温和约束。
```

AsymHFL 通信部分是：

```text
1. 每轮先评估客户端准确率。
2. 在 CIFAR-100 public images 上收集所有客户端 logits。
3. 准确率低的客户端向准确率高的客户端做 KL distillation。
```

RAHFL 用 public logits 绕开了模型异构问题，因为不同模型只需要输出同样的 10 类 logits，不需要对齐 feature 维度。

## 4. 已经尝试过的主要路线

### 4.1 PRIME + D2C

最开始想把 RAHFL 的 AugMix 换成更强的 PRIME 增强，并提出 D2C 通信：

```text
FedPRIME-D2C = PRIME local training + D2C public-logit debias communication
```

D2C 设计思想：

```text
根据客户端在 public CIFAR-100 上的平均预测估计类别 prior，
再用 prior debias、class-balanced aggregation、complementary KD 修正 public logits。
```

结果：

```text
PRIME + LogitAvg final avg_acc 约 52.10
FedPRIME-D2C final avg_acc 约 52.31
Oracle D2C final avg_acc 约 51.74
RAHFL baseline final avg_acc 约 56.41
```

结论：

```text
D2C 几乎没有超过普通 LogitAvg。
即使用 oracle prior 也没有改善。
说明用跨域 CIFAR-100 public logits 去估计 CIFAR-10 private prior 这条路不可靠。
```

### 4.2 FedPRIME-PAIR / CPAD

尝试把 public logits 的知识传递粒度从客户端级降到类别对级：

```text
类别对 margin
类别对 expertise
pairwise BCE distillation
```

结果：

```text
FedPRIME-PAIR final avg_acc 约 50.15
best avg_acc 约 51.10
```

结论：

```text
pairwise public-logit boundary distillation 没有超过 LogitAvg，更没有接近 RAHFL。
说明仅靠 public logits 的类别对边界，在跨域 CIFAR-100 上仍然很难传递有效 CIFAR-10 类别知识。
```

### 4.3 PRAC-HFL

尝试设计一个接收端自适应通信机制：

```text
对每个 candidate teacher 做 head-only virtual update；
用本地 held-out/private route risk 判断 teacher 是否有帮助；
只接受能降低风险的 teacher logits。
```

早期结果有一点希望，但出现 NaN。安全版本稳定后结果下降。

结果：

```text
PRAC public1 final avg_acc = 54.63, worst_acc = 41.88
PRAC public4 final avg_acc = 52.96, worst_acc = 43.27
RAHFL final avg_acc = 56.41, worst_acc = 44.72
AugMix+DCL local-only final avg_acc = 56.11, worst_acc = 44.23
```

结论：

```text
PRAC 通信不是完全无效，但没有带来稳定正收益。
AugMix+DCL local-only 几乎追平 RAHFL，说明性能大头来自本地鲁棒训练。
PRAC 可能引入负迁移。
```

### 4.4 CARA-L / NIR-DCL

尝试改进 RAHFL 的 DCL，本地做 Non-IID-aware robust contrastive learning。

设计包括：

```text
1. class-balanced DCL
2. local feature queue
3. strong-view reliability gate
4. stable relation alignment
```

结果：

```text
NIR-DCL local-only final avg/worst = 53.30 / 36.01
NIR-DCL + AsymHFL final avg/worst = 57.36 / 46.23
RAHFL baseline final avg/worst = 56.41 / 44.72
```

结论：

```text
NIR-DCL local-only 反而比 RAHFL local-only 差很多。
但 NIR-DCL + AsymHFL 超过 RAHFL。
可能说明它不是单独增强本地训练，而是改变了本地表征，使其更适配 AsymHFL 通信。
但这个故事不够直接，有点难讲。
```

### 4.5 FedCARA / CARA-C

尝试把通信从整体准确率路由改为类别级可靠性路由：

```text
teacher class reliability = teacher 在 class c 上的准确率
student class need = 1 - student 在 class c 上的准确率
class weight = reliability * need
```

结果：

```text
FedCARA final avg/worst = 55.88 / 45.93
RAHFL final avg/worst = 56.41 / 44.72
CARA-L + AsymHFL final avg/worst = 57.36 / 46.23
```

结论：

```text
FedCARA 提升了 worst-client accuracy，但 average accuracy 没超过 RAHFL。
它偏向弱客户端/弱类别，可能牺牲整体平均性能。
纯替换 AsymHFL 不如原始 AsymHFL 稳。
```

### 4.6 SARA

当前最有希望的路线是 SARA：

```text
SARA = Skew-Aware Robust Alignment
```

它仍然依托 RAHFL 的本地 AugMix/JSD 四视图训练，但替换 DCL 的关系学习部分。

SARA 主要做三件事：

```text
1. 类别偏斜校准：
   根据本地 class count 给 tail class 更大 contrastive 权重。

2. strong-view reliability：
   根据 strong AugMix view 的 true-class margin 判断该增强视图是否可靠。

3. stable relation alignment：
   用 softmax(sim/T) 替代原始 DCL 中更激进的 softmax(exp(sim)/T)。
```

核心直觉：

```text
RAHFL-DCL 在 Non-IID 下容易被 head class 主导。
SARA 试图让 tail class 在鲁棒关系学习中有更大梯度声音，
同时降低被强增强破坏的视图对关系学习的负面影响。
```

重要实验结果：

```text
RAHFL alpha=0.5 seed0:
  final avg/worst = 56.41 / 44.72

SARA + AsymHFL alpha=0.5 seed0:
  final avg/worst = 57.83 / 46.59

RAHFL alpha=0.5 seed1:
  final avg/worst = 56.645 / 45.29

SARA + AsymHFL alpha=0.5 seed1:
  final avg/worst = 57.2975 / 46.23

SARA + AsymHFL alpha=0.5 seed2:
  final avg/worst = 58.0025 / 45.90
```

alpha=0.3：

```text
RAHFL final avg/worst = 45.8425 / 41.9200
SARA final avg/worst = 46.7325 / 42.7700
gap = +0.89 avg, +0.85 worst
```

alpha=0.1：

```text
RAHFL final avg/worst = 35.6825 / 29.3300
SARA final avg/worst = 35.9625 / 29.1000
gap = +0.28 avg, -0.23 worst
```

结论：

```text
SARA 在 alpha=0.5 有比较明显提升。
在 alpha=0.3 有稳定但不大的提升。
在 alpha=0.1 基本打平，没有大胜。
```

最大问题：

```text
SARA local-only 很弱：
  final avg/worst = 54.10 / 32.06

SARA + AsymHFL 很强：
  final avg/worst = 57.83 / 46.59
```

这说明 SARA 不是一个独立更强的本地模块，而更像是让本地表征更适配 AsymHFL。这个现象有一定研究价值，但也导致论文动机比较难写。

### 4.7 SARA residual / CCAD

后来尝试在 AsymHFL 上加辅助通信项：

```text
SARA + receiver-side class residual
SARA + CCAD corruption-consistent asymmetric distillation
```

SARA residual 结果：

```text
SARA residual final avg/worst = 57.655 / 46.54
SARA + AsymHFL final avg/worst = 57.83 / 46.59
```

结论：

```text
residual 仍然超过 RAHFL，但没有超过更简单的 SARA + AsymHFL。
它不适合作为主线创新。
```

CCAD 设计思想：

```text
对 public image 做 clean/augmented public views。
teacher 如果在 public clean/aug views 上预测一致且自信，就认为可靠。
student 如果不稳定或不自信，就认为需要学习。
在 AsymHFL 基础上加一个 corruption-consistent residual KD。
```

但我对这个方向也没有足够信心，因为 AugMix/JSD 本身已经在约束增强一致性，再额外做 public consistency 可能只是重复机制。

## 5. 当前最关键的实验事实

目前最重要的一组对比是：

| 方法 | final avg_acc | final worst_acc | 结论 |
|---|---:|---:|---|
| RAHFL | 56.41 | 44.72 | 当前强基线 |
| AugMix+DCL local-only | 56.11 | 44.23 | 几乎追平 RAHFL |
| PRIME + LogitAvg | 约 52.10 | 约 39.72 | 明显弱于 RAHFL |
| FedPRIME-D2C | 约 52.31 | 约 39.78 | D2C 基本无效 |
| Oracle D2C | 约 51.74 | 约 39.13 | oracle prior 也没救 |
| FedPRIME-PAIR | 约 50.15 | 约 39.83 | 类别对 public logits 无效 |
| PRAC-HFL public4 | 52.96 | 43.27 | 通信有行为但负迁移 |
| NIR-DCL + AsymHFL | 57.36 | 46.23 | 小幅超过 RAHFL |
| FedCARA | 55.88 | 45.93 | worst 提升，avg 不够 |
| SARA + AsymHFL | 57.83 | 46.59 | 当前最好结果 |
| SARA residual | 57.655 | 46.54 | 不如 SARA + AsymHFL |

最重要的观察：

```text
1. RAHFL 的强度主要来自本地 AugMix/JSD/DCL。
2. AugMix+DCL local-only 已经几乎追平 RAHFL。
3. 我设计的多个 public-logit 通信替代方案都没有稳定超过 AsymHFL。
4. 目前最好的 SARA 仍然依赖 AsymHFL，创新点显得偏小。
5. 在 extreme alpha=0.1 下，SARA 并没有明显超过 RAHFL。
```

## 6. 为什么现在还撑不起一篇小论文

我目前最担心的是：虽然有一些正结果，但还不够像一篇完整论文。

### 6.1 提升幅度不够大

SARA 在 alpha=0.5 seed0 上提升：

```text
+1.42 avg_acc
+1.87 worst_acc
```

这看起来不错，但不是压倒性优势。

alpha=0.3：

```text
+0.89 avg_acc
+0.85 worst_acc
```

alpha=0.1：

```text
+0.28 avg_acc
-0.23 worst_acc
```

这说明方法并没有在越极端 Non-IID 下越强，反而在最极端情况下基本打平。

### 6.2 创新点容易被认为太小

当前最好方法 SARA + AsymHFL 仍然依赖：

```text
AugMix
JSD
RAHFL-style four-view local training
AsymHFL public logits communication
```

SARA 只是替换或修正了 DCL 的关系学习部分。

可能被质疑为：

```text
只是在 RAHFL 的 DCL 上加了 class weight、view reliability 和更稳定的 relation loss。
```

如果没有非常强的理论动机和诊断实验，这确实像“小修小补”。

### 6.3 通信创新没有站住

我尝试过很多通信路线：

```text
D2C
Oracle D2C
PAIR/CPAD
PRAC-HFL
CARA-C
CCAD
receiver-side residual
```

但结果都不够理想。

这导致一个问题：

```text
如果最终还使用 AsymHFL，那么论文看起来像：
RAHFL 本地 DCL 的小改进 + 原始 AsymHFL。
```

导师可能会认为工作量和创新性不足。

### 6.4 public logits 的信息瓶颈非常明显

RAHFL 用 CIFAR-100 public images 来传 CIFAR-10 知识。但在数据异构尤其 missing/tail class 下，这件事本身很困难。

已有诊断显示：

```text
RAHFL 对 missing class 的知识迁移并不真正成功。
D2C / Oracle D2C / PAIR 也没能让 missing/tail class 有明显改善。
```

这说明：

```text
跨域 CIFAR-100 public logits 能传递的信息很有限。
```

如果继续依赖 public logits，可能很难得到突破性框架。

### 6.5 现有实验还不够完整

还缺：

```text
1. alpha=1.0 正常 Non-IID 下的验证。
2. 真正不同 partition seeds 的多 seed 结果。
3. RAHFL seed=2 匹配对照。
4. tail_acc / head_acc / per-client / per-class 的系统诊断。
5. SARA 各组件消融。
6. 是否需要 40 epoch pretraining 的公平强基线对比。
```

所以现在不能很有底气地说：

```text
已经形成完整论文证据链。
```

## 7. 当前我真正卡住的问题

现在我面临的问题不是代码跑不起来，而是研究设计卡住了。

具体是：

```text
1. RAHFL 太强，尤其是本地 AugMix/JSD/DCL。
2. 我尝试替换 AugMix 为 PRIME，没有打过 RAHFL。
3. 我尝试多个 public-logit 通信模块，也没有稳定打过 AsymHFL。
4. 当前最好的 SARA 只是小幅改 DCL，并且仍然依赖 AsymHFL。
5. 老师不希望我换方向，希望继续完成“抗数据损坏 + 模型异构 + 数据异构”这条线。
6. 我需要一个更有说服力、更像完整论文贡献的新框架。
```

我不希望继续“想到一个点就跑一个实验”，因为这样很像盲盒试错，算力和时间都耗不起。

我希望先从理论层面想清楚：

```text
到底应该如何同时解决 corrupted data、model heterogeneity 和 label-skew Non-IID？
```

## 8. 我希望新的框架满足什么条件

理想的新框架应该满足：

```text
1. 不是简单调 loss 权重。
2. 不只是 RAHFL-DCL 的小修小补。
3. 最好能解释为什么 RAHFL 在数据异构下仍然不足。
4. 能自然支持模型异构，不要求所有客户端 feature 维度一致。
5. 能在 corrupted data 下有明确鲁棒机制。
6. 能处理 label-skew Non-IID，尤其 tail/worst client。
7. 不要过度依赖 public logits 的跨域语义幻觉，除非能说明为什么 public logits 仍然有效。
8. 如果继续用 public data，需要有比 AsymHFL 更清晰的知识选择机制。
9. 如果不用 public logits，需要说明异构模型之间怎么通信。
10. 最好能形成明确公式和论文故事，而不是拼装多个 trick。
```

## 9. 可以考虑的方向，但我还没有想通

### 9.1 继续基于 RAHFL 强基座

保留：

```text
AugMix + JSD + 四视图鲁棒训练
```

重点改：

```text
DCL 的关系学习机制
AsymHFL 的 teacher selection / distillation granularity
```

优点：

```text
最容易超过 RAHFL。
代码已有基础。
实验成本较低。
```

缺点：

```text
容易被认为只是 RAHFL 上的小改进。
创新性压力大。
```

### 9.2 基于 FedProto / prototype 的异构通信

思路：

```text
用类别原型或关系原型代替 public logits。
由于不同模型 feature 维度不同，不能直接聚合 feature。
需要设计模型无关的 prototype relation / distance / distribution。
```

优点：

```text
更像模型异构通信创新。
不一定依赖 CIFAR-100 public data。
```

缺点：

```text
FedProto 类方法已有很多工作。
如果只是上传 prototype / relation graph，也容易撞已有方法。
如何在 corruption 下证明 prototype 更可靠也不容易。
```

### 9.3 基于对比学习的更先进改进

老师建议查监督对比学习后续工作，把更先进的思想整合到当前框架。

可能方向：

```text
class-balanced supervised contrastive learning
debaised contrastive learning
hard negative / false negative correction
long-tailed contrastive learning
robust contrastive learning under augmentation noise
```

优点：

```text
和 RAHFL-DCL 的缺陷直接相关。
```

缺点：

```text
如果只是搬一个已有 contrastive trick，创新不够。
必须和 corrupted heterogeneous FL 场景结合出新问题。
```

### 9.4 完全摆脱 public logits

思路：

```text
不用 public CIFAR-100 logits，而是通信某种模型无关结构知识。
例如类别关系、排序关系、决策边界、prototype geometry、校准统计等。
```

优点：

```text
可以绕开 public data 跨域不足。
```

缺点：

```text
理论和实现都很难。
之前尝试 pairwise boundary / relation graph 效果不好。
```

## 10. 希望其他 AI 帮我回答的问题

我希望其他 AI 帮我重点思考以下问题：

```text
1. 在“抗数据损坏 + 模型异构 + 数据异构”的场景下，
   有没有比 RAHFL 更自然、更有创新性的整体框架？

2. 如果继续基于 RAHFL，如何把创新点从“小改 DCL”提升为完整方法？

3. public logits 是否天然有信息瓶颈？
   如果有，应该用什么替代通信内容？

4. 能否设计一种 model-agnostic 的通信对象，
   既不要求 feature 维度相同，又能传递 label-skew 下的 tail/missing class 知识？

5. 能否围绕 corrupted data 提出比 AugMix/JSD 更有理论支撑的鲁棒学习机制？

6. 如果必须保留 AugMix/JSD，如何让新方法不像 RAHFL 的微小增量？

7. 有没有监督对比学习的新进展，特别适合：
   label-skew Non-IID、long-tail、augmentation noise、corruption robustness？

8. 是否可以构造一个新的本地-通信协同框架：
   本地学习负责 corruption-invariant relation，
   通信负责跨客户端补足 skewed relation？

9. 如何设计实验，证明新方法不是调参，而是真正解决了 RAHFL 的某个机制缺陷？

10. 如果最终只能比 RAHFL 高 1% 左右，
    论文应该如何构造贡献和故事，才有小论文可能？
```

## 11. 当前最真实的判断

目前我的真实处境是：

```text
1. 我已经做了大量代码实现和实验尝试。
2. 许多看起来合理的通信创新都没有超过 RAHFL。
3. RAHFL 的本地 AugMix/JSD/DCL 比预想中强很多。
4. 当前最好结果 SARA + AsymHFL 有正信号，但创新性和提升幅度都不够稳。
5. 如果继续只做小修小补，论文风险很大。
6. 需要重新从理论上设计一个更清晰、更有辨识度的框架。
```

一句话总结：

```text
我不是没有进展，而是已经排除了多条无效路线，并确认 RAHFL 的真正强点在本地鲁棒关系学习。
现在最需要的是一个新的、能正面解释并解决 RAHFL 在数据异构下不足的框架，而不是继续盲目叠加小模块。
```
