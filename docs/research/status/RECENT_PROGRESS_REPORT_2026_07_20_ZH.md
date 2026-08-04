# FedPRIME-D2C 近期研究进展汇报

更新时间：2026-07-20

## 1. 本阶段研究目标

本项目希望在以下联合场景中研究异构联邦视觉学习：

```text
模型异构
+ label-skew 数据异构
+ 图像数据损坏
+ 客户端之间无法直接聚合模型参数
```

前期实验表明，单纯把 PRIME、DCL、public-logit 蒸馏等现有组件重新组合，
很难形成稳定且可解释的提升。因此近期工作从“继续调模块”转向两件事：

1. 明确 RAHFL 尚未解决的具体失败模式；
2. 对新方法的本地机制和通信机制分别做小预算 Go/No-Go 验证。

当前研究主线为：

```text
CLE-HFL 问题设定 + FedEASE v2.1 候选方法
```

---

## 2. 新问题：CLE-HFL

### 2.1 问题定义

CLE-HFL 全称为：

```text
Corruption-Label Entanglement in Heterogeneous Federated Learning
```

普通数据损坏设定通常默认损坏类型与类别相互独立。CLE-HFL 进一步考虑：

> 在一个客户端内部，某些类别更经常伴随特定损坏，模型可能把损坏模式当成
> 预测类别的捷径，而没有真正学习稳定的类别语义。

例如，在某个客户端的训练数据中：

```text
猫经常伴随 noise；
狗经常伴随 blur；
汽车经常伴随 weather。
```

如果测试时交换这些组合，例如“猫 + blur”，依赖 shortcut 的模型就会明显退化。

### 2.2 可控协议

协议使用两个独立参数：

```text
alpha：控制客户端之间的 label-skew；越小表示类别分布越不均衡。
gamma：控制类别与损坏的纠缠强度；越大表示 shortcut 越强。
```

近期实验固定：

```text
alpha=0.5
seed=0
clients=4
samples_per_client=10000
models=ResNet10 / ResNet12 / ShuffleNet / MobileNetV2
```

只改变 `gamma`，从而观察性能变化是否确实来自 corruption-label entanglement。

### 2.3 新增评价指标

除平均准确率和最差客户端准确率外，加入：

```text
WCCA：Worst Class-Corruption Accuracy
      关注最困难的类别-损坏组合，越高越好。

CFG：Counterfactual Gap
     同一类别在不同损坏环境下最好与最差表现的平均差距，越低越好。
```

Avg/Worst 反映整体性能，WCCA/CFG 用于揭示平均准确率掩盖的 shortcut 问题。

---

## 3. RAHFL 在 CLE-HFL 下的诊断结果

使用完整 RAHFL 风格基线：

```text
AugMix + JSD + DCL + AsymHFL
```

在其他条件不变时得到：

| gamma | Avg Acc ↑ | Worst Acc ↑ | WCCA ↑ | CFG ↓ |
|---:|---:|---:|---:|---:|
| 0.0 | 52.17 | 44.17 | 35.35 | 2.54 |
| 0.6 | 50.82 | 42.83 | 25.88 | 5.91 |
| 0.9 | 46.72 | 38.16 | 19.32 | 10.91 |

从 `gamma=0.0` 增加到 `gamma=0.9`：

```text
Avg Acc   下降 5.45 点
Worst Acc 下降 6.02 点
WCCA      下降 16.03 点
CFG       上升 8.37 点
```

这组结果说明：即使 RAHFL 使用了强数据增强、JSD 一致性和 DCL，随着
corruption-label 纠缠增强，其反事实类别-损坏泛化仍显著恶化。

目前能够得出的严谨结论是：

> CLE-HFL 在 `alpha=0.5, seed=0` 下已经得到初步实验支持，并暴露了 RAHFL
> 在 corruption-label shortcut 方面的盲点。

尚不能声称该场景已经被完整验证，因为仍缺少多 seed、其他 alpha 和真实数据验证。

---

## 4. 第一版解决方案及失败分析

第一版 FedCLEAR 使用：

```text
CCRE：反事实损坏风险约束
+ IRD：跨客户端关系蒸馏
```

40 轮 `gamma=0.9` 结果：

| 方法 | Avg Acc ↑ | Worst Acc ↑ | WCCA ↑ | CFG ↓ |
|---|---:|---:|---:|---:|
| RAHFL | 46.72 | 38.16 | 19.32 | 10.91 |
| FedCLEAR v0.1 | 45.41 | 36.42 | 17.80 | 11.42 |

第一版方法没有超过 RAHFL。诊断发现：

1. CCRE 的训练代理风险下降，但私有反事实测试仍保留原有 shortcut；
2. IRD 的跨域公共教师分歧较高，公共关系并未提供可靠知识；
3. 将不可靠通信知识加入训练，反而带来负迁移。

这次负结果促使当前方法不再把“多加一个蒸馏损失”当作默认解决方案，而是先验证
本地去 shortcut 机制，再单独验证通信。

---

## 5. 当前候选框架：FedEASE v2.1

FedEASE 的设计目标是：

```text
先在客户端内部降低类别与损坏环境的依赖，
再只传递跨环境和跨客户端都较一致的类别关系。
```

整体结构为：

```text
RAHFL robust local base
+ PEW
+ BER
+ CDep
+ EBST-v2
+ class-wise SCP
```

### 5.1 保留的强鲁棒训练基座

保留 RAHFL 的：

```text
AugMix + JSD + DCL
```

原因是现有实验已经证明该本地训练是一个较强的抗损坏基础。近期工作不再把
“换掉所有 RAHFL 模块”作为目标，而是针对 CLE-HFL 中尚未解决的 shortcut
进行有明确作用对象的修改。

### 5.2 PEW：Public Environment Witness

PEW 使用无类别语义要求的公共图像和已知 corruption operators 学习环境识别器，
用于替代实验中的真实 corruption group 标签。

```text
Oracle 模式：直接读取数据生成阶段保存的环境 ID，用于验证理论上限。
Learned 模式：由 PEW 预测环境，用于检验实际可部署性。
```

PEW 只估计环境，不向客户端提供 CIFAR-10 类别知识。

### 5.3 BER：Balanced Environment Risk

BER 在同一个类别内部平衡不同损坏环境的训练贡献，避免 dominant corruption
控制该类别的分类风险。

通俗地说：

> 如果“猫 + noise”很多，而“猫 + blur”很少，BER 不让大量 noise 样本完全淹没
> 少量 blur 样本。

类别-环境计数只在客户端本地使用，不上传服务器。

### 5.4 CDep：Conditional Environment Dependence Penalty

CDep 只在同一类别内部，约束视觉表示与环境变量的统计相关性：

```text
在已经知道图片都是“猫”的条件下，
模型表示不应仍然主要由 noise / blur / weather / digital 决定。
```

它使用固定随机投影，避免通过一个可学习 projection head 人为制造低相关性。
CDep 是条件依赖代理项，不能声称严格实现统计独立。

### 5.5 EBST-v2：环境平衡结构通信

EBST-v2 不聚合异构模型参数，而是通信模型无关的类别对 margin 关系。

主要安全修正包括：

```text
1. 只有同时具备类别 c 和竞争类别 j 支持的客户端，才能成为该类别对的来源；
2. 每个接收客户端使用 leave-one-out 教师，不把自己的输出再教回自己；
3. 只有跨环境、跨客户端都较一致的关系才通过 gate；
4. 只更新分类器头，并按类别执行 SCP 和通信梯度范数限制。
```

服务器只接收阈值化支持掩码和关系统计，不接收精确类别数量。

### 5.6 SCP：类别级安全通信投影

SCP 比较本地分类梯度与通信梯度。当两者冲突时，对通信梯度进行投影和限幅，
减少通信损害已有本地能力的风险。SCP 是保护机制，不作为主要创新点。

---

## 6. FedEASE 本地机制探针

为了避免直接跑完整 40 轮后才发现模块无效，先进行匹配的 12 轮实验：

```text
CLE-HFL alpha=0.5, gamma=0.9, seed=0
四个异构模型
无独立预训练
两组都不通信
唯一变量：是否启用 Oracle BER+CDep
```

结果：

| 方法 | Avg Acc ↑ | Worst Acc ↑ | WCCA ↑ | CFG ↓ |
|---|---:|---:|---:|---:|
| Oracle local control | 37.58 | 30.11 | 13.70 | 10.86 |
| Oracle BER+CDep | **41.62** | **35.52** | **14.00** | **6.16** |
| 差值 | **+4.04** | **+5.41** | **+0.30** | **-4.70** |

进一步观察：

```text
四个客户端最终准确率全部提升；
最差 corruption-group accuracy 提升 6.26 点；
最差 client-corruption accuracy 提升 9.48 点；
最后五轮 WCCA 均值由 8.91 提升至 16.22；
最后五轮 CFG 均值由 10.95 降低至 5.48。
```

这说明 BER+CDep 联合本地机制在 Oracle 环境信息下具有明确正向信号。
但该实验尚未区分 BER 和 CDep 各自的独立贡献。

---

## 7. 通信机制实验

### 7.1 旧 EBST：负结果

在 Oracle BER+CDep 上加入旧 EBST+SCP：

| 方法 | Avg Acc ↑ | Worst Acc ↑ | WCCA ↑ | CFG ↓ |
|---|---:|---:|---:|---:|
| BER+CDep local | 41.62 | 35.52 | 14.00 | 6.16 |
| 旧 EBST+SCP | 38.70 | 34.72 | 15.33 | 6.42 |
| 差值 | -2.92 | -0.80 | +1.33 | +0.26 |

旧版通信导致 client 2 从 `45.22%` 下降到 `34.72%`。原因不是通信没有执行，
而是旧 gate 只检查环境稳定性，无法识别接收客户端和类别级冲突；旧 SCP 又以
整个 classifier head 为粒度，掩盖了局部类别负迁移。

### 7.2 EBST-v2：安全修复有效，但平均收益尚不稳定

EBST-v2 加入类别对来源资格、recipient LOO teacher、跨客户端一致性 gate 和
class-wise SCP。

| 方法 | Avg Acc ↑ | Worst Acc ↑ | WCCA ↑ | CFG ↓ |
|---|---:|---:|---:|---:|
| BER+CDep local | 41.62 | 35.52 | 14.00 | 6.16 |
| BER+CDep + EBST-v2 | **41.95** | **36.23** | **14.70** | **5.19** |
| 最终轮差值 | +0.33 | +0.71 | +0.70 | -0.97 |

通信诊断：

```text
有效环境比例：54.63%
有效类别对比例：67.75%
每个类别对平均来源数：2.16
平均通信 gate：21.01%
SCP 冲突率：47.59%
SCP 保留通信梯度范数：57.63%
```

安全性方面取得了实质改善：

```text
client 0: +0.71
client 1: -0.14
client 2: +0.51
client 3: +0.22
```

旧版 client 2 的整体崩塌已经消失。但最后五轮均值显示：

```text
Avg 差值   = -0.165
Worst 差值 = +0.440
WCCA 差值  = +0.765
CFG 差值   = 0.000
```

因此，最终轮 `Avg +0.33` 不能被当作稳定通信增益。当前最准确的结论是：

> EBST-v2 已经从有害通信修正为基本安全、偏向改善最差性能的通信，但尚未证明
> 它能够稳定提升平均准确率。

仍然存在个别类别退化，例如 client 3 class 3 约下降 `11.23` 点。因此不能直接
进入完整 40 轮训练，也不应只通过调大或调小通信权重继续试错。

---

## 8. PEW 可部署性探针

Oracle 实验使用数据生成时保存的真实 corruption group，只能验证方法上限。
为了检验实际不知道环境标签时是否仍然有效，进一步运行了 12 轮 learned PEW
local probe。

结果：

| 方法 | Avg Acc ↑ | Worst Acc ↑ | WCCA ↑ | CFG ↓ |
|---|---:|---:|---:|---:|
| local control | 37.58 | 30.11 | 13.70 | 10.86 |
| Oracle BER+CDep | 41.62 | 35.52 | 14.00 | 6.16 |
| Learned PEW BER+CDep | **40.37** | **35.42** | **13.93** | **6.37** |

Learned PEW 相对 control：

```text
Avg   +2.79
Worst +5.31
WCCA  +0.23
CFG   -4.49
```

Learned PEW 相对 Oracle BER+CDep：

```text
Avg   -1.25
Worst -0.10
WCCA  -0.08
CFG   +0.22
```

预先冻结的门槛为 `40.5/34.0/WCCA 13/CFG 7`。Learned PEW 通过了 Worst、
WCCA 和 CFG，最终 Avg 为 `40.3694`，距离门槛仅 `0.1306`；第 10 轮最佳
Avg 达到 `41.4194`。

PEW 自身的诊断为：

```text
公共验证 environment accuracy：epoch 3 最好 57.4%，最终 epoch 4 为 52.5%
私有数据精确 environment-group accuracy：38.83%
各客户端 unknown rate：约 49.8% 至 57.7%
```

虽然精确环境分类准确率不高，但 PEW 保留了 Oracle 几乎全部 Worst/WCCA/CFG
收益，说明连续环境表示和粗粒度环境划分已经包含可用信息。当前应定义为“接近
通过、值得继续”，而不是严格通过：Avg 略低于门槛，而且当前代码保存最后一个
PEW epoch，而不是公共验证集上的最佳 checkpoint。

---

## 9. 与 RAHFL 的当前关系

同一 12 轮位置的 RAHFL 参考为：

| 方法 | Avg Acc ↑ | Worst Acc ↑ | WCCA ↑ | CFG ↓ |
|---|---:|---:|---:|---:|
| RAHFL round 11 | 37.46 | 30.70 | 8.15 | 9.73 |
| FedEASE Oracle + EBST-v2 round 11 | 41.95 | 36.23 | 14.70 | 5.19 |
| FedEASE Learned PEW local round 11 | 40.37 | 35.42 | 13.93 | 6.37 |
| Oracle+EBST-v2 相对 RAHFL | +4.49 | +5.53 | +6.55 | -4.54 |
| Learned PEW local 相对 RAHFL | +2.91 | +4.73 | +5.78 | -3.36 |

Learned PEW local 相对同轮 RAHFL 为：

```text
Avg +2.91, Worst +4.73, WCCA +5.78, CFG -3.36
```

这些结果说明 FedEASE 在机制探针预算下具有较强信号，但需要强调：

1. Oracle+EBST-v2 仍使用真实 environment，Learned PEW local 尚未加入通信；
2. 12 轮结果不能与 RAHFL 的 40 轮最终结果直接比较；
3. 当前提升的大头来自 BER+CDep，EBST-v2 的独立平均收益仍较弱；
4. 当前正式压缩包只输出了 random 测试，完整 clean/same/swapped/unseen 仍待补齐。

所以目前还不能对外声称“FedEASE 已经正式击败 RAHFL”。

---

## 10. 本阶段已经完成的工程工作

1. 完成 CLE-HFL 数据协议生成、审计和可复现打包；
2. 完成 RAHFL 在不同 `gamma` 下的诊断实验；
3. 增加 Avg/Worst/WCCA/CFG 及 class-corruption 细分评价；
4. 实现完整可开关的 FedEASE v2.1：PEW、BER、CDep、EBST、SCP；
5. 实现 EBST-v2 的类别对来源筛选、LOO 教师、agreement gate 和 class-wise SCP；
6. 建立 OpenI 一键入口、结果打包回传和 heartbeat 日志；
7. 完成针对性单元测试、真实四模型 smoke test 和四次正式 12 轮机制探针；
8. 将正结果、负结果及停止条件持续记录到项目记忆，避免重复烧算力。

---

## 11. 当前结论

### 已确认

```text
1. CLE-HFL 在当前受控协议下确实暴露了 RAHFL 的 corruption-label shortcut。
2. Oracle BER+CDep 能明显改善 Avg、Worst 和 CFG。
3. 旧 EBST 会产生严重客户端/类别级负迁移。
4. EBST-v2 成功消除了旧版客户端整体崩塌，并改善 Worst/WCCA/CFG。
5. Learned PEW 虽然精确环境识别较弱，仍保留了大部分 Oracle 下游收益。
```

### 尚未确认

```text
1. 验证集校准后的 PEW 能否严格通过 Avg 门槛。
2. Learned PEW 与 EBST-v2 组合后是否仍保持正向收益。
3. EBST-v2 是否具有稳定的平均准确率贡献。
4. 完整 FedEASE 在 40 轮、多 seed 下是否稳定超过 RAHFL。
5. BER 与 CDep 各自贡献多大。
6. CLE-HFL 是否能在更多 alpha、seed 和真实数据上成立。
```

---

## 12. 下一步计划

### 第一步：修正 PEW 模型选择与 unknown 校准

当前公共验证 environment accuracy 在 epoch 3 达到 `57.4%`，但代码保存的最终
epoch 只有 `52.5%`。下一次实验前应：

```text
使用公共验证指标选择 PEW checkpoint；
在公共验证集上校准 unknown threshold；
不使用私有测试标签参与模型或阈值选择。
```

### 第二步：只跑一次 Learned PEW + EBST-v2 组合探针

```text
保持当前 12 轮、本地 batch、通信预算和所有其他配置不变；
不重新运行 RAHFL、Oracle local 或旧 EBST；
比较已经保存的 Learned PEW local 与组合候选。
```

如果组合探针在 Avg/Worst 上不低于 Learned PEW local，并继续改善 WCCA/CFG：

```text
再进入 FedEASE 与 RAHFL 的 40 轮、多 seed 正式对比。
```

如果组合通信再次损害 Avg/Worst：

```text
停止 EBST 扩展，重新定义接收客户端类别级验收机制；
不通过单独调整 lambda 继续试错。
```

如果后续仍需改进 EBST-v2，应增加接收客户端类别级验收或 trust-region 保护，
而不是只调整 `lambda`。

---

## 13. 下午口头汇报建议

可以按下面的顺序汇报：

> 这周我没有继续直接堆模块，而是先把 RAHFL 在数据损坏和数据异构同时存在时
> 的失败模式做了可控化。新协议用 gamma 控制类别与损坏的虚假关联。实验中
> gamma 从 0 增加到 0.9 后，RAHFL 的平均准确率下降 5.45 点，最差类别-损坏
> 准确率下降 16 点，说明这个 shortcut 问题确实存在。

> 在方法上，我把本地去 shortcut 和跨客户端通信拆开验证。本地的 BER+CDep
> 在相同 12 轮下使平均准确率提升 4.04 点、最差客户端提升 5.41 点、CFG 降低
> 4.70 点。第一版通信出现了明显负迁移，所以我没有直接跑完整实验，而是根据
> 诊断改成类别对来源筛选、leave-one-out 教师和类别级安全投影。新版已经消除
> 客户端崩塌，Worst 和 WCCA 有提升，但平均精度最后五轮还没有稳定提高。

> 我还完成了 learned PEW 探针。PEW 对私有环境的精确识别率只有 38.83%，但最终
> Avg/Worst 为 40.37/35.42，相对本地 control 仍提升 2.79/5.31 点，CFG 降低
> 4.49 点，说明不使用真实环境标签时大部分收益仍然能够保留。

> 因此现在的进展是：新问题和本地机制已经得到正向信号，PEW 接近通过，通信也
> 从有害修到了基本安全，但完整方法还不能宣称完成。下一步先用公共验证集选择
> PEW checkpoint，再做一次 learned PEW+EBST-v2 组合探针；通过后才投入 40 轮
> 和多 seed，避免继续盲目试错。

### 老师可能追问：为什么只跑 12 轮？

回答：

> 12 轮是机制 Go/No-Go 探针，用相同预算快速判断一个模块是否值得进入 40 轮。
> 它不能代替正式结果，但能够避免对已经产生负迁移的通信浪费完整训练资源。

### 老师可能追问：现在是否已经超过 RAHFL？

回答：

> 同一 12 轮预算下，Learned PEW local 比 RAHFL 高约 2.91 点 Avg 和 4.73 点
> Worst；但还没有完成通信组合、40 轮和多 seed，所以只能说出现了较强正向
> 信号，不能说已经正式超过。

### 老师可能追问：通信模块到底有没有用？

回答：

> EBST-v2 对最差客户端、WCCA 和 CFG 有改善，并且解决了旧版负迁移；但平均准确率
> 的最后五轮均值没有提升，因此目前只能证明通信更安全、偏向改善尾部性能，不能
> 声称它已经稳定提高整体性能。
