# CLE-HFL / FedEASE 当前研究现状与外部评审材料

更新时间：2026-07-22  
用途：提交给外部 AI 或研究人员，独立审核当前问题设定、实验结论、方法漏洞和下一步通信方向。  

> 请将本文视为一份研究审稿材料，而不是要求继续包装现有方法。请优先指出不可识别假设、与已有工作的重合、理论漏洞和最低成本的证伪实验。

---

## 1. 研究目标

项目最初希望同时处理：

```text
1. 模型异构：不同客户端使用 ResNet10、ResNet12、ShuffleNet、MobileNetV2；
2. 数据异构：客户端类别分布服从 Dirichlet label-skew；
3. 数据损坏：训练和测试图片包含多种 corruption；
4. 联邦通信：异构模型之间不能直接做参数平均。
```

主要对手是 RAHFL：

```text
RAHFL = AugMix + CE/JSD + DCL + AsymHFL public-logit communication
```

早期实验发现，在普通 RAHFL-style 场景中，本地 AugMix/JSD/DCL 已贡献大部分性能，AsymHFL 的增益较小。因此，仅替换一个局部损失或对 public logits 重新加权，创新性和实际收益都不足。

---

## 2. 新问题：CLE-HFL

CLE-HFL 全称：

```text
Corruption-Label Entanglement in Heterogeneous Federated Learning
```

它研究的不是“某个客户端整体都受到同一种损坏”，而是：

> 在同一个客户端内部，损坏模式与类别标签形成统计绑定，使模型可以利用损坏外观预测类别，从而学习 corruption-label shortcut。

设：

- (Y)：类别标签；
- (G)：损坏环境；
- (K)：客户端；
- \(\phi_k(c)\)：客户端 (k) 为类别 (c) 指定的主导损坏组；
- \(\gamma\in[0,1]\)：损坏与类别的纠缠强度。

当前协议使用：

\[
P_k(G=g\mid Y=c)
=
\gamma\,\mathbf 1[g=\phi_k(c)]
+
(1-\gamma)\operatorname{Uniform}(G).
\]

解释：

```text
gamma=0.0：损坏基本独立于类别；
gamma=0.6：中等 corruption-label entanglement；
gamma=0.9：强纠缠，模型很容易学习 shortcut。
```

不同客户端使用不同的 \(\phi_k\)，所以相同类别在不同客户端上可能绑定不同损坏。

### 2.1 当前实验协议

```text
private task: corrupted CIFAR-10
public query data: CIFAR-100（跨域、无标签）
clients: 4
models: ResNet10 / ResNet12 / ShuffleNet / MobileNetV2
label skew: Dirichlet alpha=0.5
seed: 目前 CLE 主要结论只有 seed=0
```

当前数据生成器把 CIFAR-C corruption 暂时归并为：

```text
noise / blur / weather / digital
```

测试阶段构造 class-corruption counterfactual combinations，用于检查模型是否把特定损坏当成类别捷径。

### 2.2 场景成立到什么程度

当前只能声称：

> CLE-HFL 在现有 CIFAR-10/CIFAR-C、四组 corruption、alpha=0.5、seed=0 的受控协议下能够稳定暴露 RAHFL 的 shortcut failure。

目前不能声称：

```text
1. 四组 corruption 覆盖现实中的全部损坏；
2. 该现象已经在真实联邦数据集上验证；
3. 任意 unseen corruption 都满足相同规律；
4. 当前问题已经通过多 seed 证明。
```

---

## 3. 评价指标

### 3.1 Avg Accuracy

四个客户端测试准确率的平均值，越高越好。

### 3.2 Worst Accuracy

四个客户端中最低的测试准确率，越高越好。

### 3.3 WCCA

Worst Class-Corruption Accuracy。对每个类别查看其在不同 corruption 环境下的准确率，取最差环境，再在类别间汇总：

\[
\operatorname{WCCA}
=
\frac{1}{C}\sum_c\min_g\operatorname{Acc}(c,g).
\]

WCCA 越高，说明模型在不利类别-损坏组合下越稳健。

### 3.4 CFG

Counterfactual Gap。衡量同一类别更换 corruption 后性能变化的幅度，可概括为：

\[
\operatorname{CFG}
=
\frac{1}{C}\sum_c
\left(
\max_g\operatorname{Acc}(c,g)
-
\min_g\operatorname{Acc}(c,g)
\right).
\]

CFG 越低越好。高 CFG 表明模型可能依赖类别相关的损坏 shortcut。

WCCA/CFG 使用测试阶段的环境标注计算，原则上可以作为评价指标；这些标注不应输入训练算法或教师路由。

---

## 4. CLE-HFL 的基线证据

RAHFL 在相同 `alpha=0.5, seed=0` 下的 40 轮结果：

| gamma | Avg Acc ↑ | Worst Acc ↑ | WCCA ↑ | CFG ↓ |
|---:|---:|---:|---:|---:|
| 0.0 | 52.17 | 44.17 | 35.35 | 2.54 |
| 0.6 | 50.82 | 42.83 | 25.88 | 5.91 |
| 0.9 | 46.72 | 38.16 | 19.32 | 10.91 |

从 `gamma=0.0` 墦加到 `gamma=0.9`：

```text
Avg Acc   -5.45
Worst Acc -6.01
WCCA      -16.03
CFG       +8.37（越高越差）
```

因此，问题信号在当前受控协议下是明显且单调的。但这仍属于 seed0 benchmark evidence，不是最终普适结论。

---

## 5. 普通 label-skew 场景的历史结果

这些结果不是 CLE-HFL 主结果，但解释了为什么项目转向新问题。

### 5.1 RAHFL 强基线

```text
RAHFL alpha=0.5 seed0 final Avg/Worst = 56.41/44.72
AugMix+DCL local-only                = 56.11/44.23
```

这表明本地 AugMix/JSD/DCL 已承担大部分性能，原始 AsymHFL 在该设置下的最终增益约为 `+0.30/+0.49`。

### 5.2 SARA + AsymHFL

```text
seed0: 57.8300/46.59
seed1: 57.2975/46.23
seed2: 58.0025/45.90

RAHFL seed0: 56.410/44.72
RAHFL seed1: 56.645/45.29
```

SARA + AsymHFL 有约 1--2 个点收益，但 SARA local-only 较弱，而且继续复用 AsymHFL，方法创新不足，因此没有作为最终主线。

### 5.3 早期 public-logit 负结果

```text
PRIME + LogitAvg       Avg/Worst about 52.10/39.72
FedPRIME-D2C           Avg/Worst about 52.31/39.78
Oracle D2C             Avg/Worst about 51.74/39.13
FedPRIME-PAIR          best Avg about 51.10
```

这些结果说明，简单重新加权跨域 CIFAR-100 public logits 没有可靠解决 label-skew 下的知识传递。

---

## 6. CLE-HFL 下的方法实验

### 6.1 FedCLEAR v0.1：明确负结果

```text
FedCLEAR v0.1 = CCRE local counterfactual risk + IRD public-logit communication
```

40 轮 `alpha=0.5, gamma=0.9, seed=0`：

| 方法 | Avg ↑ | Worst ↑ | WCCA ↑ | CFG ↓ |
|---|---:|---:|---:|---:|
| RAHFL | **46.72** | **38.16** | **19.32** | **10.91** |
| FedCLEAR v0.1 | 45.41 | 36.42 | 17.80 | 11.42 |

FedCLEAR v0.1 四项指标均未超过 RAHFL。主要问题是跨域公共 Logit 仍然混有语义偏差和 shortcut，所谓不变响应并未形成可靠教师。

### 6.2 FedEASE 本地机制探针

FedEASE v2.1 包含：

```text
PEW：Public Environment Witness
BER：Balanced Environment Risk
CDep：Conditional Dependence regularization
EBST/EBST-v2：Environment-Balanced Structural Transfer
SCP：Safe Communication Projection
```

12 轮 `alpha=0.5, gamma=0.9, seed=0`：

| 方法 | Avg ↑ | Worst ↑ | WCCA ↑ | CFG ↓ |
|---|---:|---:|---:|---:|
| local control | 37.5813 | 30.1100 | 13.700 | 10.855 |
| Oracle BER+CDep local | **41.6206** | **35.5175** | **14.000** | **6.155** |
| learned PEW BER+CDep（旧校准） | 40.3694 | 35.4225 | 13.925 | 6.370 |
| calibrated PEW BER+CDep local | **42.8469** | **36.2300** | **19.775** | **6.5725** |

解释边界：

```text
1. Oracle BER+CDep 相对 control 有明显正收益；
2. learned PEW 保留了大部分 Oracle 收益；
3. PEW 最佳 epoch 恢复和 threshold 校准进一步改善了结果；
4. 目前只能证明 PEW+BER+CDep 组合有效，不能证明 BER 和 CDep 单独有效；
5. 这些模块使用固定环境体系，因此只能视为机制证据/环境信息上界，而非最终方法。
```

### 6.3 EBST 旧通信：负结果

| 方法 | Avg ↑ | Worst ↑ | WCCA ↑ | CFG ↓ |
|---|---:|---:|---:|---:|
| Oracle BER+CDep local | 41.6206 | 35.5175 | 14.000 | 6.155 |
| + EBST + SCP | 38.7038 | 34.7225 | 15.325 | 6.415 |

EBST 导致 `Avg -2.9169`，属于明确的通信负迁移。

### 6.4 Oracle EBST-v2：安全改善但平均收益不稳定

| 方法 | Avg ↑ | Worst ↑ | WCCA ↑ | CFG ↓ |
|---|---:|---:|---:|---:|
| Oracle BER+CDep local | 41.6206 | 35.5175 | 14.000 | 6.155 |
| + EBST-v2 + class-wise SCP | 41.9469 | 36.2275 | 14.700 | 5.190 |

最终轮看似改善，但最近五轮 Avg 相对 local 为 `-0.1648`。因此只能说安全修正减少了旧 EBST 的崩溃，不能证明稳定通信收益。

### 6.5 最关键的校准 PEW 通信归因实验

严格匹配的 12 轮 A/B：

| 方法 | Final Avg ↑ | Final Worst ↑ | Final WCCA ↑ | Final CFG ↓ |
|---|---:|---:|---:|---:|
| calibrated PEW + BER+CDep local | **42.8469** | **36.2300** | 19.775 | **6.5725** |
| calibrated PEW + BER+CDep + EBST-v2 | 42.6331 | 35.2975 | **20.675** | 7.2900 |
| 通信差值 | -0.2138 | -0.9325 | +0.900 | +0.7175（变差） |

最近五轮均值：

| 方法 | Avg ↑ | Worst ↑ | WCCA ↑ | CFG ↓ |
|---|---:|---:|---:|---:|
| calibrated local | 40.4278 | **36.2890** | **17.965** | **6.427** |
| calibrated + EBST-v2 | **40.4526** | 35.9870 | 17.400 | 6.666 |
| 通信差值 | +0.0249 | -0.3020 | -0.565 | +0.239（变差） |

逐客户端最终差值：

```text
client 0: -0.9325
client 1: -0.6050
client 2: +1.0825
client 3: -0.4000
```

结论：EBST-v2 只帮助一个客户端，却伤害三个客户端；平均收益基本为零。此前完整组合的提升主要来自 PEW 校准和 BER+CDep 本地训练，不来自通信。

---

## 7. 固定损坏依赖代码审计

### 7.1 方法级高风险依赖

| 模块 | 当前实现 | 固定损坏依赖 | 当前定位 |
|---|---|---|---|
| PEW | 分类 `clean/noise/blur/weather/digital/unknown` | **直接依赖** | 诊断/Oracle 近似 |
| BER | 按 `class × environment_id` 平衡风险 | **直接依赖 PEW 环境 ID** | 机制证据 |
| CDep | 约束类内特征与 one-hot 环境或 PEW embedding 的依赖 | **当前间接依赖固定 PEW** | 数学形式可改，当前实现非最终 |
| EBST | 统计 `class × environment × competing class` 关系 | **直接依赖** | 负结果，归档 |
| EBST-v2 | 在 EBST 上增加来源资格、LOO、gate、class-wise SCP | **直接依赖** | 归因失败，归档 |

PEW 的固定环境定义：

```text
PEW_ENVIRONMENT_NAMES = clean + four corruption groups + unknown
```

因此，PEW -> BER/CDep -> EBST 是一整条闭集环境链，而不是只有 PEW 一个模块存在问题。

### 7.2 不依赖固定环境标签的部分

| 模块 | 依赖情况 | 说明 |
|---|---|---|
| JSD | 不依赖损坏类别 | 只约束多个视图的预测一致性 |
| DCL | 不依赖损坏类别 | 使用 clean/strong/weak 特征与类别标签 |
| SCP | 不依赖损坏类别 | 只判断本地梯度和通信梯度是否冲突 |
| WCCA/CFG | 测试时使用环境标签 | 可以作为评价，不应参与训练或路由 |

### 7.3 AugMix 的边界

AugMix 使用有限的增强操作集合，例如：

```text
autocontrast/equalize/posterize/rotate/solarize/shear/translate
以及可选 color/contrast/brightness/sharpness
```

它不读取 CLE 的四个 corruption group，也不把样本分类为某种环境，所以不属于 taxonomy leakage。但它仍然具有“训练增强集合有限”的通用鲁棒性假设，不能据此声称对任意未知损坏都有理论保证。

---

## 8. 当前已经成立和没有成立的结论

### 8.1 已有实验支持

```text
1. 在当前受控 CLE-HFL 协议下，gamma 增大时 RAHFL 单调退化；
2. Avg/Worst 不能完整揭示 shortcut，WCCA/CFG 提供了额外诊断；
3. 有正确环境信息时，BER+CDep local 能明显改善 Avg/Worst/CFG；
4. learned/calibrated PEW 能保留这部分本地收益；
5. EBST 会造成负迁移；
6. EBST-v2 虽更安全，但没有稳定正通信收益。
```

### 8.2 尚未成立

```text
1. 还没有一个不读取固定 corruption taxonomy 的最终主方法；
2. 还没有一个稳定优于 matching local-only 的新通信模块；
3. 还没有在 CLE-HFL 40 轮上超过 RAHFL 的最终框架；
4. 还没有多 seed、真实数据集或 held-out corruption family 证据；
5. PEW 的 unknown 机制没有证明可泛化到真正未见的损坏；
6. 当前方法不能支持“任意未知损坏”的强表述。
```

---

## 9. 主要研究困难与审稿攻击点

### 9.1 四组 corruption 的人为性

当前 benchmark 和 PEW 都使用 noise/blur/weather/digital。作为评价协议可以解释为可复现分组，但如果算法也读取这些组，容易被认为针对 benchmark 定制。

### 9.2 unknown 的真实性

当前 unknown 主要由预设 corruption 组合构造，并非真正来自未见现实损坏。它不能自动支持 open-world robustness 的强主张。

### 9.3 跨域公共数据

公共数据是 CIFAR-100，私有任务是 CIFAR-10。历史 D2C、PAIR、FedCLEAR 等结果均表明，跨域 public logits 可能不携带可靠的任务类别语义。

### 9.4 shortcut 的不可识别性

只观察单一客户端数据时，算法无法无条件判断某个相关性是语义还是 shortcut。必须额外利用以下至少一种信息：

```text
环境标签、人工干预、公共反事实数据、跨客户端变化或结构先验。
```

任何新方法都必须明确自己使用了哪一种，而不能声称无假设识别。

### 9.5 数据异构与知识缺失

如果所有客户端都没有学到某一类别的有效知识，通信不能凭空创造该类语义。当前目标应是正常到较强 label-skew 下减少负迁移，而不是保证完全补全全网缺失类别。

### 9.6 RAHFL 路由公平性

原始 RAHFL 源码使用测试准确率选择通信方向，存在测试信息用于训练决策的风险。正式论文应同时报告原始复现版本和无测试泄漏版本，避免只靠该问题削弱对手。

---

## 10. 下一通信方向的硬约束

新的通信机制必须满足：

```text
1. 不读取 noise/blur/weather/digital/unknown 标签；
2. 不使用测试标签或测试准确率路由；
3. 不依赖模型参数或特征维度一致；
4. 不直接上传原始 class count；
5. 必须明确利用 CLE-HFL 的结构，而不是普通加权平均；
6. 必须先在离线诊断中预测正/负迁移，再消耗 GPU；
7. 相对 matching local-only 必须获得通信净收益。
```

---

## 11. 值得排重的候选研究假设：跨客户端类别边界共识

这不是已经确定的方法，只是当前最值得做文献排重和离线证伪的方向。

### 11.1 动机

CLE-HFL 中，不同客户端的类别-损坏绑定 \(\phi_k\) 不同：

```text
共享语义边界应在多个客户端之间相对一致；
客户端特有 shortcut 应造成类别边界关系在客户端之间波动。
```

### 11.2 候选通信对象

客户端 (k) 在本地真实任务数据上计算类别对 margin：

\[
M_{k,c,j}
=
\mathbb E_{x:y=c}
\left[
\bar z_{k,c}(x)-\bar z_{k,j}(x)
\right].
\]

它是 (C\times C) 的模型无关决策关系，不是特征原型，也不需要公共 CIFAR-100。

服务器计算稳健共识和跨客户端方差：

\[
\mu_{c,j}=\operatorname{RobustMean}_k(M_{k,c,j}),
\qquad
v_{c,j}=\operatorname{Var}_k(M_{k,c,j}).
\]

候选可信度：

\[
q_{c,j}=\exp(-v_{c,j}/\tau).
\]

只传递跨客户端一致的类别边界；高度不一致的边界被视为可能受到 label-skew、架构差异或 corruption-label shortcut 干扰。

客户端候选对齐损失：

\[
\mathcal L_{comm}
=
\sum_{(x,y)}\sum_{j\ne y}
q_{y,j}
\operatorname{SmoothL1}
\left(
m_i(x,y,j),\mu_{y,j}
\right).
\]

### 11.3 为什么还不能直接实现

```text
1. 可能与 class-relation distillation、classifier geometry、FedProto 派生工作相近；
2. 高跨客户端方差也可能来自模型架构异构，而不只是 shortcut；
3. label-skew 下缺失类没有可靠 margin；
4. 上传每客户端关系矩阵可能泄露类别支持信息；
5. 需要研究 secure aggregation、clipping 或噪声机制；
6. 必须先验证 margin variance 能否预测已有实验中的类别级负迁移。
```

因此，下一步应先做文献排重和已有 checkpoint 离线诊断，而不是直接编码并跑 40 轮。

---

## 12. 希望外部评审重点回答的问题

请不要只给一个新名字或模块排列组合，重点回答：

1. **场景价值**：CLE-HFL 是否构成有意义的新问题？它与 spurious correlation、domain generalization、corruption robustness FL 的区别是什么？
2. **协议风险**：四组 corruption 作为 benchmark 是否可接受？还需要怎样的 held-out/真实数据协议？
3. **可识别性**：在不提供 corruption group 的前提下，哪些假设足以识别 corruption-label shortcut？
4. **文献排重**：是否已有方法通信 class-pair margin、classifier geometry 或跨客户端关系矩阵？最近和最相似的工作是什么？
5. **通信方向**：跨客户端 margin 共识能否从架构差异、label-skew 和 shortcut 中分离共享语义？必要条件是什么？
6. **隐私**：如何在不暴露单客户端 class count/支持类集合的情况下估计跨客户端均值与方差？
7. **替代方案**：如果 margin 共识不可行，请提出一个不使用固定 corruption taxonomy、不依赖同域公共数据、支持模型异构的通信机制。
8. **最小证伪实验**：在不跑完整训练的情况下，如何使用已有 checkpoint 判断新通信是否可能有效？
9. **投稿判断**：若最终只在 CLE-HFL 上提升约 2--3 点，还需要哪些数据集、理论、消融和多 seed 才可能达到 CCF-B 会议标准？

---

## 13. 当前冻结决策

```text
[保留] CLE-HFL 问题、gamma 协议、WCCA/CFG 作为当前研究对象和评价工具
[保留] AugMix/JSD/DCL 作为强本地基座，但不声称其为原创
[保留] SCP 作为通用安全组件，不作为核心创新
[诊断] PEW + BER + CDep：证明环境信息有用，但当前硬 taxonomy 版本非最终方法
[归档] FedCLEAR v0.1、D2C、PAIR、PRAC、EBST、EBST-v2 等负路线
[停止] CRST 公共损坏响应子空间方案，不进入实现
[待排重] 不依赖 public data 的跨客户端类别边界共识通信
[禁止] 在理论和离线信号明确前继续 40 轮盲跑
```

---

## 14. 相关项目文件

```text
AGENTS.md
docs/project/CURRENT_PROJECT_MEMORY.md
docs/project/PROJECT_STATE.md
docs/project/TODO_NEXT.md
docs/research/status/RECENT_PROGRESS_REPORT_2026_07_20_ZH.md
docs/archive/methods/FEDCLEAR_CLE_HFL_PROPOSAL_ZH.md
docs/archive/methods/FEDEASE_V2_1_FRAMEWORK_AND_IMPLEMENTATION_ZH.md
```

本文件只用于外部评审，不代表第 11 节候选方法已经成立或已经成为项目最终主线。
