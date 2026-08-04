# FedCLEAR/CLE-HFL 当前研究现状与方案审核材料

更新时间：2026-07-11  
用途：提交给外部 AI 或研究人员，重新审核问题设定、实验结果和下一步方法。  
当前目标：在新的 CLE-HFL 场景下，相对严格匹配的 RAHFL 基线将最终平均准确率提高约 3 个百分点，同时改善最差客户端与最差类别-损坏组合性能。

---

## 0. 希望审核者回答的问题

请不要默认认可当前方案，也不要只给出泛化的“可以试试”。希望重点回答：

1. CLE-HFL 是否是一个独立、合理、可发表的问题，而不是人为制造 RAHFL 的弱点？
2. 当前证据是否足以说明 `corruption-label entanglement` 确实构成了 RAHFL 的系统性失败模式？
3. 将公共数据从跨域 CIFAR-100 换成同域无标签 CIFAR-10，是否构成过强假设或主要攻击点？
4. 如果不允许使用同域公共数据，怎样在模型异构条件下进行有效通信？
5. PCCD 的理论链条是否真的针对 CLE-HFL，还是把一致性蒸馏重新包装了一遍？
6. 哪些设计需要保留、删除或替换，才能形成一套理论明确且有机会稳定超过 RAHFL 约 3 个点的方法？
7. 最小但有判别力的下一组实验应该是什么？

---

## 1. 研究目标与硬约束

目标场景同时包含：

```text
模型异构 + 标签分布异构 + 数据损坏 + 损坏与标签的虚假关联
```

客户端模型分别为：

```text
ResNet10 / ResNet12 / ShuffleNet / MobileNetV2
```

因此不能直接平均模型参数，也不能假设不同模型具有同维度特征。当前通信路线只能使用共享的十类输出概率空间，或提出另一种真正与架构无关的知识载体。

希望同时满足：

1. 不上传私有图片；
2. 不上传私有标签或逐类样本数；
3. 不使用测试标签选择教师；
4. 不依赖模型参数同构；
5. 不能只在极端 Non-IID 下有效；
6. 不能以平均准确率下降换取单个诊断指标改善；
7. 正式结论需要多随机种子，而不是依赖单次最好结果。

---

## 2. 新问题：CLE-HFL

CLE-HFL 表示：

```text
Corruption-Label Entanglement in Heterogeneous Federated Learning
异构联邦学习中的损坏-标签纠缠
```

### 2.1 它和普通数据损坏有什么区别

普通 corruption robustness 通常假设损坏与标签基本独立。例如猫、狗、汽车都有可能遇到 noise、blur 或 compression。

CLE-HFL 研究的是客户端内部出现条件依赖：

```text
某客户端的猫主要伴随 noise；
某客户端的狗主要伴随 blur；
另一个客户端的猫可能主要伴随 digital corruption。
```

模型可能把损坏模式当成类别捷径：

```text
noise -> cat
blur  -> dog
```

部署时，如果同一类别出现在另一种损坏下，或者同一种损坏与另一类别组合，预测便会显著下降。

### 2.2 协议中的两个异构维度

第一层是标签数量偏斜：

\[
P_k(Y) \sim \operatorname{Dirichlet}(\alpha).
\]

目前主要诊断设置为 `alpha=0.5`。

第二层是类别条件损坏偏斜。对客户端 `k` 和类别 `c`，指定客户端特有的主导损坏组 `\phi_k(c)`：

\[
P_k(G=g\mid Y=c)
=
\gamma\,\mathbf 1[g=\phi_k(c)]
+
(1-\gamma)\operatorname{Uniform}(G).
\]

其中：

```text
gamma=0.0：损坏基本不与类别绑定；
gamma=0.6：中等纠缠；
gamma=0.9：强纠缠。
```

不同 gamma 数据集保持原始 CIFAR-10 图片、Dirichlet 私有划分、模型、训练预算和测试协议一致，只改变 `P_k(G|Y)`。

### 2.3 当前使用的损坏家族

当前协议将常见损坏算子归入四个大类：

```text
noise / blur / weather / digital
```

它们是损坏家族，不是只有四个具体算子。尽管如此，该设计仍有潜在攻击点：

1. 四个家族是否足够覆盖现实中的 shortcut？
2. 方法是否硬编码这些家族？
3. 在未见过的 corruption 上是否仍有效？
4. 如果公共干预也只使用相同家族，提升是否来自记住协议？

正式方法必须使用可扩展、标签无关的干预算子池，并用未参与训练的损坏进行测试。

---

## 3. 评价指标

### 3.1 平均与最差客户端准确率

```text
avg_acc：四个异构客户端测试准确率的平均值；
worst_acc：四个客户端中最低的测试准确率。
```

### 3.2 WCCA

Worst Class-Corruption Accuracy：

\[
\operatorname{WCCA}
=
\min_{c,g}\operatorname{Acc}(c,g).
\]

它寻找所有“类别 × 损坏”组合中最差的一组。例如：

```text
cat + noise       = 70%
cat + blur        = 65%
cat + weather     = 30%
cat + digital     = 68%
...
```

如果全表最低值为 12%，则 WCCA 为 12%。它用于发现平均准确率掩盖的局部崩溃，越高越好。

### 3.3 CFG

Counterfactual Gap：

\[
\operatorname{CFG}
=
\frac{1}{C}
\sum_c
\left[
\max_g\operatorname{Acc}(c,g)
-
\min_g\operatorname{Acc}(c,g)
\right].
\]

它衡量同一类别换一种损坏后性能变化多大。CFG 越大，说明模型越可能依赖类别相关的损坏捷径；越低越好。

WCCA/CFG 的改善不能以 `avg_acc` 或 `worst_acc` 崩溃为代价。

---

## 4. RAHFL 基线与场景证据

RAHFL 由以下部分组成：

```text
AugMix + CE/JSD + DCL + AsymHFL
```

- AugMix/JSD：本地多视图预测一致性；
- DCL：本地干净、弱增强、强增强特征之间的判别式对比约束；
- AsymHFL：在公共数据上进行异构客户端之间的非对称知识蒸馏。

在 `alpha=0.5, seed=0`、相同模型和训练预算下，RAHFL 的 CLE-HFL 结果为：

| gamma | avg_acc | worst_acc | WCCA | CFG |
| ---: | ---: | ---: | ---: | ---: |
| 0.0 | 52.17 | 44.17 | 35.35 | 2.54 |
| 0.6 | 50.82 | 42.83 | 25.88 | 5.91 |
| 0.9 | 46.72 | 38.16 | 19.32 | 10.91 |

从 `gamma=0.0` 到 `gamma=0.9`：

```text
avg_acc   -5.45
worst_acc -6.01
WCCA     -16.02
CFG       +8.37
```

### 4.1 当前可以支持的结论

当 corruption-label entanglement 增强时，RAHFL 的整体性能、最差客户端性能和最差类别-损坏性能均系统下降，而同类跨损坏差距明显扩大。

这说明 CLE-HFL 至少是一个可测量的失败模式，且 RAHFL 的 AugMix/JSD/DCL/AsymHFL 没有自动解决它。

### 4.2 当前不能宣称的结论

1. 不能仅凭一组数据、一个 seed 声称这是普遍现实规律；
2. 不能证明四个损坏家族覆盖全部现实 corruption；
3. 不能证明 RAHFL 的退化只来自 shortcut，而完全没有普通难度增加的贡献；
4. 不能在没有其他 spurious-correlation/Federated-DG 基线时声称全面领先该领域。

---

## 5. 上一版 FedCLEAR v0.1：CCRE + IRD

上一版方法试图同时改造本地训练和通信：

```text
FedCLEAR v0.1 = CCRE + IRD
```

### 5.1 CCRE

CCRE 在私有样本上产生多个标签无关损坏视图，使用类别均衡的最坏环境风险，尝试防止 head class 和容易损坏视图主导训练。

它希望直接提高 WCCA，并降低同一类别在不同损坏上的性能差距。

### 5.2 IRD

IRD 仍使用跨域 CIFAR-100 公共图片。各客户端对同一公共图片的多个干预视图进行预测，提取所谓不变响应，再通过跨客户端中位数/残差方式构造教师并蒸馏。

### 5.3 完整 gamma=0.9 结果

| 方法 | avg_acc | worst_acc | WCCA | CFG |
| --- | ---: | ---: | ---: | ---: |
| RAHFL | **46.72** | **38.16** | **19.32** | **10.91** |
| FedCLEAR v0.1 | 45.41 | 36.42 | 17.80 | 11.42 |
| 差值（FedCLEAR - RAHFL） | -1.31 | -1.74 | -1.52 | +0.51 |

四个核心指标全部没有超过 RAHFL，因此这是明确的负结果。

### 5.4 失败诊断

1. CCRE 的代理损失下降，但私有样本本身已经带有原始 corruption；在其上再次施加增强，并没有生成真正“去掉原 shortcut”的反事实样本。
2. IRD 使用 CIFAR-100 作为 CIFAR-10 模型的公共载体。客户端很可能是在跨域图片上共同无知，而不是共享可靠 CIFAR-10 语义。
3. IRD 后期 anchor disagreement 的最后十轮均值约为 `0.891`，说明教师共识并不可靠。
4. 最坏风险目标可能过度正则化，损害了普通分类边界。
5. 同时替换本地训练和通信，使负结果难以准确归因。

结论是：冻结 CCRE+IRD，不继续靠调权重挽救。

---

## 6. 当前候选框架：FedCLEAR-PCCD

当前方案不是已经获得正结果的方法，而是**已实现、尚未进行正式效果验证的候选通信框架**。

### 6.1 控制变量设计

固定两边本地训练完全一致：

```text
RAHFL：        AugMix + CE + JSD + DCL + AsymHFL
FedCLEAR-PCCD：AugMix + CE + JSD + DCL + PCCD
```

模型、私有数据、公共数据、优化器、batch、round 和 seed 均一致，唯一方法变量为 `AsymHFL -> PCCD`。

这只能证明在给定公共数据假设下 PCCD 是否优于 AsymHFL，不能自动证明整个系统比原论文设定更弱假设。

### 6.2 PCCD 的输入

对每张无标签公共图片 `u`，生成同内容的多个标签保持视图：

\[
u,\;T_1(u),\ldots,T_G(u).
\]

当前候选算子池包括：

```text
identity / gaussian noise / blur / brightness /
contrast / pixelation / haze
```

### 6.3 客户端跨视图共识

客户端 `k` 对公共样本 `u` 的视图 `g` 输出：

\[
p_{k,g}(c\mid u)=\operatorname{softmax}(z_{k,g}(u))_c.
\]

使用 log-opinion pooling：

\[
s_{k,c}(u)
=
\frac{1}{G+1}
\sum_{g=0}^{G}
\log(p_{k,g}(c\mid u)+\epsilon),
\]

\[
q_k(c\mid u)
=
\frac{\exp(s_{k,c}(u))}
{\sum_j\exp(s_{k,j}(u))}.
\]

直觉是：某类别只有在该公共内容的多个损坏视图上都得到支持，才会在 `q_k` 中保持高概率；仅由单个损坏触发的预测会被几何平均压低。

### 6.4 连续置信度

\[
r_k(u)
=
1-\frac{H(q_k(\cdot\mid u))}{\log C}.
\]

低熵、跨视图一致的客户端获得更高权重；接近均匀的共同无知获得较小权重。这里没有手工置信度阈值。

### 6.5 Leave-one-out 教师

对接收客户端 `i`：

\[
q_{-i}(c\mid u)
=
\frac{\sum_{k\neq i}r_k(u)q_k(c\mid u)}
{\sum_{k\neq i}r_k(u)+\epsilon}.
\]

教师本身的权重为：

\[
m_{-i}(u)
=
1-\frac{H(q_{-i}(\cdot\mid u))}{\log C}.
\]

这样每张公共样本都有自己的教师，不再用一个客户端的整体准确率决定它是否教所有样本，也避免把接收客户端自己的预测放回教师。

### 6.6 成对反事实蒸馏

\[
\mathcal L_{PCCD}^{(i)}
=
\frac{1}{B(G+1)}
\sum_{u\in B_{pub}}
m_{-i}(u)
\sum_{g=0}^{G}
\operatorname{KL}
\left(
q_{-i}(\cdot\mid u)
\parallel
p_{i,g}(\cdot\mid u)
\right).
\]

希望同时实现：

1. 从其他架构客户端接收共享类别概率知识；
2. 同一内容在不同损坏视图下靠近同一跨客户端教师；
3. 不确定教师自动减小梯度；
4. 不上传模型参数、特征或私有类别数量。

### 6.7 条件性理论假设

PCCD 依赖：

1. 公共干预保持真实类别；
2. 公共内容与私有任务共享语义；
3. 不同客户端 shortcut 不完全同向；
4. 对足够多公共样本，跨客户端弱多数具有正确正 margin；
5. 异构模型共享相同类别输出坐标。

如果所有客户端都稳定地预测错误，PCCD 只会形成错误共识。它不可能无条件保证提高 3 个点。

---

## 7. 最大争议：同域无标签 CIFAR-10 public

### 7.1 当前拟议划分

CIFAR-10 train 共 50,000 张：

```text
private：  40,000 = 4 clients x 10,000
public：    5,000，无标签
reserved：  5,000，不参与训练
test：     独立 CIFAR-10 test 10,000，只评价
```

public 只从 private 40,000 张的索引补集中抽取，因此 private/public/test 三者不重叠。训练 loader 不读取 public 标签。

### 7.2 为什么会提出这个改变

旧方案使用 CIFAR-100 公共图片，但实验显示跨域预测可能只是共同无知，无法可靠承载 CIFAR-10 类别语义。PCCD 要在同一语义内容的多种 corruption views 上消除客户端特有 shortcut，因此提出使用同任务域无标签公共池。

### 7.3 为什么它仍是主要攻击点

即使没有样本泄漏和标签泄漏，这仍然加强了信息条件：

1. 原始 RAHFL 使用跨域 CIFAR-100，而新方法获得了同域 CIFAR-10 内容；
2. 性能提升可能来自公共数据语义更接近任务，而不是 PCCD；
3. 某些真实联邦场景未必能获得 5,000 张同域公共图片；
4. 5,000/40,000 的公共-私有比例并不算极小；
5. 如果论文把它隐藏成普通工程细节，审稿人很容易认为比较不公。

### 7.4 两种“公平”不能混淆

内部模块公平：

```text
RAHFL 和 PCCD 都使用相同 CIFAR-10 public 5k，
因此可以判断 PCCD 是否优于 AsymHFL。
```

方法假设公平：

```text
原始 RAHFL 只需要跨域 CIFAR-100，
PCCD 需要同域 CIFAR-10，
因此不能仅凭前述 A/B 宣称在相同数据假设下全面击败原始 RAHFL。
```

正式论文至少需要以下对照：

| 公共载体 | RAHFL | PCCD | 回答的问题 |
| --- | --- | --- | --- |
| CIFAR-100 跨域 | 是 | 是 | 相同弱公共假设下，PCCD 是否仍有效 |
| CIFAR-10 同域无标签 | 是 | 是 | 相同强公共假设下，通信模块谁更强 |
| 无公共数据或合成 carrier | 可选 | 可选 | 方法对 public data 的依赖程度 |

如果 PCCD 只在 CIFAR-10 public 上提高，而 RAHFL 也因同域 public 得到相同幅度提升，则 PCCD 的核心贡献不成立。

---

## 8. 当前尚未运行的 PCCD 最小验证

已准备两份完全匹配的 12-round probe：

```text
A：AugMix + CE + JSD + DCL + AsymHFL
B：AugMix + CE + JSD + DCL + PCCD

alpha=0.5
gamma=0.9
seed=0
public=CIFAR-10 unlabeled 5k
```

它们只能作为筛选实验。进入 40 轮前，PCCD 在第 12 轮附近应同时达到：

```text
avg_acc delta    >= +1.5
worst_acc delta  >= +1.0
WCCA delta       >= +4.0
CFG delta        <= -1.5
```

正式期望为：

```text
avg_acc    +3.0
worst_acc  +2.0
WCCA       +4.0
CFG        -2.0
```

但由于同域 public 假设存在争议，在执行实验前应先决定：

1. 是否接受“少量同域无标签公共数据”作为论文场景假设；
2. 是否将 public size 降低到 500/1000 并画规模曲线；
3. 是否必须先跑 CIFAR-100 版本，避免方案建立在更强数据条件上；
4. 是否改为不需要自然同域图片的共享合成 carrier；
5. 是否放弃 public-logit 路线，改用另一种模型无关通信对象。

---

## 9. 当前研究判断

### 已经相对可靠的部分

1. CLE-HFL 的 gamma 协议能稳定制造可测量的 corruption-label shortcut；
2. RAHFL 随 gamma 增大出现单调且明显的退化；
3. WCCA 和 CFG 比单独平均准确率更能揭示失败；
4. FedCLEAR v0.1 没有效果，不能继续包装成正结果；
5. 跨域 CIFAR-100 public logits 的语义承载能力是实际瓶颈之一。

### 尚未成立的部分

1. PCCD 尚无正式实验结果；
2. 尚未证明同域 public 的收益不是数据条件收益；
3. 尚未证明 PCCD 能在 CIFAR-100 或未见 corruption 下有效；
4. 尚未达到相对 RAHFL `+3 avg_acc`；
5. 尚未完成多 seed、不同 alpha/gamma 与更多数据集验证；
6. 尚未完成正式文献排重，不能宣称 PCCD 绝对首创。

---

## 10. 希望外部 AI 给出的最终建议格式

请按以下顺序给出明确判断：

1. **问题判断**：CLE-HFL 是否值得继续，最大场景漏洞是什么？
2. **公共数据判断**：是否应允许同域无标签 public；若允许，怎样把假设降到审稿可接受？
3. **方法判断**：PCCD 各公式中哪些合理，哪些可能无效或与已有方法高度相似？
4. **替代方案**：如果不能使用同域 public，给出一套模型异构可用、且真正对应 CLE-HFL 的通信机制。
5. **理论链条**：从 CLE-HFL 的生成假设，到优化目标，再到 WCCA/CFG 的可验证关系。
6. **最小实验**：在最多约 7 小时 V100 预算下，先跑哪两个实验最能决定方向生死？
7. **成功标准**：什么结果才足以继续投入 40 轮、多 seed 和正式论文实验？
8. **投稿风险**：即使达到 `avg_acc +3`，还需要哪些实验和贡献才能支撑 CCF-B 级会议投稿？

请避免提出大量无依据模块的排列组合。优先选择假设更弱、变量更少、能被消融直接验证的方案。

