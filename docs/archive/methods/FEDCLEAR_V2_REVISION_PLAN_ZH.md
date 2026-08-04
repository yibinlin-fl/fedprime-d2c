# FedCLEAR v2 修改方案：成对反事实共识蒸馏

> 状态说明：本文件保留为第一版设计记录。最新冻结框架与理论推导请阅读 `docs/archive/methods/FEDCLEAR_LATEST_THEORY_FRAMEWORK_ZH.md`。

工作名称：FedCLEAR-v2 / PCCD

英文模块名：Paired Counterfactual Consensus Distillation

适用场景：CLE-HFL（Corruption-Label Entanglement in Heterogeneous Federated Learning）

状态：方法设计与实验决策稿，尚未编码

---

## 1. 先说结论

当前 FedCLEAR v0.1 不应该继续通过调 `lambda_ccre`、IRD temperature 或增加训练轮数来硬救。

它的主要问题不是没有收敛，而是三个理论假设被完整实验否定：

1. 在已经损坏的私有图片上叠加新损坏，并不等价于替换原始 corruption；
2. `CE + 12*JSD + worst-view CCRE` 容易用判别能力换取局部最坏风险；
3. 跨域 CIFAR-100 上的 standardized-logit median 不是可靠的 CIFAR-10 语义教师。

因此 v2 不继续堆补丁，而做三项结构性修改：

```text
修改 1：恢复 RAHFL 的强本地判别基座
修改 2：只在干净、无标签、同任务域公共样本上构造真正成对的损坏视图
修改 3：用样本级、leave-one-out 的不变共识教师替换 IRD median teacher
```

最终结构：

```text
FedCLEAR-v2
  = AugMix + JSD + DCL
  + Paired Counterfactual Consensus Distillation (PCCD)
```

这里不把 AugMix、JSD、DCL 作为创新点。它们是经过实验验证的强鲁棒本地基座。

论文贡献应集中在：

1. CLE-HFL 新问题和可控协议；
2. WCCA、CFG 等专用评价指标；
3. 面向模型异构与 corruption-label shortcut 的 PCCD 通信机制。

---

## 2. 当前实验告诉了我们什么

### 2.1 RAHFL 在 CLE-HFL 下存在明确短板

固定 `alpha=0.5, seed=0`：

| gamma | avg_acc | worst_acc | WCCA | CFG |
| ---: | ---: | ---: | ---: | ---: |
| 0.0 | 52.17 | 44.17 | 35.35 | 2.54 |
| 0.6 | 50.82 | 42.83 | 25.88 | 5.91 |
| 0.9 | 46.72 | 38.16 | 19.32 | 10.91 |

从 `gamma=0.0` 到 `0.9`：

```text
avg_acc  -5.45
worst_acc -6.01
WCCA     -16.02
CFG       +8.37（越低越好）
```

这说明 CLE-HFL 场景本身成立：RAHFL 能抵抗普通 corruption，但无法阻止类别和损坏之间的客户端特有 shortcut。

### 2.2 FedCLEAR v0.1 没有解决问题

`alpha=0.5, gamma=0.9, seed=0, 40 rounds`：

| 指标 | RAHFL | FedCLEAR v0.1 | 差值 |
| --- | ---: | ---: | ---: |
| avg_acc | 46.72 | 45.41 | -1.31 |
| worst_acc | 38.16 | 36.42 | -1.74 |
| WCCA | 19.32 | 17.80 | -1.52 |
| CFG | 10.91 | 11.42 | +0.51 |

但 v0.1 在 34/40 轮的 WCCA 高于 RAHFL，后 10 轮 WCCA 均值高约 2.75。

这说明“关注最差 class-corruption 单元”不是完全错误；错误在于实现方式牺牲了判别性，而且 IRD 没有形成可靠教师。

---

## 3. 为什么不继续使用私有数据 CCRE

设客户端私有观测为：

\[
x^{obs}=T_{g_k(y)}(x^{clean})
\]

其中 `g_k(y)` 是客户端 `k` 上与类别 `y` 纠缠的损坏。

当前 v0.1 对 `x_obs` 再应用随机算子 `T_h`：

\[
\tilde{x}=T_h(x^{obs})=T_h(T_{g_k(y)}(x^{clean}))
\]

这并没有移除 `g_k(y)`。

例如：

```text
原观测：cat + noise
新视图1：cat + noise
新视图2：cat + noise + blur
新视图3：cat + noise + brightness
```

所有视图都保留 `noise -> cat`，所以 JSD 和 CCRE 即使降到很低，也不能证明 shortcut 被消除。

因此 v2 首版应移除私有 CCRE，不再把“在已损坏私有图片上叠加增强”称为 counterfactual intervention。

---

## 4. 新的数据假设

### 4.1 无标签同任务域公共池

从 CIFAR-10 训练集预留一个与所有客户端私有集、验证集和测试集互斥的无标签公共池：

\[
D_{pub}=\{u_b\}_{b=1}^{B_{pub}}
\]

建议：

```text
CIFAR-10 train 50,000
  40,000：4 个客户端私有训练集，每个 10,000
   5,000：无标签公共池
   5,000：服务器/协议验证池或保留不用

CIFAR-10 test 10,000
  只用于最终评价和每轮日志，不允许参与 teacher 路由
```

正式实现前必须审计当前 CLE-HFL 数据索引，确保三部分没有样本重叠。

### 4.2 为什么公共池必须同任务域

PCCD 要传播的是 CIFAR-10 类别语义。CIFAR-100 公共图像与任务类别不一致，客户端在其上的预测可能只是共同无知。

同任务域不等于使用公共标签：

```text
服务器和客户端只看到公共图片，不使用其标签。
```

该假设比“完全不需要公共数据”更强，但 RAHFL 本身已经依赖公共数据。论文必须明确声明：

> 本方法面向存在小规模、无标签、任务相关公共代理池的模型异构联邦学习。

所有依赖公共数据的基线必须获得完全相同的公共池，不能只让 FedCLEAR-v2 使用。

---

## 5. PCCD 核心方法

## 5.1 成对损坏视图

对于同一个干净公共样本 `u`，服务器构造：

\[
\mathcal{V}(u)=\{u^{(0)},u^{(1)},\ldots,u^{(G)}\}
\]

其中：

\[
u^{(0)}=u,\qquad u^{(g)}=T_g(u)
\]

这些视图共享完全相同的内容，只改变环境损坏，因此才是可信的 paired intervention。

训练算子不应硬编码为 evaluation 中的四个 group。建议使用更宽的标签无关算子池：

```text
noise family
blur family
brightness / contrast
compression / pixelation
fog / haze
color / frequency perturbation
```

每批随机采样 2~3 个视图，测试时额外包含未参与公共训练的 corruption，用于验证不是记忆四个 group。

---

## 5.2 客户端的视图共识分布

客户端 `k` 对同一个公共样本的所有视图输出：

\[
p_{k,g}(u)=\operatorname{softmax}(z_{k,g}(u))
\]

不用 standardized logits。标准化会删除置信尺度，而 v0.1 已证明尺度被删除后形成的 median teacher 不可靠。

对同一客户端的不同视图使用 log-opinion pool：

\[
q_k(c\mid u)
=
\frac{
\exp\left(\frac{1}{G+1}\sum_{g=0}^{G}\log(p_{k,g,c}(u)+\epsilon)\right)
}{
\sum_{j=1}^{C}
\exp\left(\frac{1}{G+1}\sum_{g=0}^{G}\log(p_{k,g,j}(u)+\epsilon)\right)
}
\]

直觉：

```text
一个类别只有在所有损坏视图下都保持较高概率，才会在 q_k 中保持较高概率。
只在 noise 视图下突然变高的 shortcut 类别会被几何平均压低。
```

相比直接平均 logits，这个操作显式要求跨视图一致证据。

---

## 5.3 无额外阈值的可信度

客户端共识置信度：

\[
r_k(u)=1-\frac{H(q_k(\cdot\mid u))}{\log C}
\]

其中 `r_k(u)` 自然位于 `[0,1]`：

```text
q_k 接近均匀分布：r_k 接近 0，说明客户端不知道；
q_k 低熵且跨视图一致：r_k 接近 1。
```

不再额外设置手工 confidence threshold，减少超参数和离散行为。

---

## 5.4 Leave-one-out 样本级教师

对于接收客户端 `i`，服务器只融合其他客户端：

\[
q_{-i}(c\mid u)
=
\frac{
\sum_{k\neq i} r_k(u)q_k(c\mid u)
}{
\sum_{k\neq i}r_k(u)+\epsilon
}
\]

教师自身置信度：

\[
m_{-i}(u)=1-\frac{H(q_{-i}(\cdot\mid u))}{\log C}
\]

这里的关键变化是：

```text
RAHFL：一个客户端整体更强，就整体当 teacher；
IRD：所有客户端 standardized logits 逐坐标 median；
PCCD：对每个公共样本分别判断，只有跨损坏一致且低熵的客户端贡献更大。
```

模型异构不构成障碍，因为所有模型最终都输出同一个 `C` 维类别概率空间。

---

## 5.5 成对反事实蒸馏

接收客户端 `i` 在同一公共样本的所有视图上拟合 leave-one-out 教师：

\[
\mathcal{L}_{PCCD}^{(i)}
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
\right)
\]

这一项同时完成两件事：

1. 从其他架构客户端迁移任务语义；
2. 强制同一语义在不同 corruption 下匹配同一个教师分布。

PCCD 不需要：

```text
测试集准确率路由
private class count 上传
corruption group 标签
客户端参数或特征维度一致
standardized-logit median
oracle prior
```

---

## 6. 本地训练目标

v2 首版直接恢复已验证稳定的 RAHFL 本地训练：

\[
\mathcal{L}_{local}
=
\mathcal{L}_{CE}
+\lambda_{jsd}\mathcal{L}_{JSD}
+\lambda_{dcl}\mathcal{L}_{DCL}
\]

其中：

```text
CE：保证类别判别；
JSD：保证 AugMix 多视图预测一致；
DCL：保证 clean / weak / strong 特征具有类内紧凑和类间分离结构。
```

通信阶段：

\[
\mathcal{L}_{comm}=\lambda_p\mathcal{L}_{PCCD}
\]

首版仅保留一个新增权重：

```text
lambda_p = 1.0
```

不要同时加入 CCRE、SARA、CARA、PRAC 或额外 prototype loss。先证明 PCCD 本身是否有效。

---

## 7. 为什么这次理论上比 v0.1 更完整

### 7.1 真正改变 corruption，且保持语义不变

PCCD 的所有公共视图都从同一个未损坏公共样本生成，因此不存在：

```text
原 shortcut corruption 在所有新视图中残留
```

### 7.2 同时保留 invariance 与 discrimination

v0.1 主要加强最坏风险，却移除了 DCL。v2 用 DCL 保持判别结构，只把不变约束放到公共通信阶段。

这与领域泛化研究中“只做不变性可能降低判别能力”的观察一致。

### 7.3 不再聚合共同无知

同任务域公共数据包含真实 CIFAR-10 语义，视图共识与置信度让不稳定客户端自动弱化。

### 7.4 利用 CLE-HFL 的关键可识别性

CLE-HFL 中不同客户端的 `class -> corruption` 映射不同。

对于同一公共语义样本：

```text
各客户端的 shortcut 响应不同；
跨视图、跨客户端稳定的预测成分更可能来自语义内容。
```

PCCD 正是利用这一互补性构造教师，而不是假设任意 public consensus 都是知识。

---

## 8. 具体训练例子

假设公共池中有一张未标注的汽车图片 `u`。

服务器生成：

```text
u0 = 原图
u1 = 高斯噪声
u2 = 模糊
u3 = 亮度变化
```

三个发送客户端输出：

```text
client 0：四个视图都较稳定地预测汽车
client 1：原图和噪声预测汽车，模糊图错误预测猫
client 2：四个视图都接近均匀分布
```

经过视图 log-opinion pool：

```text
q0：汽车概率高、熵低，r0 高
q1：汽车仍占优势但被模糊视图压低，r1 中等
q2：近似均匀，r2 接近 0
```

服务器给接收客户端 3 构造：

```text
teacher ≈ q0 的汽车知识 + 少量 q1
几乎忽略 q2 的共同无知
```

客户端 3 随后让汽车原图、噪声图、模糊图、亮度图都拟合同一个 teacher。

这才对应：

```text
语义保持为汽车，损坏发生变化，预测保持稳定。
```

---

## 9. 与现有工作的区别和排重风险

### 9.1 RAHFL

RAHFL 使用 AugMix/JSD/DCL 本地鲁棒训练，并根据客户端级能力进行非对称公共 logits 学习。

PCCD 的区别：

```text
客户端级路由 -> 样本级 leave-one-out 教师
单公共视图 -> 同内容成对损坏视图
整体准确率能力 -> 跨损坏一致语义证据
测试准确率路由 -> 无测试信息通信
```

### 9.2 FedIIR

FedIIR 通过跨客户端梯度对齐学习隐式不变关系。PCCD 不交换或对齐模型梯度，而在共同类别输出空间完成 paired-view 共识，因此可直接支持模型异构。

### 9.3 FedCD

FedCD 已经研究联邦领域泛化中的 spurious correlation，并使用 feature intervener 与 risk extrapolation aggregation。

PCCD 必须强调：

```text
研究问题是 corruption-label entanglement；
客户端模型架构异构；
通信对象是同一公共内容的成对损坏响应；
不要求参数/梯度可聚合。
```

### 9.4 FedKA

FedKA 使用全局工作空间的特征分布匹配和客户端投票伪标签。PCCD 不能只写成“投票伪标签”，必须突出 log-opinion paired-view consensus 与反事实多视图蒸馏。

正式投稿前仍需做更完整的文献检索。当前只能说路线具备可区分性，不能声称从未有人研究相似思想。

---

## 10. 公平实验要求

一旦公共池从 CIFAR-100 改为无标签 CIFAR-10，旧的 RAHFL `46.72` 不能直接作为唯一公平基线。

必须运行：

```text
RAHFL + 同一个无标签 CIFAR-10 public pool
FedCLEAR-v2 + 同一个无标签 CIFAR-10 public pool
```

两者必须保持：

```text
相同私有划分
相同模型列表
相同 optimizer/lr
相同 batch size
相同 local epoch
相同 communication rounds
相同 public batch 数量
相同 seed
测试集只用于评价
```

同时建议保留旧 CIFAR-100 结果作为 cross-domain public data 补充实验，而不是主公平表。

---

## 11. 不盲跑的实验门槛

## 11.1 第一阶段：12 轮 probe

只跑：

```text
alpha=0.5
gamma=0.9
seed=0
12 rounds
```

同时运行相同公共池的 RAHFL probe。

只有同时满足以下条件，才进入 40 轮：

```text
avg_acc     至少比同轮 RAHFL 高 1.5
worst_acc   至少高 1.0
WCCA        至少高 4.0
CFG         至少低 1.5
teacher confidence 不塌缩
public view disagreement 随训练下降
```

如果 avg/worst 仍下降，即使 WCCA 上升，也停止 full run。

## 11.2 40 轮目标

用户期望超过旧 RAHFL 约 3 个平均准确率点，对应旧参考：

```text
旧 RAHFL gamma=0.9 avg_acc = 46.72
期望目标                      >= 49.72
```

但论文应以“同公共池的新 RAHFL 结果”为正式基线。

建议成功标准：

```text
avg_acc     >= matching RAHFL + 3.0
worst_acc   >= matching RAHFL + 2.0
WCCA        >= matching RAHFL + 4.0
CFG         <= matching RAHFL - 2.0
```

达到一次不等于可以投稿，还需要 seed 0/1/2、gamma 0/0.6/0.9 和 alpha 多设置验证。

---

## 12. 实现计划

### 阶段 A：数据协议

1. 审计当前 CIFAR-10 训练索引；
2. 固化私有 40k / 公共 5k / 保留 5k；
3. 公共池不保存或暴露标签给训练代码；
4. 保存 split manifest 与 SHA256；
5. RAHFL 和 FedCLEAR-v2 共用同一 public indices。

建议文件：

```text
scripts/prepare_cle_in_domain_public.py
outputs/partitions/cle_alpha05_gamma09_seed0_public5k.npz
```

### 阶段 B：PCCD 模块

建议文件：

```text
fedprime/methods/pccd.py
```

实现：

```text
paired public view generation
per-client log-opinion consensus
normalized-entropy confidence
leave-one-out teacher
paired-view KL distillation
diagnostic metrics
```

### 阶段 C：Runner 接入

建议保留统一 runner，并新增：

```text
method_name: fedclear_v2
local_module: rahfl_dcl
communication: pccd
```

不得修改旧 `rahfl`、`fedclear` 路径的行为。

### 阶段 D：必须输出的诊断

每轮至少记录：

```text
avg_acc
worst_acc
WCCA
CFG
pccd_loss
teacher_confidence_mean
teacher_entropy_mean
client_view_disagreement
leave_one_out_teacher_disagreement
```

### 阶段 E：测试

```text
1. 单元测试：概率归一化、leave-one-out、无 NaN
2. 2 轮本地 3050 smoke test
3. RAHFL 回归测试
4. 12 轮 OpenI probe
5. 通过门槛后才跑 40 轮
```

---

## 13. 风险和诚实预期

### 风险 1：RAHFL 也会从同任务域 public pool 获益

这是公平实验必须接受的。不能通过只给自己的方法更好公共数据制造 3 点优势。

### 风险 2：伪标签确认偏差

如果所有客户端都稳定地预测错误，PCCD 仍可能形成错误教师。leave-one-out 和熵权重只能缓解，不能从理论上完全消除。

### 风险 3：公共数据假设

如果论文声称完全 data-free，该方案不成立。论文必须限定在 public-data-assisted heterogeneous FL。

### 风险 4：无法保证提高 3 点

当前方法设计比 v0.1 更符合问题结构，但任何人在实验前承诺一定超过 3 点都不科学。

合理期待是：它有机会恢复 RAHFL 在 `gamma=0.9` 相对 `gamma=0` 损失的 5.45 个点中的一部分。是否能恢复 3 点，必须由 12 轮门槛先验证。

---

## 14. 当前决策

```text
[冻结] FedCLEAR v0.1 = CCRE + IRD，作为负结果
[保留] CLE-HFL 场景、gamma 协议、WCCA/CFG 指标
[恢复] AugMix + JSD + DCL 强本地基座
[替换] IRD -> PCCD
[新增] 无标签同任务域公共池与成对损坏视图
[先跑] 12 轮 matching RAHFL vs FedCLEAR-v2
[禁止] 未通过 probe 就跑 40 轮或多 seed
```

这是当前最值得实现、也最能解释为什么可能超过 RAHFL 的修改方向。

---

## 15. 相关工作入口

1. FedIIR: Out-of-Distribution Generalization of Federated Learning via Implicit Invariant Relationships, ICML 2023.
   https://proceedings.mlr.press/v202/guo23b.html
2. FedCD: Reducing Spurious Correlation for Federated Domain Generalization, arXiv 2024.
   https://arxiv.org/abs/2407.19174
3. FedKA: Feature Distribution Matching for Federated Domain Generalization, ACML 2023.
   https://proceedings.mlr.press/v189/sun23a.html
4. FIXED: Frustratingly Easy Domain Generalization with Mixup, CPAL 2024.
   https://proceedings.mlr.press/v234/lu24a.html
