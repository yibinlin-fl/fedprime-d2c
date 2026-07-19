# FedCLEAR 最新理论闭环框架

版本：2026-07-11 Theory-Closed Draft

方法工作名：FedCLEAR-PCCD

核心模块：Paired Counterfactual Consensus Distillation

研究场景：CLE-HFL（Corruption-Label Entanglement in Heterogeneous Federated Learning）

状态：最新方法定义，取代此前 CCRE+IRD 主线；尚未编码

---

## 1. 先澄清 DCL 与 JSD

### 1.1 我们并没有证明 DCL 没用

目前已有实验只能支持：

```text
1. AugMix + JSD + DCL local-only 是一个很强的本地基座；
2. SARA、CARA、NIR-DCL 等 DCL 改版没有形成稳定、可独立复现的收益；
3. RAHFL 的 AsymHFL 在旧 alpha=0.5 设置下相对 local-only 收益很小；
4. CLE-HFL 下尚未跑过严格的 AugMix+JSD 对 AugMix+JSD+DCL 独立消融。
```

因此：

```text
“DCL 的独立贡献未知”是正确结论；
“DCL 已经被证明完全没用”是不严谨结论。
```

如果此前把它简单表述为“DCL 没用”，这个表述需要在这里纠正。

### 1.2 为什么最新框架仍固定使用 DCL

最新实验不是要测试 DCL，而是要测试 PCCD 通信。

我们让两个方法使用完全相同的本地训练：

```text
RAHFL baseline：AugMix + JSD + DCL + AsymHFL
FedCLEAR-PCCD ：AugMix + JSD + DCL + PCCD
```

这样本地模块被控制为相同变量。最终差异只能来自：

```text
AsymHFL 与 PCCD 的通信差异
```

恢复 DCL 不是赌它能额外提高准确率，而是避免在验证通信时同时替换本地表示学习，从而再次出现“到底是哪一块导致变化”的不可解释问题。

### 1.3 为什么继续固定 JSD

JSD 的作用不是消除 corruption-label shortcut。

对同一个私有样本的多个 AugMix 视图：

\[
\mathcal L_{JSD}
=
\frac{1}{V}\sum_{v=1}^{V}
\operatorname{KL}(p^{(v)}\parallel \bar p),
\qquad
\bar p=\frac{1}{V}\sum_v p^{(v)}
\]

它控制的是模型对局部图像扰动的预测敏感性。

JSD 的能力边界必须写清楚：

```text
它能提高普通 corruption robustness；
但如果所有视图都保留同一个原始 shortcut，它不能识别和消除该 shortcut。
```

继续使用 JSD 的理由有两个：

1. 它是 RAHFL 强基座的固定组成部分；
2. PCCD 只覆盖有限公共样本，JSD 继续为全部私有样本提供局部平滑。

二者不解决同一个层面：

```text
JSD：私有样本上的局部 perturbation consistency；
PCCD：公共成对干预上的跨客户端 shortcut cancellation。
```

因此保留 JSD 不是把旧模块重新包装成创新，也不是靠盲目叠加提高准确率。

---

## 2. 最新框架只改变一个核心变量

最终冻结结构：

```text
本地阶段：完全复用同一套 AugMix + CE + JSD + DCL

RAHFL 通信：AsymHFL
FedCLEAR 通信：PCCD
```

FedCLEAR-PCCD：

\[
\boxed{
\text{FedCLEAR-PCCD}
=
\underbrace{\text{AugMix+CE+JSD+DCL}}_{\text{固定强本地基座}}
+
\underbrace{\text{PCCD}}_{\text{唯一新增方法模块}}
}
\]

当前 CCRE+IRD 完整结果保留为失败方法，不继续调参。

---

## 3. CLE-HFL 因果模型

设：

```text
Y：真实类别
S：由类别决定的语义内容
G：损坏环境或成像条件
K：客户端
X：观测图像
```

生成关系：

\[
Y\rightarrow S\rightarrow X,
\qquad
K\rightarrow P(G\mid Y,K),
\qquad
G\rightarrow X
\]

CLE-HFL 的核心不是图像单纯被损坏，而是：

\[
P(G\mid Y,K)\neq P(G\mid K)
\]

也就是说，在客户端 `k` 内，某些类别会高概率绑定某些 corruption。

模型可能学习：

```text
noise -> cat
blur  -> car
weather -> bird
```

而不是学习真实语义 `S`。

由于不同客户端使用不同映射 `phi_k(Y)`，shortcut 是客户端特有的，而真实语义在客户端之间共享。

PCCD 的目标就是利用：

```text
语义跨客户端稳定；
shortcut 随客户端和 corruption view 改变。
```

---

## 4. 为什么旧 CCRE 不能提供理论闭环

私有样本已经是：

\[
x^{obs}=T_{g_k(y)}(x^{clean})
\]

旧 CCRE 再应用增强：

\[
\tilde x=T_h(x^{obs})=T_h(T_{g_k(y)}(x^{clean}))
\]

原 shortcut `g_k(y)` 仍然保留在所有视图中。

因此最小化旧 CCRE 只能保证：

```text
模型在保留原 shortcut 的情况下适应更多叠加损坏。
```

它不能推出：

```text
模型已经不依赖原 shortcut。
```

最新方法不再在私有已损坏图片上声称执行 counterfactual intervention。

---

## 5. 可识别性要求：真正成对的公共干预

### 5.1 无标签同任务域公共池

从 CIFAR-10 train 中预留：

\[
D_{pub}=\{u_b\}_{b=1}^{B_{pub}}
\]

要求：

```text
与客户端私有训练集互斥；
与验证集和测试集互斥；
训练代码不读取 public label；
所有基线使用相同 public indices。
```

建议划分：

```text
40,000 private：4 clients x 10,000
 5,000 public：无标签任务域公共池
 5,000 reserved：协议验证或保留不用
10,000 CIFAR-10 test：只评价
```

### 5.2 从同一内容生成 paired views

对于同一个未损坏公共样本 `u`：

\[
u^{(0)}=u,
\qquad
u^{(g)}=T_g(u),\quad g=1,\ldots,G
\]

这一次所有视图都从同一个内容直接生成：

```text
car clean
car noise
car blur
car brightness
```

而不是：

```text
car noise
car noise+blur
car noise+brightness
```

因此可以合理写成对 corruption 变量的受控干预。

### 5.3 不针对四个 benchmark group 写死

公共算子从比 evaluation 更宽的标签无关算子池随机采样：

```text
noise family
blur family
color / contrast / brightness
compression / pixelation
fog / haze
frequency perturbation
```

正式测试需要额外保留未参加公共训练的 corruption，验证方法不是记住四个组。

---

## 6. PCCD 方法定义

## 6.1 客户端跨视图语义分数

客户端 `k` 对公共样本 `u` 的第 `g` 个视图输出：

\[
p_{k,g}(c\mid u)=\operatorname{softmax}(z_{k,g}(u))_c
\]

定义 log-opinion 语义分数：

\[
s_{k,c}(u)
=
\frac{1}{G+1}
\sum_{g=0}^{G}
\log(p_{k,g}(c\mid u)+\epsilon)
\]

客户端不变分布：

\[
q_k(c\mid u)
=
\frac{\exp(s_{k,c}(u))}
{\sum_{j=1}^{C}\exp(s_{k,j}(u))}
\]

几何平均的含义是：

```text
某类别只有在所有 corruption views 下都有证据，q_k 才会保持较高；
只在单个损坏视图上突然升高的 shortcut 类别会被压低。
```

## 6.2 连续可信度，不设置人工阈值

\[
r_k(u)
=
1-\frac{H(q_k(\cdot\mid u))}{\log C}
\in [0,1]
\]

```text
低熵且跨视图一致：r_k 高；
接近均匀、共同无知：r_k 低。
```

不设置 `confidence_threshold`，减少超参数和人为开关。

## 6.3 Leave-one-out 教师

对接收客户端 `i`：

\[
q_{-i}(c\mid u)
=
\frac{
\sum_{k\neq i}r_k(u)q_k(c\mid u)
}{
\sum_{k\neq i}r_k(u)+\epsilon
}
\]

教师权重：

\[
m_{-i}(u)
=
1-\frac{H(q_{-i}(\cdot\mid u))}{\log C}
\]

每个公共样本都拥有自己的教师，不再使用客户端整体准确率决定谁教所有图片。

## 6.4 成对反事实蒸馏

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
\right)
\]

该损失同时要求：

```text
从其他异构客户端接收类别语义；
同一内容在不同 corruption 下保持同一预测；
不确定教师自动产生较小梯度。
```

---

## 7. 条件性理论保证

任何深度学习方法都无法在实验前无条件证明一定比 RAHFL 高 3 个点。

我们能做的是给出明确假设，并证明 PCCD 的优化目标在这些假设下直接控制 CLE-HFL 的失败量，而不是仅凭直觉。

### 7.1 假设

**A1：标签保持干预。** 对公共样本 `u`，所有 `T_g(u)` 具有相同真实类别。

**A2：公共任务相关性。** 公共池与私有任务共享类别语义，而非 CIFAR-100 式跨域共同无知。

**A3：shortcut 多样性。** 不同客户端的 corruption-label shortcut 不完全相同，客户端偏差不是同方向完全相关。

**A4：弱多数正确。** 对足够多的公共样本，至少一组跨视图稳定客户端对真实类别具有正 margin。

**A5：共享输出语义。** 模型架构可以不同，但类别输出坐标含义一致。

这些不是隐藏条件，必须在论文方法与限制部分明确写出。

### 7.2 命题一：跨视图 pooling 抑制视图特有 shortcut

假设客户端 log-probability 可分解为：

\[
\log p_{k,g,c}(u)
=
a_{k,c}(u)+b_{k,g,c}(u)
\]

其中 `a` 是内容相关稳定项，`b` 是 corruption-view 残差。

则：

\[
s_{k,c}(u)
=
a_{k,c}(u)
+
\frac{1}{G+1}\sum_g b_{k,g,c}(u)
\]

如果公共干预足够多样，使得：

\[
\left|
\frac{1}{G+1}\sum_g b_{k,g,c}(u)
\right|
\le \eta
\]

且真实类别的稳定 margin 大于 `2 eta`，则视图 pooling 后真实类别排序不会被 corruption 残差翻转。

这说明 PCCD 需要的是多样、成对、标签保持的公共干预，而不是任意随机增强。

### 7.3 命题二：跨客户端共识削弱客户端特有偏差

进一步写成：

\[
s_{k,c}(u)=a_c(u)+d_{k,c}(u)+e_{k,c}(u)
\]

其中：

```text
a_c：跨客户端共享语义；
d_kc：客户端特有 shortcut；
e_kc：有限视图估计误差。
```

若不同客户端 `d_kc` 不完全同向，且加权均值有界，则 leave-one-out 共识中的偏差随有效客户端数量增加而下降。

直观上：

```text
client 0 把 noise 当作 cat；
client 1 把 blur 当作 cat；
client 2 把 digital 当作 cat；
真正跨客户端稳定的是 cat 的内容，而不是某一种 corruption。
```

这正是 CLE-HFL 使用 client-specific mapping 的理论价值。

### 7.4 命题三：PCCD loss 足够小时保持教师决策

设教师真实类别 margin：

\[
\Delta(u)
=
q_{-i,y}(u)-\max_{c\neq y}q_{-i,c}(u)>0
\]

由 Pinsker 不等式：

\[
\|p_{i,g}-q_{-i}\|_1
\le
\sqrt{2\operatorname{KL}(q_{-i}\parallel p_{i,g})}
\]

如果：

\[
\operatorname{KL}(q_{-i}\parallel p_{i,g})
<
\frac{\Delta(u)^2}{2}
\]

则学生在视图 `g` 上不会翻转教师的类别决策。

因此，只要教师正确且具有 margin，降低 PCCD loss 会直接约束不同 corruption views 上的分类错误。

### 7.5 命题四：PCCD 同时界定跨视图不一致

对任意两个视图 `g,h`：

\[
\operatorname{TV}(p_{i,g},p_{i,h})
\le
\operatorname{TV}(p_{i,g},q_{-i})
+
\operatorname{TV}(q_{-i},p_{i,h})
\]

结合 Pinsker 不等式，两个 KL 项同时下降时，学生跨 corruption view 的预测差异也受到上界控制。

这给出了 PCCD 与 CFG、shortcut sensitivity 之间的直接理论联系。

### 7.6 理论不能保证的部分

如果所有客户端都对同一公共样本稳定地预测错误，或者所有客户端拥有完全相同 shortcut，PCCD 会形成错误共识。

因此不能声称：

```text
无条件消除所有虚假相关；
必然超过 RAHFL 三个点；
不需要任务相关公共数据。
```

理论保证是条件性的，但条件与 CLE-HFL 协议和诊断指标可以对应验证。

---

## 8. DCL、JSD 与 PCCD 各自解决什么

| 模块 | 数据位置 | 解决目标 | 不能解决什么 |
| --- | --- | --- | --- |
| CE | 私有有标签数据 | 学习类别判别 | 不防 shortcut |
| JSD | 私有 AugMix 视图 | 局部预测平滑 | 不能移除所有视图共享的原 shortcut |
| DCL | 私有特征 | 保持类内/类间判别结构 | 不直接识别 corruption-label shortcut |
| PCCD | 公共成对干预视图 | 跨客户端识别并传播损坏不变语义 | 依赖任务相关公共池与弱多数正确 |

四项不是四个都宣称创新。

论文中只有 PCCD 是方法创新；CE/JSD/DCL 是固定训练基座。

---

## 9. 为什么这不是试盲盒

实验只改变一个变量：

```text
RAHFL local = FedCLEAR local
RAHFL public data = FedCLEAR public data
RAHFL optimizer = FedCLEAR optimizer
RAHFL rounds = FedCLEAR rounds

唯一差异：AsymHFL vs PCCD
```

因此实验结果可以直接回答：

> 在相同强本地基座和相同公共数据下，PCCD 是否比 AsymHFL 更适合 CLE-HFL？

不再同时更换增强、对比学习、通信和公共数据后，再猜是谁起作用。

---

## 10. 公平性与测试泄漏

当前 unified RAHFL 会在每轮通信前使用 `test_loader` 准确率进行路由。

正式论文至少要报告：

```text
RAHFL-original：复现原测试准确率路由，用作最强复现基线；
RAHFL-val：使用独立 validation split 路由，无测试泄漏；
FedCLEAR-PCCD：不使用 test/validation label 路由。
```

所有方法使用同一无标签任务域公共池。

旧 CIFAR-100 public 结果可作为跨域公共池补充实验，但不能和新的 CIFAR-10 public 主实验混为同一个公平对比。

---

## 11. 最小实验，不做大规模盲跑

### 11.1 先运行两个 12-round probe

```text
实验 A：AugMix+JSD+DCL + AsymHFL
实验 B：AugMix+JSD+DCL + PCCD

alpha=0.5
gamma=0.9
seed=0
相同 in-domain unlabeled public 5k
```

### 11.2 进入 40 轮的门槛

第 12 轮附近，实验 B 必须同时满足：

```text
avg_acc    >= 实验 A + 1.5
worst_acc  >= 实验 A + 1.0
WCCA       >= 实验 A + 4.0
CFG        <= 实验 A - 1.5
teacher entropy 不塌缩
paired-view disagreement 下降
```

只提高 WCCA、但 avg/worst 下降，视为失败。

### 11.3 40 轮目标

正式目标相对 matching RAHFL：

```text
avg_acc    +3.0
worst_acc  +2.0
WCCA       +4.0
CFG        -2.0
```

达到单 seed 目标后，再运行 seed 1/2、gamma 0/0.6 和其他 alpha。

---

## 12. 实现结构

### 12.1 数据

```text
scripts/prepare_cle_in_domain_public.py
fedprime/data/cle_public.py
```

保存：

```text
private indices
public indices
reserved indices
SHA256
```

### 12.2 方法

```text
fedprime/methods/pccd.py
fedprime/methods/fedclear_pccd.py
```

核心函数：

```text
build_paired_public_views
log_opinion_consensus
normalized_entropy_confidence
leave_one_out_teacher
pccd_distillation_loss
```

### 12.3 配置

```text
configs/debug_fedclear_pccd.yaml
configs/openi_v100_rahfl_cle_indomain_probe.yaml
configs/openi_v100_fedclear_pccd_probe.yaml
```

新增主要参数仅保留：

```text
public_num_views: 3
lambda_pccd: 1.0
warmup_rounds: 3
```

不加入 CCRE temperature、confidence threshold、adaptive beta、EMA prior 等旧参数。

### 12.4 每轮诊断

```text
avg_acc
worst_acc
WCCA
CFG
pccd_loss
teacher_entropy
teacher_confidence
paired_view_disagreement
teacher_margin
```

---

## 13. 创新点如何写

### 贡献一：CLE-HFL 问题与协议

首次系统刻画模型异构联邦学习中客户端特有的 corruption-label entanglement，并提供可控 `gamma` 协议。

### 贡献二：PCCD

提出面向同一公共内容的成对 corruption intervention，通过跨视图 log-opinion pooling 和跨客户端 leave-one-out 共识，提取可在异构模型间传播的损坏不变语义。

### 贡献三：细粒度评价

使用 WCCA、CFG 和 worst-client-group 指标揭示平均准确率掩盖的 shortcut failure。

这三个贡献共同构成论文，而不是仅靠“把 DCL 换了一个名字”。

---

## 14. 相关工作边界

FedIIR 已通过跨客户端梯度对齐学习隐式不变关系；FedCD 已研究联邦领域泛化中的 spurious correlation；FedKA 已使用公共空间投票伪标签。

PCCD 必须保持以下区别：

```text
模型架构异构；
问题是 corruption-label entanglement；
知识单位是同一公共内容的 paired corruption response；
通信位于共享类别概率空间；
不交换或聚合模型参数、梯度和特征；
不使用测试准确率选择教师。
```

正式投稿前仍需进行系统文献排重，不能提前宣称绝对首创。

---

## 15. 最新冻结决策

```text
[保留] CLE-HFL 场景、gamma 协议、WCCA/CFG
[冻结] CCRE+IRD 为失败历史方法
[固定] RAHFL 与 FedCLEAR 使用相同 AugMix+JSD+DCL 本地训练
[新增] 同任务域无标签 public 5k
[新增] paired corruption views
[替换] AsymHFL -> PCCD
[先跑] matching 12-round A/B probe
[禁止] probe 未通过就跑 40 轮、多 seed 或大规模调参
```

这份框架的理论主张不是“必然提高 3 点”，而是：

> 在标签保持公共干预、客户端 shortcut 多样、弱多数教师正确的条件下，PCCD 直接约束 corruption view 间预测差异，并在教师具有正 margin 时保证学生保持其类别决策。

这是当前可以被公式推导、被指标验证、也可以被实验否证的完整方法假设。

---

## 16. 相关论文

1. FedIIR: https://proceedings.mlr.press/v202/guo23b.html
2. FedCD: https://arxiv.org/abs/2407.19174
3. FedKA: https://proceedings.mlr.press/v189/sun23a.html
4. FIXED: https://proceedings.mlr.press/v234/lu24a.html
