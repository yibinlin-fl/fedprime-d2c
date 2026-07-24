# CLE-HFL v2 与 FedFalsify v0.3 当前框架

## 1. 当前研究问题

当前目标不是证明模型只能处理预先规定的 `noise/blur/weather/digital` 四类损坏，而是研究：

> 在模型异构、标签分布异构以及类别与具体损坏算子发生虚假关联时，如何避免联邦知识迁移进一步强化这种 shortcut，并提升未见损坏上的泛化能力？

旧 CLE-HFL 把四个损坏大类直接作为环境单位。CLE-HFL v2 改用具体损坏算子作为原子环境，四个家族名称只保留在审计元数据中，不进入任何训练或路由计算。

## 2. CLE-HFL v2 协议

### 2.1 标签异构

先使用 Dirichlet 分布划分 CIFAR-10 私有训练样本：

\[
p_k \sim \operatorname{Dirichlet}(\alpha), \qquad
D_k = \operatorname{Partition}(D_{\mathrm{train}}, p_k).
\]

当前正式数据使用：

```text
客户端数 K = 4
每客户端样本数 = 10,000
alpha = 0.5
seed = 0
```

### 2.2 损坏-标签纠缠

协议使用 15 个 CIFAR-C 风格具体算子。对客户端 \(k\) 和类别 \(c\)，随机指定一个训练阶段可见的主导算子 \(\phi_k(c)\)。

\[
P_k(o\mid y=c)
=
\gamma\mathbf{1}[o=\phi_k(c)]
+
(1-\gamma)\frac{1}{|\mathcal O_{\mathrm{seen}}|}.
\]

其中 \(\gamma\) 控制 shortcut 强度。当前正式数据使用 \(\gamma=0.9\)，主导算子的理论出现率为：

\[
0.9+\frac{0.1}{11}=0.9091.
\]

正式数据的实际主导匹配率为 `0.91015`，四个客户端分别为：

```text
0.9095 / 0.9101 / 0.9064 / 0.9146
```

### 2.3 Seen/Unseen 算子

训练仅使用 11 个 seen 算子：

```text
gaussian_noise, shot_noise,
defocus_blur, glass_blur, motion_blur,
snow, frost,
brightness, contrast, elastic_transform, jpeg_compression
```

完全不进入任何客户端训练数据的 4 个 unseen 算子：

```text
impulse_noise, zoom_blur, fog, pixelate
```

每个描述性家族各留出一个 unseen 算子。这里的家族只用于构造一个覆盖较均衡的评估切分；FedFalsify 不接收家族、算子名或 seen/unseen 标签。

### 2.4 反事实评估

从 CIFAR-10 测试集每类固定抽取 100 张图。每张图分别施加所有算子：

```text
test_seen     = 1,000 x 11 = 11,000
test_unseen   = 1,000 x 4  = 4,000
test_balanced = 15,000
test_clean    = 1,000
```

同一个类别在不同算子下都被评价，因此可直接测量模型是否依赖训练时的损坏-标签捷径。

## 3. FedFalsify v0.3

FedFalsify 不使用 public data，也不使用测试集进行教师选择。当前完整结构为：

```text
固定私有 fit/audit 划分
+ 接收端类别级候选证据
+ 非劣 UCB 否决
+ classifier-head TAU Top-1 路由
+ Conservative Margin Transfer
+ AugMix/JSD/DCL 本地鲁棒训练
```

### 3.1 固定 fit/audit 划分

每个客户端的私有训练数据按类别分为：

```text
D_fit   : 正常模型训练与 TAU 计算
D_audit : 只做候选教师证据审核
D_test  : 只做最终评价，绝不参与路由
```

当前 audit 比例为 15%。划分索引会保存为 `.npz`，控制组与候选方法读取同一文件。

### 3.2 类别级配对正确性证据

对接收端 \(r\)、类别 \(c\)、候选源 \(s\)，在接收端的 audit 样本上定义：

\[
d_i =
\mathbf{1}[\hat y_s(x_i)=y_i]
-
\mathbf{1}[\hat y_r(x_i)=y_i].
\]

类别级平均优势与标准误为：

\[
\hat\Delta_{s\rightarrow r,c}
= \frac{1}{n}\sum_i d_i,
\qquad
\operatorname{SE}
= \sqrt{\frac{\operatorname{Var}(d)}{n}}.
\]

非劣上界：

\[
\operatorname{UCB}_{s\rightarrow r,c}
= \hat\Delta_{s\rightarrow r,c}
+ \kappa\operatorname{SE}.
\]

若：

\[
\operatorname{UCB}<0,
\]

说明现有配对证据支持该源模型比接收端更差，候选被直接否决。它不是要求源模型已经显著更强，只排除有统计证据表明会更差的源。

### 3.3 Head-TAU 梯度兼容性

对未被否决的候选，计算：

\[
\tau_{s\rightarrow r,c}
=
\cos\left(
\nabla_{\theta_h}\mathcal L_{\mathrm{CMT}},
\nabla_{\theta_h}\mathcal L_{\mathrm{CE}}
\right),
\]

其中 \(\theta_h\) 是接收模型分类头。TAU 衡量“接受这个源的边界知识”与“接收端自己的监督目标”是否同向。

每个接收端、每个类别只选择 TAU 最大的一个候选源：

\[
s^*_{r,c}
=
\arg\max_{s:\operatorname{UCB}_{s\rightarrow r,c}\ge 0}
\tau_{s\rightarrow r,c}.
\]

### 3.4 保守边界迁移 CMT

先对每张图的 logits 做样本内标准化，消除异构模型的整体平移和尺度差异：

\[
\bar z = \frac{z-\operatorname{mean}(z)}
{\operatorname{std}(z)+\epsilon}.
\]

对真实类 \(y\) 与非目标类 \(j\)，定义边界：

\[
m_j = \bar z_y-\bar z_j.
\]

客户端只补齐教师比自己更大的正边界：

\[
\mathcal L_{\mathrm{CMT}}
=
\frac{1}{C-1}
\sum_{j\ne y}
\left[
\operatorname{clip}(m^{s^*}_j,0,m_{\max})
-m^r_j
\right]_+.
\]

并且只有教师在该样本上预测正确时才激活迁移。它不要求两个模型的 feature 维度相同，因此适配 ResNet、ShuffleNet 和 MobileNet 等异构模型。

### 3.5 本地总目标

\[
\mathcal L_{\mathrm{local}}
=
\mathcal L_{\mathrm{CE}}
+\lambda_{\mathrm{JSD}}\mathcal L_{\mathrm{JSD}}
+\mathcal L_{\mathrm{DCL}}
+\lambda_{\mathrm{CMT}}\mathcal L_{\mathrm{CMT}}.
\]

前三项沿用 RAHFL 的强本地鲁棒基座；CMT 是当前通信项。

## 4. 方法为什么不依赖固定损坏类别

训练路径只接收：

```text
image
label
receiver/source logits
fit/audit predictions
classifier-head gradients
```

以下信息只保存在数据审计和评价器中：

```text
operator_id
operator_name
family
seen/unseen
severity
```

因此可替换算子集合、改变 seen/unseen 切分，甚至换成未命名的真实损坏数据，而无需修改 FedFalsify 的路由和迁移公式。

## 5. 当前指标

对 all、seen、unseen 分别记录：

```text
Avg Acc
Worst Client Acc
Worst Operator Acc
Worst Client-Operator Acc
WCCA
CFG
```

WCCA 是所有有效“类别 × 算子”单元格中的最低准确率：

\[
\operatorname{WCCA}
=
\min_{c,o}\operatorname{Acc}(c,o).
\]

CFG 是每个类别跨算子的最大、最小准确率差，再对类别平均：

\[
\operatorname{CFG}
=
\frac{1}{C}\sum_c
\left[
\max_o\operatorname{Acc}(c,o)
-
\min_o\operatorname{Acc}(c,o)
\right].
\]

WCCA 越高越好，CFG 越低越好。

## 6. 当前实现与验证

核心实现：

```text
fedprime/data/corruptions.py
scripts/prepare_cle_v2_data.py
scripts/audit_cle_v2_data.py
fedprime/data/loaders.py
fedprime/engine/operator_metrics.py
fedprime/methods/fedfalsify/evidence.py
fedprime/methods/fedfalsify/router.py
fedprime/methods/fedfalsify/transfer.py
fedprime/methods/fedfalsify_experiment.py
```

OpenI 入口：

```text
scripts/openi_cle_v2_entry.py
```

正式数据包：

```text
local_runs/cle_hfl_v2_prepared/
  cle_hfl_v2_prepared_alpha05_gamma09_seed0_split0.tar.gz
```

大小约 346 MiB，包含 CLE-HFL v2 私有数据、CIFAR-100 public tarball 和协议审计文件。

本地验证：

```text
协议、配置与 FedFalsify 测试：22 passed
RTX 3050 两轮端到端 smoke：通过
round 0：warmup，CMT=0
round 1：66 candidates，12 inferior rejected，4 routes，CMT=1.0123
RTX 3050 RAHFL v2 一轮 smoke：通过，AsymHFL 与本地损失均为有限值
```

smoke 准确率不构成研究结果。

## 7. 下一次正式实验

先跑 12 轮，不直接跑 40 轮：

```text
RAHFL
严格 fit-only control
FedFalsify v0.3
```

冻结判断标准：

```text
1. FedFalsify 的 Avg、Worst、WCCA 高于严格 control；
2. FedFalsify 的 CFG 不高于严格 control；
3. unseen Avg/WCCA 不退化，unseen CFG 不上升；
4. 再判断是否值得与 RAHFL 扩展到 40 轮和多 seed。
```

当前只能说“框架和新协议已实现并通过工程验证”，还不能声称已经在 CLE-HFL v2 上击败 RAHFL。
