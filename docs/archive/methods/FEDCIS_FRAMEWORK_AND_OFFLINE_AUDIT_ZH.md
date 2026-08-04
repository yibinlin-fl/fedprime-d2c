# FedCIS 修正版框架与离线审计规范

更新日期：2026-08-03

## 0. 文档定位

FedCIS 的全称暂定为：

```text
Federated Class-conditional Invariant Sensitivity
联邦类别条件不变敏感场学习
```

FedCIS 是 CLE-HFL v2 下的新候选方法，不是已经验证有效的正式主线。
当前状态如下：

```text
框架定义：已形成修正版候选
代码实现：仅完成独立 Audit A/B，不存在联邦训练 runner
离线审计：NO-GO（2026-08-03）
正式训练：禁止启动
研究结论：不能声称有效或超过 RAHFL
```

正式离线审计结果：

```text
真实类别跨种子子空间相似度 = 0.1673
类别打乱对照               = 0.1669
等秩随机对照               = 0.1197
跨客户端同类别相似度       = 0.1269
跨客户端异类别相似度       = 0.1318
真实正交攻击胜过两个对照   = 30.30%（冻结门槛 >=60%）
```

因此，当前 DCT 低秩投影、双 AugMix 视图和广义特征分解组成的 FedCIS-v0
没有证明可识别出类别语义敏感子空间。本文档保留为完整假设与反证记录，
不再授权实现 Audit C、12 轮或 40 轮 FedCIS。

本文件冻结四件事：

1. FedCIS 试图解决的具体问题；
2. 原始提案中必须修正的公式和实现风险；
3. 方法能够与不能够覆盖的边界；
4. 进入 12 轮训练前必须通过的离线审计门槛。

## 1. 目标场景

FedCIS 专门面向 CLE-HFL v2：

```text
model heterogeneity
+ label-skew Non-IID
+ corruption-label entanglement
+ seen/unseen corruption generalization
```

当前正式协议为：

```text
K=4 heterogeneous clients
Dirichlet alpha=0.5
entanglement gamma=0.9
11 seen concrete corruption operators
4 unseen concrete corruption operators
operator metadata unavailable to training and communication
```

核心失败模式是：客户端内部的类别与损坏发生条件依赖。例如，同一类别
`cat` 在不同客户端分别主要与 blur、noise、brightness 或 JPEG 绑定。模型
可能利用损坏模式预测类别，而不是学习可迁移语义。

FedCIS 的研究问题是：

> 在不共享公共图片、不聚合异构模型参数、不使用损坏名称或算子 ID 的前提
> 下，能否共享同一类别在不同客户端中的决策敏感结构，并抑制客户端特有的
> corruption-label shortcut？

## 2. 核心识别假设

对于客户端 `k` 和类别 `c`，输入敏感场可被概念性地写成：

\[
j_{k,c}=j_c^{\mathrm{shared}}+j_{k,c}^{\mathrm{specific}}+\epsilon_{k,c}.
\]

FedCIS 假设：

1. 同一类别在不同客户端之间存在可恢复的共享敏感子空间；
2. 客户端特有 shortcut 在该共享子空间的正交补中占据更大能量；
3. 在正交补中构造最坏方向扰动，并保持分类结果，可以减少 CLE shortcut。

这三个命题目前只是待验证的可识别性假设。尤其不能直接写成：

```text
跨客户端共同方向必然等于语义；
跨客户端不同方向必然等于 shortcut。
```

真实语义梯度可能随图片位置、姿态和模型架构变化；纹理、边缘或共同预处理
偏置也可能形成跨客户端一致方向。因此，离线识别审计是实现 runner 前的
强制前置条件。

## 3. 总体结构

修正版 FedCIS-v0 定义为：

\[
\boxed{
\text{FedCIS-v0}
=
\text{AugMix/JSD/DCL robust base}
+
\text{class-conditional sensitivity statistics}
+
\text{server invariant-subspace recovery}
+
\text{orthogonal projected counterfactual training}
}
\]

第一阶段冻结 RAHFL 的本地鲁棒基座，仅把 AsymHFL 替换为 FedCIS 协作
模块。这样后续 A/B 实验能够把增量归因到新的通信载荷。

```text
RAHFL   = AugMix/JSD/DCL + AsymHFL
FedCIS  = AugMix/JSD/DCL + sensitivity-subspace collaboration
```

## 4. 本地鲁棒基座

客户端 `k` 保留现有 RAHFL 本地损失：

\[
\mathcal L_k^{\mathrm{base}}
=
\mathcal L_{\mathrm{CE}}
+\lambda_J\mathcal L_{\mathrm{JSD}}
+\lambda_D\mathcal L_{\mathrm{DCL}}.
\]

AugMix 在这里不是 FedCIS 的创新点。它只负责提供随机、多样的局部扰动，
帮助暴露模型敏感方向。FedCIS 不应声称重新发明或改进 AugMix。

## 5. 类别条件输入敏感统计

### 5.1 Margin 与输入梯度

对真实类别 `c` 定义分类 margin：

\[
M_{k,c}(x)
=
z_{k,c}(x)-\operatorname{LSE}_{r\ne c}z_{k,r}(x).
\]

对同一私有样本生成两个独立 AugMix 视图：

\[
x^{(1)}=a_1(x),\qquad x^{(2)}=a_2(x).
\]

计算输入 margin 梯度：

\[
h_{k,c}^{(v)}(x)
=
\nabla_{x^{(v)}}M_{k,c}(x^{(v)}).
\]

它表示局部改变输入时，哪些方向最容易改变类别 `c` 相对其他类别的决策
margin。

### 5.2 模型无关投影

所有客户端共享相同输入空间，但内部特征维度不同。使用固定正交输入基
`Psi` 将梯度压缩：

\[
s_{k,c}^{(v)}
=
\frac{\Psi^\top h_{k,c}^{(v)}}
{\|\Psi^\top h_{k,c}^{(v)}\|_2+\epsilon}.
\]

第一版只允许使用不依赖损坏标签的固定多尺度 DCT 基。它不是
`noise/blur/weather/digital` 分类器，也不能读取 CLE-HFL v2 的 operator
元数据。建议审计投影维度为 `r=32`。

### 5.3 修正后的 PSD 二阶统计

原提案直接使用 `E[s1 s2^T]`，该矩阵不保证对称或半正定，不能直接作为
稳定的广义特征值问题左项。修正版定义：

\[
a_{k,c}=\frac{s_{k,c}^{(1)}+s_{k,c}^{(2)}}{2},
\qquad
d_{k,c}=s_{k,c}^{(1)}-s_{k,c}^{(2)}.
\]

类别稳定统计：

\[
A_{k,c}=\mathbb E[a_{k,c}a_{k,c}^{\top}].
\]

视图变化统计：

\[
N_{k,c}=\mathbb E[d_{k,c}d_{k,c}^{\top}].
\]

`A` 和 `N` 都是对称半正定矩阵。客户端只能在本地支持足够的类别上计算
统计。第一版审计建议每个客户端、每个类别最多采样 16 个样本。

## 6. 通信载荷

客户端上传：

\[
\{A_{k,c},N_{k,c},m_{k,c}\}_{c=1}^{C},
\]

其中 `m` 是固定形状的有效支持掩码。为了减少直接暴露类别数量，第一版不
上传精确 `n_{k,c}`，服务器对支持合格的客户端做等权聚合。

FedCIS 不上传：

```text
private images
public logits
prototypes
model parameters
single-sample Jacobians
corruption IDs/names/families/severity
```

但类别条件二阶统计和支持掩码仍可能泄露类别存在性与群体属性。论文只能
声称降低直接样本泄露风险，不能声称零隐私泄露。正式版本需要评估裁剪、
安全聚合或差分隐私的兼容性。

## 7. 服务器端不变子空间恢复

对至少有两个支持合格客户端的类别 `c`，服务器计算：

\[
\bar A_c
=
\frac{1}{|\mathcal K_c|}
\sum_{k\in\mathcal K_c}A_{k,c},
\qquad
\bar N_c
=
\frac{1}{|\mathcal K_c|}
\sum_{k\in\mathcal K_c}N_{k,c}.
\]

原提案中的 `(A_k-A_bar)^2` 含义不明确，也不保证期望的 PSD 结构。修正为：

\[
D_c
=
\frac{1}{|\mathcal K_c|}
\sum_{k\in\mathcal K_c}
(A_{k,c}-\bar A_c)(A_{k,c}-\bar A_c)^{\top}.
\]

服务器求解：

\[
\bar A_c u
=
\lambda
(\bar N_c+\mu D_c+\epsilon I)u.
\]

选取最大的前 `q` 个广义特征向量：

\[
U_c=[u_{c,1},\ldots,u_{c,q}],
\qquad
P_c=U_cU_c^{\top}.
\]

第一版建议审计 `q=4`，并将 `U_c` 下发给客户端。支持客户端少于两个、
矩阵条件数异常或有效样本不足时必须 abstain，客户端只执行本地基座。

## 8. 正交投影反事实训练

对本地真实类别 `c`，将 margin 梯度限制在全局子空间正交补：

\[
h_{k,c}^{\perp}
=
\Psi(I-P_c)\Psi^{\top}\nabla_xM_{k,c}(x).
\]

原提案使用 `+h_perp`，这会沿 margin 上升方向移动，通常使样本更容易。
为了暴露模型在非共识方向上的脆弱性，修正版必须使用 margin 下降方向：

\[
\delta_{k,c}^{\perp}
=
-\varepsilon
\frac{h_{k,c}^{\perp}}
{\|h_{k,c}^{\perp}\|_2+\epsilon}.
\]

\[
x^{\mathrm{cf}}
=
\operatorname{clip}(x+\delta_{k,c}^{\perp}).
\]

等价实现可以沿交叉熵损失的正梯度方向。构造 `delta` 后必须 `detach`，
避免反事实训练通过输入梯度产生完整二阶反向传播。

修正版 v0 不启用原始 `L_sens` 梯度范数正则，因为它需要参数与输入之间的
混合二阶导数，显存和运行成本高，也会把“子空间知识”与普通 input-gradient
regularization 混杂。v0 只使用：

\[
\mathcal L_k
=
\mathcal L_k^{\mathrm{base}}
+\lambda_{\mathrm{cf}}
\operatorname{CE}(f_k(x^{\mathrm{cf}}),y)
+\lambda_{\mathrm{cons}}
\operatorname{JS}(p_k(x),p_k(x^{\mathrm{cf}})).
\]

`L_sens` 只能在 v0 通过后作为单独消融研究，不能默认加入。

## 9. 联邦轮流程

```text
warmup: only AugMix/JSD/DCL local robust training
  -> clients compute class-conditional sensitivity statistics
  -> upload fixed-shape A/N/support tensors
  -> server recovers U_c and abstains on unsupported classes
  -> server broadcasts low-rank U_c
  -> clients construct detached orthogonal projected counterfactuals
  -> next local phase optimizes base + counterfactual CE/JSD
  -> refresh sensitivity subspaces periodically
```

如果未来进入 runner，候选初值可以是 3 轮 warmup、每 5 轮刷新一次子空间；
这不是当前已验证超参数。

## 10. 适用边界

### 10.1 能够主张的目标

FedCIS 最直接面向：

```text
classes observed by at least two clients
client-specific corruption-label shortcut
model-heterogeneous collaboration through a common input space
taxonomy-free seen/unseen corruption evaluation
WCCA improvement and CFG reduction
```

### 10.2 不能主张的目标

FedCIS 不能保证：

1. 为完全缺失类别创造视觉语义；
2. 区分所有客户端共同拥有的相同 shortcut；
3. 对任意未知损坏无条件泛化；
4. 共同输入敏感方向必然等于人类语义；
5. 上传二阶统计等于零隐私泄露；
6. 在四客户端下稳定恢复每个类别的低秩子空间。

完全缺失类别仍属于独立的信息缺失问题。FedCIS 的主要论文主张应聚焦于
“类别存在但与不同客户端损坏条件纠缠”的可审计目标。missing-class 结果必须
报告，但不能作为 FedCIS 已解决的贡献。

## 11. 原提案问题清单

正式实现必须确认下列问题均已修正：

```text
[x] counterfactual direction changes from margin ascent to margin descent
[x] nonsymmetric cross-view moment changes to PSD A/N statistics
[x] ambiguous matrix square changes to outer-product dispersion
[x] full second-order L_sens removed from v0
[x] exact class count removed from default payload
[x] zero-privacy claim removed
[x] missing-class limitation stated explicitly
[ ] semantic/shortcut subspace identifiability verified empirically
[ ] cross-architecture comparability verified empirically
[ ] RTX 3050 memory/runtime feasibility verified
```

## 12. 强制离线审计

FedCIS 不允许先实现 12/40 轮 runner。必须先复用现有 CLE-HFL v2 数据、
fit/audit split 和本地 checkpoint，完成以下阶段。

### Audit A：数值与计算可行性

要求：

1. 四种异构模型均能得到有限的 `A/N/U/delta/loss/gradient`；
2. 广义特征值分解无 NaN、Inf 或严重病态；
3. RTX 3050 可以完成至少四个客户端的审计批次；
4. `detach(delta)` 后不存在意外二阶计算图；
5. 相同输入、checkpoint 和随机种子可复现子空间。

任一失败即 NO-GO，不进入识别审计。

### Audit B：子空间可识别性

必须包含三类对照：

```text
true class-matched global U_c
class-shuffled global U_perm(c)
equal-rank random orthogonal U_rand
```

并在至少三个独立 AugMix 种子上检验：

1. 真实 `U_c` 的跨种子投影相似度高于随机和打乱类别对照；
2. 同类别跨客户端相似度高于不同类别相似度；
3. 正交补 projected attack 的 audit 损失增量高于等秩随机子空间；
4. 以上优势不能只由单一模型或单一类别贡献。

冻结门槛：至少 60% 的可审计 `client x class` 目标同时满足方向性条件，且
真实子空间的总体优势必须高于两个对照。若真实、随机和打乱类别结果接近，
核心识别假设判定为 NO-GO。

operator 元数据只能在所有 `U/delta` 产生之后用于审计归因，绝不能参与子
空间构造、超参数选择或通信。

### Audit C：匹配的一步更新审计

所有分支必须从相同 checkpoint、相同 fit batch、相同优化器状态出发：

```text
C0: base CE/JSD/DCL step
C1: C0 + random-subspace counterfactual step
C2: C0 + class-shuffled-subspace counterfactual step
C3: C0 + true FedCIS counterfactual step
```

每次更新后恢复原 checkpoint。只在 `D_audit` 上评价，不使用最终测试标签
选择子空间、阈值、超参数或更新。

允许进入 12 轮 runner 的冻结门槛：

1. `C3` 的平均类别条件 audit loss 优于 `C0/C1/C2`；
2. 至少 60% 的可审计 `client x class` 目标相对 `C0` 非负；
3. audit Avg 不下降；
4. audit Worst 和 WCCA 不下降；
5. audit CFG 不增加；
6. seen/unseen audit accuracy 均不下降；
7. 所有更新有限且没有客户端出现集中性崩溃。

任何一项核心指标门槛失败，都不得通过调整单个 `lambda` 或 `epsilon` 直接
进入正式实验。应先归因识别假设、投影基或反事实方向是否失败。

## 13. 12 轮与 40 轮门槛

只有 Audit A/B/C 全部通过后，才允许实现一个严格 12 轮 A/B probe：

```text
strict control: AugMix/JSD/DCL, same fit/audit split
candidate:      strict control + FedCIS-v0
```

12 轮使用相同数据、模型初始化、随机种子、训练样本、优化器和预算。最终
测试集只用于报告。建议继续采用最后五轮门槛：

```text
Avg >= strict control + 1.0 point
Worst delta >= 0
WCCA delta >= 0
CFG delta <= 0
seen/unseen accuracy delta >= 0
```

未全部通过时，不运行 40 轮，不只调 `lambda_cf`、`epsilon` 或子空间维度。

只有 12 轮通过后，才准备 40 轮、多 seed、多 gamma，并加入 RAHFL-val 的
严格无测试泄漏对照。

## 14. 必须避免的论文表述

不能使用：

```text
完全创新
理论上必然有效
共同敏感方向就是纯语义
彻底消除 shortcut
零隐私泄露
对任意未知 corruption 保证鲁棒
```

在审计通过前，只能表述为：

> FedCIS 提出一个待验证的类别条件输入敏感子空间假设，尝试在共同输入空间
> 中为异构模型建立不依赖 public data 的协作载荷。

## 15. 当前研究决策

```text
FedCIS-v0: CONDITIONAL GO for offline audit only
FedCIS 12-round runner: BLOCKED
FedCIS 40-round experiment: BLOCKED
```

下一步只实现离线审计所需的最小模块，不改动现有正式训练 runner，不启动
OpenI/Kaggle 付费实验。
