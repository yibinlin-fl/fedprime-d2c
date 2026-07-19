# FedEASE v2.1 理论框架与完整候选实现状态

更新时间：2026-07-19

## 1. 当前研究主线

当前问题仍是 CLE-HFL：

```text
模型异构
+ label-skew Non-IID
+ client-specific corruption-label entanglement
```

已完成的 RAHFL 诊断表明，当 `alpha=0.5, seed=0` 固定、`gamma` 从 0 增加到 0.9 时，
RAHFL 的 Avg/Worst/WCCA 显著下降，CFG 显著上升。因此 CLE-HFL 具有初步受控实验依据。

FedCLEAR v0.1（CCRE + IRD）已经是负结果。PCCD 是历史候选，不再覆盖当前 FedEASE 主线。

## 2. FedEASE 的完整理论结构

```text
FedEASE
  = RAHFL robust local base
  + class-conditional environment invariance
  + environment-balanced structural transfer
  + optional safe communication projection
```

完整目标计划为：

\[
\mathcal L_k
=
\mathcal L_{BER}
+\lambda_{jsd}\mathcal L_{JSD}
+\lambda_{dcl}\mathcal L_{DCL}
+\lambda_{dep}\mathcal L_{CDep}
+\lambda_{rel}\mathcal L_{EBST}.
\]

重要修正：`BER` 替换原始 clean CE，而不是在完整 CE 上再重复叠加一份分类损失。

### 2.1 PEW：Public Environment Witness

使用与分类标签无关的公共图像和已知 corruption operators 训练环境识别器。
PEW 只预测 corruption environment，不提供分类知识。

代码同时支持两条可切换路径：

```text
environment_mode: oracle   # 使用数据生成器保存的 train_corruption_ids.npy
environment_mode: learned  # 使用 PEW 预测环境、置信度和连续环境 embedding
```

PEW 在无标签 CIFAR-100 图像上在线合成 `clean/noise/blur/weather/digital/unknown`
环境监督，同时预测损坏严重度。低置信度预测进入 `unknown`，避免强行分配到四个已知组。
PEW 的实现完成不代表估计质量已经通过；正式实验必须先检查其验证准确率、私有环境组准确率和
unknown rate。

### 2.2 BER：Balanced Environment Risk

BER 解决同一类别中 dominant corruption group 控制普通经验风险的问题。

客户端从本地标签和环境 ID 计算：

\[
n_{k,c,e}=|\{i:y_i=c,e_i=e\}|.
\]

目标环境权重为：

\[
a_{k,c,e}
=
\frac{\min(n_{k,c,e},n_{cap})^{\gamma_s}}
{\sum_{e'}\min(n_{k,c,e'},n_{cap})^{\gamma_s}}.
\]

每个样本在全数据 BER 中的系数为：

\[
w_{k,i}
=
\frac{1}{|\mathcal C_k|}
\frac{a_{k,y_i,e_i}}{n_{k,y_i,e_i}}.
\]

对均匀 minibatch 使用：

\[
\widehat{\mathcal L}_{BER}
=
\frac{N_k}{B}\sum_{i\in batch}w_{k,i}CE_i,
\]

这是全数据 BER 的随机估计，避免“少数组没有进入当前 batch 就无法平衡”的问题。
组计数只在客户端本地使用，不上传服务器。

### 2.3 CDep：Conditional Environment Dependence Penalty

CDep 只在同一类别内部测量 representation 和 environment 的相关性。

Oracle 模式使用 environment one-hot；learned 模式使用 PEW 的连续环境 embedding。
任务表示先经过固定随机投影：

\[
z_i^p=z_iP_k,
\]

其中 `P_k` 根据 seed 固定，不参与训练，因此不能通过 projection-head collapse 伪造低依赖。

对每个有效类别计算标准化条件交叉协方差：

\[
\mathcal L_{CDep}
=
\frac{1}{|\mathcal C_k^{valid}|}
\sum_c
\operatorname{mean}
\left(
\operatorname{Cov}(\bar z^p,\bar e\mid Y=c)^2
\right).
\]

它只能表述为降低类内线性环境依赖 surrogate，不能声称严格条件独立。

### 2.4 EBST：Environment-Balanced Structural Transfer

EBST 的通信对象是客户端内标准化的 `class x environment x class` margin relation。
服务器先在相同环境内聚合客户端，再在环境之间等权形成共识，并使用跨环境方差门控过滤
shortcut-heavy 关系。

客户端上传的不是模型参数或原始特征，而是按真实类别和环境累计的分类边界 margin：

\[
R_{k,c,e,j}=\mathbb E[z_{k,c}(x)-z_{k,j}(x)\mid y=c,e],
\]

每个 `(c,e)` 行先做 z-score，降低异构模型 logit 尺度差异。服务器先对同一环境内的客户端
等权聚合，再对有效环境等权聚合。stability gate 使用跨环境归一化方差：

\[
G_{c,j}=\exp(-\widehat{\operatorname{Var}}_e(R_{c,e,j})/\tau_g).
\]

客户端通过 gated Huber loss 对齐分类头的相对 margin；主干特征在该通信损失中 detach，
从而将跨模型通信限制在共享的类别关系空间。

EBST 的目标是帮助本地有少量该类别的客户端，不能声称凭空恢复完全缺失类别的视觉知识。

### 2.5 SCP：Safe Communication Projection

SCP 是可选负迁移保护，不是主要创新。它比较主任务梯度 `g_p` 与 EBST 通信梯度 `g_c`。
若 `g_p^T g_c < 0`，则把通信梯度投影到不与主任务冲突的半空间：

\[
g_c' = g_c - \frac{g_p^T g_c}{\|g_p\|_2^2}g_p.
\]

当前实现只对分类头执行 SCP，并记录 conflict rate、gradient cosine 和 projection ratio。

## 3. 当前数据是否需要重做

第一阶段不需要重做数据。当前直接使用：

```text
local_runs/cle_hfl_prepared/
  cle_hfl_prepared_alpha05_gamma09_seed0/
```

其中已经包含：

```text
client_i/train_images.npy
client_i/train_labels.npy
client_i/train_corruption_ids.npy
client_i/train_corruption_method_ids.npy
test_balanced/*
cifar_100/cifar-100-python.tar.gz
```

Oracle BER/CDep 只需要 train image、label 和 corruption group ID。

`clean/same/random/swapped/unseen` 评估协议已经生成；这属于补充测试划分，
不是更换 CIFAR-10/CIFAR-100 数据源。OpenI 上传包为：

```text
local_runs/cle_hfl_prepared/fedease_cle_prepared_alpha05_gamma09_seed0.tar.gz
```

## 4. 2026-07-19 已实现内容

新增：

```text
fedprime/data/fedease.py
fedprime/methods/balanced_environment_risk.py
fedprime/methods/conditional_dependence.py
fedprime/methods/environment_witness.py
fedprime/methods/environment_structural_transfer.py
fedprime/methods/safe_communication_projection.py
fedprime/methods/local_fedease.py
fedprime/methods/fedease.py
fedprime/engine/cle_metrics.py
scripts/prepare_fedease_eval_data.py
scripts/analyze_fedease_probe.py
scripts/openi_fedease_entry.py
```

统一 runner 已支持：

```text
method_name: fedease
method.cl_module: fedease
method.communication: none | local_only | ebst
method.fedease.environment_mode: oracle | learned
```

已增加逐轮诊断：

```text
fedease_clean_ce
fedease_classification_loss
fedease_ber_loss
fedease_jsd_loss
fedease_dcl_loss
fedease_cdep_loss
fedease_cdep_valid_classes
fedease_cdep_mean_abs_covariance
fedease_ber_valid_groups
fedease_ebst_loss
fedease_scp_conflict
fedease_scp_gradient_cosine
fedease_stability_gate_mean
```

调试与 OpenI 正式配置：

```text
configs/debug_fedease_oracle_control.yaml
configs/debug_fedease_oracle_ber.yaml
configs/debug_fedease_oracle_cdep.yaml
configs/debug_fedease_oracle_ber_cdep.yaml
configs/debug_fedease_oracle_ebst.yaml
configs/debug_fedease_pew.yaml
configs/openi_v100_fedease_oracle_control_probe.yaml
configs/openi_v100_fedease_oracle_ber_cdep_probe.yaml
configs/openi_v100_fedease_pew_probe.yaml
configs/openi_v100_fedease_ebst_probe.yaml
configs/openi_v100_fedease_full.yaml
```

含义：

```text
A. RAHFL local control: AugMix/JSD + DCL
B. control + BER
C. control + CDep
D. control + BER + CDep
```

Oracle control 与 BER+CDep 两组均禁用通信，目的是先验证本地理论机制，不使用
AsymHFL、PCCD 或测试准确率路由。EBST probe 和 full 配置才启用新通信。

## 5. 验证状态

已完成：

```text
Python compile check: passed
FedEASE targeted unit/config tests: 19 passed
OpenI entry dry-run: passed
```

一个真实数据、四异构模型、两轮、每客户端每轮单 local batch 的 EBST smoke 已完成：

```text
round 1 avg_acc=12.11
round 1 worst_acc=9.38
EBST loss=0.4411
valid environment fraction=0.650
mean stability gate=0.406
SCP conflict rate=0.75
non-finite loss/gradient: none
evaluation splits: clean/same/random/swapped/unseen all executed
```

该 smoke 只证明数据、AugMix、DCL、BER、CDep、反向传播和日志接口能接通；
只测试极少 batch，准确率/WCCA/CFG 没有任何科研意义。

## 6. 已实现但尚未获得正式实验结论

```text
多 seed / 第二数据集
Oracle control vs Oracle BER+CDep 正式 12 轮结果
PEW 正式环境识别质量
EBST 相对 BER+CDep 的独立通信增益
40 轮完整方法结果
```

因此当前代码是“完整、可开关的候选实现”，但不是“已被实验验证有效的论文方法”。

## 7. 下一次实验门槛

先进行相同数据、模型、优化器和轮数下的：

```text
control vs BER+CDep
```

如果联合版在 random/balanced 测试上表现出：

```text
WCCA 上升
CFG 下降
Avg/Worst 不崩溃
clean accuracy 后续下降不超过约 1 point
```

再运行 PEW probe；只有 PEW 环境估计可靠时，才比较 Oracle EBST 与 learned PEW+EBST。

若 Oracle BER+CDep 都无法改善 WCCA/CFG，则即使 PEW/EBST/SCP 代码已经存在，也不应直接跑
40 轮完整版；应停止扩展实验并重新检查本地机制。
