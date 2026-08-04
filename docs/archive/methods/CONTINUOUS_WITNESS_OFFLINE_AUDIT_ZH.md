# Taxonomy-Free Continuous Nuisance Witness 离线审计

更新日期：2026-08-03

## 1. 目的

本次工作尝试把 FedEASE 中依赖固定
`clean/noise/blur/weather/digital/unknown` 标签的 PEW、BER 和 CDep，替换成
不读取 corruption 名称、类别、family、severity 或 seen/unseen 信息的连续
nuisance witness。

该工作只进行匹配的一步更新审计，不实现 12/40 轮联邦 runner，也不运行
AsymHFL。

## 2. 连续 witness

对输入图像 `x` 提取 22 维确定性连续描述：

```text
RGB channel mean/std                         6
horizontal/vertical gradient mean/std       4
Laplacian absolute response mean/std        2
saturation mean/std                         2
radial log-spectrum energy                  8
total                                      22
```

这些量描述图像的连续颜色、频率和空间退化状态。算法不把样本划入预定义
corruption 类别，也不读取 CLE-HFL v2 的 operator 元数据。

## 3. 连续风险与条件依赖

在类别 `c` 内标准化 witness `w`。令单样本分类损失为 `ell`，连续平衡风险为：

```text
L_CBR(c) = mean(ell | y=c)
           + rho * || Cov(w, ell | y=c) ||_2 / sqrt(dim(w))
```

第二项是风险对连续 nuisance 分布均值移动的一阶响应。总体目标对本地出现的
类别等权平均。

同时在类别内部约束模型决策向量 `z` 与 witness 的标准化交叉协方差：

```text
L_CDep(c) = || Cov(normalize(z), normalize(w) | y=c) ||_F^2
```

一步候选更新使用：

```text
L = L_CBR + 12 * L_JSD + L_DCL + lambda_cdep * L_CDep
```

## 4. 匹配控制

四个分支从完全相同的 checkpoint、fit batch 和随机状态出发：

```text
base:      原始 CE + JSD + DCL
true:      真实图像对应的连续 witness
shuffled:  在真实类别内部循环打乱 witness
random:    在类别内部随机正交混合，精确保留均值和完整协方差
```

operator ID 只在所有更新完成后用于计算 audit WCCA/CFG，绝不进入 witness、
loss 或分支选择。

## 5. RTX 3050 正式审计结果

```text
branch      Avg       Worst     WCCA    CFG       audit loss
base        87.8277   83.4000   0.0     88.0423   0.400166
true        87.8277   83.3333   0.0     86.7923   0.401964
shuffled    87.8444   83.6667   0.0     88.0423   0.401336
random      87.7111   83.0000   0.0     88.0423   0.402281
```

可审计 `client x class` 目标为 33 个。真实 witness 的类别 audit loss 同时优于
base、shuffled 和 random 的比例为：

```text
33.33% < frozen gate 60%
```

门槛结果：

```text
CFG 至少降低 0.25：       PASS
Worst 不低于全部控制：    FAIL
WCCA 不低于全部控制：     PASS（所有分支均为 0）
audit loss 低于全部控制： FAIL
目标成功率 >=60%：         FAIL
overall：                  NO-GO
```

## 6. 结论与边界

真实 witness 将 CFG 从 `88.04` 降到 `86.79`，说明连续描述捕捉到了一部分
corruption-label dependence；但该信号没有转化为稳定的 Worst、audit loss
或跨类别收益，因此不能进入 12 轮 local-only，更不能直接接入 AsymHFL。

本审计还有两个边界：

1. 复用的 RAHFL checkpoint 曾用全部本地训练样本训练，因此这是机制诊断，
   不是严格独立验证结果；
2. 本地 held-out audit 中极少数 class/operator 组合样本较少，导致所有分支
   WCCA 都为 0，WCCA 在本次一步审计中缺乏区分度。

上述边界不会推翻 NO-GO，因为平均 audit loss、Worst 和 33 个类别目标的
冻结门槛仍然失败。禁止仅调 `rho`、`lambda_cdep` 或描述维度绕过结论。

## 7. 实现位置

```text
fedprime/methods/continuous_nuisance.py
scripts/audit_continuous_nuisance.py
tests/test_continuous_nuisance.py
local_test_outputs/continuous_witness_audit_20260803/
```
