# FedFalsify v0.1 离线审计指南

## 目的

FedFalsify 正式训练会引入私有 `fit/audit` 划分、客户端快照流转、类别级证据估计和梯度一致性计算。直接跑 12 或 40 轮成本高，而且可能再次陷入“先训练、后发现理论前提不成立”。

因此，编码正式 runner 前先验证三个问题：

1. 外来客户端的类别知识是否会在接收客户端环境中失效；
2. FRA 门能否覆盖足够多的类别与训练样本；
3. CMT 是否比直接 KD 安全，TAU 是否能预测真实一步更新收益。

## 审计边界

当前历史检查点训练时使用了全部私有训练数据，无法事后把其中一部分称为独立 audit 数据。为了避免伪造独立性，本审计采用：

- 历史 RAHFL 检查点：仅作为待审计模型；
- `test_same/client_k`：只用于离线机制验证；
- 客户端原训练数据：只提供拟合 batch 和估计未来 15% 分层 audit 的样本覆盖率。

`test_same` 标签绝不能进入正式 FedFalsify 训练路由，也不能被包装成论文中的无泄漏训练结果。

正式实验必须在训练开始前固定：

```text
D_k = D_k_fit union D_k_audit
D_k_fit intersection D_k_audit = empty
```

所有基线都必须使用相同的 `D_fit`，最终测试集保持完全不可见。

## 三项审计

### 1. Foreign Transfer Tensor

对每个 `source j -> receiver k -> class c`，在 receiver 的独立同环境样本上同时评估 source 与 receiver：

```text
T[j,k,c] = Acc(f_j; D_same[k,c])
```

外来环境生存差距为：

```text
Gap[j,c] = T[j,j,c] - mean_{k != j} T[j,k,c]
```

差距随 `gamma` 增大，说明客户端知识越来越依赖自己的损坏-类别映射。

### 2. Gate Coverage

用真实客户端标签计数投影未来 15% 分层 audit 划分，再从独立 `test_same` 预测池 bootstrap：

```text
paired advantage = mean(1[source correct] - 1[receiver correct])
```

保守优势再加入方差惩罚和小样本 shrinkage。报告：

- 有足够 audit 样本的类别比例；
- source-receiver-class 激活比例；
- 投影到实际 fit 样本后的通信覆盖率。

### 3. Exact One-Step Audit

从同一检查点分别执行一个冻结 BatchNorm 统计的参数更新：

```text
CE only
CE + fixed margin
CE + direct peer KD
CE + CMT
CMT only
```

在与证据 batch 不重叠的独立 audit-loss batch 上计算：

```text
delta = L_audit(before) - L_audit(after)
```

`delta > 0` 表示更新有利。通信项的真实净贡献必须使用：

```text
increment = delta(CE + communication) - delta(CE only)
```

不能把 CE 本身造成的下降算给通信模块。

TAU 是 audit CE 梯度与 CMT 梯度的余弦相似度。它只是一阶方向判据，不是完整训练收益保证。

## 当前执行结果

汇总文件：

```text
deliverables/fedfalsify_offline_audit/FEDFALSIFY_OFFLINE_AUDIT_ZH.md
deliverables/fedfalsify_offline_audit/fedfalsify_offline_audit_summary.csv
```

原始审计输出：

```text
outputs/fedfalsify_audit/foreign_tensor/
outputs/fedfalsify_audit/gate_coverage/
outputs/fedfalsify_audit/gate_coverage_min5/
outputs/fedfalsify_audit/one_step/
```

当前冻结结论是：问题信号成立，直接 KD 不安全，CMT 有小幅正作用，TAU 有筛选能力，但 FRA 覆盖率不足。暂不投入 40 轮训练。

追加的 v0.2 来源排序审计显示：

```text
TAU Top-1 coverage                = 100% / 100% / 100%
TAU Top-1 positive precision      = 91.4% / 94.3% / 85.7%
TAU Top-1 mean CMT-over-CE gain   = .00354 / .00367 / .00320
gamma                             = 0.0 / 0.6 / 0.9
```

因此下一候选不是放松所有安全约束，而是：

```text
TAU 作为动作安全门
每个 receiver-class 只选一个 TAU 最大的 source
FRA 从硬门降级为软排序先验或平局处理
```

低成本 head-only TAU 审计结果：

```text
gamma                              0.0      0.6      0.9
与 full TAU 的 Top-1 来源一致率    80.0%    62.9%    74.3%
positive precision                91.4%    91.4%    85.7%
mean CMT-over-CE increment        .00348   .00317   .00281
```

它略逊于 full-model TAU，但分类头参数只占四个异构模型总参数的约
`0.067%-0.224%`。如果进入 12 轮 probe，应默认采用 head-only TAU，
同时保留 full TAU 作为昂贵的机制上界。
