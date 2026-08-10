# FedLENS-MPIE 径向有序表征确认审计

Updated: 2026-08-10

## 冻结背景

FedLENS-PIE v1 的 seed-0 Audit A 为 `NO-GO (6/7)`。唯一失败项是 held-out
severity Spearman `0.498202 < 0.500000`；检索、非坍缩和内容泄漏门槛均通过。
不得修改旧门槛、覆盖旧报告或追加随机种子追认 v1 成功。

MPIE 是一次结构性修订，不是阈值、维度或 loss weight 调参。它将连续环境表示写成：

```text
z(x) = r(x) * normalize(d(x))

direction d: 干预机制
radius r:    干预强度
```

公共训练为每个随机机制构造两张不同内容图像，并对每张图像生成严格有序的
`low < high` 干预。机制方向通过跨内容、跨强度配对学习；半径通过 pairwise ordinal
loss 学习。训练不读取 corruption family、五分类标签或公共语义标签。

## Matched control

确认审计必须同时训练：

```text
control:   PIE-v1 encoder + unordered loss
candidate: MPIE radial encoder + ordinal loss
```

两臂共享完全相同的四视图 batch（content A/B x severity low/high）、公共图像索引、
干预程序、训练 seed、optimizer steps、batch size、epoch、audit views 和评估代码。
Control 将 low/high 当作两个普通 program，不接收顺序关系；candidate 接收
`low < high`，并按方向/半径分解表征。

## 新确认划分

本确认审计在实现前冻结为：

```text
public split seed: 1
train images:      5000
audit images:      1000 (disjoint)
epochs:            5
embedding dim:     16
max chain length:  2
held-out operators:
  shot_noise
  motion_blur
  frost
  jpeg_compression
```

上述 concrete operator 身份仅用于生成与审计划分。family 映射不进入模型、loss、
路由或门槛。

## 冻结门槛

Candidate 必须首先通过原 Audit A 的全部绝对门槛：

```text
seen retrieval lift                  >= 5.0
held-out retrieval lift              >= 3.0
seen severity Spearman               >= 0.50
held-out severity Spearman           >= 0.50
active dimension fraction            >= 0.75
seen content probe accuracy          <= max(5%, 2 x chance)
held-out content probe accuracy      <= max(5%, 2 x chance)
```

此外必须通过 matched attribution 门槛：

```text
held-out severity delta (MPIE - PIE) >= +0.020
seen retrieval lift delta            >= -0.500
held-out retrieval lift delta        >= -0.500
```

只有绝对门槛和归因门槛全部通过，MPIE 才获得 Phase 2 PBR 的实现资格。任一失败即
`NO-GO`；不得在看到结果后改 split、seed、窗口或阈值。

## 当前边界

以下边界在确认结果产生前冻结：MPIE 不接入 CLE-HFL runner，不实现 PBR，也不改变
hard PEW + hard BER 的当前正式地位。后文结果不反向修改这些约束。

## 确认结果（2026-08-10）

seed-1 matched confirmatory Audit 已按上述冻结协议完成。两臂均使用相同的 5,000
训练图像、1,000 审计图像和四视图干预 batch：

```text
metric                       PIE control   MPIE candidate   delta
seen retrieval lift             5.776407        7.032147   +1.255741
held-out retrieval lift         4.688454        4.189092   -0.499362
seen severity Spearman          0.644638        0.517250   -0.127388
held-out severity Spearman      0.521280        0.447701   -0.073579
active dimension fraction       1.000000        1.000000    0.000000
seen content probe accuracy     0.013400        0.001675   -0.011725
held-out content accuracy       0.018364        0.008347   -0.010017
```

MPIE 的机制确实生效：最后一轮训练的平均半径从 low `1.219` 增长到 high `2.241`，
seen retrieval 也明显提高。因此失败不是空模块或实现未启用。它在新 held-out 算子上
的 severity Spearman 反而下降 `0.073579`，绝对门槛和归因门槛同时失败。

严格结论：`NO-GO`。冻结 MPIE，不继续调 ordinal margin、loss weight、维度、seed
或 operator split；PBR 不获得实现资格。隔离代码仅保留作可复现负证据，不接入当前
runner。

```text
local_runs/fedlens_mpie_confirmatory_seed1/confirmatory_report.json
local_runs/fedlens_mpie_confirmatory_seed1/pie_v1_matched.pt
local_runs/fedlens_mpie_confirmatory_seed1/mpie_v2.pt
```
