# CLE-HFL v2 Cross-Binding-Map map1 Formal 结果

日期：2026-09-12

## 结论

冻结判定为：

```text
GO_PEW_BER_CROSS_MAP1
```

在只改变客户端特定 class-operator binding map、其余训练和评价条件固定的第二张 CLE mapping
上，baseline 再次形成显著 directional shortcut，冻结的 coarse PEW+BER 再次显著降低 DSA。
I0、L0、C1、C2 四项预注册门槛全部通过。

这使论文可以主张：原固定场景中的 shortcut 形成与 PEW+BER 缓解不依赖唯一一张偶然 binding
map。它仍不证明跨 partition、训练 seed、corruption 库、severity、数据集或真实场景泛化。

## 协议

```text
mode: formal
scenario_id: cle_hfl_v2_cross_map1_seed0_split0
partition_seed: 0
binding_map_seed: 1
evaluation_seed: 20260909
train_seed: 0
rounds: 12
local_batches_per_client_round: 16
private_samples: 40000
paired grid: 1000 sources x 15 operators x 4 clients
base: AugMix/JSD/DCL + strict AsymHFL-val
plugin: base + frozen public PEW + hard BER
CDep: false
```

PEW checkpoint SHA256：
`BC9FF7523B8474774B36E02507A544FEBD76363772E9B865191FD3119960DCBB`。

## 冻结门槛与结果

| Gate | 冻结要求 | 结果 | 判定 |
|---|---|---:|---|
| I0 | 两臂输入、初始化和训练随机轨迹匹配 | 48 条 local trace 完全匹配 | PASS |
| L0 | 两臂 operator-grid pooled accuracy 均不低于 20% | Base 20.9450%，Plugin 21.4583% | PASS |
| C1 | Base DSA≥0.05，且高于 shuffled null p95，p≤0.01 | 0.113761 > 0.020252，p=0.000999 | PASS |
| C2 | DSA reduction≥0.02，CI95 下界>0，4/4 客户端同向 | 0.064230，CI95 [0.063105, 0.065328]，4/4 | PASS |

### DSA

```text
Base DSA:                 0.1137607581
PEW+BER DSA:              0.0495309845
absolute reduction:       0.0642297736
relative reduction:       56.46%
source-bootstrap CI95:    [0.0631047287, 0.0653276183]
```

逐客户端 DSA：

| Client | Base | PEW+BER | Reduction |
|---:|---:|---:|---:|
| 0 | 0.124739 | 0.060544 | 0.064195 |
| 1 | 0.078365 | 0.030683 | 0.047682 |
| 2 | 0.140930 | 0.055316 | 0.085614 |
| 3 | 0.111009 | 0.051581 | 0.059428 |

### 报告性任务指标

这些指标按冻结协议完整报告，但不是本次 cross-map 判定的主 gate。

| Last-5 metric | Base | PEW+BER | Plugin−Base |
|---|---:|---:|---:|
| Avg | 20.0337 | 20.7377 | +0.7040 |
| Worst | 15.5680 | 16.6440 | +1.0760 |
| WCCA | 0.6000 | 0.9500 | +0.3500 |
| CFG | 28.5000 | 18.6450 | -9.8550 |

本次新 map 的四项聚合任务指标方向均有利，但这不能覆盖原 Stage-2 的冻结整体 NO-GO，也不能
升级为“所有架构、所有场景无损”的结论。

## 与原 binding map 的联合解释

| 场景 | Base DSA | PEW+BER DSA | Reduction | Relative |
|---|---:|---:|---:|---:|
| 原 seed0_split0 map | 0.119644 | 0.041252 | 0.078392 | 65.52% |
| 新 cross-map1 | 0.113761 | 0.049531 | 0.064230 | 56.46% |

两张不同 binding map 中，Base 都形成约 0.11--0.12 的 directional shortcut，PEW+BER 都使其
下降超过 0.06，且每张 map 的 4/4 客户端都同向。这是“跨 binding-map 复现”，不是广义
cross-domain 或真实世界外推。

## 独立审计

- OpenI 输入审计：PASS；manifest SHA256
  `027C965117871CB9B4692594E8FAE28A23DAA66BADB469ECD3440EDDB4D54FEF`。
- 从 `CROSS_MAP1_PREDICTIONS.npz` 独立重算 DSA、逐客户端效应、bootstrap、shuffled-binding
  null 和 operator-grid accuracy，与平台 `RESULT_SUMMARY.json` 完全一致。
- 概率全部有限，最大概率和误差为 `2.3842e-7`。
- 两臂 48 条 paired local traces 完全匹配。
- Formal 总实测耗时 `8311.44 s`，约 `2.31 V100 GPU-hours`。
- 原始结果包大小 `4,457,718 bytes`，SHA256
  `8DA64B4E6669CE7534ADEA023E54EFAEB51353073EE85261E0F1768F2BD688E1`。

## 论文边界与下一步

允许新增的主张是：CLE shortcut 形成和 PEW+BER 缓解已在两张只改变 binding direction 的受控
mapping 上复现。不得写成跨全部 CLE 条件泛化，也不得把 source-bootstrap CI 当作训练 seed 或
场景级置信区间。

map2 尚未授权，也不是确认本次 GO 所必需的自动续跑项。是否补 map2 应由投稿证据预算决定；
当前应先把 map1 纳入主表、机制图、理论文档与论文初稿，再审计剩余最关键缺口。
