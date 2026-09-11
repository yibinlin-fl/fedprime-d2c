# CLE-v2 Native FedDF-fidelity × PEW+BER Formal 结果

日期：2026-09-11

## 结论

冻结 Formal verdict 为：

```text
NO_GO_PEW_BER_FEDDF_PLUGIN_SEED0
```

该判定不能改写为插件 Formal GO。它同时包含两个需要严格分开的发现：

1. **CLE directional shortcut 抑制显著成立。** PEW+BER 将 pooled DSA 从
   `0.136989` 降至 `0.029012`，绝对下降 `0.107977`（相对下降 `78.82%`）；
   source-paired bootstrap CI95 为 `[0.106498, 0.109417]`，4/4 客户端均同向。
2. **“通用插件且不损失平均任务效用”没有成立。** last-5 Avg 下降 `0.6660 pp`，
   因此 P2 失败；两臂 operator-grid pooled accuracy 都低于冻结的 `20%` 学习下限，
   因此 L0 失败。

论文中可以把它作为跨第二通信底座的机制证据和效用边界，但不能写成 PEW+BER 已被证明为
普适、无代价、即插即用的 HFL 插件。

## 冻结协议与完整性

```text
protocol: cle_v2_feddf_plugin_contract_v1
scenario: cle_hfl_v2_paired_factorial_seed0_split0_v1
training seed: 0
rounds: 12
local batches/client/round: 16
base: standard single-view CE + feddf_fidelity
candidate: PEW-grouped hard BER-weighted CE + identical feddf_fidelity
AugMix/JSD/DCL: disabled in both arms
CDep: disabled in both arms
scientific_evidence: true
```

输入审计为 PASS。两臂配置哈希与实际文件一致，48 条配对 local trace 完全匹配；预测缓存形状为
`[2,4,1000,15,10]`，所有值有限，概率和最大误差为 `2.384e-7`。独立复算的 DSA、bootstrap、
operator-grid accuracy、last-5 和 gate 均与平台 JSON 一致。

## 正式结果

### DSA

| 指标 | Base | PEW+BER | Candidate − Base / Reduction |
|---|---:|---:|---:|
| pooled DSA | 0.136989 | 0.029012 | reduction 0.107977 |
| relative DSA reduction | — | — | 78.82% |

逐客户端 DSA reduction：

```text
[0.150575, 0.074155, 0.053119, 0.154059]
```

source-paired bootstrap：

```text
CI95 [0.106498, 0.109417]
```

### 任务效用

| 指标 | Base | PEW+BER | Candidate − Base |
|---|---:|---:|---:|
| operator-grid pooled accuracy | 18.4917 | 18.4517 | -0.0400 pp |
| last-5 Avg | 18.2777 | 17.6117 | -0.6660 pp |
| last-5 Worst | 15.9080 | 16.1227 | +0.2147 pp |
| last-5 WCCA | 0.1000 | 0.0000 | -0.1000 pp |
| last-5 CFG | 30.7950 | 17.5600 | -13.2350 pp |

逐客户端 operator-grid candidate-minus-base：

```text
[+0.5200, +0.9133, -1.4667, -0.1267] pp
```

这表明 pooled operator-grid 几乎不变，但逐客户端效用仍不一致，不能建立 architecture-uniform
utility claim。

## 冻结门槛

| Gate | 条件摘要 | 结果 |
|---|---|---|
| I0 | 输入、哈希、轨迹、预测与分析完整 | PASS |
| L0 | 两臂 operator-grid pooled accuracy 均至少 20% | FAIL |
| P1 | DSA reduction ≥0.02，CI95 下界 >0，4/4 客户端同向 | PASS |
| P2 | last-5 Avg delta ≥0，且 operator-grid pooled 不劣于 1 pp | FAIL |

P2 中 operator-grid 子条件通过（`-0.04 pp`），但 last-5 Avg 子条件失败（`-0.666 pp`）。
L0 与 P2 失败后，P1 不能覆盖总 verdict。

## 允许与禁止的论文表述

允许：

- 在固定 CLE-v2 强绑定场景、training seed 0 下，PEW+BER 的 directional shortcut 抑制从
  AsymHFL 迁移到了 native-CE FedDF-fidelity 协议。
- 该迁移体现在 DSA 大幅下降、4/4 客户端同向和严格为正的 source-bootstrap CI。
- shortcut 抑制与任务效用改善不是同一命题；FedDF 实验显示二者可能解耦。

禁止：

- “PEW+BER 是经 Formal 验证的通用无损插件”。
- “PEW+BER 在所有架构/客户端上改善准确率”。
- “FedDF 官方完整 recipe 已被复现”。当前仅为 protocol-matched `feddf_fidelity` adapter。
- 用 smoke、benchmark、事后改门槛或追加 seed 覆盖冻结 NO-GO。

## 与已有 AsymHFL 结果的联合解释

```text
AsymHFL: DSA 0.119644 -> 0.041252 (reduction 65.52%), last-5 Avg +1.7323 pp
FedDF:   DSA 0.136989 -> 0.029012 (reduction 78.82%), last-5 Avg -0.6660 pp
```

两种底座都支持“PEW+BER 能抑制固定强 CLE directional shortcut”；但准确率收益依赖底座/
客户端配置，故最终定位应是“具有跨通信底座捷径抑制证据、同时存在效用权衡的 taxonomy-assisted
本地缓解模块”，而不是“普适性能增强插件”。

## 成本与原始产物

```text
training: 1400.8453 s
analysis: 43.4742 s
total: 1444.3195 s = 0.4012 V100 GPU-hours
peak CUDA memory: 2253.138 MB (both arms)
archive bytes: 4,496,390
archive SHA256: A6328074227F08A6C89BF68727C4725CEEE826EB9D08DB09626C74A241E20F38
```

原始包：

```text
outputs/openi_downloads/cle_v2_feddf_plugin_seed0_formal/
  cle_v2_feddf_plugin_seed0_formal_outputs.tar.gz
```
