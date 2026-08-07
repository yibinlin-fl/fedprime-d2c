# CLE-HFL 外部基线 12 轮筛选结果

日期：2026-08-07

## 审计结论

- 七个 arm 均包含完整 round 0--11。
- 所有配置均为 `seed=0`、`cle_hfl_v2`、strict fit/audit，生成配置与 resolved 配置一致。
- 独立复算与归档中的 `cle_external_baselines_seed0_12round.json` 完全一致。
- final-test 仅用于报告；本结果是固定场景单训练种子的 12 轮筛选，不是多场景显著性结论。

## 最后五轮均值

| 方法 | Avg | Worst | WCCA | CFG |
|---|---:|---:|---:|---:|
| Local-only | 30.4367 | 25.7573 | 0.3000 | 30.8350 |
| FedMD | 23.2023 | 19.1293 | 0.4000 | 39.0150 |
| RHFL | 17.1053 | 14.9800 | 0.0000 | 36.4550 |
| FedProto | 24.1250 | 20.5493 | 0.7000 | 37.5850 |
| AugHFL | 26.6993 | 20.5587 | 0.0000 | 34.7150 |
| RAHFL | 30.0853 | 25.0427 | 0.8500 | 30.4400 |
| **Candidate** | **34.0230** | **28.9467** | **5.9000** | **24.1200** |

Candidate 相对 RAHFL：

```text
Avg   +3.9377
Worst +3.9040
WCCA  +5.0500
CFG   -6.3200
```

该差值与此前 strict seed-0 A/B 完全一致，说明通信接口重构没有改变正式 AsymHFL 对照结果。

## Seen / Unseen 最后五轮

| 方法 | Seen Avg | Seen Worst | Seen WCCA | Seen CFG | Unseen Avg | Unseen Worst | Unseen WCCA | Unseen CFG |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Local-only | 30.3909 | 25.7364 | 0.3000 | 30.7450 | 30.5625 | 25.8150 | 0.8500 | 18.0450 |
| FedMD | 22.9232 | 18.7418 | 0.4000 | 38.4150 | 23.9700 | 20.1950 | 2.0000 | 23.0350 |
| RHFL | 16.8959 | 14.8182 | 0.0000 | 36.3600 | 17.6812 | 15.2850 | 0.6500 | 21.5450 |
| FedProto | 23.7482 | 20.1873 | 0.7000 | 37.1550 | 25.1613 | 21.2350 | 3.4500 | 23.5700 |
| AugHFL | 26.6105 | 20.5236 | 0.0000 | 34.6550 | 26.9437 | 20.6550 | 0.5500 | 20.0150 |
| RAHFL | 29.9164 | 24.8709 | 0.9000 | 30.1000 | 30.5500 | 25.5150 | 1.9500 | 17.3350 |
| **Candidate** | **33.7077** | **28.8200** | **5.9000** | **23.5300** | **34.8900** | **29.2950** | **9.1500** | **13.5250** |

## 运行代价

| 方法 | 平均每轮秒数 | 12轮总分钟 | 峰值显存 MiB |
|---|---:|---:|---:|
| Local-only | 94.76 | 18.95 | 5389.2 |
| FedMD | 95.75 | 19.15 | 3767.3 |
| RHFL | 185.21 | 37.04 | 3766.4 |
| FedProto | 177.11 | 35.42 | 3941.5 |
| AugHFL | 95.79 | 19.16 | 3766.0 |
| RAHFL | 96.24 | 19.25 | 5394.9 |
| Candidate | 97.75 | 19.55 | 5394.6 |

Candidate 相对 RAHFL 平均每轮增加约 `1.51 s`（约 `1.6%`）。PEW 准备耗时约 `10.02 s`。RHFL 和 FedProto 的额外 fit 遍历导致每轮时间接近翻倍，正式论文需披露该统一实现下的计算口径。

## PEW诊断

```text
private group accuracy  62.21%
validation env accuracy 57.40%
ECE                     0.0341
NLL                     1.0825
unknown AUROC           0.8167
```

## 判断

1. Candidate 是本轮唯一在 Avg、Worst、WCCA、CFG 四项上同时明显超过 RAHFL 的方法。
2. Candidate 在 seen 与 unseen operator 上均领先，提升不是只来自 seen corruption。
3. FedMD、RHFL、FedProto、AugHFL 在当前 CLE 场景均低于 RAHFL，且多数低于 Local-only，说明无选择或不适配 CLE 的通信会传播低质量知识。
4. Local-only 最后五轮 Avg/Worst 略高于 RAHFL，但 WCCA/CFG 更差；RAHFL 最终轮四项均优于 Local-only。12轮短窗不足以声称 RAHFL 全面胜过 Local-only。
5. 目前不能写成完整 SOTA 结论：仅有固定场景 seed0，且尚缺 FedDF、KT-pFL、FCCL。下一步优先运行 A0--A6 消融，再决定基线多种子与缺失方法补充。
