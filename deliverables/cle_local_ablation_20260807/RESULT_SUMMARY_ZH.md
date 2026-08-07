# CLE-HFL A0--A6 12轮消融结果

日期：2026-08-07

## 审计

- A0--A6均包含完整round 0--11。
- 所有arm均固定seed 0、strict fit/audit和AsymHFL-val。
- 生成配置与resolved配置完全一致。
- 独立复算与归档分析JSON完全一致。

## 最后五轮均值

| Arm | Avg | Worst | WCCA | CFG |
|---|---:|---:|---:|---:|
| A0 RAHFL | 30.0853 | 25.0427 | 0.8500 | 30.4400 |
| A1 BER-only | 34.6320 | 29.4280 | 7.2500 | 24.6400 |
| A2 CDep-only | 30.4070 | 24.7707 | 1.1500 | 30.7750 |
| A3 Full | 34.0230 | 28.9467 | 5.9000 | 24.1200 |
| A4 Uncalibrated PEW | 33.5820 | 28.6040 | 5.0500 | 27.0700 |
| A5 Shuffled PEW | 31.5437 | 26.0147 | 2.5500 | 37.3750 |
| A6 Oracle family | **35.1200** | **30.7253** | **7.7000** | **20.6900** |

## 关键归因

BER-only相对A0：

```text
Avg +4.5467, Worst +4.3853, WCCA +6.4000, CFG -5.8000
```

CDep-only相对A0：

```text
Avg +0.3217, Worst -0.2720, WCCA +0.3000, CFG +0.3350
```

Full相对BER-only（最后五轮）：

```text
Avg -0.6090, Worst -0.4813, WCCA -1.3500, CFG -0.5200
```

Full相对BER-only（最终轮）：

```text
Avg -0.3033, Worst -0.9733, WCCA +1.5000, CFG -3.1250
```

因此，BER是当前增益的主要来源。CDep单独没有形成有效提升；与BER组合后主要改善最终轮的WCCA/CFG，但最后五轮Avg、Worst、WCCA下降。当前证据不足以把CDep写成稳定的独立贡献。

Full相对固定阈值0.55：

```text
Avg +0.4410, Worst +0.3427, WCCA +0.8500, CFG -2.9500
```

自动校准阈值为0.0，PEW私有family诊断准确率为62.21%；固定0.55时准确率仅39.99%。校准PEW的贡献得到支持。

Full相对Shuffled PEW：

```text
Avg +2.4793, Worst +2.9320, WCCA +3.3500, CFG -13.2550
```

打乱样本与环境的对应后CFG严重恶化，说明提升依赖正确的环境结构，不只是增加一个随机正则项。

Oracle相对Full：

```text
Avg +1.0970, Worst +1.7787, WCCA +1.8000, CFG -3.4300
```

Oracle上界全面领先，说明PEW估计质量仍是可改进瓶颈；Oracle使用私有operator metadata，只用于分析，不能作为可部署方法。

## Seen / Unseen

A1 BER、A3 Full和A6 Oracle均同时改善seen与unseen。最后五轮：

```text
                 Seen Avg/Worst     Unseen Avg/Worst
A0 RAHFL         29.916 / 24.871    30.550 / 25.515
A1 BER           34.324 / 29.358    35.479 / 29.620
A3 Full          33.708 / 28.820    34.890 / 29.295
A6 Oracle        34.937 / 30.695    35.624 / 30.650
```

## 计算代价

各arm平均每轮约96--98秒、峰值显存约5.39 GiB，Full没有明显增加每轮计算或显存。

## 结论与下一步

1. PEW+BER是当前最清晰、最稳定的核心机制。
2. 校准和环境对应关系均有明确因果证据。
3. CDep需要通过已经预先准备的lambda敏感性实验进一步判定；不应仅凭当前结果宣称其稳定有效。
4. 下一任务建议运行 `openi_cle_sensitivity_entry.py --arms=base,cdep_l001,cdep_l010`，比较lambda 0.01/0.05/0.10。若CDep仍不能稳定超过BER-only，应考虑把主方法简化为PEW+BER，并将CDep降为可选分析模块。
