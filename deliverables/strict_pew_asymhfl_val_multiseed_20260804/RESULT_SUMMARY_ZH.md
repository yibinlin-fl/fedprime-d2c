# strict PEW + AsymHFL-val 三训练种子正式结果

日期：2026-08-04

## 结论

固定 CLE-HFL v2 `alpha05_gamma09_seed0_split0` 场景上的 training seeds
0/1/2 严格 12 轮 A/B 判定为 `GO`。三个 seed 均单独通过原四项完整门槛，
九项预注册多 seed 判据全部通过。

这排除了 seed-0 训练随机性的偶然正结果，但尚未检验跨 CLE 场景 seed
泛化，也不是 40 轮最终论文结果。

## 每个 seed 的 last-five delta

| Training seed | Avg | Worst | WCCA | CFG | 单 seed 完整门槛 |
|---:|---:|---:|---:|---:|:---:|
| 0 | +3.9377 | +3.9040 | +5.0500 | -6.3200 | PASS |
| 1 | +4.7977 | +3.8893 | +4.3500 | -8.3000 | PASS |
| 2 | +5.0287 | +4.8573 | +7.2500 | -5.5250 | PASS |
| mean | **+4.5880** | **+4.2169** | **+5.5500** | **-6.7150** | 3/3 |
| sample std | 0.5749 | 0.5547 | 1.5133 | 1.4290 | — |

## 三 seed 绝对 last-five 指标

| 方法 | Avg | Worst | WCCA | CFG |
|---|---:|---:|---:|---:|
| control mean | 30.2639 | 25.0436 | 0.6167 | 31.2317 |
| candidate mean | 34.8519 | 29.2604 | 6.1667 | 24.5167 |
| candidate − control | +4.5880 | +4.2169 | +5.5500 | -6.7150 |

control 的三 seed sample std 为 Avg `0.1633`、Worst `0.3520`、WCCA
`0.2082`、CFG `0.6860`；candidate 对应为 `0.7206`、`0.3021`、
`1.4189`、`1.4076`。

## 最后一轮快照

| Training seed | Avg delta | Worst delta | WCCA delta | CFG delta |
|---:|---:|---:|---:|---:|
| 0 | +5.1267 | +2.9533 | +8.7500 | -7.7250 |
| 1 | +5.5450 | +0.8333 | +6.7500 | -7.1000 |
| 2 | +6.5933 | +2.3067 | +10.7500 | -4.4500 |
| mean | +5.7550 | +2.0311 | +8.7500 | -6.4250 |

最后一轮四项方向在三个 seed 上仍全部有利。seed 1 的 final Worst 提升为
`+0.8333`，低于单 seed 门槛中的 `+1.0`，但冻结判据从一开始定义为
last-five 均值；其 seed-1 last-five Worst 提升为 `+3.8893`。因此不改变
正式判定，同时保留这一末轮波动作为 40 轮 durability 设计时的观察点。

## 扩展稳定性

- 15/15 个 last-five seed-round 均同时改善 Avg、Worst、WCCA，并降低 CFG。
- 三 seed mean `worst_group_acc +7.31`。
- 三 seed mean `worst_client_group_acc +7.20`。
- seen Avg/Worst：`+4.48/+4.28`。
- unseen Avg/Worst：`+4.90/+4.06`。
- 最弱 seed 的 Avg/Worst/WCCA 仍分别为 `+3.9377/+3.8893/+4.3500`。
- 最不利的 CFG delta 仍为 `-5.5250`。

## 预注册多 seed 判据

以下九项全部通过：

```text
mean Avg   >= +1.5
mean Worst >= +1.0
mean WCCA  >=  0.0
mean CFG   <= -1.0
every-seed Avg   > 0
every-seed Worst > 0
every-seed WCCA >= 0
every-seed CFG   < 0
at least 2/3 seeds pass the full original gate
```

实际为 3/3 seed 通过完整门槛。

## 完整性与公平性

```text
seed1 archive sha256:
7f4889fe20a11b7c446355b1b643fce1974f13a57f38e320c74a96a817cbd32e

seed2 archive sha256:
10d78afce660776cfcd95bcf0c12b420dac545c46b3810b00bfba44afd001eab

partition sha256 for seeds 0/1/2:
75c6bd9dc4b7714f505eea2c047f1b882582da311d00d099b6caac1b5ba4d2ec
```

- 两个新归档均为 30 个成员，无不安全路径或链接。
- 六条正式 arm 均完整包含 rounds 0-11，核心指标无缺失。
- seed 1/2 归档 comparison 与独立重算完全一致。
- 每个 seed 内 data、models、train、checkpoints、strict split 和通信协议匹配。
- candidate 仅增加预定的 calibrated PEW、BER 和 CDep 路径，并保留 DCL。

## 科学边界与下一步

当前结果支持：calibrated PEW + BER+CDep 在固定 strict CLE-HFL v2 场景和
AsymHFL-val 通信下，对训练随机种子具有稳定正贡献。

当前结果不支持：跨类别划分/算子映射泛化、40 轮长期稳定性或最终论文方法
声明。下一步可单独预注册 40 轮 durability probe；场景 seed 泛化应作为另一
个实验变量处理。现有 PEW/BER/CDep 参数应冻结，不根据本结果继续调参。
