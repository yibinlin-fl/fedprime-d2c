# Strict PEW-LOO 12轮结果摘要

日期：2026-08-09

## 结论

三臂均完整运行 round 0--11。Strict PEW-LOO 的协议审计及全部四个预注册门槛
通过，结论为 `GO`。这证明 PEW+BER 在公共 PEW 训练和验证均未见四个目标算子
时，仍能显著优于同任务 RAHFL。

## Last-five

| 方法 | Avg | Worst | WCCA | CFG |
|---|---:|---:|---:|---:|
| RAHFL | 30.0853 | 25.0427 | 0.8500 | 30.4400 |
| 标准 PEW+BER | 34.6320 | 29.4280 | 7.2500 | 24.6400 |
| Strict-LOO PEW+BER | 34.9880 | 31.2973 | 5.4500 | 24.3300 |

Strict-LOO 减 RAHFL：

```text
Avg +4.9027, Worst +6.2547, WCCA +4.6000, CFG -6.1100
```

四个预注册门槛全部通过：

```text
Avg >= +1.5      PASS
Worst >= +1.0    PASS
WCCA >= 0        PASS
CFG <= -1.0      PASS
```

Strict-LOO 减标准 PEW+BER：

```text
Avg +0.3560, Worst +1.8693, WCCA -1.8000, CFG -0.3100
```

因此 Strict-LOO 相对标准 PEW 并非所有指标都更好：平均准确率、最差客户端和
CFG 更好，但 WCCA 低 1.8。此实验的主要用途是验证算子级泛化，不应被表述成
Strict-LOO 在所有指标上全面优于标准 PEW。

## 协议审计

留出的算子为：

```text
impulse_noise, zoom_blur, fog, pixelate
```

四个客户端的私有 fit 中上述算子计数全部为 0；Strict PEW 的公共训练和验证
operator pools 也全部排除了它们。标准与 Strict 配置除实验名、PEW checkpoint
和预注册的 exclude_operators 外一致。

独立读取三个 metrics.csv 后重新计算的 last-five 与压缩包内报告一致。三臂的
历史可复现性也成立：同任务 RAHFL 和标准 PEW+BER 精确复现了此前 A0/A1 的
last-five 数值。

## PEW诊断

| 指标 | 标准PEW | Strict-LOO PEW |
|---|---:|---:|
| 私有group accuracy | 62.2100 | 68.3225 |
| 公共验证环境准确率 | 57.4000 | 58.5000 |
| unknown AUROC | 0.8167 | 0.8210 |
| NLL | 1.0825 | 0.9968 |
| ECE | 0.0341 | 0.0370 |

Strict-LOO 没有因为少四个训练算子而崩溃，反而在多数识别诊断上略好。但这仍是
单个 scenario seed、单个 training seed 的结果，不应解释为统计显著提升。

## 论文可声称的边界

可以声称：PEW 不依赖看到测试中的完全相同 concrete corruption operators；在
四个已知 family 内，每个 family 留出一个算子后仍保持显著收益。

不能声称：PEW 已经适用于任意未知损坏、全新损坏 family、复合损坏或所有真实
世界域偏移。后续论文中应称为 `operator-level leave-one-out generalization`。

原始压缩包：`outputs/cle_pew_loo_12round_seed0_outputs.tar.gz`。
