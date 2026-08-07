# CLE CDep 12轮敏感性结果（2026-08-07）

## 完整性与公平性

- 压缩包：`outputs/cle_sensitivity_12round_outputs.tar.gz`
- 三个 arm 均包含 round 0--11，无缺轮。
- `base`、`cdep_l001`、`cdep_l010` 除实验名和
  `method.fedease.cdep.lambda` 外配置一致。
- 四个客户端的 PEW 注释文件在三个 arm 间逐字节一致。
- `base` 的核心训练指标复现此前 A3 full；运行时间/显存存在正常波动。

## 最后5轮结果

| 方法 | CDep λ | Avg | Worst | WCCA | CFG |
|---|---:|---:|---:|---:|---:|
| BER-only A1 | 0 | 34.6320 | 29.4280 | 7.2500 | 24.6400 |
| CDep 0.01 | 0.01 | 34.1847 | 29.1053 | 6.0000 | 24.4350 |
| CDep base | 0.05 | 34.0230 | 28.9467 | 5.9000 | 24.1200 |
| CDep 0.10 | 0.10 | 33.9827 | 29.0373 | 5.9000 | 25.2500 |

相对 BER-only：

| CDep λ | ΔAvg | ΔWorst | ΔWCCA | ΔCFG |
|---:|---:|---:|---:|---:|
| 0.01 | -0.4473 | -0.3227 | -1.2500 | -0.2050 |
| 0.05 | -0.6090 | -0.4813 | -1.3500 | -0.5200 |
| 0.10 | -0.6493 | -0.3907 | -1.3500 | +0.6100 |

CFG 越低越好。λ=0.01 是三个 CDep 设置中准确率最好的，但仍同时低于
BER-only 的 Avg、Worst 和 WCCA；其 CFG 改善只有 0.2050。λ=0.10 还使 CFG
恶化 0.6100。

## Seen/Unseen 诊断

相对 BER-only，三个 CDep 设置在 seen 和 private-unseen 的 Avg、Worst、WCCA
均下降。λ=0.01 的 private-unseen 差值为：

```text
Avg -0.7050, Worst -0.3950, WCCA -2.7000, CFG +0.6050
```

因此没有证据表明当前 CDep 帮助未见算子泛化。

## 机制诊断

最后5轮 CDep 代理量随 λ 增大而下降：

```text
λ=0.01: loss 0.3176, mean_abs_covariance 0.4424
λ=0.05: loss 0.3013, mean_abs_covariance 0.4237
λ=0.10: loss 0.2963, mean_abs_covariance 0.4181
```

这说明优化器确实在压低当前类条件线性依赖代理量，但代理量下降没有转化为
更好的稳健分类指标。问题不是 CDep 没有执行，而是当前 batch-local 代理目标
与最终稳健性收益没有形成可靠一致性。

## 决策

当前 CDep 形式未证明对 BER 有稳定增量贡献，不能作为已验证核心模块。若仍要
保留该研究方向，只允许一次有结构变化的 CDep-v2 对照（跨 batch 统计、PEW
置信度/支持度门控和 warm-up），并预注册相对 BER-only 的通过门槛；不得继续
只扫 λ。若 CDep-v2 仍失败，主方法冻结为 calibrated PEW + BER。
