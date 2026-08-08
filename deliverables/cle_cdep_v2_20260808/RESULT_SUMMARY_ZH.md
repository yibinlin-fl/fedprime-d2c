# CDep-v2 12轮单臂结果（2026-08-08）

## 完整性

- 压缩包：`outputs/cle_cdep_v2_12round_outputs.tar.gz`
- round 0--11 完整，无缺轮。
- 自动报告、比较文件、配置、PEW诊断和四客户端注释齐全。
- 独立重算与自动报告最后5轮数值一致。

## 冻结门槛的机械结果

| 方法 | Avg | Worst | WCCA | CFG |
|---|---:|---:|---:|---:|
| 历史 PEW+BER A1 | 34.6320 | 29.4280 | 7.2500 | 24.6400 |
| CDep-v2 | 34.9927 | 29.6280 | 7.0000 | 23.8500 |
| 差值 | +0.3607 | +0.2000 | -0.2500 | -0.7900 |

自动四门槛通过三项，WCCA 非劣门槛失败 `0.25`，因此归档的机械结果为
`pass=false`。不得事后修改门槛或改用最后一轮覆盖最后5轮。

相对 RAHFL A0，CDep-v2 最后5轮为：

```text
Avg +4.9074, Worst +4.5853, WCCA +6.1500, CFG -6.5900
```

## CDep-v2 机制诊断

CDep-v2 正常执行，不是空模块：

```text
round 0/1/2 buffer: 1229.49 / 1939.55 / 2183.50（四客户端批次均值）
round 0/1/2 ramp:   0 / 0 / 0.3333
最后5轮有效类别:    7.0477
最后5轮有效环境组:  41.1299
最后5轮buffer:      2694.91
最后5轮CDep loss:   0.04103
```

最后5轮 seen 相对历史 PEW+BER 略有改善，但 private-unseen 略降：

```text
seen:          Avg +0.5264, Worst +0.2800, WCCA -0.2500, CFG -0.4700
private-unseen: Avg -0.0950, Worst -0.0200, WCCA -0.6000, CFG -0.4550
```

## 关键公平性缺陷

本次 CDep-v2 使用了新路径的 PEW checkpoint，重新训练出的 PEW 与历史 A1
并不相同：

```text
                    历史A1       CDep-v2
private group acc   62.210%      68.075%
unknown threshold    0.000        0.220
validation env acc  57.400%      54.400%
ECE                   0.0341       0.0689
unknown AUROC          0.8167       0.7902
```

四个客户端 `pew_predictions/client_*.npz` 的 SHA256 均不一致。因此历史 A1
与本次 CDep-v2 没有冻结相同 PEW checkpoint/注释，自动比较虽然按预注册公式
计算正确，但不能把差值因果归因给 CDep-v2。PEW质量也不是单方向全面改善，不能
简单做数值校正。

## 科学结论与下一步

当前结果应标记为 `INCONCLUSIVE_FOR_ATTRIBUTION`：

- CDep-v2 的实现、缓存、门控和损失均正常；
- 单臂结果接近历史 PEW+BER，并优于旧 CDep-v1；
- 但历史对照和候选的 PEW 注释不匹配；
- 不能据此保留或淘汰 CDep-v2。

下一次必须在同一个 OpenI 任务中只跑两个 arm：

```text
control:   同一新PEW + BER，CDep关闭
candidate: 同一新PEW + BER + CDep-v2
```

两个 arm 必须复用同一个 PEW checkpoint 与逐字节相同的注释，并保持模型初始
化、训练随机性、fit/audit、通信和评估一致。仍使用原四门槛，不得修改。无需再跑
CDep-v1。
