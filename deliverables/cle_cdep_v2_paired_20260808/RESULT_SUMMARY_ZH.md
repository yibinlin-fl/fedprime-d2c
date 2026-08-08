# CDep-v2 共享 PEW 配对实验结果

日期：2026-08-08

## 结论

这次实验完成了严格的同 PEW 配对归因。Control 与 Candidate 均完整运行
0--11 轮，四个客户端的 PEW annotation NPZ 均逐字节一致。CDep-v2 的机制
确实激活，但相对 `calibrated PEW + BER` 的四个预注册 last-five 门槛全部失败。

最终决策：`NO-GO`。冻结 CDep-v1/CDep-v2，不再调 CDep 的结构、lambda 或
阈值。论文主方法的本地鲁棒模块冻结为 `calibrated PEW + BER`。

## 配对审计

```text
control:   calibrated PEW + BER
candidate: 同一个 calibrated PEW + BER + CDep-v2
scenario seed: 0
training seed: 0
rounds: 12 + 12
PEW checkpoint: outputs/pew_checkpoints/cle_cdep_v2_paired_seed0.pt
PEW annotations byte-identical: true (4/4 clients)
```

Resolved configs 只在实验名和 CDep-v2 开关/参数上不同。两臂 PEW 的 private
group accuracy 均为 62.21%，校准阈值均为 0.0，验证环境准确率均为 57.4%，
ECE 均为 0.03412，unknown AUROC 均为 0.81671。

## Last-five 主结果

```text
metric       control    CDep-v2    candidate-control
Avg          34.6320    34.4387    -0.1933
Worst        29.4280    29.2000    -0.2280
WCCA          7.2500     6.6500    -0.6000
CFG          24.6400    25.0850    +0.4450
```

预注册门槛及结果：

```text
Delta Avg   >=  0.0   failed
Delta Worst >=  0.0   failed
Delta WCCA  >=  0.0   failed
Delta CFG   <= -0.5   failed
overall              failed (0/4)
```

CFG 越低越好，因此 `+0.4450` 也是退化。最后一轮虽然 Avg、WCCA 略正，
但预注册决策窗口是 last-five，不能在看到结果后改用单轮数值。

## Seen / Unseen 诊断

```text
metric                candidate-control, last-five
seen Avg              -0.1882
seen Worst            -0.2182
seen WCCA             -0.6000
seen CFG              +0.5250
private-unseen Avg    -0.2075
private-unseen Worst  -0.2550
private-unseen WCCA   -0.5500
private-unseen CFG    +0.2300
```

CDep-v2 在 seen 与 private-unseen 两侧方向一致地退化，不存在“总体均值掩盖了
unseen 收益”的证据。

## 机制与效率

CDep-v2 并非未运行。Candidate last-five 的 CDep loss 为 0.04171，平均有效
环境组为 41.8807，buffer 样本数为 2730.86。平均每轮时间从 97.7426 秒增加
到 99.2407 秒，约增加 1.53%；峰值显存基本不变（约 5.39 GiB）。

因此失败应解释为“有效执行但没有带来增益”，而不是实现空转。

## 科研决策

1. 冻结 CDep-v1 和 CDep-v2；不再做 lambda、buffer、置信度阈值等局部调参。
2. 当前最终本地方法冻结为 `calibrated PEW + BER`。
3. 之前完整候选中的正收益主要由 BER 驱动；论文不能再把 CDep 写成已验证
   的核心贡献。
4. 下一步先把尚未运行的通信正交性、跨场景、压力网格和第二数据集入口中的
   Candidate 改成冻结的 `PEW + BER`，做聚焦测试后再提交 OpenI。
5. PEW 的全局未见损坏泛化仍需 strict PEW-LOO 单独验证；当前 unseen 仅能称为
   private-fit holdout。

## 证据

```text
outputs/cle_cdep_v2_paired_12round_outputs.tar.gz
outputs/cle_cdep_v2_paired_20260808/
outputs/cle_cdep_v2_paired_20260808/outputs/cle_cdep_v2_paired_comparison.json
```
