# CLE-v2 Native FedDF × PEW/BER OpenI Benchmark

日期：2026-09-11

## 结论

```text
verdict: BENCHMARK_ONLY_NO_SCIENTIFIC_DECISION
execution_integrity: PASS
formal_authorized: false
```

本任务是1轮、每客户端8个local batches的V100成本与稳定性测试，不是科学实验。两臂为：

```text
fd_b = standard CE + FedDF-fidelity
fd_p = PEW-grouped hard BER-weighted CE + identical FedDF-fidelity
```

两臂均禁用AugMix/JSD/DCL/CDep。输入审计PASS；40,000个私有样本、1,000个paired source与15个
operator的输入清单一致。两份配置SHA256与冻结contract一致，4条standard single-view batch trace
完全匹配。预测缓存形状为`2 x 4 x 20 x 15 x 10`，全部有限，最大概率和误差`1.79e-7`。

## 执行与成本

```text
training_seconds: 88.6199
analysis_seconds: 8.2919
total_seconds: 96.9118
peak_cuda_memory_mb: 2224.13
FedDF server updates: 1 per arm
```

两臂round time为28.73/28.22秒；teacher entropy、teacher disagreement与FedDF loss均为有限值。
按12轮、16 local batches/client/round及完整评价保守估计Formal约0.5--1.0 V100 GPU-hour，实际值
必须由Formal timing记录。

## 仅用于异常筛查的数值

```text
DSA: base 0.012033, plugin 0.003176, reduction 0.008857
source-bootstrap CI95 [0.006390, 0.012240]
client reductions [0.024299, 0.010545, 0.000585, -0.000001]
operator-grid Avg: base 10.3333, plugin 9.1667
truncated reporting Avg: base 8.8867, plugin 8.9355
```

这些结果只使用20个source、1轮训练和截断测试，且两臂均未达到20%学习下限。分析器显示的P1/P2
失败不构成Formal NO-GO；同样，DSA方向为正和Avg近似持平也不构成GO。允许结论只有：代码、
配置、配对轨迹、FedDF通信、checkpoint推理、分析和成本链路均可执行，能够进入经用户明确批准的
12轮Formal。

原始包：

```text
cle_v2_feddf_plugin_seed0_benchmark_outputs.tar.gz
bytes: 94367
sha256: 828921F200FF6A862C2397ACA9C4F856E15C4CA6110DB70E8D45C862E18FE25D
```
