# CLE-v2 Moderate Axis II Plain Benchmark 审计

Date: 2026-10-13

## 结论

`axis2_plain` benchmark通过工程门槛。Local/ERM、FedMD、FedProto和FedDF均在冻结moderate
输入上完成2轮、每客户端每轮8个local batches；四臂配置、输出、checkpoint派生预测、统一指标
分析和完成标记齐全。该结果只授权后续Formal，不构成论文科学证据。

## 协议

```text
protocol: cle_v2_moderate_submission_v1
source: alpha=1.0, rho=0.5, gamma=0.9, map2, partition/train seed0
rounds: 2
max_local_batches: 8
batch_size: 64
public_batches_per_round: 4
loader: standard single-view for all four arms
skip_nonfinite: false
```

## 完整性与配对

- `COMPLETION_axis2_plain.json`: 4/4 complete。
- 四臂均有2行metrics、8行local-batch trace，loss有限。
- 四个trace文件SHA256完全相同：
  `726C3E7D1BCE4965F4345B676B5FDCE556D971F34DED0A888822F71DC238B749`。
- 通信分别为none/FedMD/FedProto/FedDF；所有其他冻结训练预算一致。
- DSA、operator-grid、clean、balanced-corrupted、seen、unseen和rho05 mixed指标均成功生成。

## 成本

```text
training_seconds: 298.182
analysis_seconds: 221.839
total: 520.021 sec = 8.667 min
```

按`40/2 * 64/8 = 160`倍训练更新量机械外推，四臂Formal训练约13.25 V100小时，约79.5积分；
每臂平均约3.31小时/19.9积分。该外推只用于拆分规划，正式时间以实际任务为准。

## Benchmark数值边界

两轮仅16个local batches，模型仍接近初始化，准确率与DSA均不得写入论文或用于方法排序。
例如FedDF benchmark DSA较小但不超过其shuffled-null p95，这不能解释为FedDF消除了CLE。

## 下一步

`axis2_plain`四个方法已允许进入单臂Formal。只有两个有积分账号时，优先将每个方法作为独立任务，
由两个账号并行各跑一臂，完成后再运行余下两臂；禁止跨账号拼接同一方法的轮次。AugHFL/RAHFL
使用不同的多视图路径，仍需先通过`axis2_robust` benchmark，不能由本结果代替。
