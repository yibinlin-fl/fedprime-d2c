# CLE-v2 Moderate Axis II Robust Benchmark 审计

Date: 2026-10-13

## 结论

`axis2_robust` benchmark通过。AugHFL与RAHFL均在冻结moderate输入上完成2轮x8 local batches，
多视图AugMix/JSD路径、通信策略、DCL差异、输出和统一分析均符合合同。Benchmark只授权后续
Formal，不构成论文证据。

## 工程审计

```text
AugHFL: loader=augmix, communication=aughfl_fidelity, cl=none, lambda_jsd=12
RAHFL:  loader=augmix, communication=asymhfl_val, cl=dcl, lambda_jsd=12
skip_nonfinite=false
```

- 2/2 arms complete；各2行metrics、8行trace。
- local/communication loss全部有限，无非有限值跳过。
- 两臂trace SHA256完全一致：
  `AE1C036CBA84D8F785E21E41D35E802C1890FF161B46BEC9A4F052FBB1A8FA4E`。
- DSA、operator-grid、clean、balanced-corrupted、seen、unseen与rho05 mixed分析均生成。

## 成本

```text
training_seconds: 765.594
analysis_seconds: 122.334
total: 887.928 sec = 14.799 min
```

按`40/2 * 64/8 = 160`倍训练更新量机械外推，两臂Formal训练约34.03 V100小时；平均每臂
约17.01小时、约102积分，建议单臂至少预留110--120积分。该估计不含排队和安装波动。

## 数值边界

两轮benchmark准确率与DSA不得用于方法排序或写入论文。RAHFL的两轮DSA超过其shuffled null
也只说明分析链工作，不是最终shortcut结论。

## 优先级

由于robust两臂成本高，当前先完成更核心且较便宜的`axis2_asym` benchmark及
AsymHFL--ERM/AsymHFL+PEW+BER Formal，再根据剩余积分分别运行AugHFL和RAHFL单臂Formal。
