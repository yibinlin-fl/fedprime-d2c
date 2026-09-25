# held-out map2 执行与配对验证（2026-09-25）

## 结论

```text
REAL_INPUT_AUDIT_PASS
FEDMD_FOUR_OBJECTIVE_SMOKE_PASS
FEDMD_FOUR_OBJECTIVE_PAIRING_PASS
HFL_TEN_ARM_TWO_ROUND_PAIRING_PASS
BENCHMARK_READY
FORMAL_NOT_AUTHORIZED
```

本报告只确认代码、输入、checkpoint、分析链和本地batch配对；所有smoke/pairing性能数值均不是
科学证据，不得写入论文结果表。

## 输入

```text
archive: local_runs/cle_v2_cross_scenario/cle_hfl_v2_cross_maps1_2_seed0_split0_with_pew.tar.gz
bytes: 1385820059
sha256: BEA8E98737BF881C701DCFFF05F4E04C3A1E6095B7CF7702A5177260C2F186F5
scenario_id: cle_hfl_v2_cross_map2_seed0_split0
manifest_sha256: A5D272149F6F31FC1CE117C6B936AA35FB12B423A1449A076DBD8A6094535196
pew_checkpoint_sha256: BC9FF7523B8474774B36E02507A544FEBD76363772E9B865191FD3119960DCBB
private_samples: 40000
paired_sources: 1000
operators: 15
```

输入审计状态为`PASS`。

## FedMD四目标smoke

固定FedMD symmetric public-logit exchange，真实运行：

```text
ERM
CVaR-DRO
PEW+GroupDRO
PEW+BER
```

每臂1轮、每客户端1个local batch。首次smoke发现PEW本地训练器的通信组合白名单未登记已有的
`fedmd`策略；ERM/CVaR能运行，两个PEW臂在构造阶段被拒绝。最小修复只把`fedmd`加入允许列表，
没有改变PEW、BER、GroupDRO、CVaR或FedMD的优化公式和训练逻辑，并新增实例化回归测试。

修复后4/4臂全部完成：

- 4个客户端checkpoint齐全；
- paired DSA分析成功；
- `paired_local_traces.matched=true`；
- CDep未使用；
- 科学判定固定为`SMOKE_ONLY_NO_SCIENTIFIC_DECISION`。

## 十个HFL方法两轮pairing

真实map2输入、training seed 0下运行：

```text
Local/ERM
FedMD
FedProto
FedTGP
FedDF
KT-pFL
FCCL
RHFL
AugHFL
RAHFL
```

每臂2轮、每客户端每轮2个local batches。审计结果：

```text
trace rows per arm: 8
10/10 arm_matches: true
all_arms_match: true
```

这确认所有方法使用相同私有batch顺序；通信阶段读取私有DataLoader不会再污染后续本地训练轨迹。

## 下一步

只授权benchmark：

1. FedMD四目标benchmark，用于估算四臂40轮成本；
2. HFL十臂按`cheap_a / cheap_b / fedtgp / rhfl`分片benchmark，用于估算各分片成本及双账号分配。

benchmark通过后仍需用户单独授权Formal。
