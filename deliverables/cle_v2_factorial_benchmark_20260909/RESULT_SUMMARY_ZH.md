# CLE-v2 × PEW+BER 八臂 OpenI Benchmark 结果

日期：2026-09-09

## 判定

```text
benchmark_execution: PASS
integrity_audit: PASS
paired_local_traces: PASS
scientific_evidence: false
formal_authorized: false
```

本次 benchmark 证明八臂 OpenI 链路、冻结输入、PEW lineage、单进程 DataLoader 修复、
checkpoint、配置和配对训练轨迹均可执行。它不构成准确率、DSA、机制或插件效果证据。

## 输入与结果完整性

```text
archive: cle_v2_factorial_seed0_benchmark_outputs.tar.gz
bytes: 723935489
sha256: 3D3112775B69A794729F4391E8A45EFA3B31F738332D2EA3D90520BDA11194E4
data manifest: 6D48D332CE719F7C0ED3BFB52565B906936E446AB980AEE25CD342D564240C1E
PEW checkpoint: BC9FF7523B8474774B36E02507A544FEBD76363772E9B865191FD3119960DCBB
private samples: 40000
DSA grid: 1000 sources × 15 operators
```

完整性审计为 `PASS`。gamma00 四臂的 local batch/AugMix trace 完全相同；gamma09
四臂也完全相同。全部配置为 `num_workers=0, batch_size=64, rounds=1,
max_local_batches=8, max_test_batches=1, max_audit_batches=1`。

## V100S 成本

外层八臂运行耗时为 `1550.4681 s = 25.8411 min`。各臂 round time：

| arm | seconds | peak CUDA MB |
|---|---:|---:|
| h0_b | 175.5903 | 5393.37 |
| h9_b | 179.1000 | 5393.37 |
| l0_b | 175.4372 | 5388.61 |
| l9_b | 177.3235 | 5388.61 |
| h0_p | 178.9319 | 5394.32 |
| h9_p | 182.3481 | 5394.32 |
| l0_p | 181.6424 | 5386.69 |
| l9_p | 175.9387 | 5386.69 |

八臂 round time 合计 `1426.3121 s`，均值 `178.2890 s/arm`；baseline 均值
`176.8627 s`，plugin 均值 `179.7153 s`。峰值显存 `5394.32 MB`，显存不是瓶颈。

每客户端正式 fit loader 为 `floor(fit_size/64)=132` batches，而 benchmark 仅执行 8，
局部训练批次数放大 `16.5×`。按当前 round time 做一阶换算：

```text
1426.3121 s × 16.5 × 12 / 3600 = 78.4472 V100 GPU-hours
```

该数值只是量级估算，不是精确报价：benchmark 将 capped evaluation 一起放大，另一方面
Formal 会把 public batches 从 1 增至 4，并打开完整 audit/test/DSA，所以实际成本仍有显著
不确定性。合理规划量级约为 `80 GPU-hours`，不应直接启动完整 Formal。

## 边界与下一步

- benchmark 中的一轮准确率很低是预期现象，禁止作为论文结果。
- 当前未执行 paired DSA bootstrap、binding shuffle 或冻结六门槛判定。
- 不批准 `mode=formal`，不批准付费长任务。
- 下一步应先缩减无科学必要的计算/保存开销，或设计一个代表性 full-round 成本测试；任何
  协议调整必须保持最终 DSA、last-5 指标和冻结门槛不变，并重新获得用户批准。

