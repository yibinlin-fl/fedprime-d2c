# K1-C-Minimal OpenI 平台 Benchmark 审计

Date: 2026-09-04

## 结论

```text
verdict: BENCHMARK_ONLY_NO_SCIENTIFIC_DECISION
cost decision: PASS_TO_USER_FORMAL_APPROVAL
```

本结果只通过工程成本门，不证明 CRSF 有效。按本次 V100 实测线性外推，Minimal Formal 约
`894.39 s = 14.91 min = 0.2484 single-GPU hours`。由于只直接计时了 H9/ResNet10，并以代理比例
估计 oracle 阶段，正式运行应预留 2--3 倍裕量，即约 30--45 分钟。该成本已显著低于废止的
K1-C-FULL 路线，可以提交用户决定是否运行 Formal。

## 原始产物

```text
archive: outputs/openi_downloads/cle_k1_c_minimal_benchmark_seed0/cle_k1_c_minimal_seed0_benchmark_outputs.tar.gz
bytes:   19,398
sha256:  D16E82F85FFA636DBEE50086BF6A083F932BB1F8833F3F7E366F5E90AF24F2D4
```

## 实测与外推

```text
scope: H9 / ResNet10 / Bank A
correction: 512 carriers x 16 probes
active arms: CRSF + RawSpec, one accepted step each

correction one-step/two-arm wall time: 11.7909 s
prefix total:                         6.9237 s
sample unseen 128x8/three arms:       1.3286 s

projected four-context correction:    56.3491 s
projected full Bank-B unseen eval:    664.2975 s
projected oracle eval proxy:          173.7393 s
projected total:                      894.3860 s
projected single-GPU hours:           0.248441
```

资源峰值：

```text
bounded correction prefix arrays: 570,425,344 bytes = 544 MiB
CUDA peak allocated:              358,510,080 bytes = 341.9 MiB
sample unseen peak arrays:          9,961,472 bytes = 9.5 MiB
transformed-input disk cache:               0 bytes
```

## 数值与隔离检查

- CRSF 第一步 objective 从 `1.0` 降至 `0.981134`，anchor KL `0.001071 < 0.02`。
- RawSpec 第一步 objective 从 `1.0` 降至 `0.939968`，anchor KL `0.002066 < 0.02`。
- 两条干预臂均无 contract failure，且只修改冻结协议允许的 ResNet10 `layer4` 卷积权重。
- OpenI manifest 明确记录 `evaluation_extracted=false`、`training_performed=false`、
  `communication_modified=false`、`full_checkpoints_written=false`。
- `forbidden_assets_loaded=[]`；benchmark 没有读取标签、CLE binding、DSA/WCCA/CFG。
- artifact manifest 的 5 个文件全部重新计算 SHA256 一致。
- OpenI 的 `git_commit` 字段因运行环境无 `.git` 显示 `UNAVAILABLE`，但 source manifest 中6个
  源文件的 SHA256 均与本地提交 `c561e88` 工作树逐项一致，因此不存在代码版本漂移。

## 下一步边界

若用户接受成本，只允许运行：

```bash
python scripts/openi_cle_k1_c_minimal_entry.py --mode=formal --confirm-formal
```

Formal 才会回答 `chi_response下降是否伴随DSA下降`。Formal 失败即停止 CRSF；通过也只授权
B-to-A 与其余架构 replication，不授权直接进入完整联邦训练。
