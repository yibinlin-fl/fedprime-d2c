# strict PEW + AsymHFL-val seed-0 正式结果

日期：2026-08-04

## 结论

严格 12 轮 A/B 的 seed-0 归因结论为 `GO`。candidate 相对 control 的
last-five 四项冻结门槛全部通过，但仍需匹配的多 seed 复验，不能直接作为
40 轮或最终论文方法结论。

## 核心结果

| 范围 | Avg | Worst | WCCA | CFG |
|---|---:|---:|---:|---:|
| control final | 31.3667 | 27.9667 | 1.5000 | 29.4000 |
| candidate final | 36.4933 | 30.9200 | 10.2500 | 21.6750 |
| final delta | +5.1267 | +2.9533 | +8.7500 | -7.7250 |
| control last-five | 30.0853 | 25.0427 | 0.8500 | 30.4400 |
| candidate last-five | 34.0230 | 28.9467 | 5.9000 | 24.1200 |
| last-five delta | **+3.9377** | **+3.9040** | **+5.0500** | **-6.3200** |

冻结门槛为 Avg `>= +1.5`、Worst `>= +1.0`、WCCA `>= 0`、CFG
`<= -1.0`；四项全部通过。

## 稳定性与扩展指标

- rounds 7-11 中，candidate 每轮都同时提高 Avg、Worst、WCCA，并降低 CFG。
- last-five `worst_group_acc +6.00`。
- last-five `worst_client_group_acc +6.46`。
- seen Avg/Worst：`+3.79/+3.95`。
- unseen Avg/Worst：`+4.34/+3.78`。
- round-0 `col_loss` 两臂完全相同：`0.2274829049905141`。

## 完整性

```text
archive: outputs/strict_pew_asymhfl_val_probe_outputs.tar.gz
sha256: 77109f7a382b1271317a3afd89a30ae27170e8003977225a3ace5dd7ace9f3d9
members: 30
rounds: control 0-11, candidate 0-11
missing gated metrics: 0
recomputed comparison: exact match
unsafe archive paths: none
```

## 下一步

运行匹配的 12 轮多 seed 复验。在复验确认前，不启动 40 轮实验，不形成
最终方法主张，也不通过盲目调整 lambda、阈值或 rank 抢救结果。
