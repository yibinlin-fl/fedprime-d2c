# CLE K1-A Head-only SDMN OpenI Smoke 审计

日期：2026-09-02

## 结论

```text
SMOKE_ONLY_NO_SCIENTIFIC_DECISION
```

Smoke执行与产物完整性全部通过，允许进入只使用public surgery objective和anchor KL的数值
calibration。该结果不允许判断Targeted优于control，也不允许运行formal K1-A或完整训练。

## 归档

```text
file:   cle_k1_sdmn_headonly_seed0_smoke_outputs.tar.gz
bytes:  138746333
sha256: D75911EE280E5A9C3FF09E0B98DBB7C3411544ED50CA19431417007163C8F327
```

归档38个成员、28个文件，无绝对路径、`..`、符号链接或硬链接。`artifact_manifest.json`
全部文件的bytes与SHA256匹配，无缺失或篡改。

## 协议与资产

- OpenI input仍为535,256,689-byte Phase-B0 package，SHA256为
  `DFB766F6494A5F61AA16F45666EC250A30501066AB54D89C984CD2324293B9BC`。
- 16/16 H0/H9/L0/L9 checkpoint存在。
- CIFAR-100 public tar与CLE evaluation arrays存在。
- `full_training_performed=false`，`communication_modified=false`，`formal_locked=true`。
- mode与config均为`smoke`，`scientific_decision_allowed=false`。

## Public split

```text
discover: 8 unique
surgery:  16 unique
holdout:  16 unique
```

三组两两交集均为0，三个index array的独立hash重算均与manifest一致。公共标签未提取、未使用。

## Probe与control

```text
selected high-risk Bank-A probes: [17, 7]
sensitivity-matched random probes: [11, 1]
weights sum: 1.0
max |cos(targeted, sham)|: 0.120274 <= 0.2
```

## 优化轨迹

| arm | objective | max anchor KL | accepted | rollback |
|---|---:|---:|---:|---:|
| Targeted | 51.9535 -> 48.7139 -> 45.5894 | 0.001373 | 2/2 | 0 |
| Direction-Sham | 3.4621 -> 2.5832 -> 1.8403 | 0.002898 | 2/2 | 0 |
| Random-Probe | 32.0418 -> 29.2625 -> 26.6006 | 0.002536 | 2/2 | 0 |
| Generic Invariance | 88.5172 -> 84.3133 -> 80.2310 | 0.000871 | 2/2 | 0 |

全部objective与anchor有限，所有step不增并被trust region接受。

## Checkpoint与response

- Frozen checkpoint与原H9 client0逐tensor完全一致。
- Targeted、Direction-Sham、Random-Probe和Generic-Invariance均只改变
  `linear.weight`与`linear.bias`；所有backbone、BN和projector tensor逐值不变。
- 五份unseen-bank response均为`(16,2,10)`、有限、class-centered residual在`6e-6`以内。
- 从五份原始response独立重算S/Dcf/K/R，与`result.json`最大绝对误差为`0.0`。

Smoke中的R变化没有统计意义：只有一个client、一个fold、16个holdout carriers和2个unseen
probes，严禁据此比较方法优劣。

## 下一步

只运行：

```bash
python scripts/openi_cle_k1_sdmn_headonly_entry.py --mode=calibration
```

Calibration覆盖H9/L9共8个client与AB/BA两fold，只允许读取discover+surgery carriers及训练bank。
禁止读取holdout bank、CLE binding、DSA、WCCA、CFG或private test。返回结果经审计后，再冻结
formal maximum surgery steps和optimizer/backtracking contract。
