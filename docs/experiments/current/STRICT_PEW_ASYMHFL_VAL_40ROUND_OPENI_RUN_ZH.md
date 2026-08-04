# strict PEW + AsymHFL-val 40 轮 durability 运行说明

Updated: 2026-08-04

## 目的

training seeds 0/1/2 的 strict 12-round A/B 已全部通过。本实验只回答：
在不改变方法、数据、划分或通信协议时，seed-0 的优势能否维持到 40 轮。

```text
control   = AugMix + JSD + DCL + strict AsymHFL-val
candidate = AugMix + JSD + DCL + calibrated PEW + BER+CDep
            + strict AsymHFL-val
```

这是一项 durability attribution probe，不是新的调参实验。

## 冻结协议

```text
CLE scenario: alpha05_gamma09_seed0_split0
training seed: 0
fit/audit split: outputs/partitions/strict_cle_v2_alpha05_gamma09_seed0_split0.npz
strict_fit_audit.seed: 0
models: ResNet10, ResNet12, ShuffleNet, MobileNetV2
local epochs: 1
public batches per round: 4
rounds: 40
checkpoints.save_final: false
```

除 `rounds: 12 -> 40` 和实验输出名外，配置必须与已通过的 seed-0 12 轮
配置一致。不得加载 12 轮 checkpoint、不得 resume、不得修改 PEW、BER、
CDep、DCL、学习率或阈值。

## OpenI 设置

数据集继续使用：

```text
cle_hfl_v2_prepared_alpha05_gamma09_seed0_split0.tar.gz
```

启动文件：

```text
scripts/openi_strict_pew_asymhfl_40round_entry.py
```

运行参数：

```text
--mode=both
```

如果平台使用独立参数框，填写 `mode=both`。必须让 control 和 candidate
在同一个任务中顺序运行，不得拆开。

## 正式配置

```text
configs/openi_v100_rahfl_val_cle_v2_40round_probe.yaml
configs/openi_v100_fedease_pew_asymhfl_val_cle_v2_40round_probe.yaml
```

## 预期输出

```text
strict_pew_asymhfl_val_40round_seed0_outputs.tar.gz
outputs/strict_pew_asymhfl_val_40round_seed0_comparison.json
```

启动日志应包含：

```text
Methods: ['control', 'candidate']
Training seed: 0
Rounds: 40
CLE scenario/split: seed0/split0 (fixed)
```

## 预注册 durability 判据

主判据使用最后十轮 candidate-minus-control 均值：

```text
last-10 Avg   >= +1.5
last-10 Worst >= +1.0
last-10 WCCA  >=  0.0
last-10 CFG   <= -1.0
```

晚期崩溃保护使用最后五轮均值：

```text
last-5 Avg   > 0
last-5 Worst > 0
last-5 WCCA  >= 0
last-5 CFG   < 0
```

八项全部通过才判定 `GO`。任一项失败即为 `NO-GO`，不通过修改窗口、
lambda、threshold、rank 或选择有利轮次抢救。

## 下一步边界

如果 seed-0 40 轮 GO，再用同一 40 轮协议复验 training seeds 1/2。只有
三训练种子的 40 轮结果稳定，才讨论长期稳定方法主张。CLE scenario seed
泛化是另一独立变量，应在 durability 之后单独设计。
