# strict PEW + AsymHFL-val 多 seed 复验说明

Updated: 2026-08-04

## 目的

seed-0 strict 12-round A/B 已通过冻结门槛。当前复验只检验该提升是否对
训练初始化、数据顺序和训练随机性稳定，不改变 CLE 场景、fit/audit split、
方法参数或训练预算。

```text
control   = AugMix + JSD + DCL + strict AsymHFL-val
candidate = AugMix + JSD + DCL + calibrated PEW + BER+CDep
            + strict AsymHFL-val
```

## 冻结变量

seed 1/2 两组任务继续使用：

```text
dataset scenario: alpha05_gamma09_seed0_split0
dataset archive: cle_hfl_v2_prepared_alpha05_gamma09_seed0_split0.tar.gz
fit/audit split: outputs/partitions/strict_cle_v2_alpha05_gamma09_seed0_split0.npz
strict_fit_audit.seed: 0
models: ResNet10, ResNet12, ShuffleNet, MobileNetV2
rounds: 12
local epochs: 1
public batches per round: 4
```

只改变顶层 `seed` 和实验输出名。每个 seed 的 control/candidate 必须放在
同一个 OpenI 任务中用 `--mode=both` 顺序运行，保证匹配初始化逻辑生效。

## OpenI 任务 1：training seed 1

启动文件：

```text
scripts/openi_strict_pew_asymhfl_entry.py
```

参数：

```text
--mode=both --train_seed=1
```

预期配置：

```text
configs/openi_v100_rahfl_val_cle_v2_trainseed1_probe.yaml
configs/openi_v100_fedease_pew_asymhfl_val_cle_v2_trainseed1_probe.yaml
```

预期归档：

```text
strict_pew_asymhfl_val_trainseed1_probe_outputs.tar.gz
```

## OpenI 任务 2：training seed 2

使用相同启动文件，参数：

```text
--mode=both --train_seed=2
```

预期配置：

```text
configs/openi_v100_rahfl_val_cle_v2_trainseed2_probe.yaml
configs/openi_v100_fedease_pew_asymhfl_val_cle_v2_trainseed2_probe.yaml
```

预期归档：

```text
strict_pew_asymhfl_val_trainseed2_probe_outputs.tar.gz
```

两个 seed 建议创建为两个独立 OpenI 任务，避免单个任务超时或失败后全部重跑。
不要并行拆开同一个 seed 的 control 和 candidate。

## 单 seed 输出

入口会分别生成：

```text
outputs/strict_pew_asymhfl_val_trainseed1_comparison.json
outputs/strict_pew_asymhfl_val_trainseed2_comparison.json
```

下载后保留原始压缩包，将其放到本地 `outputs/`，再解压到互不覆盖的独立目录。

## 三 seed 汇总

seed 1/2 均完成后运行：

```bash
python scripts/analyze_strict_pew_asymhfl_multiseed.py \
  --comparison 0=outputs/strict_pew_asymhfl_val_probe_20260804/outputs/strict_pew_asymhfl_val_comparison.json \
  --comparison 1=<seed1解压目录>/outputs/strict_pew_asymhfl_val_trainseed1_comparison.json \
  --comparison 2=<seed2解压目录>/outputs/strict_pew_asymhfl_val_trainseed2_comparison.json \
  --output outputs/strict_pew_asymhfl_val_multiseed_comparison.json
```

## 预注册多 seed 判据

以下判据已经写入分析器，结果返回后不得修改：

```text
三 seed mean Avg   >= +1.5
三 seed mean Worst >= +1.0
三 seed mean WCCA  >=  0.0
三 seed mean CFG   <= -1.0

每个 seed Avg   > 0
每个 seed Worst > 0
每个 seed WCCA >= 0
每个 seed CFG   < 0

至少 2/3 seed 单独通过原四项完整门槛
```

全部通过才把多 seed 判定记为 `GO`。否则为 `NO-GO`，停止 40 轮升级，
不通过盲目调整 lambda、threshold 或 rank 抢救。

## 当前边界

本轮只做 training-seed 稳定性，不改变 CLE 的类别划分、算子映射或
fit/audit split。若 training seeds 0/1/2 通过，再单独设计场景 seed 泛化和
40 轮正式实验；不要把两类随机性混入当前归因复验。

## 2026-08-04 正式结果

training seeds 0/1/2 均完成，独立重算结果为：

```text
seed   Avg       Worst     WCCA      CFG       full gate
0      +3.9377   +3.9040   +5.0500   -6.3200   PASS
1      +4.7977   +3.8893   +4.3500   -8.3000   PASS
2      +5.0287   +4.8573   +7.2500   -5.5250   PASS
mean   +4.5880   +4.2169   +5.5500   -6.7150
```

三 seed 全部通过原完整门槛，九项预注册多 seed 判据全部通过，最终判定
为 `GO`。该结论证明固定 CLE 场景下的训练随机种子稳定性，但不等价于
跨场景泛化或 40 轮最终结果。详细报告：

```text
deliverables/strict_pew_asymhfl_val_multiseed_20260804/RESULT_SUMMARY_ZH.md
```
