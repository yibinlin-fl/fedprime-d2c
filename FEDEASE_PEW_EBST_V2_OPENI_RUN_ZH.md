# FedEASE Learned PEW + EBST-v2 OpenI 运行说明

更新时间：2026-07-21

## 今晚实验只回答一个问题

在相同的 CLE-HFL `alpha=0.5, gamma=0.9, seed=0` 和 12 轮预算下，验证：

```text
learned PEW + BER + CDep + EBST-v2 + class-wise SCP
```

能否在不损害 learned PEW 本地结果的前提下，通过联邦通信继续改善
`Worst/WCCA/CFG`。

这不是 40 轮正式实验，也不重复运行 RAHFL、Oracle control 或 Oracle BER+CDep。

## 本次修正

```text
1. PEW 训练后恢复 public validation accuracy 最好的 epoch；
2. unknown threshold 不再固定为 0.55，而是在 public validation 上自动选择；
3. 通信只使用已经通过安全性探针的 EBST-v2；
4. 保持 3 轮 warmup、pair-qualified source、LOO teacher 和 class-wise SCP。
```

校准只使用合成的无标签 public corruption 任务标签，不读取 CIFAR-10 私有标签或测试标签。

## OpenI 表单

```text
计算资源：1 x V100 32GB
镜像：ubuntu22.04-cuda11.8.0-py310-torch2.1.0-tf2.14.0
数据集：openi_cle_rahfl_diagnostic
项目：fedprime-d2c
代码分支：main
启动文件：scripts/openi_fedease_entry.py
运行参数：--mode=pew_ebst_v2_probe
```

旧数据集中的 `cle_hfl_prepared_alpha05_gamma09_seed0.tar.gz` 足够，不需要重新上传。

## 必须出现的日志

PEW 阶段：

```text
[setup] restored best PEW checkpoint from epoch=...
[setup] calibrated PEW unknown threshold=...
[heartbeat] PEW inferred client=...
```

第 0 至 2 轮是通信 warmup。第 3 轮开始必须出现：

```text
[heartbeat] EBST-v2 LOO aggregated ...
valid_pairs=...
sources=...
ebst=...
scp_conflict=...
```

## Go/No-Go 判据

已存储的 learned PEW 本地 12 轮结果：

```text
Avg=40.3694
Worst=35.4225
WCCA=13.925
CFG=6.370
```

组合实验至少应满足：

```text
Avg >= 40.37
Worst > 35.42
WCCA >= 13.93
CFG <= 6.37
```

同时检查最后 5 轮均值，不能只看最后一轮偶然波动。若 Avg 明显回退，则停止当前硬分类
PEW + EBST 路线，不进入 40 轮；下一步转向 taxonomy-free continuous environment
embedding、Soft-BER 和 Soft-EBST 的正式重构。

## 自动回传

任务完成后会自动回传：

```text
fedease_pew_ebst_v2_probe_outputs.tar.gz
summary.csv
summary.md
```
