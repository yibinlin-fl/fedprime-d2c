# FedEASE EBST-v2 OpenI 启动说明

更新时间：2026-07-20

## 实验目的

本次只验证修正后的通信模块：

```text
Oracle BER+CDep + EBST-v2 + source-agreement gate + class-wise SCP
```

它与已经跑完的 Oracle BER+CDep 本地基线使用相同数据、seed、模型、优化器和
12 轮训练预算。不要运行 `--mode=full`。

## OpenI 表单

```text
计算资源：1 x V100 32GB
镜像：ubuntu22.04-cuda11.8.0-py310-torch2.1.0-tf2.14.0
数据集：openi_cle_rahfl_diagnostic
项目：fedprime-d2c
代码分支：main
启动文件：scripts/openi_fedease_entry.py
运行参数：--mode=ebst_v2_probe
```

运行参数必须恰好使用两个半角连字符。旧数据集中的
`cle_hfl_prepared_alpha05_gamma09_seed0.tar.gz` 可以直接复用，不需要重新上传数据。

## 正常日志

轮次 0 至 2 是关系统计 warmup。第 3 轮开始应看到：

```text
[heartbeat] EBST-v2 LOO aggregated ...
valid_env=...
valid_pairs=...
sources=...
mean_gate=...
```

每轮本地日志还应包含：

```text
ebst=...
scp_conflict=...
gate=...
valid_pairs=...
sources=...
```

`valid_pairs` 和 `sources` 在正式通信轮应为有限且非零的数；所有 loss 必须为有限值。

## 自动回传

任务结束后入口会通过 c2net 自动回传：

```text
fedease_ebst_v2_probe_outputs.tar.gz
summary.csv
summary.md
```

下载压缩包后放入本地 `outputs/` 目录再进行分析。

## 继续条件

对照 Oracle BER+CDep 本地基线：

```text
Avg=41.6206
Worst=35.5175
WCCA=14.000
CFG=6.155
```

EBST-v2 至少达到以下条件才继续完整方法：

```text
Avg > 42.1
Worst > 36.0
WCCA >= 14.0
CFG <= 6.2
```

未通过时停止该通信路线，不通过单独调小 lambda 反复试跑。
