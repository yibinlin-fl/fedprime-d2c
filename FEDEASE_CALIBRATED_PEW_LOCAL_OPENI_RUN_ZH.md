# FedEASE 校准 PEW Local-only 控制实验

更新时间：2026-07-22

## 实验目的

本实验不再寻找新精度，而是隔离通信贡献。它与已经跑完的
`calibrated PEW + BER+CDep + EBST-v2` 使用完全相同的数据、seed、模型、PEW、
BER、CDep、本地训练和 12 轮预算，唯一差异是关闭 EBST-v2 和 SCP。

```text
calibrated PEW + BER+CDep local-only
```

## OpenI 表单

```text
计算资源：1 x V100 32GB
镜像：ubuntu22.04-cuda11.8.0-py310-torch2.1.0-tf2.14.0
数据集：openi_cle_rahfl_diagnostic
项目：fedprime-d2c
代码分支：main
启动文件：scripts/openi_fedease_entry.py
运行参数：--mode=pew_calibrated_local_probe
```

旧数据集可直接复用，不需要重新上传。

## 正常日志

PEW 阶段应出现：

```text
[setup] restored best PEW checkpoint from epoch=...
[setup] calibrated PEW unknown threshold=...
```

每轮协作阶段应出现：

```text
[heartbeat] communication disabled for local-only probe
```

所有轮次的 `ebst` 和 `col_loss` 应为 `0`。

## 对照值与判定

已跑完的组合实验：

```text
final Avg/Worst/WCCA/CFG = 42.6331/35.2975/20.675/7.290
last-five mean           = 40.4526/35.9870/17.400/6.666
```

计算：

```text
EBST-v2 contribution = combination - calibrated local-only
```

只有当组合实验最后五轮 Avg 至少高 `0.5`，且 Worst/WCCA 不降低、CFG 不升高，
才认为 EBST-v2 通信具有值得继续的独立贡献。否则当前提升应主要归因于 PEW 校准，
停止硬分类 EBST 路线，不运行 40 轮。

任务结束后自动回传：

```text
fedease_pew_calibrated_local_probe_outputs.tar.gz
summary.csv
summary.md
```
