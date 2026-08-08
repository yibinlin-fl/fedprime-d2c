# CLE-HFL CDep-v2 单臂 OpenI 实验

更新日期：2026-08-07

## 目的

当前 batch-local CDep 在 λ=0.01/0.05/0.10 下均未超过匹配的
calibrated PEW+BER。CDep-v2 是最后一次结构性改造，只运行一个新 arm，复用
已有 PEW+BER 与 CDep-v1 结果，不重复消耗算力。

CDep-v2 使用客户端本地、跨 batch 的类别×环境特征缓存，以 PEW 置信度加权，
低支持组不参与，并采用两轮 warm-up 与三轮线性 ramp。缓存和统计不离开客户端，
只使用 fit 数据，不改变 AsymHFL-val 通信和 audit 路由。

## OpenI 填写

```text
代码分支：main
数据集：openi_cle_hfl_v2_alpha05_gamma09
启动文件：scripts/openi_cle_cdep_v2_entry.py
运行参数：留空
```

不要添加 `--skip_train` 或 `--no_upload`。

## 冻结配置

```text
scenario seed=0
training seed=0
rounds=12
fit/audit split=seed0_split0
communication=AsymHFL-val
projection_dim=64
lambda=1.0
buffer_size_per_group=64
min_confidence=0.20
min_group_count=4
min_environments=2
warmup_rounds=2
ramp_rounds=3
```

`min_confidence=0.20` 略高于六分类均匀概率 `1/6`；合格样本仍继续按连续
PEW 置信度加权。λ=1.0 是归一化质心偏移目标的自然系数，不进行扫参。

## 正常日志

前两轮应看到：

```text
cdep_v2_buffer=... cdep_v2_ramp=0.00
```

第2轮开始：

```text
cdep_v2_buffer=>0 cdep_v2_ramp=0.33 cdep=>0
```

随后 ramp 为 `0.67`，第4轮起为 `1.00`。若正式运行中缓存始终为0、有效组始终
为0或第4轮以后 CDep 始终为0，则结果不得用于方法判断，应先诊断。

## 输出与冻结门槛

最终自动上传：

```text
cle_cdep_v2_12round_outputs.tar.gz
```

压缩包包含自动比较：

```text
outputs/cle_cdep_v2_comparison.json
```

相对既有 matched calibrated PEW+BER A1 的最后5轮门槛：

```text
ΔAvg   >= 0
ΔWorst >= 0
ΔWCCA  >= 0
ΔCFG   <= -0.5
```

四项必须全部通过才保留 CDep-v2。失败则主方法冻结为 calibrated PEW+BER，
不再继续 CDep 调参或改造。
