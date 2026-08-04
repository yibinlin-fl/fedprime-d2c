# FedFalsify OpenI 严格 A/B Probe 运行指南

## 1. 本次实验验证什么

本次不是 40 轮正式实验，而是 12 轮 Go/No-Go probe。它只回答：

> 在相同 `D_fit`、相同模型、相同优化器和相同随机种子下，基于
> `D_audit` 的 head-TAU Top-1 路由与 CMT 是否优于纯本地训练？

入口会依次运行：

1. `strict fit-only control`：只在 `D_fit` 上运行 AugMix + JSD + DCL；
2. `FedFalsify`：同样的本地训练，加 3 轮 warmup、head-TAU 路由和 CMT。

两组实验读取同一个固定 `.npz` 划分。最终测试集只用于每轮记录指标，
不参与教师选择。

## 2. 数据集

继续挂载已有数据集：

```text
openi_cle_rahfl_diagnostic
```

代码会在挂载目录中自动查找：

```text
cle_hfl_prepared_alpha05_gamma09_seed0.tar.gz
```

不需要重新上传数据。该压缩包中的 CIFAR-100 对本实验不会被通信模块使用；
FedFalsify 不依赖 public data。

## 3. OpenI 页面填写

推荐镜像：

```text
ubuntu22.04-cuda11.8.0-py310-torch2.1.0-tf2.14.0
```

资源：

```text
1 x V100 32 GB
```

启动文件：

```text
scripts/openi_fedfalsify_entry.py
```

运行参数留空即可。默认会安装依赖、导入数据、跑两组 12 轮实验、比较并回传。

若镜像已具备 requirements 中的依赖，可添加：

```text
--skip_install
```

不要填写四个连字符，例如错误的 `----skip_install`。

## 4. 关键配置

严格对照：

```text
configs/openi_v100_fedfalsify_fit_control_probe.yaml
```

FedFalsify：

```text
configs/openi_v100_fedfalsify_probe.yaml
```

共同设置：

```text
rounds: 12
local_epochs: 1
batch_size: 64
seed: 0
audit_ratio: 0.15
min_audit_per_class: 5
min_fit_per_class: 2
```

FedFalsify 独有设置：

```text
warmup_rounds: 3
fit_samples_per_class: 16
audit_samples_per_class: 16
teacher policy: positive head-TAU Top-1
lambda_cmt: 0.5
FRA weight: 0.0
```

FRA 只记录，不作为硬门控。这是离线审计得出的明确结论，而不是待搜索参数。

## 5. 正常日志

warmup：

```text
[heartbeat] FedFalsify warmup 1/3; CMT disabled
```

开始路由：

```text
[heartbeat] ... building receiver-side head-TAU routes
[heartbeat] FedFalsify routes=.../40 mean_tau=... elapsed=...s
```

每轮结果：

```text
[round 003] avg_acc=... worst_acc=... wcca=... cfg=...
local_loss=... cmt_loss=... routes=... tau=... elapsed=...s
```

第 0-2 轮 `cmt_loss=0`、`routes=0` 是正常 warmup。第 3 轮后应出现正路由数。

## 6. 自动输出

平台最终回传：

```text
fedfalsify_probe_outputs.tar.gz
fedfalsify_probe_comparison.json
```

压缩包包含：

```text
outputs/probe_fedfalsify_fit_control_alpha05_gamma09_seed0/
outputs/probe_fedfalsify_alpha05_gamma09_seed0/
outputs/partitions/fedfalsify_v1_cle_alpha05_gamma09_seed0.npz
outputs/fedfalsify_probe_comparison.json
```

不会保存模型 checkpoint，避免输出文件膨胀。

## 7. 冻结的 Go/No-Go 标准

比较 last-five mean，而不是只看最后一轮：

```text
Avg:   FedFalsify > control
Worst: FedFalsify >= control
WCCA:  FedFalsify >= control
CFG:   FedFalsify <= control
```

四项同时满足才进入 40 轮验证。若失败，则停止该通信路线，不通过盲目调
`lambda_cmt` 延长试错。

## 8. 当前本地验证

3050 两轮 debug 已通过：

```text
round 0: routes=0,  cmt_loss=0
round 1: routes=11, mean_tau=0.9473, cmt_loss=1.2705
tests: 14 passed
```

debug 每客户端只训练两个 batch，其准确率没有研究意义；它只证明数据划分、
路由、CMT、反向传播、评估和日志链路均正常。
