# CLE-HFL full-epoch Local 成本 benchmark（2026-10-09）

## 结论

OpenI `1 x V100 32GB` 上，held-out map2 的 `Local/ERM` 两轮完整 strict-fit epoch
benchmark 执行通过，但确认当前每账号约 130 积分不足以安全完成一条 40 轮单臂 Formal。

```text
training_seconds = 5032.9761 s = 83.883 min
analysis_seconds = 36.9806 s
round 0           = 2492.0652 s
round 1           = 2530.1851 s
mean per round    = 2511.1252 s = 41.852 min
linear 40-round training estimate = 100445.0 s = 27.90 h
V100 point estimate at 6 points/h = 167.4 points (training only)
```

考虑依赖安装、输入解压、完整 Formal 分析和波动后，单臂建议至少预留 `180` 积分；
130 积分只对应 21.67 个 V100 小时，不能直接启动当前 one-shot 40-round Formal。

## 协议核验

```text
scenario_id        = cle_hfl_v2_cross_map2_seed0_split0
binding_map_seed   = 2
partition_seed     = 0
training seed      = 0
rounds             = 2
pretrain_epochs    = 0
local_epochs       = 1
max_local_batches  = null
batch_size         = 64
public batches     = 4 / round
fit samples/client = 8498, 8500, 8499, 8497
batches/client     = 133 / round
optimizer steps    = 532 / round across four clients
```

输入审计、输出、配置、batch trace和分析均完整；`all_arms_match=true`。该任务是成本
benchmark，不是科学证据。两轮准确率、DSA等数值禁止进入论文结果表。

## 文件

```text
archive = outputs/openi_downloads/s2_hfl_context/cle_hfl_context_local_benchmark_outputs.tar.gz
size    = 2274837 bytes
SHA256  = 3362835993AF5062CA98AAAEF307D31B2C41CABF019F25785994161C9927952B
```

## 下一决策

在启动正式实验前二选一：

1. 每个账号继续累计到至少 180 积分，再按完整单臂运行；或
2. 先实现并验证能够保存模型、优化器、通信状态、DataLoader/RNG状态和batch trace的可审计
   分段续跑协议。

若考虑A100，应先用完全相同的两轮Local benchmark测量实际点数效率；不能仅凭显卡理论性能
假定它更省积分。
