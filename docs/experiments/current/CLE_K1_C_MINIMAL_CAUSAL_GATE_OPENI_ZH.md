# CLE K1-C-Minimal：CRSF 最小因果干预门控

Updated: 2026-09-04

Implementation status: focused regression `14/14 PASS`; real H9/ResNet10 CUDA smoke completed with
`SMOKE_ONLY_NO_SCIENTIFIC_DECISION`. OpenI benchmark independently passed the engineering cost gate:
projected Minimal Formal `14.91 min / 0.2484 single-GPU hours`, with a conservative 30--45 minute
budget. This is still `BENCHMARK_ONLY_NO_SCIENTIFIC_DECISION`; no Formal scientific result exists.

## 状态与目的

旧版 K1-C-FULL 在任何 Formal 科学结果产生前被替代，状态冻结为：

```text
K1-C-FULL = SUPERSEDED_BEFORE_FORMAL
```

旧规范和实现保留作 provenance，不删除、不覆盖，也不再启动其 calibration/formal。
K1-C-Minimal 只回答一个因果问题：

```text
主动降低 response-spectrum concentration，真实 CLE 的 DSA 是否随之下降？
```

K1-C0 的 10/10 PASS 只是观察性前提：强 CLE checkpoint 存在异常响应谱集中；它不预测
CRSF 一定有效。Minimal 通过实际修改 late backbone block 检验因果箭头。

## 预结果冻结协议

唯一配置源：

```text
configs/cle_k1_c_minimal_seed0.json
```

Formal primary：

```text
systems:       H9 + L9
architectures: ResNet10(client0) + MobileNetV2(client3)
fold:          A -> B only
arms:          Frozen / CRSF / RawSpec
```

Correction 使用原冻结 `D_surgery` 2,000 池中由 seed `20260913` 无放回预选的 512 个
carrier，以及 Bank A 64 recipes 中由 seed `20260914` 预选的16个 probe。配置保存选择算法、
probe ids、完整池 hash、选择位置 hash、全局 index hash 和 Bank A/B hash；运行时必须重建并
逐项校验，禁止结果后改选。

优化固定为 Adam、初始 LR `1e-4`、5个 accepted steps、anchor KL `<=0.02`、每步最多12次
确定性 LR halving。每个 candidate update 后必须重新计算完整 `512x16` exact objective 和
512-carrier anchor KL；不满足 objective monotonicity 或 KL 时同时恢复参数和 Adam state。
不再进行独立三学习率 calibration。

## 严格未见评价

Correction 完成后，使用完整、未参与 correction 的 `D_holdout` 2,000 carriers 和 Bank B
全部64 probes 流式计算 Frozen/CRSF/RawSpec 的 `chi_unseen` 与 response energy。评价规模不
缩水，也不得用 Bank B 结果更新、回溯或选择模型。

全部 taxonomy-free 指标、moments/Gram、参数 delta、trace 与 source/input/selection manifest
写入后，生成 `primary_taxonomy_free_manifest.json` 封存。只有封存完成后才允许打开 Phase-B0
evaluation、CLE binding、真实 operator/family 和标签，计算 DSA 及报告性 WCCA/CFG/accuracy。

## 冻结 Gate

H9 与 L9 分别必须满足：

```text
mean CRSF unseen chi reduction >= 15%
ResNet10 与 MobileNetV2 两个 client 均为正
mean CRSF response-energy retention >= 50%
CRSF chi reduction - RawSpec chi reduction >= 10pp

DSA absolute reduction >= 0.05 OR relative reduction >= 25%
两个 selected client 的 DSA reduction 均为正
CRSF DSA reduction - RawSpec DSA reduction >= 0.02
```

WCCA、CFG、Avg、Worst、Clean 完整报告但不参与本次因果 kill gate，也不得用于调参。

全部 gate 通过：

```text
GO_TO_K1_C_MINIMAL_REPLICATION
```

只允许追加 `B->A + ResNet12 + ShuffleNet` replication，不允许直接进入40轮训练。

任一 gate 失败：

```text
NO_GO_CRSF_INTERVENTION
```

停止 CRSF，不补 fold/架构，不恢复 K1-C-FULL。

## 运行顺序与算力止损

```bash
# 1. execution-only
python scripts/openi_cle_k1_c_minimal_entry.py --mode=smoke

# 2. 受限平台计时；不作科学判断
python scripts/openi_cle_k1_c_minimal_entry.py --mode=benchmark

# 3. 只有 benchmark 成本经用户确认后才允许运行
python scripts/openi_cle_k1_c_minimal_entry.py --mode=formal --confirm-formal
```

OpenI 的键值运行参数界面必须填写：

```text
mode             formal
confirm-formal   true
```

CLI 裸开关与 OpenI 显式布尔值两种形式均受支持。

Benchmark 只运行 `H9 / ResNet10 / Bank A`，测一小步 CRSF/RawSpec 和受限 Bank B 流式前向，
输出阶段计时、GPU峰值显存、内存数组峰值估计、正式 correction/unseen/oracle 的 ETA 和单卡
GPU-hours。它不能读取 evaluation、标签、binding、DSA/WCCA/CFG，verdict 固定为
`BENCHMARK_ONLY_NO_SCIENTIFIC_DECISION`。

## 成本与方法边界

K0-B 当前只作离线审计。K1-C-Minimal 仍是 checkpoint-level causal kill test，不是最终训练
算法。即使通过，也必须先设计并验证 stochastic CRSF compression/efficiency gate；最终版本
要求推理零额外开销、通信近零、内存有界和现实训练开销。`512x16` 也不能未经验证直接塞入
每轮客户端训练。
