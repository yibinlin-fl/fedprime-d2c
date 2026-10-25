# CLE-v2 Moderate Axis II 与剂量响应 Formal 结果

Updated: 2026-10-25

## 结论

四个单臂 OpenI Formal 均完整结束，训练、恢复包、正式分析、四客户端 checkpoint、配置合同和
`scientific_evidence=true`均通过验收。结果支持两个结论：

1. 在固定 `alpha=1.0, rho=0.5, map2, seed0/split0` 的 moderate 协议中，ERM 的 DSA 随
   `gamma=0 -> 0.5 -> 0.9` 从接近零单调升高，支持 class--corruption binding 强度与
   directional shortcut 的剂量响应。
2. 在完全匹配的 `gamma=0.9` AsymHFL 通信下，PEW+BER 将 pooled DSA 从 `0.184514` 降至
   `0.090024`，绝对下降 `0.094490`，相对下降 `51.21%`，且四个异构客户端均下降。

该结果同时暴露真实边界：PEW+BER 没有消除 shortcut，且 clean accuracy 下降 `4.350` 个百分点、
`rho=.5` mixed accuracy 下降 `1.793` 个百分点。因此它支持“显著抑制 CLE，并改善多数 corrupted
评价”，不支持“无损”“完全消除”或“任意 HFL 通用插件”。

## 协议与完整性

```text
protocol: cle_v2_moderate_submission_v1
source protocol: cle_hfl_v2_moderate_alpha10_rho05_gamma_grid_seed0_split0_v1
private task: CIFAR-10
clients/models: ResNet10, ResNet12, ShuffleNet, MobileNetV2
map seed: 2
partition/training seed: 0/0
Dirichlet alpha: 1.0
private corruption rate rho: 0.5
rounds: 40
max local batches/client/round: 64
batch size: 64
public batches/round: 4
pretrain epochs: 0
input bytes: 893421220
input SHA256: DEA800EC1465C45037E74E4B948DFB8D778F1354E8522B551438C6A515C9E588
```

每臂均有 40 个 metrics rows（round 0--39）、160 条本地轨迹记录、4 个可下载 checkpoint、
完整 predictions 和正式 `RESULT_SUMMARY.json`。四份配置文件的 SHA256 均与各自合同一致。
16个checkpoint均已在本机CPU环境重新加载；ResNet10/ResNet12/ShuffleNet/MobileNetV2分别含
162/186/606/702个tensor，所有浮点tensor均为有限值。恢复包与正式包的共同训练文件逐文件一致；
唯一预期差异是`RUN_TIMING.json`由分析前的`training_complete_analysis_pending`更新为最终
`complete`。
正式产物没有内嵌 Git revision；任务启动记录指向远端 `main` 提交
`125580e705e08727419151337f97f3f7b854c5ab`，但这项代码身份不能仅凭下载包再次独立恢复，是
本轮唯一 provenance 限制，不要求为此重跑。

## 配对审计

- `asymhfl_erm` 与 `asymhfl_pew_ber` 的 160/160 条 round/client/batch、标签和图像哈希完全一致；
  两份完整 trace 的 SHA256 均为
  `5AEB50E9D1BBFC3F7F6C3081218F16AA10140148579EF4D0FC420CAE19A53DB7`。
- `gamma=0/0.5/0.9` 的 160/160 条 round/client/batch 与标签哈希一致；图像哈希全部随 gamma
  改变，符合只改变 corruption operator draw 的协议。
- 数据 manifest 记录 source indices、labels、corruption mask、severity draws、fit/audit split、
  initial states 均跨 gamma 固定；operator draw 只随 gamma 改变。
- 四份 prediction cache 均为 `[4 clients, 1000 sources, 15 operators, 10 classes]`，labels、binding
  与 operator names 完全一致。

## Moderate 剂量响应

| gamma | 目标 | DSA ↓ | shuffled-null p95 | p-value | Grid Acc. ↑ | Clean Acc. ↑ | Corrupt Acc. ↑ | Mixed Acc. ↑ |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | ERM | -0.000730 | 0.005108 | 0.546454 | 48.480 | 52.325 | 47.853 | 50.089 |
| 0.5 | ERM | 0.039679 | 0.008662 | 0.000999 | 48.395 | 54.350 | 47.930 | 51.140 |
| 0.9 | ERM | 0.184514 | 0.031311 | 0.000999 | 37.193 | 49.700 | 37.008 | 43.354 |
| 0.9 | PEW+BER | 0.090024 | 0.020321 | 0.000999 | 38.160 | 45.350 | 37.773 | 41.562 |

5,000 次 source-paired bootstrap 的 DSA 对比：

| 对比 | bootstrap mean | 95% CI |
|---|---:|---:|
| gamma .5 - gamma 0 | 0.040409 | [0.038967, 0.041950] |
| gamma .9 - gamma .5 | 0.144856 | [0.142085, 0.147653] |
| gamma .9 - gamma 0 | 0.185194 | [0.181629, 0.188850] |

这支持单调剂量响应，但不声称 DSA 与 gamma 线性。`gamma=0` 接近预注册零点且未超过 shuffled
null；`gamma=.5/.9` 均显著沿真实 binding 方向移动。

## AsymHFL 方法对照

| 方法 | DSA ↓ | Grid Acc. ↑ | Clean Acc. ↑ | Corrupt Acc. ↑ | Seen Acc. ↑ | Unseen Acc. ↑ | Mixed Acc. ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| AsymHFL--ERM | 0.184514 | 37.193 | 49.700 | 37.008 | 36.541 | 38.294 | 43.354 |
| AsymHFL+PEW+BER | 0.090024 | 38.160 | 45.350 | 37.773 | 37.461 | 38.631 | 41.562 |
| candidate - control | -0.094490 | +0.967 | -4.350 | +0.765 | +0.920 | +0.337 | -1.793 |

`DSA_ERM - DSA_PEW+BER` 的 5,000 次 source-paired bootstrap mean 为 `0.094456`，95% CI
为 `[0.091877, 0.097081]`。四客户端 DSA 下降分别为：

```text
ResNet10     0.034947
ResNet12     0.086444
ShuffleNet   0.113152
MobileNetV2  0.143417
```

因此 shortcut 抑制不是由单个架构驱动。另一方面，mixed accuracy 的客户端变化为
`[-3.573, +0.477, +1.510, -5.583]` 个百分点，说明任务效用并非架构一致地改善。

旧训练记录的 last-10 补充变化为：Avg `-0.620`、Worst `-1.695`、WCCA `+2.250`、CFG
`-7.502` 个百分点。正式论文的统一 checkpoint split inference 优先于这些 legacy logger 指标。

## 成本

| arm | training h | analysis h | total h |
|---|---:|---:|---:|
| asymhfl_erm | 10.730 | 2.287 | 13.018 |
| asymhfl_pew_ber | 10.444 | 2.325 | 12.769 |
| erm_g00 | 11.018 | 1.973 | 12.992 |
| erm_g05 | 11.155 | 1.903 | 13.058 |

分析约占每个任务 1.9--2.3 小时；它执行 paired grid、clean、balanced-corrupted、seen、unseen
五组正式推理和 shuffled-binding null，不是重复训练。

## 科学判定与边界

```text
execution: PASS (4/4)
artifact/config/checkpoint audit: PASS
matched AsymHFL A/B: PASS
moderate dose response: SUPPORTED at training seed 0
PEW+BER CLE suppression: SUPPORTED at training seed 0
utility-preserving claim: NOT SUPPORTED
universal plug-in claim: NOT SUPPORTED
training-seed/partition/dataset generality: NOT TESTED by this result
```

该 Formal 可回填 moderate dose-response 表和 Axis II 的两条 AsymHFL 行。它不能替代 strong
Axis I 三 seed 结果、local-first 多 seed、第二 private dataset 或 taxonomy mismatch 实验。
