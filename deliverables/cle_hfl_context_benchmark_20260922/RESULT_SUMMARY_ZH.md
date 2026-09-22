# S2-v2 十臂 HFL Context Benchmark 分析

日期：2026-09-22

## 1. 输入与判定

```text
archive = cle_hfl_context_benchmark_outputs.tar.gz
bytes = 20,131,052
sha256 = BE84CE654B48F21A72890220D00ED62FCBD1BD72376FAD95ADAD2E722C40FB75
protocol = cle_hfl_context_table_v2
mode = benchmark
scientific_evidence = false
```

冻结判定：

```text
EXECUTION_BENCHMARK = PASS
FORMAL_UNDER_9H = NO_GO
PAIRING_INTEGRITY = FAIL_FOR_FEDTGP_AND_RHFL
FORMAL_AUTHORIZED = false
```

本结果只能用于执行、协议和成本审计，所有benchmark准确率、DSA与方法排序禁止进入论文。

## 2. 完整性

- 输入审计`PASS`：40,000个private samples、1,000个DSA sources、15个operators；
- 十臂均生成config、1行metrics、operator/group表和预测缓存；
- 十份config均为`rounds=1`、`max_local_batches=8`、`max_test_batches=1`；
- 十臂metrics无NaN；分析摘要含十行且明确`scientific_evidence=false`；
- GPU峰值最高为AugHFL的约10.60 GB，V100 32 GB显存充足。

## 3. 真实成本

总训练`7039.47 s = 117.32 min`，分析`191.91 s = 3.20 min`。逐臂round时间：

| Arm | 时间（min） | 峰值显存（MB） |
|---|---:|---:|
| RHFL adapter | 45.95 | 3617.1 |
| FedTGP adapter | 45.73 | 3958.9 |
| AugHFL fidelity | 3.13 | 10596.3 |
| Local/ERM | 3.05 | 2886.1 |
| FedDF fidelity | 2.98 | 2886.1 |
| RAHFL fidelity | 2.95 | 5393.4 |
| FCCL adapter | 2.90 | 2877.9 |
| KT-pFL fidelity | 2.90 | 2886.1 |
| FedMD adapter | 2.88 | 3619.2 |
| FedProto adapter | 2.85 | 2886.1 |

FedTGP的100个server prototype epochs和RHFL的private-data confidence/SCE路径主导成本。仅把
benchmark逐轮时间乘40就得到约78.2小时；Formal还把local batches由8增至16，并解除完整
test/audit上限，因此78.2小时只是乐观下限。用户剩余额度不足，不得启动当前十臂40轮Formal。

## 4. 公平性异常

Local、FedMD、FedProto、FedDF、KT-pFL、FCCL、AugHFL、RAHFL八臂的四客户端local trace hash
完全一致；FedTGP与RHFL彼此一致，但与前述八臂不同。

原因是FedTGP/RHFL在pre-local communication中遍历private fit loader，推进了DataLoader的独立
generator；当前`paired_local_rng`只重置全局RNG，没有恢复DataLoader generator state。因此两臂
随后local training读取了不同batch轨迹。该差异是可修复的工程混杂，但修复前不允许Formal。

最小修复原则：在每轮pre-local communication前保存每个private loader generator state，通信后
恢复，再进入local phase；post-local通信则应保存local结束后的state并在通信后恢复。修复后运行
至少2轮小型配对benchmark，要求10/10 arms的local trace一致。

## 5. FedProto首轮边界

FedProto benchmark与Local/ERM完全相同是当前实现的预期round-0行为，不是效应结论。代码在
`round_idx == 0`时尚无global prototypes并返回0；round 1以后才聚合prototype并用于local MSE。
因此单轮benchmark只能证明入口可执行，不能证明FedProto通信效果。修复后的验证至少需要2轮。

## 6. 下一步

1. 不运行S2-v2 Formal；
2. 修复private-loader generator state隔离并补2轮、低batch配对Kill Test；
3. 不因成本擅自减少FedTGP官方示例的server epochs或RHFL核心步骤；若需要精简表，必须重新定义
   context-table范围，不能把轻量改版冒充当前adapter；
4. 剩余额度优先用于M3或S1 benchmark，而非十臂长任务。
