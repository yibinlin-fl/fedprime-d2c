# CLE-HFL held-out map2 最终实验协议（2026-09-25）

## 冻结公共条件

```text
scenario_id: cle_hfl_v2_cross_map2_seed0_split0
binding_map_seed: 2
partition_seed: 0
evaluation_seed: 20260909
formal rounds: 40
formal local batches per client-round: 16
main field-table training seed: 0
```

所有比较必须复用相同初始状态、私有 batch 轨迹、公共数据、评价 grid 和 DSA 实现。Smoke 与
benchmark 只验证执行和成本，不能作为论文证据。

## A. 已完成主方法比较

固定 strict AsymHFL-val 通信，比较：

```text
ERM / CVaR-DRO / PEW+GroupDRO / PEW+BER
```

40轮、training seeds 0/1/2 已完成。该实验回答 BER 是否优于普通 ERM、generic tail-risk 和
共享同一 PEW 分组的 GroupDRO；它不是任意通信算法上的万能插件实验。

## B. 待运行 FedMD 跨通信四目标复现

固定 FedMD symmetric public-logit exchange，重复同一四目标：

```text
FedMD+ERM
FedMD+CVaR-DRO
FedMD+PEW+GroupDRO
FedMD+PEW+BER
```

入口：

```text
scripts/openi_cle_v2_fedmd_objectives_entry.py
```

必须先做 smoke/benchmark 和配对审计；Formal 仍需用户明确授权。该实验回答主方法排序是否依赖
strict AsymHFL-val 通信，不声称覆盖所有 HFL 通信机制。

## C. 待运行十二行 HFL 领域比较

十个原生/协议匹配 HFL 行：

```text
Local/ERM
FedMD
FedProto
FedTGP
FedDF
KT-pFL
FCCL
RHFL
AugHFL
RAHFL
```

再复用主方法 seed-0 的两行：

```text
AsymHFL--ERM
AsymHFL+PEW+BER
```

运行入口：

```text
scripts/openi_cle_hfl_context_entry.py
```

十臂可按 `cheap_a / cheap_b / fedtgp / rhfl` 分片跨两个相同 OpenI 环境运行。分片结果先由
`scripts/merge_cle_hfl_context_shards.py` 审计合并，再由
`scripts/merge_cle_hfl_domain_table.py` 与既有 AsymHFL seed-0 Formal 合并。最终合并要求场景、
轮数、training seed、evaluation grid 和本地 batch 轨迹全部一致。

该表回答现有 HFL 通信/鲁棒机制是否自然消除 CLE，以及本文方法在领域中的位置。它不是
untouched official-recipe leaderboard，也不识别 BER 的因果贡献；BER 的归因由四目标表承担。

## D. 其余投稿前证据

1. local-first factorial 的 training seeds 1/2：由于原 8 小时任务已停止，需先做成本重构，
   不得直接恢复长任务。
2. 第二受控私有数据集：验证现象与缓解不只存在于当前 CIFAR-10 类受控任务。
3. bounded taxonomy stress：只检验有限的 unseen/compound corruption 退化，不包装开放世界保证。
4. JTT Formal：仅当正文要给出直接 JTT 胜负数字时才需要。
