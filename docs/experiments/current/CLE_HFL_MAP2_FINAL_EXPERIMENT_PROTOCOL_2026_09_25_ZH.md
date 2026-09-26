# CLE-HFL held-out map2 最终实验协议（2026-09-25）

> 2026-09-27状态：领域表暂停新增Formal。已完成Local/FedMD/RAHFL-style显示约20--23%的共同
> 低学习水平；旧AugHFL包含486次非有限梯度跳过而无效；旧FedProto运行因每轮完整private-fit
> prototype扫描在第7轮被主动停止。先执行
> `CLE_HFL_LEARNING_FLOOR_KILL_TEST_2026_09_27_ZH.md`，再决定本表最终预算。
> 2026-09-27补充：修复后FedProto已通过真实map2两轮pairing，batch trace完全匹配；这只解除
> 工程阻塞，不恢复旧中止运行的科学效力。Local b32真实smoke也已通过，下一步仍是b32/b64
> benchmark，而不是直接启动Formal。

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

## B. 运行中的 FedMD 跨通信四目标复现

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

smoke、benchmark和配对审计已通过；用户已启动map2、training seed 0、40轮Formal。该实验回答
主方法排序是否依赖strict AsymHFL-val通信，不声称覆盖所有HFL通信机制。seed 1/2不自动启动：
若seed 0通过全部冻结门槛，它已承担跨通信复现；只有结果临界或正文要提出FedMD训练稳定性主张时
才追加。

## C. 待运行十行 practical HFL 领域比较

八个原生/协议匹配 HFL 行：

```text
Local/ERM
FedMD
FedProto
FedDF
KT-pFL
FCCL
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

八臂必须按 `local / fedmd / fedproto / feddf / kt_pfl / fccl / aughfl / rahfl` 单臂分片运行。
原`cheap_a/cheap_b`继续用于benchmark，不用于最终Formal打包。分片结果先由
`scripts/merge_cle_hfl_context_shards.py --profile practical`审计合并，再由
`scripts/merge_cle_hfl_domain_table.py` 与既有 AsymHFL seed-0 Formal 合并。最终合并要求场景、
轮数、training seed、batch预算、evaluation grid和本地batch轨迹全部一致。

FedTGP与RHFL benchmark单轮分别约45.5和46.3分钟，40轮各约31小时，当前移为条件性实验室
服务器扩展。不得降低其内部计算预算后混入40轮主表。原full profile与合并能力保留，未来长算力
齐备时仍可生成十二行表。

四目标FedMD中的ERM不能直接复用为领域表FedMD：前者明确使用`standard`本地loader，后者属于
领域表既有统一AugMix-view loader协议；虽然都不启用JSD/DCL，但resolved config并不等价，必须
独立运行。

该表回答现有 HFL 通信/鲁棒机制是否自然消除 CLE，以及本文方法在领域中的位置。它不是
untouched official-recipe leaderboard，也不识别 BER 的因果贡献；BER 的归因由四目标表承担。

## D. 其余投稿前证据

1. local-first factorial 的 training seeds 1/2：由于原 8 小时任务已停止，需先做成本重构，
   不得直接恢复长任务。
2. 第二受控私有数据集：验证现象与缓解不只存在于当前 CIFAR-10 类受控任务。
3. bounded taxonomy stress：只检验有限的 unseen/compound corruption 退化，不包装开放世界保证。
4. JTT Formal：仅当正文要给出直接 JTT 胜负数字时才需要。
