# CLE-HFL held-out map2 最终实验协议（2026-09-25）

> 2026-09-28状态：七月配置复核已经定位共同低准确率的主因。旧领域表每轮16 batches、40轮仅
> 约4.8个本地epoch；七月`gamma=0.9`诊断是40轮、每轮完整10,000张私有数据，约40个epoch，
> Avg为46.72%。因此b32/b64试探被最终full-epoch目标取代，旧16-batch数据全部降级为pilot。
> 旧AugHFL的486次非有限梯度和旧FedProto全量额外扫描仍分别判为无效/中止；相应代码已修复。

## 冻结公共条件

```text
scenario_id: cle_hfl_v2_cross_map2_seed0_split0
binding_map_seed: 2
partition_seed: 0
evaluation_seed: 20260909
formal rounds: 40
local training per client-round: one complete strict-fit epoch
formal max_local_batches: null
batch size: 64
public batches per round: 4
pretrain epochs: 0
main field-table training seed: 0
```

所有比较必须复用相同初始状态、私有 batch 轨迹、公共数据、评价 grid 和 DSA 实现。Smoke 与
benchmark 只验证执行和成本，不能作为论文证据。

## A. Axis I：主方法比较（旧结果降级，full-epoch待重跑）

固定 strict AsymHFL-val 通信，比较：

```text
ERM / CVaR-DRO / PEW+GroupDRO / PEW+BER
```

旧40轮、training seeds 0/1/2使用每轮16 batches，仅作为low-budget机制筛选。最终表必须按
full-epoch协议重跑。该实验回答BER是否优于普通ERM、generic tail-risk和共享同一PEW分组的
GroupDRO；它不是任意通信算法上的万能插件实验。

## B. Axis I扩展：FedMD跨通信四目标复现（保留，full-epoch待重跑）

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

旧入口及结果使用16-batch预算，不再进入最终表。候选实验永久保留，后续按full-epoch协议先跑
seed 0；它回答主方法排序是否依赖strict AsymHFL-val通信，不声称覆盖所有HFL通信机制。seed 1/2
不自动启动，只有正文要提出FedMD训练稳定性主张时才追加。

## C. Axis II：待运行的full-epoch practical HFL领域比较

正文优先的六个原生/协议匹配HFL行：

```text
Local/ERM
FedMD
FedProto
FedDF
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

正文优先按`local / fedmd / fedproto / feddf / aughfl / rahfl`单臂分片运行。
原`cheap_a/cheap_b`继续用于benchmark，不用于最终Formal打包。分片结果先由
`scripts/merge_cle_hfl_context_shards.py --profile practical`审计合并，再由
`scripts/merge_cle_hfl_domain_table.py` 与既有 AsymHFL seed-0 Formal 合并。最终合并要求场景、
轮数、training seed、batch预算、evaluation grid和本地batch轨迹全部一致。

FedTGP与RHFL benchmark单轮分别约45.5和46.3分钟，40轮各约31小时，当前移为条件性实验室
服务器扩展。不得降低其内部计算预算后混入40轮主表。原full profile与合并能力保留，未来长算力
齐备时仍可补KT-pFL/FCCL/RHFL/FedTGP并生成扩展表。这些候选不得从项目记忆中删除。

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
