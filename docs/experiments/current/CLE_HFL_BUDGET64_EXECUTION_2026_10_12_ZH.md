# CLE-HFL 40轮×64 batches最终预算协议（2026-10-12）

## 决策

在任何新Formal结果产生前，将最终统一预算冻结为：

```yaml
pretrain_epochs: 0
rounds: 40
local_epochs: 1
max_local_batches: 64
batch_size: 64
public_batches_per_round: 4
skip_nonfinite: false
```

held-out map2每客户端完整strict-fit loader为133 batches，因此该预算对应每客户端约：

```text
40 * 64 / 133 = 19.248 local epoch equivalents
```

它保留40个通信轮，是旧16-batch协议的4倍本地更新量，同时约为full-epoch协议成本的一半。
选择发生在新Formal结果产生前，依据2026-10-09成本benchmark，而不是依据方法结果。

## 为什么替代两个旧预算

```text
旧16 batches/round：40轮约4.81 local epochs，学习不足，只保留为pilot。
旧133 batches/round：40轮约40 local epochs，Local单臂实测推算约27.9 V100小时/167.4积分。
新64 batches/round：40轮约19.25 local epochs，Local单臂预计约13.45 V100小时/80.7积分。
```

所有新主表比较必须使用相同64-batch预算。旧16-batch与133-batch结果均不得混入新表。

## Axis I：AsymHFL四目标主比较

固定strict AsymHFL-val通信：

```text
erm / cvar_dro / pew_groupdro / pew_ber
```

入口现支持每条arm独立执行：

```text
scripts/openi_cle_v2_spurious_final_entry.py
```

OpenI参数：

```ini
mode = benchmark | formal
train_seed = 0
arm = erm | cvar_dro | pew_groupdro | pew_ber
confirm_formal = false | true
data_source =
skip_install = false
```

每个单臂包独立保存输入审计、配置哈希、checkpoint、预测缓存、metrics与batch trace。四包到齐后
使用：

```text
scripts/merge_cle_v2_spurious_final_shards.py
```

合并器强制检查场景、partition、binding map、evaluation/training seed、40轮、64-batch预算、配置
哈希、输入指纹、评价grid和跨账号batch trace；只有四臂全部Formal且轨迹一致才计算冻结对比门槛。

## Axis II：HFL领域表

入口继续支持单臂分片：

```text
scripts/openi_cle_hfl_context_entry.py
```

正文优先：

```text
local / fedmd / fedproto / feddf / aughfl / rahfl
```

OpenI参数：

```ini
mode = benchmark | formal
shard = <单臂名>
confirm_formal = false | true
data_source =
skip_install = false
```

八臂practical合并能力仍保留，但当前优先六臂；KT-pFL/FCCL/FedTGP/RHFL是条件性扩展。

## 证据边界

- benchmark只验证执行与成本，不是论文证据；
- Formal单臂可以作为该方法的描述性结果，但Axis I胜负必须等待四臂严格合并；
- 不以官方RAHFL `40 pretrain + 40 communication`自居；官方recipe如需复现应单列；
- 主表报告学习曲线与last-10/last-5，审计是否仍处于明显快速上升期；
- 不因单臂中途结果修改轮数、batch预算、门槛或方法超参数。
