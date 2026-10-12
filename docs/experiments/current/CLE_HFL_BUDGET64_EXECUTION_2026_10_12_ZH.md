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
新64 batches/round：40轮约19.25 local epochs；实际成本必须按各方法合法loader分别benchmark。
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

### Axis II本地训练链路（v6修复）

2026-10-12的首个`local` budget64 benchmark暴露出旧v5协议仍让`Local/ERM`使用默认AugMix
loader。即使`lambda_jsd=0`，代码仍生成多视图、执行三倍前向并计算零权重JSD，而且增强视图会
更新BatchNorm统计；因此该实现既浪费算力，也不能严格称为纯单视图ERM。该benchmark只作工程
诊断，其耗时与数值不得用于最终协议或论文。

v6在任何Axis II Formal之前冻结方法原生loader边界：

```text
standard单视图：Local / FedMD / FedProto / FedTGP / FedDF / KT-pFL / FCCL / RHFL
AugMix多视图： AugHFL / RAHFL
```

配对审计相应改为：同一loader家族要求完整图像batch哈希一致；跨loader家族要求标签batch轨迹
一致，并继续由合同锁定相同split、loader seed、初始化、训练seed和预算。禁止为了全表tensor
哈希一致而让普通HFL基线生成无效AugMix视图。修复后的standard与AugMix代表臂都必须重新完成
smoke/benchmark；旧v5 Axis II包禁止进入最终合并。

旧v5 `local`两轮产物的诊断数值为：训练`3831.226s`，round-1/2分别为`1866.961s`与
`1953.184s`，分析`35.665s`，显存峰值约`2.89GB`。batch trace显示4个图像视图，直接证实
默认AugMix loader泄漏。若错误地线性外推40轮约需`21.28h/127.7`个V100积分；这个估计仅说明
旧实现浪费程度，绝不能作为v6预算。作为参考，同日Axis I standard-loader PEW+BER两轮训练仅
`412.342s`，旧Local链路慢约`9.29x`。v6仍需重新benchmark获取合法成本。

v6修复后`local`两轮benchmark已PASS：训练`819.799s`，分析`45.094s`，两轮分别为
`332.911s`与`476.385s`，显存峰值约`1.29GB`；trace确认每批只有一个标准视图。相对旧v5
训练耗时下降约`78.6%`（`4.67x`加速）。机械外推40轮训练约`4.55h/27.3`个V100积分；该数值
仅用于预算，benchmark的accuracy/DSA不得作为科学证据。standard-loader门槛已通过，仍需
`aughfl`多视图代表臂benchmark后才可冻结Axis II Formal分配。

## 证据边界

- benchmark只验证执行与成本，不是论文证据；
- Formal单臂可以作为该方法的描述性结果，但Axis I胜负必须等待四臂严格合并；
- 不以官方RAHFL `40 pretrain + 40 communication`自居；官方recipe如需复现应单列；
- 主表报告学习曲线与last-10/last-5，审计是否仍处于明显快速上升期；
- 不因单臂中途结果修改轮数、batch预算、门槛或方法超参数。
