# CLE-HFL v2 Cross-Binding-Map 复现实验

Updated: 2026-09-11

状态：S0协议冻结、S1数据生成与PEW复用完成、S2本地smoke完成。2026-09-12用户明确选择
跳过OpenI benchmark并授权只运行map1 Formal；map2 Formal尚未授权。代码已push，正式输入包已
上传OpenI。当前仍无cross-map科学结果。

## 实验问题

固定训练seed、客户端非IID划分、source、severity、模型初始权重、公共数据、PEW checkpoint和
评价grid，只更换客户端特定的class-operator binding map。检验固定`seed0_split0`上的发现是否
依赖某一张偶然mapping。

每张mapping只有两臂：

```text
h9_b = AugMix/JSD/DCL + strict AsymHFL-val
h9_p = h9_b + frozen public PEW + hard BER
```

CDep禁用。PEW不重新训练；复用同一个只由公共CIFAR-100合成corruption训练的checkpoint，随后
分别对两张新mapping的私有样本生成PEW伪环境标注。真实operator/binding仍只用于数据生成、
审计与最终DSA报告。

## 为什么这比“换一个随机seed”更关键

训练seed只能检查优化随机性；新binding map直接改变shortcut的方向。如果baseline仍在新方向
形成显著DSA，且PEW+BER仍降低它，才能支持机制和缓解不是记住一张特定映射。它仍不等于真实
数据外推，也不覆盖新的partition、架构分配或corruption库。

## 冻结变量与唯一变化

```text
partition_seed: 0
evaluation_seed: 20260909
train_seed: 0
binding_map_seed: 1 or 2   # 唯一场景变化
private samples: 40000 per map package
paired grid: 1000 sources x 15 operators x 4 clients
PEW checkpoint SHA256: BC9FF7523B8474774B36E02507A544FEBD76363772E9B865191FD3119960DCBB
```

跨包审计确认gamma0图像/operator完全相同；gamma0.9只随binding变化，labels、source ids和
severity ids逐样本相同。两图共享split、公共数据、初始状态、评价grid和PEW checkpoint哈希。

## S2 smoke结果

两张mapping均完成`1 round x 1 local batch/client`的两臂CUDA smoke；配置、初始权重、严格
fit/audit隔离、PEW annotation加载、BER分组、checkpoint保存、paired local trace和20-source
DSA分析全部跑通。Smoke中模型尚未达到学习下限，所有准确率和DSA数值禁止引用为论文证据。

聚焦测试：`20 passed`。

## 阶段与门槛

```text
benchmark: 1 round x 8 local batches/client；只验证平台、显存、耗时和产物
formal:    12 rounds x 16 local batches/client/round；必须显式confirm_formal=true
```

Formal对每张mapping冻结：

```text
I0 paired integrity:
  两臂输入、初始化、训练随机轨迹匹配

L0 learning floor:
  base与plugin operator-grid pooled accuracy均 >= 20%

C1 new-map shortcut formation:
  base DSA >= 0.05
  base DSA > shuffled-binding null p95
  permutation p <= 0.01

C2 mitigation replication:
  DSA(base)-DSA(plugin) >= 0.02
  source-bootstrap CI95 lower > 0
  4/4 clients reduction > 0
```

Avg/Worst/WCCA/CFG完整报告，但不是本次cross-map复现的主gate；不得用它们事后覆盖C1/C2。
benchmark永远不给科学判定。

## 2026-09-12 Formal授权范围

```text
authorized: mode=formal, map_seed=1, confirm_formal=true
not authorized: map_seed=2 or map_seed=all
benchmark: explicitly skipped by user
```

同协议旧Stage-2两臂Formal实测训练加分析为`8116.51 s`，即约`2.25 V100 GPU-hours`。新map1
预计约`2.3--3.0 V100 GPU-hours`；这是参考估算而非新入口benchmark实测。若map1任务失败，先
诊断失败，不得自动创建map2或重跑Formal。

## 当前唯一有效入口与数据包

```text
entry: scripts/openi_cle_v2_cross_scenario_entry.py
bundle: local_runs/cle_v2_cross_scenario/cle_hfl_v2_cross_maps1_2_seed0_split0_with_pew.tar.gz
bytes: 1385820059
SHA256: BEA8E98737BF881C701DCFFF05F4E04C3A1E6095B7CF7702A5177260C2F186F5
OpenI dataset: chujiu/CLE_v2_CrossBindingMaps_Seed0_PEW_20260911
upload: 100%, 1.3 GiB, 2.5 MiB/s, 00:09:36
```

禁止使用旧的`openi_cle_cross_scenario_40round_entry.py`及其`seed1_split1/seed2_split2`数据包：
它们同时改变partition、重训PEW并包含CDep，不能回答当前纯PEW+BER的单变量cross-map问题。
