# CLE-v2 Moderate Axis II AsymHFL Benchmark

日期：2026-10-14

## 判定

`axis2_asym` benchmark通过，授权`asymhfl_erm`与`asymhfl_pew_ber`分别进入seed-0、40轮Formal。
Benchmark不是科学证据，其DSA和accuracy不得进入论文。

## 审计

```text
mode                 benchmark
shard                axis2_asym
arms                 asymhfl_erm / asymhfl_pew_ber
rounds               2
max_local_batches    8
batch_size           64
public_batches       4
train_seed           0
map_seed             2
alpha/rho/gamma      1.0 / 0.5 / 0.9
completed            2/2
scientific_evidence  false
```

两臂各有8条`round x client`训练轨迹，SHA256完全一致：

```text
726C3E7D1BCE4965F4345B676B5FDCE556D971F34DED0A888822F71DC238B749
```

训练总耗时`227.981s`，分析`156.547s`。由每臂round耗时按更新量机械外推：

| Formal单臂 | 估计V100小时 | 估计积分（6/小时） |
|---|---:|---:|
| AsymHFL--ERM | 4.05 | 24.3 |
| AsymHFL+PEW+BER | 3.95 | 23.7 |

考虑安装、解压、正式分析和波动，每个账号建议预留30--35积分。正式任务应拆成两个单臂并行，
不得跨账号拼接轮次。

## 推荐并行分配

若有三个可用账号：

```text
账号A  feddf_fidelity Formal       -> Axis II第4行
账号B  asymhfl_erm Formal          -> Axis II第7行
账号C  asymhfl_pew_ber Formal      -> Axis II第8行
```

三项完成后，八行领域表将完成第1--4、7、8行；第5/6行AugHFL/RAHFL因每臂约17小时/102积分
继续暂缓。

## 原始包

```text
cle_v2_moderate_axis2_asym_benchmark_outputs.tar.gz
bytes   4379936
SHA256  A01B841E21228DBDD754C79FC4824F8D9BC4B4E61B2512B7799738957507FF65
```
