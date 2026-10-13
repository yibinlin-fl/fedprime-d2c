# CLE-v2 Moderate Axis II Plain Formal（seed 0）

日期：2026-10-14

## 1. 结论

`Local/ERM`、`FedMD`和`FedProto`均在同一moderate协议下完成40轮Formal，并通过合同、完成标记、
训练轨迹与统一评价审计。三种方法的DSA均显著高于shuffled-binding零基准，说明仅使用普通本地
ERM或两种经典异构联邦通信并不会自动消除class--corruption directional shortcut。

FedProto相对Local的DSA降低`0.007587`（约`4.51%`），同源source bootstrap 95% CI为
`[-0.009359,-0.005880]`；但其operator-grid、clean、balanced-corrupted与rho=.5 mixed准确率
均略低于Local。因此该结果只能说明FedProto在本固定场景下有小幅shortcut缓解，不能称其解决
CLE，也不能称其在效用上优于Local。FedMD与Local的DSA差异为`-0.000539`，95% CI跨零。

## 2. 冻结协议与执行审计

三臂共同使用：

```text
private task       CIFAR-10
binding map        held-out map2
Dirichlet alpha    1.0
corruption rate    0.5
CLE gamma          0.9
training seed      0
pretrain epochs    0
communication      40 rounds
local budget       64 batches/client/round
batch size         64
public batches     4/round
loader             standard single-view
skip_nonfinite     false
```

每个包均有完整`CONTRACT`、`COMPLETION`、40行round metrics、160行local batch trace、统一预测缓存
及Formal分析。三臂local batch trace的SHA256完全一致：

```text
5AEB50E9D1BBFC3F7F6C3081218F16AA10140148579EF4D0FC420CAE19A53DB7
```

这证明三臂的40轮、4客户端训练batch轨迹匹配；不同账号的运行时间不用于方法比较。

## 3. 正式结果

| 方法 | DSA↓ | Grid Acc↑ | Clean↑ | Corr-Balanced↑ | Seen↑ | Unseen↑ | rho=.5 Mixed↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Local/ERM | 0.168374 | **39.830** | **51.425** | **39.508** | **39.407** | **39.788** | **45.467** |
| FedMD | 0.167835 | 38.732 | 50.400 | 38.378 | 38.057 | 39.263 | 44.389 |
| FedProto | **0.160788** | 39.250 | 50.800 | 38.937 | 38.943 | 38.919 | 44.868 |

三臂shuffled-binding检验：

| 方法 | observed DSA | shuffled p95 | permutation p |
|---|---:|---:|---:|
| Local/ERM | 0.168374 | 0.026743 | 0.000999 |
| FedMD | 0.167835 | 0.027091 | 0.000999 |
| FedProto | 0.160788 | 0.026462 | 0.000999 |

## 4. 配对差异

基于同一1000个source进行10000次source bootstrap：

| DSA contrast | 点估计 | 95% CI | 解释 |
|---|---:|---:|---|
| FedMD - Local | -0.000539 | [-0.002522, 0.001420] | 无稳定差异 |
| FedProto - Local | -0.007587 | [-0.009359, -0.005880] | 小幅但稳定降低 |
| FedProto - FedMD | -0.007047 | [-0.008879, -0.005250] | 小幅但稳定降低 |

这些区间只覆盖固定checkpoint后的source抽样变化，不覆盖training seed、partition、dataset或
binding-map总体不确定性。

## 5. 运行成本

| 方法 | training seconds | V100 hours | 约合6积分/小时 |
|---|---:|---:|---:|
| Local/ERM | 12043.509 | 3.345 | 20.1 |
| FedMD | 12111.212 | 3.364 | 20.2 |
| FedProto | 10442.972 | 2.901 | 17.4 |

## 6. 论文允许与禁止的主张

允许：

- moderate CLE在三种普通训练/通信机制下均产生显著directional shortcut；
- FedMD没有可辨别的DSA改善；FedProto只有小幅改善且未提高本组统一utility；
- 经典异构通信本身不足以替代针对CLE结构的干预。

禁止：

- 不能在第7/8行Formal到齐前宣称PEW+BER在moderate Axis II中胜出；
- 不能把seed-0结果写成training-seed稳定性；
- 不能据此宣称FedProto普遍优于FedMD或Local；
- 不能将moderate CIFAR-10结果写成第二数据集或真实部署证据。

## 7. 原始归档

```text
Local archive   2407720 bytes
SHA256          9CCDBBA73F2A1F755387847A3A2534645C6E42E1D5AEFAEAA067CB611E8983A0

FedMD archive   2384711 bytes
SHA256          9251FAA0637167F6DADF6085158C5651AC98FBF63090A34012085300B08FF06E

FedProto archive 2406097 bytes
SHA256           12B8B2D2D145A62F114F9FE92E8C1A6B57B35E6A65054C2100BFF336754A92BB
```

原始包位于`outputs/openi_downloads/moderate_axis2_*_formal_seed0_20261013/`，不纳入Git。
