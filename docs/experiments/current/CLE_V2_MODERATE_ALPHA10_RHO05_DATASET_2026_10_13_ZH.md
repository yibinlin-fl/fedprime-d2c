# CLE-v2 中等难度补充场景数据集

Updated: 2026-10-13

## 定位

本数据集是现有 strong CLE/map2 主场景的补充，不替换原来的
`alpha=0.5, corruption_rate=1.0, gamma=0.9` stress setting。它用于回答：当标签偏斜和私有
损坏率更接近 RAHFL 的常用设置时，CLE 是否仍随绑定强度形成剂量响应，以及 PEW+BER 是否仍能
抑制 strong CLE。

## 冻结生成参数

```yaml
private_dataset: CIFAR-10
public_dataset: CIFAR-100
num_clients: 4
samples_per_client: 10000
partition: Dirichlet
dirichlet_alpha: 1.0
corruption_rate: 0.5
gamma: [0.0, 0.5, 0.9]
binding_map_seed: 2
partition_seed: 0
fit_audit_split_seed: 0
seen_operators: 11
unseen_operators: 4
```

每个客户端/样本位置先用冻结随机数产生一个 corruption mask；该 mask 在三种 gamma 之间完全
相同。未损坏样本保持 clean，`train_is_corrupted=false`、`train_corruption_ids=255`、severity=0。
被损坏样本使用同一个 source、label 和 severity，只改变 operator 采样分布：

```text
P(E=e | Y=c, C=1) = (1-gamma)/11 + gamma * 1[e=b_k(c)]
```

因此 gamma 比较是配对的 binding-strength 对比。由于该补充场景同时把 alpha 从0.5改为1.0、
corruption rate从1.0改为0.5，它不是单独识别 alpha 或 corruption rate 因果效应的消融。

## 生成与审计结果

```text
protocol = cle_hfl_v2_moderate_alpha10_rho05_gamma_grid_seed0_split0_v1
actual corruption rate = 0.499325
bound-operator rate among corrupted samples:
  gamma=0.0: 0.092275
  gamma=0.5: 0.543701
  gamma=0.9: 0.907401
integrity audit = PASS
private samples = 40000
paired DSA grid = 1000 sources x 15 operators, severity 3
```

PEW 不重新训练。数据包复用正式 map2 冻结 checkpoint：

```text
PEW SHA256 = BC9FF7523B8474774B36E02507A544FEBD76363772E9B865191FD3119960DCBB
gamma00 diagnostic coarse-group accuracy = 62.0825%
gamma05 diagnostic coarse-group accuracy = 61.8875%
gamma09 diagnostic coarse-group accuracy = 61.4025%
```

这些 PEW 数字只描述 proxy quality，不证明 PEW+BER 有效。

## 在论文中的身份

该制品可以作为论文“受控CLE benchmark protocol / evaluation testbed”的一部分，具体包括：

- 将客户端标签偏斜`alpha`、损坏暴露率`rho`和class--operator binding强度`gamma`显式分开；
- 在gamma条件间冻结source、label、mask、severity、split与初始化，支持配对剂量响应；
- 同时提供训练数据、sealed operator metadata、paired DSA grid、PEW proxy与完整manifest；
- 允许把普通corruption difficulty、label skew和directional shortcut分开报告。

它不应单独宣称为“新真实数据集”：private task仍是CIFAR-10，corruption与binding均为受控合成。
论文贡献措辞应是“a controlled CLE-HFL construction and evaluation protocol”，并与paired DSA、
local-first归因和PEW+BER共同构成闭环。若公开发布，应同时提供生成器、固定seeds/maps、manifest、
hash、信息边界和许可证说明，而不是只发布一个tar包。

## 本地制品

```text
directory:
C:\Users\asus\Desktop\FedPRIME-D2C\local_runs\cle_v2_moderate_alpha10_rho05\cle_hfl_v2_moderate_alpha10_rho05_map2_seed0_split0

archive:
C:\Users\asus\Desktop\FedPRIME-D2C\local_runs\cle_v2_moderate_alpha10_rho05\cle_hfl_v2_moderate_alpha10_rho05_map2_seed0_split0_with_pew.tar.gz

archive bytes = 893421220
archive SHA256 = DEA800EC1465C45037E74E4B948DFB8D778F1354E8522B551438C6A515C9E588
```

2026-10-13二次包审计：压缩包217个entries（其中178个文件）均可流式读取，解压总字节`986797088`；三组PEW
annotation均为每客户端10000项、取值在`0..5`，压缩包大小与SHA256保持一致。

当前只完成数据生成、PEW annotation 和完整性审计。尚未冻结训练矩阵、实现对应 OpenI runner，
也未授权 benchmark/Formal；不得把数据统计或 smoke 当作论文方法证据。
