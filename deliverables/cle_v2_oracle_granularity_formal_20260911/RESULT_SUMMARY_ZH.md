# CLE-v2 Oracle 粒度 Kill Test Formal 结果

日期：2026-09-11

## 正式结论

```text
verdict: NO_GO_OPERATOR_GRANULARITY_GAP
hierarchical_pew_training_authorized: false
```

固定 `seed0_split0`、gamma0.9、training seed 0、12 rounds、每客户端每轮16个local batches，
比较三臂：

```text
h9_of = Oracle family BER
h9_oo = Oracle operator BER
h9_ro = 类别内打乱operator的随机对照BER
```

Oracle使用私有真实corruption metadata，只是不可部署的机制上界，不是候选训练方法。

## 完整性

输入审计PASS：40,000个私有训练样本、1,000个paired source、15个operator；PEW checkpoint
虽包含在复用包中，但三臂均未使用learned PEW或CDep。三臂配置哈希与冻结contract一致，48条
local batch/AugMix trace完全匹配。缓存预测全部有限，概率和最大浮点误差为`2.38e-7`。

原始包：

```text
cle_v2_oracle_granularity_seed0_formal_outputs.tar.gz
bytes: 6633253
sha256: B133F578255308764A1AB6ECD41ACB66421FA75A73A75AF7864BA16681D7054C
```

## 核心结果

| arm | pooled DSA | operator-grid Avg | last-5 Avg | last-5 Worst | zero-recall总数 |
|---|---:|---:|---:|---:|---:|
| Oracle family | 0.017232 | 21.9333 | 21.0960 | 17.8400 | 13 |
| Oracle operator | 0.015783 | 21.5200 | 20.0273 | 15.8827 | 13 |
| Random operator | 0.067895 | 18.7817 | 18.3933 | 14.7867 | 10 |

```text
DSA(RO)-DSA(OO) = 0.052112, CI95 [0.051231, 0.052983]
DSA(OF)-DSA(OO) = 0.001449, CI95 [0.001086, 0.001790]
```

按客户端的`DSA(OF)-DSA(OO)`约为：

```text
client0 +0.020798
client1 +0.007219
client2 -0.022287
client3 +0.000065
```

operator细分收益并不跨客户端一致；client2上反而更差。

## 冻结门槛

| gate | 结果 | 原因 |
|---|---|---|
| I0 | PASS | 48条训练轨迹匹配 |
| L0 | FAIL | random arm grid Avg 18.7817，低于20% |
| G1 | PASS | 真实operator优于类别内随机分组，差值0.052112且CI下界大于0 |
| G2 | FAIL | family到operator仅改善0.001449，远低于0.02 |
| G3 | FAIL | grid与零召回子条件通过；last-5 Avg下降1.0687pp，超过1pp容忍线0.0687pp |

G3虽然只窄幅失败，但不是主结论来源。即使G3通过，G2仍差约13.8倍，Formal verdict不会改变。

## 允许与禁止的解释

允许：真实环境对应关系相对随机分组明显有用；粗family分组已经捕获了几乎全部可由真实
operator分组获得的DSA收益；因此粗PEW可以用于operator-level DSA问题，细化到operator不是当前
主要瓶颈。

禁止：不得据此实现层次化/operator PEW，不得调BER权重、门槛或补seed复活粒度路线；不得把
Oracle称为可部署方法；不得声称结果解决了ShuffleNet/client2上的任务效用问题。

与固定场景历史结果作上下文比较时，baseline DSA为0.119644、learned coarse PEW+BER为0.041252，
而Oracle family为0.017232。该跨实验差值提示粗粒度伪环境质量仍存在上界空间，但它不是本次
预注册的因果对比，也不授权复活已冻结的WEC-BER或直接训练新PEW。

## 成本

```text
training: 12365.30 s
analysis: 61.15 s
total: 12426.45 s = 3.4518 V100 GPU-hours
```

