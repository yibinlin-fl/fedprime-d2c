# CLE-v2 Mechanism Stage-1 Formal 结果

日期：2026-09-10  
状态：`GO_CLE_V2_MECHANISM_STAGE1`

## 协议

固定CLE-v2 `seed0_split0`场景、training seed 0、12 rounds、每客户端每轮最多16 local batches、
batch size 64。四臂为HFL/Local × gamma0/gamma0.9，均使用AugMix/JSD/DCL baseline；本阶段不训练、
加载或审计PEW，不启用BER/CDep。最终checkpoint在1,000 source × 15 operator × 4 client完整
operator grid上评价。

输入包含40,000个可用私有样本，但受每轮batch上限与重复洗牌影响，不能解释为每个模型完整
遍历40,000个唯一样本。

## 主结果

| Arm | Scope | Gamma | Operator-grid Acc (%) | Pooled DSA |
|---|---|---:|---:|---:|
| h0_b | HFL | 0.0 | 24.9300 | -0.000245 |
| h9_b | HFL | 0.9 | 21.4367 | 0.119644 |
| l0_b | Local | 0.0 | 25.5683 | -0.001914 |
| l9_b | Local | 0.9 | 22.4233 | 0.106046 |

| Estimand | Estimate | 95% source-bootstrap CI |
|---|---:|---:|
| HFL CLE effect | 0.119889 | [0.118050, 0.121562] |
| Local CLE effect | 0.107960 | [0.106145, 0.109678] |
| Communication add-on | 0.011929 | [0.011115, 0.012712] |

`Local/HFL share=90.05%`。H9真实binding的DSA为0.119644，shuffled-binding null p95为0.029559，
置换检验`p=0.000999`。L0/M1/M2/M3全部通过，不触发32-batch补跑。

## 解释边界

DSA的0.119889是0--1概率尺度上的约12个百分点定向概率质量变化，不是0.1个百分点准确率。
它约为shuffled-null p95的4.05倍，且四个客户端的HFL CLE effect均为正。operator-grid accuracy
同时从HFL gamma0的24.93%降至gamma0.9的21.44%，说明该机制伴随可见任务损害。

90.05%是两个同尺度、匹配contrast的描述性比值：`Local CLE effect / HFL CLE effect`。等价分解为
`0.119889 = 0.107960 + 0.011929`。它说明HFL中观察到的大部分CLE效应在无通信的Local训练中已
出现，但不是把每个样本的因果来源精确分成90.05%与9.95%。通信客户端差值有正有负，因此只能
声称pooled平均上存在较小附加放大。

本结果是12-round正式机制证据，因为预注册学习门已通过；它不宣称达到40-round饱和性能。
source-bootstrap不覆盖训练随机性，也不支持跨CLE场景外推。本阶段未使用PEW/BER，不能证明插件
有效。

## 完整性与成本

- 结果包：`cle_v2_mechanism_stage1_seed0_formal_outputs.tar.gz`
- bytes：8,979,168
- SHA256：`A23312C97A1B9FE2A4DAA8341BB83CD326553A550726CE626097FF68AF4FF4B2`
- 独立概率缓存形状：`[4,4,1000,15,10]`
- 概率和最大误差：`2.09e-7`
- 独立复算与正式汇总最大差：0
- 总耗时：15,726.38秒（约4小时22分）
- 峰值显存：约5.4GB

## 下一优先级

先在相同固定场景和可负担预算下做`h9 Base vs h9 Base+PEW+BER`纯插件A/B。已有三seed正结果的
candidate实际包含`PEW/BER+CDep`，可以作为组合插件稳定性证据，但不能替代PEW+BER-only归因。
确认纯插件有效后，再做PEW family/Oracle family/Oracle operator/random grouping粒度消融，最后
选择一个代表性异构联邦底座验证插件性。
