# CLE-HFL论文主结果表

日期：2026-09-12

## 建议正文主表

**Table X. PEW+BER对CLE directional shortcut的缓解、跨binding-map复现及跨通信底座边界。**
所有实验均为CLE-HFL v2、training seed 0、12轮、每客户端每轮16个local batches；结果来自
Formal而非smoke/benchmark。`Delta`均为Plugin减Base；DSA一列用向下箭头表示越低越好。

| Setting | Communication/private objective | Base DSA ↓ | +PEW/BER DSA ↓ | Abs. reduction ↑ | Relative ↓ | ΔAvg (pp) | ΔWorst (pp) | ΔWCCA (pp) | ΔCFG (pp) | Interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Original CLE-v2 | strict AsymHFL-val + AugMix/JSD/DCL | 0.119644 | 0.041252 | 0.078392 | 65.52% | +1.7323 | -0.3760 | +0.3500 | -9.2600 | mitigation GO；uniform utility未建立 |
| New binding map1 | strict AsymHFL-val + AugMix/JSD/DCL | 0.113761 | 0.049531 | 0.064230 | 56.46% | +0.7040 | +1.0760 | +0.3500 | -9.8550 | cross-map I0/L0/C1/C2全部PASS |
| Original CLE-v2 | native-CE FedDF-fidelity | 0.136989 | 0.029012 | 0.107977 | 78.82% | -0.6660 | +0.2147 | -0.1000 | -13.2350 | cross-base mitigation GO；utility trade-off |

## 表下注释

1. 原map与新map1共享partition、evaluation/training seed、模型初始化、public data、evaluation
   source grid及同一冻结PEW；新实验只改变客户端特定class-operator binding direction。
2. 新map1的DSA reduction source-bootstrap CI95为`[0.063105,0.065328]`，4/4客户端同向；
   Base DSA高于shuffled-binding null p95（`0.113761>0.020252`, `p=0.000999`）。
3. FedDF仅为protocol-matched `feddf_fidelity` adapter，不声称完整复现官方recipe；其两臂未达到
   预注册20% operator-grid学习下限，因此只作为支持性的跨底座机制证据。
4. 原Stage-2与FedDF的冻结overall NO-GO必须保留。表中“mitigation GO”只针对DSA抑制，不等于
   普适无损插件。
5. 本表不把source-bootstrap解释为跨训练seed或跨scenario置信区间。

## 在论文中的位置

建议放在Experiments开头、CLE形成与local-first机制表之后。逻辑顺序为：先证明shortcut存在并
定位为local-first，再用本表回答干预是否有效、是否依赖唯一binding map、是否依赖唯一通信底座。
