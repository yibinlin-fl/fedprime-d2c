# CLE-HFL 论文闭环状态

Updated: 2026-09-11

## 论文主线

```text
CLE-HFL受控场景
  -> paired counterfactual + operator-level DSA证明定向shortcut
  -> HFL-vs-Local证明机制主要local-first
  -> coarse PEW+BER作为taxonomy-assisted本地干预
  -> Oracle/random粒度消融界定环境信息与粒度边界
```

论文定位不是“PEW网络结构创新”，而是CLE-HFL问题、配对诊断、local-first机制归因、一个简单
有效干预及其边界的闭环研究。

## 一、Oracle粒度消融表

| grouping | pooled DSA | operator-grid Avg | 允许结论 |
|---|---:|---:|---|
| Oracle family | 0.017232 | 21.9333 | 粗真实环境分组的不可部署上界 |
| Oracle operator | 0.015783 | 21.5200 | 细粒度只额外降低0.001449 DSA |
| Random operator | 0.067895 | 18.7817 | 破坏样本—环境对应后显著变差 |

`RO-OO=0.052112`且CI95下界大于0，说明真实环境对应关系重要；`OF-OO=0.001449<0.02`，
说明operator细化没有实质收益。Formal verdict为`NO_GO_OPERATOR_GRANULARITY_GAP`，禁止继续
层次化/operator PEW。

## 二、DSA三个理论与缓存性质

1. operator条件交换时DSA严格为0；exchangeable projection数值为`-2.50e-20`，gamma0经验
   最大绝对DSA为`0.001914`。
2. DSA对预测概率混合是线性的；HFL/Local 11点混合曲线严格单调，最大仿射误差
   `2.78e-17`。
3. 同一operator附近的增强预测JSD可以为0，同时跨operator的DSA仍为`0.119644`；因此
   AugMix/JSD一致性不推出CLE directional shortcut消失。

这些性质解释指标零点、shortcut混合强度和JSD不充分性，不声称跨场景因果充分性。

## 三、最终插件

最终方法保持`frozen coarse PEW + hard BER`，安装在客户端本地训练阶段：PEW提供粗family
伪环境，BER在类别内部平衡环境风险；AugMix/JSD/DCL继续承担局部增强一致性和鲁棒表征，通信
策略本身不由插件修改。固定AsymHFL场景seed-0纯插件Formal将DSA从`0.119644`降到`0.041252`
（降低65.52%，CI95 `[0.077191,0.079601]`），Avg增加1.7323pp，但原完整效用门因Worst失败。

当前新增的FedDF-fidelity两臂使用`standard CE vs PEW-grouped BER-weighted CE`，两臂均不使用
AugMix/JSD/DCL，并保持FedDF通信完全相同。它用于验证PEW+BER能否作为真正的本地目标插件迁移
到第二底座；实现仍是protocol-matched fidelity adapter，不是未经修改的官方完整recipe。

## 四、必须明确的限制

- CLE抑制在固定强CLE场景中成立；跨CLE mapping、跨数据集仍未建立。
- 架构与客户端数据绑定，当前不能声称所有架构任务效用一致改善。
- PEW依赖人工corruption family taxonomy；不是taxonomy-free方法。
- Oracle只用于机制上界，不可部署。
- coarse family足够不等于learned PEW已达到Oracle质量。
- smoke和benchmark只验证执行/成本，不能进入科学结果表。

## 当前唯一待决实验

```text
FedDF-fidelity + standard CE
vs
FedDF-fidelity + coarse PEW/BER-weighted CE
```

先完成OpenI benchmark；只有执行、轨迹、数值稳定性和成本均通过，且用户另行明确批准，才运行
12轮Formal。此后停止方法扩展，进入主表、消融表、机制图和逐句证据审计。
