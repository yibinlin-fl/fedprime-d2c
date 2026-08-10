# 类条件反事实退化遗憾 Audit 0

日期：2026-08-10

## 1. 目标

本审计只回答一个问题：不使用 PEW 环境分类、损坏 family、operator ID 或
severity 标签时，同一样本在独立 AugMix 干预下的正确类 margin 降幅，能否稳定预测
下一次独立干预中的脆弱性，并优于原始 CE 与 JSD disagreement。

暂用名称：

```text
Class-Conditional Counterfactual Regret (C3R)
类条件反事实退化遗憾
```

Audit 0 不实现训练损失、不连接正式 runner、不启动 12/40 轮实验。

## 2. 与当前方法的关系

若后续晋级，C3R 的论文主候选将替换 hard PEW + hard BER：

```text
control   = AugMix/JSD/DCL + hard PEW + hard BER
candidate = AugMix/JSD/DCL + C3R
```

两者代码可以共存以便形成 A/B，但主方法不同时启用两者。AsymHFL-val 保持不变，
不作为本阶段创新。

## 3. 冻结信号定义

对样本 `i` 的原始视图和两个独立强视图，计算正确类 margin：

```text
m0 = z0[y] - max(z0[not y])
m1 = z1[y] - max(z1[not y])
m2 = z2[y] - max(z2[not y])
regret = max(0, m0 - min(m1, m2))
```

训练损失若获准进入下一阶段，原始视图 margin 必须 stop-gradient，防止模型通过主动
降低 `m0` 缩小 regret。Audit 0 只读 logits，不产生梯度。

三个待比较信号冻结为：

```text
C3R: regret
CE:  原始视图逐样本交叉熵
JSD: 原始/强1/强2预测分布的逐样本 Jensen-Shannon divergence
```

所有信号先在各语义类别内部转为百分位秩，再跨类别汇总，避免 label-skew 下多数类
仅凭样本量主导结果。

## 4. 数据角色和轻量训练

正式运行固定使用 CLE-HFL v2 `seed0_split0` 的持久化 strict split：

```text
clients: 1 and 3
models: ResNet12 and Mobilenetv2
local base: AugMix/JSD/DCL, no PEW, no BER
training epochs: 3
augmentation seeds: 1701, 1702, 1703
```

每个客户端只在原 `fit_indices` 内按类别确定性划分互斥的 probe 子集。仅当类别至少有
32 个 fit 样本时才进入 probe；probe 数量为该类 fit 的 15%，下限 16、上限 64，
其余 fit 样本用于轻量训练。原 private `audit` 保持为 AsymHFL routing 专用，final
test 完全不读取。

训练数据中的 operator ID 也不进入模型或信号，只允许在结果阶段汇总
`class x operator` cell。

## 5. 跨随机视图评价

使用有向配对：

```text
1701 -> 1702
1702 -> 1703
1703 -> 1701
```

源 seed 计算 C3R/CE/JSD，目标 seed 计算：

```text
target regret
robust error: 任一强视图分类错误
flip error: 原始视图正确、但任一强视图错误
```

因此 C3R 不能用生成自身的同一组强视图作为预测标签。

## 6. 冻结 GO/NO-GO 门槛

两个客户端合计 6 个有向 seed pair。只有全部门槛通过，才允许进入匹配的一步更新
审计：

```text
G0 validity:
   每个客户端三 seed 平均原始视图 accuracy >= 20%
   每个有向 pair 的 target flip prevalence 在 [2%, 80%]

G1 activity:
   每个客户端 regret>0 比例的三 seed 中位数 >= 20%
   每个客户端 regret p90 的三 seed 中位数 >= 0.25 margin

G2 persistence:
   6 个 pair 的类别内 regret Spearman 中位数 >= 0.25

G3 predictive AUROC:
   6 个 pair 的 C3R -> target flip AUROC 中位数 >= 0.60
   且比每个 pair 中较强的 CE/JSD 基线的中位优势 >= +0.02

G4 tail enrichment:
   C3R 类别内 top-25% 对 target flip 的富集倍数中位数 >= 1.30
   且不低于较强的 CE/JSD 基线

G5 cell relevance:
   每个 cell 至少 8 个 probe 样本；两客户端合计至少 20 个有效 cell
   C3R cell mean 与 target robust-error rate 的 Spearman 中位数 >= 0.30

G6 cross-client consistency:
   C3R AUROC 在至少 4/6 pair 上超过 CE 和 JSD
   且两个客户端各至少有一个 pair 获胜
```

如果因模型没有学到最低 accuracy 或 flip prevalence 不可识别而失败，结论为
`INVALID_PROBE`，允许只提高预注册的训练充分性后重跑；不得改变信号公式或其余门槛。
其他门槛失败均为 `NO-GO`，不得通过调整 margin 定义、top fraction 或门槛复活。

## 7. 晋级后的唯一下一步

若 Audit 0 为 `GO`：

```text
同一初始化与 fit/probe 划分
control   = AugMix/JSD/DCL 的一步更新
candidate = control + stop-gradient C3R loss 的一步更新
```

只有一步更新对最差类别/cell 有正归因且平均性能不回退，才允许把 C3R 接入可选本地
runner。Smoke accuracy 仍只证明执行。
