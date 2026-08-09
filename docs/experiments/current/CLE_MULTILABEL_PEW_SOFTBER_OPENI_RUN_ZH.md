# Multi-label PEW + Soft-BER 配对筛选

Updated: 2026-08-09

## 目的

检验“把复合损坏硬归为单个环境”是否限制了现有 PEW+BER。该实验不是通信实验；两臂都冻结使用 strict AsymHFL-val。

```text
hard_pew_ber       = 原版硬环境标签 PEW + 硬分组 BER
multilabel_softber = 软组合标签 PEW + 概率责任 Soft-BER
```

公共数据仍然不使用语义类别标签。对人工依次施加两个 corruption family 的图像，新 PEW 使用两个同时为 1 的 multi-hot family 目标和独立 sigmoid/BCE 监督；推断时将已知 family membership 归一化为责任概率。私有样本保存完整环境概率，Soft-BER 按该概率分摊逐样本分类损失：

```text
R(c,e) = sum_i 1[y_i=c] q(i,e) loss_i / sum_i 1[y_i=c] q(i,e)
```

原硬路径、历史 checkpoint 和历史结果全部保留。checkpoint 带有 `label_mode` 元数据，硬/软版本不能交叉加载。

## 公平性冻结项

两臂固定：CLE-HFL v2 `alpha=0.5, gamma=0.9, scenario seed=0`、training seed 0、同一 persisted fit/audit split、12 轮、相同模型/优化器/AugMix/JSD/DCL、相同 strict AsymHFL-val 与 final-test reporting-only 协议。仅 PEW 标签表达、对应 checkpoint 和 BER assignment 不同。

这是 12 轮机制筛选，不是最终多种子或 40 轮结论。通过后才考虑耐久性复验。

## OpenI 填写

```text
数据集: openi_cle_hfl_v2_alpha05_gamma09
代码分支: main（提交推送后）
启动文件: scripts/openi_cle_multilabel_softber_entry.py
运行参数: 留空
```

脚本在一个任务中按顺序运行两臂，不需要新增 `arms` 参数。

## 输出

```text
cle_multilabel_softber_seed0_12round_outputs.tar.gz
```

归档包含两臂 resolved config、训练输出、诊断汇总和冻结判定。last-five promotion gates 为：Avg 至少 `+0.5`，Worst/WCCA 不下降，CFG 不增大；四项必须全部通过。

## 当前验证

实现完成后的当前路径回归为 `71 passed`。一轮小数据 smoke 已完成 PEW 训练/校准、四客户端 Soft-BER 更新、strict audit 路由、final report 与扩展评估；smoke accuracy 仅证明路径可运行，不作为科研证据。
