# CLE-HFL 基线忠实度修复说明

Updated: 2026-08-09

## 定位

历史名称 `aughfl`、`feddf`、`kt_pfl` 保持冻结，用于解释已经完成的 12 轮筛选结果。
本次新增三个互不覆盖的实现：

```text
aughfl_fidelity
feddf_fidelity
kt_pfl_fidelity
```

这些实现修复论文核心流程与项目旧适配之间的明确偏差，但仍运行在统一 CLE-HFL 数据、四模型、
strict fit/audit/final-test 和公共 batch 预算下。因此应称为 `protocol-matched fidelity repair`，
不能称作未经修改的官方完整 recipe。

## AugHFL fidelity

参考发布源码：`local_runs/reference_sources/AugHFL/HHF/AugHFL.py` 与
`Dataset/dataaug.py`。

已修复：

- 每个公共样本为四个客户端分别生成独立 clean/aug1/aug2 triplet；
- 使用发布实现的公共输入归一化 `(x - 0.5) / 0.5`；
- 使用每客户端增强一致性的倒数形成全局教师权重；
- 公共协作阶段按发布代码重新创建 Adam optimizer；
- 保存教师权重熵、最小/最大权重和平均 view consistency。

有意未复制发布代码中把负 log-probability clamp 到正数的数值问题；当前仍向 KLDivLoss 提供
合法的 log-probability，避免梯度被 clamp 清零。

## FedDF fidelity

参考官方仓库：`local_runs/reference_sources/FedDF/codes/FedDF-code`。

已修复：

- 每轮先做私有 local update，再做 server fusion；
- 在 server fusion 开始时冻结所有 post-local client teacher 快照；
- 整个 server distillation 阶段教师不随 student 更新漂移；
- 采用官方 `avg_logits` 教师及 forward KL；
- 为每个异构架构的 server student 使用独立 Adam 和 cosine schedule；
- 保存教师熵、教师分歧和server update次数。

当前每种架构只有一个客户端，因此官方的“同架构客户端FedAvg”是恒等操作。每个客户端模型
同时充当该架构的server student，这是四种完全不同架构下的必要适配。

## KT-pFL fidelity

参考 NeurIPS 2021 论文 Algorithm 1 与 Eqs. (6)--(7)。没有把非官方第三方实现当作真值。

已修复：

- 每轮先私有 local update，再执行个性化公共蒸馏；
- 系数矩阵通过 softmax 保持逐行随机矩阵；
- 模型蒸馏结束后重新计算冻结的客户端soft predictions，再更新系数矩阵；
- 系数目标按各客户端fit样本占比加权；
- 使用显式 Frobenius uniform regularization，而不是被元素均值额外缩小的近似项；
- 保存系数损失、熵、对角/非对角均值和每轮漂移。

统一 CLE 筛选配置保留 `local_epochs=1`、每轮4个公共batch；原论文CIFAR-10使用更长的
local训练和更大的公共数据迭代预算。二者必须作为不同实验配置，不应混在同一张公平预算表。

## 入口与验证

```text
scripts/openi_cle_baseline_fidelity_entry.py
tests/test_cle_baseline_fidelity.py
tests/test_cle_remaining_baselines.py
```

本地接口/机制测试：`30 passed`。

本地一轮三臂烟雾测试通过；FedDF与KT-pFL日志确认协作阶段位于local update之后，三个方法的
专用诊断字段均为有限值。烟雾准确率不是科研结果。

正式运行说明见：

```text
docs/experiments/current/CLE_BASELINE_FIDELITY_OPENI_RUN_ZH.md
```
