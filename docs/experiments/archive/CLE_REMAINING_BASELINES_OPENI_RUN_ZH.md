# CLE-HFL v2 剩余异构基线运行说明

更新：2026-08-09

## 目的

补齐 RAHFL 主实验表中首轮外部基线筛选尚未覆盖的三个方法：
`FedDF`、`KT-pFL`、`FCCL`。它们通过现有 `CommunicationContext`
统一接口运行，不改变 CLE 数据、四个异构模型、本地训练预算或评测角色。

这是 12 轮筛选，不是最终长轮数结论。只有有竞争力的基线才晋级 40 轮。

## OpenI 填写

```text
数据集: openi_cle_hfl_v2_alpha05_gamma09
启动文件: scripts/openi_cle_remaining_baselines_entry.py
运行参数: 无（默认依次运行全部五臂）
```

默认顺序：

```text
feddf, kt_pfl, fccl, rahfl, pew_ber
```

如果只想跑部分臂，添加参数名 `arms`，参数值用逗号分隔，例如：

```text
feddf,kt_pfl,fccl
```

预期压缩包：

```text
outputs/cle_remaining_baselines_seed0_12round_outputs.tar.gz
```

## 公平性合同

所有臂固定 scenario seed 0、training seed 0、`split0`、12 轮、四个公共
batch/round、相同四个模型和优化器。私有 fit 只产生本地梯度；三个新增通信方法
只使用公共无标签图像；final-test 只报告。`rahfl` 和最终 `pew_ber` 同任务重跑，
用于排除跨任务环境差异。

三个适配器复现论文/公开代码的核心机制，而不是照搬其不同数据集、模型、轮数和
优化器的整套脚本：

```text
FedDF  : 全体客户端 logits 求均值，分别蒸馏回每个异构学生模型
KT-pFL : 每个接收客户端维护一行可学习知识系数，形成个性化软教师
FCCL   : 按公开代码计算公共 logits 的经验交叉相关损失
```

论文中应称为“under our matched CLE-HFL protocol 的核心机制复现”，不要称为
原仓库配置的逐行复跑。

## 结果判断

分析器默认报告各臂相对同任务 `rahfl` 的 final 和 last-five 差值。12 轮仅用于：

```text
明显落后 -> 不晋级
接近或超过 RAHFL -> 进入 40 轮匹配复验
早期曲线仍上升且差距很小 -> 不直接误杀，先检查曲线再决定
```
