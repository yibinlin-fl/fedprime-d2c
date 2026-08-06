# CLE-HFL 论文证据补全：OpenI 运行顺序

Updated: 2026-08-06

本文档对应论文实验补全，不把本地测试或 smoke accuracy 当作科研结论。所有正式比较必须保持 matched seed、模型初始化、fit/audit 划分、公共批次、评估频率与最终测试协议一致；final-test 标签仅用于报告。

## 已实现的十步工作

1. 通信层已从本地目标中解耦，并保留 strict AsymHFL-val 路由。
2. A0--A6 局部消融已实现：RAHFL、BER、CDep、完整方法、未校准 PEW、打乱 PEW、oracle family。
3. 局部消融 12 轮入口和分析器已实现。
4. Local/HFL/AsymHFL-val 的 2x3 因子实验已实现。
5. FedMD、RHFL、FedProto、AugHFL、RAHFL 外部基线已适配到同一 CLE 协议。
6. 外部基线 12 轮入口和分析器已实现。
7. scenario seed 1/2 数据及 40 轮跨场景入口已准备；training seed 固定为 0，仅改变场景。
8. alpha `{0.1,0.5,1.0}` x gamma `{0.5,0.9}` 压力网格已准备。
9. CIFAR-100-private/CIFAR-10-public 第二数据集、正确归一化和 12 轮 A/B 已准备。
10. PEW ECE/NLL/unknown-AUROC/confusion matrix、逐轮耗时、峰值显存及 OFAT 敏感性入口已实现。

以上表示“代码和数据准备完成”，不表示 OpenI 正式训练已经完成。

## 单任务队列的推荐顺序

OpenI 同时只能运行一个任务时，按下列顺序逐个提交。每个入口内部也会顺序执行所选 arms，并尽早打包已有结果。

```text
1. python scripts/openi_cle_local_ablation_entry.py --arms=all
2. python scripts/openi_cle_external_baselines_entry.py --arms=all
3. python scripts/openi_cle_communication_factorial_entry.py --arms=all
4. python scripts/openi_cle_cross_scenario_40round_entry.py --scenario_seed=1 --mode=both
5. python scripts/openi_cle_cross_scenario_40round_entry.py --scenario_seed=2 --mode=both
6. python scripts/openi_cle_stress_grid_entry.py --cells=all --mode=both
7. python scripts/openi_cle_cifar100_entry.py --mode=both
8. python scripts/openi_cle_sensitivity_entry.py --arms=all
```

若算力紧张，先运行 1、2、7；它们分别回答“贡献来自哪里”“是否优于外部方法”“是否跨数据集”。第 3 步用于判断创新是否依赖 AsymHFL；第 4--6 步补跨场景与强度证据；第 8 步补机制诊断。

## 第二数据集资产

```text
archive:
  local_runs/cle_hfl_v2_second_dataset/
  cle_hfl_v2_prepared_cifar100_alpha05_gamma09_seed0_split0.tar.gz

SHA-256:
  AF554AC3B9B46D38571445DDE84647965341444DAD175A1E4191851B8DD01EB4

private:
  CIFAR-100, 4 clients x 10,000 samples, alpha=0.5, gamma=0.9

public:
  CIFAR-10 labels unused

evaluation:
  22,000 seen + 8,000 unseen + 2,000 clean
```

协议审计通过：seen/unseen 不重叠、训练端 unseen 泄漏为 0、实际主导算子比例为 `0.90815`（理论 `0.90909`）。该数据包约 422 MiB，需要作为 OpenI 数据集上传，不要提交 Git。

## 外部基线口径

- FedMD：公共数据上的对称 logit distillation。
- RHFL：本地 symmetric cross entropy + 官方式置信度加权公共蒸馏。
- FedProto：使用四种模型已有的统一 1024 维 embedding，按类别先在客户端求特征原型、再跨客户端等权聚合，并以 MSE 联合本地分类目标；第 0 轮不使用尚不存在的全局原型。
- AugHFL：本地 AugMix/JSD + 公共 clean/augmented consistency 加权。
- RAHFL：AugMix/JSD/DCL + strict AsymHFL-val，作为匹配 control。

所有基线均走相同数据、模型集合和报告指标。正式成表前仍需将实现细节与原论文/官方代码逐项核对并在论文中披露适配差异。

RAHFL 论文 Table 5--7 的异构比较还包括 FedDF、KT-pFL 和 FCCL。当前首批先运行 FedMD、RHFL、FedProto、AugHFL 与 RAHFL；前三个缺失方法在首批筛选后按证据需要补入，不能把当前集合写成论文的完整 SOTA 集合。

## 晋级规则

- 12 轮只做筛选；明显失败配置不晋级 40 轮。
- 关键外部基线、完整方法和必要消融才做 40 轮多场景复验。
- 旧的固定场景 gate 不能自动当作第二数据集或外部基线的录用门槛；新场景先报告 effect size 和稳定性，再预注册后续 gate。
- 分析输出需同时保留 Avg、Worst、WCCA、CFG、seen/unseen、PEW 校准、耗时与显存。

## 返回文件

每个任务结束后把对应 `*_outputs.tar.gz` 放入 `outputs/`。不要只返回比较 JSON，因为后续需要独立复算逐轮指标并审计 resolved config。
