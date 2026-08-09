# FedPRIME-D2C 当前代码架构

Updated: 2026-08-09

本文件只描述清理后的活动代码。清理前的完整历史架构保存在：

```text
docs/archive/legacy/ARCHITECTURE_PRE_CLEANUP_2026_08_09.md
```

## 当前研究主线

```text
CLE-HFL v2 prepared data
  -> strict fit/audit split
  -> heterogeneous client models
  -> PEW environment annotations (hard ID, or optional compositional probabilities)
  -> BER/Soft-BER + AugMix/JSD + DCL local training
  -> selected communication strategy
  -> final-test reporting and CLE/operator metrics
```

正式协议边界：

- `fit` 仅用于梯度更新。
- 客户端私有 `audit` 仅用于 AsymHFL-val 路由。
- final-test 标签仅用于报告，不参与调参、选教师或早停。
- PEW 使用无标签公共图像上的人工已知 corruption 训练；私有 operator 元数据仅用于诊断评估。
- 默认路径继续使用硬 PEW+BER；实验性 `multi_label` 路径把复合 corruption 表示为 family 概率，并由 Soft-BER 进行分数责任聚合。
- PEW checkpoint 保存标签模式，禁止硬/软 checkpoint 误复用。

## 核心运行入口

```text
scripts/run_experiment.py
```

入口按 `method_name` 惰性加载实验类，不再静态导入已经冻结的方法。当前主要 runner：

```text
fedprime/methods/rahfl_asymhfl.py  统一异构通信、评估和轮次控制
fedprime/methods/fedease.py        PEW 准备、校准和私有环境推断
```

## 当前本地训练

```text
fedprime/methods/local_fedease.py
fedprime/methods/balanced_environment_risk.py
fedprime/methods/environment_witness.py
```

`local_fedease.py` 只保留当前选定目标：

```text
classification = hard BER over class x predicted-environment groups
              or Soft-BER over class x probabilistic environment responsibilities
local objective = classification + lambda_jsd * JSD + DCL
```

CDep、EBST 和 SCP 已从活动实现中移除。历史公式、配置和结果位置见：

```text
docs/archive/methods/NEGATIVE_CODE_REMOVAL_INDEX_ZH.md
```

通用/历史但仍可运行的本地模块独立放置：

```text
fedprime/methods/local_rahfl.py  AugMix/JSD/DCL、NIR-DCL、SARA
fedprime/methods/nir_dcl.py
fedprime/methods/sara.py
fedprime/methods/local_prime.py  PRIME 兼容本地训练
```

## 数据与严格划分

```text
fedprime/data/strict_fit_audit.py  中性的 strict fit/audit 划分与 loader
fedprime/data/fedease.py           PEW/BER 所需数据包装和统计
fedprime/data/loaders.py           通用、CLE 和公共数据 loader
fedprime/data/corruptions.py       corruption 操作
fedprime/data/partition.py         通用分区
```

`strict_fit_audit.py` 原先位于带有 FedFalsify 名称的模块中。当前文件只承载通用协议，不包含已否定的 FedFalsify router。

## 通信与外部基线

```text
fedprime/communication/public_logits.py  通信上下文和当前核心策略接口
fedprime/communication/baselines.py      匹配预算的外部基线 runtime adapters
```

`baselines.py` 集中保存共享同一 `CommunicationContext` 的轻量适配器，便于强制相同轮次、公共 batch、数据角色和指标记录。它不是官方源码的混合拷贝。

官方/发布源码只作为只读参考，彼此隔离且默认不跟踪：

```text
local_runs/reference_sources/AugHFL/
local_runs/reference_sources/FCCL/
local_runs/reference_sources/FedDF/
local_runs/reference_sources/FedProto/
local_runs/reference_sources/RHFL/
```

RAHFL 和 PRIME 的 vendor/runtime 依赖分别位于：

```text
RAHFL-master/
PRIME-augmentations-main/
```

基线忠实度边界和未复现的官方 recipe 细节记录在：

```text
docs/research/baselines/BASELINE_FIDELITY_REPAIR_ZH.md
deliverables/baseline_fairness_audit_20260809/BASELINE_FAIRNESS_AUDIT_ZH.md
```

## 指标与产物

```text
fedprime/engine/cle_metrics.py       CLE-HFL 聚合指标
fedprime/engine/operator_metrics.py  seen/unseen operator 指标
outputs/                             原始输出、压缩包、checkpoint（不跟踪）
deliverables/                        解析报告、图表和表格
local_runs/                          参考源码、缓存和临时运行（不跟踪）
```

## 维护边界

- 新方法必须通过统一通信上下文或明确的新接口接入，不得把官方仓库训练循环直接揉进 runner。
- 外部基线适配器与官方参考源码分离；适配器负责公平协议，参考源码负责忠实度核对。
- 被冻结的方法不重新加入活动 registry；需要追溯时读取归档或 Git 历史。
- 修改 strict split、PEW+BER 本地目标或 AsymHFL-val 路由时，必须运行对应聚焦回归和一轮 smoke。
