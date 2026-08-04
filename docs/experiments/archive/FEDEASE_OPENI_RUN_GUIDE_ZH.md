# FedEASE v2.1 OpenI 运行指南

更新时间：2026-07-19

## 1. 这次上传哪个数据集

第一次 `oracle_probe` 可以直接复用 OpenI 现有数据集 `openi_cle_rahfl_diagnostic`。
入口按文件名递归查找，不要求 OpenI 页面中的数据集显示名称固定；旧数据中的：

```text
cle_hfl_prepared_alpha05_gamma09_seed0.tar.gz
```

已经足够完成 Oracle control vs Oracle BER+CDep 的 random/balanced 核心 A/B。
旧包没有 `clean/same/swapped/unseen`，因此这四套扩展评价会显示 missing warning，但不会阻止训练和
random 指标比较。

需要完整五套评价时，再上传以下单个文件并创建新 OpenI 数据集：

```text
local_runs/cle_hfl_prepared/fedease_cle_prepared_alpha05_gamma09_seed0.tar.gz
```

文件约 `623.29 MiB`，包含：

```text
4 个 CLE-HFL 私有客户端训练集
CIFAR-100 public 数据
clean / same / random / swapped / unseen 五套评价数据
数据协议 metadata 与 SHA256 audit
```

脚本会通过 `c2net.context.prepare()` 获取挂载路径，并在 `/tmp/dataset` 等位置自动查找该包，
无需手写容器内绝对路径。

## 2. OpenI 任务表单

推荐：

```text
计算资源：英伟达 GPU，1 x V100 32GB
镜像：ubuntu22.04-cuda11.8.0-py310-torch2.1.0-tf2.14.0
项目：fedprime-d2c
代码分支：包含本次 FedEASE 完整实现的分支
启动文件：scripts/openi_fedease_entry.py
```

运行参数必须使用两个半角连字符，例如：

```text
--mode=oracle_probe
```

不要写成 `----mode=full`。

## 3. 四种启动模式

### 3.1 第一优先：Oracle 本地机制 A/B

```text
--mode=oracle_probe
```

它依次运行：

```text
Oracle control：AugMix/JSD/DCL
Oracle BER+CDep：AugMix/JSD/DCL + BER + CDep
```

两者均为 12 轮、无通信、相同 seed/数据/模型/优化器。输出会自动比较 Avg、Worst、WCCA、CFG。
这是当前必须先跑的实验。

### 3.2 第二步：PEW 可学习环境估计

```text
--mode=pew_probe
```

训练 5 epoch 的 PEW，再用预测环境运行 12 轮 BER+CDep。重点检查：

```text
PEW validation environment accuracy
private group accuracy
unknown rate
WCCA / CFG / Avg / Worst
```

### 3.3 第三步：Oracle EBST 通信

```text
--mode=ebst_probe
```

使用 Oracle 环境运行 12 轮 BER+CDep+EBST+stability gate+SCP。这样先隔离通信模块本身，
不把 PEW 估计误差混入 EBST 判断。

### 3.4 最后：完整 FedEASE

```text
--mode=full
```

运行 20 epoch PEW 和 40 轮 learned BER+CDep+EBST+gate+SCP。只有前三个 probe 均通过后才运行。

## 4. Go/No-Go 顺序

```text
1. oracle_probe：BER+CDep 是否提高 WCCA、降低 CFG，且 Avg/Worst 不明显下降？
2. pew_probe：PEW 是否能可靠识别私有 corruption environment？
3. ebst_probe：EBST 是否在 BER+CDep 之上带来独立增益？
4. 三项均通过后才运行 full。
```

建议门槛：

```text
WCCA 上升
CFG 下降
Avg/Worst 不下降超过约 1 point
PEW 不应大量退化为 unknown
EBST 的 gate、active weight、SCP conflict 均应有有限且非零的诊断值
```

## 5. 日志与回传

启动入口使用无缓冲 Python，并在 PEW epoch、客户端本地 batch、每轮通信和扩展评价阶段输出 heartbeat。
任务结束后通过 `c2net` 自动回传：

```text
fedease_oracle_probe_outputs.tar.gz
fedease_pew_probe_outputs.tar.gz
fedease_ebst_probe_outputs.tar.gz
fedease_full_outputs.tar.gz
```

压缩包包含实验目录、逐轮指标、五套评价 CSV/JSON、summary 和方法状态文档。

## 6. 重要边界

代码已完成专项测试和本地真实数据 smoke，不等于方法已经优于 RAHFL。第一次 OpenI 任务应选择
`oracle_probe`，不要直接运行 `full`。如果 Oracle BER+CDep 本身不能改善 CLE-HFL 指标，继续训练
PEW 或 EBST 只会增加变量和算力成本。
