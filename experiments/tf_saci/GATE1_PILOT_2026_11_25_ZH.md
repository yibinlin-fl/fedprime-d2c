# TF-SACI Gate 1-Pilot 固定参数探索性实验

原开发集审计和 Gate 1 预注册保留。Pilot 由作者批准，在独立开发集未解决时检验 SSA 可行性。试验只用预先固定的 `phi_tex`、`lambda_abs=0.5`，分别训练 `ssa_poe`（eta=1.0）和 `ssa_poe_ce_anchor`（eta=0.5）。不使用 10,000 张候选图或超参数搜索。探索性结果不转作独立确认。原 Go/No-Go 数值作为报告目标，未达到时记录每项权衡。CC-Mix 和 PS-PID 均关闭。

## 数据、配置与集成

- 复用主协议 AsymHFL-val 与 `moderate_arm_config("asymhfl_erm", mode="formal")`。map2，partition/training seed 0，alpha=1.0，rho=0.5，gamma=0.9，四端初始化权重原 SHA256，零预训练，40 轮，每轮每端 1 local epoch、最多 64 个本地 batch、每批 64 张，公开 CIFAR-100 5,000 抽样、每轮 4 个 batch、每批 128 张。路由仍使用四端独立 audit 和原 AsymHFL 规则。两个 arm 每端同一组 fit 样本按同一标准视图训练。
- 本机打包的训练文件为固定私有图像、任务标签、源 ID、fit/audit 行号、公共 CIFAR-100 tar、四个初始化权重、四份 Gate 0 `gamma09/phi_tex` 缓存。原数据、初始化和缓存按冻结 SHA256 校验。该训练目录不含 corruption ID、私有 binding 元数据、候选开发图、正式测试或 paired DSA。训练进程只接收该目录。每批用 client + 局部行号 + source ID + 任务标签校对缓存。冻结偏置附着于存盘图像 ID，标准随机裁切和翻转沿用基线，不为额外视图复用偏置。
- 沿用原训练循环、本地优化、梯度裁剪和 KD。局部任务损失使用原均值 CE 的凸组合，记录两项损失和各自加权 logit 梯度范数。公共蒸馏、审计与最终推理使用原模型 logits。显式保留旧逐轮测评开启顺序 DataLoader 时的 CPU RNG 消耗，避免取消测评后改变公开采样。

## 本机验收

17 项单测通过。RTX 3050 Laptop 上执行了两臂各 2 轮、每轮每端 2 个私有 batch 的 smoke，正式训练预算未在本机运行。Smoke 用 640 张公共图强制覆盖迭代器重建，公开 batch size 和每轮批次数保持原值。16 个本地变换+样本身份轨迹、8 个公共 batch、四端各 12 个 audit batch 的哈希在两臂一致。8 个本地首 batch 图像与标签哈希还与冻结的 OpenI ERM 历史回传一致。每端每轮首 batch 的加权损失复算通过。初次 GPU smoke 的哈希追踪报错已修复，后续两臂均完成。

## V100 执行、回传与统一评价

分配两个 OpenI 任务，每任务 1×V100 32 GB。入口 `scripts/openi_tf_saci_pilot_entry.py` 复用 c2net 及原 OpenI 回传工具。输入使用新的**私有**训练白名单包，下载目录和精确的参数见本机启动卡。回传第 20/30 轮权重快照，以及最终 40 轮 recovery 包、训练日志、状态和清单。中间包保存模型权重用于核验或推理，完整 optimizer 续跑状态属于另一项能力。两臂各自保留其失败或部分回传。

只有两臂均返回符合冻结合同的第 40 轮权重，才统一运行 `python -m experiments.tf_saci.pilot_evaluate`。分析器先核对两臂配置、输入、运行时、私有批次轨迹以及权重 SHA256，连同旧 ERM、PEW+BER 归档的权重身份先落盘，再打开 sealed 网格。输出四行 pooled/逐客户端的 DSA、Clean、Balanced Corrupted、Mixed、Operator-grid、WCCA、CFG，训练与评价成本、源图配对 DSA 区间。旧目标为 DSA ≤ 0.090024、Clean ≥ 48.700%、Mixed ≥ 42.354%。一次固定 seed 的结果用于决定是否重新设计独立开发及确认性评价。
