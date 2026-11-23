# TF-SACI Gate 1 开发集来源审计和预注册

**判定 `HOLD_DEV_PROVENANCE`。拒绝把当前 10,000 个候选作为独立开发集。开发集和最终选定的 lambda 仍为空。Gate 1 未启动。**

## 来源审计

主评价使用原始 CIFAR-10 train 的源图 ID，按四客户端、三 gamma 的 `train_source_indices.npy` 和固定 fit/audit 索引追踪。先验证数据清单、源 ID 文件、分区文件和 Kaggle 历史输入的文件哈希，再计算交集。当前 fit/audit 的 40,000 张源图互异。候选是原始训练 ID `[0,50000)` 减去这 40,000 个 ID。

| 可验证的用途 | 用途中的源图数 | 与候选重合 |
| --- | ---: | ---: |
| 当前 map2 fit | 33,995 | 0 |
| 当前 map2 路由 audit | 6,005 | 0 |
| Phase-A1a 40 轮正式实验 fit | 33,994 | 6,826 |
| Phase-A1a 40 轮正式实验路由 audit | 6,006 | 1,138 |
| 历史 Axis I map2 fit / audit | 33,994 / 6,006 | 6,826 / 1,138 |
| 11 月 Kaggle 损坏分配预检 fit / route / eval | 8,000 / 1,000 / 1,000 | 1,606 / 203 / 206 |

已核验历史交集的**去重并集为 8,363**，另有 1,637 个 ID 未被这些记录覆盖。历史 Axis I 和 Phase-A1a 的重合不重复计数。Kaggle 预检使用 CIFAR-10 **训练**源做历史评价，因此其中 206 个候选已经出现在历史评价数据里。没有证据表明这些样本曾用于 SSA 的 lambda 选择，本轮也未进行选参。这里的结论针对开发集是否具备所要求的独立性。

- **主协议最终测试和 sealed DSA。** 准备脚本从官方 CIFAR-10 test split 采样构造 Clean、Balanced 和 paired DSA。候选来自 train split，分区来源不同，即使整数 ID 相等也无交集。本审计核验的是源命名空间，不是跨分区的图像内容去重。未打开正式测试图像、标签、sealed 预测或指标缓存。
- **初始化和公共资源。** 主协议的四份初始权重由 seed 0 构建模型并保存，预训练为 0。主协议公共蒸馏和冻结 PEW 的公共数据源是 CIFAR-100 train，与候选的 CIFAR-10 train 源命名空间不同。旧的同域 CIFAR-10 public/PCCD 设计明确从旧私有分区的补集取样，目前没有找到可关联实际运行的 `public_indices.npy`。M3 CIFAR-100 入口配置了 CIFAR-10 公共载体及 PEW，但本地预期 benchmark 回传目录为空，这只能说明仍缺实际用量证据。历史公共数据的使用记录因此无法完整排除剩下的 1,637 个候选。
- **历史选参证据范围。** Phase-A1a 和 Axis I 有正式实验记录，Kaggle 有验收过的固定输入和预检评价。更早的 prepared 数据包在当前可见归档中缺少可直接归属的 `train_source_indices.npy`；不将它们计入交集，也不据此认证剩余候选。主协议的正式评测保留现状。

复算命令在实验 worktree 的根目录执行。程序只加载清单和索引数组，不输出源 ID 明细。

```powershell
python -m experiments.tf_saci.audit_dev_sources --local-runs ..
```

对照来源为 `docs/handoffs/latest.md` 的 Phase-A1a 正式记录、`docs/research/status/CLE_AXIS1_FROZEN_BUDGET_HISTORY_2026_11_05_ZH.md`、`docs/research/status/HFL_PEER_CORRUPTION_KAGGLE_RESULTS_2026_11_10_ZH.md` 及 `scripts/prepare_cle_v2_factorial_data.py`、`scripts/prepare_cle_in_domain_public.py`、`scripts/openi_cle_cifar100_submission_entry.py` 的数据来源实现。

## 替代开发集

优先用**独立采集、与 CIFAR-10 类别一致**的开发图像，保留原 map2 数据、初始化、模型和正式锚点。可审计的候选是 [CIFAR-10.1 v6 官方发布](https://github.com/modestyachts/CIFAR-10.1)。它提供独立发布的一批图像，发布方记录了与原 CIFAR-10 的近重复检查。它与 CIFAR-10 仍共享上游图像来源，项目内还需核验历史使用、与 CIFAR-100 公共载体的重复以及使用许可。冻结版本和资产哈希前，它只是候选。若无法完成该证明，另采一批有可验证来源和分类标签的同类图像。

开发集冻结应记录来源、样本 ID、图像和标签 SHA256、类支持、重复排查范围，以及损坏视图生成清单。开发 Clean 用原始图像，Corrupted 在同一批独立源图上按预先列定、与类别无关的公开损坏算子和强度生成；Mixed 为两者 50/50。其标签只用于模型训练结束后的开发评价。用独立开发效用选参，正式 DSA 只在锁定候选后打开。另一方案是在新来源内事先固定 fit/audit/dev/test 并重跑两条基线，这会产生新比较表，无法沿用旧的冻结锚点数值。

## Gate 1 协议和判定

`gate1_protocol.json` 记录预注册方案，当前不能用于启动训练。保持 `asymhfl_val`、map2、`alpha=1.0`、`rho=0.5`、`gamma=0.9`、training/partition seed 0、相同初始化、局部 epoch 1、无预训练。40 轮，每轮每端最多 64 个私有 batch、每 batch 64 张，每轮 4 个公开 batch、每 batch 128 张，公开池和路由规则同原主评价。

令 \(z=f_k(T(x_i))\) 是标准训练变换后主模型的 logits，\(b_i\) 是对同一存盘 fit 图计算的冻结 OOF 偏置，\(B\) 为 batch size。

\[
L_\eta=(1-\eta)\,\mathrm{CE}_{\mathrm{mean}}(z,y)
 +\eta\,\mathrm{CE}_{\mathrm{mean}}(z+\lambda\,\operatorname{sg}(b_i),y).
\]

SSA-PoE 用 \(\eta=1\)，SSA-PoE+CE-anchor 用 \(\eta=0.5\)。两项各为同批均值，权重和为 1。原始 CE 与 PoE 的 logit 梯度权重分别为 \((1-\eta)/B\) 和 \(\eta/B\)，分别乘以对应 softmax 残差。这对齐损失归约与权重，实际梯度范数仍可能不同。后续记录两项 batch CE、总损失及两路加权 logit 梯度的范数，检查 \(\lambda=0\) 时两臂损失和梯度与 ERM 一致。当前损失原语已有单测，训练集成尚未发生。

**数据接入要求。** 冻结基线用 standard 单视图。包装其原有 transform 和 sampler，将 `(client_id, local_row_id, source_id, task_label)` 和变换后的图一起返回，进入本地任务损失前逐项对缓存。\(b_i\) 附着在存盘样本身份，标准随机变换可能改变像素统计。本轮把这作为预定义设计，避免按 batch 下标错配，禁止将原样本的偏置重用于额外 AugMix 视图。先做无损失变更的配对 smoke，比较私有图像和标签轨迹、路由 audit 和公开批次。推理、通信 logits 和正式评价始终走主模型原始输出。

**正式数据隔离。** 现有 `CorruptionSkewClientDataset` 构造器即使设置 `return_corruption=False` 仍加载真实训练损坏 ID；现有训练主循环还会构建 `test_balanced` 并逐轮评价。Gate 1 必须在独立入口中用训练输入白名单和不加载损坏 ID 的样本读取方式，隔离正式评价根目录，并在缺失路由 audit 时失败，杜绝回退到测试数据。正式评价脚本接受冻结配置与 checkpoint 哈希后运行。对四臂复用原主协议 DSA、Grid、Clean、Corrupted、Mixed 定义，并在 `test_balanced` 采用原算子×类别 WCCA（最小有效组准确率）和 CFG（各类有效算子准确率极差的均值）。同时汇报 pooled/各客户端、源图配对区间、原始分数及开销。任一指标缺失均先记录失败。

**开发选参。** 候选 \(\lambda\in\{0.5,1.0\}\)，两个方法共用同一个 lambda。只看第 40 轮的独立开发结果。一个 lambda 合格需两个方法各自的 Clean 和 Mixed 均比同一开发集上的旧 ERM 低不超过 1.0 pp。合格者以两法开发 Mixed 的平均值最高为优先，同分时取较小 lambda。无合格值即停止，继续封存正式测试。已有开发候选 checkpoint 可以直接锁定后进入最终推理，避免重复训练。

**正式 Go/No-Go。** 每个方法必须同时满足 pooled DSA **≤ 0.090024**（相对 ERM **至少下降 0.094490**），Clean **≥ 48.700%**，Mixed **≥ 42.354%**。Clean/Mixed 门槛给 ERM 各 1.0 pp 容忍度，且均高于冻结的 PEW+BER（45.350% / 41.562%）。统一用冻结表精度作判定，并保留原始数据。任一方法三关全过才考虑下一阶段；两法都未过即终止当前 SSA 路线。seed 0 的通过只作为阶段证据。正式结果不用于再选超参。
