# PEW 小论文网页端 GPT 写作交接包

日期：2026-09-06

## 一句话定位

本文研究模型异构联邦学习中的 **class-corruption entanglement（CLE）**：当类别与数据损坏在客户端内形成系统性虚假相关时，模型可能借 corruption 猜类别，而非学习 corruption-invariant semantics。论文的保守主线是：提出一个受控 CLE-HFL v2 benchmark，并用 **calibrated hard PEW + hard BER** 作为 taxonomy-assisted 的经验缓解方法；不把它包装成 taxonomy-free 新理论或普适 SOTA。

## 必须保持的事实

1. Phase-A0 的 paired counterfactual 与 DSA 已证明 strong CLE directional shortcut 存在；后续 taxonomy-free 方法失败不否定场景本身。
2. HFL-vs-Local 归因显示该机制主要是 local-first，通信放大/坏教师不是主故事。
3. 最终方法是 `calibrated hard PEW + hard BER + AugMix/JSD/DCL + strict AsymHFL-val`，CDep 已从最终方法移除。
4. PEW 在公共 CIFAR-100 carrier 上使用人工 corruption taxonomy 合成六类环境监督；不读取私有 corruption metadata 训练，但绝不 taxonomy-free。
5. BER 是类别均匀、类别内按截断 support 的平方根加权平均环境风险，不是 GroupDRO/CVaR，也不应声称新的 DRO 原理。
6. exact 最终方法的主证据只有固定 `seed0_split0`、训练 seed 0、12 轮，以及同 seed 的消融、LOO 和效率。
7. 历史 3-seed 和 40-round candidate 含 CDep，只能作为早期完整系统的 supporting evidence。
8. strict LOO 只证明已知 family 内的具体 operator 留出泛化，不证明全新 family 或任意真实 corruption。
9. smoke/benchmark 不是科学证据；最新 LCRE/CVRS 等失败路线不进入本文方法。
10. 不使用“首次”“taxonomy-free”“适用于任意未知损坏”“显著优于”“SOTA”等当前证据不能支持的表述。

## 方法摘要

PEW 环境集合是 `clean/noise/blur/weather/digital/unknown`。小型卷积网络在 5000 张公共图像的合成损坏版本上训练：环境分类 CE 加 `0.25` 倍严重度 CE，5 epochs，公共验证集选择最佳 checkpoint，并在公共验证集上校准 unknown rejection threshold。冻结后，它为每个私有 fit 样本生成 hard pseudo-environment。

客户端只在自己的 strict fit 子集上计算 `n[c,e]`。有效组要求 `n[c,e] >= 2`，类内权重正比于 `min(n[c,e],32)^0.5`，再对有效类别均匀平均。完整本地损失：

```text
L_local = L_BER + 12 L_JSD + L_DCL
```

通信仍是 strict AsymHFL-val：fit 用于训练，client-private audit 用于通信路由，最终测试标签只报告。

## 可直接使用的核心结果

### 主结果

| 方法 | Last-five Avg | Worst | WCCA | CFG |
|---|---:|---:|---:|---:|
| PEW+BER | **34.6320** | **29.4280** | **7.2500** | **24.6400** |
| Local-only | 30.4367 | 25.7573 | 0.3000 | 30.8350 |
| RAHFL | 30.0853 | 25.0427 | 0.8500 | 30.4400 |
| AugHFL | 26.6993 | 20.5587 | 0.0000 | 34.7150 |
| FedProto | 24.1250 | 20.5493 | 0.7000 | 37.5850 |
| FedDF | 23.6607 | 19.2507 | 0.3500 | 38.3950 |
| KT-pFL | 23.6587 | 19.5467 | 0.3500 | 38.7300 |
| FCCL | 23.3163 | 19.2280 | 0.7000 | 37.4000 |
| FedMD | 23.2023 | 19.1293 | 0.4000 | 39.0150 |
| RHFL | 17.1053 | 14.9800 | 0.0000 | 36.4550 |

PEW+BER 相对 RAHFL：Avg `+4.5467 pp`，Worst `+4.3853 pp`，WCCA `+6.4000 pp`，CFG `-5.8000 pp`。

### 关键消融

| Arm | 正确名称 | Avg | Worst | WCCA | CFG |
|---|---|---:|---:|---:|---:|
| A0 | RAHFL control | 30.0853 | 25.0427 | 0.8500 | 30.4400 |
| A1 | Calibrated hard PEW + hard BER | **34.6320** | **29.4280** | **7.2500** | 24.6400 |
| A2 | CDep only | 30.4070 | 24.7707 | 1.1500 | 30.7750 |
| A3 | PEW+BER+CDep | 34.0230 | 28.9467 | 5.9000 | 24.1200 |
| A4 | Fixed-threshold PEW+BER | 33.5820 | 28.6040 | 5.0500 | 27.0700 |
| A5 | Shuffled PEW labels + BER | 31.5437 | 26.0147 | 2.5500 | 37.3750 |
| A6 | Oracle family + BER | 35.1200 | 30.7253 | 7.7000 | **20.6900** |

注意：历史结果把 A1 写成 `BER-only`，论文必须改名，因为它仍使用 learned PEW labels。

### Operator-level LOO

strict LOO 从公共 PEW train/val 和私有 fit 中同时排除：`impulse_noise, zoom_blur, fog, pixelate`。

| 方法 | Avg | Worst | WCCA | CFG |
|---|---:|---:|---:|---:|
| RAHFL | 30.0853 | 25.0427 | 0.8500 | 30.4400 |
| Standard PEW+BER | 34.6320 | 29.4280 | 7.2500 | 24.6400 |
| Strict-LOO PEW+BER | **34.9880** | **31.2973** | 5.4500 | **24.3300** |

### 诊断与效率

```text
PEW private group accuracy       62.21%
PEW public val env accuracy      57.40%
PEW unknown AUROC                0.8167
PEW validation ECE               0.0341
PEW validation NLL               1.0825
PEW+BER matched time/round       ~97.7 s
RAHFL matched time/round         ~94.7 s
round-time overhead              ~3.2%
```

效率限制：PEW witness 训练和私有标注发生在 round 计时前，所以 3.2% 不是完整端到端 overhead。另一批统一运行得到 1.6%，正文若只选一个数字，优先用同一 remaining-baselines 归档中的 matched 3.2%，并在附录说明测量批次差异。

## 建议论文结构

1. Introduction：模型异构 FL 中类别—损坏虚假相关的问题、local-first 机制、本文保守贡献。
2. Related Work：spurious correlation under corruption；heterogeneous FL/KD；pseudo-group robust learning；class-conditional balancing。
3. CLE-HFL v2：客户端、类别、operator mapping；fit/audit/test 隔离；paired counterfactual 与 DSA。
4. Method：公共 taxonomy 构造、hard PEW、公共阈值校准、hard BER、与 AugMix/JSD/DCL 和 AsymHFL-val 的组合。
5. Experiments：主表、A0--A6、operator LOO、诊断、效率。
6. Discussion：为何 shuffled labels 失败、oracle gap、local-first 含义。
7. Limitations：taxonomy、人造场景、单 exact seed、LOO 边界、历史 CDep 证据归属。
8. Conclusion：受控 benchmark 和 taxonomy-assisted mitigation 的经验结论，不作普适方法宣称。

## 建议贡献表述

可以写：

- 构建了一个受控的 model-heterogeneous CLE-HFL benchmark，使 client-specific class-corruption mappings 和 operator-cell harm 可被系统评估。
- 通过 paired counterfactual/DSA 与 HFL-vs-Local 对照，实证展示 shortcut 是 strong、directional 且主要 local-first。
- 研究一个 taxonomy-assisted 两阶段基线：公共合成环境 witness 为私有样本生成粗环境标签，随后在每个类别内平衡预测环境风险。
- 在固定 CLE 场景的 matched seed-0 实验中，该基线改善 Avg/Worst/WCCA/CFG；消融和 family-internal operator LOO 支持其环境结构解释。

禁止写：

- 第一个 class-corruption spurious correlation 问题。
- 第一个 pseudo-group 后再做 robust/balanced learning 的方法。
- PEW 自动发现潜在环境、taxonomy-free 或不需要环境监督。
- BER 是新 DRO、GroupDRO 或 CVaR。
- exact PEW+BER 已完成多种子和 40 轮验证。
- 对任意未知 corruption、新 family、复合 corruption 或真实世界 shift 有保证。

## 给网页端 GPT 的直接指令

将本文件连同同目录的 `PEW_CODE_AUDIT_ZH.md`、`EVIDENCE_LEDGER_ZH.md` 和三张 CSV 上传给网页端 GPT，然后发送：

```text
请基于我上传的 PEW 论文交接包写一版中文学术小论文初稿。必须严格遵守证据台账和 claim boundaries，不得自行补造实验、显著性、文献结论或多种子结果。论文定位为“CLE-HFL 受控 benchmark + taxonomy-assisted empirical mitigation”，最终方法固定为 calibrated hard PEW + hard BER + AugMix/JSD/DCL + strict AsymHFL-val，不包含 CDep。历史 3-seed/40-round 结果含 CDep，只能在讨论或补充材料中明确标为早期完整系统的支持性证据，不能放入 exact 方法主结果。A1 不得称为 BER-only。请先输出：题目候选、摘要、完整章节初稿、三张表的正文解读、Limitations、以及需要我补充的引用占位符清单。对任何你无法从材料确认的事实，用 [待确认] 标记，不要猜测。
```

## 当前下一步

现在即可在网页端生成第一版文字，不需要等待新实验。初稿完成后，应回到本仓库做一次“逐句证据审计”，重点检查：方法是否误称 taxonomy-free、A1 命名、CDep 归属、exact seed 数、LOO 外推和效率计时口径。
