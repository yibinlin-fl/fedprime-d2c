# CLE-HFL V0.6-SKILL 修改日志

**Base**: `CLE_HFL_LATEX_V0_5_EXPERIMENT_SKELETON_OVERLEAF_20260922.zip`  
**New version**: V0.6-SKILL  
**Editing workflow**: Paper Review → Academic Writing Skills → Scientific Writing → Defensive Writing Auditor → Humanizer → Final Fact Audit  

## 1. 总体变化

V0.6-SKILL 没有添加新方法、新 loss、新通信模块或新实验结果。它做的是：

1. 根据最新 handoff 把 **map2 三 training-seed Formal** 正确传播为 Axis I 主证据；
2. 把 Experiments 重构为 **Axis I / Axis II / Shared validation**，避免把不同问题混成总排行榜；
3. 把 V0.5 skeleton 中尚未获最新 authority 冻结的具体 M3/S1 协议恢复为 generic `[EVIDENCE NEEDED]`；
4. 去掉重复 reviewer-facing defense，但保留科学必要边界；
5. 对 Abstract、Introduction、Related Work、Method、Experiments、Discussion、Conclusion 做 ML-conference prose 和 Humanizer pass；
6. 保留 citation key、label/ref、公式含义和所有已有实验数字。

主正文+附录英文 token-style word count 由约 **5373** 降到约 **4926**；删减主要来自重复解释、defensive prose 和内部协议辩护，而不是证据本身。

## 2. 分节修改

| Section | V0.5 → V0.6-SKILL |
|---|---|
| Abstract | 强化唯一科学闭环；显式写 `seed-0 local-first` 与 `map2 three matched training seeds`；FedDF utility failure 保留；删除完整缺口清单。 |
| Introduction | 压成快速问题进入；四项 contribution 保留；引入 Axis I/II；不把 PEW/BER 包装成新网络结构；减少“我们不声称……”式防御。 |
| Related Work | 缩短为 HFL / corruption / spurious-group robustness / federated shortcut 四条邻近线；保留 FedCD preprint、PEW taxonomy-assisted、CVaR fidelity 边界。 |
| Method | 保留 CLE operator-level 与 PEW family-level 区分；DSA scope 更直接；paired cancellation/non-identifiability/source-level inference 收紧；local-first 改为形成分析并保留 M2 pending；BER 增加“压缩而非完全等权”的直观解释。 |
| Experiments | 改为 Axis I matched objectives、shared validation、Axis II HFL context、Oracle boundary、pending evidence；map2 seeds 0/1/2 升级为核心 Formal；JTT 继续 screen-only。 |
| Discussion | 从重复 disclaimer 改为三条实质边界：shortcut vs utility、proxy/taxonomy、statistical/external validity；明确 map2 三 seed 与其余 seed0 的证据层级。 |
| Conclusion | 只总结已支持链条；不再写 submission-management 语言；保留未解决的 M2/M3/taxonomy boundary。 |
| Appendix | 保留 DSA numerical identity、proxy construction、PEW/BER audit、Oracle、JTT screen boundary；改进“numerical identity check”措辞，避免把机器精度当统计显著性。 |

## 3. 表格与实验矩阵修改

### `protocol_information_boundary.tex`

删除旧 skeleton 中未被最新 handoff 冻结的具体 future protocol；只保留已完成/已冻结的信息边界和协议角色。

### `local_first_multiseed.tex`

seed 0 保留；seeds 1/2 明确 pending。没有使用 source bootstrap 替代 training-seed uncertainty。

### `map2_method_comparison.tex`

主表使用已完成的 seeds 0/1/2 Formal mean±SD；JTT Formal row 仅在 internal draft skeleton 模式下显示，且仍为 pending。

### `cross_setting_dataset.tex`

保留 original map / map1 / FedDF-fidelity；future second-private-task row 只写 `dataset/protocol pending`，不指定数据集。

### `hfl_context_baselines.tex`

Axis II 只保留 protocol-matched context skeleton；所有科学指标 pending。benchmark/smoke 不进入 Formal 表。

### `appendix/taxonomy_stress.tex`

移除旧 skeleton 提前锁定的 motion-blur / leave-one-operator-out 方案；改回 `[EVIDENCE NEEDED: bounded taxonomy stress test; protocol pending.]`。

## 4. 去 defensive prose 的代表性变化

处理原则不是删除限制，而是把“想象 reviewer → 自我辩护”改成“事实 → 一次边界”。例如：

- 多次 `We do not claim...` → 直接写支持范围；
- `safe interpretation is therefore...` → 直接给 matched result 和 utility boundary；
- `This is not evidence for...` 重复 → 在 Discussion 集中陈述未支持外推；
- `for a fair comparison` 类措辞 → 直接列出固定条件。

仍保留 FedDF gate failure、learning-floor failure、CVaR c0/c1 exception、source-bootstrap scope 等真实解释边界。

## 5. Humanizer 处理

- Unicode em dash: 清零；
- 模板化自动总结句：显著减少；
- reviewer-facing meta prose：清理；
- 技术 hyphenation 如 `source-paired`, `operator-grid`, `family-level`, `protocol-matched` 因领域语义需要保留，不机械拆除；
- citations、numbers、equations、defined terms、labels/ref 不因 Humanizer 修改。

## 6. 未解决、未填造的内容

仍保持：

- `[EVIDENCE NEEDED: M2 multi-training-seed stability for the matched HFL-versus-Local four-arm factorial; training seeds 1/2 remain pending.]`
- `[EVIDENCE NEEDED: second private dataset; dataset and protocol pending.]`
- `[EVIDENCE NEEDED: bounded taxonomy stress test; protocol pending.]`
- Axis II S2-v2 Formal scientific metrics pending；
- JTT Formal optional/pending；
- 统一 computation-cost timing pending。

这些缺口是研究状态，不属于写作问题，因此没有被 Skill 自动“修复”。
