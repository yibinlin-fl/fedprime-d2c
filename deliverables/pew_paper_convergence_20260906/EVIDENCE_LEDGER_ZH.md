# PEW 论文证据台账

日期：2026-09-06

## 证据分层

| ID | 证据 | 方法是否为当前精简 PEW+BER | 可用于什么主张 | 禁止外推 |
|---|---|---|---|---|
| E1 | 固定 `seed0_split0`、训练 seed 0、12 轮完整基线比较 | 是 | 在同一 CLE-HFL v2 协议下，PEW+BER 的 Avg/Worst/WCCA/CFG 优于所列基线 | 多场景、统计显著性、任意预算 SOTA |
| E2 | A0--A6 matched 消融 | 是，A1 为当前方法 | 有信息的 PEW 分组、公共阈值校准和 BER 是 seed-0 增益的主要来源；CDep 无稳定独立贡献 | 把 A1 写成“没有 PEW 的 BER-only”；把 oracle 写成可部署方法 |
| E3 | strict operator-level LOO | 是 | 每个已知 family 留出一个具体 operator 后，收益仍存在 | 未见 family、任意未知/复合 corruption、现实域迁移 |
| E4 | PEW 诊断：private group acc 62.21%，public val env acc 57.40%，unknown AUROC 0.8167 | 是 | PEW 提供了有信息但不完美的粗环境标签 | PEW 精确恢复真实 corruption；诊断标签参与训练 |
| E5 | matched 逐轮计时：PEW+BER 约 97.7 s，RAHFL 约 94.7 s | 是 | 训练轮内额外时间约 3.2% | 完整端到端仅增加 3.2%；PEW 准备发生在 round 计时前 |
| E6 | 12 轮训练 seed 0/1/2 与 40 轮 seed 0 正结果 | 否，历史 candidate 含 CDep | 早期完整系统在固定 CLE 场景具有训练随机性和长程稳定性的支持性迹象 | 写成 exact PEW+BER 的多种子或 40 轮证据 |
| E7 | CDep matched 消融/敏感性 | 否 | 支持从最终方法移除 CDep | 把 CDep 包装为当前贡献 |
| E8 | 2026-08-16 claim audit | 适用于当前定位 | PEW+BER 是 taxonomy-assisted 强基线/经验机制；CLE-HFL 可作为受控 benchmark extension | “首次”、taxonomy-free、新 DRO 原理、强方法创新或广义 SOTA |

## 精确结果

当前精简方法相对 RAHFL 的 seed-0、last-five 差值为：

```text
Avg   +4.5467 pp
Worst +4.3853 pp
WCCA  +6.4000 pp
CFG   -5.8000 pp
```

strict LOO 相对 RAHFL：

```text
Avg   +4.9027 pp
Worst +6.2547 pp
WCCA  +4.6000 pp
CFG   -6.1100 pp
```

历史含 CDep 的三训练种子 last-five 平均差值：

```text
Avg   +4.5880 pp
Worst +4.2169 pp
WCCA  +5.5500 pp
CFG   -6.7150 pp
```

历史含 CDep 的 40 轮 seed-0 last-ten 差值：

```text
Avg   +4.9292 pp
Worst +3.2987 pp
WCCA  +9.8750 pp
CFG   -5.4700 pp
```

## 原始证据定位

- 主表与完整基线：`deliverables/cle_remaining_baselines_20260809/RESULT_SUMMARY_ZH.md`
- 首批外部基线与 PEW 诊断：`deliverables/cle_external_baselines_20260807/RESULT_SUMMARY_ZH.md`
- A0--A6 消融：`deliverables/cle_local_ablation_20260807/RESULT_SUMMARY_ZH.md`
- operator-level LOO：`deliverables/cle_pew_loo_20260809/RESULT_SUMMARY_ZH.md`
- 历史三训练种子：`deliverables/strict_pew_asymhfl_val_multiseed_20260804/RESULT_SUMMARY_ZH.md`
- 历史 40 轮：`deliverables/strict_pew_asymhfl_val_40round_seed0_20260805/RESULT_SUMMARY_ZH.md`
- 论文主张审计：`docs/research/status/PEW_BER_PAPER_CLAIM_AUDIT_2026_08_16_ZH.md`

## 当前证据缺口

1. exact calibrated hard PEW + hard BER 尚无独立训练 seed 1/2。
2. exact 方法尚无 40 轮耐久性结果；历史 40 轮含 CDep。
3. 只有一个固定 CLE 数据场景，尚无跨 mapping、alpha/gamma 或真实数据验证。
4. operator-level LOO 只是在已知四个 family 内留出具体算子。
5. 现有新颖性审计不支持把 PEW+BER 写成强方法首创。

这些缺口不阻止现在写保守的小论文初稿，但必须进入 Limitations，且不得靠措辞隐藏。
