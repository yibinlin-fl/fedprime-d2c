# CLE-HFL v2 × PEW+BER 八臂配对闭环

状态：OpenI benchmark 已通过完整性审计；一阶 Formal 成本约 78.45 V100 GPU-hours，
Formal 因成本未授权。

## 研究问题

同一套 CLE-HFL v2 源样本上回答两座桥：

1. **机制桥**：从 `gamma=0` 到 `gamma=0.9`，concrete-operator DSA 是否显著上升；该上升在 Local 中是否已经存在，从而支持 local-first，而不是把通信写成根因。
2. **插件桥**：固定同一个公共 PEW 后，PEW+BER 是否在 HFL 和 Local 中都降低强 CLE 带来的 DSA 增量，同时保住 Avg/Worst/WCCA/CFG。

这不是新方法搜索，不包含 CDep、EBST、WEC-BER、CRSF、CVRS 或任何 taxonomy-free 路线复活。

## 数据与可见性

- 私有底图为 CIFAR-10；4 客户端，每客户端 10,000 个互不重叠 source。
- `gamma00` 与 `gamma09` 逐客户端共享 source、label、severity draw、Dirichlet partition、fit/audit split 和模型初始化；只允许 operator draw 随 gamma 改变。
- 15 个 CIFAR-C core operator 中固定 11 个 seen operator 用于训练；4 个 unseen operator 不进入私有训练。
- paired DSA test 为 1,000 个 class-balanced source × 15 operator，固定 severity=3，共 15,000 张。
- 训练只读取图像、类别标签以及 PEW 伪环境标注。真实 operator/family/severity/binding 只供最终报告和 DSA，禁止用于梯度、通信路由、选择、早停或调参。
- PEW 只在公共 CIFAR-100 合成环境上训练一次；checkpoint、阈值和两条件 annotations 全部冻结并校验 SHA256。

## 八臂

| arm | 通信 | gamma | 本地目标 |
|---|---|---:|---|
| `h0_b` | strict AsymHFL-val | 0.0 | AugMix/JSD/DCL |
| `h9_b` | strict AsymHFL-val | 0.9 | AugMix/JSD/DCL |
| `l0_b` | Local | 0.0 | AugMix/JSD/DCL |
| `l9_b` | Local | 0.9 | AugMix/JSD/DCL |
| `h0_p` | strict AsymHFL-val | 0.0 | PEW+BER + AugMix/JSD/DCL |
| `h9_p` | strict AsymHFL-val | 0.9 | PEW+BER + AugMix/JSD/DCL |
| `l0_p` | Local | 0.0 | PEW+BER + AugMix/JSD/DCL |
| `l9_p` | Local | 0.9 | PEW+BER + AugMix/JSD/DCL |

所有臂固定 training seed 0、12 rounds、local epoch 1、Adam `lr=0.001`、batch size 64、`lambda_jsd=12`。插件固定 hard PEW、hard BER、`support_gamma=0.5`、`count_cap=32`、`min_group_count=2`，不使用 CDep。每轮/客户端/本地 epoch 使用独立 paired RNG reset；同一 gamma 的四臂必须通过首批 source/AugMix trace 全等审计。

## 主要量

令 `D(a)` 为 arm `a` 在固定网格上的 pooled concrete-operator DSA：

```text
HFL CLE effect          = D(h9_b)-D(h0_b)
Local CLE effect        = D(l9_b)-D(l0_b)
Communication add-on    = HFL CLE effect-Local CLE effect
HFL plugin mitigation   = [D(h9_b)-D(h0_b)]-[D(h9_p)-D(h0_p)]
Local plugin mitigation = [D(l9_b)-D(l0_b)]-[D(l9_p)-D(l0_p)]
```

CI 使用 source-paired bootstrap 2,000 次。binding-specificity 使用保持客户端 operator 边际的 class-binding shuffle 1,000 次。同步报告四客户端估计值、last-5 Avg/Worst/WCCA/CFG 和通信交互；不得用准确率替代 DSA。

## Formal 前冻结门槛

- `M1`：HFL CLE effect `>=0.10`，CI95 下界 `>0`，4/4 客户端为正。
- `M2`：`h9_b` observed DSA 高于 shuffled-binding null p95，置换 `p<=0.05`。
- `M3`：Local CLE effect `>=0.05`，CI95 下界 `>0`，4/4 为正，且 `Local/HFL CLE effect >=0.50`。
- `P1`：HFL plugin mitigation `>=0.02`，CI95 下界 `>0`，4/4 为正。
- `P2`：Local plugin mitigation `>=0.02`，CI95 下界 `>0`，4/4 为正。
- `P3`：强 CLE HFL 插件相对 baseline，last-5 `Avg>=+1.5`、`Worst>=+1.0`、`WCCA>=0`、`CFG<=-1.0`。

六项全过才记为 `GO_SEED0_FACTORIAL_CLOSURE`。任一失败即 `NO_GO_SEED0_FACTORIAL_CLOSURE`，不得事后改窗口、门槛、PEW 阈值、BER 参数或补 seed 翻案。即使 seed-0 GO，也只闭合固定场景；多 seed、40 rounds、跨 scenario 和更多 baseline 均需另行批准。

## 已完成的本地验证

- `tests/test_cle_v2_factorial.py`：5/5 PASS。
- 正式输入静态审计 PASS：40,000 私有 source 无重叠；paired 条件、split、初始化、DSA grid、公共 PEW lineage/hash 全过。
- 公共 PEW：5 epochs；最佳 validation environment accuracy `55.4%`，冻结 unknown threshold `0.27`；私有 oracle-family accuracy 仅作诊断，为 gamma00 `57.4575%`、gamma09 `62.0425%`。
- 正式输入八臂一批次 CUDA smoke 全部完成；同一 gamma 的 HFL/Local × baseline/plugin 轨迹逐客户端完全一致；BER 实际触发。

这些只证明数据、加载、训练、检查点、轨迹和分析链路可执行，不是效果证据。

## OpenI 顺序

必须先运行 `mode=benchmark`：正式 batch size 64，每 arm 1 round、每客户端 8 个 local batches，并限制评价批数；仅用于估时和检查 V100S 链路。下载 benchmark 并确认成本后，用户再次明确批准，才可运行 `mode=formal --confirm_formal=true`。

运行稳定性约束：所有模式固定 `num_workers=0`。AugMix 变换含局部 Lambda，启用 worker
不会改变科学协议但会引入跨平台序列化/worker 崩溃风险。2026-09-09 修复后，首臂 `h0_b`
已按完整 benchmark 设置在本地 CUDA 跑通。

```text
入口：scripts/openi_cle_v2_factorial_entry.py
输入：cle_hfl_v2_paired_factorial_seed0_split0_with_pew.tar.gz
benchmark 输出：cle_v2_factorial_seed0_benchmark_outputs.tar.gz
formal 输出：cle_v2_factorial_seed0_formal_outputs.tar.gz
```

Formal 会自动运行八臂、完整 paired DSA 分析、冻结门槛判定并封装结果。benchmark/smoke 不作科学判定。

2026-09-09 benchmark 结果：八臂耗时 `1550.4681 s`，峰值显存 `5394.32 MB`，gamma00
和 gamma09 各自的四臂训练轨迹完全匹配，审计 `PASS`。正式 fit 为 132 batches/客户端，
相对 benchmark 的 8 batches 放大 16.5 倍；按 12 rounds 一阶换算约 78.45 GPU-hours，
且 Formal 还会打开完整 audit/test/DSA。判定为 `BENCHMARK_PASS / FORMAL_NOT_AUTHORIZED`。
详见 `deliverables/cle_v2_factorial_benchmark_20260909/RESULT_SUMMARY_ZH.md`。
