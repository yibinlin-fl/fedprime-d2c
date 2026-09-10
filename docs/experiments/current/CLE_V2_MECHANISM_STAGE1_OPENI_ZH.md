# CLE-v2 Mechanism Stage-1 OpenI 运行说明

状态：2026-09-10 seed-0、12-round Formal 已完成；四项冻结门槛全部通过，正式 GO。

## 目标与边界

Stage-1 只验证 CLE-v2 的 operator-level directional shortcut 与 local-first 归因：

```text
h0_b = gamma0   + HFL   + RAHFL baseline
h9_b = gamma0.9 + HFL   + RAHFL baseline
l0_b = gamma0   + Local + RAHFL baseline
l9_b = gamma0.9 + Local + RAHFL baseline
```

Stage-1 不训练、加载或审计 PEW，不启用 BER/CDep，也不判断插件效果。它复用现有数据包，
其中包含 PEW 文件不代表本阶段使用这些文件。

## 冻结训练协议

Formal 固定：training seed 0、12 rounds、每客户端每轮最多 16 local batches、batch size
64、Adam `lr=0.001`、AugMix/JSD/DCL、`lambda_jsd=12`、4 public batches/round、完整
client-private audit routing、`num_workers=0`。训练过程的 test 只取1 batch作运行诊断；完整
1,000 source × 15 operator、severity 3 的 paired DSA 只在最终 checkpoint 上计算。

Benchmark 固定为同样的4臂与每客户端16 batches，但只跑1轮；bootstrap/shuffle 分别为
100/100，仅验证链路和成本。Formal 使用2,000次 source-paired bootstrap 和1,000次
binding shuffle。

## 数学对象

```text
HFL CLE effect       = DSA(h9_b) - DSA(h0_b)
Local CLE effect     = DSA(l9_b) - DSA(l0_b)
Communication add-on = HFL CLE effect - Local CLE effect
Local/HFL share      = Local CLE effect / |HFL CLE effect|
```

## 冻结门槛

- `L0`：四臂最终 operator-grid pooled accuracy 均 `>=20%`。
- `M1`：HFL CLE effect `>=0.10`，CI95 下界 `>0`，4/4客户端为正。
- `M2`：`h9_b` observed DSA 高于 shuffled-binding null p95，置换 `p<=0.05`。
- `M3`：Local CLE effect `>=0.05`，CI95 下界 `>0`，4/4客户端为正，且
  `Local/HFL CLE effect>=0.50`。

Formal 判定：

```text
L0失败                              -> INCONCLUSIVE_UNDERTRAINED
L0通过且M1/M2/M3全部通过             -> GO_CLE_V2_MECHANISM_STAGE1
L0通过但任一机制门失败                -> NO_GO_CLE_V2_MECHANISM_STAGE1
```

只有 `INCONCLUSIVE_UNDERTRAINED` 才允许按预注册规则另行讨论32 batches版本；机制失败不能靠
加训练量翻案。只有 Stage-1 GO 才允许讨论插件阶段。

## 本地验证

- `tests/test_cle_v2_factorial.py + tests/test_cle_v2_mechanism_stage1.py`：10/10 PASS。
- 机制范围输入审计 PASS：40,000私有样本、1,000×15 DSA grid；`pew_audited=false`。
- 四臂一批次 CUDA smoke 全部完成并保存最终模型。
- gamma00 的 HFL/Local local batch/AugMix trace 完全一致；gamma09 同样一致。
- 四臂最终 checkpoint 的小型平衡 source DSA/ bootstrap/shuffle 分析链执行完成。
- smoke 数值不得作为科学结果。

## Formal 结果（2026-09-10）

结果包 `cle_v2_mechanism_stage1_seed0_formal_outputs.tar.gz` 为8,979,168 bytes，SHA256：
`A23312C97A1B9FE2A4DAA8341BB83CD326553A550726CE626097FF68AF4FF4B2`。四臂均完成12轮，
输入审计与配对轨迹审计通过，独立从 `STAGE1_PREDICTIONS.npz` 复算得到：

```text
arm    pooled operator-grid acc    pooled DSA
h0_b   24.9300                     -0.000245
h9_b   21.4367                      0.119644
l0_b   25.5683                     -0.001914
l9_b   22.4233                      0.106046

HFL CLE effect       0.119889  CI95 [0.118050, 0.121562]
Local CLE effect     0.107960  CI95 [0.106145, 0.109678]
communication add-on 0.011929
Local/HFL share      90.05%
h9 shuffled null p95 0.029559  p=0.000999
```

L0/M1/M2/M3 全部 PASS，verdict 为 `GO_CLE_V2_MECHANISM_STAGE1`，不允许也不需要触发32-batch
欠训练补跑。总实测时间15,726.38秒（约4小时22分），其中分析91.68秒；峰值显存约5.4GB。

该 Formal 是 fixed CLE scenario、training seed 0 的机制证据。source-paired bootstrap 不覆盖
训练随机性，结果不自动外推到其他训练 seed 或 CLE mapping。40,000 是输入审计中的可用私有样本
总数；由于每轮每客户端最多16×64次取样，不能写成每个模型完整遍历了40,000个唯一样本。

## 已完成的 OpenI 启动记录

本次完成的 Formal 使用：

```text
启动文件：scripts/openi_cle_v2_mechanism_stage1_entry.py
参数：mode=formal, confirm_formal=true
```

此前本地smoke与OpenI成本评估只用于执行验证；用户明确批准后才解锁上述Formal。Stage-1现已
得到正式GO，不需要重复运行benchmark、Formal或32-batch版本。

```text
benchmark 输出：cle_v2_mechanism_stage1_seed0_benchmark_outputs.tar.gz
formal 输出：cle_v2_mechanism_stage1_seed0_formal_outputs.tar.gz
```

下载包保存配置、指标、配对轨迹、完整 final operator probabilities、DSA 和审计，但主动排除
checkpoint，避免数百MB冗余。benchmark 不是科学证据。
