# CLE-HFL K1-A：Head-only SDMN Checkpoint Surgery

状态：INSPECT、smoke与calibration均已审计通过；Formal K1-A协议已冻结并实现，等待OpenI正式运行。

## 1. 科学问题

K0-B已证明taxonomy-free generic probes可以检测到CLE模型中的
`carrier-stable + class-selective directional response`。K1-A只回答：

> 按检测到的方向定向修改冻结模型的分类头，能否在未见probe bank上降低R，并进一步降低真实CLE DSA？

这不是完整训练，不修改RAHFL/AsymHFL通信，也不恢复PEW/BER/PNCB。

## 2. INSPECT结果

- 复用Phase-B0的16个H0/H9/L0/L9 round-40 checkpoint。
- 四种异构模型均具有`model.backbone`和`model.linear`，可以只更新最后线性分类头。
- 复用K0-B冻结Bank A/B、response定义以及`rho/K/R`实现。
- 复用同一1,000个CIFAR-100 discover index；标签不加载、不使用。
- Phase-B0输入包同时包含原CLE的`test_images.npy/test_labels.npy`，可在primary产物封存后复用
  DSA/WCCA/CFG evaluator。

## 3. Public split

```text
D_discover = K0-B原1,000张，seed 20260901
D_surgery  = 2,000张未见CIFAR-100
D_holdout  = 2,000张未见CIFAR-100
split seed = 20260906
```

Surgery/Holdout从排除完整K0-B 1,000张后的49,000张中无放回抽取，三者严格互斥。保存index与
dtype/shape/bytes联合SHA256。公共标签不会读入。

## 4. 实现对象

核心代码：

```text
fedprime/engine/cle_sdmn_headonly.py
scripts/run_cle_k1_sdmn_headonly.py
scripts/openi_cle_k1_sdmn_headonly_entry.py
tests/test_cle_sdmn_headonly.py
```

实现：

- K0-B完全相同的class-vs-rest centered response；
- active median + top-20% rho probe selection；
- 固定discover direction和rho-normalized weights；
- exact full-carrier SDMN moment；
- Direction-Sham class permutation，`|cos|<=0.2`；
- sensitivity-matched Random-Probe；
- Generic Invariance；
- head-only更新、backbone/BN/dropout冻结；
- public anchor KL trust region与确定性rollback/backtracking；
- checkpoint、trace、direction、selection、feature-cache hash、unseen response与artifact manifest。

预结果固定seed：

```text
sham RNG base seed = 20260907
```

## 5. 当前运行顺序

本地或OpenI smoke：

```bash
python scripts/openi_cle_k1_sdmn_headonly_entry.py --mode=smoke
```

Smoke只用：

```text
H9 client0
Fold A->B
8 discover / 16 surgery / 16 holdout carriers
20 Bank-A discover recipes（按冻结规则选出2个高风险probe）
2 Bank-B unseen recipes
2 optimization steps
Frozen + Targeted + Direction-Sham + Random-Probe + Generic-Invariance
```

Smoke只能输出：

```text
SMOKE_ONLY_NO_SCIENTIFIC_DECISION
```

本地RTX 3050真实资产smoke已通过，并重复执行验证确定性：

```text
INSPECT checkpoints: 16/16
selected Bank-A probes: [17, 7]
matched random probes:  [11, 1]
Targeted objective:      51.9535 -> 45.5894
maximum anchor KL:       0.002899（全部四个手术臂）
split/selection hashes:  repeat一致
feature hashes:          repeat一致
metrics/traces:          repeat逐值一致
focused K1/K0 tests:     20/20 PASS
verdict:                 SMOKE_ONLY_NO_SCIENTIFIC_DECISION
```

这些数值只验证实现、梯度、对照臂、trust region与未见bank接口，不作为Targeted有效性的证据。

OpenI smoke随后完成并通过独立审计：

```text
archive bytes:  138746333
archive sha256: D75911EE280E5A9C3FF09E0B98DBB7C3411544ED50CA19431417007163C8F327
manifest errors: 0
16 checkpoints: present
public split overlaps: 0/0/0
checkpoint change scope: linear.weight/linear.bias only
maximum anchor KL: 0.002898
unseen metric recompute max error: 0.0
verdict: SMOKE_ONLY_NO_SCIENTIFIC_DECISION
```

报告：

```text
deliverables/cle_k1_sdmn_headonly_20260902/SMOKE_AUDIT_ZH.md
```

Smoke审计后再运行数值校准：

```bash
python scripts/openi_cle_k1_sdmn_headonly_entry.py --mode=calibration
```

Calibration对H9/L9全部8个client、A->B/B->A两fold，只使用discover+surgery及训练bank，候选
LR为`1e-4/3e-4/1e-3`，每个10步。它禁止读取holdout bank、DSA、WCCA、CFG、CLE binding和
private test。选择满足有限值、至少8/10步objective不增、anchor KL `<0.005`的最大LR。

Calibration正式审计结果：

```text
archive: cle_k1_sdmn_headonly_seed0_calibration_outputs.tar.gz
bytes:   43365
sha256:  5B1740185723823862D722C4ABE724AC4B1A21B7405C065385D1E330F3ABF80C
1e-4 pass: 16/16
3e-4 pass:  2/16
1e-3 pass:  0/16
verdict: CALIBRATION_PASS_READY_FOR_PROTOCOL_FREEZE
```

原始JSON存在一个不影响数值结果的schema覆盖：候选标量`learning_rate`被同名的10步LR轨迹覆盖。
轨迹完整、所有候选恒定且可由首元素无歧义恢复，因此不重跑。代码已将字段拆成
`candidate_learning_rate`和`effective_learning_rate_trace`；正式标量清单冻结于：

```text
configs/cle_k1_sdmn_headonly_calibration_seed0.json
```

## 6. Formal协议

正式优化合同在看任何正式结果前冻结为：

```text
optimizer          Adam
steps              10
anchor KL          <= 0.02
backtracking       x 0.5
maximum rollback   12
LR                 每个system/client/fold读取冻结manifest
```

Formal固定运行H9与L9、4个异构client、A->B/B->A两fold，每个fold均从原round-40 checkpoint
独立开始。每个fold运行五臂：

```text
Frozen
Targeted SDMN
Direction-Sham
Random-Probe
Generic Invariance
```

正式入口：

```bash
python scripts/openi_cle_k1_sdmn_headonly_entry.py --mode=formal
```

仍复用Phase-B0输入数据集，不需要上传新数据。Formal的2,000张surgery与calibration完全一致；
新增2,000张holdout后不会改变surgery pool：

```text
discover sha256 = 731B8CFF...57F6CA
surgery sha256  = B5441E50...05334A
holdout sha256  = 321C0910...E240EE
```

## 7. 两阶段结果封存

Primary阶段只允许读取无标签CIFAR-100、冻结PRIME bank和checkpoint。它先生成并哈希：

```text
probe selection / direction
optimization traces
five-arm checkpoints
opposite-bank responses and S/Dcf/K/R
primary_result.json
primary_artifact_manifest.json
```

只有primary seal写完后，才动态加载CLE binding、16个真实corruption与test labels，计算
DSA/WCCA/CFG/Avg/Worst/Clean。Primary未通过或真实DSA不下降时不得将普通head扰动描述为
CLE shortcut removal。

正式判决仅有：

```text
GO_TO_TRAINING_INTEGRATION
MECHANISM_PASS_INTEGRATION_NEEDS_REDESIGN
NO_GO_DIRECTIONAL_SURGERY
```

Formal依然不是完整训练，不修改通信，不进入K1之后的training integration。

## 8. 旧Formal lock说明

早期版本因缺少最大步数与优化器合同而主动拒绝Formal。上述四项现在均已完成，所以只解除
Formal K1-A入口；完整CLE-HFL训练、通信修改和K1之后阶段仍保持禁止。
