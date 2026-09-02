# CLE-HFL K1-A：Head-only SDMN Checkpoint Surgery

状态：实现与smoke阶段。Formal保持锁定。

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

## 6. Formal lock

网页handoff没有明确冻结optimizer、formal maximum surgery steps及backtracking细节。当前实现为
smoke/calibration使用Adam、KL rollback、factor 0.5、最多12次rollback；这些不是formal科学结论。

Formal入口会主动拒绝执行，直到：

1. calibration产物完成并审计；
2. 每个client/fold LR写入冻结manifest；
3. maximum surgery steps与optimizer/backtracking规则正式冻结；
4. formal primary封存、oracle解锁和最终三类verdict实现完成。

不得绕过该锁直接运行formal或完整CLE-HFL训练。
