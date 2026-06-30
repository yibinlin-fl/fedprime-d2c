# FedPRIME-D2C / PRAC-HFL Current Project Memory

Updated: 2026-07-01

## Current Goal

Build a paper-worthy heterogeneous FL method for:

```text
model heterogeneity + data heterogeneity / Non-IID + data corruption robustness
```

The current baseline to beat is the unified-runner RAHFL baseline.

## Current Main Method

The current mainline is:

```text
NIR-DCL = RAHFL AugMix/JSD local robust training + Non-IID-aware Robust DCL
```

The project is no longer centered on D2C, FedPRIME-PAIR, or current PRAC-HFL
communication. Those are now historical diagnostic experiments. PRAC-HFL runner
is still useful for local-only experiments because it has robust Kaggle heartbeat
logging and can skip communication with `warmup_rounds: 999`.

## NIR-DCL Design

NIR-DCL modifies only the local DCL branch:

```text
CE(clean)
+ lambda_jsd * JSD(clean, aug1, aug2)
+ lambda_nir * NIR-DCL(clean_feature, weak_feature, strong_feature)
```

Implemented components:

```text
1. Class-balanced DCL
   Average per-class losses first, then average over classes present in the batch.

2. Client-local feature queue
   Each client keeps a private per-class feature queue to provide extra positives
   and negatives when a Non-IID mini-batch has too few tail-class samples.

3. Strong-view reliability gate
   Down-weights relation alignment when the strong AugMix view has a poor
   true-class margin.

4. Stable relation alignment
   Replaces the original DCL `softmax(exp(sim) / T)` style relation with a more
   stable `softmax(sim / T)` KL alignment.
```

Key files:

```text
fedprime/methods/nir_dcl.py
fedprime/methods/local_rahfl.py
configs/kaggle_t4_nir_dcl_local_only.yaml
configs/debug_nir_dcl_local_only.yaml
configs/kaggle_t4_nir_dcl_rahfl.yaml
```

## PRAC-HFL Design

Local training follows RAHFL:

```text
AugMix multi-view training
+ CE
+ JSD consistency
+ RAHFL DCLLoss
```

Communication replaces RAHFL AsymHFL:

```text
1. Server selects a public CIFAR-100 mini-batch.
2. Clients upload public logits.
3. Candidate teacher logits are forwarded to each receiver.
4. Receiver performs head-only virtual KD toward each teacher.
5. Receiver evaluates private route CE risk before/after the virtual teacher step.
6. Positive teacher/class effects construct a personalized mixed teacher.
7. Receiver performs a mixed-teacher head step.
8. Independent accept batch decides whether to keep or revert the step.
```

Key implementation:

```text
fedprime/methods/prac_hfl.py
fedprime/methods/local_rahfl.py
configs/kaggle_t4_prac_hfl.yaml
configs/debug_prac_hfl_cifar10c.yaml
scripts/run_kaggle_prac.sh
```

Latest pushed commits:

```text
fa108f7 实现PRAC-HFL接收端自适应通信
5e476ea 增强PRAC-HFL数值稳定性
```

## Safe PRAC-HFL Settings

Current safe config uses:

```text
warmup_rounds: 3
risk_lambda_aug: 0.0
risk_lambda_js: 0.0
virtual_lr: 0.005
head_max_grad_norm: 1.0
train.max_grad_norm: 5.0
train.skip_nonfinite: true
```

Why:

```text
First PRAC run produced NaN at round 029.
Likely causes:
- virtual head distillation step too large
- route risk used CE + AugCE + 12*JSD, causing noisy/huge deltas
- no communication warmup
```

## Current Baselines and Historical Results

RAHFL unified-runner baseline:

```text
config: configs/kaggle_t4_rahfl.yaml
final: avg_acc=56.41, worst_acc=44.72
setting: alpha=0.5, corrupted train/test rate=1, 4 heterogeneous clients
important: no independent 40-epoch pretraining
```

This RAHFL number is a fair resource-limited runner baseline, not full paper reproduction.

D2C / public-logit prior route:

```text
PRIME + LogitAvg final avg_acc≈52.10, worst_acc≈39.72
FedPRIME-D2C final avg_acc≈52.31, worst_acc≈39.78
Oracle D2C final avg_acc≈51.74, worst_acc≈39.13
```

Conclusion:

```text
D2C did not meaningfully beat LogitAvg.
Even oracle prior did not fix it.
D2C is archived as a negative/diagnostic result.
```

FedPRIME-PAIR / CPAD route:

```text
FedPRIME-PAIR final avg_acc≈50.15, worst_acc≈39.83
Best avg_acc≈51.10
CPAD did not beat LogitAvg.
```

Conclusion:

```text
Pairwise public-logit boundary distillation is also archived as a negative/diagnostic route.
```

PRAC-HFL first run before safe fix:

```text
Visible attachment rounds: 001-028
At round 028:
  RAHFL same-round avg_acc=53.21, worst_acc=41.64
  PRAC-HFL avg_acc=53.86, worst_acc=39.52
Best visible PRAC avg_acc=53.86 at round 028
Best visible PRAC worst_acc=42.15 at round 027
Mean accept_rate over visible rounds≈15.18%
Round 029 produced NaN and invalidated the rest of that run.
```

Interpretation:

```text
PRAC-HFL has the strongest signal among our proposed communication variants.
It can match or slightly exceed same-round RAHFL average accuracy before NaN.
Worst-client accuracy remains less stable.
Safe run from commit 5e476ea is the next required experiment.
```

Safe PRAC-HFL public1 result from `outputs/prac_hfl_results.tar.gz`:

```text
config used public_batches_per_round=1
final avg_acc=54.63, final worst_acc=41.88
best avg_acc=55.53 at round 38
best worst_acc=43.43 at round 36
mean accept_rate after warmup=30.4%
```

Important interpretation:

```text
This is a stable low-public-communication result, not the strict fair comparison
against RAHFL. RAHFL uses public_batches_per_round=4.
The main config configs/kaggle_t4_prac_hfl.yaml has been corrected to
public_batches_per_round=4 and experiment_name prac_hfl_cifar10c_alpha05_cr1_t4_public4.
The old public1 setting is preserved as configs/kaggle_t4_prac_hfl_public1_lite.yaml.
```

Safe PRAC-HFL public4 fair result from `outputs/prac_hfl_public4_results.tar.gz`:

```text
config used public_batches_per_round=4
final avg_acc=52.96, final worst_acc=43.27
best avg_acc=52.96 at round 39
best worst_acc=43.27 at round 39
mean accept_rate after warmup=25.5%
mean avg_delta after warmup=-0.0045
```

Comparison:

```text
RAHFL public4 final: avg_acc=56.41, worst_acc=44.72
PRAC public4 gap:    avg_acc=-3.45, worst_acc=-1.45
PRAC public1 final:  avg_acc=54.63, worst_acc=41.88
```

Interpretation:

```text
PRAC communication is not empty: accept_rate is nonzero and checkpoints change.
However, public4 did not improve over public1. More public batches lowered avg_acc
but improved final worst_acc over public1, suggesting PRAC may help weak clients
while causing average-performance negative transfer.
We still need AugMix+DCL local-only to decide whether PRAC adds real value over
local robust training alone.
```

Local-only control config:

```text
configs/kaggle_t4_augmix_dcl_local_only.yaml
method_name: prac_hfl
warmup_rounds: 999
meaning: AugMix + CE + JSD + DCL local training, no PRAC communication for all 40 rounds
```

AugMix+DCL local-only result from Kaggle log:

```text
final avg_acc=56.11, final worst_acc=44.23
best avg_acc=56.94 at round 38
best worst_acc=44.23 at round 39
prac_loss=0 and accept_rate=0 for all rounds
non-finite gradient warnings=822, all from client 2, skipped by skip_nonfinite=true
```

Comparison:

```text
RAHFL final:        avg_acc=56.41, worst_acc=44.72
PRAC public1 final: avg_acc=54.63, worst_acc=41.88
PRAC public4 final: avg_acc=52.96, worst_acc=43.27
Local-only final:   avg_acc=56.11, worst_acc=44.23
Local-only best avg exceeds RAHFL final avg by +0.53
```

Interpretation:

```text
Current PRAC communication does not add positive average-accuracy gain over
AugMix+DCL local robust training. The strongest evidence now is that most of the
performance comes from RAHFL-style local robust learning. Current PRAC should be
treated as weak/negative transfer unless redesigned. The main research direction
should shift toward Non-IID-aware robust DCL/local representation learning, with
communication as a secondary module.
```

## Deliverables

Comparison workbook and figures:

```text
deliverables/prac_vs_rahfl_analysis/rahfl_prac_hfl_comparison.xlsx
deliverables/prac_vs_rahfl_analysis/round_comparison.csv
deliverables/prac_vs_rahfl_analysis/avg_accuracy_curve.png
deliverables/prac_vs_rahfl_analysis/worst_accuracy_curve.png
deliverables/prac_vs_rahfl_analysis/prac_diagnostics.png
```

## Kaggle Running Notes

Use Python streaming launcher. Do not use a long silent `%%bash` cell.

Dataset:

```text
Kaggle input dataset name: fedprime-data
DATA_SOURCE=/kaggle/input/fedprime-data
```

Before running PRAC-HFL, verify:

```text
git log -1 --oneline
expected: 5e476ea 增强PRAC-HFL数值稳定性
```

Expected PRAC logs:

```text
[setup] PRAC-HFL ...
[heartbeat] round 000 start
[heartbeat] round 000 local client 0 start
[heartbeat] round 000 PRAC warmup: skip communication
[heartbeat] round 003 running PRAC communication
[heartbeat] round xxx PRAC client y accept/reject ...
[round xxx] avg_acc=... worst_acc=... accept_rate=... pos_teacher=... avg_delta=...
```

## Next Required Experiments

1. Treat current PRAC communication as weak/negative transfer under the current design.
2. Shift main method design toward Non-IID-aware DCL/local robust representation learning.
3. If communication is kept, redesign it with held-out route/accept split and weaker aggregated updates.

```text
RAHFL local only
RAHFL local + Average KD
RAHFL local + AsymHFL
RAHFL local + PRAC-HFL
```

Purpose:

```text
Separate the contribution of AugMix+DCL local training from the communication method.
```
