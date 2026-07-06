# FedPRIME-D2C / PRAC-HFL Current Project Memory

Updated: 2026-07-05

## Current Goal

Build a paper-worthy heterogeneous FL method for:

```text
model heterogeneity + data heterogeneity / Non-IID + data corruption robustness
```

The current baseline to beat is the unified-runner RAHFL baseline.

## Current Main Method

The current mainline is:

```text
SARA + AsymHFL = AugMix/JSD + Skew-Aware Robust Alignment + RAHFL AsymHFL
```

The project is no longer centered on D2C, FedPRIME-PAIR, PRAC-HFL, or FedCARA.
Those are historical diagnostic experiments. PRAC-HFL runner is still useful for
local-only controls because it has robust Kaggle heartbeat logging and can skip
communication with `warmup_rounds: 999`.

SARA is currently the best-performing mainline because SARA + original RAHFL
AsymHFL is the first setting that beats the fair RAHFL baseline on both final
average accuracy and final worst-client accuracy.

Key files/configs:

```text
fedprime/methods/sara.py
fedprime/methods/local_rahfl.py
fedprime/methods/rahfl_asymhfl.py
configs/kaggle_t4_sara_local_only.yaml
configs/kaggle_t4_sara_rahfl.yaml
configs/debug_sara_local_only.yaml
```

Latest pushed SARA commit:

```text
9df13a7 实现SARA偏斜感知鲁棒对齐
```

## SARA Design

SARA means:

```text
Skew-Aware Robust Alignment
```

It replaces RAHFL's DCL branch while keeping the RAHFL-style robust local base:

```text
CE(clean)
+ lambda_jsd * JSD(clean, aug1, aug2)
+ lambda_sara * SARA(clean_feature, weak_feature, strong_feature)
```

SARA contains:

```text
1. Skew-aware supervised contrastive alignment
   Uses client class counts to rebalance contrastive contributions from head and
   tail classes under label-skew Non-IID.

2. PRIME/AugMix-view reliability gate
   Uses strong-view true-class margin to down-weight unreliable augmented views.

3. Relation alignment
   Uses stable softmax(sim / T) relation matching instead of the more fragile
   softmax(exp(sim) / T) style relation.
```

Current implementation uses AugMix/JSD views from the RAHFL local base, not PRIME.
PRIME remains a historical route unless explicitly resumed.

## SARA Results - 2026-07-02

Result archive:

```text
outputs/sara_rahfl_results.tar.gz
```

Contained runs:

```text
SARA local-only:
  config: configs/kaggle_t4_sara_local_only.yaml
  method_name: prac_hfl
  communication disabled by warmup_rounds=999
  final avg_acc   = 54.10
  final worst_acc = 32.06
  best avg_acc    = 54.59 at round 38
  best worst_acc  = 33.96 at round 17

SARA + AsymHFL:
  config: configs/kaggle_t4_sara_rahfl.yaml
  method_name: rahfl
  cl_module: sara
  final avg_acc   = 57.83
  final worst_acc = 46.59
  best avg_acc    = 57.83 at round 39
  best worst_acc  = 46.59 at round 39
```

Main comparisons:

```text
RAHFL baseline:
  final avg_acc   = 56.41
  final worst_acc = 44.72

AugMix+DCL local-only:
  final avg_acc   = 56.11
  final worst_acc = 44.23

FedCARA v1:
  final avg_acc   = 55.88
  final worst_acc = 45.93
```

Interpretation:

```text
SARA local-only is not strong and should not be claimed as a standalone local
training improvement. It appears to over-regularize or hurt weak clients.

SARA + AsymHFL is currently the best mainline result:
  vs RAHFL: +1.42 avg_acc, +1.87 worst_acc
  vs AugMix+DCL local-only: +1.72 avg_acc, +2.36 worst_acc
  vs FedCARA: +1.95 avg_acc, +0.66 worst_acc

The key story is synergy:
  SARA alone may be too strict, but it changes local robust representations in
  a way that makes AsymHFL public-logit communication more effective under
  label-skew Non-IID.
```

## SARA Alpha=0.5 Seed Validation - 2026-07-05

New result archives:

```text
outputs/rahfl_seed1_results.tar.gz
outputs/sara_rahfl_seed12_results.tar.gz
```

Completed runs:

```text
RAHFL seed=1:
  config: configs/kaggle_t4_rahfl_seed1.yaml
  final avg/worst = 56.645 / 45.29
  best avg/worst  = 56.645 / 45.29

SARA + AsymHFL seed=1:
  config: configs/kaggle_t4_sara_rahfl_seed1.yaml
  cl_module: sara
  final avg/worst = 57.2975 / 46.23
  best avg/worst  = 57.2975 / 46.23
  paired gap vs RAHFL seed=1 = +0.6525 avg_acc, +0.94 worst_acc

SARA + AsymHFL seed=2:
  config: configs/kaggle_t4_sara_rahfl_seed2.yaml
  cl_module: sara
  final avg/worst = 58.0025 / 45.90
  best avg/worst  = 58.0025 / 45.90
```

SARA final results across alpha=0.5 seeds 0/1/2:

```text
seed0: 57.83   / 46.59
seed1: 57.2975 / 46.23
seed2: 58.0025 / 45.90

mean final avg_acc   = 57.71
mean final worst_acc = 46.24
population std avg   = 0.30
population std worst = 0.28
```

Available RAHFL final results across alpha=0.5 seeds 0/1:

```text
seed0: 56.41  / 44.72
seed1: 56.645 / 45.29

mean final avg_acc   = 56.5275
mean final worst_acc = 45.005
```

Important caveat:

```text
The archived partition files named seed0/seed1/seed2 have identical SHA-256
prefixes and identical client_class_counts in these runs. Therefore the current
alpha=0.5 seed validation is best interpreted as different training/randomness
seeds on the same fixed label-skew partition, not as different data partitions.

This is still useful for training stability, but formal paper claims should not
describe it as cross-partition validation unless new genuinely distinct
partition files are generated and audited.
```

Partition seed bug fix - 2026-07-06:

```text
Root cause:
  fedprime/data/loaders.py reused RAHFL-master/Dataset/sampling.py for IID and
  Dirichlet splits. That vendor file resets random.seed(0) and np.random.seed(0)
  at import time, so missing seed1/seed2 partition files could be generated with
  seed0 randomness despite different config seed names.

Fix:
  fedprime/data/loaders.py now implements local IID/Dirichlet partition
  generation with np.random.default_rng(partition_seed).
  All experiment runners and partition/audit/diagnostic scripts pass config.seed
  into partition_private_data().

Verification:
  A temporary alpha=0.5 seed0/1/2 generation produced distinct .npz SHA-256
  prefixes and each client had exactly 10000 samples.

Important:
  Existing historical archives are not changed. If a .npz already exists, the
  runner still loads it for reproducibility. To get genuinely different
  partitions, generate a new partition pack after this fix.
```

Interpretation:

```text
SARA + AsymHFL remains positive under the seed=1 matched comparison and SARA
seed=2 is also strong. The gain over RAHFL is smaller than the original seed0
gap but remains positive on both final average accuracy and final worst-client
accuracy for the completed matched seed=1 run.

The mainline claim is strengthened:
  SARA does not appear to be a seed0-only accident under the fixed alpha=0.5
  partition. It still needs RAHFL seed=2 and stronger/non-extreme alpha checks
  before final paper-level claims.
```

## SARA Alpha=0.3 Validation - 2026-07-06

Result archive:

```text
outputs/sara_vs_rahfl_alpha03_results.tar.gz
```

Setting:

```text
alpha=0.3
seed=0
corrupt_rate=1
rounds=40
same fixed partition for RAHFL and SARA
```

Results:

```text
RAHFL alpha=0.3:
  final avg/worst = 45.8425 / 41.9200
  best  avg/worst = 46.3825 / 43.1300

SARA + AsymHFL alpha=0.3:
  final avg/worst = 46.7325 / 42.7700
  best  avg/worst = 47.0825 / 44.1100
```

SARA gap:

```text
final avg_acc   +0.8900
final worst_acc +0.8500
best avg_acc    +0.7000
best worst_acc  +0.9800
```

Trend:

```text
SARA beats RAHFL in 36/40 rounds for avg_acc and 36/40 rounds for worst_acc.

Last-10-round mean gap:
  avg_acc   +0.6942
  worst_acc +0.5270
```

Partition audit:

```text
nonzero_classes_per_client = [7, 6, 7, 10]
max_client_class_proportion = [0.3625, 0.3669, 0.3583, 0.3716]
```

Interpretation:

```text
The alpha=0.3 split is clearly label-skewed and both methods use identical
client class counts. SARA still wins consistently, but the gain is modest and
smaller than the alpha=0.5 seed0 gain. Do not overclaim a large severe-Non-IID
advantage from this single alpha=0.3 run. Continue with alpha=0.1 and alpha=1.0.
```

Next required validation:

```text
1. Run RAHFL seed=2 under alpha=0.5 for the missing matched control.
2. Generate or verify genuinely distinct alpha=0.5 seed partitions if the paper
   needs cross-partition multi-seed claims.
3. Run alpha=0.1 to test stronger label skew.
4. Run alpha=1.0 to ensure normal/non-extreme Non-IID does not collapse.
5. Only redesign communication after these validations. Do not replace AsymHFL
   immediately, because SARA + AsymHFL is currently the strongest evidence.
```

Alpha validation preparation:

```text
New configs:
  configs/kaggle_t4_rahfl_alpha01.yaml
  configs/kaggle_t4_sara_rahfl_alpha01.yaml
  configs/kaggle_t4_sara_rahfl_alpha03.yaml
  configs/kaggle_t4_sara_rahfl_alpha10.yaml

New Kaggle launcher:
  scripts/run_kaggle_sara_vs_rahfl_alpha01.sh
  scripts/run_kaggle_sara_alpha0103.sh
  scripts/run_kaggle_sara_alpha0310.sh

New partition tools:
  scripts/build_partition_pack.py
  scripts/import_partition_pack.py

Local generated pack:
  local_runs/sara_partitions_alpha01_alpha03
  local_runs/sara_partitions_alpha01_alpha03.tar.gz
  local_runs/sara_partitions_alpha03_alpha10
  local_runs/sara_partitions_alpha03_alpha10.tar.gz

Suggested Kaggle dataset name:
  sara-partitions-alpha01-alpha03
  sara-partitions-alpha03-alpha10
```

The partition pack only contains fixed `.npz` partition files and audit metadata,
not CIFAR image data. It should be mounted together with the existing
`fedprime-data` dataset. This avoids re-uploading the large CIFAR-10-C/CIFAR-100
prepared data while keeping alpha=0.3 and alpha=1.0 partitions reproducible.

For the alpha=0.1 paired comparison, use:

```text
configs/kaggle_t4_rahfl_alpha01.yaml
configs/kaggle_t4_sara_rahfl_alpha01.yaml
scripts/run_kaggle_sara_vs_rahfl_alpha01.sh
```

Mount:

```text
/kaggle/input/fedprime-data
/kaggle/input/sara-partitions-alpha01-alpha03
```

If `PARTITION_SOURCE` is empty, the alpha=0.1 paired launcher will generate the
missing alpha=0.1 partition on the fly. It also performs a partition-only audit
for `configs/kaggle_t4_rahfl_alpha03.yaml`, so the final result archive includes
both:

```text
outputs/partitions/cifar10c_alpha01_seed0_clients4_samples10000.npz
outputs/partitions/cifar10c_alpha03_seed0_clients4_samples10000.npz
```

This lets the user download one archive and later reuse the alpha=0.3 partition
without another Kaggle data-generation pass.

RAHFL multi-seed control preparation:

```text
New matching RAHFL seed configs:
  configs/kaggle_t4_rahfl_seed1.yaml
  configs/kaggle_t4_rahfl_seed2.yaml

New Kaggle launcher:
  scripts/run_kaggle_rahfl_seed12.sh
```

These configs are copied from the original unified-runner RAHFL baseline and
only change `seed`, `experiment_name`, and fixed partition path. They do not add
SARA-specific stability settings, so they remain an original RAHFL control.

Use the remaining RAHFL seed=2 control when preparing formal mean/std reporting:

```text
SARA seed=1 vs RAHFL seed=1: completed and positive
SARA seed=2 vs RAHFL seed=2: SARA completed, RAHFL seed=2 pending
```

Current alpha=0.5 validation status - 2026-07-05:

```text
Done:
  SARA + AsymHFL seed=1
  SARA + AsymHFL seed=2
  RAHFL seed=1

Pending:
  RAHFL seed=2
  alpha=0.3/0.1/1.0 SARA checks
  partition-generation audit because archived alpha=0.5 seed partition files
  are identical despite different seed names.
```

## CARA-L Design

CARA-L is the paper-facing name for the previously implemented NIR-DCL local
module. It modifies only the local DCL branch:

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

## CARA-C Design

CARA-C is the FedCARA communication module. It replaces RAHFL AsymHFL's
overall-accuracy teacher routing with class-wise reliable teaching.

Original AsymHFL:

```text
If overall_acc(student) <= overall_acc(teacher),
the student learns the teacher's full public softmax distribution.
```

CARA-C:

```text
For receiver i, teacher j, class c:
  weight_{i,j,c} = reliability_{j,c} * need_{i,c}

where:
  reliability_{j,c} = per-class acc of teacher j on class c
  need_{i,c}        = 1 - per-class acc of receiver i on class c
```

The first implementation also uses:

```text
better_only: true
only teach class c if teacher_acc_{j,c} > student_acc_{i,c} + margin
```

The public-logit KL becomes:

```text
weighted_KL = sum_c weight_{i,j,c} * p_teacher,c * log(p_teacher,c / p_student,c)
```

Key files/configs:

```text
fedprime/methods/rahfl_asymhfl.py
configs/debug_fedcara_cifar10c.yaml
configs/kaggle_t4_fedcara.yaml
```

## CARA-L / NIR-DCL Results - 2026-07-01

Two Kaggle runs finished:

```text
outputs/nir_dcl_local_only_results.tar.gz
outputs/nir_dcl_rahfl_results.tar.gz
```

Results:

```text
NIR-DCL local-only:
  final avg_acc   = 53.30
  final worst_acc = 36.01
  best avg_acc    = 54.74 at round 37
  best worst_acc  = 37.37 at round 26

NIR-DCL + AsymHFL:
  final avg_acc   = 57.36
  final worst_acc = 46.23
  best avg_acc    = 57.89 at round 34
  best worst_acc  = 46.33 at round 34
```

Comparison:

```text
RAHFL baseline final:        avg_acc=56.41, worst_acc=44.72
AugMix+DCL local-only final: avg_acc=56.11, worst_acc=44.23

NIR-DCL local-only gap vs AugMix+DCL local-only:
  avg_acc=-2.81
  worst_acc=-8.22

NIR-DCL + AsymHFL gap vs RAHFL:
  avg_acc=+0.95
  worst_acc=+1.51
```

Interpretation:

```text
NIR-DCL alone hurts local-only performance, especially worst-client accuracy.
However, NIR-DCL combined with AsymHFL exceeds the RAHFL baseline on both
average accuracy and worst-client accuracy under the current alpha=0.5 setting.

This suggests NIR-DCL may improve the quality/compatibility of public-logit
communication even if it is too restrictive as a purely local objective.
The next research story should focus on synergy:
  RAHFL local DCL is strong by itself;
  CARA-L regularizes local representations so AsymHFL communication becomes
  more beneficial under Non-IID label skew.
```

## Next FedCARA Experiment

Run:

```text
configs/kaggle_t4_fedcara.yaml
```

Compare against:

```text
RAHFL baseline:        56.41 / 44.72
CARA-L + AsymHFL:      57.36 / 46.23
```

Goal:

```text
FedCARA should ideally match or exceed CARA-L + AsymHFL.
If it beats 57.36 / 46.23, the communication innovation is immediately useful.
If it stays above RAHFL but below CARA-L + AsymHFL, CARA-C still has a valid
class-aware communication story but needs tuning.
```

## FedCARA v1 Result - 2026-07-01

Result archive:

```text
outputs/fedcara_results.tar.gz
```

FedCARA v1:

```text
config: configs/kaggle_t4_fedcara.yaml
method_name: fedcara
local: CARA-L
communication: CARA-C class-weighted public-logit KD
```

Final and best metrics:

```text
FedCARA:
  final avg_acc   = 55.88
  final worst_acc = 45.93
  best avg_acc    = 56.86 at round 34
  best worst_acc  = 45.93 at round 39

RAHFL baseline:
  final avg_acc   = 56.41
  final worst_acc = 44.72

CARA-L + AsymHFL:
  final avg_acc   = 57.36
  final worst_acc = 46.23
```

Interpretation:

```text
FedCARA v1 does not beat RAHFL on final average accuracy:
  avg_acc gap vs RAHFL = -0.53

But it does beat RAHFL on final worst-client accuracy:
  worst_acc gap vs RAHFL = +1.21

It is also below CARA-L + original AsymHFL:
  avg_acc gap = -1.48
  worst_acc gap = -0.30
```

Current judgment:

```text
CARA-C v1 is not the final communication module yet.
It appears to bias learning toward weaker clients/classes, improving worst_acc
but sacrificing average accuracy. The class-aware communication direction is not
dead, but pure replacement of AsymHFL with hard class weighting is too conservative.

Best next version should be hybrid:
  keep part of original AsymHFL full-softmax KD
  add CARA-C class-aware weighted KD as an auxiliary or residual term
instead of fully replacing AsymHFL.
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
