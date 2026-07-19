# FedPRIME-D2C Agent Entry

This file is the first memory entry for future Codex sessions.

## Safety Rules

Do not batch-delete files or directories.

Never use:

```text
del /s
rd /s
rmdir /s
Remove-Item -Recurse
rm -rf
```

If deletion is needed, delete only one explicit file path at a time. If batch deletion is needed, stop and ask the user to delete manually.

## Read Order

When resuming this project, read these files first:

```text
CURRENT_PROJECT_MEMORY.md
PROJECT_STATE.md
TODO_NEXT.md
ARCHITECTURE.md
EXPERIMENT_GUIDE_ZH.md
```

Use `CURRENT_PROJECT_MEMORY.md` as the cleanest current-state summary. Older files may contain historical notes.

## Current Mainline Override - 2026-07-19

The newest candidate is:

```text
CLE-HFL + FedEASE v2.1
```

Latest corrective probe implementation - 2026-07-20:

```text
communication: ebst_v2
pair-qualified sources + recipient LOO teacher + source-agreement gate
+ class-wise SCP + per-class communication norm cap
```

Run only:

```text
scripts/openi_fedease_entry.py --mode=ebst_v2_probe
```

Config: `configs/openi_v100_fedease_ebst_v2_probe.yaml`.
Guide: `FEDEASE_EBST_V2_OPENI_RUN_ZH.md`.
The old dataset `openi_cle_rahfl_diagnostic` is sufficient. Do not run full mode.

Read first:

```text
FEDEASE_V2_1_FRAMEWORK_AND_IMPLEMENTATION_ZH.md
```

The complete switchable candidate is implemented:

```text
Oracle or learned PEW environment
+ BER + fixed-random-projection CDep
+ preserved AugMix/JSD/DCL
+ EBST + stability gate + classifier-head SCP
+ clean/same/random/swapped/unseen evaluation
```

Core files:

```text
fedprime/data/fedease.py
fedprime/methods/balanced_environment_risk.py
fedprime/methods/conditional_dependence.py
fedprime/methods/environment_witness.py
fedprime/methods/environment_structural_transfer.py
fedprime/methods/safe_communication_projection.py
fedprime/methods/local_fedease.py
fedprime/methods/fedease.py
fedprime/engine/cle_metrics.py
```

Formal OpenI entry: `scripts/openi_fedease_entry.py`.
Read `FEDEASE_OPENI_RUN_GUIDE_ZH.md` and run `--mode=oracle_probe` first.
The complete code is not yet a positive research result. Do not run `--mode=full`:
Oracle BER+CDep passed, but the Oracle EBST communication probe failed.

EBST probe result - 2026-07-20:

```text
BER+CDep local final Avg/Worst/WCCA/CFG = 41.6206/35.5175/14.000/6.155
BER+CDep+EBST+SCP final                 = 38.7038/34.7225/15.325/6.415
delta                                  = -2.9169/-0.7950/+1.325/+0.260
```

Current EBST is an archived negative communication result. It was active, but
caused severe class-specific regression on client 2. Do not tune its lambda or
run full mode blindly. Redesign pairwise source eligibility and recipient/class
specific safety before another communication experiment.

Oracle probe result - 2026-07-20:

```text
control final Avg/Worst/WCCA/CFG = 37.5813/30.1100/13.70/10.855
BER+CDep final                  = 41.6206/35.5175/14.00/6.155
delta                           = +4.0394/+5.4075/+0.30/-4.70
```

The local mechanism gate passed. The subsequent `--mode=ebst_probe` failed as
recorded above. Do not run full mode.

## Current Mainline Override - 2026-07-11

The newest research mainline supersedes the older FedCLEAR v0.1 and corruption-skew notes below:

```text
CLE-HFL + FedCLEAR-PCCD
FedCLEAR-PCCD = fixed AugMix/JSD/DCL local base + paired counterfactual consensus distillation
```

Read first:

```text
FEDCLEAR_LATEST_THEORY_FRAMEWORK_ZH.md
```

Implementation:

```text
fedprime/methods/pccd.py
fedprime/methods/fedclear_pccd.py
fedprime/methods/rahfl_asymhfl.py
scripts/prepare_cle_in_domain_public.py
scripts/openi_fedclear_pccd_entry.py
```

FedCLEAR v0.1 (`CCRE + IRD`) is a completed negative result:

```text
gamma=0.9 final: avg=45.41, worst=36.42, WCCA=17.80, CFG=11.42
matching RAHFL:   avg=46.72, worst=38.16, WCCA=19.32, CFG=10.91
```

Immediate next experiment is a matching 12-round A/B probe with the same
unlabeled in-domain CIFAR-10 public pool:

```text
RAHFL: configs/openi_v100_rahfl_cle_indomain_probe.yaml
PCCD:  configs/openi_v100_fedclear_pccd_probe.yaml
entry: scripts/openi_fedclear_pccd_entry.py --method rahfl|pccd|both
```

Do not run a 40-round PCCD experiment unless the probe passes all avg/worst/WCCA/CFG gates.

## Current Mainline

New immediate experiment direction:

```text
FedSARA-CS on corruption-skew protocol
```

Core idea:

```text
model heterogeneity + label-skew Non-IID + corruption-skew Non-IID
```

Key files:

```text
scripts/prepare_corruption_skew_data.py
scripts/import_fedsara_cs_data.py
scripts/run_openi_fedsara_cs.sh
configs/openi_v100_rahfl_cs_alpha05_rho07.yaml
configs/openi_v100_fedsara_cs_alpha05_rho07.yaml
FEDSARA_CS_SCENARIO_OPENI_GUIDE_ZH.md
```

Prepared dataset:

```text
local_runs/fedsara_cs_prepared/fedsara_cs_prepared_alpha05_rho07_seed0.tar.gz
```

Both RAHFL-CS and FedSARA-CS formal configs use:

```text
pretrain_epochs: 40
rounds: 40
```

Current active method:

```text
SARA + AsymHFL = AugMix/JSD + Skew-Aware Robust Alignment + RAHFL AsymHFL
```

Current main implementation:

```text
fedprime/methods/sara.py
fedprime/methods/local_rahfl.py
fedprime/methods/rahfl_asymhfl.py
configs/kaggle_t4_sara_rahfl.yaml
configs/kaggle_t4_sara_local_only.yaml
configs/debug_sara_local_only.yaml
```

Legacy/diagnostic runner implementation:

```text
fedprime/methods/prac_hfl.py
configs/kaggle_t4_nir_dcl_local_only.yaml
configs/debug_nir_dcl_local_only.yaml
scripts/run_kaggle_prac.sh
```

D2C, FedPRIME-PAIR, and current PRAC-HFL communication are historical/diagnostic
routes. PRAC-HFL runner is still reused because it has the best Kaggle heartbeat
logging and can disable communication with `warmup_rounds: 999`.

## Latest Important Commits

Latest SARA implementation commit:

```text
9df13a7
```

Previous PRAC-HFL implementation commit:

```text
fa108f7
```

Kaggle PRAC-HFL runs should verify:

```text
git log -1 --oneline
expected SARA implementation is present: 9df13a7 or later
```

## Current Experiment Facts

RAHFL unified-runner fair baseline:

```text
avg_acc=56.41
worst_acc=44.72
no independent 40-epoch pretraining
config: configs/kaggle_t4_rahfl.yaml
```

First PRAC-HFL run before safe fix:

```text
visible rounds: 001-028
round 028 PRAC avg_acc=53.86, worst_acc=39.52
best visible PRAC avg_acc=53.86
best visible PRAC worst_acc=42.15
round 029 became NaN and invalidated later results
```

Safe PRAC-HFL changes:

```text
warmup_rounds: 3
risk_lambda_aug: 0.0
risk_lambda_js: 0.0
virtual_lr: 0.005
head_max_grad_norm: 1.0
train.max_grad_norm: 5.0
train.skip_nonfinite: true
```

Safe PRAC-HFL public1 result:

```text
public_batches_per_round=1
final avg_acc=54.63
final worst_acc=41.88
best avg_acc=55.53
best worst_acc=43.43
```

Safe PRAC-HFL public4 fair result:

```text
public_batches_per_round=4
final avg_acc=52.96
final worst_acc=43.27
best avg_acc=52.96
best worst_acc=43.27
gap vs RAHFL final: avg_acc=-3.45, worst_acc=-1.45
```

Interpretation:

```text
PRAC communication has nonzero behavior, but public4 did not improve average accuracy.
It may help worst-client accuracy somewhat while introducing negative transfer.
The decisive next control is AugMix+DCL local-only.
```

Local-only control:

```text
configs/kaggle_t4_augmix_dcl_local_only.yaml
method_name: prac_hfl
warmup_rounds: 999
meaning: AugMix + CE + JSD + DCL local training only, no PRAC communication.
```

Local-only control result:

```text
final avg_acc=56.11
final worst_acc=44.23
best avg_acc=56.94 at round 38
best worst_acc=44.23 at round 39
all PRAC metrics are zero
```

Interpretation:

```text
AugMix + DCL local-only nearly matches RAHFL and beats both PRAC public1/public4
on final average accuracy. Current PRAC communication does not provide positive
average-accuracy gain over local robust training and should not remain the main
claim without redesign.
```

Historical negative/diagnostic routes:

```text
PRIME + LogitAvg final avg_acc about 52.10
FedPRIME-D2C final avg_acc about 52.31
Oracle D2C final avg_acc about 51.74
FedPRIME-PAIR final avg_acc about 50.15, best avg about 51.10
Safe PRAC-HFL public4 final avg_acc about 52.96
```

Current new route:

```text
CARA-L local-only:
  config: configs/kaggle_t4_nir_dcl_local_only.yaml
  debug:  configs/debug_nir_dcl_local_only.yaml
  communication: disabled by warmup_rounds=999
  goal: improve AugMix+DCL local-only under label-skew Non-IID before revisiting communication

CARA-L + AsymHFL:
  config: configs/kaggle_t4_nir_dcl_rahfl.yaml
  goal: optional stronger comparison after local-only signal is known

FedCARA:
  config: configs/kaggle_t4_fedcara.yaml
  debug:  configs/debug_fedcara_cifar10c.yaml
  local module: CARA-L
  communication module: CARA-C
```

Latest NIR-DCL result:

```text
NIR-DCL local-only final avg/worst = 53.30/36.01
NIR-DCL + AsymHFL final avg/worst = 57.36/46.23
RAHFL baseline final avg/worst = 56.41/44.72
```

Interpretation:

```text
NIR-DCL local-only is worse than AugMix+DCL local-only, but NIR-DCL + AsymHFL
beats RAHFL by +0.95 avg_acc and +1.51 worst_acc under alpha=0.5.
The promising claim is not "NIR-DCL alone is stronger"; it is "NIR-DCL improves
the compatibility of local robust features with AsymHFL communication under
Non-IID label skew."
```

Latest FedCARA v1 result:

```text
FedCARA final avg/worst = 55.88/45.93
FedCARA best avg/worst = 56.86/45.93
RAHFL baseline final avg/worst = 56.41/44.72
CARA-L + AsymHFL final avg/worst = 57.36/46.23
```

Interpretation:

```text
FedCARA v1 improves worst-client accuracy over RAHFL (+1.21) but loses average
accuracy (-0.53). It is below CARA-L + original AsymHFL. CARA-C v1 is therefore
not final; it should become a hybrid auxiliary communication term rather than a
full replacement for AsymHFL.
```

Latest SARA result - 2026-07-02:

```text
Result archive:
  outputs/sara_rahfl_results.tar.gz

SARA local-only:
  config: configs/kaggle_t4_sara_local_only.yaml
  method_name: prac_hfl
  communication disabled by warmup_rounds=999
  final avg/worst = 54.10 / 32.06
  best avg/worst  = 54.59 / 33.96

SARA + AsymHFL:
  config: configs/kaggle_t4_sara_rahfl.yaml
  method_name: rahfl
  cl_module: sara
  final avg/worst = 57.83 / 46.59
  best avg/worst  = 57.83 / 46.59

RAHFL baseline:
  config: configs/kaggle_t4_rahfl.yaml
  final avg/worst = 56.41 / 44.72
```

Interpretation:

```text
SARA local-only is weak, especially on worst-client accuracy. It should not be
claimed as a standalone local-training improvement.

SARA + AsymHFL is currently the best mainline result:
  gap vs RAHFL = +1.42 avg_acc and +1.87 worst_acc.

The working hypothesis is synergy:
  SARA changes local representation geometry/class balance in a way that makes
  public-logit AsymHFL communication more effective under label-skew Non-IID,
  even though SARA alone over-regularizes weak clients.
```

## Kaggle Run Memory

Kaggle notebooks are often executed as one-shot background runs. Therefore:

```text
1. Print heartbeat logs before long local/client loops.
2. Use unbuffered Python output where possible.
3. Avoid relying on interactive state after a run starts.
4. Verify repo head immediately after clone.
5. Import prepared data from /kaggle/input/fedprime-data.
6. Do not run slow data downloads if the mounted prepared dataset exists.
```

Known prepared-data layout:

```text
/kaggle/input/fedprime-data/cifar_10_c
/kaggle/input/fedprime-data/cifar_100
/kaggle/input/fedprime-data/outputs/partitions
```

Copy these into:

```text
RAHFL-master/Dataset/cifar_10_c
RAHFL-master/Dataset/cifar_100
outputs/partitions
```

## Git And Artifact Hygiene

The following are intentionally ignored and should not be pushed by default:

```text
outputs/
local_runs/
logs/
runs/
wandb/
*.pt
*.pth
*.ckpt
*.npy
*.npz
RAHFL-master/Dataset/cifar_10_c/
RAHFL-master/Dataset/cifar_10/
RAHFL-master/Dataset/cifar_100/
RAHFL-master/Dataset/cifar_100_c/
PRIME-augmentations-main/data/
```

Use `deliverables/` for manually prepared figures, tables, and documents for reports. Do not commit large experiment outputs unless the user explicitly requests it.

## Research Direction

The current practical direction is:

```text
Use RAHFL local robust learning as the strong base.
Use SARA to make robust local contrastive alignment skew-aware.
Keep AsymHFL for now because SARA + AsymHFL is the first result that beats RAHFL
on both final average accuracy and final worst-client accuracy.
Beat the fair RAHFL baseline under Non-IID + corruption + model heterogeneity.
```

Immediate next experiment:

```text
Do not replace communication yet.
First verify SARA + AsymHFL:
  1. rerun seeds 1 and 2 under alpha=0.5
  2. run alpha=0.3 and alpha=0.1 for stronger Non-IID
  3. run alpha=1.0 to check normal/non-extreme Non-IID
  4. rerun RAHFL for matching seeds/settings if SARA remains positive
```
