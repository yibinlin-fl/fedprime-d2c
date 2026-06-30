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

## Current Mainline

Current active method:

```text
NIR-DCL = RAHFL AugMix/JSD local robust training + Non-IID-aware Robust DCL
```

Current main implementation:

```text
fedprime/methods/nir_dcl.py
fedprime/methods/local_rahfl.py
fedprime/methods/prac_hfl.py
configs/kaggle_t4_nir_dcl_local_only.yaml
configs/debug_nir_dcl_local_only.yaml
scripts/run_kaggle_prac.sh
```

D2C, FedPRIME-PAIR, and current PRAC-HFL communication are historical/diagnostic
routes. PRAC-HFL runner is still reused because it has the best Kaggle heartbeat
logging and can disable communication with `warmup_rounds: 999`.

## Latest Important Commits

Latest pushed mainline commit:

```text
5e476ea
```

Previous PRAC-HFL implementation commit:

```text
fa108f7
```

Kaggle PRAC-HFL runs should verify:

```text
git log -1 --oneline
expected head starts with: 5e476ea
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
NIR-DCL local-only:
  config: configs/kaggle_t4_nir_dcl_local_only.yaml
  debug:  configs/debug_nir_dcl_local_only.yaml
  communication: disabled by warmup_rounds=999
  goal: improve AugMix+DCL local-only under label-skew Non-IID before revisiting communication

NIR-DCL + AsymHFL:
  config: configs/kaggle_t4_nir_dcl_rahfl.yaml
  goal: optional stronger comparison after local-only signal is known
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
First improve local DCL to be Non-IID-aware.
Beat the fair RAHFL baseline under Non-IID + corruption + model heterogeneity.
```

Immediate next experiment:

```text
Use the local-only result as the decision point.
Shift focus from current PRAC communication to Non-IID-aware robust DCL/local representation learning,
unless PRAC is redesigned with held-out routing and weaker/aggregated communication.
```
