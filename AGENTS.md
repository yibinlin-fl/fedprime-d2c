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
PRAC-HFL = RAHFL local robust training + receiver-adaptive safe communication
```

Current main implementation:

```text
fedprime/methods/prac_hfl.py
fedprime/methods/local_rahfl.py
configs/kaggle_t4_prac_hfl.yaml
configs/debug_prac_hfl_cifar10c.yaml
scripts/run_kaggle_prac.sh
```

D2C and FedPRIME-PAIR are historical diagnostic routes, not the current main method.

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

Historical negative/diagnostic routes:

```text
PRIME + LogitAvg final avg_acc about 52.10
FedPRIME-D2C final avg_acc about 52.31
Oracle D2C final avg_acc about 51.74
FedPRIME-PAIR final avg_acc about 50.15, best avg about 51.10
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
Improve communication beyond AsymHFL through receiver-side validation and safe adaptive knowledge transfer.
Beat the fair RAHFL baseline under Non-IID + corruption + model heterogeneity.
```
