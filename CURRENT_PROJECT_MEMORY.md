# FedPRIME-D2C / PRAC-HFL Current Project Memory

Updated: 2026-06-29

## Current Goal

Build a paper-worthy heterogeneous FL method for:

```text
model heterogeneity + data heterogeneity / Non-IID + data corruption robustness
```

The current baseline to beat is the unified-runner RAHFL baseline.

## Current Main Method

The current mainline is:

```text
PRAC-HFL = RAHFL local robust training + receiver-adaptive safe communication
```

The project is no longer centered on D2C or FedPRIME-PAIR. Those are now historical diagnostic experiments.

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

1. Rerun safe PRAC-HFL public4 using `configs/kaggle_t4_prac_hfl.yaml`.
2. Compare against RAHFL 56.41 / 44.72.
3. If safe PRAC-HFL is stable and close, run:

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
