# FedPRIME-D2C Session Handoff

Updated: 2026-08-04

## Current Objective

Study heterogeneous federated learning under simultaneous model heterogeneity,
label-skew Non-IID, and corruption-label entanglement. The current formal
benchmark is four-client CLE-HFL v2 (`alpha=0.5`, `gamma=0.9`, seed 0), with 11
seen and 4 unseen concrete corruption operators. Operator metadata is available
to evaluation only.

## Experiment Running Now

OpenI is running one strict 12-round A/B attribution probe:

```text
control:
  AugMix + JSD + DCL + strict AsymHFL-val

candidate:
  AugMix + JSD + DCL
  + calibrated PEW
  + BER + CDep
  + the same strict AsymHFL-val
```

Fairness contract:

```text
same four heterogeneous models
same seed and matched model initialization
same persisted class-stratified 85/15 fit/audit split
fit-only local gradients
audit-only AsymHFL teacher ordering
same CIFAR-100 public data and 4 public batches/round
final CLE test labels used only for reporting
```

OpenI assets:

```text
dataset: cle_hfl_v2_prepared_alpha05_gamma09_seed0_split0.tar.gz
entry: scripts/openi_strict_pew_asymhfl_entry.py
argument: --mode=both
expected archive: strict_pew_asymhfl_val_probe_outputs.tar.gz
```

Formal configs:

```text
configs/openi_v100_rahfl_val_cle_v2_probe.yaml
configs/openi_v100_fedease_pew_asymhfl_val_cle_v2_probe.yaml
```

## Implementation Status

Completed on 2026-08-04:

```text
strict persisted fit/audit data path
fit-only FedEASE annotations and class/environment counts
audit-only AsymHFL routing
CLE-HFL v2 FedEASE support
PEW post-training RNG reset for matched initialization
operator-to-family mapping for diagnostic reporting only
automatic final/last-five comparison and frozen GO/NO-GO gates
OpenI packaging and c2net upload
checkpoints.save_final=false enforcement
```

Verification:

```text
46 focused tests passed
strict control one-round RTX 3050 smoke passed
candidate one-round RTX 3050 smoke passed
both arms had identical round-0 audit routing accuracies
both arms had nonzero and matched AsymHFL col_loss
candidate BER and CDep were nonzero
```

Smoke accuracy is not a research result.

## Decision After Result Arrives

Place the returned archive under `outputs/`, then analyze the generated:

```text
outputs/strict_pew_asymhfl_val_comparison.json
```

Candidate-minus-control must pass all last-five gates:

```text
Avg   >= +1.5
Worst >= +1.0
WCCA  >=  0.0
CFG   <= -1.0
```

If all pass, repeat seeds before any 40-round claim. If any fail, archive this
combination and do not rescue it with blind lambda/threshold sweeps.

## Research Memory In One Screen

Validated positive historical signal:

```text
calibrated PEW + BER+CDep local learning improved Avg/Worst/CFG on CLE-HFL v1
SARA + original AsymHFL reached 57.83/46.59 on the older alpha=0.5 setting
```

Important caveat: neither is yet a clean final paper method under strict
CLE-HFL v2 evaluation.

Frozen negative routes include D2C, Oracle D2C, FedPRIME-PAIR, PRAC-HFL,
FedCARA v1 communication, FedCLEAR/PCCD, EBST/EBST-v2, FedFalsify v0.2/v0.3,
FedCIS-v0, and the handcrafted continuous nuisance witness. Consult the index
before reopening any of them.

## Repository State

Latest committed head observed before this handoff:

```text
05236e6 实现CLE-HFL v2算子协议与FedFalsify三方实验入口
branch: main
```

The strict A/B implementation and recent audit documents are currently local
worktree changes unless committed after this handoff. Do not revert unrelated
dirty files. Large outputs, datasets, checkpoints, and `local_test_outputs/`
must remain untracked.
