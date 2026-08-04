# FedPRIME-D2C Session Handoff

Updated: 2026-08-04

## Current Objective

Study heterogeneous federated learning under simultaneous model heterogeneity,
label-skew Non-IID, and corruption-label entanglement. The current formal
benchmark is four-client CLE-HFL v2 (`alpha=0.5`, `gamma=0.9`, seed 0), with 11
seen and 4 unseen concrete corruption operators. Operator metadata is available
to evaluation only.

## Latest Formal Result

OpenI completed one strict 12-round A/B attribution probe on 2026-08-04:

```text
control:
  AugMix + JSD + DCL + strict AsymHFL-val

candidate:
  AugMix + JSD + DCL
  + calibrated PEW
  + BER + CDep
  + the same strict AsymHFL-val
```

The returned archive was validated and independently reanalyzed locally. Both
arms contain rounds 0-11 with no missing core metrics, and the recomputed
comparison exactly matches the archived comparison.

Candidate-minus-control:

```text
scope       Avg       Worst     WCCA      CFG
final       +5.1267   +2.9533   +8.7500   -7.7250
last-five   +3.9377   +3.9040   +5.0500   -6.3200
```

All four frozen last-five gates passed. Verdict: `GO` for the seed-0
attribution probe. This does not yet justify a multi-seed or final-paper claim.

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

## Decision And Next Step

Validated result locations:

```text
outputs/strict_pew_asymhfl_val_probe_outputs.tar.gz
outputs/strict_pew_asymhfl_val_probe_20260804/
```

Candidate-minus-control must pass all last-five gates:

```text
Avg   >= +1.5
Worst >= +1.0
WCCA  >=  0.0
CFG   <= -1.0
```

The gates passed for seed 0. The next scientific step is matched 12-round seed
repetition. Do not start 40 rounds or present a final method claim until the
positive effect survives those repeats. Do not start new paid runs unless the
user explicitly requests them.

Prepared repeat infrastructure:

```text
seed 1: scripts/openi_strict_pew_asymhfl_entry.py --mode=both --train_seed=1
seed 2: scripts/openi_strict_pew_asymhfl_entry.py --mode=both --train_seed=2
guide: docs/experiments/current/STRICT_PEW_ASYMHFL_VAL_MULTISEED_OPENI_RUN_ZH.md
aggregate: scripts/analyze_strict_pew_asymhfl_multiseed.py
```

The repeat keeps the CLE scenario and persisted fit/audit split fixed at
`seed0_split0`; only the top-level training seed changes. The aggregate gates
are pre-registered in the guide and analyzer. Verification completed with 9
focused tests, CLI help checks, a synthetic three-seed CLI dry-run, and
dependency checks for all four new configs. The local private CLE path is
absent until the prepared archive is imported, as expected. No paid repeat
task has been started.

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

Documentation was reorganized on 2026-08-04. The repository root now keeps
only `README.md` and `AGENTS.md`; use `docs/README_ZH.md` as the document map.
