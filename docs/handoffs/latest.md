# FedPRIME-D2C Session Handoff

Updated: 2026-08-08

## Current Objective

Study heterogeneous federated learning under simultaneous model heterogeneity,
label-skew Non-IID, and corruption-label entanglement. The current formal
benchmark is four-client CLE-HFL v2 (`alpha=0.5`, `gamma=0.9`, seed 0), with 11
seen and 4 unseen concrete corruption operators. Operator metadata is available
to evaluation only.

The immediate objective is paper-evidence completion. Communication
decoupling, A0--A6 ablations, a 2x3 local/communication factorial, five
external baseline adaptations, two cross-scenario datasets, a 2x3 CLE
strength grid, a CIFAR-100-private second dataset, and PEW
calibration/sensitivity/efficiency instrumentation were prepared on
2026-08-06. These are implementation assets awaiting formal OpenI runs, not
new research results. Run them in the order documented at:

```text
docs/experiments/current/CLE_HFL_PAPER_EVIDENCE_OPENI_RUN_ZH.md
```

CDep-v2 was implemented after all three CDep-v1 lambda settings failed to beat
PEW+BER. Its matched shared-PEW paired experiment is now complete. All four
pre-registered last-five gates failed, so CDep-v1/v2 are frozen and the final
local method is `calibrated PEW + BER`.

The immediate next paid run is now a strict PEW operator leave-one-out audit.
It preserves the original PEW as a separate arm and sequentially runs RAHFL,
standard PEW+BER, and Strict-LOO PEW+BER for 12 rounds each. Strict LOO removes
`impulse_noise`, `zoom_blur`, `fog`, and `pixelate` from both public PEW train
and validation generation while leaving private data, AugMix, BER, and
AsymHFL-val unchanged:

```text
dataset: openi_cle_hfl_v2_alpha05_gamma09
entry:   scripts/openi_cle_pew_loo_entry.py
args:    none
guide:   docs/experiments/current/CLE_PEW_LOO_OPENI_RUN_ZH.md
output:  cle_pew_loo_12round_seed0_outputs.tar.gz
```

Its primary frozen gate is Strict-LOO PEW+BER minus same-task RAHFL last-five:
`Avg >= +1.5`, `Worst >= +1.0`, `WCCA >= 0`, `CFG <= -1.0`. Do not run the
older CDep entry and do not continue CDep tuning.

Before the first paper-evidence run, the refactored AsymHFL strategy was
protected by a legacy-vs-new numerical golden regression, cross-scenario
configs were corrected to freeze training seed 0, and FedProto was replaced
with native class-wise aggregation in the models' shared 1024-dimensional
embedding space.

The untracked CIFAR-100 screening archive is:

```text
local_runs/cle_hfl_v2_second_dataset/
cle_hfl_v2_prepared_cifar100_alpha05_gamma09_seed0_split0.tar.gz
SHA256 AF554AC3B9B46D38571445DDE84647965341444DAD175A1E4191851B8DD01EB4
```

## Latest Formal Result

The matched CDep-v2 shared-PEW experiment completed on 2026-08-08. Both arms
contained rounds 0--11, their resolved configs differed only in experiment
name and CDep-v2, and all four PEW annotation files were byte-identical.
Independently recomputed candidate-minus-control last-five deltas were:

```text
Avg -0.1933, Worst -0.2280, WCCA -0.6000, CFG +0.4450
```

All four frozen gates failed (0/4). CDep-v2 was active (last-five loss 0.04171,
41.8807 valid groups, buffer size 2730.86), so this is not an empty-module
failure. Verdict: `NO-GO`. Freeze the local method as calibrated PEW+BER and
do not revive CDep-v1/v2 by further structural or hyperparameter tuning.

Report:

```text
deliverables/cle_cdep_v2_paired_20260808/RESULT_SUMMARY_ZH.md
```

The single-arm CDep-v2 12-round screen completed on 2026-08-08. The run was
complete and the mechanism was active. Mechanical last-five comparison against
historical PEW+BER A1 was:

```text
Avg +0.3607, Worst +0.2000, WCCA -0.2500, CFG -0.7900
```

Three of four frozen gates passed; WCCA non-inferiority failed by 0.25. More
importantly, the CDep-v2 run retrained a different PEW: private group accuracy
was 68.075% versus 62.21% for A1, threshold was 0.22 versus 0.0, and all four
PEW annotation hashes differed. The historical comparison is therefore not a
matched causal CDep comparison. Verdict: `INCONCLUSIVE_FOR_ATTRIBUTION`, not a
validated PASS or a definitive CDep rejection.

The subsequently completed paired experiment used one PEW checkpoint and
byte-identical annotations:

```text
control   = calibrated PEW + BER, CDep disabled
candidate = the same calibrated PEW + BER + CDep-v2
```

This historical single-arm comparison must not override the paired NO-GO.
Report:

```text
deliverables/cle_cdep_v2_20260808/RESULT_SUMMARY_ZH.md
```

The focused CDep lambda sensitivity completed on 2026-08-07. All three arms
ran rounds 0--11 with identical PEW annotations and configs differing only in
experiment name and CDep lambda. Last-five results were:

```text
method            Avg       Worst     WCCA      CFG
BER-only A1       34.6320   29.4280   7.2500   24.6400
CDep lambda .01   34.1847   29.1053   6.0000   24.4350
CDep lambda .05   34.0230   28.9467   5.9000   24.1200
CDep lambda .10   33.9827   29.0373   5.9000   25.2500
```

The dependence proxy decreased as lambda increased, but no CDep setting beat
BER-only on Avg, Worst, or WCCA. Current CDep is not a validated additive core
module. Do not continue lambda-only tuning. Report:

```text
deliverables/cle_sensitivity_20260807/RESULT_SUMMARY_ZH.md
```

The A0--A6 12-round local ablation completed on 2026-08-07. All arms were
complete and config-matched. The main attribution is:

```text
BER-only minus RAHFL, last-five:
Avg +4.5467, Worst +4.3853, WCCA +6.4000, CFG -5.8000

CDep-only minus RAHFL, last-five:
Avg +0.3217, Worst -0.2720, WCCA +0.3000, CFG +0.3350
```

BER is the dominant positive component. Calibrated PEW outperformed a fixed
0.55 threshold, shuffled PEW badly degraded CFG, and oracle family was best.
CDep alone was neutral/slightly negative; in the full method it improved
final-round fairness relative to BER-only but reduced last-five Avg/Worst/WCCA.
Run the preplanned CDep lambda sensitivity before making a final method claim.
Report:

```text
deliverables/cle_local_ablation_20260807/RESULT_SUMMARY_ZH.md
```

The 12-round external-baseline screen completed on 2026-08-07 for Local-only,
FedMD, RHFL, FedProto, AugHFL, RAHFL, and the candidate. All seven arms have
complete rounds 0--11, matched strict configs, and independently reproduced
analysis. Last-five candidate-minus-RAHFL was:

```text
Avg +3.9377, Worst +3.9040, WCCA +5.0500, CFG -6.3200
```

This exactly reproduces the earlier formal seed-0 A/B. Candidate also led on
both seen and unseen operators. The external methods did not beat RAHFL in
this fixed CLE screen. This is not yet a complete SOTA claim: it is one
scenario/training seed and FedDF, KT-pFL, and FCCL remain absent. Report:

```text
deliverables/cle_external_baselines_20260807/RESULT_SUMMARY_ZH.md
```

OpenI completed the strict 12-round A/B for matched training seeds 0/1/2 on
2026-08-04:

```text
control:
  AugMix + JSD + DCL + strict AsymHFL-val

candidate:
  AugMix + JSD + DCL
  + calibrated PEW
  + BER + CDep
  + the same strict AsymHFL-val
```

All three returned archives were validated and independently reanalyzed
locally. Every arm contains rounds 0-11 with no missing core metrics, and all
recomputed comparisons exactly match the archived comparisons. The persisted
fit/audit partition hash is identical across seeds 0/1/2.

Candidate-minus-control:

```text
seed   Avg       Worst     WCCA      CFG
0      +3.9377   +3.9040   +5.0500   -6.3200
1      +4.7977   +3.8893   +4.3500   -8.3000
2      +5.0287   +4.8573   +7.2500   -5.5250
mean   +4.5880   +4.2169   +5.5500   -6.7150
```

All three seeds passed the original full gate (3/3), and all nine
pre-registered multi-seed gates passed. Verdict: `GO` for training-seed
stability on fixed CLE `seed0_split0`. This does not yet establish
cross-scenario generalization or a 40-round final-paper result.

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
arguments: --mode=both --train_seed=0/1/2
seed0 archive: strict_pew_asymhfl_val_probe_outputs.tar.gz
seed1 archive: strict_pew_asymhfl_val_trainseed1_probe_outputs.tar.gz
seed2 archive: strict_pew_asymhfl_val_trainseed2_probe_outputs.tar.gz
```

Formal configs:

```text
configs/openi_v100_rahfl_val_cle_v2_probe.yaml
configs/openi_v100_fedease_pew_asymhfl_val_cle_v2_probe.yaml
configs/openi_v100_rahfl_val_cle_v2_trainseed1_probe.yaml
configs/openi_v100_fedease_pew_asymhfl_val_cle_v2_trainseed1_probe.yaml
configs/openi_v100_rahfl_val_cle_v2_trainseed2_probe.yaml
configs/openi_v100_fedease_pew_asymhfl_val_cle_v2_trainseed2_probe.yaml
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
pre-registered three-seed aggregate analyzer and gates
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
all six formal arms had complete rounds 0-11 and no missing core metrics
all three archived comparisons exactly matched independent recomputation
all three persisted partition files were byte-identical
```

Smoke accuracy is not a research result.

## Decision And Next Step

Validated result locations:

```text
outputs/strict_pew_asymhfl_val_probe_outputs.tar.gz
outputs/strict_pew_asymhfl_val_probe_20260804/
outputs/strict_pew_asymhfl_val_trainseed1_probe_outputs.tar.gz
outputs/strict_pew_asymhfl_val_trainseed1_20260804/
outputs/strict_pew_asymhfl_val_trainseed2_probe_outputs.tar.gz
outputs/strict_pew_asymhfl_val_trainseed2_20260804/
outputs/strict_pew_asymhfl_val_multiseed_comparison.json
```

Candidate-minus-control must pass all last-five gates:

```text
Avg   >= +1.5
Worst >= +1.0
WCCA  >=  0.0
CFG   <= -1.0
```

The 40-round training-seed 0 durability task completed and was independently
reanalyzed. Both arms contain exact rounds 0-39 with no missing core metrics;
the returned configs match the committed configs, the fixed partition hash is
unchanged, and the first 12 rounds exactly reproduce the prior formal seed-0
run. Candidate-minus-control last-ten was:

```text
Avg +4.9292, Worst +3.2987, WCCA +9.8750, CFG -5.4700
verdict: GO (8/8 gates)
```

The user explicitly requested matched 40-round repeats for training seeds 1/2.
Keep the CLE scenario, persisted fit/audit split, and all PEW/BER/CDep settings
fixed. A distinct later question is generalization across CLE scenario seeds;
do not mix it into this durability attribution.

Prepared 40-round entry points:

```text
entry seed1: scripts/openi_strict_pew_asymhfl_40round_entry.py --mode=both --train_seed=1
entry seed2: scripts/openi_strict_pew_asymhfl_40round_entry.py --mode=both --train_seed=2
overnight: scripts/openi_strict_pew_asymhfl_40round_entry.py --mode=both --train_seed=all
analyzer: scripts/analyze_strict_pew_asymhfl_40round.py
guide: docs/experiments/current/STRICT_PEW_ASYMHFL_VAL_40ROUND_OPENI_RUN_ZH.md
expected seed1 archive: strict_pew_asymhfl_val_40round_trainseed1_outputs.tar.gz
expected seed2 archive: strict_pew_asymhfl_val_40round_trainseed2_outputs.tar.gz
```

The overnight mode runs pending seeds `[1, 2]` sequentially and uploads seed1
before starting seed2. Based on the completed seed0 task, expected total runtime
is roughly 4.5 hours; the OpenI job time limit must allow adequate margin.

## Research Memory In One Screen

Validated positive historical signal:

```text
calibrated PEW + BER+CDep passed strict CLE-HFL v2 training seeds 0/1/2
SARA + original AsymHFL reached 57.83/46.59 on the older alpha=0.5 setting
```

Important caveat: 40-round durability is currently established for training
seed 0 only and remains fixed to one CLE scenario; it is not yet a final-paper
cross-seed or cross-scenario result.

Frozen negative routes include D2C, Oracle D2C, FedPRIME-PAIR, PRAC-HFL,
FedCARA v1 communication, FedCLEAR/PCCD, EBST/EBST-v2, FedFalsify v0.2/v0.3,
FedCIS-v0, and the handcrafted continuous nuisance witness. Consult the index
before reopening any of them.

## Repository State

Latest pushed head before the current seed1/2 preparation:

```text
1eb5ba6 记录三种子结果并准备40轮耐久性实验
branch: main
```

The seed-0 40-round result record and seed1/2 preparation are the intended scope
of the next commit. Do not revert or stage unrelated dirty files. Large
outputs, datasets, checkpoints, and `local_test_outputs/` must remain untracked.

Documentation was reorganized on 2026-08-04. The repository root now keeps
only `README.md` and `AGENTS.md`; use `docs/README_ZH.md` as the document map.
