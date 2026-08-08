# FedPRIME-D2C Project State

Last updated: 2026-08-09

## Strict PEW Operator-LOO Result - GO - 2026-08-09

Added an optional `method.fedease.pew.exclude_operators` protocol. The default
is empty, so all historical PEW configs and checkpoints retain their original
behavior. Strict checkpoints record their exclusions and fail closed if a
config attempts to reuse a checkpoint trained under another exclusion policy.

The new OpenI entry `scripts/openi_cle_pew_loo_entry.py` runs three matched
12-round arms: RAHFL, original PEW+BER, and Strict-LOO PEW+BER. Strict LOO
removes `impulse_noise`, `zoom_blur`, `fog`, and `pixelate` from public PEW
training and validation. It also audits zero occurrences in private fit,
records the final public operator pools, disables CDep, and applies the original
four candidate-vs-RAHFL gates. Focused verification: `29 passed`; entry dry-run
and all three environment/path checks passed.

The formal three-arm run completed. Strict-LOO minus RAHFL last-five was
`Avg +4.9027`, `Worst +6.2547`, `WCCA +4.6000`, `CFG -6.1100`; all four
pre-registered gates passed. All held-out private-fit counts were zero and the
four operators were absent from Strict PEW public train/validation pools.

Guide: `docs/experiments/archive/CLE_PEW_LOO_OPENI_RUN_ZH.md`.
Report: `deliverables/cle_pew_loo_20260809/RESULT_SUMMARY_ZH.md`.

## CDep-v2 Matched Paired Decision - NO-GO - 2026-08-08

The strict shared-PEW paired experiment completed for 12-round PEW+BER control
and 12-round PEW+BER+CDep-v2 candidate. Both arms contain rounds 0--11; all
four client PEW annotations are byte-identical; resolved configs differ only
in experiment name and CDep-v2 settings.

Independent last-five candidate-minus-control recomputation:

```text
Avg -0.1933, Worst -0.2280, WCCA -0.6000, CFG +0.4450
```

All four frozen gates failed. CDep-v2 was active, with last-five mean loss
0.04171, 41.8807 valid groups, and 2730.86 buffer samples. Final decision:
freeze CDep-v1/v2 and freeze the local method as calibrated PEW+BER.

Before another paid run, prepared paper-evidence configs must be updated to
use the frozen PEW+BER candidate. Completed guide:
`docs/experiments/archive/CLE_CDEP_V2_PAIRED_OPENI_RUN_ZH.md`.
Report: `deliverables/cle_cdep_v2_paired_20260808/RESULT_SUMMARY_ZH.md`.

## CDep-v2 Single-Arm Screen Implemented - 2026-08-07

The single-arm OpenI run completed on 2026-08-08. Its implementation and
diagnostics passed, but attribution is inconclusive because the run retrained a
different PEW from historical A1. A paired shared-PEW entry is now required;
the single-arm entry must not be treated as the final CDep decision.

The paired entry is implemented at
`scripts/openi_cle_cdep_v2_paired_entry.py`. It runs control then candidate
with one shared checkpoint, verifies byte-identical PEW annotations, computes
the unchanged frozen gates, and packages both arms. The new focused suite is
`23 passed`; dry-run config and environment checks passed.

- Added `BufferedConditionalMomentAlignment`, a bounded client-local
  class/environment feature memory with PEW-confidence weighting and support
  gates. Stored features are detached and never communicated.
- Added two warm-up rounds and a three-round activation ramp; the legacy CDep
  and PEW+BER paths remain available and unchanged by configuration.
- Added round diagnostics for active groups, memory size, ramp, and loss.
- Added `scripts/openi_cle_cdep_v2_entry.py`, a single-arm 12-round OpenI entry
  with automatic packaging and frozen comparison against existing matched
  PEW+BER A1.
- Focused unit, config, sensitivity, local-ablation, strict-AsymHFL, and
  communication golden tests: `24 passed`.
- Three-round local smoke passed. Memory grew `8 -> 20 -> 29`; ramp was
  `0, 0, 0.33`; round-2 CDep-v2 loss was nonzero. Smoke accuracy is not
  scientific evidence.

Historical single-arm entry: `scripts/openi_cle_cdep_v2_entry.py`.
Completed paired entry: `scripts/openi_cle_cdep_v2_paired_entry.py` with no arguments.
Guide: `docs/experiments/archive/CLE_CDEP_V2_PAIRED_OPENI_RUN_ZH.md`.

## Strict PEW + AsymHFL-val A/B Probe Ready - 2026-08-04

- Added a reusable strict fit/audit mode to the unified RAHFL runner.
- Local gradients are fit-only; AsymHFL routing is client-private audit-only;
  final-test labels are no longer used to select teachers.
- Added fit-only FedEASE annotated loaders and fit-only class/environment
  counts, while preserving old configs when the strict switch is disabled.
- Added matched 12-round control/candidate configs, automatic comparison,
  OpenI packaging/upload, heartbeat logs, and a Chinese run guide.
- Both one-round RTX 3050 smoke runs passed; 46 focused tests pass.
- Formal 12-round research result has not been run. No effectiveness claim may
  be made from the smoke accuracy.

Entry: `scripts/openi_strict_pew_asymhfl_entry.py --mode=both`.

## Continuous Witness Matched Audit Completed - NO-GO - 2026-08-03

- Added `fedprime/methods/continuous_nuisance.py` with a taxonomy-free
  continuous image descriptor, continuous balanced risk, conditional
  decision/witness covariance, and exact matched controls.
- Added `scripts/audit_continuous_nuisance.py` for matched full-model one-step
  updates and held-out local Avg/Worst/WCCA/CFG evaluation.
- Added six focused tests; the combined new FedCIS/continuous-witness suite is
  `14 passed`.
- The true witness uniquely improved CFG by `1.25`, but slightly harmed Worst
  and audit loss, and won only `33.33%` of client-class targets.
- Decision: do not run local-only or AsymHFL with this witness formulation.

Read: `docs/archive/methods/CONTINUOUS_WITNESS_OFFLINE_AUDIT_ZH.md`.

## FedCIS Audit A/B Completed - NO-GO - 2026-08-03

- Implemented taxonomy-free input-sensitivity extraction, deterministic
  multiscale DCT projection, PSD class statistics, guarded generalized-eigen
  subspace recovery, detached orthogonal margin-descent counterfactuals, and
  class-shuffled/equal-rank-random controls.
- Added eight focused unit tests; all pass.
- Ran the full offline Audit A/B on the local RTX 3050 with four heterogeneous
  RAHFL checkpoints and AugMix seeds 0/1/2.
- All ten classes had enough support, so the failure is not caused by class
  abstention alone.
- True cross-seed similarity was `0.1673`, nearly identical to shuffled
  `0.1669`; matched-class cross-client similarity `0.1269` was below
  mismatched-class `0.1318`.
- Only `30.30%` of 33 attack targets beat both controls, below the frozen 60%
  gate.
- Decision: stop FedCIS-v0 before Audit C and before any federated runner.

Artifacts: `local_test_outputs/fedcis_audit_20260803/`.

## FedCIS-v0 Candidate Frozen For Offline Audit - 2026-08-03

- Kept the formal problem as four-client CLE-HFL v2.
- Defined FedCIS as a replacement candidate for AsymHFL while freezing the
  AugMix/JSD/DCL local robust base.
- Replaced the original nonsymmetric cross-view statistic with PSD view-mean
  and view-difference second moments.
- Replaced the ambiguous client matrix square with an outer-product dispersion.
- Corrected the counterfactual direction from margin ascent to margin descent.
- Removed full second-order sensitivity regularization from v0; projected
  counterfactual perturbations must be detached.
- Restricted the default payload to fixed-shape class support masks and
  class-conditional second moments; exact class counts are not required.
- Recorded explicit limitations for missing classes, shared shortcuts, privacy,
  arbitrary unseen corruptions, and K=4 subspace recovery.
- Froze a three-stage offline audit: numerical feasibility, subspace
  identifiability, and matched one-step causal update.

Status:

```text
FedCIS offline audit: NO-GO after Audit A/B
FedCIS implementation: standalone audit only
FedCIS 12-round runner: BLOCKED
FedCIS 40-round experiment: BLOCKED
```

Specification: `docs/archive/methods/FEDCIS_FRAMEWORK_AND_OFFLINE_AUDIT_ZH.md`.

## Robust Frontier Offline Audit Completed - 2026-07-26

- Added a taxonomy-free class-pair robust-margin audit.
- Added per-sample z-score normalization so heterogeneous model logit scales
  are comparable.
- Reused stored CLE-HFL v2 data, final RAHFL checkpoints, and operator-level
  evaluation CSV; no training was performed.
- The frontier score predicts local seen/unseen vulnerability with Spearman
  `0.434/0.559`.
- Source frontier advantage predicts actual seen/unseen advantage with
  Spearman `0.319/0.548`.
- Direct positive-source routing precision is only `52.94%/52.94%`; the
  full-coverage route is rejected.
- Top-quartile all-view routes show a stable exploratory signal across three
  augmentation seeds: seen precision `77.78%-88.89%`, unseen
  `88.89%-100%`, with route-set Jaccard `0.80`.
- Current decision: no 40-round FedRIFT. Next validate only stable,
  high-confidence, abstaining transfer with a one-step fit/audit head audit.

Artifacts:
`deliverables/robust_frontier_audit_20260726/ROBUST_FRONTIER_AUDIT_ZH.md`.

### One-step matched control

- Tested the seven cross-seed-stable routes with head-only updates.
- Added an identical CE-only update control so normal local optimization is not
  misattributed to communication.
- Frontier loss slightly reduced target-class audit CE on all seven routes.
- It did not change seen accuracy, and unseen mean improved only `+0.0357`.
- Overall audit accuracy changed by `-0.0095`.
- Decision: the frontier can diagnose/select reliability but cannot itself
  serve as the communication knowledge payload.
- No 12/40-round frontier-transfer experiment is justified.

## CLE-HFL v2 Probe Completed - 2026-07-24

- All RAHFL, strict fit-only control, and FedFalsify v0.3 runs completed for
  12 rounds.
- FedFalsify versus strict control final delta:
  `Avg +0.3183`, `Worst -0.4067`, `WCCA +0.250`, `CFG +1.600`.
- FedFalsify versus strict control last-five delta:
  `Avg +0.1180`, `Worst -0.4373`, `WCCA +0.850`, `CFG +2.185`.
- FedFalsify has a real early communication signal, but later class-operator
  negative transfer harms fairness and counterfactual generalization.
- The frozen gate failed. A 40-round FedFalsify run is not justified.
- RAHFL reached `33.8267/27.0400/WCCA 0.250/CFG 30.050`, but this is not yet a
  strict fair comparison: it uses all local samples and final-test accuracy for
  routing, while FedFalsify reserves 15% audit data and never routes on test.
- Next required implementation/run is strict RAHFL-val on the identical
  fit/audit split. Existing control and FedFalsify runs remain reusable.

Analysis:
`deliverables/cle_hfl_v2_probe_analysis_20260724/`.

## CLE-HFL v2 Operator Protocol Implemented - 2026-07-24

- Replaced the next benchmark protocol's four broad corruption groups with 15
  concrete CIFAR-C-style operators.
- Added a deterministic 11-seen/4-unseen operator split.
- Added randomized client/class dominant-operator assignment.
- Added class-balanced clean/seen/unseen/all evaluation data.
- Kept operator IDs strictly outside all FedFalsify training and routing paths.
- Added operator-level Avg/Worst/WCCA/CFG and full per-round
  client/class/operator CSV output.
- Added RAHFL, strict fit-only control, and FedFalsify v0.3 12-round configs.
- Added one OpenI entry with automatic import, heartbeat logging, packaging,
  and c2net upload.
- Formal data and protocol audit completed.
- Focused tests: `22 passed`.
- RTX 3050 two-round active-communication smoke passed.
- RTX 3050 one-round RAHFL/AsymHFL v2 smoke passed.

Status: formal 12-round result exists. The protocol is executable, but the
current FedFalsify v0.3 method failed the frozen gate.

## FedFalsify v0.3 Implemented And Smoke-Tested - 2026-07-23

- Added paired standard error and one-sided non-inferiority UCB.
- Added a switchable `noninferiority_veto` before TAU Top-1.
- Added candidate-level UCB, eligibility, and rejection-reason logs.
- Added route-level eligible/rejected counts and selected mean UCB metrics.
- Preserved v0.2 behavior when the new switch is disabled.
- Added independent local-debug and OpenI candidate-only configurations.
- Focused tests passed: `15 passed`.
- RTX 3050 two-round smoke passed through warmup and active CMT communication.

Smoke communication round:

```text
candidates=99
eligible=84
statistically inferior rejected=15
active routes=11/40
cmt_loss=1.2705
```

The method is executable but not yet a positive research result. Run one
12-round candidate-only probe next and compare it with the stored strict
fit-only control.

### v0.3 probe completed - 2026-07-24

```text
final delta vs strict control:
  Avg +1.2844, Worst +0.3450, WCCA +3.200, CFG -0.3500

last-five delta vs strict control:
  Avg +1.3846, Worst +1.1195, WCCA +1.475, CFG +0.6385
```

The frozen gate passed three of four metrics and failed CFG. The veto cut
selected nonpositive-advantage teachers from 54.17% to 27.98%, and Avg/Worst
improved in all 9 communication rounds. Current status: positive partial
communication result, not approved for a 40-round run.

## FedFalsify v0.2 Strict Probe Ready - 2026-07-23

- Implemented deterministic class-stratified `D_fit/D_audit` persistence.
- Rare classes remain in fit when a valid audit-and-fit split is impossible.
- Implemented per-round frozen peer snapshots.
- Implemented receiver-private head-TAU Top-1 class routing.
- Implemented CMT as an optional local AugMix/JSD/DCL loss.
- Added a strict fit-only control using the identical split.
- Added OpenI A/B entry, automatic comparison, packaging, and c2net upload.
- Confirmed no final test metric participates in routing.
- Passed 14 focused tests and two local 3050 end-to-end debug runs.

Current status: ready for the 12-round OpenI Go/No-Go probe, not ready for a
40-round claim.

### Probe completed

The strict probe has completed. Last-five FedFalsify minus control is:

```text
Avg +0.5450, Worst +0.2810, WCCA +0.310, CFG +0.446
```

Because lower CFG is better, the gate failed. Post-warmup Avg improves in all
9/9 rounds, but 54.17% of selected teachers have nonpositive paired accuracy
advantage. v0.2 is archived as a weak positive communication result and must
not be run for 40 rounds unchanged.

Next candidate: TAU Top-1 preceded by a statistical non-inferiority veto.

## FedFalsify v0.1 Offline Gate Completed - 2026-07-23

The project now has a tested offline audit implementation for:

```text
Foreign Transfer Tensor
FRA paired advantage and projected gate coverage
CMT / fixed-margin / direct-KD controls
TAU gradient agreement
exact frozen-BN one-step parameter update
```

Tests: `tests/test_fedfalsify_audit.py` -> `10 passed`.

The three-gamma real-checkpoint audit says that direct KD is increasingly
harmful and that CMT is mildly positive, but the original FRA hard gate is too
sparse. FedFalsify v0.1 is therefore a No-Go for a 40-round run. The next
candidate is TAU-first top-1 source selection with FRA demoted to a ranking
prior. See `docs/experiments/archive/FEDFALSIFY_AUDIT_GUIDE_ZH.md` and the latest section of
`docs/project/CURRENT_PROJECT_MEMORY.md`.

The follow-up source-ranking audit supports that revision:

```text
gamma                         0.0      0.6      0.9
TAU top-1 coverage            100%     100%     100%
positive precision            91.4%    94.3%    85.7%
mean increment over CE        .00354   .00367   .00320
```

Before a 12-round runner, validate a cheaper head-only or last-block TAU against
the current full-model TAU.

## Calibrated PEW Local Attribution Result - EBST-v2 Rejected - 2026-07-22

The required 12-round calibrated PEW local-only control completed:

```text
local final Avg/Worst/WCCA/CFG     = 42.8469/36.2300/19.775/6.5725
EBST-v2 final                      = 42.6331/35.2975/20.675/7.2900
EBST-v2 minus local                = -0.2138/-0.9325/+0.900/+0.7175

local last-five mean               = 40.4278/36.2890/17.965/6.427
EBST-v2 last-five mean             = 40.4526/35.9870/17.400/6.666
EBST-v2 minus local last-five      = +0.0249/-0.3020/-0.565/+0.239
```

The two runs match in PEW checkpoint/threshold, inferred environments, data,
models, seed, optimizer, and training budget. The only experimental difference
is EBST-v2/SCP communication, so attribution is valid. EBST-v2 fails the frozen
gate: average gain is effectively zero, Worst and CFG regress, and three of four
clients lose accuracy. The calibrated PEW + BER+CDep local mechanism is the
validated positive component. Hard-taxonomy EBST-v2 is now a negative archived
route; a 40-round run is blocked pending communication redesign.

## Calibrated PEW + EBST-v2 Probe Result - Positive but Attribution Unresolved - 2026-07-21

The 12-round combination probe completed:

```text
final Avg/Worst/WCCA/CFG = 42.6331/35.2975/20.675/7.290
last-five mean           = 40.4526/35.9870/17.400/6.666
```

Relative to old learned-PEW local, final Avg is `+2.2638` and final WCCA is
`+6.75`; last-five Avg/Worst/WCCA improve and CFG drops by `0.8795`. PEW private
group accuracy rises from `38.83%` to `63.59%` after best-epoch restoration and
automatic threshold calibration.

The experiment is confounded for communication attribution: rounds 0-2 already
improve before EBST-v2 starts. Therefore the complete candidate has a positive
signal, but EBST-v2 itself is not yet validated. A matching calibrated learned
PEW local-only probe is required before any 40-round run.

The matching control is implemented in
`configs/openi_v100_fedease_pew_calibrated_local_probe.yaml` and exposed as
`scripts/openi_fedease_entry.py --mode=pew_calibrated_local_probe`. Fifteen
targeted tests and the formal dependency/path check pass.

## Calibrated PEW + EBST-v2 Combination Probe Ready - 2026-07-21

The next OpenI experiment is implemented and locally verified:

```text
scripts/openi_fedease_entry.py --mode=pew_ebst_v2_probe
configs/openi_v100_fedease_pew_ebst_v2_probe.yaml
```

Changes are intentionally isolated: best-validation PEW checkpoint restoration,
validation-calibrated unknown threshold, and one new 12-round learned
PEW+EBST-v2 config/entry. Historical Oracle, PEW local, legacy EBST, and Oracle
EBST-v2 routes are unchanged. `26` targeted tests and a real-data smoke pass.

The old OpenI dataset `openi_cle_rahfl_diagnostic` is sufficient. Full mode is
still blocked pending this result.

## FedEASE Learned PEW Probe Result - Near Pass - 2026-07-20

The 12-round learned-environment local probe completed:

```text
control:           37.5813 / 30.1100 / WCCA 13.700 / CFG 10.855
Oracle BER+CDep:   41.6206 / 35.5175 / WCCA 14.000 / CFG  6.155
PEW BER+CDep:      40.3694 / 35.4225 / WCCA 13.925 / CFG  6.370
```

PEW retains `+2.7881` Avg, `+5.3125` Worst, and `-4.485` CFG relative to the
local control. It is only `0.095` below Oracle on Worst and `0.075` below Oracle
on WCCA, but is `1.2513` lower on Avg. The frozen gate passed three metrics and
missed `Avg >= 40.5` by `0.1306`.

Private exact environment-group accuracy is `38.83%`, with roughly half of the
private samples assigned to `unknown`. Nevertheless downstream tail and CFG
benefits are largely preserved, suggesting PEW embeddings/coarse partitions are
useful even when exact taxonomy prediction is imperfect.

The public validation environment accuracy peaked at `57.4%` on epoch 3, while
the saved final epoch reports `52.5%`. PEW checkpoint selection and unknown-
threshold calibration should be corrected before one deployable PEW+EBST-v2
combination probe. Full 40-round mode remains blocked.

## FedEASE EBST-v2 Corrective Probe Result - Mixed/Insufficient - 2026-07-20

The 12-round OpenI probe completed on CLE-HFL `alpha=0.5, gamma=0.9, seed=0`:

```text
Oracle BER+CDep local:          41.6206 / 35.5175 / WCCA 14.000 / CFG 6.155
Oracle BER+CDep+EBST-v2+SCP:   41.9469 / 36.2275 / WCCA 14.700 / CFG 5.190
final delta:                   +0.3263 / +0.7100 / +0.700 / -0.965
last-five mean delta:          -0.1648 / +0.4400 / +0.765 /  0.000
```

This is a major safety improvement over legacy EBST: no client collapses, and
client 2 changes by `+0.5125` rather than `-10.4950`. Communication is active
with mean valid-pair fraction `0.6775`, source count `2.1635`, and gate `0.2101`.
Class-wise SCP detects conflicts in `47.59%` of updates and retains `57.63%` of
the communication-gradient norm.

The predeclared average gate (`Avg > 42.1`) was not met. The last-five average is
also slightly worse than local-only, so this is not a stable positive average-
accuracy result. EBST-v2 is classified as safety-correct but average-neutral;
full mode remains blocked.

## FedEASE EBST-v2 Corrective Implementation - 2026-07-20

Implemented a separate EBST-v2 route without changing the archived legacy EBST:

```text
communication: ebst_v2
```

The correction directly addresses the failed probe:

```text
source eligibility is now class-pair specific;
each recipient receives a leave-one-out teacher;
the gate includes cross-client relation disagreement;
SCP projects and caps each classifier class row independently;
formal communication starts after three warmup rounds.
```

The formal 12-round OpenI probe is configured in
`configs/openi_v100_fedease_ebst_v2_probe.yaml`. Targeted tests report `24 passed`,
and a two-round real-data smoke exercised LOO aggregation and class-wise SCP with
finite losses and gradients. The completed result is recorded above.

## FedEASE Oracle EBST Communication Probe - Negative Result - 2026-07-20

The 12-round Oracle EBST probe completed on the same CLE-HFL
`alpha=0.5, gamma=0.9, seed=0` setting as the positive local probe.

```text
Oracle BER+CDep local:       Avg=41.6206, Worst=35.5175, WCCA=14.000, CFG=6.155
Oracle BER+CDep+EBST+SCP:   Avg=38.7038, Worst=34.7225, WCCA=15.325, CFG=6.415
EBST delta:                  Avg=-2.9169, Worst=-0.7950, WCCA=+1.325, CFG=+0.260
```

EBST executed normally (`mean loss=0.1392`, `mean gate=0.3905`, valid environment
fraction `1.0`). SCP detected conflicts in about `45.31%` of batches, but retained
about `98.08%` of the communication-gradient norm. Client 2 collapsed from
`45.2175` to `34.7225`; this dominates the average-accuracy regression.

Therefore the result rejects the current EBST communication design, not the
Oracle BER+CDep local mechanism. Full FedEASE and PEW+EBST training are blocked
until communication is redesigned. No claim should be made that the current
stability gate or SCP prevents negative transfer.

## FedEASE Oracle Local Formal Probe Result - 2026-07-20

The first formal FedEASE mechanism probe completed successfully on OpenI using
the existing gamma=0.9 CLE-HFL package. Both experiments are local-only and use
the same data, seed, models, optimizer, round count, and batch budget.

```text
Oracle control:       Avg=37.5813, Worst=30.1100, WCCA=13.70, CFG=10.855
Oracle BER+CDep:      Avg=41.6206, Worst=35.5175, WCCA=14.00, CFG= 6.155
Delta:                Avg=+4.0394, Worst=+5.4075, WCCA=+0.30, CFG=-4.70
```

All clients and all corruption groups improve. The final worst corruption-group
accuracy rises by `+6.2575`, and worst client-corruption accuracy rises by `+9.48`.
The automatic Go/No-Go decision is `pass=true`.

This validates the joint Oracle BER+CDep local mechanism, not PEW, and does not
separate BER from CDep. The subsequent Oracle EBST probe failed as recorded above.

## FedEASE v2.1 Complete Candidate Implementation - 2026-07-19

The full switchable candidate is now implemented for CLE-HFL:

```text
PEW learned environment estimation
+ Oracle/learned BER and CDep
+ EBST environment-balanced structural communication
+ cross-environment stability gate
+ classifier-head SCP negative-transfer protection
+ clean/same/random/swapped/unseen evaluation
```

Formal OpenI files:

```text
scripts/openi_fedease_entry.py
configs/openi_v100_fedease_oracle_control_probe.yaml
configs/openi_v100_fedease_oracle_ber_cdep_probe.yaml
configs/openi_v100_fedease_pew_probe.yaml
configs/openi_v100_fedease_ebst_probe.yaml
configs/openi_v100_fedease_full.yaml
docs/experiments/archive/FEDEASE_OPENI_RUN_GUIDE_ZH.md
```

Prepared upload artifact:

```text
local_runs/cle_hfl_prepared/fedease_cle_prepared_alpha05_gamma09_seed0.tar.gz
about 623.29 MiB
```

Verification completed:

```text
Python compile passed
19 targeted FedEASE tests passed
OpenI entry dry-run passed
two-round four-model real-data EBST smoke passed
all five evaluation splits executed
```

Smoke diagnostics at round 1:

```text
avg/worst=12.11/9.38 (not a research result)
EBST loss=0.4411
valid environment fraction=0.650
mean gate=0.406
SCP conflict rate=0.75
no NaN/non-finite gradient
```

Research status remains unvalidated. The first formal task is `--mode=oracle_probe`.
Do not infer that PEW or EBST is effective from smoke tests, and do not run the 40-round
full candidate before the staged Oracle/PEW/EBST gates pass.

## FedSARA-CS Corruption-Skew Protocol - 2026-07-08

Implemented a new corruption-skew scenario for a stronger paper motivation:

```text
model heterogeneity + label-skew Non-IID + corruption-skew Non-IID
```

Instead of only using RAHFL-style globally random corruption, each client now has
a dominant corruption group:

```text
client 0: noise
client 1: blur
client 2: weather
client 3: digital
```

Generated formal prepared dataset:

```text
local_runs/fedsara_cs_prepared/fedsara_cs_prepared_alpha05_rho07_seed0.tar.gz
```

Dataset metadata:

```text
alpha = 0.5
rho = 0.7
seed = 0
clients = 4
samples_per_client = 10000
balanced test groups = noise / blur / weather / digital
```

The audit files show both label-skew and corruption-skew are present:

```text
local_runs/fedsara_cs_prepared/fedsara_cs_prepared_alpha05_rho07_seed0/
  cifar_10_cs/alpha05_rho07_seed0/audit/client_label_counts.csv
  cifar_10_cs/alpha05_rho07_seed0/audit/client_corruption_counts.csv
```

Implemented code/configs:

```text
fedprime/data/corruptions.py
fedprime/data/loaders.py
fedprime/methods/rahfl_asymhfl.py
scripts/prepare_corruption_skew_data.py
scripts/import_fedsara_cs_data.py
scripts/run_openi_fedsara_cs.sh
configs/openi_v100_rahfl_cs_alpha05_rho07.yaml
configs/openi_v100_fedsara_cs_alpha05_rho07.yaml
configs/debug_rahfl_cs.yaml
configs/debug_fedsara_cs.yaml
docs/experiments/archive/FEDSARA_CS_SCENARIO_OPENI_GUIDE_ZH.md
```

Formal comparison to run on OpenI:

```text
RAHFL-CS:
  AugMix/JSD + DCL + AsymHFL
  config: configs/openi_v100_rahfl_cs_alpha05_rho07.yaml

FedSARA-CS:
  AugMix/JSD + SARA + CS-AsymHFL
  config: configs/openi_v100_fedsara_cs_alpha05_rho07.yaml
```

Both formal configs enable 40-epoch local CE pretraining:

```text
pretrain_epochs: 40
rounds: 40
```

The pretraining path uses a plain corruption-skew CE loader for efficiency and
fairness. Formal training rounds still use the AugMix/JSD local base.

Local smoke tests passed:

```text
python scripts/run_experiment.py --config configs/debug_fedsara_cs.yaml
python scripts/run_experiment.py --config configs/debug_rahfl_cs.yaml
```

The smoke tests validated:

```text
1. corruption-skew data import
2. CIFAR-100 public tar fallback loader
3. RAHFL-CS runner path
4. FedSARA-CS runner path
5. metrics.csv, corruption_group_acc.csv, client_group_acc.csv output
```

## SARA + AsymHFL Seed Validation - 2026-07-05

New archives analyzed:

```text
outputs/rahfl_seed1_results.tar.gz
outputs/sara_rahfl_seed12_results.tar.gz
```

Alpha=0.5, corrupt_rate=1, 40-round unified-runner results:

```text
RAHFL seed0:          56.41   / 44.72
RAHFL seed1:          56.645  / 45.29

SARA + AsymHFL seed0: 57.83   / 46.59
SARA + AsymHFL seed1: 57.2975 / 46.23
SARA + AsymHFL seed2: 58.0025 / 45.90
```

Seed1 matched comparison:

```text
SARA - RAHFL = +0.6525 avg_acc, +0.94 worst_acc
```

SARA seeds0/1/2 mean final:

```text
avg_acc   = 57.71
worst_acc = 46.24
```

Important caveat:

```text
The archived alpha=0.5 partition files named seed0/seed1/seed2 are byte-identical
by SHA-256 prefix and have identical client_class_counts. Current alpha=0.5
multi-seed evidence validates training/randomness stability on one fixed
label-skew partition, not cross-partition robustness.
```

Current next actions:

```text
1. Run the missing RAHFL seed=2 matched control.
2. Verify/generate genuinely distinct fixed partitions if cross-partition claims
   are needed.
3. Run SARA + AsymHFL at alpha=0.3, alpha=0.1, and alpha=1.0 before changing
   the communication module.
```

## FedPRIME-PAIR Implementation - 2026-06-25

The project now includes a switchable first implementation of the new
FedPRIME-PAIR route:

```text
FedPRIME-PAIR = PRIME + CBCL + CPAD
```

Latest code/runtime update:

```text
105e6c6 optimize FedPRIME-PAIR heartbeat logging and CBCL forward pass
9942276 make import_prepared_data.py accept both --destination and --repo-root
```

The previous Kaggle full run that showed no round output for hours should not
be interpreted as a valid algorithm result. It was started before the heartbeat
and CBCL-forward optimization. The current full config now prints:

```text
[heartbeat] round 000 start
[heartbeat] round 000 local client 0 start
[heartbeat] FedPRIME-PAIR local phase, client=0 batch=50 loss=...
[heartbeat] round 000 local client 0 done ...
```

If a fresh Kaggle run reaches setup successfully but prints no heartbeat for
more than about 10 minutes, stop it and inspect the setup/logs instead of
waiting multiple hours.

The CBCL optimization is an engineering fix, not an algorithmic change. RAHFL
heterogeneous models normally return `(logits, embedding)`. The first CBCL
implementation used `forward_logits(model, views)`, which kept only logits and
discarded the already-computed embedding, then ran the backbone again to obtain
features for contrastive learning. The current code calls `model(views)` once
and reuses both outputs:

```text
logits_all, features_all = output[0], output[1]
```

Only models that return logits alone fall back to a second backbone call.

Kaggle data import helper:

```text
scripts/import_prepared_data.py
```

now accepts both of the following equivalent arguments:

```bash
--destination /kaggle/working/fedprime-d2c
--repo-root /kaggle/working/fedprime-d2c
```

The motivation is the completed D2C diagnostics:

```text
PRIME + D2C final:      avg_acc=52.31, worst_acc=39.78
PRIME + LogitAvg final: avg_acc=52.10, worst_acc=39.72
Oracle D2C final:       avg_acc=51.74, worst_acc=39.13
RAHFL final:            avg_acc=56.41, worst_acc=44.72
```

Current D2C is effectively tied with LogitAvg, so the new route no longer uses
public logits to infer a private prior. Instead, it estimates which client is
reliable on each directed class-pair boundary and distills public logits at the
class-pair level.

Implemented files:

```text
fedprime/methods/cpad.py
fedprime/methods/fedprime_pair.py
fedprime/methods/local_prime.py       # added PRIME+CBCL local training
scripts/analyze_pair_expertise.py
```

Entry point:

```text
method_name: fedprime_pair
```

Configs:

```text
configs/debug_fedprime_pair_cifar10c.yaml
configs/kaggle_t4_fedprime_pair_full.yaml
```

Default full setting:

```text
PRIME + JSD + CBCL + CPAD
cpad.warmup_rounds = 3
leave_one_out = true
```

All major modules are configurable:

```yaml
method:
  use_prime: true
  use_cbcl: true
  use_cpad: true
```

Local smoke test passed on the local `pytorch` conda environment:

```text
config: configs/debug_fedprime_pair_cifar10c.yaml
round 0: avg_acc=11.52, worst_acc=10.00,
         local_loss=5.1416, cpad_loss=0.7056,
         expertise_mean=0.7439
```

Generated outputs:

```text
outputs/debug_fedprime_pair_cifar10c/metrics.csv
outputs/debug_fedprime_pair_cifar10c/checkpoints/client_*.pt
outputs/debug_fedprime_pair_cifar10c/pair_expertise/round_000.npz
outputs/debug_fedprime_pair_cifar10c/pair_expertise_analysis/
```

The smoke result is only a path validation result, not a performance result.
The next formal run should be the 40-round Kaggle T4 full config.

## Current State - 2026-06-06

The first Kaggle core comparison exposed and helped isolate a PRIME numerical
stability bug:

```text
RAHFL = AugMix + DCL + AsymHFL
FedPRIME-D2C = PRIME + 3 local-only warmup rounds + D2C
```

Configs:

```text
configs/kaggle_t4_rahfl.yaml
configs/kaggle_t4_fedprime_d2c_warmup3.yaml
```

Both methods use the same prepared data, heterogeneous models, optimizer
settings, and fixed Non-IID partition:

```text
outputs/partitions/cifar10c_alpha05_seed0_clients4_samples10000.npz
```

Kaggle prepared data is stored as the mounted dataset `fedprime-data`. Kaggle
automatically exposes its contents below `/kaggle/input`. The import helper:

```text
scripts/import_prepared_data.py
```

automatically locates the mounted CIFAR data and copies it into:

```text
RAHFL-master/Dataset/cifar_10_c
RAHFL-master/Dataset/cifar_100
outputs/partitions
```

This avoids downloading CIFAR data again for every new Kaggle session.

RAHFL completed all 40 rounds successfully:

```text
round 0: avg_acc=22.94 worst_acc=21.00 local_loss=15.1687 col_loss=0.1735
round 39: avg_acc=56.41 worst_acc=44.72 local_loss=12.2930 col_loss=1.7927
```

Important interpretation:

```text
This is a valid baseline for the current resource-limited unified runner.
It is not a full reproduction of the paper's strongest RAHFL result.
```

The paper first pre-trains each local model for 40 epochs and then runs 40
communication rounds. The current runner starts from random initialization,
uses 4 public batches per round instead of the full 5000-image public set, and
uses a more severe alpha=0.5 plus corruption-rate=1 Non-IID setting. See
`docs/experiments/guides/EXPERIMENT_GUIDE_ZH.md` for the complete comparison.

The original FedPRIME-D2C warmup=3 run diverged:

```text
rounds 0-2: local_loss=nan while d2c_loss=0
round 3 onward: local_loss=nan and d2c_loss=nan
```

This proves D2C was not the initial cause. The failure began during PRIME local
training before communication was enabled.

Root cause and fix:

```text
Root cause: ShuffleNet PRIME JSD could have finite loss but non-finite gradients
because softmax targets underflowed to exact zero inside KLDiv.

Fix: clamp and renormalize each JSD target distribution before KLDiv.
Added: first-failure finite checks, gradient checks, optional gradient clipping,
and scripts/diagnose_prime_stability.py.
```

Local verification after the fix:

```text
ResNet10: PASS
ResNet12: PASS
ShuffleNet: PASS
Mobilenetv2: PASS
```

All four clients completed a full local PRIME epoch without NaN/Inf. The main
Kaggle warmup config does not enable gradient clipping, matching the completed
RAHFL optimizer settings. Non-finite gradient detection remains enabled and
will stop the run immediately if numerical instability returns.

The detailed Chinese experiment/configuration and metric guide is:

```text
docs/experiments/guides/EXPERIMENT_GUIDE_ZH.md
```

The repaired FedPRIME-D2C warmup=3 run has now completed successfully:

```text
round 0:  avg_acc=19.92 worst_acc=18.89 local_loss=1.6707 d2c_loss=0.0000
round 2:  avg_acc=25.93 worst_acc=24.03 local_loss=1.4589 d2c_loss=0.0000
round 3:  avg_acc=28.15 worst_acc=15.74 local_loss=1.4052 d2c_loss=1.8864
round 37: avg_acc=52.83 worst_acc=38.39 local_loss=0.9280 d2c_loss=1.0381
round 39: avg_acc=52.31 worst_acc=39.78 local_loss=0.9123 d2c_loss=1.0764
```

Comparison against the completed lightweight RAHFL baseline:

```text
                         final avg_acc   final worst_acc
RAHFL                         56.41             44.72
FedPRIME-D2C warmup=3         52.31             39.78
gap                            -4.10             -4.94
```

Interpretation:

```text
The numerical fix worked: all 40 rounds are finite and local_loss decreases.
PRIME + D2C learns substantially, but does not beat RAHFL in this first valid run.
The weakest client drops sharply when D2C first turns on at round 3, then recovers.
This suggests the current early D2C teacher/prior may be too aggressive for weak clients.
```

Current action:

```text
Save the Kaggle outputs and run underrepresented-class diagnosis.
Do not run multi-seed yet.
```

The strict T4-safe LogitAvg+PRIME control experiment has also completed:

```text
LogitAvg+PRIME round 39: avg_acc=52.10, worst_acc=39.72
LogitAvg+PRIME best avg: 52.19 at round 37
LogitAvg+PRIME best worst: 39.98 at round 38
```

Direct communication comparison:

```text
                         final avg_acc   final worst_acc   best avg_acc
LogitAvg+PRIME                 52.10             39.72          52.19
FedPRIME-D2C                   52.31             39.78          52.83
D2C improvement                +0.21             +0.06          +0.64
```

Interpretation:

```text
Current D2C is effectively tied with plain LogitAvg; the tiny gains are below
what should be treated as a meaningful single-seed improvement.
The main bottleneck is now confirmed to be the D2C mechanism, not numerical
stability. Predicted priors estimated from cross-domain CIFAR-100 public images
may be close to uniform or unreliable, causing prior debias, class balancing,
and complementary KD to degenerate toward ordinary logit averaging.
```

Current next action:

```text
Run a T4-safe Oracle Prior D2C diagnostic first.
If Oracle Prior substantially beats 52.31, redesign predicted-prior estimation.
If Oracle Prior remains near 52, inspect aggregation and complementary KD.
```

## Oracle Prior Diagnostic Implementation - 2026-06-07

The T4-safe Oracle Prior diagnostic and predicted-vs-true prior logging are now
implemented:

```text
configs/kaggle_t4_fedprime_d2c_oracle_warmup3.yaml
configs/debug_fedprime_d2c_oracle.yaml
fedprime/engine/prior_diagnostics.py
scripts/analyze_priors.py
```

Low-intrusion guarantee:

```text
The existing D2CServer.build_teacher() compatibility API remains available.
Diagnostics use a separate build_teacher_with_diagnostics() API.
When prior_diagnostics.enabled is false, the normal runner does not record or
export diagnostic values.
Regression tests prove the default predicted-prior teacher and prior are
element-for-element equal to the previous D2C formula.
```

Oracle formal-run outputs:

```text
prior_diagnostics.csv  complete per-round/public-batch/client prior vectors
prior_summary.json     aggregate L1/KL/cosine/entropy/top-match statistics
priors/round_*.npz     compact full-prior snapshots for selected rounds
```

Analyze after the run:

```bash
python scripts/analyze_priors.py \
  --experiment_dir outputs/fedprime_d2c_oracle_cifar10c_alpha05_cr1_t4_warmup3
```

Decision target:

```text
PRIME+LogitAvg is already 52.10.
An Oracle result near 60 would demonstrate that D2C has roughly the desired
+8-point headroom and that predicted-prior estimation is the main bottleneck.
An Oracle result still near 52 means the current D2C formulas need redesign.
```

Verification:

```text
5 unit/regression tests pass.
The local full debug run could not start because the local CIFAR-100 directory
failed torchvision integrity validation. Run the formal smoke/full experiment
with the complete Kaggle mounted prepared dataset.
```

## Kaggle Background-Run Constraint

Kaggle `Save Version` / background execution is not interactive:

```text
Once execution starts, no new diagnostic cell can be run and no cell can be
edited. Any change requires cancelling the run and starting a fresh version.
```

Therefore every future Kaggle experiment must be provided as a complete
pre-validated sequence that automatically:

```text
clones code -> imports mounted data -> checks CUDA/config/paths -> starts
training with unbuffered logging -> analyzes results -> packages outputs
```

Do not advise running another cell while a background version is executing.
Inside a normal Python cell use `%cd /kaggle/working/fedprime-d2c`; use plain
`cd /kaggle/working/fedprime-d2c` only inside a cell beginning with `%%bash`.

## Kaggle Streaming Launcher Rule - 2026-06-25

For long Kaggle `Save Version` runs, do **not** use a long `%%bash` cell as the
primary launcher. Kaggle/IPython may buffer `%%bash` stdout until the subprocess
finishes, which can make a live experiment look frozen for hours while only
showing kernel/debugger warnings.

Use a single Python streaming launcher cell instead. It must run shell commands
through `subprocess.Popen`, read stdout line by line, and print a driver
heartbeat every 60 seconds. Do not call `sys.stdout.reconfigure(...)` inside
Kaggle notebooks, because Kaggle uses an `OutStream` object that does not
provide `reconfigure`.

The streaming launcher must verify these items before training:

```text
1. print START immediately
2. list /kaggle/input
3. clone or pull the repo with GIT_TERMINAL_PROMPT=0
4. print git log -1 --oneline
5. confirm commit is 8a4ee15 or later for FedPRIME-PAIR
6. run scripts/run_kaggle_pair.sh with RUN_DEBUG=1 and PYTHONUNBUFFERED=1
```

A healthy FedPRIME-PAIR Kaggle run should show:

```text
===== START FedPRIME-PAIR Kaggle streaming launcher =====
8a4ee15 增强FedPRIME-PAIR启动日志与Kaggle一键脚本
===== FedPRIME-PAIR Kaggle one-shot launcher =====
===== Importing prepared Kaggle data =====
===== Running FedPRIME-PAIR debug smoke =====
===== Running FedPRIME-PAIR full experiment =====
[setup] FedPRIME-PAIR loading private labels
[heartbeat] round 000 local client 0 start
[heartbeat] FedPRIME-PAIR local phase, client=0 batch=50 loss=...
```

If a future Kaggle log only shows:

```text
Debugger warning: It seems that frozen modules are being used
```

for many minutes, do not interpret it as training progress. Stop the run and
check whether the saved notebook version contains the Python streaming launcher
cell and whether `git log -1` reaches `8a4ee15` or later.

## Local RTX 3050 Oracle Validation - 2026-06-07

The Oracle implementation has now been validated through a real one-round
end-to-end run on the local RTX 3050 Laptop GPU:

```text
torch: 2.8.0+cu126
GPU: NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB
runtime: about 35 seconds
round 0: avg_acc=9.74, worst_acc=9.09, local_loss=2.4352, d2c_loss=1.0566
```

The run successfully completed:

```text
PRIME local training
Oracle D2C teacher construction
client public-data distillation
full shared-test evaluation
metrics/prior CSV and JSON export
selected-round NPZ export
prior analysis plots
four final client checkpoints
```

The original local CIFAR-100 extracted directory has a Windows ACL problem and
cannot be read. For the validation, the existing CIFAR-100 tar archive was
extracted into `outputs/local_debug_data/cifar_100` without modifying or
deleting the inaccessible directory.

The first real prior diagnostic strongly supports the current hypothesis:

```text
predicted normalized entropy mean: 0.999900
oracle normalized entropy mean:    0.748475
prior L1 mean:                     0.907714
prior KL mean:                     0.568954
top-class match:                   0.50
```

The predicted prior is almost perfectly uniform in this early debug round,
while the real client priors are clearly skewed. The formal 40-round Oracle run
is still required to measure the performance upper bound.

## Full Oracle Prior Result - 2026-06-07

The full 40-round T4 Oracle Prior experiment completed and was extracted under:

```text
outputs/oracle_result_extracted/
```

Results:

```text
Oracle final:      avg_acc=51.74, worst_acc=39.13
Oracle best avg:   52.65 at round 37
Oracle best worst: 39.89 at round 38

Predicted D2C final: avg_acc=52.31, worst_acc=39.78
LogitAvg final:      avg_acc=52.10, worst_acc=39.72
RAHFL final:         avg_acc=56.41, worst_acc=44.72
```

Oracle did not improve D2C. Relative to predicted-prior D2C:

```text
final avg_acc:   -0.57
final worst_acc: -0.65
best avg_acc:    -0.18
best worst_acc:  +0.11
```

Important interpretation:

```text
Predicted priors are too smooth and imperfect, but prior-estimation error alone
does not explain the D2C bottleneck. The current formulas do not benefit from a
true private-label prior and may over-correct clients.
```

Oracle communication caused a strong early weak-client shock:

```text
round 2 -> 3 worst_acc: 24.03 -> 17.39
round 5 worst_acc:       10.42
round 4 d2c_loss:         3.99
```

The current Oracle experiment must not be described as a guaranteed performance
upper bound. The true prior is from private CIFAR-10 labels while teacher logits
come from cross-domain CIFAR-100 images, so standard label-shift prior
correction assumptions do not hold.

Highest-priority next diagnostic:

```text
Oracle + no prior debias
```

The term `- beta * log(prior)` can add about `+3.45` logit to a missing class
when `beta=0.5` and `p_min=0.001`, making it the strongest candidate for the
early harmful communication shock. After that, separately ablate class-balanced
aggregation and complementary KD, and test a smaller/ramped D2C strength.

Final-checkpoint underrepresented-class diagnosis further confirms the failure
mode:

```text
client 2: head_acc=75.48, tail_acc=4.63, missing_acc=0.00
client 3: head_acc=74.37, tail_acc=0.00, missing_acc=0.00
```

Client 2 is missing classes 8/9 and client 3 is missing class 9. Oracle D2C
learned none of those missing classes. The current complementary KD therefore
does not achieve its intended purpose of transferring missing-class knowledge.

## RAHFL Missing/Tail Diagnostic - 2026-06-08

RAHFL-original was rerun with the same T4-safe configuration and fixed
partition, then diagnosed with `scripts/diagnose_underrepresented.py`.

Overall result reproduced the previous baseline:

```text
RAHFL final:      avg_acc=56.41, worst_acc=44.72
RAHFL best avg:   56.41 at round 39
RAHFL best worst: 44.72 at round 39
```

Underrepresented-class result:

```text
client 0: overall=66.27, head=78.39, tail=38.00, missing=nan
client 1: overall=64.92, head=79.54, tail=30.80, missing=nan
client 2: overall=44.71, head=84.14, tail=8.80,  missing=0.00
client 3: overall=49.74, head=82.03, tail=1.73,  missing=0.00
```

The fixed partition has:

```text
client 2 missing classes: 8, 9
client 3 missing classes: 9
```

RAHFL-original still obtains `0%` missing accuracy for all missing classes.
Therefore, in the current alpha=0.5 setting, RAHFL's higher average accuracy
does not mean it transfers completely missing CIFAR-10 classes through
cross-domain CIFAR-100 public logits. Its advantage mostly comes from stronger
head-class performance and modest tail-class gains on classes that are still
present locally.

This result supports the paper angle:

```text
RAHFL can improve robust heterogeneous collaboration, but it does not explicitly
solve class-missing knowledge transfer under label-skew Non-IID data.
```

Next design work should preserve PRIME and the public-logit communication
interface, but should no longer rely on private-prior debiasing or assume
cross-domain CIFAR-100 logits automatically carry missing target-class
semantics.

## Pending PRIME Local-Backbone Control - 2026-06-23

The next required fairness control is now configured but has not yet been run:

```text
configs/kaggle_t4_rahfl_prime.yaml
```

It runs:

```text
PRIME + DCL + original RAHFL AsymHFL
```

It is deliberately identical to the completed T4-safe RAHFL configuration in
all data, fixed partition, heterogeneous models, optimizer, rounds, and public
communication budget. The only change is the local robust learner:

```text
AugMix -> PRIME
```

This control must be completed before interpreting any new PRIME-based
communication method. It answers whether PRIME improves the RAHFL local
backbone under the same Non-IID/corruption setting.

Local path verification is complete:

```text
config: configs/debug_rahfl_prime_cifar10c.yaml
round:  1
result: avg_acc=10.50, worst_acc=8.25, local_loss=18.0854, col_loss=0.2041
```

This is a random-initialization smoke result, not a performance result. It
confirms that PRIME three-view training, original DCL, original AsymHFL public
distillation, four heterogeneous models, metrics, and checkpoints run through
one complete round together. The formal 40-round T4 result is still pending.

## Resume Update - 2026-06-05

Completed since the previous state update:

- Added the Chinese experiment/configuration guide:

```text
docs/experiments/guides/EXPERIMENT_GUIDE_ZH.md
```

It records today's warmup=3 Kaggle comparison, all existing experiment configs,
their purposes, expected outputs, recommended execution order, and missing
T4-safe configs.

- Fixed the Kaggle formal comparison OOM by adding T4-safe configs:

```text
configs/kaggle_t4_rahfl.yaml
configs/kaggle_t4_fedprime_d2c.yaml
configs/kaggle_t4_fedprime_d2c_warmup3.yaml
```

These keep the same shared partition file and method choices, but lower
`batch_size` to `64` and `public_batch_size` to `128`. The original full configs
are still available for larger GPUs.

- Updated the default urgent Kaggle comparison to use FedPRIME-D2C with
  `d2c_warmup_rounds: 3`. The original `warmup=0` config remains available for
  the later warmup ablation.

- Added fixed shared partition indices for stricter fair comparison.
  - Config field: `data.partition_indices_path`
  - Main alpha=0.5 comparison now shares:

```text
outputs/partitions/cifar10c_alpha05_seed0_clients4_samples10000.npz
```

- Verified that RAHFL and FedPRIME-D2C load the same Non-IID client split.
- Added explanatory comments in `fedprime/methods/d2c.py` for:
  - prior debias
  - class-balanced aggregation
  - sample confidence
  - complementary KD
  - adaptive beta
  - EMA prior
  - self-gate
  - oracle prior
- Added optional FedPRIME-D2C + DCL configs:

```text
configs/fedprime_d2c_dcl_cifar10c.yaml
configs/fedprime_d2c_dcl_cifar10c_alpha01.yaml
configs/debug_fedprime_d2c_dcl_cifar10c.yaml
```

- Verified debug FedPRIME-D2C + DCL smoke run:

```text
[round 000] avg_acc=9.08 worst_acc=7.66 local_loss=16.6238 d2c_loss=0.5184
```

- Bound the local repo to GitHub SSH remote and pushed to:

```text
git@github.com:yibinlin-fl/fedprime-d2c.git
```

- Added Kaggle launcher:

```text
scripts/run_kaggle.sh
```

Current Kaggle default is intentionally **without DCL** for the main claim:

```text
RAHFL = AugMix + DCL + AsymHFL
FedPRIME-D2C = PRIME + D2C
```

Run on Kaggle:

```bash
git clone https://github.com/yibinlin-fl/fedprime-d2c.git
cd fedprime-d2c
RUN_DEBUG=1 bash scripts/run_kaggle.sh
```

This first runs `configs/debug_fedprime_d2c_cifar10c.yaml`, then runs:

```text
configs/cifar10c_rahfl.yaml
configs/fedprime_d2c_cifar10c.yaml
```

Latest commits:

```text
5973fd0 支持DCL增强版D2C并固定公平划分
734e5ca kaggle 一键启动脚本
321cabd 调整kaggle默认对比为无DCL主框架
```

New implementation work after the Kaggle launcher:

- Added `method.d2c_warmup_rounds`.
  - Default is `0`, so existing experiments are unchanged.
  - If set to `3` or `5`, the first rounds run local PRIME only and skip D2C.
- Added `method.communication`.
  - `d2c`: full D2C teacher.
  - `logit_avg`: plain public-logit averaging teacher.
- Added LogitAvg+PRIME baseline configs:

```text
configs/logitavg_prime_cifar10c.yaml
configs/logitavg_prime_cifar10c_alpha01.yaml
configs/debug_logitavg_prime_cifar10c.yaml
```

- Added underrepresented class diagnosis:

```text
scripts/diagnose_underrepresented.py
```

It loads trained checkpoints and reports per-client `head_acc`, `tail_acc`,
and `missing_acc` according to each client's private class distribution.

## Quick 5-Round Decision Criteria

Purpose:

```text
Use a short run to decide whether FedPRIME-D2C has a promising trend before
spending many Kaggle GPU hours on full 40-round experiments.
```

Do **not** judge only by the absolute accuracy at round 5. In early rounds,
both methods may still be near random or unstable. Judge by trends:

1. `avg_acc` trend:
   - Promising: FedPRIME-D2C average accuracy rises at a similar or faster rate than RAHFL.
   - Warning: FedPRIME-D2C stays flat near random accuracy while RAHFL clearly rises.

2. `worst_acc` trend:
   - Promising: FedPRIME-D2C improves the worst client or narrows the gap to RAHFL.
   - Very important because D2C is designed to help clients under Non-IID label skew.
   - Warning: average accuracy rises but `worst_acc` collapses or remains far below RAHFL.

3. Gap by round 5:
   - Acceptable: FedPRIME-D2C is close to RAHFL, for example within about 3-5 accuracy points, and still improving.
   - Strong warning: FedPRIME-D2C is more than about 8-10 points behind RAHFL and the gap is widening.

4. `d2c_loss` behavior:
   - Promising: finite, stable, not exploding to `nan` or very large values.
   - Warning: `d2c_loss` becomes `nan`, explodes, or dominates training.

5. Local loss behavior:
   - Promising: finite and generally decreasing or stable.
   - Warning: loss explodes or becomes `nan`.

6. Final interpretation:
   - If FedPRIME-D2C is slightly behind in 5 rounds but improving, continue to 40 rounds.
   - If FedPRIME-D2C is clearly flat while RAHFL improves, inspect D2C hyperparameters first:
     `beta`, `eta`, `temperature`, `lambda_d2c`, `use_sample_confidence`,
     and whether PRIME local training is learning.
   - If FedPRIME-D2C loses badly without DCL but FedPRIME-D2C + DCL performs well later,
     the likely story is that D2C is useful but local representation learning needs the DCL module.

## Resume Update - 2026-06-04

Completed in the latest continuation:

- Prepared local CIFAR data with `scripts/prepare_data.py`.
- Generated RAHFL-style CIFAR-10-C caches for rates `0`, `0.5`, and `1`.
- Confirmed `data.private_root` and `data.public_root` pass `scripts/check_environment.py`.
- Ran partition audit for `configs/fedprime_d2c_cifar10c.yaml`.
- Added a tiny debug config:

```text
configs/debug_fedprime_d2c_cifar10c.yaml
```

- Ran one debug FedPRIME-D2C smoke training successfully:

```text
[round 000] avg_acc=9.72 worst_acc=9.04 local_loss=2.4352 d2c_loss=0.8128
```

- Added FedPRIME-D2C + DCL configs and verified a debug smoke run:

```text
configs/fedprime_d2c_dcl_cifar10c.yaml
configs/fedprime_d2c_dcl_cifar10c_alpha01.yaml
configs/debug_fedprime_d2c_dcl_cifar10c.yaml
[round 000] avg_acc=9.08 worst_acc=7.66 local_loss=16.6238 d2c_loss=0.5184
```

Additional environment dependencies installed in local `pytorch` env:

```text
pandas
seaborn
scikit-learn
```

Code/config changes from this continuation:

- `.gitignore` now ignores `RAHFL-master/Dataset/cifar_10/`.
- `requirements.txt` now includes RAHFL runner dependencies.
- `configs/debug_fedprime_d2c_cifar10c.yaml` was added for local smoke tests.

## Goal

Build and evaluate **FedPRIME-D2C**, a robust heterogeneous federated learning framework for:

- model heterogeneity
- data heterogeneity / Non-IID label skew
- common corruption robustness

The main target baseline is **RAHFL**. The core paper claim should be:

> RAHFL mainly addresses unreliable or corrupted collaborators by asking which client is more reliable. FedPRIME-D2C instead addresses Non-IID prior-contaminated public logits by debiasing client logits, constructing class-balanced teachers, and applying personalized complementary KD.

## Current Repository State

The repository is initialized as a Git repo.

Recent commits:

- `5b73341` - `项目基础构造初始化提交`
- `d992cab` - `完成RAHFL+PRIME+DCL修正 表格汇总 损坏评估 多种子与断点`
- `745dde0` - `数据异构审计异构与数据自动下载`

Current working tree was clean after the last commit.

## Major Code Areas

```text
fedprime/
  augmentations/
    prime_adapter.py
    prime.py
    rand_filter.py
    diffeomorphism.py
    color_jitter.py
  data/
    loaders.py
    partition.py
    corruptions.py
  models/
    factory.py
    resnet.py
    shufflenet.py
    mobilenet_v2.py
  methods/
    d2c.py
    local_prime.py
    local_rahfl.py
    fedprime_d2c.py
    rahfl_asymhfl.py
  engine/
  utils/
configs/
scripts/
docs/
```

## Implemented Methods

### FedPRIME-D2C

Main runner:

```text
fedprime/methods/fedprime_d2c.py
```

Implemented modules:

- Local PRIME robust learning
- CE + JSD local loss
- Public logits communication
- Predictive prior estimation
- Prior logit debiasing
- Class-balanced aggregation
- Sample confidence weighting
- Personalized complementary KD
- Optional oracle prior
- Optional adaptive beta
- Optional EMA prior
- Optional self-gate
- Checkpoint loading / resume

Core D2C implementation:

```text
fedprime/methods/d2c.py
```

### RAHFL Baseline

Unified runner:

```text
fedprime/methods/rahfl_asymhfl.py
```

Modes:

```yaml
method_name: rahfl
```

Runs:

```text
AugMix + DCL + AsymHFL
```

```yaml
method_name: rahfl_prime
```

Runs:

```text
PRIME + DCL + AsymHFL
```

This is the strong baseline. It replaces AugMix-style strong augmentation with PRIME while preserving the RAHFL DCL idea and AsymHFL communication.

## PRIME Reuse

PRIME is not rewritten. The code reuses the official implementation under:

```text
PRIME-augmentations-main/
```

Thin adapters are in:

```text
fedprime/augmentations/
```

`prime_adapter.py` builds the official `GeneralizedPRIMEModule` and returns:

```text
clean + prime_aug1 + prime_aug2
```

for CE + JSD training.

## Data Format

The current training pipeline follows the RAHFL-style cached numpy format:

```text
cifar_10_c/
  train/
    random_corrupt_0.npy
    random_corrupt_0.5.npy
    random_corrupt_1.npy
    labels.npy
  test/
    random_corrupt_0.npy
    random_corrupt_0.5.npy
    random_corrupt_1.npy
    labels.npy
```

Meaning:

- `random_corrupt_0.npy`: clean CIFAR-10 cached in RAHFL format
- `random_corrupt_0.5.npy`: 50 percent random corrupted CIFAR-10
- `random_corrupt_1.npy`: 100 percent random corrupted CIFAR-10

This is not the official CIFAR-10-C folder layout. It is the data format expected by RAHFL code.

Official CIFAR-10-C usually has files like:

```text
gaussian_noise.npy
shot_noise.npy
motion_blur.npy
fog.npy
jpeg_compression.npy
labels.npy
```

Those are more useful for corruption-group evaluation.

## Current Default Experiment

Default config:

```text
configs/fedprime_d2c_cifar10c.yaml
```

Important fields:

```yaml
private_corrupt_rate: 1
test_corrupt_rate: 1
partition: dirichlet
dirichlet_alpha: 0.5
```

So by default:

- private training uses `train/random_corrupt_1.npy`
- testing uses `test/random_corrupt_1.npy`
- public data uses CIFAR-100
- client split uses Dirichlet Non-IID label skew

## Data Preparation

Implemented:

```text
scripts/prepare_data.py
```

Example:

```powershell
D:\anaconda3\Scripts\conda.exe run -n pytorch python scripts\prepare_data.py --config configs\fedprime_d2c_cifar10c.yaml --download --rates 0 0.5 1
```

This downloads CIFAR-10/CIFAR-100 through torchvision and creates RAHFL-style random corrupted CIFAR-10 files.

## Data Heterogeneity Audit

Implemented:

```text
scripts/audit_partition.py
```

Example:

```powershell
D:\anaconda3\Scripts\conda.exe run -n pytorch python scripts\audit_partition.py --config configs\fedprime_d2c_cifar10c.yaml
```

Outputs:

```text
outputs/partition_audit/<experiment_name>/
  client_class_counts.csv
  client_class_proportions.csv
  client_class_counts.png
  partition_summary.json
```

Use this to prove Non-IID label skew exists.

## Core Run Commands

Check environment:

```powershell
D:\anaconda3\Scripts\conda.exe run -n pytorch python scripts\check_environment.py --config configs\fedprime_d2c_cifar10c.yaml
```

Prepare data:

```powershell
D:\anaconda3\Scripts\conda.exe run -n pytorch python scripts\prepare_data.py --config configs\fedprime_d2c_cifar10c.yaml --download --rates 0 0.5 1
```

Audit partition:

```powershell
D:\anaconda3\Scripts\conda.exe run -n pytorch python scripts\audit_partition.py --config configs\fedprime_d2c_cifar10c.yaml
```

Run FedPRIME-D2C only:

```powershell
D:\anaconda3\Scripts\conda.exe run -n pytorch python scripts\run_experiment.py --config configs\fedprime_d2c_cifar10c.yaml
```

Run core comparison:

```powershell
D:\anaconda3\Scripts\conda.exe run -n pytorch python scripts\run_grid.py configs\cifar10c_rahfl.yaml configs\cifar10c_rahfl_prime.yaml configs\fedprime_d2c_cifar10c.yaml
```

Run multi-seed:

```powershell
D:\anaconda3\Scripts\conda.exe run -n pytorch python scripts\run_multiseed.py --config configs\fedprime_d2c_cifar10c.yaml --seeds 0 1 2
```

Summarize results:

```powershell
D:\anaconda3\Scripts\conda.exe run -n pytorch python scripts\summarize_results.py --outputs outputs
```

## Local Environment

Checked conda env:

```text
env name: pytorch
python: 3.11.13
torch: 2.8.0+cu126
CUDA available: True
GPU: NVIDIA GeForce RTX 3050 Laptop GPU
```

Installed missing PRIME dependencies:

```text
einops
opt-einsum
```

Smoke tests passed:

- D2C tensor smoke test
- PRIME augmentation smoke test
- runner import test

## Current Known Limitations

- Full training has not yet been run.
- `prepare_data.py` creates simplified random corruptions, not the full official CIFAR-10-C corruption suite.
- corruption group evaluation exists, but requires official-style per-corruption `.npy` files.
- No real result table yet because no training outputs exist.
- No paper figures generated yet except partition audit heatmap.

## Tomorrow Resume Prompt

## 2026-06-29 Current Mainline: PRAC-HFL

The current main experimental direction is PRAC-HFL. D2C and FedPRIME-PAIR are now historical diagnostic experiments rather than the main method.

```text
PRAC-HFL = RAHFL local robust training + receiver-adaptive safe communication
```

Local training follows the strong RAHFL local baseline:

```text
AugMix multi-view training + CE + JSD consistency + RAHFL DCLLoss
```

Implementation files:

```text
fedprime/methods/local_rahfl.py
fedprime/methods/prac_hfl.py
configs/kaggle_t4_prac_hfl.yaml
configs/debug_prac_hfl_cifar10c.yaml
scripts/run_kaggle_prac.sh
```

Latest pushed commits:

```text
fa108f7 实现PRAC-HFL接收端自适应通信
5e476ea 增强PRAC-HFL数值稳定性
```

Safe PRAC-HFL config:

```text
warmup_rounds: 3
risk_lambda_aug: 0.0
risk_lambda_js: 0.0
virtual_lr: 0.005
head_max_grad_norm: 1.0
train.max_grad_norm: 5.0
train.skip_nonfinite: true
```

Historical results now archived:

```text
RAHFL unified runner: final avg_acc=56.41, worst_acc=44.72, no 40-epoch pretraining.
PRIME + LogitAvg: final avg_acc≈52.10, worst_acc≈39.72.
FedPRIME-D2C: final avg_acc≈52.31, worst_acc≈39.78.
Oracle D2C: final avg_acc≈51.74, worst_acc≈39.13.
FedPRIME-PAIR: final avg_acc≈50.15, worst_acc≈39.83.
```

Current interpretation:

```text
D2C and PAIR are negative/diagnostic public-logit communication results.
PRAC-HFL has the strongest signal so far.
First PRAC run reached round 028 with avg_acc=53.86, higher than same-round RAHFL 53.21,
but worst_acc=39.52 was below same-round RAHFL 41.64, and round 029 produced NaN.
Safe PRAC-HFL must be rerun from commit 5e476ea.
```

Generated comparison deliverables:

```text
deliverables/prac_vs_rahfl_analysis/rahfl_prac_hfl_comparison.xlsx
deliverables/prac_vs_rahfl_analysis/avg_accuracy_curve.png
deliverables/prac_vs_rahfl_analysis/worst_accuracy_curve.png
deliverables/prac_vs_rahfl_analysis/prac_diagnostics.png
```

Kaggle memory:

```text
Use Python streaming launcher, not long silent %%bash cells.
Dataset input: /kaggle/input/fedprime-data
Before running PRAC-HFL, verify: git log -1 --oneline == 5e476ea 增强PRAC-HFL数值稳定性
Expected logs include [heartbeat] round xxx ... and [round xxx] avg_acc=... worst_acc=...
```

Use:

```text
读取 docs/project/PROJECT_STATE.md 和 docs/project/TODO_NEXT.md，继续推进 FedPRIME-D2C 项目。
```

## Remaining RAHFL-table baselines - 2026-08-09

The unified communication registry now includes FedDF, KT-pFL, and FCCL.
All three implement the published/released core communication mechanism through
`CommunicationContext`, use the same public batch budget, and do not read audit
or final-test labels. The matched OpenI entry is
`scripts/openi_cle_remaining_baselines_entry.py`; formal results are pending.
