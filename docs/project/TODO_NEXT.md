# TODO Next

## Immediate - Run Strict PEW Operator-LOO - 2026-08-08

The matched shared-PEW CDep-v2 experiment is complete. Candidate-minus-control
last-five was `Avg -0.1933`, `Worst -0.2280`, `WCCA -0.6000`, `CFG +0.4450`;
all four frozen gates failed. Freeze CDep-v1/v2 and freeze the final local
method as calibrated PEW+BER. Before propagating it to all large paper runs,
test whether PEW depends on seeing the four private-unseen concrete operators
in its public synthetic supervision.

Run exactly one three-arm OpenI task:

```text
dataset: openi_cle_hfl_v2_alpha05_gamma09
entry: scripts/openi_cle_pew_loo_entry.py
arguments: none
archive: cle_pew_loo_12round_seed0_outputs.tar.gz
```

It sequentially runs RAHFL, original PEW+BER, and Strict-LOO PEW+BER for 12
rounds each. Strict LOO excludes `impulse_noise`, `zoom_blur`, `fog`, and
`pixelate` from public PEW train/validation generation. The old PEW remains
unchanged and uses a separate checkpoint.

Primary Strict-LOO-minus-RAHFL last-five gates remain:

```text
Avg >= +1.5, Worst >= +1.0, WCCA >= 0, CFG <= -1.0
```

After this result:

1. If all gates pass, freeze Strict-LOO-compatible PEW+BER and remove CDep from
   the Candidate arms of the prepared communication
   factorial, cross-scenario, alpha/gamma stress, and second-dataset entries.
2. If a gate fails, do not launch the old paper-evidence queue; decide whether
   to implement Multi-label PEW + Soft-BER or reduce the paper claim.
3. Keep PEW calibration, BER, strict fit/audit roles, AsymHFL-val, seeds, and
   reporting unchanged.
4. Run focused config/unit tests and audit one resolved control/candidate pair.
5. Update the paper-evidence run guide with the revised frozen method.
6. Only then choose the next paid run; do not submit the old CDep-bearing
   candidate configs.

After propagation, the recommended next scientific run is communication
orthogonality, followed by cross-scenario seeds 1/2, alpha/gamma stress, and
the CIFAR-100-private second dataset.

The CDep lambda sensitivity is complete. Lambda 0.01, 0.05, and 0.10 all lost
to matched BER-only on last-five Avg, Worst, and WCCA. Current batch-local CDep
is not a validated additive module. Do not continue lambda-only tuning.

CDep-v2 is implemented and locally verified. The original single-arm entry is
now historical and must not be rerun as the final attribution experiment:

```text
dataset: openi_cle_hfl_v2_alpha05_gamma09
entry: scripts/openi_cle_cdep_v2_entry.py
arguments: none
expected archive: cle_cdep_v2_12round_outputs.tar.gz
```

Its automatic historical comparison is retained as provenance only. The paired
entry has now completed and failed all frozen gates. It must not be rerun or
used as a new tuning surface.

The PEW generalization implementation is now ready. The four
private-unseen seed-0 operators (`impulse_noise`, `zoom_blur`, `fog`,
`pixelate`) are absent from private fit data and the Strict variant now also
excludes them from public PEW train and validation generation. Report existing
results as `private-unseen`; the new experiment is specifically a PEW operator
LOO audit, not a claim that every generic augmentation is globally unseen.

After the local method and strict PEW protocol are frozen, run the matched
CIFAR-100-private second-dataset A/B, followed by communication orthogonality,
cross-scenario seeds 1/2, alpha/gamma stress tests, and missing
FedDF/KT-pFL/FCCL baselines.

## Historical Completed Strict 12-Round A/B Probe - 2026-08-04

This experiment is complete and retained only as historical provenance:

```text
control   = AugMix/JSD/DCL + strict AsymHFL-val
candidate = calibrated PEW/BER+CDep + the same strict AsymHFL-val
```

Do not rerun it as the current method decision. Its candidate included the now
frozen CDep-v1. Inspect `outputs/strict_pew_asymhfl_val_comparison.json` only
when historical provenance is needed.

Proceed only if the candidate-minus-control last-five result satisfies all:

```text
Avg >= +1.5
Worst >= +1.0
WCCA >= 0.0
CFG <= -1.0
```

If the gate fails, archive this combined route. Do not attribute old PEW local
gains to communication and do not tune only lambda/threshold values.

## Immediate Decision After Continuous-Witness Audit - 2026-08-03

The taxonomy-free continuous witness captured a CFG signal but failed the
pre-registered overall gate. Cancel the following work for this formulation:

```text
12-round continuous-witness local-only training
continuous-witness + strict AsymHFL
40-round OpenI/Kaggle runs
single-parameter rho/lambda tuning
```

Do not confuse the isolated CFG reduction with a validated method. Any next
candidate must explain why it should preserve worst-client performance and
must pass a new cheap matched audit before platform training.

## Immediate Decision After FedCIS Audit - 2026-08-03

FedCIS-v0 failed its frozen identifiability gate. The following tasks are
cancelled:

```text
matched one-step FedCIS Audit C
FedCIS 12-round runner
FedCIS 40-round OpenI/Kaggle experiment
rank/epsilon/loss-weight-only tuning
```

Before proposing another paid experiment, the next research task must change
the knowledge signal or the local objective, not merely retune this sensitivity
subspace. Preserve the audit implementation as a reusable negative-result and
falsification tool.

Recommended next decision session:

1. Freeze and summarize all validated positive components and failed
   communication payloads.
2. Choose exactly one new hypothesis with a distinct payload and a cheap
   matched offline causal test.
3. Define its failure gate before implementation.
4. Do not start another 12/40-round run until that offline test passes.

## Immediate - FedCIS-v0 Offline Identifiability Audit - 2026-08-03

The current candidate is FedCIS-v0 under the existing K=4 CLE-HFL v2
protocol. Previous FedCFSA K=8 work is historical and is not the next task.

Do not implement a 12/40-round runner. Work only in this order:

1. Implement a standalone audit extractor for normalized class-margin input
   gradients projected onto a fixed multiscale DCT basis.
2. Compute PSD view-mean `A` and view-difference `N` statistics on the existing
   persisted fit split.
3. Recover class subspaces with a numerically guarded generalized eigensolver.
4. Repeat over three independent AugMix seeds.
5. Compare true class-matched, class-shuffled, and equal-rank random subspaces.
6. Verify projected counterfactuals use margin descent and detach the direction.
7. Run matched one-step `base/random/shuffled/true` updates from identical
   checkpoints and optimizer states.
8. Evaluate only on `D_audit`; final-test labels must not select any subspace,
   threshold, hyperparameter, or update.

Frozen gate before a 12-round runner:

```text
true subspace separates from random and class-shuffled controls
>=60% auditable client x class targets satisfy directional checks
mean class-conditional audit loss improves over all controls
audit Avg/Worst/WCCA/seen/unseen deltas are nonnegative
audit CFG delta is nonpositive
no nonfinite values or concentrated client collapse
```

If the true subspace is indistinguishable from controls, archive FedCIS rather
than tuning only rank, epsilon, or loss weights.

Read: `docs/archive/methods/FEDCIS_FRAMEWORK_AND_OFFLINE_AUDIT_ZH.md`.

## Historical - Paused K=8 Checkpoint-Level Reliability Audit

The CPU-only FedCFSA source-redundancy sweep is complete:

```text
strong coverage = >=3 supported sources from >=3 distinct dominant environments

K=4:  mean 56.29%, worst seed 43.75% -> NO-GO
K=8:  mean 94.61%, worst seed 88.52% -> GO
K=10: mean 96.73%, worst seed 93.06% -> GO
K=20: mean/worst 100.0%             -> GO
```

This establishes only data-level source availability among auditable target
classes. It does not establish model reliability or semantic-anchor utility.

Historical proposed work, currently superseded by FedCIS-v0:

1. Extend CLE-HFL v2 preparation to a standard K=8 full-data Dirichlet
   protocol; do not reuse the K=4 `10,000 samples/client` capacity constraint.
2. Cycle the four architectures twice across eight clients.
3. Persist an 85/15 fit/audit split.
4. Train only a short strict local robust-base checkpoint probe; no
   communication and no final-test routing.
5. Re-run robust-frontier estimation over three augmentation seeds.
6. Require at least three stable reliable sources for a substantial fraction
   of auditable receiver-class targets.
7. Only if this passes, implement synthetic-anchor A/B/C/D/E one-step audits.

Do not run RAHFL, FedCFSA anchors, or a 40-round experiment yet.

Read:

```text
deliverables/fedcfsa_source_redundancy_audit_20260727/AUDIT_REPORT_ZH.md
```

## Historical - Paused FedCFSA Source-Redundancy Route

The FedCFSA coverage audit completed before image condensation was implemented.

```text
7 stable routes / 6 receiver-class targets
only 1 target has two stable sources
5/7 routes have no other stable source to validate the generator
2/7 routes have only one stable validator
0/7 routes have two independent stable validators
```

Therefore the current four-client cross-falsification formulation is NO-GO.
The later CPU sweep shows conditional data-level feasibility from K=8, but
checkpoint-level reliability remains unverified. Do not generate anchors and
do not run FedCFSA training yet.

The next research decision is one of:

1. Expand CLE-HFL v2 to 8--10 clients and require at least three independent
   source candidates for a communicated class.
2. Keep four clients but downgrade cross-falsification to one-peer validation,
   accepting a weaker mechanism and novelty claim.
3. Introduce an external semantic verifier, explicitly accepting the new
   pretrained-model assumption.
4. Archive FedCFSA and design a semantic payload that does not require
   source redundancy.

Read:

```text
deliverables/fedcfsa_coverage_audit_20260727/FEDCFSA_COVERAGE_AUDIT_ZH.md
```

## Immediate - High-Confidence Frontier One-Step Audit

The taxonomy-free robust-frontier audit is complete.

```text
direct positive-source precision: seen 52.94%, unseen 52.94% -> NO-GO
top-quartile all-view precision across augmentation seeds:
  seen   77.78% / 88.89% / 88.89%
  unseen 88.89% / 100.00% / 88.89%
stable routes common to all three seeds: 7
```

Do not implement the original full-coverage/global-median FedRIFT and do not
run 40 rounds.

Next work, in order:

1. Reuse the persisted CLE-HFL v2 fit/audit split.
2. Freeze source and receiver backbones; update only the receiver classifier
   head.
3. Use only the seven routes stable across all three augmentation seeds.
4. Apply a class-pair margin lower-bound loss on receiver fit samples.
5. Evaluate CE, class accuracy, and negative transfer on receiver audit data.
6. Never use final-test labels for source selection, thresholds, or acceptance.
7. Proceed to a 12-round method only if most stable routes are nonnegative and
   aggregate Worst/CFG proxies do not regress.

Read:

```text
deliverables/robust_frontier_audit_20260726/ROBUST_FRONTIER_AUDIT_ZH.md
```

Status: completed on 2026-07-27 with a matched CE-only control.

```text
7/7 routes slightly improve target-class audit loss,
but mean seen/unseen accuracy increment is 0.0000/+0.0357,
and mean all-audit accuracy increment is -0.0095.
```

Direct boundary transfer is rejected. Do not run it for 12 or 40 rounds.

Next theoretical decision:

1. Keep the robust frontier only as a taxonomy-free, high-confidence
   reliability/abstention gate.
2. Define a separate sample-level semantic payload that is implementable under
   heterogeneous FL and does not expose private receiver images.
3. Before a runner exists, test that payload with the same matched one-step
   candidate-versus-control audit.
4. If no such payload can be justified, archive the frontier route rather than
   returning to threshold tuning.

## Immediate - Strict RAHFL-val Fairness Repair

The CLE-HFL v2 three-way 12-round probe is complete. FedFalsify v0.3 failed the
frozen gate against its strict fit-only control:

```text
final delta:     Avg +0.3183, Worst -0.4067, WCCA +0.250, CFG +1.600
last-five delta: Avg +0.1180, Worst -0.4373, WCCA +0.850, CFG +2.185
```

Do not run FedFalsify for 40 rounds and do not tune only `kappa` or
`lambda_cmt`.

The next required work is:

1. Add a strict RAHFL-val mode that loads the same persisted FedFalsify
   fit/audit split.
2. Train every RAHFL client only on `D_fit`.
3. Compute AsymHFL route accuracy only on `D_audit`.
4. Keep final test labels out of all routing and training decisions.
5. Run only this strict RAHFL-val for 12 rounds under the frozen CLE-HFL v2
   setup.
6. Reuse the completed strict control and FedFalsify results offline.
7. Only after the fair comparison, decide whether to redesign or archive the
   FedFalsify communication route.

Reason: the completed RAHFL run used 100% local data and final-test accuracy
routing, while control/FedFalsify used about 85% fit data and held out 15% for
audit. Its current numerical lead is not a formal fair baseline.

User decision: pause this run because it does not answer the more important
method-design question. Do not implement or launch strict RAHFL-val until the
external theoretical review is complete.

Immediate non-compute task:

1. Review `docs/archive/methods/FEDFALSIFY_LATEST_EXTERNAL_AI_DISCUSSION_BRIEF_ZH.md`.
2. Discuss whether CLE-HFL v2 should remain the paper problem.
3. Decide whether to retain only the fit/audit falsification principle or
   archive FedFalsify entirely.
4. Freeze all training runs until a new method has a clearly stated information
   source, mechanism, and one-probe Go/No-Go criterion.

## Immediate - FedFalsify v0.3 Candidate-Only Gate

Implementation and RTX 3050 smoke are complete.

Next:

1. Push the v0.3 implementation when requested.
2. On OpenI, use the existing gamma=0.9 CLE-HFL dataset.
3. Start `scripts/openi_fedfalsify_v03_entry.py` with no arguments.
4. Download `fedfalsify_v03_probe_outputs.tar.gz`.
5. Compare offline against the stored strict fit-only control.
6. Require last-five `Avg >= control + 1.0`, nonnegative Worst/WCCA deltas, and
   nonpositive CFG delta.
7. If the gate fails, archive FedFalsify communication instead of tuning only
   thresholds or `lambda_cmt`. If it passes, prepare a matching 40-round run.

Status: completed. Avg/Worst/WCCA passed; CFG failed.

Immediate diagnostic work:

1. Do not run v0.3 for 40 rounds yet.
2. Extend CLE evaluation output to persist per-round
   `client x class x corruption` correct/total/accuracy matrices.
3. Keep training behavior and all v0.3 hyperparameters frozen.
4. Use the richer diagnostics to identify which class-corruption cells create
   the round-7 CFG spike and whether they coincide with zero-UCB route churn.
5. Decide whether a temporal route-consistency mechanism is theoretically
   justified. Do not implement one based only on the aggregate CFG curve.

## Completed - FedFalsify v0.2 Strict Gate

1. On OpenI, keep using dataset `openi_cle_rahfl_diagnostic`.
2. Start `scripts/openi_fedfalsify_entry.py` with no arguments.
3. Download `fedfalsify_probe_outputs.tar.gz` after completion.
4. Compare candidate versus strict fit-only control using last-five means:
   Avg must improve, Worst/WCCA must not decline, CFG must not increase.
5. Only if all four gates pass, implement matching 40-round runs and repeat
   seeds. Otherwise archive this communication path instead of tuning only
   `lambda_cmt`.

Status: completed. The gate failed on CFG.

Next implementation:

1. Add paired correctness UCB to each receiver/class/source audit.
2. Reject only sources with `paired_advantage + kappa * SE < 0`.
3. Run TAU Top-1 among surviving sources; abstain when none survive.
4. Keep the same split, warmup, CMT, data, models, and optimizer.
5. Run a candidate-only 12-round v0.3 probe and compare offline with the stored
   strict control. Do not run 40 rounds yet.

## 2026-07-23 Decision: Revise FedFalsify Before Training

FedFalsify v0.1 offline audit is complete. Do not launch its original hard
`FRA AND TAU` gate for 40 rounds.

Next steps, in order:

1. Define FedFalsify v0.2 as receiver-side TAU-first top-1 source selection.
2. Keep FRA as a soft ranking prior/tie-breaker, not a mandatory positive gate.
3. Source-ranking audit is complete: TAU top-1 has 100% receiver-class coverage
   and 85.7%-94.3% positive one-step precision across the three gammas.
4. Add a compute-efficient head-only or last-block TAU option and measure its
   agreement with full-model TAU.
5. Only if the low-cost TAU preserves the signal, implement a 12-round probe
   with a real pre-training `fit/audit` split.
6. Use identical `fit` data for RAHFL and FedFalsify controls; never route using
   final test labels.

Completed artifacts:

```text
docs/experiments/archive/FEDFALSIFY_AUDIT_GUIDE_ZH.md
deliverables/fedfalsify_offline_audit/
outputs/fedfalsify_audit/
```

## 2026-07-22 Decision: Stop EBST-v2 and Redesign Communication

The calibrated local-only attribution control is complete:

```text
local final:       42.8469/36.2300/WCCA 19.775/CFG 6.5725
EBST-v2 final:     42.6331/35.2975/WCCA 20.675/CFG 7.2900

local last-five:   40.4278/36.2890/WCCA 17.965/CFG 6.427
EBST-v2 last-five: 40.4526/35.9870/WCCA 17.400/CFG 6.666
```

EBST-v2 fails the frozen survival rule. Do not run 40 rounds, do not rerun the
same probe, and do not spend a run on lambda-only tuning.

Next work, in order:

```text
1. Freeze calibrated PEW + BER+CDep as the validated local mechanism.
2. Archive EBST-v2 as a negative communication result.
3. Define a taxonomy-free communication redesign using continuous environment
   embeddings or another recipient/class-specific trust mechanism.
4. Prove on stored logits/diagnostics that the proposed teacher signal has
   positive recipient-class utility before implementing another paid run.
5. Only after a 12-round matched probe passes Avg/Worst/WCCA/CFG gates, schedule
   a 40-round comparison against RAHFL.
```

## 2026-07-21 Next Required Control: Calibrated PEW Local-only

The calibrated PEW + EBST-v2 combination reached:

```text
final:    42.6331/35.2975/WCCA 20.675/CFG 7.290
last-five 40.4526/35.9870/WCCA 17.400/CFG 6.666
```

Do not run 40 rounds yet. The next experiment must use the same new PEW best
checkpoint selection and automatic threshold, but set communication to `none`.
It should match the combination config in all other data/model/train settings.

Implementation is complete. Run:

```text
startup: scripts/openi_fedease_entry.py
argument: --mode=pew_calibrated_local_probe
dataset: openi_cle_rahfl_diagnostic
```

Decision rule:

```text
EBST-v2 survives only if the combination improves last-five Avg by a meaningful
margin (target >= +0.5) without reducing Worst/WCCA or increasing CFG versus the
new calibrated local-only control. If not, attribute the current gain to PEW
calibration and stop the hard-taxonomy EBST communication route.
```

## 2026-07-21 Run Tonight: Calibrated PEW + EBST-v2

Implementation and local verification are complete. Run exactly:

```text
startup: scripts/openi_fedease_entry.py
argument: --mode=pew_ebst_v2_probe
dataset: openi_cle_rahfl_diagnostic
```

Compare the result to stored learned PEW local:

```text
Avg=40.3694, Worst=35.4225, WCCA=13.925, CFG=6.370
```

Decision rule:

```text
Go only if final and last-five Avg do not regress, Worst/WCCA improve, and CFG
does not increase. Otherwise stop the hard-taxonomy PEW+EBST route and implement
continuous environment embeddings + Soft-BER + Soft-EBST. Do not run 40 rounds.
```

## 2026-07-20 Next Action: Calibrated PEW + EBST-v2 Combination Probe

The learned PEW local probe is complete:

```text
PEW BER+CDep final = 40.3694 / 35.4225 / WCCA 13.925 / CFG 6.370
gate               = 40.5    / 34.0    / WCCA 13.0   / CFG 7.0
```

It passed three gates and missed final Avg by `0.1306`. It retains most Oracle
Worst/WCCA/CFG performance, so do not discard the learned environment route.

Before another OpenI run:

```text
1. save/select the PEW checkpoint using public validation rather than last epoch;
2. calibrate the unknown threshold on the public validation split;
3. add a 12-round PEW + BER+CDep + EBST-v2 combination probe;
4. preserve the current local and communication budgets and all frozen settings;
5. do not rerun RAHFL, Oracle local, or run 40-round full mode yet.
```

Combination Go/No-Go should require the deployable candidate to exceed the
matching 12-round RAHFL trajectory broadly and avoid regression relative to the
stored PEW local result, especially on Avg and Worst.

## 2026-07-20 EBST-v2 Decision: Safety Fixed, Full Run Still Blocked

The corrective probe is complete:

```text
BER+CDep local:       41.6206 / 35.5175 / WCCA 14.000 / CFG 6.155
BER+CDep+EBST-v2:    41.9469 / 36.2275 / WCCA 14.700 / CFG 5.190
final delta:          +0.3263 / +0.7100 / +0.700 / -0.965
last-five avg delta:  -0.1648
```

Decision:

```text
Do not run --mode=full or PEW+EBST-v2 yet.
Do not count the final-round +0.326 avg as a stable communication gain.
Keep EBST-v2 as evidence that pair-qualified LOO transfer and class-wise SCP
remove the catastrophic client-2 regression seen in legacy EBST.
```

Before another paid experiment:

```text
1. design recipient-class acceptance using a local calibration/trust-region test;
2. require that a class-row communication update does not increase its local
   class-conditional risk beyond a small tolerance;
3. keep pair-qualified LOO teachers and class-wise SCP unchanged;
4. unit-test the acceptance path and run only one matching 12-round probe.
```

## 2026-07-20 EBST Probe Decision: Stop Full FedEASE Expansion

The Oracle EBST communication probe is complete:

```text
BER+CDep local:       41.6206 / 35.5175 / WCCA 14.000 / CFG 6.155
BER+CDep+EBST+SCP:   38.7038 / 34.7225 / WCCA 15.325 / CFG 6.415
delta:                -2.9169 / -0.7950 / +1.325 / +0.260
```

Decision:

```text
Do not run --mode=full.
Do not promote current EBST as the communication contribution.
Do not spend a run only tuning EBST lambda.
Keep Oracle BER+CDep as the validated positive local mechanism.
```

Before another communication run, redesign and unit-test these two missing safeguards:

```text
1. relation source eligibility must require trustworthy support for both classes
   in each pair, not only support for the true class;
2. transfer safety must be recipient/class specific, because the current global
   classifier-head SCP hides severe per-class conflicts.
```

The next compute run is intentionally undecided. A PEW-only local probe answers
whether learned environments preserve the Oracle local gain, but it does not fix
the communication failure. Run it only if that deployability evidence is currently
more valuable than redesigning communication.

## 2026-07-20 Next Run: Oracle EBST Communication Probe

The Oracle local mechanism gate passed:

```text
control final:       37.5813 / 30.1100 / WCCA 13.70 / CFG 10.855
BER+CDep final:      41.6206 / 35.5175 / WCCA 14.00 / CFG  6.155
candidate delta:     +4.0394 / +5.4075 / +0.30 / -4.70
```

Run next on the existing `openi_cle_rahfl_diagnostic` dataset:

```text
startup: scripts/openi_fedease_entry.py
argument: --mode=ebst_probe
config:  configs/openi_v100_fedease_ebst_probe.yaml
```

This adds only `EBST + stability gate + SCP` to Oracle BER+CDep. It keeps PEW out
of the experiment so communication can be judged without environment-estimation error.

This experiment is complete and failed the gate: average and worst accuracy fell,
and CFG increased slightly. The historical instructions above are retained only
to document the decision process.

## 2026-07-19 Immediate Mainline: FedEASE v2.1 Formal Probe Sequence

Current implementation scope:

```text
Complete switchable candidate implemented:
Oracle/PEW + BER + CDep + EBST + stability gate + SCP
+ clean/same/random/swapped/unseen evaluation
```

Implementation completion is not evidence of effectiveness. Do not run the 40-round full method first.

First decision experiment:

```text
OpenI entry: scripts/openi_fedease_entry.py
runtime parameter: --mode=oracle_probe
A. configs/openi_v100_fedease_oracle_control_probe.yaml
D. configs/openi_v100_fedease_oracle_ber_cdep_probe.yaml
```

The two formal configs already match on:

```text
alpha=0.5
gamma=0.9
seed=0
models=ResNet10/ResNet12/ShuffleNet/MobileNetV2
optimizer
rounds
batch budget
test split
```

Decision metrics:

```text
Avg and Worst: must not collapse
WCCA: higher is better
CFG: lower is better
CDep valid classes: must remain nonzero often enough
CDep covariance: should decline over training without accuracy collapse
```

If the joint method is positive, run `--mode=pew_probe`, then `--mode=ebst_probe`.
If it is negative, stop formal FedEASE expansion even though later modules are already coded.

Current data can be reused directly:

```text
local_runs/cle_hfl_prepared/cle_hfl_prepared_alpha05_gamma09_seed0/
```

OpenI should upload the complete prepared package:

```text
local_runs/cle_hfl_prepared/fedease_cle_prepared_alpha05_gamma09_seed0.tar.gz
```

Implemented formal sequence:

```text
1. --mode=oracle_probe  # 12 + 12 rounds, control vs BER+CDep
2. --mode=pew_probe     # 5-epoch PEW + 12-round learned BER+CDep
3. --mode=ebst_probe    # 12-round Oracle BER+CDep+EBST+gate+SCP
4. --mode=full          # 20-epoch PEW + 40-round complete method
```

Read `docs/experiments/archive/FEDEASE_OPENI_RUN_GUIDE_ZH.md` before creating the OpenI task.

## 2026-07-11 Immediate Run: Matching RAHFL vs FedCLEAR-PCCD Probe

FedCLEAR v0.1 (`CCRE + IRD`) is frozen as a negative result. Do not tune it or
run more seeds.

Latest method:

```text
FedCLEAR-PCCD = fixed AugMix/JSD/DCL local base + PCCD communication
```

First upload the new public package alongside the existing CLE gamma=0.9 private package:

```text
local_runs/cle_hfl_indomain_public/
  cle_hfl_indomain_public_alpha05_gamma09_seed0.tar.gz
```

OpenI startup:

```text
scripts/openi_fedclear_pccd_entry.py
```

Run the two matching tasks, preferably separately so each output is recoverable:

```text
--method rahfl
--method pccd
```

Configs:

```text
configs/openi_v100_rahfl_cle_indomain_probe.yaml
configs/openi_v100_fedclear_pccd_probe.yaml
```

Both runs use the exact same private data, in-domain unlabeled public 5k,
models, optimizer, local module, batch budgets, seed, and 12 rounds. The only
method difference is AsymHFL vs PCCD.

After both outputs are available, run or let the entry run:

```text
scripts/analyze_pccd_probe.py
```

Only run 40 rounds if all tail-mean gates pass:

```text
avg_acc delta   >= +1.5
worst_acc delta >= +1.0
WCCA delta      >= +4.0
CFG delta       <= -1.5
```

## 2026-07-10 Immediate Run: FedCLEAR 12-Round OpenI Probe

FedCLEAR v0.1 is implemented and the local two-round smoke test passed.

Do not spend the remaining OpenI budget on a 40-round run first. Run:

```text
startup file:
  scripts/openi_fedclear_entry.py

runtime parameter:
  --mode probe

config selected by the entry:
  configs/openi_v100_fedclear_cle_gamma09_probe.yaml
```

The probe setting is:

```text
alpha=0.5
gamma=0.9
seed=0
rounds=12
local_epochs=1
batch_size=64
public_batches_per_round=4
warmup_rounds=3
full counterfactual test evaluation
```

Use the already uploaded OpenI dataset containing:

```text
cle_hfl_prepared_alpha05_gamma09_seed0.tar.gz
```

Primary decision metrics:

```text
avg_acc
worst_acc
WCCA          # higher is better
CFG           # lower is better
ccre_loss
ccre_worst_view_risk
ird_loss
ird_anchor_disagreement
ird_worst_view_kl
```

Decision after 12 rounds:

```text
Positive mechanism signal:
  FedCLEAR WCCA is clearly above the matching RAHFL round-11 value,
  CFG is lower, avg/worst trends are not collapsing, and all diagnostics are finite.

Then:
  run --mode full for 40 rounds.

Negative mechanism signal:
  WCCA does not improve, CFG does not fall, or anchor disagreement stays high.

Then:
  do not burn points on the 40-round run; inspect CCRE/IRD diagnostics first.
```

Current RAHFL 40-round gamma=0.9 reference:

```text
avg_acc=46.72
worst_acc=38.16
WCCA=19.32
CFG=10.91
```

The fair early-round comparison must use the archived RAHFL metrics at the same
round, not compare FedCLEAR round 11 directly with RAHFL round 39.

The OpenI entry performs this comparison automatically. Same-round reference:

```text
RAHFL round 11:
  avg_acc=37.4575
  worst_acc=30.6950
  WCCA=8.1500
  CFG=9.7250

RAHFL rounds 9-11 mean:
  avg_acc=36.6488
  worst_acc=30.4125
  WCCA=8.1833
  CFG=10.6558
```

## 2026-07-10 CLE-HFL Status: Failure Mode Initially Validated

The first RAHFL-only CLE-HFL diagnostic has finished.

Result archive and analysis:

```text
outputs/cle_rahfl_diagnostic_outputs.tar.gz
outputs/cle_rahfl_diagnostic_analysis/
```

Fixed setting:

```text
alpha=0.5
seed=0
clients=4
samples_per_client=10000
baseline=RAHFL / AugMix-JSD + DCL + AsymHFL
```

CLE-HFL signal:

```text
gamma=0.0:
  avg_acc=52.17, worst_acc=44.17, WCCA=35.35, CFG=2.54

gamma=0.6:
  avg_acc=50.82, worst_acc=42.83, WCCA=25.88, CFG=5.91

gamma=0.9:
  avg_acc=46.72, worst_acc=38.16, WCCA=19.32, CFG=10.91
```

Interpretation:

```text
As corruption-label entanglement gets stronger, RAHFL gets worse:
  avg_acc   -5.45
  worst_acc -6.02
  WCCA      -16.02
  CFG       +8.37

This initially validates CLE-HFL as a failure-mode benchmark: RAHFL has hidden
counterfactual class-corruption weakness under corruption-label shortcut.
```

Immediate research step is now implemented and awaiting the probe result:

```text
Test FedCLEAR for gamma=0.9 and determine whether it:
  1. improves WCCA,
  2. reduces CFG,
  3. keeps avg_acc/worst_acc competitive with or better than RAHFL.
```

Experiments after a positive probe:

```text
1. Run the 40-round FedCLEAR gamma=0.9 full config.
2. Run SARA + AsymHFL under CLE-HFL gamma=0.9 as a method baseline.
3. Add at least one matched validation seed after a positive full result.
4. Later, rerun gamma=0.0 and gamma=0.9 for FedCLEAR to show the
   method specifically addresses entanglement, not only generic robustness.
```

Implemented method:

```text
FedCLEAR:
  CCRE class-conditional counterfactual worst-risk learning
  IRD invariant-anchor / shortcut-residual heterogeneous distillation
```

## 2026-07-08 Immediate Mainline: FedSARA-CS on Corruption-Skew

New immediate direction:

```text
Run RAHFL-CS vs FedSARA-CS under the new corruption-skew protocol.
```

Why this matters:

```text
The old random-corruption setting made the story too close to RAHFL.
The new protocol introduces corruption-skew across clients, so the paper can
target model heterogeneity + label-skew + corruption-skew together.
```

Prepared dataset:

```text
local_runs/fedsara_cs_prepared/fedsara_cs_prepared_alpha05_rho07_seed0.tar.gz
```

Upload this tarball to OpenI / 启智 as a dataset, suggested name:

```text
fedsara-cs-alpha05-rho07-seed0
```

Configs to run:

```text
configs/openi_v100_rahfl_cs_alpha05_rho07.yaml
configs/openi_v100_fedsara_cs_alpha05_rho07.yaml
```

Both configs use:

```text
pretrain_epochs: 40
rounds: 40
```

Debug configs already passed locally:

```text
configs/debug_rahfl_cs.yaml
configs/debug_fedsara_cs.yaml
```

OpenI launcher:

```text
scripts/run_openi_fedsara_cs.sh
```

Detailed run guide:

```text
docs/experiments/archive/FEDSARA_CS_SCENARIO_OPENI_GUIDE_ZH.md
```

Primary metrics to inspect:

```text
avg_acc
worst_acc
worst_group_acc
worst_client_group_acc
```

Decision rule:

```text
If FedSARA-CS beats RAHFL-CS on avg_acc and clearly improves worst_group_acc or
worst_client_group_acc, this becomes the new paper story.

If FedSARA-CS only ties RAHFL-CS, inspect corruption_group_acc.csv and
client_group_acc.csv to see whether it improves specific corruption-skew
failure modes before changing the method again.
```

## 2026-07-05 Immediate Mainline: SARA + AsymHFL

Current best single run:

```text
SARA + AsymHFL
config: configs/kaggle_t4_sara_rahfl.yaml
archive: outputs/sara_rahfl_results.tar.gz
alpha=0.5, seed=0, corrupt_rate=1, rounds=40
final avg_acc   = 57.83
final worst_acc = 46.59
```

New alpha=0.5 seed validation archives:

```text
outputs/rahfl_seed1_results.tar.gz
outputs/sara_rahfl_seed12_results.tar.gz
```

Completed alpha=0.5 results:

```text
RAHFL seed0:          56.41   / 44.72
RAHFL seed1:          56.645  / 45.29

SARA + AsymHFL seed0: 57.83   / 46.59
SARA + AsymHFL seed1: 57.2975 / 46.23
SARA + AsymHFL seed2: 58.0025 / 45.90

Seed1 paired gap:
  SARA - RAHFL = +0.6525 avg_acc, +0.94 worst_acc

SARA seeds0/1/2 mean final:
  avg_acc   = 57.71
  worst_acc = 46.24
```

Interpretation:

```text
SARA local-only is not strong. It hurts weak-client performance.
SARA + AsymHFL remains strong and now has a positive seed=1 matched comparison
against RAHFL on both final average and final worst-client accuracy.

Important caveat:
  The archived alpha=0.5 partition files named seed0/seed1/seed2 have identical
  hashes and identical client_class_counts. Treat current seed validation as
  training/randomness stability on one fixed partition, not as cross-partition
  validation.

2026-07-06 fix:
  partition_private_data no longer calls RAHFL-master/Dataset/sampling.py, whose
  import-time seed=0 reset caused newly generated seed1/seed2 partition files to
  duplicate seed0. New partition generation now uses config.seed explicitly.
  Historical archives remain unchanged; regenerate partition packs for true
  cross-partition experiments.

Do not replace AsymHFL yet. The next priority is to verify whether this gain is
stable across Non-IID strengths and to complete the missing RAHFL seed=2 control.
```

Next experiments:

```text
1. Run RAHFL seed=2 at alpha=0.5 for the missing matched control.
2. Generate or verify genuinely distinct alpha=0.5 partitions if needed for
   paper-level cross-partition multi-seed reporting.
3. Run SARA + AsymHFL at alpha=0.3 and alpha=0.1.
4. Run SARA + AsymHFL at alpha=1.0 to check non-extreme Non-IID.
5. Then decide whether a communication module replacement is necessary.
```

Matching RAHFL seed=1/2 controls are now prepared:

```text
configs/kaggle_t4_rahfl_seed1.yaml
configs/kaggle_t4_rahfl_seed2.yaml
scripts/run_kaggle_rahfl_seed12.sh
```

Alpha=0.1 paired comparison is now prepared:

```text
configs/kaggle_t4_rahfl_alpha01.yaml
configs/kaggle_t4_sara_rahfl_alpha01.yaml
scripts/run_kaggle_sara_vs_rahfl_alpha01.sh
```

The alpha=0.1 launcher also writes alpha=0.3 partition files into outputs via a
partition-only audit. It does not train alpha=0.3 in that run.

Use the existing partition pack:

```text
/kaggle/input/sara-partitions-alpha01-alpha03
```

Or leave `PARTITION_SOURCE` empty to generate alpha=0.1 and alpha=0.3 seed0
partition files inside the result archive.

2026-07-06 rerun note:

```text
First alpha=0.1 attempt crashed in RAHFL at round 007 due to non-finite gradient.
Rerun from the latest commit where configs/kaggle_t4_rahfl_alpha01.yaml includes:
  max_grad_norm: 5.0
  skip_nonfinite: true
  local_log_interval: 50
```

Alpha=0.1 result received - 2026-07-06:

```text
Archive:
  outputs/sara_vs_rahfl_alpha01_results.tar.gz

RAHFL final/best:
  35.6825 / 29.3300

SARA final:
  35.9625 / 29.1000

SARA gap:
  final avg  +0.28
  final worst -0.23
```

Conclusion:

```text
Alpha=0.1 is not a big win. It is basically a tie.
Do not claim SARA becomes stronger as Non-IID becomes extreme based on current
results. The next useful check is alpha=1.0, then decide whether a stronger
method change is required.
```

Last-shot method result - 2026-07-07:

```text
SARA + receiver-side class-aware residual AsymHFL
```

Run:

```text
configs/kaggle_t4_sara_residual_rahfl.yaml
scripts/run_kaggle_sara_residual_alpha05.sh
```

Why alpha=0.5 first:

```text
This is the known setting where SARA already beats RAHFL.
If residual cannot improve or at least preserve this setting, do not spend
remaining compute on alpha=0.1.
```

Decision:

```text
Observed result:
  final avg/worst = 57.655 / 46.54

Compared with RAHFL:
  +1.245 avg_acc, +1.82 worst_acc

Compared with SARA + AsymHFL:
  -0.17 avg_acc, -0.05 worst_acc

Conclusion:
  Residual AsymHFL is not the new mainline. It beats RAHFL but does not beat the
  simpler SARA + AsymHFL. Keep it as a diagnostic/optional fairness variant.
```

New communication test - 2026-07-07:

```text
SARA + CCAD
CCAD = Corruption-Consistent Asymmetric Distillation
```

Run:

```text
configs/kaggle_t4_sara_ccad.yaml
scripts/run_kaggle_sara_ccad_alpha05.sh
```

Why:

```text
Instead of class-count reweighting, CCAD uses public-sample corruption
consistency to calibrate AsymHFL communication at the sample level. It is a
stronger communication-motivation candidate than SARA residual while preserving
the stable AsymHFL route.
```

Decision after alpha=0.5 seed=0:

```text
If CCAD >= SARA + AsymHFL:
  promote CCAD as the main communication innovation and then test alpha=0.3/0.1/1.0.

If CCAD beats RAHFL but not SARA + AsymHFL:
  keep it as a communication diagnostic; the paper mainline remains FedSARA.

If CCAD underperforms RAHFL:
  stop public-logit communication redesign and focus on consolidating FedSARA.
```

Recommended order now:

```text
1. Run only the missing RAHFL seed=2 control, or adapt the launcher to skip
   already completed seed=1.
2. Do not claim cross-partition robustness from the current alpha=0.5 seed pack.
3. Prioritize alpha=0.3/0.1/1.0 SARA runs before redesigning communication.
4. Report seed-matched mean/std once RAHFL seed=2 is available.
```

Current status - 2026-07-05:

```text
Done:
  - SARA + AsymHFL seed=1, alpha=0.5
  - SARA + AsymHFL seed=2, alpha=0.5
  - RAHFL seed=1, alpha=0.5

Pending:
  - RAHFL seed=2, alpha=0.5
  - alpha=0.1/1.0 validation
  - partition audit/generation fix if distinct seed partitions are required
```

Alpha=0.3 validation result - 2026-07-06:

```text
Archive:
  outputs/sara_vs_rahfl_alpha03_results.tar.gz

RAHFL alpha=0.3:
  final avg/worst = 45.8425 / 41.9200
  best  avg/worst = 46.3825 / 43.1300

SARA + AsymHFL alpha=0.3:
  final avg/worst = 46.7325 / 42.7700
  best  avg/worst = 47.0825 / 44.1100

Gap:
  final +0.89 avg, +0.85 worst
  best  +0.70 avg, +0.98 worst

Trend:
  SARA wins 36/40 rounds on avg_acc and 36/40 rounds on worst_acc.
  Last-10-round mean gap is +0.6942 avg and +0.5270 worst.
```

Interpretation:

```text
Positive but modest. This supports SARA's robustness at alpha=0.3, but does not
yet prove a large severe-Non-IID advantage. Continue alpha=0.1 and alpha=1.0
before changing the method.
```

Prepared alpha partition pack:

```text
local_runs/sara_partitions_alpha01_alpha03
local_runs/sara_partitions_alpha01_alpha03.tar.gz

Contains:
  alpha=0.1 seeds 0/1/2
  alpha=0.3 seeds 0/1/2

Suggested Kaggle dataset name:
  sara-partitions-alpha01-alpha03

local_runs/sara_partitions_alpha03_alpha10
local_runs/sara_partitions_alpha03_alpha10.tar.gz

Contains:
  alpha=0.3 seeds 0/1/2
  alpha=1.0 seeds 0/1/2

Suggested Kaggle dataset name:
  sara-partitions-alpha03-alpha10
```

When running alpha=0.3/1.0 on Kaggle, mount both:

```text
/kaggle/input/fedprime-data
/kaggle/input/sara-partitions-alpha03-alpha10
```

When running alpha=0.3/0.1 on Kaggle, mount both:

```text
/kaggle/input/fedprime-data
/kaggle/input/sara-partitions-alpha01-alpha03
```

## 当前工作方向 - 2026-06-30

### 总方向

当前项目不要再定位成“简单替换 RAHFL 的通信模块”。

更合适的研究方向是：

```text
面向数据损坏 + 模型异构 + 数据异构的 Non-IID-aware 鲁棒异构联邦学习框架
```

核心问题不是“把 RAHFL 的粗粒度通信改成细粒度通信”这么简单，而是：

```text
RAHFL 主要解决了模型异构和数据损坏，
但在 label-skew Non-IID 场景下，
它的本地 DCL 和客户端级 AsymHFL 通信都没有充分考虑类别分布偏斜。
```

因此后续工作应围绕两条主线展开：

```text
1. Non-IID-aware robust local representation learning
   改进 AugMix + DCL，使其在类别不均衡、tail class、weak client 下更稳。

2. Receiver-safe heterogeneous communication
   改进 public-logit 通信，使客户端只吸收对自己本地分布真正有益的外部知识。
```

论文动机应表述为：

```text
RAHFL 使用强本地鲁棒增强和客户端级非对称通信，
但整体客户端准确率在 label-skew Non-IID 下不是可靠的知识可迁移性指标。
高准确率客户端可能只擅长自己的 head classes，
并不一定能为其他客户端的 tail/missing classes 提供可靠知识。

因此，我们研究如何在数据损坏和模型异构同时存在时，
进行 Non-IID-aware 的鲁棒表征学习和安全知识迁移。
```

### 当前实验事实

已有结果：

```text
RAHFL public4:
  final avg_acc   = 56.41
  final worst_acc = 44.72

PRAC-HFL public1:
  final avg_acc   = 54.63
  final worst_acc = 41.88
  best avg_acc    = 55.53

PRAC-HFL public4:
  final avg_acc   = 52.96
  final worst_acc = 43.27

AugMix + DCL local-only:
  final avg_acc   = 56.11
  final worst_acc = 44.23
  best avg_acc    = 56.94 at round 38
  best worst_acc  = 44.23 at round 39
```

当前结论：

```text
local-only 最终 avg_acc=56.11，几乎追平 RAHFL final 56.41；
local-only best avg_acc=56.94，已经超过 RAHFL final 56.41；
local-only final 同时高于 PRAC public1 和 PRAC public4。

这说明当前性能主要来自 RAHFL-style AugMix + DCL 本地鲁棒学习，
而当前 PRAC 通信没有提供稳定正增益，甚至可能引入负迁移。
后续不应继续盲目微调 PRAC 超参数。
```

### 立即工作重点

当前第一优先级不再是继续修 PRAC 通信，而是设计：

```text
Non-IID-aware robust DCL / local representation learning
```

可优先考虑：

```text
1. class-balanced DCL
   防止 head class 在 DCL 中支配特征空间。

2. tail-aware supervised contrastive learning
   提升少样本类和 weak client 的表征质量。

3. corruption-view reliability weighting
   对过强、语义可能被破坏的增强视图降低对比拉近强度。

4. client-adaptive contrastive loss strength
   根据客户端类别偏斜程度调整 DCL 权重。

5. communication as secondary
   只有在本地 Non-IID-aware DCL 稳定后，再考虑轻量安全通信。
```

PRAC 可以作为历史探索保留，但不能作为当前论文主线。

### 已实现的新版本 - NIR-DCL

2026-07-01 已实现：

```text
NIR-DCL = Non-IID-aware Robust DCL
```

实现文件：

```text
fedprime/methods/nir_dcl.py
fedprime/methods/local_rahfl.py
fedprime/methods/prac_hfl.py
fedprime/methods/rahfl_asymhfl.py
```

配置入口：

```text
configs/debug_nir_dcl_local_only.yaml
configs/kaggle_t4_nir_dcl_local_only.yaml
configs/kaggle_t4_nir_dcl_rahfl.yaml
```

当前最应该先跑：

```text
configs/kaggle_t4_nir_dcl_local_only.yaml
```

原因：

```text
先验证只改本地 DCL 是否能超过 AugMix+DCL local-only 和 RAHFL。
如果 NIR-DCL local-only 没有提升，就不要急着接通信。
如果 NIR-DCL local-only 已经明显提升，再跑 NIR-DCL + AsymHFL。
```

### CARA-L / NIR-DCL 首轮结果

已完成：

```text
NIR-DCL local-only:
  final avg_acc   = 53.30
  final worst_acc = 36.01
  best avg_acc    = 54.74

NIR-DCL + AsymHFL:
  final avg_acc   = 57.36
  final worst_acc = 46.23
  best avg_acc    = 57.89

RAHFL baseline:
  final avg_acc   = 56.41
  final worst_acc = 44.72
```

结论：

```text
NIR-DCL local-only 不成立，明显弱于 AugMix+DCL local-only。
但 NIR-DCL + AsymHFL 超过 RAHFL，说明 NIR-DCL 可能不是单独提升本地性能，
而是让本地表征更适合 AsymHFL public-logit 通信。
```

下一步不要马上大规模消融。更合理的下一步：

```text
1. 先复跑 seed=1 或 alpha=0.3 中的一个，确认这个提升不是 seed-0 偶然。
2. 补 tail_acc / per-client / per-class 指标，确认 worst_acc 提升来自哪里。
3. 如果还有算力，再跑 alpha=1.0，确认正常 Non-IID 不掉。
```

### 已实现 FedCARA / CARA-C

2026-07-01 新增：

```text
FedCARA = AugMix + CARA-L + CARA-C
```

其中：

```text
CARA-L: 原 NIR-DCL 的正式命名，负责类别自适应鲁棒本地对齐。
CARA-C: 新通信模块，负责类别自适应可靠教师蒸馏。
```

CARA-C v1 公式：

```text
w_{i,j,c} = acc_{j,c} * (1 - acc_{i,c})
```

默认还加一个安全门：

```text
only use class c if acc_{j,c} > acc_{i,c}
```

然后在 public logits 上做 class-weighted KL：

```text
L = sum_c w_{i,j,c} * p_j,c * log(p_j,c / p_i,c)
```

配置入口：

```text
configs/debug_fedcara_cifar10c.yaml
configs/kaggle_t4_fedcara.yaml
```

下一步先跑：

```text
configs/kaggle_t4_fedcara.yaml
```

比较：

```text
RAHFL baseline:   56.41 / 44.72
CARA-L+AsymHFL:   57.36 / 46.23
FedCARA v1:       55.88 / 45.93
```

结论：

```text
FedCARA v1 final avg_acc 没超过 RAHFL，但 worst_acc 超过 RAHFL。
说明当前 CARA-C 过于偏向弱类/弱客户端，提升公平性但牺牲平均精度。
```

下一步若继续改通信，建议不要纯替换 AsymHFL，而是做 hybrid：

```text
L_comm = L_AsymHFL + lambda_cara * L_CARA-C
```

或者：

```text
teacher selection 仍用 AsymHFL overall routing，
但 KD loss 中加入 class-aware residual weight。
```

### 必须补的严谨性

后续正式实验前，必须避免审稿人攻击：

```text
1. 不能只在 alpha=0.1 极端 Non-IID 上赢。
2. 至少覆盖 IID、alpha=1.0、0.5、0.3、0.1。
3. 正常场景不能明显低于 RAHFL。
4. severe Non-IID 下要重点看 avg_acc、worst_acc、tail_acc。
5. 通信路由不能使用最终测试集。
6. PRAC 的 route/accept 应使用本地 held-out validation split。
```

推荐最终实验定位：

```text
正常/IID 场景：保持 RAHFL 级别鲁棒性，不显著下降。
中重度 Non-IID：提升 worst-client / tail-class / average accuracy。
通信成本：如果 public1 接近 public4，应强调低通信开销。
```

### 当前不要做的事

暂时不要继续：

```text
1. 盲目设计新的 public-logit 通信模块。
2. 只在 PRAC 上反复调超参数。
3. 只追求 alpha=0.1 上超过 RAHFL。
4. 只报告 avg_acc，不分析 worst_acc / tail_acc。
5. 把工作讲成“RAHFL 粗粒度，我细粒度”。
```

更好的表述是：

```text
RAHFL-inspired but Non-IID-aware:
我们沿用强鲁棒本地增强思想作为公平基座，
但针对 RAHFL 在 label-skew 下的本地对比学习偏斜和客户端级通信偏差进行改进。
```

## Current Authoritative Next Steps - 2026-06-30

### Now: run AugMix + DCL local-only control

The key unanswered question is whether current PRAC communication adds value
beyond RAHFL-style local robust training.

Run:

```text
configs/kaggle_t4_augmix_dcl_local_only.yaml
```

Meaning:

```text
method_name: prac_hfl
warmup_rounds: 999
40 rounds of AugMix + CE + JSD + DCL local training
no PRAC communication in any round
```

Compare against:

```text
RAHFL public4 final:      avg_acc=56.41, worst_acc=44.72
PRAC public1 final:       avg_acc=54.63, worst_acc=41.88
PRAC public4 final:       avg_acc=52.96, worst_acc=43.27
```

Decision rule:

```text
If local-only >= PRAC public1/public4:
  current PRAC communication has weak or negative contribution.

If PRAC > local-only but still < RAHFL:
  PRAC has useful signal, but communication strength / accept policy needs tuning.

If local-only is much lower than PRAC:
  PRAC communication is useful and should be optimized rather than discarded.
```

Important current interpretation:

```text
PRAC is not empty: accept_rate is nonzero.
But public4 lowered avg_acc compared with public1 while improving final worst_acc.
This suggests weak-client help plus average-performance negative transfer.
```

## Current Authoritative Next Steps - 2026-06-25

### Now: run the new FedPRIME-PAIR full experiment

FedPRIME-PAIR has been implemented as a switchable method:

```text
method_name: fedprime_pair
FedPRIME-PAIR = PRIME + CBCL + CPAD
```

Smoke test passed:

```text
configs/debug_fedprime_pair_cifar10c.yaml
round 0: avg_acc=11.52, worst_acc=10.00, local_loss=5.1416, cpad_loss=0.7056
```

The next formal Kaggle run is:

```text
configs/kaggle_t4_fedprime_pair_full.yaml
```

Important runtime note:

```text
Use code at or after commit 8a4ee15.
The setup cell must show git log -1 containing 8a4ee15 or a later commit.
```

This version includes:

```text
1. FedPRIME-PAIR heartbeat logs in the full run.
2. CBCL forward-pass optimization that reuses model embeddings.
3. Kaggle data import compatibility for both --destination and --repo-root.
```

Expected full-run heartbeat:

```text
[heartbeat] round 000 start
[heartbeat] round 000 local client 0 start
[heartbeat] FedPRIME-PAIR local phase, client=0 batch=50 loss=...
```

Preferred Kaggle entry:

```text
Use the Python streaming launcher cell, not a long %%bash cell.
The Python cell should call:
  RUN_DEBUG=1 PYTHONUNBUFFERED=1 bash scripts/run_kaggle_pair.sh
```

This script performs data import, environment check, partition audit, optional
debug smoke, full FedPRIME-PAIR training, pair-expertise analysis, summary, and
output packaging in one uninterrupted background-safe command.

Reason:

```text
Kaggle/IPython may buffer %%bash stdout until the subprocess exits.
The Python streaming launcher uses subprocess.Popen and prints a driver
heartbeat every 60 seconds, so the run never appears silent.
Do not use sys.stdout.reconfigure in Kaggle; its OutStream has no reconfigure.
```

If a fresh full run reaches training but prints no heartbeat for about 10
minutes, stop it. Do not wait for hours. Inspect whether the notebook pulled
the latest commit and whether data import completed successfully.

It matches the previous T4 fair setting:

```text
rounds=40
local_epochs=1
batch_size=64
public_batch_size=128
public_batches_per_round=4
fixed partition: outputs/partitions/cifar10c_alpha05_seed0_clients4_samples10000.npz
```

The first comparison should be against the already reproduced baseline:

```text
RAHFL final: avg_acc=56.41, worst_acc=44.72
```

Primary decision criteria:

```text
1. final/best avg_acc vs 56.41
2. final/best worst_acc vs 44.72
3. cpad_loss finite and not exploding
4. pair_expertise heatmaps show client-class-pair differences
5. underrepresented diagnosis after checkpoints exist
```

If full FedPRIME-PAIR is below RAHFL, do not immediately redesign again. First
rerun with switches:

```yaml
method.use_cbcl: false   # PRIME + CPAD
method.use_cpad: false   # PRIME + CBCL local-only control
```

This identifies whether CBCL or CPAD is the bottleneck.

Kaggle prepared dataset remains:

```text
fedprime-data
```

Use `scripts/import_prepared_data.py` before training.

### Previous D2C diagnostic context

The repaired FedPRIME-D2C warmup=3 experiment completed all 40 rounds without
NaN/Inf, but did not beat RAHFL:

```text
RAHFL final:            avg_acc=56.41, worst_acc=44.72
FedPRIME-D2C final:     avg_acc=52.31, worst_acc=39.78
FedPRIME-D2C best avg:  avg_acc=52.83 at round 37
LogitAvg+PRIME final:   avg_acc=52.10, worst_acc=39.72
Oracle D2C final:       avg_acc=51.74, worst_acc=39.13
```

Conclusion:

```text
Old D2C public-prior debiasing is not the current main route.
FedPRIME-PAIR/CPAD is the new implementation to validate.
```

## Historical D2C Next Steps - 2026-06-07

### Now: diagnose why D2C collapses toward LogitAvg

The repaired FedPRIME-D2C warmup=3 experiment completed all 40 rounds without
NaN/Inf.

```text
RAHFL final:            avg_acc=56.41, worst_acc=44.72
FedPRIME-D2C final:     avg_acc=52.31, worst_acc=39.78
FedPRIME-D2C best avg:  avg_acc=52.83 at round 37
```

Conclusion:

```text
PRIME + D2C is numerically stable and learns, but the first valid run does not
beat RAHFL. The final gaps are -4.10 avg_acc and -4.94 worst_acc.
```

The strict LogitAvg+PRIME control has completed:

```text
LogitAvg+PRIME final: avg_acc=52.10, worst_acc=39.72
FedPRIME-D2C final:   avg_acc=52.31, worst_acc=39.78
D2C gain:             avg_acc=+0.21, worst_acc=+0.06
```

This is effectively a tie. Current D2C does not yet provide a meaningful gain
over ordinary public-logit averaging.

Run the underrepresented-class diagnosis before ending the Kaggle session:

```bash
python scripts/diagnose_underrepresented.py \
  --config configs/kaggle_t4_fedprime_d2c_warmup3.yaml \
  --checkpoint_dir outputs/fedprime_d2c_cifar10c_alpha05_cr1_t4_warmup3/checkpoints
```

Then summarize and preserve:

```bash
python scripts/summarize_results.py --outputs outputs
```

Collect:

```text
outputs/fedprime_d2c_cifar10c_alpha05_cr1_t4_warmup3/metrics.csv
outputs/fedprime_d2c_cifar10c_alpha05_cr1_t4_warmup3/underrepresented_accuracy.csv
outputs/fedprime_d2c_cifar10c_alpha05_cr1_t4_warmup3/checkpoints/
outputs/summary.csv
```

### Next experiments, in priority order

0. Run the T4-safe PRIME local-backbone control before implementing another
   communication mechanism:

```text
configs/kaggle_t4_rahfl_prime.yaml
PRIME + DCL + original AsymHFL
```

This keeps all settings equal to `configs/kaggle_t4_rahfl.yaml` and changes
only `AugMix -> PRIME`. Compare its final/best `avg_acc`, `worst_acc`, and
underrepresented-class diagnosis against RAHFL. Do not attribute any later
PRIME-based communication gain to the communication module until this control
has been measured.

1. Run a T4-safe Oracle Prior D2C experiment. This is the highest-information
   next diagnostic:

```text
Implementation completed:
  configs/kaggle_t4_fedprime_d2c_oracle_warmup3.yaml
  fedprime/engine/prior_diagnostics.py
  scripts/analyze_priors.py

Formal Kaggle run is still pending.
```

Local end-to-end Oracle debug is complete on the RTX 3050. It produced finite
losses and all diagnostic outputs. The initial predicted prior had normalized
entropy `0.9999`, strongly indicating near-uniform prior collapse. The next
required experiment remains the full 40-round Kaggle Oracle run.

The full 40-round Oracle run is now complete:

```text
Oracle final:        avg_acc=51.74, worst_acc=39.13
Predicted D2C final: avg_acc=52.31, worst_acc=39.78
LogitAvg final:      avg_acc=52.10, worst_acc=39.72
```

Oracle is not better, so do not spend the next run only improving predicted
prior estimation. The next experiment priority is:

```text
1. T4-safe Oracle + no prior debias
2. Oracle with beta=0.1 or beta=0.2
3. Oracle + no class-balanced aggregation
4. Oracle + no complementary KD
5. smaller/ramped lambda_d2c
```

The first target is prior debias because missing classes receive up to about
`+3.45` logit under the current `beta=0.5, p_min=0.001` configuration.

The Oracle final checkpoints also show:

```text
client 2 missing_acc=0.00, tail_acc=4.63
client 3 missing_acc=0.00, tail_acc=0.00
```

Future D2C redesigns must be judged primarily by weak-client `tail_acc` and
`missing_acc`, not only average accuracy. A method that does not improve these
metrics does not validate the complementary-knowledge claim.

RAHFL-original missing/tail has now been measured:

```text
RAHFL final avg/worst: 56.41 / 44.72
client 2 missing_acc: 0.00
client 3 missing_acc: 0.00
```

So RAHFL's strong average performance does not demonstrate missing-class
transfer in this fixed alpha=0.5 split. The next research step is not another
RAHFL missing run. It is to design a public-logit communication module that can
explicitly improve missing/tail classes, or to test whether a same-domain
balanced public CIFAR-10 subset is required for that goal.

Run:

```bash
python scripts/run_experiment.py \
  --config configs/kaggle_t4_fedprime_d2c_oracle_warmup3.yaml
python scripts/analyze_priors.py \
  --experiment_dir outputs/fedprime_d2c_oracle_cifar10c_alpha05_cr1_t4_warmup3
```

```text
If Oracle Prior improves substantially:
  predicted prior from cross-domain CIFAR-100 is the primary bottleneck.

If Oracle Prior remains near 52:
  class-balanced aggregation and/or complementary KD are the bottleneck.
```
2. Inspect `tail_acc` and `missing_acc` for both D2C and LogitAvg checkpoints.
3. Predicted-vs-true prior logging is implemented. After Oracle training, export
   and analyze `prior_diagnostics.csv`, `prior_summary.json`, and prior plots.
4. Inspect whether predicted priors are nearly uniform under temperature=3
   using normalized entropy, L1/KL, cosine similarity, and heatmaps.
5. The round-3 worst-client drop also suggests
   early D2C may be too aggressive.
6. Only after Oracle Prior, test targeted D2C stabilization:

```text
longer warmup
EMA prior
self-preserving gate
smaller lambda_d2c or beta
```

7. Run a T4-safe alpha=0.1 Severe Non-IID comparison after D2C is competitive.
8. After confirming the design is promising, add a strong RAHFL comparison:

```text
40 local pretraining epochs before communication
larger/full public communication budget per round
the same strengthened training budget for FedPRIME-D2C
```

9. Warmup ablation:

```text
configs/kaggle_t4_fedprime_d2c.yaml
configs/kaggle_t4_fedprime_d2c_warmup3.yaml
```

10. Create T4-safe controlled configs for:

```text
RAHFL+PRIME = PRIME + DCL + AsymHFL
FedPRIME-D2C+DCL = PRIME + DCL + D2C
```

11. Run D2C component ablations.
12. Run seeds 0, 1, 2 only after the design is stable.
13. Evaluate official CIFAR-10-C corruption groups later.

Full experiment descriptions and configuration paths:

```text
docs/experiments/guides/EXPERIMENT_GUIDE_ZH.md
```

### Kaggle execution rule

Kaggle background `Save Version` runs cannot be modified or inspected with new
cells after starting. Prepare and validate the entire notebook before launch.
Future launch snippets must automatically perform setup, checks, training,
analysis, and result packaging. Never rely on adding a diagnostic cell during
an active background run.

### Resume prompt

```text
读取 docs/project/ARCHITECTURE.md、docs/project/PROJECT_STATE.md、docs/experiments/guides/EXPERIMENT_GUIDE_ZH.md 和 docs/project/TODO_NEXT.md，
继续推进 FedPRIME-D2C。先检查当前 Kaggle 核心对比是否完成，并分析 summary.csv
以及两个 metrics.csv。
```

## Historical Next Steps

The section below records earlier plans and may be outdated. Use the
`Current Authoritative Next Steps - 2026-06-06` section above first.

## Immediate Next Steps

0. Current continuation checkpoint.

Done:

- local data prepared
- environment check passed
- partition audit generated
- debug FedPRIME-D2C smoke run passed

Output:

```text
outputs/debug_fedprime_d2c_cifar10c/metrics.csv
outputs/partition_audit/fedprime_d2c_cifar10c_alpha05_cr1/
```

1. Check Git status.

```powershell
git status --short
```

2. Prepare local data if cloning on a new machine.

```powershell
D:\anaconda3\Scripts\conda.exe run -n pytorch python scripts\prepare_data.py --config configs\fedprime_d2c_cifar10c.yaml --download --rates 0 0.5 1
```

3. Run environment check again.

```powershell
D:\anaconda3\Scripts\conda.exe run -n pytorch python scripts\check_environment.py --config configs\fedprime_d2c_cifar10c.yaml
```

Expected after data preparation:

```text
einops: OK
opt_einsum: OK
data.private_root: OK
data.public_root: OK
```

4. Run partition audit.

```powershell
D:\anaconda3\Scripts\conda.exe run -n pytorch python scripts\audit_partition.py --config configs\fedprime_d2c_cifar10c.yaml
```

Inspect:

```text
outputs/partition_audit/<experiment_name>/client_class_counts.png
```

5. Start with a tiny smoke training config before full training.

Use the committed debug config:

Run:

```powershell
D:\anaconda3\Scripts\conda.exe run -n pytorch python scripts\run_experiment.py --config configs\debug_fedprime_d2c_cifar10c.yaml
```

6. Run core comparison.

```powershell
D:\anaconda3\Scripts\conda.exe run -n pytorch python scripts\run_grid.py configs\cifar10c_rahfl.yaml configs\cifar10c_rahfl_prime.yaml configs\fedprime_d2c_cifar10c.yaml
```

For the stricter controlled comparison with DCL on both PRIME methods:

```powershell
D:\anaconda3\Scripts\conda.exe run -n pytorch python scripts\run_grid.py configs\cifar10c_rahfl_prime.yaml configs\fedprime_d2c_dcl_cifar10c.yaml
```

7. Summarize results.

```powershell
D:\anaconda3\Scripts\conda.exe run -n pytorch python scripts\summarize_results.py --outputs outputs
```

8. Mechanism diagnostics after checkpoints exist.

Run LogitAvg+PRIME to check whether D2C beats plain public-logit averaging:

```powershell
D:\anaconda3\Scripts\conda.exe run -n pytorch python scripts\run_experiment.py --config configs\logitavg_prime_cifar10c.yaml
```

Diagnose whether weak / underrepresented client classes improved:

```powershell
D:\anaconda3\Scripts\conda.exe run -n pytorch python scripts\diagnose_underrepresented.py --config configs\fedprime_d2c_cifar10c.yaml --checkpoint_dir outputs\fedprime_d2c_cifar10c_alpha05_cr1\checkpoints
```

## Experimental Priorities

### Priority 1: Make Training Run

Goal:

- one complete FedPRIME-D2C run
- no shape/device/data bugs

### Priority 2: Core Battle

Run:

- RAHFL
- RAHFL + PRIME + DCL
- FedPRIME-D2C

Same config settings:

- `dirichlet_alpha: 0.5`
- `private_corrupt_rate: 1`
- `test_corrupt_rate: 1`

### Priority 3: Severe Non-IID

Run:

```text
dirichlet_alpha: 0.1
```

This is the most important setting for the paper story.

### Priority 4: Ablations

Run configs under:

```text
configs/ablations/
```

Most important:

- no prior debias
- no class-balanced aggregation
- no complementary KD
- oracle prior

### Priority 5: Clean vs Corrupted Test

Create or edit configs:

```yaml
test_corrupt_rate: 0
```

and compare against:

```yaml
test_corrupt_rate: 1
```

## Questions To Revisit

1. Should the main paper setting train on corrupted private data or clean private data?

Current default follows RAHFL: corrupted private train + corrupted test.

2. Should we add official CIFAR-10-C download/format support for corruption group evaluation?

Current `prepare_data.py` creates RAHFL-style random corrupted CIFAR-10. Official CIFAR-10-C per-corruption files are still needed for detailed group evaluation.

3. Should local pretraining be added before communication?

RAHFL paper uses local pretraining. Current unified runner supports checkpoint loading but does not yet include a dedicated pretraining script in `fedprime`.

## If Continuing With Codex Tomorrow

## 2026-06-29 Next Actions

Current mainline:

```text
PRAC-HFL
```

D2C and FedPRIME-PAIR are now historical diagnostic results, not the active main method.

Immediate next task:

```text
Rerun safe PRAC-HFL on Kaggle from commit 5e476ea.
```

Before running, verify:

```text
git log -1 --oneline
5e476ea 增强PRAC-HFL数值稳定性
```

Use:

```text
configs/kaggle_t4_prac_hfl.yaml
scripts/run_kaggle_prac.sh
```

Safe PRAC-HFL has:

```text
warmup_rounds=3
CE-only route/accept risk
virtual_lr=0.005
head_max_grad_norm=1.0
train.max_grad_norm=5.0
skip_nonfinite=true
```

Judgment criteria:

```text
No NaN through round 039.
avg_acc should approach or exceed RAHFL 56.41.
worst_acc should approach or exceed RAHFL 44.72.
accept_rate should not remain zero forever.
avg_delta should become less negative than the first unstable run.
```

If safe PRAC-HFL works:

```text
1. Run PRAC-HFL multi-seed.
2. Run RAHFL local-only / Average-KD / AsymHFL / PRAC-HFL communication ablation.
3. Add underrepresented head/tail/missing diagnosis to PRAC-HFL checkpoints.
```

If safe PRAC-HFL still underperforms:

```text
1. Try accept gate with patience or EMA route risk.
2. Try classwise=false model-level PRAC to reduce noisy class routing.
3. Try full-model accepted KD only after head-only version is stable.
```

Tell Codex:

```text
读取 docs/project/PROJECT_STATE.md 和 docs/project/TODO_NEXT.md，继续推进 FedPRIME-D2C 项目。先检查 git 状态，然后准备数据和跑一个 debug smoke training。
```
