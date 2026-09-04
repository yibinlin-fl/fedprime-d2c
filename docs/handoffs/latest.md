# FedPRIME-D2C Session Handoff

Updated: 2026-09-04

## Latest Decision: P4 Rejects HFL Decoupling; Marginal Profile Cannot Specify Pairwise Action

P4 used only sealed K0-B/P2, Phase-A1a, P3-A clean-base and P3-A permutation/null outputs. It ran no
model inference, GPU/OpenI, PRIME, training or permutation search. The harmful matrix uses exactly the
original DSA binding-family and valid-source condition, only for post-hoc explanation.

```text
verdict: GENERIC_ROUTING_ALIGNS_WITH_HARMFUL_ROUTING_BUT_TARGET_RULE_FAILED
recommendation: INFORMATION_MAY_EXIST_IDENTIFIABILITY_AUDIT_REQUIRED
method_go: false
```

The proposed HFL-vs-Local decoupling hypothesis is false: H9 mean generic/harmful cosine/Spearman is
`0.8496/0.7091`, versus L9 `0.8459/0.4667`; both have Top-3 overlap `0.8333`. The split instead comes
from how the fixed rank reversal acts on the full pairwise harmful-routing matrix. H9 targeted signed
destructive percentile is only `42.5%`, while L9 is `100.0%`; this matches the P3-A DSA split of
`60.1%` versus `0.0%` (lower DSA percentile is better).

Generic probes therefore recover class marginal salience but have not identified the pairwise map
needed for intervention. P2/P3-A remain NO-GO. Do not search a better permutation, use oracle M_harm
as a loss, start P3-B or train. If the user continues, the only permitted next step is a paper-level
identifiability audit of whether taxonomy-free observables can determine the missing pairwise map.

```text
report: deliverables/post_no_go_p4_routing_targetability_gap_20260904/P4_ROUTING_TARGETABILITY_GAP_AUDIT.md
analyzer: scripts/analyze_post_no_go_p4_routing_targetability_gap.py
```

## Latest Decision: P3-A Complete / Generic Profile Not a Valid HFL Targeter

The user authorized the exact missing-data completion: the same 1,000 Phase-A1a CIFAR-10 clean
sources were forwarded through all 16 H0/H9/L0/L9 round-40 checkpoints. This was clean-only local
RTX 3050 inference: no training, backward, corruption/PRIME generation or checkpoint modification.
Disabling Ampere TF32 matched the sealed V100 reference within `5.36e-7` maximum probability error,
with 100% argmax agreement. The output is `4 x 4 x 1000 x 10`, SHA256
`4D24CFC...0A7A7F`.

The original frozen P3-A then ran as a pure CPU output-space counterfactual. K0-B Bank-A + carrier
half Ua alone defined each arm/client rank-reversal class permutation. Seed `20260904` generated
1,000 unique random derangements per arm/client before binding, corruption family or DSA were read.
All response magnitude/geometry/K0-B-risk invariants passed; maximum error was `3.64e-10`.

```text
verdict: CLASS_IDENTITY_CAUSAL_BUT_GENERIC_PROFILE_NOT_TARGETING
status:  NO_GO_TO_METHOD
```

H9 DSA changed `0.204270 -> -0.013220` and L9 `0.205189 -> -0.061513`, with 4/4 clients positive.
However, the decisive HFL targeting gate failed: H9 targeted was at random-null percentile `60.1%`,
not the required bottom 10%. L9 alone was at `0.0%`. Thus changing class identity strongly changes
signed DSA, but the P2 taxonomy-free profile does not identify a specifically harmful HFL routing
assignment better than random. P2 remains an observational descriptor only. Do not run P3-B, tune
the permutation/profile, design a routing loss or start training.

```text
spec: docs/experiments/archive/P3A_ROUTING_IDENTITY_CAUSAL_AUDIT_ZH.md
clean exporter: scripts/run_p3a_clean_base_completion.py
analyzer: scripts/analyze_post_no_go_p3a_routing_identity.py
report: deliverables/post_no_go_p3a_routing_identity_causal_audit_after_clean_completion_20260904/P3A_ROUTING_IDENTITY_CAUSAL_AUDIT.md
```

## Latest Decision: P2 CLE-Specific Class-Visible Routing Audit Complete

P2 reused the complete K0-B round-40 H0/H9/L0/L9 response grid: four heterogeneous clients, two
independent 64-recipe PRIME banks and two disjoint 500-carrier halves. It ran pure NumPy analysis
only: no checkpoint load, model inference, PRIME generation, training, GPU or OpenI. All 16 response
hashes and their Phase-B0 final-round checkpoint lineage matched. Taxonomy-free outputs were written
and SHA256-sealed before Phase-A1a DSA and original K0-B risk were opened.

```text
CLE_SPECIFIC_CLASS_VISIBLE_ROUTING
+ CLASS_ROUTING_EXCEEDS_GENERIC_FRAGILITY
status: CANDIDATE_MECHANISM_FOR_CAUSAL_AUDIT
```

Across all eight system/bank/half pooled slices, the mean-client H9/H0 or L9/L0 ratios were
`1.596--2.008x` for output-spectrum concentration, `4.178--5.262x` for normalized positive
class-routing strength and `2.176--2.614x` for class-profile concentration. Every slice had 4/4
positive-chi clients. Strong-CLE class profiles were highly stable: minimum cross-half cosine
`0.997664`, minimum cross-bank cosine `0.975400`. Raw centered-response energy increased only
`2.154--3.187x`, so the normalized structural contrast is not just uniform generic fragility.

The new object does not obviously reduce to K0-B R. Across all 16 observations, positive routing
strength correlates with DSA at Pearson/Spearman `0.9810/0.8969`, compared with K0-B R at
`0.8648/0.8616`; its residual Pearson association after K0-B R is `0.9305`. More importantly, across
the eight matched CLE effects, positive routing-strength delta tracks DSA delta at
`0.9459/0.9524`, while `chi_out` delta does not (`-0.0030/0.1667`). Therefore the candidate is the
explicit stable nuisance-to-class routing profile/strength, not another scalar spectrum detector.

This is still observational, single-seed and retrospective. It does not revive CRSF and does not
authorize training or a new loss. The next permitted step is a paper-only causal-routing audit
design that must isolate routing identity/strength from raw response magnitude and K0-B detection;
no paid experiment is currently authorized.

```text
report:   deliverables/post_no_go_cle_specific_routing_audit_20260904/P2_CLE_SPECIFIC_ROUTING_AUDIT.md
seal:     deliverables/post_no_go_cle_specific_routing_audit_20260904/primary_taxonomy_free_manifest.json
analyzer: scripts/analyze_post_no_go_cle_specific_routing_p2.py
```

## Latest Decision: Post-NO-GO Class-Readout Audit P1 Complete

The final zero-cost representation audit loaded only CPU state dictionaries and existing
K1-C-Minimal moments/Grams/DSA. It ran no model inference, PRIME generation, optimization, GPU or
OpenI job. All checkpoint/classifier hashes and representation dimensions matched; all 24 response
matrices reconstructed their saved Grams with maximum error `9.55e-13`.

```text
READOUT_COUPLING_REMAINS_INTACT
+ GLOBAL_CHI_MISSES_CLASS_VISIBLE_GEOMETRY
+ READOUT_WEIGHTED_GEOMETRY_TRACKS_DSA
status: CANDIDATE_MECHANISM_FOR_NEXT_AUDIT
```

CRSF's class/probe routing-matrix cosine is `0.995453` for ResNet10 and at least `0.999689` for
MobileNetV2; dominant-mode readout-coupling cosine is at least `0.997850`; Top-3/Top-5 class-norm
sets remain unchanged. Yet the exploratory readout-weighted chi tracks DSA changes better than
global chi: after removing the functionally duplicated L9/ResNet rows, descriptive Pearson/Spearman
are `0.9980/0.9429` versus `0.9643/0.7143`. RawSpec supplies sign counterexamples where global chi
falls while readout-weighted chi and DSA worsen.

This is not a method GO and does not revive CRSF. H0/L0 feature-space responses, independent seeds
and full client/architecture coverage are missing, so CLE specificity and causal mediation remain
unproven. No new training or paid experiment is authorized.

```text
report:   deliverables/post_no_go_class_readout_audit_20260904/P1_CLASS_READOUT_ROUTING_AUDIT.md
assets:   deliverables/post_no_go_class_readout_audit_20260904/ARTIFACT_AVAILABILITY.md
analyzer: scripts/analyze_post_no_go_class_readout_p1.py
```

## Latest Decision: Post-NO-GO Mechanism Audit P0 Complete

The CPU-only P0 audit is complete. It used only existing K0-B, K1-B0, K1-C0 and K1-C-Minimal
artifacts; it loaded no checkpoint, generated no PRIME view, ran no model forward/backward pass and
used no GPU or OpenI job.

The allowed mechanism diagnoses are:

```text
GLOBAL_SPECTRUM_IS_WEAK_PROXY
+ CLASS_ROUTING_REMAINS_INTACT
+ ARCHITECTURE_DEPENDENT_CONTROLLABILITY
+ SPECTRAL_REDUCTION_BY_TAIL_REDISTRIBUTION
```

CRSF preserved the class-routing vector almost exactly: H9/L9 client-binding cosines were
`0.998579/0.998559`, rank correlations `1.000000/0.998496`, and Top-3/Top-5 overlap was 100%.
Pooled chi-to-DSA conversion efficiency was only `0.433/0.423`. Spectrally, CRSF reduced lambda-1
by 6.43% on average but increased tail energy by 6.21%; the principal-vector cosine remained
`0.999693`. ResNet10 carried most of the weak effect, while MobileNetV2 barely responded. H9/L9
client0 outputs are bit-identical and must not be counted as independent ResNet replications.

```text
report:   deliverables/post_no_go_mechanism_audit_20260904/POST_NO_GO_AUDIT.md
inputs:   deliverables/post_no_go_mechanism_audit_20260904/ARTIFACT_AVAILABILITY.md
analyzer: scripts/analyze_post_no_go_mechanism_p0.py
```

K1-C-Minimal remains `NO_GO_CRSF_INTERVENTION`; this audit does not authorize tuning, B-to-A,
additional architectures, replication, full training or a new method. There is currently no paid
experiment to run. The next step is a research decision about whether CLE should be approached at
the class-conditional routing level or whether the representation-intervention branch should stop;
P0 itself does not make that decision.

## Latest Decision: K1-C-Minimal Causal Intervention NO-GO

The old exact K1-C-FULL route is frozen as `SUPERSEDED_BEFORE_FORMAL`. It produced no calibration or
formal scientific result and must not be restarted. Its specification and implementation remain only
for provenance. K1-C0 remains a 10/10 observational GO: it established a CLE-associated generic
response-spectrum concentration, not that CRSF surgery works.

K1-C-Minimal is now implemented to test only the missing causal arrow:

```text
reduce response-spectrum concentration -> does real CLE DSA decrease?
```

The preregistered primary uses H9/L9, ResNet10/client0 and MobileNetV2/client3, A-to-B only, and
Frozen/CRSF/RawSpec. Correction uses 512 prespecified D_surgery carriers and 16 prespecified Bank-A
probes for five accepted steps at initial LR 1e-4. Every step retains exact post-update objective,
anchor-KL <= 0.02, rollback and deterministic LR halving. There is no separate LR calibration.
Evaluation remains full and independent: D_holdout 2,000 x all 64 Bank-B probes, sealed before CLE
binding and DSA are opened.

```text
spec:    docs/experiments/current/CLE_K1_C_MINIMAL_CAUSAL_GATE_OPENI_ZH.md
config:  configs/cle_k1_c_minimal_seed0.json
runner:  scripts/run_cle_k1_c_minimal.py
OpenI:   scripts/openi_cle_k1_c_minimal_entry.py
tests:   tests/test_cle_k1_c_minimal.py
```

Focused regression is 14/14 PASS. A real-checkpoint CUDA smoke completed with verdict
`SMOKE_ONLY_NO_SCIENTIFIC_DECISION`: frozen selection hashes verified, both CRSF and RawSpec accepted
one exact step, all metrics were finite, bounded in-memory streaming wrote no transformed-input disk
cache, and no oracle/evaluation assets were loaded. This is execution evidence only.

The OpenI benchmark completed and independently passed the engineering cost gate. It ran one H9/ResNet10 context without
oracle/evaluation assets, used 544 MiB bounded prefix arrays and 341.9 MiB peak CUDA allocation, and
projected Minimal Formal at 894.39 seconds / 0.2484 single-GPU hours. Allow 30--45 minutes as a
conservative 2--3x envelope because MobileNetV2 was not directly timed and oracle cost was proxied.
This was not scientific evidence.

Minimal Formal has now completed and was independently recomputed from the sealed response moments and
oracle predictions. Verdict: `NO_GO_CRSF_INTERVENTION`. H9/L9 CRSF unseen-chi reductions were only
`5.369%/5.452%` versus the frozen 15% gate; CRSF-minus-RawSpec advantages were only
`3.838/3.959 pp` versus 10 pp. DSA reductions were `0.005237/0.005193` (`2.322%/2.308%`) versus the
`0.05 or 25%` gate, with only `0.00549/0.00590` advantage over RawSpec versus 0.02. ResNet10 carried
most of the weak effect; MobileNetV2 chi and DSA changes were near zero.

All 31 artifact hashes and 18 pre-oracle seal hashes matched. Raw moments/Gram and all six prediction
files reproduced chi, DSA and reporting metrics exactly. All eight intervention traces completed five
accepted monotone steps; maximum final KL was 0.019646. This is a scientific method failure, not an
execution failure. Stop CRSF: no B-to-A, remaining architectures, tuning, replication or full training.
K1-C0 remains an observational mechanism result only; K0-B remains offline audit only.

```text
benchmark archive sha256: D16E82F85FFA636DBEE50086BF6A083F932BB1F8833F3F7E366F5E90AF24F2D4
formal archive sha256:    E07B9E75E2AEDDE0C1B3A4FF018CE0B4FD90EAA6CB88144D6E0D98588E43D4CA
report: deliverables/cle_k1_c_minimal_formal_20260904/RESULT_SUMMARY_ZH.md
```

## Previous Objective: K1-B0 CDR-SNR Shared Representation Localization

## Current Objective: K1-B0 CDR-SNR Shared Representation Localization

K1-A head-only SDMN formal is now frozen as `NO_GO_DIRECTIONAL_SURGERY`; do not tune or revive it.
K0-B's taxonomy-free detector remains valid, but changing only the classifier head did not close the
detect -> intervene -> CLE reduction loop.

The active stage is the zero-training K1-B0 localization gate. It asks whether the frozen K0-B
high-risk PRIME probes expose a carrier-stable, high-vs-energy-matched-specific and cross-bank
transferable nuisance subspace in the penultimate representations of H9/L9, stronger than matched
H0/L0. It does not yet perform SNR surgery or full FL training.

```text
spec:      docs/experiments/current/CLE_K1_B0_CDR_SNR_OPENI_ZH.md
engine:    fedprime/engine/cle_shared_nuisance_routing.py
analyzer:  scripts/analyze_cle_k1_b0_cdr_snr.py
OpenI:     scripts/openi_cle_k1_b0_cdr_snr_entry.py --mode=formal
selection: fedprime/augmentations/assets/cle_k1_b0/selection_manifest.json
tests:     tests/test_cle_shared_nuisance_routing.py
```

The selection manifest was reconstructed from the independently audited K0-B formal archive
`1E02A16C...88608`; it freezes H9/L9 active probes, top-20% rho probes and weights for both 64-recipe
banks. Local INSPECT verified all 16 frozen checkpoints, both bank hashes, D_select hash
`731B8CFF...F6CA`, and D_rep hash `321C0910...40EE`. The tiny local smoke passed with verdict
`SMOKE_ONLY_NO_SCIENTIFIC_DECISION`; no labels/evaluation assets were read and no optimizer,
backward, training or checkpoint write occurred.

Next action: run the unchanged 535,256,689-byte Phase-B0 OpenI dataset with
`scripts/openi_cle_k1_b0_cdr_snr_entry.py --mode=formal`. Download the compact
`cle_k1_b0_cdr_snr_seed0_formal_outputs.tar.gz`, audit all 20 frozen gates, and stop. A GO only
authorizes design of cross-bank SNR surgery; it does not authorize starting it automatically.

## Latest Formal Gate: K0-B v2 Taxonomy-Free Generic Probe

K0-A formally passed and K0-B v2 is now implemented without training or checkpoint writes. Its
primary object is carrier-stable plus class-selective response, not ordinary split-direction
reproducibility:

```text
spec:       docs/experiments/current/CLE_GENERIC_PROBE_K0B_OPENI_ZH.md
bank code:  fedprime/augmentations/frozen_prime.py
statistics: fedprime/engine/cle_generic_probe_gate.py
analyzer:   scripts/analyze_cle_generic_probe_k0b.py
OpenI:      scripts/openi_cle_generic_probe_k0b_entry.py --mode=smoke
tests:      tests/test_cle_generic_probe_k0b.py
```

Two complete 64-recipe PRIME banks are versioned under
`fedprime/augmentations/assets/cle_generic_probe_k0b/`. Their preregistered canonical hashes are
`6CAE529D...3DC01` and `4A53497E...BF4E`. Every primitive composition, spectral state,
displacement field, color coefficient, filter kernel, strength, mixture weight and depth is fixed
before inference and reused for every carrier.

The formal OpenI result was independently verified from all 16 raw response tensors and is
`GO_TO_K1_CHECKPOINT_SURGERY`. HFL and Local each passed all eight frozen gates; generic-fragility
kill was false. HFL K delta was `+0.252727`, combined R ratio `4.901569`, and both bank ratios were
`5.739226/4.317300`. Local K delta was `+0.232752`, combined R ratio `4.385780`, and both bank ratios
were `5.166668/4.094945`. Both systems had 4/4 positive-R clients. Manual S/Dcf/K/R recomputation
matched the returned metrics to maximum absolute error `5.56e-17`.

```text
result:  cle_generic_probe_k0b_seed0_formal_outputs.tar.gz
bytes:   234888047
sha256:  1E02A16C765D8AB976A692D444FA9DAEBE38C30F8279CD6DCCFC49D1BFF88608
report:  deliverables/cle_generic_probe_k0b_20260902/RESULT_SUMMARY_ZH.md
```

K1-A head-only SDMN is now implemented through INSPECT, smoke and numerical-calibration modes:

```text
spec:    docs/experiments/current/CLE_K1_SDMN_HEADONLY_OPENI_ZH.md
engine:  fedprime/engine/cle_sdmn_headonly.py
runner:  scripts/run_cle_k1_sdmn_headonly.py
OpenI:   scripts/openi_cle_k1_sdmn_headonly_entry.py --mode=smoke
tests:   tests/test_cle_sdmn_headonly.py
```

INSPECT verified all 16 checkpoints, both frozen banks, CIFAR-100 and the existing CLE evaluator
assets. Focused K1/K0 regression is 20/20 PASS. The local H9-client0/A-to-B tiny smoke ran Frozen,
Targeted, Direction-Sham, Random-Probe and Generic-Invariance, wrote full checkpoints/traces/hashes,
kept every anchor KL below 0.003, and reproduced identical split/selection/features/metrics/traces in
a second run. Its verdict is only `SMOKE_ONLY_NO_SCIENTIFIC_DECISION`.

The OpenI smoke is now independently audited and passed: archive/manifest hashes matched, public
split overlap was zero, all four surgery objectives decreased, maximum anchor KL was `0.002898`,
all checkpoints changed only `linear.weight/bias`, and raw unseen responses reproduced reported
S/Dcf/K/R exactly. Verdict remains `SMOKE_ONLY_NO_SCIENTIFIC_DECISION`.

The OpenI calibration artifact was independently audited. All 16 client/fold cases passed at
`1e-4`, two passed at `3e-4`, and none passed at `1e-3`. The raw result had a JSON naming collision
where the per-step LR trace overwrote the candidate scalar; the underlying traces are complete and
the scalar is unambiguously recoverable, so no rerun is required. The corrected schema and frozen
per-client/fold LR values are versioned in
`configs/cle_k1_sdmn_headonly_calibration_seed0.json`.

Formal K1-A is now implemented with Adam, 10 steps, anchor KL `<=0.02`, backtracking factor `0.5`
and at most 12 rollbacks. Its 2,000-image surgery pool preserves the exact calibration hash after
adding a disjoint 2,000-image holdout. It runs Frozen/Targeted/Direction-Sham/Random-Probe/Generic-
Invariance for H9/L9 and both A-to-B/B-to-A folds. Taxonomy-free unseen-bank artifacts are written
and hashed before the CLE binding, true corruption grid, DSA/WCCA/CFG or task labels are opened.

Next action: run the unchanged Phase-B0 OpenI dataset with
`scripts/openi_cle_k1_sdmn_headonly_entry.py --mode=formal`. Stop when K1-A returns one of its three
frozen verdicts; do not start full training, modify communication, or revive PNCB/PEW/BER.

## Latest Formal Gate: K0-A Public-Carrier Transfer Oracle

The CLE-HFL topic remains active, but PNCB-SCDW remains stopped. The only current method-candidate
gate is a zero-training test of the cross-carrier directional moment:

```text
spec:     docs/experiments/current/CLE_PUBLIC_CARRIER_K0A_OPENI_ZH.md
engine:   fedprime/engine/cle_public_carrier_moment.py
analyzer: scripts/analyze_cle_public_carrier_k0a.py
OpenI:    scripts/openi_cle_public_carrier_k0a_entry.py --mode=smoke
tests:    tests/test_cle_public_carrier_k0a.py
```

K0-A reuses the existing 535,256,689-byte Phase-B0 input archive containing CIFAR-100 and all 16
frozen round-40 H0/H9/L0/L9 checkpoints. It selects 1,000 CIFAR-100 train images with seed 20260901,
does not use their labels, and applies the existing 16 operators at severity 3 only as an oracle
mechanism audit. Blind centered class-vs-rest logit responses are saved and hashed before client-class
binding and operator-family truth are opened for scoring.

Focused regression and local smoke passed. The formal OpenI result was independently recomputed and
is `GO_TO_K0_B`: HFL and Local each passed all 10 preregistered gates. H9/H0 mAP was
`0.796627/0.406176` (delta `+0.390451`); L9/L0 was `0.811235/0.411342` (delta
`+0.399894`). Both systems had 4/4 positive clients and both null tests at `p=0.000999001`.
Directional-strength and coherence bootstrap CI95 lower bounds were strictly positive.

```text
result:  cle_public_carrier_k0a_seed0_formal_outputs.tar.gz
bytes:   48651705
sha256:  AA260672FED05C991DDEF2308342BD88150CA8A36FD8366EF9A9E85B2E523168
report:  deliverables/cle_public_carrier_k0a_20260901/RESULT_SUMMARY_ZH.md
```

The next action is K0-B taxonomy-free generic-probe design. K0-A used the true operator bank and
binding only for oracle scoring, so it does not authorize DME/K1 training. Preserve the local-first
interpretation; the result does not support communication amplification.

Full self-contained handoff for GPT Web discussion:

```text
docs/research/status/CLE_PNCB_SCDW_CURRENT_RESEARCH_HANDOFF_FOR_GPTWEB_2026_08_31_ZH.md
```

It records the problem, evidence chain, PNCB/SCDW mathematics, weakest assumptions, Phase-B0 data
and gates, smoke audit, conditional Phase-B1 design, claim boundaries and eight external-review
questions. A formal-result addendum now supersedes its earlier pending status.

## Latest Formal Result: PNCB Bridge NO-GO

The frozen Phase-B0 formal run completed and was independently checked from its returned probability
cache:

```text
result:  cle_public_canonicalization_phase_b0_seed0_formal_outputs.tar.gz
bytes:   31788115
sha256:  8F824A6EF21AFDF8E8CF089530786882FE684504079C17160DFD2205D140BE2C
verdict: NO_GO_PNCB_BRIDGE

PASS: G1 semantic preservation, G4 HFL retrieval,
      G6 relative overlay margin, G7 clean artifact null
FAIL: G2 old-nuisance contraction, G3 family separability reduction,
      G5 Local retrieval
```

The PNCB completed all 10 epochs and its training loss decreased by 18.77%, so the failure is not an
execution failure. It preserved semantics (worst canonical accuracy delta `-0.5875pp`) but increased
within-source cross-operator variance by `12.2267%` instead of contracting it by at least `25%`.
Family separability fell by only `9.4729%` versus the frozen `30%` requirement. Local gamma9
retrieval was mAP `0.593924`, hit `0.65`; both missed their gates. G6 passed only because overlay was
even more dispersive; it cannot override absolute G2 failure.

Decision: stop the current PNCB-SCDW route. Do not implement or run Phase-B1 classifier/SCDW A/B/C,
do not tune SCDW weights, and do not rescue the bridge by epoch/channel/loss-only tuning. SCDW as an
abstract object was not directly trained, but its required contraction bridge is absent. Any future
bridge must be a new paper candidate with a new pre-result argument and gate, not a renamed revival.

```text
report: deliverables/cle_public_canonicalization_phase_b0_20260831/RESULT_SUMMARY_ZH.md
```

## Latest Paper Design: Public Canonicalization + Signed Directional Withdrawal

The paper-only intervention-bridge design is complete:

```text
docs/research/status/CLE_PUBLIC_CANONICALIZATION_DIRECTIONAL_WITHDRAWAL_DESIGN_2026_08_30_ZH.md
```

Directly overlaying another fixed corruption bank on already-corrupted private images is rejected:
it does not overwrite the original degradation and revives an artificial taxonomy. The conditional
candidate instead uses the public unlabeled data already required by heterogeneous logit
communication to learn a frozen Public Nuisance Canonicalization Bridge (PNCB). The bridge is trained
only by public corrupted-to-source reconstruction; it consumes no private class, corruption, family
or binding metadata.

For private image `X`, the client obtains `C(X)` and estimates, for every wrong class, the signed
probability withdrawal `p(c|X)-p(c|C(X))` over samples whose task label is not `c`. The proposed SCDW
loss penalizes only a positive one-sided lower confidence bound, stop-grads the standard-error
threshold and canonical probability, and keeps the existing AugMix/JSD/DCL objective. A separate
canonical-view CE term is mandatory, as is a `bridge-only` arm for attribution. The method adds no
communication and is compatible with heterogeneous backbones because it operates in input and class-
probability spaces.

This is only `CONDITIONAL GO`. Its weakest assumptions are semantic preservation, contraction of the
original hidden degradation, label-independent public reconstruction, paired observability and
limited cancellation of harmful directions. The next gate is bridge-only, before classifier
training:

```text
Identity bridge
vs AugMix overlay
vs public canonicalizer

audit: semantic preservation, source-conditioned nuisance contraction,
       H9/L9-vs-H0/L0 hidden-binding retrieval, clean artifact null,
       and per-client consistency
```

If the public canonicalizer cannot preserve semantics while contracting the old degradation, the
candidate is `NO-GO`; SCDW weights must not be tuned to rescue it. If it passes, the first classifier
experiment is a matched `baseline / bridge-only / bridge+SCDW` 12-round screen. Only the Phase-B0
bridge harness and execution smoke are authorized; no formal OpenI run or classifier training has
started, and no communication method, PEW/BER revival or fixed corruption labels should be added.

Phase-B0 implementation is now complete without classifier training:

```text
spec:       docs/experiments/current/CLE_PUBLIC_CANONICALIZATION_PHASE_B0_ZH.md
model:      fedprime/models/public_canonicalizer.py
engine:     fedprime/engine/cle_directional_withdrawal.py
train:      scripts/train_cle_public_canonicalizer_phase_b0.py
analyzer:   scripts/analyze_cle_public_canonicalization_phase_b0.py
OpenI:      scripts/openi_cle_public_canonicalization_phase_b0_entry.py
tests:      7 focused tests passed; 16/16 final checkpoints strict-loaded
smoke:      4 public images, one CPU batch, execution-only PASS
formal run: NOT STARTED
```

The four approximately 173 MiB H0/H9/L0/L9 Phase-A1a arm archives are now local and verified. Each
contains four final and four round-12 checkpoints; the Phase-B0 input intentionally extracted only
the 16 final round-40 checkpoints. The slim OpenI package is complete:

```text
input:  local_runs/cle_public_canonicalization_phase_b0/
        cle_public_canonicalization_phase_b0_seed0_inputs.tar.gz
bytes:  535256689
sha256: DFB766F6494A5F61AA16F45666EC250A30501066AB54D89C984CD2324293B9BC
entry:  scripts/openi_cle_public_canonicalization_phase_b0_entry.py --mode=smoke
```

It contains only the frozen 1,000-source evaluation arrays, CIFAR-100 public tar, 16 final
checkpoints and a per-file hash manifest; private Phase-A1a training arrays and round-12 duplicate
weights are excluded. The OpenI entry defaults to smoke and verifies both the archive hash and every
manifest file before execution.

The OpenI end-to-end smoke passed on 2026-08-30:

```text
result:  cle_public_canonicalization_phase_b0_seed0_smoke_outputs.tar.gz
bytes:   3344564
sha256:  C57D3A9BE84E04FDDBB35402DF011B59294D78B532951346476182A891E81E54
verdict: SMOKE_ONLY_NO_SCIENTIFIC_DECISION
```

All 19 manifest files verified, all 16 classifiers ran on CUDA, the temporary PNCB completed exactly
one epoch/two batches, and the analyzer used 20 balanced sources, all 16 operators and severity 3.
Independent NPZ inspection found all 12 probability tensors at shape `(4,20,16,10)`, all values
finite and maximum probability-sum error `2.38e-7`. Smoke bridge metrics are not scientific evidence:
the temporary two-batch PNCB showed semantic loss and no nuisance contraction, which is expected to
remain uninterpretable under the frozen smoke rule. The next action is user approval for one formal
`--mode=formal` bridge-only run. Do not alter the seven gates or start classifier/SCDW training. No
Phase-A1a retrain is required.

## Latest Zero-Training Gate: PIDR Recovers Hidden Binding Under Oracle Probes

The paper-only definition, identifiability counterexample, weakest-assumption analysis and 2024--2026
collision audit are complete:

```text
docs/research/status/CLE_LOCAL_FIRST_DIRECTIONAL_SHORTCUT_IDENTIFIABILITY_AUDIT_2026_08_30_ZH.md
```

The current CIFAR CLE construction is a valid controlled stress test and DSA is a binding-specific
directional diagnostic, but neither is evidence that class--corruption spurious correlation is a newly
discovered real-world phenomenon. Real acquisition/quality shortcuts are documented; the cyclic four-
family map and gamma 0.9 are synthetic controls and must be described as such.

The population local-first directional shortcut risk is the DSA causal estimand. It is not identifiable
from a single already-corrupted image, its task label and i.i.d. AugMix views: two worlds can have the
same complete observable transcript while the model uses the degradation in one and the semantic
feature in the other. A view-residual, spectral or clustering rewrite would either miss persistent base
shortcuts or revive frozen C3R/CRSR-style negatives.

The only conditional candidate is Probe-Indexed Directional Promotion Risk (PIDR). It requires a
semantic-preserving intervention bridge whose distinguishable probes overwrite rather than merely
overlay the relevant degradation and cover every harmful direction. Under those assumptions PIDR
upper-bounds the unknown binding-set probability promotion up to coverage and intervention errors.
This is not yet a method GO: FedPIN, FedCD, ShortcutProbe, FedDDL, FedCAug, GIC and recent unlabeled
debiasing methods create strong collisions.

The approved zero-training oracle gate is complete. It reused the existing round12/round40 prediction
caches; no training and no inference ran. The estimator read only probabilities, task labels and 16
probe tensor identities. Binding and operator-family metadata were opened only after all promotion
matrices existed, for scoring and null tests.

Round-40 formal result:

```text
arm   PIDR       mAP       AUC       class-to-family hit
H0    0.024559   0.441855  0.510677  0.225
H9    0.175479   0.844847  0.923906  0.850
L0    0.025401   0.430622  0.507083  0.275
L9    0.174728   0.865557  0.933177  0.875

HFL mAP delta:      +0.402993
Local mAP delta:    +0.434935
positive clients:   4/4 and 4/4
class/probe null p: 0.000999 for H9 and L9
verdict:             GO_TO_INTERVENTION_BRIDGE_DESIGN
```

All six frozen gates passed for both HFL and Local. Round12 already showed the same pattern. This
establishes oracle-side directional observability: distinguishable degradation probes can recover the
hidden class binding without giving the estimator a family taxonomy. It does not establish that
i.i.d. AugMix views overwrite the base degradation, that training without clean sources is identified,
or that PIDR is method-novel relative to ShortcutProbe/FedCD.

```text
engine:     fedprime/engine/cle_probe_directional_promotion.py
analyzer:   scripts/analyze_cle_pidr_zero_training_gate.py
tests:      10 focused tests passed
result:     deliverables/cle_pidr_zero_training_gate_20260830/
commit:     bd37fc6 (local only; not pushed)
```

Next action superseded by the PNCB-SCDW bridge-only gate above. Do not implement a classifier loss or
start 12-round training until that gate passes.

## Current CLE Result: Directional Shortcut Is Local-First; Communication Amplification NO-GO

The bad-teacher / wrong-routing / shortcut-propagation story remains parked. Historical D2C and
Oracle D2C results, FedFalsify attribution and the beta0/beta4 coupling screen do not support making
teacher selection the dominant CLE mechanism.

A narrower zero-training screen asked whether the historical `gamma=0.9` RAHFL
models directionally move predictions toward classes bound to an applied corruption family more
than matched `gamma=0.0` models. Existing assets were audited: the 272,728,582-byte historical
archive contains all four `gamma=0/0.9` client checkpoints. Existing balanced tests repeat the same
source indices across corruptions but independently sample severity and omit explicit source IDs,
so the formal audit must regenerate a deterministic paired evaluation grid from 1,000 clean images:
16 old-CLE operators, severity 3, eight checkpoints, 128,000 forward items.

Formal inference ran on OpenI; the local RTX 3050 was used only for archive/hash checks, tiny
synthetic dry-runs and an independent recomputation from the returned probability cache. The frozen
protocol used Directional Shortcut Alignment, a `gamma09-gamma00` contrast, group-size-preserving
shuffled binding maps, paired bootstrap, integrity gates and five scientific promotion gates.

Implementation and artifact status:

```text
spec:           FROZEN
implementation: COMPLETE
focused tests:  3 passed
checkpoint load: 8/8 strict CPU PASS
input package:  cle_shortcut_alignment_phase_a0_seed0_inputs.tar.gz
input bytes:    184575308
input sha256:   C1F6823E186DDAF6DB44A38BCBDA300C78B9F1B4702C5E84B7AEE3A485499EFE
OpenI run:      COMPLETE
result archive: cle_shortcut_alignment_phase_a0_seed0_outputs.tar.gz
result bytes:   4883205
result sha256:  CCC91ED4EC3F08C6CA6433CA275423AD3E32EDB824993F7760F94A8A49B4B76F
code commit:    e005689 (pushed to origin/main)
```

Frozen specification:

```text
docs/experiments/current/CLE_SHORTCUT_ALIGNMENT_PHASE_A0_OPENI_ZH.md
```

All input-integrity checks passed. The OpenI summary and an independent recomputation from the
returned `predictions.npz` agree exactly:

```text
gamma00 pooled DSA: -0.0003018894
gamma09 pooled DSA:  0.2013210658
delta DSA:           0.2016229552
paired CI95:         [0.1964123272, 0.2072188988]
positive clients:    4/4
pooled shuffled p:   0.000999001
client shuffled p:   0.000999001 for all 4 clients
verdict:             GO_TO_MATCHED_PARTITION_DESIGN (G1--G5 all PASS)
```

Secondary changes from `gamma00` to `gamma09` were Avg `52.2422 -> 46.8250`, Worst
`43.4188 -> 37.9375`, WCCA `36.0000 -> 19.3125`, CFG `2.6500 -> 11.8813`, and paired
prediction-flip rate `0.148325 -> 0.353404`. This is strong evidence that CLE induces a directional
corruption-to-class shortcut under the historical RAHFL protocol. It does not yet show that the
effect is caused or amplified by federation, communication or model heterogeneity. Passing Phase-A0
permits only the matched-global Local-D/Local-E attribution design; it is not a paper or method GO.

Checkpoint provenance was re-audited on 2026-08-30. The eight weights were copied byte-for-byte
from `outputs/cle_rahfl_diagnostic_outputs.tar.gz`, specifically the four final client checkpoints
under each of `diag_rahfl_cle_alpha05_gamma00_seed0` and `diag_rahfl_cle_alpha05_gamma09_seed0`.
They are project-trained historical diagnostic models with matched alpha 0.5, seed 0, architectures,
optimizer, local epochs, public batches and AugMix/JSD+DCL+AsymHFL configuration; only the intended
CLE dataset root differs. Their resolved configs set `pretrain_epochs=0` and `rounds=40`. Therefore
Phase-A0 is an internally valid post-hoc matched diagnostic, but it is not evidence from the canonical
40-pretrain + 40-communication RAHFL schedule. The old prepared gamma00/gamma09 dataset archives are
not currently present locally, so exact client-label/source-index identity cannot now be byte-audited;
the deterministic generator uses the same alpha, seed and `partition_seed=seed`, which establishes
the intended matched partition. Any promoted paper experiment must persist partition/source hashes.

Phase-A1a completed on OpenI on 2026-08-30 using four jointly matched 40-round arms:
`H0/H9 = strict AsymHFL-val at gamma 0/0.9`, `L0/L9 = Local-only at gamma 0/0.9`.
All arms used the same code, source partition, labels, severities, persisted fit/audit split,
per-client initial weights and arm-independent private-loader RNG. Runtime hashes verified that HFL
and Local saw identical first-batch labels and all AugMix/DCL views in every round and client. Final
test labels remained reporting-only. The returned probability tensors were independently recomputed.

Round-40 formal DSA result:

```text
H0  0.0015228341        H9  0.2042704937
L0  0.0008234010        L9  0.2051892788
HFL CLE effect:          +0.2027476596
Local CLE effect:        +0.2043658778
communication A_pool:    -0.0016182182
paired source CI95:      [-0.0033365882, 0.0001283891]
positive clients:        1/4
top-1 amplification:     +0.0025799851
H9 shuffled-map p:       0.000999001
verdict:                 NO_GO_FL_SPECIFIC_AMPLIFICATION
```

G1--G3 failed; G4--G5 passed. At round 12, `A_pool=-0.0169168048` with CI entirely below zero,
but this early suppression disappeared by round 40. CLE itself is strong and reproducible in both
training systems: controls have DSA near zero, while gamma09 has DSA about 0.204; Avg falls about
6--7pp, WCCA falls 20--26pp, CFG rises from about 2.8 to about 12, and paired prediction flips rise
from about 0.15 to about 0.36. The supported mechanism is therefore `CLE -> local-first directional
shortcut`; strict AsymHFL neither materially amplifies nor durably suppresses it.

```text
code commit: 6199dd8 (pushed to origin/main)
input: cle_shortcut_amplification_phase_a1a_seed0.tar.gz
input bytes/sha256: 408228487 / 6322F16513C6980CDC5904D7EF91204A241205BC76DCCE8BC450E635519B4202
result: cle_shortcut_amplification_phase_a1a_seed0_analysis_outputs.tar.gz
result bytes/sha256: 19322309 / FDF1BEC2395334DD3816BC9C3F594B01D814DB22C0CC245CD1378A45180F397C
spec: docs/experiments/current/CLE_SHORTCUT_COMMUNICATION_AMPLIFICATION_PHASE_A1A_ZH.md
```

The user explicitly chooses to continue CLE. Do not revive bad-teacher, routing-amplification,
D2C/Oracle-D2C, PEW/BER-as-core, or tune/repeat Phase-A1a to rescue communication amplification.
The next stage is paper-only design and collision audit for a genuinely local-first directional
shortcut-suppression object that uses no environment/corruption labels, no clean counterpart and no
source index. It must be distinguished from AugMix/JSD, AugMax, consistency regularization,
IRM/VREx, GroupDRO/CVaR, counterfactual invariance and all frozen project negatives before code or
new training. The Phase-A1a bootstrap covers evaluation-source uncertainty only; seed 0 alone does
not establish training-seed stability.

## Superseding Topic-Selection State: Stop Extending RAHFL by Adding One More Difficulty

There is currently no active paper method, implementation task, local run or OpenI run. Preserve
all RAHFL/CLE-HFL/PEW+BER code and evidence, but do not restart any candidate below without a new
paper-level argument that changes its mathematical core.

The adversarial-attack / dual-robustness extension is now formally `PAPER NO-GO`. Model-
heterogeneous FL with adversarial robustness, malicious clients, logit poisoning and trustworthy
heterogeneous distillation already has direct adjacent literature. A narrower story such as
`robustness-profile conflict`, `teacher-ranking conflict`, `routing-amplified poisoning` or
`corruption-camouflaged poisoning` does not by itself create an independent method contribution;
its likely defense reduces to existing detection, trust weighting or robust logit aggregation.
Do not run the proposed Phase-A0, add PGD/FGSM/AutoAttack, modify RAHFL, implement a poisoning
harness or design a defense for this route.

Current corruption/robustness topic pool:

```text
severity heterogeneity:                 NO-GO
compound corruption:                    NO-GO as a paper core
corruption--label coupling:             EXPERIMENTAL SCREEN NO-GO
communication-induced forgetting:       low-value diagnostic; do not invest
robustness propagation:                 composition-risk conditional idea; no active test
data cleaning / sample removal:         NO-GO for the intended paper narrative
availability--corruption coupling:      NO-GO due direct collisions
dynamic corruption:                     NO-GO / highly crowded
compression robustness:                 high composition risk; not active
adversarial attack / dual robustness:   FORMAL NO-GO
PEW+BER:                                strong baseline, not the paper method core
```

The search rule is changed. Do not ask what extra factor RAHFL lacks. Start from an effective
2024--2026 centralized robustness objective or method and identify a necessary information object
that becomes unavailable, non-decomposable or incomparable in FL: global distributions, cross-
sample pairing, global hard-example order, cross-domain statistics, a common parameter space,
stable/unstable feature identities, a centralized reference model, or global corruption
statistics. A candidate advances only if all of the following are made explicit before code:

```text
centralized problem and why its method works
the exact necessary information that is broken by federation
why the fracture is nontrivial and not a toy protocol choice
the weakest assumption that repairs identifiability/executability
2024--2026 collision audit, including federated variants
available datasets/code and a cheap falsification test
```

Next action: paper-only reverse search from centralized robustness methods to genuine federated
structural fractures. Do not implement, run experiments or proactively commit during topic
selection.

## Completed Historical Result: RAHFL Corruption--Label Coupling Phase-A

This completed screen asked whether a fixed amount of annotation noise causes an additional penalty
when wrong labels concentrate on high-severity corrupted samples, and whether RAHFL/HFL amplifies
that penalty. It is retained for provenance and is not an active topic.

Frozen Phase-A1a comparison:

```text
Independent:     beta=0
Strong Coupled:  beta=4
same frozen corrupted images, disjoint client partition, fit/audit split,
20% noisy-label count and true-class -> wrong-class transition matrix
```

Each of four heterogeneous clients has 10,000 mutually exclusive CIFAR-10 samples and an exact
9,000/1,000 fit/audit split. The fit labels contain exactly 1,800 errors per client. The same noisy
label manifest is used by local CE pretraining and communication-round CE/DCL. Trusted clean audit
labels are routing-only; final test labels are reporting-only. The actual legacy RAHFL IID helper is
disjoint, but the default `Network/pretrain.py` and `HHF/RAHFL.py` IID branches independently sample
clients and may overlap; the new disjoint split is an explicit protocol choice.

Implementation status:

```text
commit: 999fee0 (pushed to origin/main on 2026-08-24)
focused tests: 2 passed
beta0/beta4 1+1 smoke: passed
artifact audit: all checks passed
formal 40+40: not started
OpenI seed0 10+10 screen: completed; no promotion
```

The artifact audit recovered a frozen image SHA256 beginning `e20128a7ad50`, verified 40,000 unique
client samples, exact 9,000/1,000 splits, identical flip matrices, clean audits and unchanged tests.
Mean severity among noisy fit samples is 2.5029 for beta=0 and 3.5699 for beta=4.

OpenI input and entry:

```text
dataset: rahfl_coupling_phase_a_seed0_prepared.tar.gz
bytes: 327018418
sha256: AE5F9524AF594963C790016FB386BD0EB600ACD84BBC1D12EF57DA7393D1835F
entry: scripts/openi_rahfl_coupling_phase_a_entry.py
args: --mode=both
```

The completed entry copied condition archives and the paired JSON summary to
`c2net_context.output_path`, then called `upload_output()`. Returned files included
`rahfl_coupling_phase_a_screen_seed0_beta0_outputs.tar.gz`,
`rahfl_coupling_phase_a_screen_seed0_beta4_outputs.tar.gz`,
`rahfl_coupling_phase_a_screen_seed0_both_outputs.tar.gz`, and
`rahfl_coupling_phase_a_screen_seed0_summary.json`.

The OpenI 10+10 screen completed on 2026-08-24. Independently recomputed beta0-minus-beta4 results
were final Avg/Worst `-1.90/-1.83pp`, last-3 Avg/Worst `-2.01/-1.37pp`, and all-10 Avg/Worst
`-0.40/-0.07pp`. Beta4 was better, not worse. Before the first collaborative phase, beta4 clean-audit
accuracy was already +2.45pp on average and higher for all four clients, so the reverse signal began
in local pretraining rather than an AsymHFL amplification failure. Configs differed only by name and
beta; both have rounds 0--9 and four checkpoints. Verdict: `SCREEN NO-GO`; do not promote to 40+40,
Local/Centralized diagnostics, beta/noise sweeps, or a post-hoc inverted paper claim. Evidence:
`deliverables/rahfl_coupling_phase_a_screen_20260824/RESULT_SUMMARY_ZH.md`.

## Latest Topic Reset: Federated Source-Graph Weak Supervision Is the Conditional Primary

After parking corruption/CLE-HFL/PEW-BER as the paper mainline, a benchmark-first screen of recent
federated topics removed heterogeneous LoRA, federated calibration/conformal prediction, open-world
category discovery, incomplete-modality multimodal FL, federated causal overlap recovery and generic
pairwise-risk optimization because direct 2024--2026 methods already cover their central mechanisms.

One candidate survives for a strict theory gate:

```text
Federated Source-Reliability Completion
under Private and Partially Overlapping Labeling Functions
```

Clients hold unlabeled data and private subsets of programmatic labeling functions. They do not send
samples, LF code or per-sample weak-label matrices; they send aggregate LF-pair agreement moments.
The union co-observation graph can contain reliability information absent from every local graph. In
the minimal binary conditionally-independent model, `M_jk = a_j a_k`. Disconnected graphs are not
globally alignable; connected bipartite graphs retain a multiplicative scale ambiguity; a connected
non-bipartite graph plus one orientation anchor can identify source reliability in the ideal model.

This is not a claim that federated weak supervision is new. WSHFL already mines and shares
parameterized candidate LFs, and US20230237321A1 already transfers weak labels through cross-client
sample similarity without sharing LF code. The only possible new core is the identifiability and
recovery of partially overlapping private source reliabilities from aggregate co-observation
statistics. WRENCH and BOXWRENCH provide public weak-supervision tasks and code.

Verdict:

```text
source-graph weak supervision: CONDITIONAL GO FOR THEORY GATE
global survival concordance:   HIGH-RISK BACKUP ONLY
implementation / experiment:   NONE / NONE
commit:                        NONE (user requested no proactive commits)
```

Evidence:

```text
docs/research/status/CCF_B_FEDERATED_TOPIC_RESET_SCREEN_2026_08_17_ZH.md
```

Next action: paper-only impossibility and identifiability audit for the source-graph candidate. It
must handle disconnected/bipartite graphs, abstention, client-dependent LF accuracy, finite-sample
error, natural WRENCH/BOXWRENCH domain partitions and exact collisions with WSHFL/FlyingSquid/the
weak-supervision patent. Do not implement or run experiments before this gate passes.

## Latest Theory Gate: Federated Ordinal Boundary Completion Is NO-GO

The proposed topic on completing missing ordinal severity boundaries across clients has completed
its paper-only audit. The application is real, and the direct Federated Ordinal Learning study
already observes that missing adjacent classes reduce the information needed to learn ordinal
boundaries. However, the candidate does not create a new identifiable FL object.

For homogeneous models, every cumulative cutpoint risk remains a weighted sum of client risks, so
one-step FedSGD exactly recovers the centralized gradient even when each client's local cutpoint
optimum diverges. Multiple-local-step bias is ordinary client drift; cutpoint-wise prevalence
correction collides with FedLC, while class deficiency is covered by FedGELA and SSDI. The proposed
boundary-complementarity statistic reduces to `Var_w(p_i,k)` via the law of total variance.

For heterogeneous models, client score/cutpoint scales are incomparable without shared parameters,
common inputs or a shared proxy, and label counts plus boundary support cannot identify the missing
input-label mapping. Adding common inputs reduces the protocol to FedMD/FedH2L plus ordinal
encoding; adding a shared proxy reduces it to FedeKD-style reliable bidirectional distillation.
FedeKD already evaluates model/data heterogeneity on RetinaMNIST and Diabetic Retinopathy.

Verdict:

```text
ordinal severity task / missing-boundary phenomenon: REAL / ALREADY DOCUMENTED
client-boundary observability diagnostic:            VALID BUT SIMPLE
homogeneous FL method novelty:                        NO-GO
heterogeneous FL without bridge:                      UNIDENTIFIABLE
heterogeneous FL with bridge:                         KD COMPOSITION / NO-GO
implementation / experiment / commit:                 NONE / NONE / NONE
```

Evidence:

```text
docs/archive/methods/FEDERATED_ORDINAL_BOUNDARY_COMPLETION_AUDIT_2026_08_17_ZH.md
```

Do not implement boundary-support weighting, cutpoint teacher routing, ordinal public-logit
distillation or a shared ordinal proxy. Select a new problem only after establishing a target that
is neither additively solved by standard federated gradients nor bridged by existing heterogeneous
KD.

## Latest New-Topic Screen: No Paper-GO Candidate Outside Model-Heterogeneous HFL Yet

The user has explicitly allowed the new paper topic to leave model-heterogeneous HFL and rejected
any CLE-HFL + PEW/BER benchmark fallback. A final problem-first screen therefore removed corruption,
public logits, heterogeneous backbones and KD from the required core.

Federated backward-compatible representations are paper `NO-GO`: centralized BCT, BC-Aligner and
multi-generation compatibility already cover the mechanism, and federating their local loss does
not create a new FL object. Client optimizer autonomy is `NO-GO` because Federated Blended
Optimization already treats heterogeneous local optimizers as black boxes, while FedPM/FedPAC
directly cover incompatible preconditioner geometries. Cross-client deduplication is `NO-GO` due
EP-MPD and FedRW. Heterogeneous label maturity/delay is real but currently reduces to local
inverse-maturity correction plus delayed-feedback learning and ordinary aggregation, so it is also
method `NO-GO`.

The final theory gate also rejected repeated adaptive use of client-private validation sets. Across
rounds, server proposals do depend on reused audit responses, so the statistical problem is real.
Define the useful diagnostic:

```text
FVI_i(T) = I(H_T ; A_i | training transcript and other clients' audits).
```

For sub-Gaussian loss, the expected client validation optimism is controlled by
`sqrt(2 sigma^2 FVI_i(T) / m_i)`. However, Reusable Holdout/Thresholdout already permits arbitrarily
adaptive queries based on the entire previous transcript. Running one mechanism per client remains
valid when the server uses other clients' answers to construct later proposals: relative to a target
audit set, the rest is adaptive post-processing and mechanisms on disjoint data. Thus the per-client
FVI vector adds reporting granularity but no new joint validity mechanism. DP-Hype additionally
covers local evaluation, noising, secure aggregation and private federated selection.

Verdict:

```text
federated adaptive validation-reuse problem: REAL
per-client FVI diagnostic:                   VALID BUT NOT A NEW METHOD
paper core / implementation / experiment:    NO-GO / NONE
all PEW/BER or model-HFL fallbacks:           EXCLUDED BY USER
```

Evidence:

```text
docs/research/status/FEDERATED_NEW_TOPIC_FINAL_SCREEN_2026_08_17_ZH.md
```

Next action: obtain the user's boundary decision: either the new topic may leave federated learning
entirely, or it must remain in FL while accepting new real data / observable side information.
Do not change code, run experiments or commit.

## Latest Theory Gate: Client Architecture Transition Core Is NO-GO

The conditional primary on personalized knowledge continuity after the same client changes
architecture has completed its paper-only identifiability/nontriviality audit. If the old client
function is queryable on the relevant input distribution, ordinary local old-to-new distillation is
the optimal projection into the new hypothesis class for the pure preservation objective; no
federated transcript can lower that objective. If old outputs are unavailable or observed only on a
finite transfer set, two worlds can have identical local observations and federated transcripts but
opposite old personalized functions on an unqueried positive-mass region, so uniform recovery is
impossible.

Adding current federated knowledge changes the objective from preservation to collaborative task
improvement and reduces to historical-local + current-global dual/multi-teacher distillation. This
directly collides with pFedSD/comprehensive KD, pFedKT and FedPSD, while AdaptFL and FedKDNAS already
cover real-time resource changes, changing client architectures and heterogeneous KD. Cross-
architecture KD and adaptive dual-teacher CAKD cover the remaining transfer mechanism.

Verdict:

```text
architecture-transition scenario reality:       GO
ATR / FTG as controlled evaluation metrics:      GO
FL-specific identifiable preservation target:   NO-GO
non-dual-teacher method object:                  NO-GO
CCF-B core / implementation / experiment:        NO-GO / NONE
```

Evidence:

```text
docs/archive/methods/CLIENT_ARCHITECTURE_TRANSITION_IDENTIFIABILITY_AUDIT_2026_08_17_ZH.md
```

Do not implement an architecture-switch runner, old+global teacher loss, architecture-specific
temperature/projector/gate, or run local/OpenI experiments. The only survivor from the prior screen
is the architecture-conditioned collaboration-benefit diagnostic, and it remains a benchmark-only
conditional route rather than a CCF-B method paper.

## Latest Topic Convergence: One Conditional Primary and One Diagnostic Backup

A top-down screen compared six non-corruption topics against current literature, frozen negative
routes, existing infrastructure and a one-to-two-month CCF-B window. Proxy-data coverage, receiver
capacity/learnability, heterogeneous label spaces and model-heterogeneous federated unlearning all
have direct collisions and are paper `NO-GO` for this project.

The only conditional primary is personalized knowledge continuity when the *same client* changes
architecture during federation. AdaptFL already assigns resource-specific architectures under
round-wise resource changes, and FedKDNAS selects an architecture every round while exchanging
public-reference logits. Therefore dynamic architecture selection itself is not new. The remaining
question is whether architecture-switch loss and personalized-knowledge continuity admit an
FL-specific object that beats the strongest local old-to-new KD and cannot reduce to an old-local +
current-federation dual teacher. This has not passed its theory gate and must not be implemented yet.

The sole backup is a diagnostic/benchmark on architecture-conditioned collaboration benefit, using
matched data twins across backbones to separate architecture from data. Collaborative fairness,
individual benefit over Local and low-end-device inclusiveness already exist, so this is not yet a
method-paper contribution and is retained only if the user accepts a diagnostic paper.

Evidence:

```text
docs/research/status/CCF_B_TOPIC_CONVERGENCE_MATRIX_2026_08_17_ZH.md
```

Next action: paper-only identifiability/nontriviality audit for architecture transitions. Do not
change code, run experiments, revive frozen routers, or commit for this screen.

## Latest Theory Gate: Missed-Knowledge Path Is Valid but Paper NO-GO

The ordered replay gate for a returning model-heterogeneous client has completed without code or
experiments. A minimal non-injective student-response construction proves that endpoint-only and
mean-teacher distillation can remain at an information-degenerate point, while chronological and
reverse replay select opposite final branches. Thus order can contain information absent from the
final aggregate teacher. A standard contraction bound also relates recovery error to initial
staleness, weighted teacher-path compression error, and local optimization/capacity error.

The paper gate nevertheless fails. Pro-KD, progressive distillation's implicit-curriculum theory,
Continuation-KD and curriculum extraction directly cover the advantage of intermediate teacher
checkpoints over a final teacher. FAPD already brings progressive, capacity-aware distillation into
federated learning; FedGKD uses historical global teachers; FedLFH uses client historical
trajectories. Path-variation recovery bounds are standard dynamic-regret/tracking results. The exact
returning-client/public-logit protocol was not found verbatim, but its method is an obvious
composition rather than a new CCF-B core.

Verdict:

```text
ordered-path mathematical phenomenon: GO
new KD principle:                     NO-GO
CCF-B rejoining-client core method:   NO-GO
implementation / experiment:          NONE
```

Evidence:

```text
docs/research/status/REJOINING_HETERO_LOGIT_RECOVERY_AUDIT_2026_08_17_ZH.md
docs/archive/methods/MISSED_KNOWLEDGE_PATH_THEORY_GATE_2026_08_17_ZH.md
```

Keep data-corruption x model-HFL parked as the current submission mainline, preserve all existing
RAHFL/CLE-HFL/PEW+BER assets, and do not implement or tune the missed-path candidate.

## Latest New-Topic Audit: Public-Logit Privacy Is Real but Not a New Core

The public-logit privacy route was audited against the exact current protocol. Every round exposes
per-client, per-public-sample 10-way probability vectors before receiver-specific AsymHFL routing.
The resulting transcript can support label-distribution and membership inference; this is a real
risk, not merely a paper story.

It does not pass the novelty and feasibility gates. PoPETs 2025 already attacks public-dataset-assisted
federated distillation, including CIFAR-10 private / CIFAR-100 public. USENIX Security 2024 directly
studies architecture-dependent privacy leakage. FedMD-NFDP, one-shot/noisy federated KD,
Selective-FD, CoFedMID and secure heterogeneous FD aggregation surround the available defenses.
Under formal DP, the worst-case sensitivity of probability releases is architecture-independent;
under empirical architecture-aware perturbation, protection is attack-specific and provides no
worst-case privacy claim. Privacy-budgeted teacher-query routing is the only less-directly occupied
formulation, but query utility is counterfactual and fewer deterministic queries do not themselves
provide DP.

Verdict:

```text
privacy risk in the current communication:              REAL
heterogeneous leakage as an empirical audit:             VALID BUT WEAK
architecture-aware logit release as a paper core:        NO-GO
privacy-budgeted teacher-query routing:                  CONDITIONAL, THEORY INCOMPLETE
implementation / local attacks / OpenI:                  NONE
```

Evidence:

```text
docs/archive/methods/MODEL_HETERO_PUBLIC_LOGIT_PRIVACY_AUDIT_2026_08_17_ZH.md
```

Do not implement per-backbone noise, temperature, top-k, argmax, query dropping or an attack-AUC
controller. The next independent topic screen should examine late-joining/intermittent heterogeneous
clients and must first distinguish itself from asynchronous FL, stale-update revival and continual HFL.

## Latest Theory Audit: Compound Degradation Does Not Rescue the Current Paper Core

The compound-degradation route was formalized without corruption labels, quality metadata, client
filtering or sample downweighting. Its sole candidate, Federated Compound Interaction Risk (FCIR),
measures whether adding an augmentation after an existing augmentation history creates positive
loss synergy beyond the same augmentation's singleton damage. A valid telescoping bound controls
long compositions when every prefix interaction is bounded.

This is only a closed augmentation-library result. It cannot cover unknown real corruptions without
a label-preserving augmentation-to-corruption coverage and loss-smoothness assumption. The local
behavior is surrounded by AugMix, AugMax, CoCor and representation straightening; communicating
class-by-augmentation interaction tables reduces to FedAvP-style shared augmentation policy. The
route also has no mechanism that increases training quality in the private weak class-environment
cells responsible for BER's WCCA/CFG gains.

Verdict:

```text
compound degradation as an evaluation extension: CONDITIONAL GO
compound degradation as a new problem claim:     NO-GO
FCIR closed-library mathematics:                  VALID BUT WEAK
FCIR as a standalone paper core:                  NO-GO
implementation / local / OpenI:                   NONE
```

Evidence:

```text
docs/archive/methods/COMPOUND_DEGRADATION_INVARIANT_KNOWLEDGE_AUDIT_2026_08_17_ZH.md
```

Do not implement FCIR, shared augmentation-interaction communication, or a renamed compound
AugMix module. If the user still rejects a taxonomy-assisted benchmark paper, stop the data
corruption x model-HFL method mainline and select a new topic with an observable target and a
short, existing-data experimental path.

## Latest Theory Audit: Latent Degradation Risk Identifiability Is Not a New Paper Core

The proposed paper route formalized class-conditional latent degradation risk as
`r_c = Pi_c rho_c`, constructed observationally equivalent worlds with different hidden
worst-cell risks, and derived the known-mixture rank condition, transcript invariance and the
additional confounding caused by model-specific risk vectors. These statements are mathematically
valid and explain why ordinary public-logit communication cannot replace missing environment side
information.

They do not pass the novelty gate. Unknown-group worst risk is already covered by fairness without
demographics and its federated version; aggregate-statistic bounds and partial identification have
direct precedents; multi-mixture component identifiability is covered by mutual-contamination theory.
The proposed heterogeneous shortcut-transfer estimand also collides with knowledge-distillation
mechanism-transfer work and AugHFL/RAHFL's corrupted-knowledge motivation.

Verdict:

```text
latent degradation risk as an internal diagnostic: VALID
P0 identifiability as a standalone paper core:      NO-GO
P1 shortcut transfer as a standalone paper core:    NO-GO
implementation / offline audit / experiment:        NONE
```

Evidence:

```text
docs/archive/methods/LATENT_DEGRADATION_RISK_IDENTIFIABILITY_AUDIT_2026_08_17_ZH.md
```

Do not implement identified-set optimization, communication difference-in-differences, or another
hidden-environment proxy. The next strategic choice must be a transparent PEW+BER empirical paper,
a real-metadata/observable-fault problem change, or leaving the data-corruption x model-HFL mainline.

## Latest Theory Audit: Safe Model-Heterogeneous FTTA Is Paper NO-GO

The proposed safe collaborative continual test-time adaptation setting was audited before any
implementation. With no target task labels, two worlds can share identical source data, target
images, public responses, model outputs and communication histories while reversing whether a
collaboration helps or harms. Therefore the sign of target collaboration harm is not identifiable
from the current observation set. Source audit certifies source behavior only.

The usual repairs do not pass the paper gate. Covariate shift plus overlap is mathematically
sufficient but untestable and implausible for arbitrary information-destroying corruptions. A
calibrated unlabeled loss proxy assumes the key risk relation and collides with AETTA and NeurIPS
2025 TTA risk monitoring. Collaborative and continual FTTA are already covered by FedTHE, ATP,
FedTSA, CoLA, FedCTTA and Latte; adding arbitrary backbones is an implementation intersection,
not yet a new identifiable object.

Verdict:

```text
fully-unlabeled safe collaborative MH-CTTA: PAPER NO-GO
implementation / local smoke / OpenI:       NONE
sparse unbiased delayed target labels:      CONDITIONAL REFRAME ONLY
```

Full report:

```text
docs/archive/methods/SAFE_MODEL_HETERO_FTTA_IDENTIFIABILITY_AUDIT_2026_08_17_ZH.md
```

The delayed-label route requires shadow candidates or randomized actions to estimate
counterfactual collaboration harm and changes the task to delayed-supervision online/continual
learning. Do not implement it unless the user explicitly accepts that problem change.

## Latest Candidate Audit: FCNT / FPER / FRT Do Not Enter Implementation

Three explicit side-information routes were formalized and checked against both frozen project
evidence and primary literature without changing training code or running experiments:

```text
FCNT: continuous nuisance coordinates + class-conditional federated OT
FPER: paired restoration intervention + degradation-effect risk
FRT:  public multi-view response tensor factorization
```

None passes the current core-method gate. FCNT is surrounded by CCDB/FG-CCDB,
class-conditioned Wasserstein DRO, FedWaD/FedDaDiL and SLOT-Align; a Wasserstein barycenter
also provides no lower mass bound for latent weak cells. FPER requires an unverifiable
label-preserving nuisance-removal oracle, does not prevent minority-cell dilution and collides with
counterfactual invariance/generation plus the frozen C3R/FedCISA reasoning. FRT lacks an identifiable
semantic/shortcut decomposition, cannot connect public responses to private weak-cell mass, and
repeats the public multi-view/shared-subspace risks already rejected by CCAD/IRD/FedCIS/EBST.

Verdict:

```text
FCNT current-protocol core:       NO-GO
FCNT with explicit real metadata: CONDITIONAL REFRAME ONLY
FPER observed-only core:          NO-GO; paired/clean ORACLE ONLY
FRT communication:               NO-GO
implementation / experiment:     NONE
```

Full report:

```text
docs/archive/methods/FCNT_FPER_FRT_THEORY_NOVELTY_AUDIT_2026_08_17_ZH.md
```

The next action is a strategic route choice, not a fourth taxonomy-free module: explicitly add a
realistically available side-information assumption, retain PEW+BER for a conservative empirical
paper, or stop CLE as the method-paper mainline. Do not implement or run any of these three candidates
before the user selects the route.

## Latest Paper-Claim Audit: PEW+BER Is Baseline GO, Core-Method NO-GO

The implementation, existing evidence and external literature were audited without
running experiments. The current exact method is a six-way public synthetic
corruption-family classifier followed by class x predicted-environment reweighted
ERM. It does not read private operator metadata for training, but it is not
taxonomy-free. BER is neither GroupDRO nor CVaR; it is a support-shrunk grouped
average risk and does not define a new robust-optimization principle.

External collision is material: Corrupted CIFAR-10 in Learning from Failure
(NeurIPS 2020) already couples labels with corruption types; SSA/BARACK already
use predicted spurious/group attributes for downstream robust training; CCDB and
FG-CCDB directly study class-conditional distribution balancing. CLE-HFL can be
positioned only as a controlled model-heterogeneous federated extension with
client-specific mappings and operator-cell evaluation, not as the first
class-corruption entanglement problem.

Verdict:

```text
PEW+BER empirical mechanism on fixed CLE:       GO
PEW+BER taxonomy-assisted diagnostic baseline:  GO
PEW+BER as the sole paper-level core method:     NO-GO
CLE-HFL as a federated benchmark extension:      CONDITIONAL GO
```

Do not spend the next stage merely adding exact PEW+BER seeds/rounds or renaming
the scenario. Preserve PEW+BER as the positive anchor; a new candidate must add
an FL-specific mathematical object and pass paper-level collision checks before
implementation. Full evidence:

```text
docs/research/status/PEW_BER_PAPER_CLAIM_AUDIT_2026_08_16_ZH.md
```

## Current Objective

The user has rejected all CLE-HFL/PEW+BER benchmark fallbacks and allowed the submission topic to
leave model-heterogeneous HFL while clarifying that heterogeneous models remain allowed and may be
desirable. Model heterogeneity is therefore a permitted protocol condition, not a required novelty
claim. The first broader FL screen and the subsequent Federated Ordinal Boundary Completion audit
have no paper-GO survivor. Preserve all prior assets and negative evidence, but do not force the
topic back into corruption, KD, public logits or architecture heterogeneity. There is currently no
active method candidate.

## Latest Theory Result: LCC - NO-GO Before Implementation

Latent Correction Conflict (LCC) was formalized as class-conditioned
per-sample last-layer gradient grouping followed by a minimum-norm common
descent update. It requires no environment labels and differs from the frozen
project methods, but it does not pass the external novelty gate:

```text
gradient clustering -> latent robust groups   GRASP collision
minimum-norm common descent                   MGDA/CAGrad collision
last-layer gradient KNN soft neighborhoods    GoG (KDD 2025) collision
```

Verdict: `THEORY NO-GO`. Do not implement LCC, change its clustering/graph, or
spend GPU/OpenI time on it. Evidence:

```text
docs/archive/methods/LCC_NOVELTY_AUDIT_ZH.md
```

## Taxonomy-Free Identifiability Boundary

Using client identity as an unlabeled mixture view was also checked before
turning it into a communication module. For class `c`, observable client risks
satisfy `r_c = Pi_c rho_c`; centered client contrasts can identify at most
`K-1` environment-risk directions. In the frozen four-client CLE mapping, the
effective family-contrast ranks by class are:

```text
class: 0 1 2 3 4 5 6 7 8 9
rank:  1 2 1 3 1 1 2 2 0 2
```

Only class 3 has full four-family contrast coverage; class 8 has none. Model
heterogeneity further confounds client-risk differences. Pure client-class
variance/DRO therefore cannot replace BER with a clean guarantee and must not
be promoted as the next communication innovation. Evidence:

```text
docs/research/status/TAXONOMY_FREE_IDENTIFIABILITY_2026_08_11_ZH.md
scripts/audit_mixture_contrast_identifiability.py
```

## Latest Result: CRSR Audit 0 - NO-GO

Class-conditional Residual Spectral Risk (CRSR) used only fit-internal labels
and predictions:

```text
r(x,c)   = softmax(f(x)) - one_hot(c)
Sigma_c  = Cov(r | y=c)
S_c      = sqrt(lambda_max(Sigma_c))
L_CRSR   = class-balanced CE + 2.0 * mean_c S_c
```

The frozen local Audit 0 completed on client 1/ResNet12 and client
3/MobileNetV2 without reading private audit or final test. Operator IDs were
used only for post-hoc cell evaluation. Independent recomputation matched the
script: G0--G3 passed; G4--G6 failed.

```text
median top share             0.752411  PASS
median direction cosine      0.975226  PASS
median transfer advantage    0.639379  PASS
median spectral cell rho     0.069658  FAIL (< 0.25)
median advantage vs baseline -0.903889 FAIL (< 0.02)
mean CE delta, clients 1/3   +0.006485 / +0.000360  FAIL
worst-cell CE delta          +0.101343 / -0.045068  FAIL
verdict                      NO-GO
```

Interpretation: the class-residual spectrum is active, stable across disjoint
splits, and nonredundant with sample CE/Brier, but it does not consistently
identify weak class-operator cells and its optimization is not mean-risk
noninferior. Freeze CRSR. Do not tune its weight, support thresholds, probe
size, power-iteration count, or gates; do not connect it to the runner or run
12/40 rounds.

Evidence and retained isolated implementation:

```text
docs/experiments/archive/CLASS_RESIDUAL_SPECTRAL_RISK_AUDIT_ZH.md
outputs/class_residual_spectral_risk_audit0/result.json
outputs/class_residual_spectral_risk_audit0/signals.npz
fedprime/methods/class_residual_spectral_risk.py
scripts/audit_class_residual_spectral_risk.py
tests/test_class_residual_spectral_risk.py
```

## Current Formal Positive Result

The selected local path remains calibrated hard PEW + hard BER; the legacy
strict three-seed positive package also included the then-active CDep term.
On fixed CLE-HFL v2 `seed0_split0`, that package's matched 12-round
training-seed 0/1/2 deltas versus AugMix/JSD/DCL control were:

```text
mean Avg +4.5880, Worst +4.2169, WCCA +5.5500, CFG -6.7150
```

The 40-round training-seed-0 durability result also passed all frozen gates.
These results establish the empirical target to preserve, not a defense of
PEW's five-family taxonomy.

## Other Frozen Recent Negatives

```text
Multi-label PEW + Soft-BER: NO-GO (0/4 matched last-five gates)
PIE/MPIE: NO-GO; do not implement PBR
C3R: NO-GO; do not implement its training loss
CRSR: NO-GO; stable geometry but invalid weak-cell surrogate
```

Also obey the permanent frozen-negative list in `AGENTS.md`; do not revive
FedCIS, continuous-witness, IRD/PCCD, or communication methods already archived
as negative. A new object must additionally be distinguished from
GroupDRO/CVaR and CCAD instead of merely renaming their objective.

## Next Action

Report the ordinal-boundary `NO-GO` and start no implementation. The next independent topic screen
must begin from an observable target that standard additive federated gradients cannot already
solve and that existing KD/proxy communication cannot trivially bridge. Homogeneous and
heterogeneous models are both allowed; architecture heterogeneity alone is not the contribution.
Do not run local/OpenI experiments or commit during topic selection.
