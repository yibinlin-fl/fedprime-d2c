# FedPRIME-D2C / PRAC-HFL Current Project Memory

Updated: 2026-09-04

## CLE PIDR Zero-Training Hidden-Binding Gate - 2026-08-30

The PIDR paper-only identifiability audit was followed by the approved zero-training gate. The new
analyzer reused Phase-A1a round12/round40 softmax caches only; it ran no training and no inference.
Promotion matrices were completed using only probabilities, task labels and probe tensor position.
The class-to-family binding and operator-family ids were withheld until final scoring and two
independent permutation nulls.

Formal round40:

```text
arm   PIDR       mAP       AUC       positive P/R       class-family hit
H0    0.024559   0.441855  0.510677  0.2465 / 0.4688   0.225
H9    0.175479   0.844847  0.923906  0.5691 / 0.9063   0.850
L0    0.025401   0.430622  0.507083  0.2448 / 0.4313   0.275
L9    0.174728   0.865557  0.933177  0.5878 / 0.9250   0.875
```

HFL/Local mAP deltas were `+0.402993/+0.434935`; both had 4/4 positive client deltas. H9 and L9
each achieved `p=0.000999` against both shuffled class maps and shuffled probe identities. All six
frozen gates passed for both systems; verdict `GO_TO_INTERVENTION_BRIDGE_DESIGN`. Round12 already
showed H9/L9 mAP `0.8135/0.8360` and hit `0.80/0.875`.

This is an oracle observability GO, not a method GO. It proves distinguishable probes can reveal the
hidden directional binding without providing the estimator a family taxonomy. It does not prove that
ordinary i.i.d. AugMix views overwrite an existing base corruption, that clean-source-free training is
identified, or that PIDR is novel relative to ShortcutProbe/FedCD. The next stage is paper-only probe-
bridge design; no training loss or 12-round A/B is authorized.

```text
code:       fedprime/engine/cle_probe_directional_promotion.py
analyzer:   scripts/analyze_cle_pidr_zero_training_gate.py
tests:      tests/test_cle_probe_directional_promotion.py
result:     deliverables/cle_pidr_zero_training_gate_20260830/
verification: 10 passed
commit:     none
```

## CLE Shortcut Communication Amplification Phase-A1a - 2026-08-30

The matched four-arm OpenI attribution completed and was independently recomputed from both returned
probability caches. It used CIFAR-10 CLE-v1, alpha 0.5, seed 0, four heterogeneous clients, no
pretraining, 40 rounds, one local epoch, AugMix/JSD+DCL, a persisted private 85/15 fit/audit split,
and shared initial weights. `H0/H9` used strict audit-only AsymHFL-val routing; `L0/L9` used exact
no-op communication. Final-test labels were reporting-only.

All integrity checks passed: source indices, labels and severity draws were paired across gamma;
the split and initial states were shared; round-12/final checkpoints existed; and every HFL/Local
pair had identical first-batch labels and all AugMix/DCL views for all 40 rounds and four clients.
The two probability tensors had shape `[4,4,1000,16,10]`, were finite and normalized. Six relevant
OpenI source-file hashes match the local checkout exactly. OpenI did not retain `.git`, so the
contract records `base_git_commit=UNAVAILABLE`; file hashes bind it to the pushed `6199dd8` code.

Formal round-40 DSA:

```text
arm                  DSA
H0                   0.0015228341
H9                   0.2042704937
L0                   0.0008234010
L9                   0.2051892788
HFL CLE effect       0.2027476596
Local CLE effect     0.2043658778
A_pool              -0.0016182182
CI95                [-0.0033365882, 0.0001283891]
A_client            [-0.0027690681, -0.0023939108, -0.0016801117, +0.0003702177]
top1 amplification   0.0025799851
H9 shuffled p        0.000999001
```

Only 1/4 clients had positive amplification. G1 minimum effect, G2 positive CI and G3 client
direction failed; G4 top-1 direction and G5 binding-specificity passed. Formal decision:
`NO_GO_FL_SPECIFIC_AMPLIFICATION`. Round 12 showed temporary suppression
`A_pool=-0.0169168048`, CI `[-0.0183210229,-0.0155504985]`, which vanished by round 40 and is not a
durable mitigation claim.

Round-40 secondary metrics:

```text
arm   Avg       Worst     WCCA      CFG       paired flip
H0    52.0125   44.6063   31.5625   2.8063    0.1460
H9    45.9641   38.1063   11.8125  11.7625    0.3612
L0    52.7969   44.6188   34.5625   2.8313    0.1525
L9    45.9000   38.1063    8.3125  12.0750    0.3591
```

Scientific interpretation: CLE robustly causes a directional corruption-to-class shortcut and a
large robustness--invariance gap, but the shortcut is already learned locally. Strict AsymHFL does
not materially amplify or durably suppress it. Do not use bad-teacher propagation, wrong routing or
communication amplification as the CLE paper story, and do not run seed sweeps, router tuning or
more rounds to rescue this rejected hypothesis. The paired bootstrap quantifies evaluation-source
uncertainty, not training-seed variability.

The user explicitly keeps CLE as the project direction. The active research target is now a
local-first, environment-label-free directional shortcut suppression module. Before implementation,
it must define an observable and differentiable risk using only the allowed training information,
avoid clean-source/corruption-label/source-index oracles, and pass a collision audit against
AugMix/JSD, AugMax, consistency regularization, IRM/VREx, GroupDRO/CVaR, counterfactual invariance,
PEW/BER, C3R, CCAD/IRD and all frozen communication routes. No method is yet approved for code.

Artifacts:

```text
code commit: 6199dd8 (origin/main)
input archive: local_runs/cle_shortcut_amplification_phase_a1a/
               cle_shortcut_amplification_phase_a1a_seed0.tar.gz
input bytes: 408228487
input SHA256: 6322F16513C6980CDC5904D7EF91204A241205BC76DCCE8BC450E635519B4202
result archive: outputs/openi_downloads/cle_shortcut_amplification_phase_a1a_seed0/
                cle_shortcut_amplification_phase_a1a_seed0_analysis_outputs.tar.gz
result bytes: 19322309
result SHA256: FDF1BEC2395334DD3816BC9C3F594B01D814DB22C0CC245CD1378A45180F397C
summary: outputs/openi_downloads/cle_shortcut_amplification_phase_a1a_seed0/extracted/
         outputs/cle_shortcut_amplification_phase_a1a_seed0_analysis/
         cle_shortcut_phase_a1a_summary.json
```

## CLE Directional Shortcut Alignment Phase-A0 - 2026-08-30

The frozen zero-training OpenI audit completed on eight historical RAHFL checkpoints: four from
matched `gamma=0.0` and four from `gamma=0.9`. It evaluated the same 1,000 balanced clean CIFAR-10
sources under all 16 historical CLE-v1 corruption operators at severity 3, for 128,000 total forward
items. All integrity checks passed. The returned summary was independently recomputed from the raw
probability cache with exact agreement.

```text
gamma00 pooled DSA: -0.0003018894
gamma09 pooled DSA:  0.2013210658
delta DSA:           0.2016229552
paired CI95:         [0.1964123272, 0.2072188988]
positive clients:    4/4
pooled shuffled p:   0.000999001
client shuffled p:   0.000999001 for all 4 clients
decision:            GO_TO_MATCHED_PARTITION_DESIGN; G1--G5 all PASS
```

Secondary changes were Avg `52.2422 -> 46.8250`, Worst `43.4188 -> 37.9375`, WCCA
`36.0000 -> 19.3125`, CFG `2.6500 -> 11.8813`, paired prediction-flip rate
`0.148325 -> 0.353404`, and family-bound top-1 bias `0.000010 -> 0.230527`.

Interpretation is deliberately limited: the historical strong-CLE models directionally exploit
corruption families as class evidence, while the independent-CLE models do not. This establishes
the shortcut mechanism and a robustness--invariance gap in the historical RAHFL protocol. It does
not identify federation, communication, teacher routing or model heterogeneity as the cause. The
next authorized scientific step is the already pre-registered matched-global Local-D/Local-E
attribution design. No new method or full training experiment is authorized by this result alone.

Provenance limitation: these are the final four-client checkpoints from the project-trained
`diag_rahfl_cle_alpha05_gamma00_seed0` and `diag_rahfl_cle_alpha05_gamma09_seed0` runs. Both resolved
configs use alpha 0.5, seed 0, AugMix/JSD+DCL+AsymHFL, `pretrain_epochs=0`, 40 communication rounds
and one local epoch per round. They are not the canonical 40-pretrain + 40-communication RAHFL
schedule. The post-hoc Phase-A0 comparison is internally matched and valid for this historical
diagnostic protocol, but a paper-level full-RAHFL claim would require a separately matched run. The
historical prepared v1 datasets are no longer local, so their source/label identity cannot be
re-hashed; the deterministic generator uses identical alpha, seed and partition seed across gamma.
Future promoted runs must persist explicit source-index, label and partition hashes.

Artifacts:

```text
code commit: e005689 (origin/main)
result archive: outputs/openi_downloads/cle_shortcut_alignment_phase_a0_seed0/
                cle_shortcut_alignment_phase_a0_seed0_outputs.tar.gz
result bytes: 4883205
result SHA256: CCC91ED4EC3F08C6CA6433CA275423AD3E32EDB824993F7760F94A8A49B4B76F
```

## Corruption/Robustness Topic-Pool Closure and New Search Rule - 2026-08-25

The project has formally stopped using the search pattern `RAHFL + one additional difficulty`.
That pattern repeatedly produces combinations already occupied by mature neighboring literature:
attack, label noise, dynamic corruption, resource heterogeneity, availability and compression. It
also encourages post-hoc renaming of a known problem rather than discovering an independent
federated object. No current candidate is authorized for implementation or experimentation.

The current topic-pool decisions are:

```text
topic                                      decision
severity heterogeneity                     NO-GO
compound corruption                        PAPER-CORE NO-GO; evaluation-only at most
corruption--label coupling                 SCREEN NO-GO; beta4 outperformed beta0
communication-induced robustness forgetting LOW-VALUE DIAGNOSTIC; DO NOT INVEST
architecture-agnostic robustness propagation COMPOSITION-RISK CONDITIONAL; NOT ACTIVE
data cleaning / damaged-sample removal      NO-GO for the intended paper narrative
availability--corruption coupling           NO-GO due literature collision
dynamic corruption                         NO-GO / highly crowded
compression robustness                     HIGH COMPOSITION RISK; NOT ACTIVE
adversarial attack / dual robustness HFL    FORMAL NO-GO
PEW+BER                                    STRONG BASELINE; CORE-METHOD NO-GO
```

The attack route is frozen broadly, not only for one proposed experiment. Do not implement the
one-shot attack Phase-A0; do not add PGD, FGSM or AutoAttack; do not implement malicious-logit,
poisoning or backdoor experiments; and do not modify RAHFL for an attack defense. Literature already
covers the relevant intersection through model-heterogeneous adversarial robustness, federated-
distillation logit poisoning and trustworthy heterogeneous logit fusion. Narrowing the story to
`robustness-profile conflict`, `teacher-ranking conflict`, `routing amplification` or
`corruption-camouflaged poisoning` is insufficient when the resulting method remains anomaly
detection, trust weighting or robust logit aggregation.

The architecture-agnostic common-corruption robustness-propagation idea is not recorded as a GO.
It remains only a composition-risk hypothesis adjacent to FedRBN, FedERL/DART, AugHFL/RAHFL and
robust distillation. No Phase-A harness or pretraining should be started unless a later paper-level
audit establishes a non-compositional mathematical core.

Future topic selection must reverse direction:

```text
effective centralized robustness method/objective
    -> identify information necessary for its correctness
    -> show that federation removes or fractures that information
    -> prove the fracture is not solved by additive local gradients, ordinary KD or a proxy
    -> audit exact federated collisions
    -> only then define a minimal falsification experiment
```

Candidate structural fractures include, but are not automatically research contributions: missing
global data distributions, unavailable cross-sample pairs, missing global hard-example rankings,
unshareable cross-domain statistics, incomparable heterogeneous parameter spaces, unavailable
stable/unstable feature identities, absent centralized reference models and unavailable global
corruption statistics. Every new candidate must state the centralized problem, why the centralized
method works, exactly what information is lost in FL, the weakest repair assumption, why the setting
is non-toy, available data/code, direct 2024--2026 collisions and the cheapest Kill Test before any
implementation.

Current operational state:

```text
active paper method: none
active implementation: none
active local/OpenI experiment: none
next action: paper-only reverse search from centralized robustness to federated structural fracture
git policy during topic selection: no proactive commit
```

## RAHFL Corruption-Dependent Label-Noise Coupling Phase-A - 2026-08-24

The active research screen has returned to corruption robustness for one narrowly defined causal
question. It does **not** revive PEW/BER or claim that class--corruption entanglement is new. It asks
whether equal-rate annotation errors become more damaging when they preferentially occur on
high-severity corrupted samples, and later whether the resulting coupling penalty is amplified by
the federated/RAHFL system relative to Local and Centralized training.

Phase-A1a freezes two paired worlds:

```text
beta=0: independent noisy-mask scoring
beta=4: severity-biased noisy-mask scoring

fixed across worlds:
images, four-client disjoint IID partition, exact 90/10 fit-audit split,
20% fit noise, per-stratum quotas, shared Gumbels, wrong-label destination
multisets and the aggregate true-class -> wrong-class transition matrix
```

Protocol details:

```text
clients: 4 heterogeneous RAHFL models
samples per client: 10,000, mutually exclusive across clients
fit/audit: exactly 9,000 / 1,000 per client
noisy fit labels: exactly 1,800 per client
pretraining: uses the regime's persisted noisy-label manifest
communication local CE/DCL: uses the same persisted noisy-label manifest
AsymHFL teacher routing: trusted clean client-private audit only
final corrupted test: reporting only
DCL: intentionally unchanged and therefore exposed to the noisy labels
```

The IID implementation distinction is now explicit. `Dataset/sampling.py::iid_sampling()` is
mutually exclusive, but the actual default IID branches in `Network/pretrain.py` and `HHF/RAHFL.py`
independently execute `np.random.permutation(50000)[:Private_Data_Len]` for each client and therefore
permit overlap. Phase-A uses a fixed disjoint IID partition as an experimental protocol choice; it
does not claim that the helper itself is buggy.

Implementation and verification were committed and pushed:

```text
commit: 999fee0 Add RAHFL label-coupling Phase-A harness
remote: https://github.com/yibinlin-fl/fedprime-d2c.git
branch: main
focused unit tests: 2 passed
beta0 1-pretrain-batch + 1-round smoke: passed
beta4 1-pretrain-batch + 1-round smoke: passed
formal training: not started
```

The generated artifact audit passed all frozen equivalence checks:

```text
unique client samples: 40,000
fit/audit per client: 9,000 / 1,000
noisy fit labels per client: 1,800
frozen image sha256: e20128a7ad506cf5ea0eccd386c81ad5b5bde8ea4ddf6dd6c9ab90e9d6a8435f
beta0 E[severity | noisy]: 2.5029
beta4 E[severity | noisy]: 3.5699
partition/split/noise count/flip matrices/test role: matched
clean audit and no audit gradient: verified
pretrain and local CE/DCL label source: identical persisted noisy manifest
```

Current cost-saving OpenI screen:

```text
beta0: 10 pretrain epochs + 10 communication rounds
beta4: 10 pretrain epochs + 10 communication rounds
entry: scripts/openi_rahfl_coupling_phase_a_entry.py
arguments: --mode=both
status: completed on 2026-08-24
```

Prepared OpenI dataset:

```text
file: outputs/rahfl_coupling_phase_a_seed0_prepared.tar.gz
upload name: rahfl_coupling_phase_a_seed0_prepared.tar.gz
size: 327,018,418 bytes (~311.9 MiB)
sha256: AE5F9524AF594963C790016FB386BD0EB600ACD84BBC1D12EF57DA7393D1835F
visibility recommendation: private
```

OpenI success-return behavior is frozen and remembered. The entry obtains the mounted dataset and
output paths through `c2net.context.prepare()`, safely extracts the exact prepared archive, runs the
artifact verifier, executes beta0 and beta4 sequentially, packages beta0 immediately after it
finishes, writes a paired descriptive summary after both finish, copies the archives/summary into
`context.output_path`, and finally calls `c2net.context.upload_output()`.

Expected returned files:

```text
rahfl_coupling_phase_a_screen_seed0_beta0_outputs.tar.gz
rahfl_coupling_phase_a_screen_seed0_beta4_outputs.tar.gz
rahfl_coupling_phase_a_screen_seed0_both_outputs.tar.gz
rahfl_coupling_phase_a_screen_seed0_summary.json
```

The 10+10 result is screening-only and cannot establish a formal paper claim. It nevertheless fails
the pre-registered promotion direction. Independent beta0-minus-beta4 recomputation gave:

```text
window       CP Avg      CP Worst
all-10       -0.40pp     -0.07pp
last-3       -2.01pp     -1.37pp
final        -1.90pp     -1.83pp
```

Beta4 was better, not worse. At round 0 before the first collaborative phase, beta4 clean-audit
accuracy averaged 50.775 versus 48.325 for beta0 (+2.45pp), and beta4 was higher for every client.
Thus the reverse signal already arose from local pretraining; the screen supplies no evidence that
RAHFL communication amplifies corruption-dependent label-noise harm. Teacher graphs were identical
in rounds 0--2 and 8--9 and differed only modestly in five middle rounds. Both conditions completed
rounds 0--9 and four final checkpoints under configs differing only in name and beta.

Decision: `SCREEN NO-GO`. Do not run 40+40, Local/Centralized causal diagnostics, intermediate beta,
noise-rate sweeps, DCL changes or AsymHFL changes for this hypothesis. Do not post-hoc invert the
claim into a paper contribution. A possible explanation is that mislabeled clear samples in beta0
provide stronger contradictory gradients than mislabeled low-information severity-4 samples, but
this is a local-learning interpretation, not an established FL-specific novelty. Full report:
`deliverables/rahfl_coupling_phase_a_screen_20260824/RESULT_SUMMARY_ZH.md`.

## Strict PEW + AsymHFL-val Formal Training-Seed Result - 2026-08-04

After FedCIS-v0 and the handcrafted continuous witness both failed their
offline gates, a strict attribution experiment compared:

```text
control   = AugMix/JSD/DCL + strict AsymHFL-val
candidate = AugMix/JSD/DCL + calibrated PEW/BER+CDep + strict AsymHFL-val
```

Both arms use CLE-HFL v2, the same four heterogeneous models, the same
persisted class-stratified 85/15 fit/audit split, four public CIFAR-100 batches
per round, and 12 rounds. Local gradients see only `fit`; AsymHFL teacher
ordering sees only each client's private `audit`; the final CLE test labels
are reporting-only. This removes the original RAHFL test-routing leakage.

Formal status:

```text
strict data/routing implementation: complete
OpenI A/B configs and entry: complete
local RTX 3050 one-round smoke: both arms passed
focused tests: 46 passed
formal 12-round seed-0 result: complete
independent recomputation: exact match
formal 12-round training seeds 1/2: complete
independent recomputation: exact match for seeds 0/1/2
verdict: GO for fixed-scenario training-seed stability
```

The smoke verified identical persisted splits and identical round-0 audit
accuracies in both arms, nonzero AsymHFL communication, and nonzero BER/CDep in
the candidate. The PEW operator-to-family mapping is used only for diagnostic
accuracy reporting; operator metadata never enters training or routing.

Run:

```text
entry: scripts/openi_strict_pew_asymhfl_entry.py
argument: --mode=both
data: cle_hfl_v2_prepared_alpha05_gamma09_seed0_split0.tar.gz
guide: docs/experiments/archive/STRICT_PEW_ASYMHFL_VAL_OPENI_RUN_ZH.md
```

Result archive and integrity:

```text
archive: outputs/strict_pew_asymhfl_val_probe_outputs.tar.gz
sha256: 77109f7a382b1271317a3afd89a30ae27170e8003977225a3ace5dd7ace9f3d9
extracted: outputs/strict_pew_asymhfl_val_probe_20260804/
members: 30
unsafe paths: none
symbolic/hard links: none
```

Candidate-minus-control:

```text
scope       Avg       Worst     WCCA      CFG
final       +5.1267   +2.9533   +8.7500   -7.7250
last-five   +3.9377   +3.9040   +5.0500   -6.3200
```

Extended last-five mean deltas were `worst_group_acc +6.00`,
`worst_client_group_acc +6.46`, seen Avg/Worst `+3.79/+3.95`, and unseen
Avg/Worst `+4.34/+3.78`. Candidate improved all four gated metrics in every
one of rounds 7-11. Round-0 `col_loss` was identical in both arms
(`0.2274829049905141`). The two resolved configs differed only in experiment
identity and the intended FedEASE/PEW/BER+CDep method fields.

All frozen last-five gates passed. This is evidence that calibrated PEW +
BER+CDep improves the matched strict AsymHFL-val pipeline on seed 0. This
seed-0 result triggered the pre-registered training-seed repeats documented
below; those repeats also passed. Do not rescue future failures with blind
lambda, threshold, or rank sweeps.

### Training-seed repeat result

The seed 1/2 repeats completed. Both kept `alpha05_gamma09_seed0_split0`, the
persisted strict fit/audit split, method parameters, models, and 12-round budget
fixed. Only the matched top-level training seed and output identity changed.

```text
seed 1: scripts/openi_strict_pew_asymhfl_entry.py --mode=both --train_seed=1
seed 2: scripts/openi_strict_pew_asymhfl_entry.py --mode=both --train_seed=2
aggregate: scripts/analyze_strict_pew_asymhfl_multiseed.py
guide: docs/experiments/archive/STRICT_PEW_ASYMHFL_VAL_MULTISEED_OPENI_RUN_ZH.md
```

Archive integrity:

```text
seed1 sha256: 7f4889fe20a11b7c446355b1b643fce1974f13a57f38e320c74a96a817cbd32e
seed2 sha256: 10d78afce660776cfcd95bcf0c12b420dac545c46b3810b00bfba44afd001eab
members per archive: 30
unsafe paths or links: none
partition sha256 (all three): 75c6bd9dc4b7714f505eea2c047f1b882582da311d00d099b6caac1b5ba4d2ec
```

Independently recomputed candidate-minus-control last-five deltas:

```text
seed   Avg       Worst     WCCA      CFG       full gate
0      +3.9377   +3.9040   +5.0500   -6.3200   PASS
1      +4.7977   +3.8893   +4.3500   -8.3000   PASS
2      +5.0287   +4.8573   +7.2500   -5.5250   PASS
mean   +4.5880   +4.2169   +5.5500   -6.7150
std     0.5749    0.5547    1.5133    1.4290
```

All three seeds passed the original full gate, and all nine pre-registered
multi-seed gates passed. Across every one of the 15 last-five seed-rounds,
candidate improved Avg, Worst, and WCCA while reducing CFG. Three-seed mean
extended deltas were `worst_group_acc +7.31`,
`worst_client_group_acc +7.20`, seen Avg/Worst `+4.48/+4.28`, and unseen
Avg/Worst `+4.90/+4.06`. Even the weakest seed direction remained positive;
the least favorable CFG delta was `-5.525`.

Decision: `GO` for training-seed stability on the fixed CLE scenario. This
passes the prerequisite for separately designing a 40-round durability probe.
It does not establish cross-scenario generalization because class partitions,
operator maps, and the fit/audit split remained fixed. Freeze current method
parameters and do not start a new paid run without an explicit user request.

### Prepared seed-0 40-round durability probe

After the training-seed repeat passed, the user requested preparation of the
fixed-scenario seed-0 40-round probe. It changes only `train.rounds` from 12 to
40 and experiment/output identity; it runs from scratch and does not load or
resume the 12-round checkpoints.

```text
entry: scripts/openi_strict_pew_asymhfl_40round_entry.py --mode=both
control: configs/openi_v100_rahfl_val_cle_v2_40round_probe.yaml
candidate: configs/openi_v100_fedease_pew_asymhfl_val_cle_v2_40round_probe.yaml
analyzer: scripts/analyze_strict_pew_asymhfl_40round.py
guide: docs/experiments/current/STRICT_PEW_ASYMHFL_VAL_40ROUND_OPENI_RUN_ZH.md
expected archive: strict_pew_asymhfl_val_40round_seed0_outputs.tar.gz
```

Pre-registered primary last-10 gates:

```text
Avg >= +1.5, Worst >= +1.0, WCCA >= 0, CFG <= -1.0
```

Pre-registered last-5 anti-collapse gates require strictly positive Avg/Worst,
nonnegative WCCA, and negative CFG deltas. All eight gates must pass. The
implementation passed 14 focused tests, both CLI help checks, config dependency
checks, and a synthetic 40-row analyzer dry-run with all eight gates producing
`GO`.

### 40-round seed-0 result and matched seed1/2 preparation

The returned seed-0 archive was safety-checked and independently reanalyzed on
2026-08-05. Both arms contain exact rounds 0-39, no core NaNs, and byte-matched
formal configs. The persisted split hash remains
`75C6BD9DC4B7714F505EEA2C047F1B882582DA311D00D099B6CAAC1B5BA4D2EC`.
The first 12 rounds exactly reproduce the earlier seed-0 formal run.

```text
last-10 candidate-minus-control:
Avg +4.9292, Worst +3.2987, WCCA +9.8750, CFG -5.4700
last-5 candidate-minus-control:
Avg +4.7140, Worst +3.1747, WCCA +9.7000, CFG -5.0800
verdict: GO (8/8 gates)
```

The user then explicitly requested 40-round training-seed repeats 1/2. The
same entry now selects them with `--train_seed=1/2`; scenario data and
`strict_fit_audit.seed=0` remain fixed. Do not tune method parameters or mix in
scenario-seed changes.

For a one-task overnight run, `--train_seed=all` executes pending seeds `[1,2]`
in order. Each seed is analyzed, packaged, and uploaded before the next begins,
so a seed2 failure does not lose a completed seed1 archive.

## Continuous Taxonomy-Free Witness Audit - 2026-08-03

After FedCIS-v0 failed, a separate taxonomy-free continuous nuisance witness
was implemented and audited. It uses a 22-dimensional deterministic descriptor
of channel moments, spatial derivatives, Laplacian response, saturation, and
radial spectrum. It never reads operator IDs/names/families/severity during
training logic.

The matched one-step branches were base, true witness, classwise shuffled
witness, and classwise moment/covariance-matched random witness.

```text
branch      Avg       Worst     WCCA    CFG       audit loss
base        87.8277   83.4000   0.0     88.0423   0.400166
true        87.8277   83.3333   0.0     86.7923   0.401964
shuffled    87.8444   83.6667   0.0     88.0423   0.401336
random      87.7111   83.0000   0.0     88.0423   0.402281
```

True witness improved CFG by `1.25`, but failed Worst, audit-loss, and target
coverage gates. It beat all controls on only `33.33%` of 33 auditable
client-class targets; the frozen requirement was 60%. Decision: NO-GO. Do not
implement a 12-round local probe or attach this version to AsymHFL.

Read: `docs/archive/methods/CONTINUOUS_WITNESS_OFFLINE_AUDIT_ZH.md`.

## FedCIS Offline Audit Result - 2026-08-03

FedCIS-v0 Audit A/B is implemented and completed locally on the RTX 3050
using all four CLE-HFL v2 RAHFL checkpoints, three independent AugMix seeds,
at most 16 fit samples per client/class, a 32-dimensional multiscale DCT
projection, and rank-4 class subspaces.

```text
valid classes per seed                         = 10/10
true class cross-seed subspace similarity      = 0.1673
class-shuffled cross-seed similarity           = 0.1669
equal-rank random similarity                   = 0.1197
cross-client matched-class similarity          = 0.1269
cross-client mismatched-class similarity       = 0.1318
auditable client x class attack targets        = 33
true orthogonal attack beats both controls     = 30.30%
frozen requirement                             = >=60%
verdict                                        = NO-GO
```

The true class-conditioned subspace is effectively indistinguishable from the
class-shuffled control, and matched classes are not more similar across clients
than mismatched classes. Therefore the central FedCIS identifiability
assumption is not supported by this audit. Do not implement Audit C, a FedCIS
runner, or 12/40-round FedCIS training. Do not tune only rank, epsilon, or loss
weights to bypass the frozen decision.

Implementation and artifacts:

```text
fedprime/analysis/fedcis.py
scripts/audit_fedcis_sensitivity.py
tests/test_fedcis_audit.py                 (8 passed)
local_test_outputs/fedcis_audit_20260803/
```

This is a negative mechanism result, not evidence that input sensitivity is
useless in general. It rejects this specific low-rank DCT, two-AugMix-view,
generalized-eigen FedCIS-v0 formulation under the current checkpoints and
four-client CLE-HFL v2 protocol.

## Current Candidate: FedCIS-v0 - 2026-08-03

The current formal problem remains the four-client CLE-HFL v2 protocol. The
new candidate is:

```text
FedCIS-v0
= AugMix/JSD/DCL robust local base
+ class-conditional input-sensitivity PSD statistics
+ model-agnostic server subspace recovery
+ detached orthogonal projected counterfactual training
```

FedCIS replaces AsymHFL as the candidate collaboration module. It does not use
public images, public logits, prototypes, model-parameter aggregation, or
corruption taxonomy metadata. All heterogeneous models share only the input
space and class space.

Important status:

```text
framework definition: corrected candidate exists
implementation: standalone Audit A/B complete
offline audit: NO-GO
12/40-round training: blocked
positive result: none
```

Corrections frozen on 2026-08-03:

```text
1. use PSD view-mean/view-difference second moments instead of E[s1 s2^T];
2. use outer-product client dispersion instead of an ambiguous matrix square;
3. generate counterfactuals along margin descent, not margin ascent;
4. detach the projected perturbation and remove full second-order L_sens from v0;
5. use fixed-shape support masks rather than exact class counts by default;
6. do not claim zero privacy, arbitrary unseen-corruption guarantees, or
   missing-class semantic completion.
```

The central identifiability assumption is unverified: cross-client matched-class
input sensitivity may reflect shared semantics, but it may also reflect common
texture, architecture, or preprocessing bias. Before any runner is implemented,
the RTX 3050 offline audit must compare:

```text
true class-matched global subspace
class-shuffled global subspace
equal-rank random orthogonal subspace
matched base-only update
```

Only true FedCIS may proceed if it separates from both controls, improves
matched audit loss, keeps audit Avg/Worst/WCCA/seen/unseen nonnegative, and does
not increase audit CFG. Operator metadata is evaluation-only and cannot enter
subspace construction or selection.

Read the authoritative specification first:

```text
docs/archive/methods/FEDCIS_FRAMEWORK_AND_OFFLINE_AUDIT_ZH.md
```

Previous FedCFSA K=8 feasibility work is archived as historical evidence. The
user has rejected changing the formal client count to K=8; it is not the next
task.

## Historical: FedCFSA Multi-Client Redundancy Audit - 2026-07-27

The four-client formal checkpoint audit remains NO-GO for strong
cross-falsification (`0/7` stable routes have two independent stable
validators). A separate CPU-only feasibility sweep tested whether this is a
four-client artifact.

Frozen sweep:

```text
standard full-CIFAR-10 Dirichlet partition (50,000 labels, variable client size)
alpha=0.5, gamma=0.9, seeds=0/1/2
K=4/8/10/20
source support: fit>=16 and audit>=5
strong coverage: >=3 source clients from >=3 distinct dominant environments
```

Mean strong coverage over auditable receiver-class targets, with worst seed:

```text
K=4:  56.29%, worst seed 43.75% -> NO-GO
K=8:  94.61%, worst seed 88.52% -> GO
K=10: 96.73%, worst seed 93.06% -> GO
K=20: 100.0%, worst seed 100.0% -> GO
```

Decision:

```text
FedCFSA source redundancy is conditionally feasible from K=8 onward.
This is not evidence that frontier routing or synthetic anchors work.
```

The sweep covers only receiver classes with enough fit/audit support. It does
not solve fully missing classes. The proposed next stage at that time was a
K=8 checkpoint-level reliability audit:

```text
1. build one K=8 CLE-HFL v2 seed-0 dataset with repeated heterogeneous models;
2. train only the strict local robust base for a short checkpoint probe;
3. recompute stable robust-frontier sources across augmentation seeds;
4. require >=3 reliable sources for enough auditable receiver-class targets;
5. only then implement the matched synthetic-anchor one-step audit.
```

Implementation/report:

```text
fedprime/analysis/fedcfsa_coverage.py
scripts/audit_fedcfsa_source_redundancy.py
tests/test_fedcfsa_coverage.py
deliverables/fedcfsa_source_redundancy_audit_20260727/AUDIT_REPORT_ZH.md
```

Verification: `18 passed`. No GPU training was run.

This K=8 proposal is paused and superseded by the 2026-08-03 FedCIS K=4
offline-audit candidate.

## Historical: FedCFSA Coverage Audit - 2026-07-27

The proposed semantic-payload candidate was:

```text
FedCFSA
= RAHFL local robust base
+ taxonomy-free robust-frontier gate
+ cross-falsified synthetic semantic anchors
+ receiver-side safe assimilation
```

Before implementing image condensation, the mandatory source-redundancy audit
was run on the seven routes stable across all three frontier augmentation
seeds.

Result:

```text
stable routes: 7, covering 6 receiver-class targets
targets with two stable sources: 1/6
routes with another stable source after excluding the generator: 2/7
routes with at least two ordinary support-qualified validators: 1/7
routes with two independent stable validators: 0/7
```

The written FedCFSA formula averages validators in
`S[receiver,class] \ {generator}`. For five of seven routes this set is empty;
for the remaining two routes it contains only one validator. The simultaneous
claims "at most two sources" and "multiple independent cross-falsifiers" are
therefore inconsistent under the current four-client severe-label-skew
protocol.

Decision for the existing four-client formal checkpoints:

```text
FedCFSA as currently written on four-client CLE-HFL v2: NO-GO
```

This rejects the four-client cross-falsification coverage assumption, not all
synthetic semantic payloads. The later multi-client CPU sweep shows that source
support is feasible from K=8, but checkpoint-level reliable-source coverage is
still unverified. Do not implement anchor condensation or run a 5/12/40-round
FedCFSA experiment yet.

```text
1. increase the formal protocol to enough clients for >=3 sources per class;
2. accept one-validator peer filtering and weaken the scientific claim;
3. introduce an external semantic verifier and explicitly change assumptions.
```

Audit:

```text
deliverables/fedcfsa_coverage_audit_20260727/FEDCFSA_COVERAGE_AUDIT_ZH.md
```

External discussion brief updated after the robust-frontier one-step audit:

```text
docs/research/status/CURRENT_RESEARCH_STATUS_RAHFL_AND_COMMUNICATION_REVIEW_20260727_ZH.md
```

It summarizes RAHFL, CLE-HFL/CLE-HFL v2, all major positive and negative
results, the public-logit bottleneck, and the unresolved semantic-payload
question. Use this file for the next external AI discussion.

## Newest Offline Decision: Robust Frontier Audit - 2026-07-26

Before implementing another communication runner, a taxonomy-free robust
frontier audit was completed on the stored CLE-HFL v2 RAHFL checkpoints.

Definition:

```text
per-sample logits are z-score normalized across classes
q[k,c,j] = lower-0.2 quantile of the worst-view normalized margin z_c-z_j
primary views = clean + two AugMix draws
diagnostic views also include the weak DCL view
```

No operator ID, name, family, split, or severity enters `q`. Operator metadata
is used only after inference to evaluate whether the score predicts final
client/class seen and unseen operator performance.

Formal seed-0 result:

```text
local frontier vs seen-worst Spearman        = 0.434
local frontier vs unseen-worst Spearman      = 0.559
source advantage vs seen advantage Spearman  = 0.319
source advantage vs unseen advantage         = 0.548

unfiltered positive-source precision:
  seen   = 52.94%
  unseen = 52.94%
```

All four correlation gates pass, but direct teacher routing fails the frozen
precision gate. The original global-median/full-coverage FedRIFT concept is
therefore **NO-GO** and must not be run for 40 rounds.

Exploratory abstention signal, repeated over augmentation seeds 0/1/2:

```text
top-quartile all-view routes: 9 routes per seed
seen precision:   77.78% / 88.89% / 88.89%
unseen precision: 88.89% / 100.00% / 88.89%
pairwise route-set Jaccard: 0.80
routes common to all three seeds: 7
```

This supports only one narrower next candidate:

```text
multi-view robust frontier + high-confidence abstention
```

Do not implement a full runner yet. First perform a fit/audit one-step
classifier-head update audit on the stable routes, with no final-test routing.

Implementation and report:

```text
fedprime/analysis/robust_frontier.py
scripts/audit_robust_frontier.py
tests/test_robust_frontier_audit.py
deliverables/robust_frontier_audit_20260726/ROBUST_FRONTIER_AUDIT_ZH.md
outputs/robust_frontier_audit_20260726/
```

Verification: `7 passed`; RTX 3050 formal audit completed three times.

### One-step transfer audit - 2026-07-27

The seven routes stable across all augmentation seeds were tested with a
matched causal control:

```text
control   = one classifier-head CE step on the identical fit batch
candidate = identical CE step + lower-tail robust-frontier margin loss
```

Each route restores the original checkpoint afterward. Candidate-minus-control:

```text
mean target-class audit accuracy = 0.0000
target-class audit loss          = slightly better on 7/7 routes
mean seen accuracy               = 0.0000
mean unseen accuracy             = +0.0357
mean all-audit accuracy          = -0.0095
```

The loss has a finite gradient, but produces no meaningful accuracy transfer.
Therefore a `C x C` robust frontier is a useful reliability diagnostic but an
insufficient knowledge payload. Direct frontier regularization remains NO-GO.

If retained, the frontier may only serve as a high-confidence abstention gate
around a separately justified semantic knowledge channel. Do not start a
12/40-round run until that payload passes its own matched one-step audit.

Implementation/output:

```text
scripts/audit_robust_frontier_one_step.py
outputs/robust_frontier_one_step_audit_20260727/
```

Verification is now `8 passed`.

## CLE-HFL v2 Formal Probe Result - 2026-07-24

The three 12-round runs in `outputs/cle_hfl_v2_probe_outputs.tar.gz` completed:

```text
metric order: Avg / Worst / WCCA / CFG; lower CFG is better

RAHFL final          = 33.8267 / 27.0400 / 0.250 / 30.050
Strict control final = 30.7550 / 24.9800 / 0.250 / 30.225
FedFalsify v0.3      = 31.0733 / 24.5733 / 0.500 / 31.825

FedFalsify - control final:
  +0.3183 / -0.4067 / +0.250 / +1.600

FedFalsify - control last-five:
  +0.1180 / -0.4373 / +0.850 / +2.185
```

FedFalsify communication is active but fails the frozen gate: it gives a small
Avg/WCCA gain while harming Worst and CFG. Do not run it for 40 rounds or tune
only `kappa`/`lambda_cmt`.

Important fairness limitation: the RAHFL run used all 10,000 local samples per
client and routes with final-test accuracy. Strict control/FedFalsify train on
about 8,500 fit samples and reserve about 1,500 audit samples; they never route
with final test labels. Therefore only `FedFalsify vs strict control` is a
strict causal comparison. RAHFL's numerical lead is diagnostic, not yet a fair
paper result.

Next required run: one 12-round strict RAHFL-val using the same persisted
fit/audit split, fit-only local training, audit-only AsymHFL routing, and final
test only for evaluation. Existing control/FedFalsify runs do not need reruns.

Analysis:

```text
deliverables/cle_hfl_v2_probe_analysis_20260724/
```

Research decision after review: do not spend the next compute budget on
strict RAHFL-val yet. The fair causal comparison already shows that FedFalsify
v0.3 gives only a tiny Avg gain while harming Worst and CFG. Pause training and
seek an external theoretical review before another communication design.

External discussion brief:

```text
docs/archive/methods/FEDFALSIFY_LATEST_EXTERNAL_AI_DISCUSSION_BRIEF_ZH.md
```

## Current Protocol Mainline: CLE-HFL v2 - 2026-07-24

The current method remains FedFalsify v0.3, but its next evaluation protocol is
now operator-level CLE-HFL v2 rather than the legacy four-group taxonomy.

```text
CLE-HFL v2:
  label skew: Dirichlet alpha=0.5
  shortcut: client/class -> concrete seen corruption operator, gamma=0.9
  seen operators: 11
  unseen operators: 4, absent from every client training set
  evaluation: clean + seen + unseen + all 15 operators
```

Corruption family labels are audit metadata only. FedFalsify receives images,
labels, predictions, and classifier-head gradients; it never receives operator
IDs, names, families, seen/unseen flags, or severities.

Implementation:

```text
fedprime/data/corruptions.py
scripts/prepare_cle_v2_data.py
scripts/audit_cle_v2_data.py
fedprime/data/loaders.py
fedprime/engine/operator_metrics.py
scripts/openi_cle_v2_entry.py
configs/openi_v100_rahfl_cle_v2_probe.yaml
configs/openi_v100_fedfalsify_v03_cle_v2_control_probe.yaml
configs/openi_v100_fedfalsify_v03_cle_v2_probe.yaml
```

Formal prepared data:

```text
local_runs/cle_hfl_v2_prepared/
  cle_hfl_v2_prepared_alpha05_gamma09_seed0_split0.tar.gz
size: 363,169,221 bytes, about 346 MiB
```

Protocol audit passed:

```text
samples/client: 10,000
expected dominant-operator rate: 0.90909
realized rate: 0.91015
unseen samples in client training: 0
test seen/unseen/all/clean: 11,000/4,000/15,000/1,000
```

Verification:

```text
focused tests: 22 passed
RTX 3050 end-to-end v2 smoke: passed
round 0 warmup
round 1: 66 candidates, 12 inferior rejected, 4 routes, CMT=1.0123
RAHFL v2 one-round smoke: passed, finite AsymHFL/local losses
```

Read:

```text
docs/archive/methods/CLE_HFL_V2_FEDFALSIFY_FRAMEWORK_ZH.md
docs/experiments/guides/CLE_HFL_V2_OPENI_RUN_GUIDE_ZH.md
```

The formal three-way 12-round probe is complete. FedFalsify failed the
Worst/CFG gate. The next run is strict RAHFL-val only; do not run 40 rounds.

## Latest Executable Candidate: FedFalsify v0.3 - 2026-07-23

FedFalsify v0.2 produced a real but weak communication signal and failed the
strict gate on CFG. Route attribution found that 54.17% of selected teachers
had nonpositive paired correctness advantage. v0.3 makes exactly one change:

```text
paired SE  = sqrt(paired_variance / n)
paired UCB = paired_advantage + kappa * paired SE
eligible   = paired UCB >= 0
```

Head-TAU Top-1 selection then runs only among eligible sources. This is a
non-inferiority veto: uncertain sources may survive, but sources statistically
supported as worse than the receiver are removed. CMT, fit/audit split, warmup,
models, optimizer, data, and all local learning remain frozen.

Core files:

```text
fedprime/methods/fedfalsify/evidence.py
fedprime/methods/fedfalsify/router.py
fedprime/methods/fedfalsify_experiment.py
configs/debug_fedfalsify_v03_cle.yaml
configs/openi_v100_fedfalsify_v03_probe.yaml
scripts/openi_fedfalsify_v03_entry.py
```

Verification:

```text
focused tests: 15 passed
RTX 3050 smoke: 2 rounds completed in 22.2 seconds
round 0: warmup, routes=0, cmt_loss=0
round 1: candidates=99, eligible=84, statistically inferior rejected=15
         routes=11, mean TAU=0.9473, finite cmt_loss=1.2705
```

The full repository test command timed out in the pre-existing Matplotlib
`test_analyze_priors.py` plotting path; no FedFalsify test failed.

Next action: run only `scripts/openi_fedfalsify_v03_entry.py`. It packages
`fedfalsify_v03_probe_outputs.tar.gz` and reuses the stored strict control
offline. Do not run 40 rounds before the stricter frozen gate passes.

### FedFalsify v0.3 OpenI result - 2026-07-24

The 12-round candidate-only probe completed:

```text
strict control final  = 37.7788/31.8025/WCCA 9.550/CFG 9.4625
FedFalsify v0.3 final = 39.0631/32.1475/WCCA 12.750/CFG 9.1125
final delta           = +1.2844/+0.3450/+3.200/-0.3500

strict control last5  = 35.2646/29.5320/WCCA 6.480/CFG 10.7440
FedFalsify v0.3 last5 = 36.6493/30.6515/WCCA 7.955/CFG 11.3825
last-five delta       = +1.3846/+1.1195/+1.475/+0.6385
```

Lower CFG is better. The frozen four-metric gate passed Avg, Worst, and WCCA,
but failed CFG. Communication is nevertheless real: v0.3 beats strict control
on Avg and Worst in all 9 post-warmup rounds. Relative to v0.2, final
Avg/Worst/WCCA/CFG all improve.

Route attribution:

```text
mean active routes:                       32.00 -> 21.44
selected nonpositive-advantage teachers: 54.17% -> 27.98%
common-route source churn:                21.03% -> 16.89%
active-route-set adjacent Jaccard:        96.90% -> 75.97%
```

The non-inferiority veto successfully improves teacher quality. The remaining
problem is unstable class-corruption disparity, plausibly associated with
routes entering and leaving around the zero-UCB boundary. This association is
not yet a causal result because the run did not persist full per-round
client/class/corruption matrices.

Decision: preserve v0.3 as a positive partial result, but do not run 40 rounds
or tune `kappa/lambda_cmt` blindly. Add evaluation-only matrix persistence and
attribute CFG spikes first.

Full report:

```text
deliverables/fedfalsify_v03_probe_analysis_20260724/FEDFALSIFY_V03_PROBE_ANALYSIS_ZH.md
```

## Previous Executable Candidate: FedFalsify v0.2 - 2026-07-23

The offline audit has now been converted into an executable, leakage-free
12-round A/B probe.

Current method:

```text
AugMix/JSD/DCL local base
+ fixed class-stratified D_fit/D_audit split
+ frozen peer model snapshots
+ receiver-side class-conditional head-TAU Top-1 source selection
+ conservative margin transfer (CMT)
```

Important restrictions:

```text
final test labels are evaluation-only;
FRA is not a hard gate and has default weight 0;
source selection uses only receiver-private D_fit and D_audit;
peer snapshots are frozen for each local round;
no public data or public logits are used.
```

Formal A/B:

```text
scripts/openi_fedfalsify_entry.py
configs/openi_v100_fedfalsify_fit_control_probe.yaml
configs/openi_v100_fedfalsify_probe.yaml
docs/experiments/archive/FEDFALSIFY_OPENI_RUN_GUIDE_ZH.md
```

The control and candidate both use the persisted split:

```text
outputs/partitions/fedfalsify_v1_cle_alpha05_gamma09_seed0.npz
```

Local 3050 end-to-end debug passed twice. Round 0 correctly disabled
communication; round 1 selected 11/40 class routes with mean head-TAU 0.9473
and a finite CMT loss. Unit tests: 14 passed. Debug accuracy is not evidence.

Next action: run the strict OpenI A/B probe and judge last-five
Avg/Worst/WCCA/CFG. Do not run 40 rounds before the frozen gate passes.

### Strict A/B result

The 12-round OpenI probe is complete:

```text
strict control final  = 37.7788/31.8025/WCCA 9.550/CFG 9.4625
FedFalsify final      = 38.5175/31.5400/WCCA 9.525/CFG 9.7175

strict control last5  = 35.2646/29.5320/WCCA 6.480/CFG 10.744
FedFalsify last5      = 35.8096/29.8130/WCCA 6.790/CFG 11.190
```

Metric order is Avg/Worst/WCCA/CFG and lower CFG is better. The last-five
delta is `+0.5450/+0.2810/+0.310/+0.446`, so the frozen gate failed only on
CFG. Avg is higher than strict control in all 9 communication rounds, which is
a real but small positive communication signal.

Route diagnosis shows why the gain is limited: 54.17% of selected sources have
nonpositive paired accuracy advantage and 61.11% have zero FRA strength, even
though mean head-TAU is 0.8725. TAU verifies gradient compatibility but not
teacher expertise.

Decision: do not run v0.2 for 40 rounds. The next candidate should add a paired
correctness non-inferiority veto (`Delta + kappa*SE >= 0`) before TAU Top-1.
This rejects statistically worse sources without restoring the sparse v0.1
hard positive-FRA gate.

Full report:

```text
deliverables/fedfalsify_probe_analysis_20260723/FEDFALSIFY_PROBE_ANALYSIS_ZH.md
```

## Latest Result: FedFalsify v0.1 Offline Audit - 2026-07-23

FedFalsify was not sent directly into another paid federated run. Its required
offline Go/No-Go audits were implemented and executed first on the stored RAHFL
CLE-HFL checkpoints for `alpha=0.5, seed=0, gamma={0.0,0.6,0.9}`.

Core implementation:

```text
fedprime/methods/fedfalsify/evidence.py
fedprime/methods/fedfalsify/transfer.py
fedprime/methods/fedfalsify/audit_runtime.py
scripts/audit_fedfalsify_foreign_tensor.py
scripts/audit_fedfalsify_gate_coverage.py
scripts/audit_fedfalsify_one_step.py
scripts/summarize_fedfalsify_audit.py
```

Key results:

```text
gamma                              0.0       0.6       0.9
foreign survival gap             0.0016    0.0692    0.1559
projected sample activation       11.56%     8.23%     4.50%
FRA+TAU triplet activation        23.81%    19.05%    14.29%
direct KD increment over CE      -0.0421   -0.0489   -0.0724
CMT increment over CE            +0.0019   +0.0019   +0.0017
TAU precision / recall            .903/1    .904/1    .857/1
```

Interpretation:

```text
1. The failure mode is measurable: as gamma grows, a model's knowledge
   survives much better in its own corruption-label environment than abroad.
2. Direct peer KD is unsafe and becomes more harmful under stronger
   entanglement.
3. CMT is directionally safer but its incremental one-step effect is small.
4. TAU predicts positive CMT increments reasonably well.
5. FRA is the bottleneck: it adds little precision while sharply reducing
   coverage, especially at gamma=0.9.
```

Frozen decision:

```text
Do not run FedFalsify v0.1 for 40 rounds.
Do not count CE improvement as communication improvement.
Do not use independent test_same labels in a formal training router.
```

Promising v0.2 revision:

```text
Make TAU the safety gate. For each receiver/class, choose the top-1
TAU-positive source; use FRA only as a ranking prior or tie-breaker rather than
as a hard prerequisite.

gamma                              0.0       0.6       0.9
TAU top-1 coverage                100%      100%      100%
positive increment precision      91.4%     94.3%     85.7%
mean CMT-over-CE increment        .00354    .00367    .00320

At gamma=0.9 this covers all 35 auditable receiver-class groups and gives
positive CMT-over-CE increment on 30/35. It is close to the offline oracle
ranking, but this remains a one-step result rather than a federated-run proof.
```

Artifacts:

```text
docs/experiments/archive/FEDFALSIFY_AUDIT_GUIDE_ZH.md
deliverables/fedfalsify_offline_audit/FEDFALSIFY_OFFLINE_AUDIT_ZH.md
deliverables/fedfalsify_offline_audit/fedfalsify_offline_audit_summary.csv
outputs/fedfalsify_audit/
outputs/fedfalsify_audit/source_ranking/
```

## Latest Result: EBST-v2 Fails Clean Attribution - 2026-07-22

Archive:

```text
outputs/fedease_pew_calibrated_local_probe_outputs.tar.gz
```

The matching calibrated PEW local-only control completed on CLE-HFL
`alpha=0.5, gamma=0.9, seed=0`. It uses the same PEW checkpoint (epoch 3),
automatic threshold (`0.0`), inferred environments (`63.59%` private group
accuracy), data, models, seed, optimizer, and 12-round budget as the completed
EBST-v2 combination. Only communication, EBST-v2, and SCP are disabled.

```text
                                   final                         last-five mean
calibrated PEW local   42.8469/36.2300/19.775/6.5725  40.4278/36.2890/17.965/6.427
calibrated PEW+EBSTv2  42.6331/35.2975/20.675/7.2900  40.4526/35.9870/17.400/6.666
EBST-v2 minus local    -0.2138/-0.9325/+0.900/+0.7175 +0.0249/-0.3020/-0.565/+0.239
```

Metric order is `Avg/Worst/WCCA/CFG`; lower CFG is better. EBST-v2 fails the
pre-registered gate (`last-five Avg >= +0.5`, no Worst/WCCA regression, no CFG
increase). Its final per-client deltas are `[-0.9325, -0.6050, +1.0825,
-0.4000]`: three clients regress and only client 2 improves. All four corruption
groups regress slightly. Large class-level transfers in both directions cancel
in the mean, confirming residual negative transfer rather than useful global
knowledge transfer.

Decision:

```text
The gain previously attributed to the complete combination comes primarily
from calibrated PEW + BER+CDep local learning, not EBST-v2 communication.
Archive EBST-v2 as a negative communication result. Do not run 40 rounds and do
not spend another run on lambda-only tuning. Redesign communication before the
next paid experiment, preferably without hard environment taxonomy.
```

Analysis artifacts:

```text
deliverables/fedease_ebst_attribution_20260722/summary.csv
deliverables/fedease_ebst_attribution_20260722/client_deltas.csv
deliverables/fedease_ebst_attribution_20260722/group_deltas.csv
deliverables/fedease_ebst_attribution_20260722/class_deltas.csv
deliverables/fedease_ebst_attribution_20260722/analysis.json
```

## Latest Result: Calibrated PEW + EBST-v2 Is Positive as a Whole but Confounded - 2026-07-21

Archive:

```text
outputs/fedease_pew_ebst_v2_probe_outputs.tar.gz
```

Final and last-five results on CLE-HFL `alpha=0.5, gamma=0.9, seed=0`:

```text
                                   final                         last-five mean
old PEW local         40.3694/35.4225/13.925/6.370   38.9004/35.0515/14.940/7.5455
calibrated PEW+EBSTv2 42.6331/35.2975/20.675/7.290   40.4526/35.9870/17.400/6.6660
delta                 +2.2638/-0.1250/+6.750/+0.920  +1.5523/+0.9355/+2.460/-0.8795
```

Metric order is `Avg/Worst/WCCA/CFG`; lower CFG is better. The last-five comparison
improves all four metrics. Final Worst is nearly tied, while final CFG is worse
because the final-round metric is volatile.

The PEW correction worked strongly:

```text
selected checkpoint: epoch 3, public validation environment accuracy 57.4%
automatic unknown threshold: 0.0
private exact environment accuracy: 38.83% -> 63.59%
private unknown rate: about 49.8%-57.7% -> 3.7%-10.6%
```

All corruption groups improve at the final round relative to old PEW local:

```text
noise +1.32, blur +2.02, weather +2.18, digital +3.54,
worst corruption group +1.32 points.
```

EBST-v2 was active in rounds 3-11:

```text
mean EBST loss=0.2528, valid environment fraction=0.5725,
valid pair fraction=0.6650, eligible sources=2.1711, mean gate=0.1877,
SCP conflict fraction=0.4697, retained communication-gradient norm=0.5805.
```

Interpretation boundary:

```text
The deployable combination is a positive 12-round candidate trajectory and is
well above same-round RAHFL. It does not yet prove that EBST-v2 adds value,
because PEW calibration and communication changed together. Rounds 0-2 already
gain +1.07 to +2.16 Avg before communication starts. Large class-specific gains
and regressions also remain. Run a matching calibrated PEW local-only control;
do not run 40 rounds yet.
```

Analysis artifacts:

```text
deliverables/fedease_pew_ebst_v2_analysis_20260721/summary.csv
deliverables/fedease_pew_ebst_v2_analysis_20260721/class_deltas.csv
deliverables/fedease_pew_ebst_v2_analysis_20260721/analysis.json
```

The required calibrated local-only attribution control is ready:

```text
entry:  scripts/openi_fedease_entry.py --mode=pew_calibrated_local_probe
config: configs/openi_v100_fedease_pew_calibrated_local_probe.yaml
guide:  docs/experiments/archive/FEDEASE_CALIBRATED_PEW_LOCAL_OPENI_RUN_ZH.md
```

It preserves calibrated PEW, BER, CDep, AugMix/JSD/DCL, models, data, seed,
optimizer, and the 12-round budget, while disabling communication, EBST-v2, and
SCP. Targeted tests and the formal environment/path check pass.

## Immediate Run: Calibrated Learned PEW + EBST-v2 - 2026-07-21

Implemented the final 12-round hard-environment-taxonomy combination probe:

```text
entry:  scripts/openi_fedease_entry.py --mode=pew_ebst_v2_probe
config: configs/openi_v100_fedease_pew_ebst_v2_probe.yaml
guide:  docs/experiments/archive/FEDEASE_PEW_EBST_V2_OPENI_RUN_ZH.md
```

The correction restores the PEW epoch with the highest synthetic public
validation environment accuracy and selects the unknown-rejection threshold on
that same public validation split. It never uses CIFAR-10 test labels for model
selection. The candidate combines learned PEW + BER+CDep with the previously
safety-corrected EBST-v2 and class-wise SCP; all previous configs remain intact.

Verification:

```text
26 targeted tests passed;
formal config dependency/path check passed;
two-round four-model real-data smoke executed calibration, LOO aggregation,
EBST loss, class-wise SCP, and extended evaluation without NaN.
```

The stored PEW local comparison point is:

```text
Avg=40.3694, Worst=35.4225, WCCA=13.925, CFG=6.370
```

This run is the final Go/No-Go test for the current hard six-way PEW route. If
communication regresses against the stored PEW local result, do not run 40
rounds or tune only lambda. Redesign the deployable method using continuous
environment embeddings, soft anchors, Soft-BER, and Soft-EBST. The four CIFAR-C
groups may remain an evaluation taxonomy, but should no longer be a hard method
assumption.

## Latest Result: Learned PEW Preserves Most Oracle Local Gain - Near Pass - 2026-07-20

Archive:

```text
outputs/fedease_pew_probe_outputs.tar.gz
```

Matching 12-round CLE-HFL `alpha=0.5, gamma=0.9, seed=0` comparison:

```text
Oracle local control:  Avg=37.5813, Worst=30.1100, WCCA=13.700, CFG=10.855
Oracle BER+CDep:       Avg=41.6206, Worst=35.5175, WCCA=14.000, CFG= 6.155
Learned PEW BER+CDep: Avg=40.3694, Worst=35.4225, WCCA=13.925, CFG= 6.370
```

Learned PEW relative to control:

```text
Avg=+2.7881, Worst=+5.3125, WCCA=+0.225, CFG=-4.485
```

Learned PEW relative to Oracle BER+CDep:

```text
Avg=-1.2513, Worst=-0.0950, WCCA=-0.075, CFG=+0.215
```

The predeclared final gate was `40.5/34.0/WCCA 13/CFG 7`. PEW passed Worst,
WCCA, and CFG, and missed Avg by only `0.1306`. Best Avg was `41.4194` at
round 10. Last-five means remain clearly better than control:

```text
Avg=+2.4306, Worst=+4.1010, WCCA=+6.030, CFG=-3.4025
```

PEW diagnostic:

```text
public validation environment accuracy: best 57.4% at epoch 3, final 52.5%
private exact group accuracy: 38.83%
per-client unknown rate: approximately 49.8% to 57.7%
```

Interpretation:

```text
The learned environment path is viable and preserves almost all Oracle tail/CFG
benefit despite imperfect exact group prediction. It is a near pass, not a strict
pass, because final Avg missed the frozen gate and PEW saved the last epoch rather
than the best validated checkpoint. Do not run 40-round full mode yet.
```

Before the next paid run, correct PEW model selection using public validation and
calibrate the unknown threshold on that validation split. Then run one matching
12-round deployable combination probe with PEW + BER+CDep + EBST-v2. Do not rerun
RAHFL or the Oracle local baselines for that probe.

## Latest Result: EBST-v2 Is Safe But Average-Neutral - 2026-07-20

Archive:

```text
outputs/fedease_ebst_v2_probe_outputs.tar.gz
```

Matching 12-round comparison on CLE-HFL `alpha=0.5, gamma=0.9, seed=0`:

```text
Oracle BER+CDep local:          Avg=41.6206, Worst=35.5175, WCCA=14.000, CFG=6.155
Oracle BER+CDep+EBST-v2+SCP:   Avg=41.9469, Worst=36.2275, WCCA=14.700, CFG=5.190
Final-round delta:              Avg=+0.3263, Worst=+0.7100, WCCA=+0.700, CFG=-0.965
Last-five-round mean delta:     Avg=-0.1648, Worst=+0.4400, WCCA=+0.765, CFG=0.000
```

The formal gate passed Worst, WCCA, and CFG, but missed the `Avg > 42.1` gate.
Because the last-five average accuracy is slightly below the local baseline, the
final-round average gain is not yet stable evidence of positive communication.

EBST-v2 was active and selective:

```text
valid environment fraction=0.5463
valid class-pair fraction=0.6775
mean eligible sources=2.1635
mean gate=0.2101
mean SCP conflict rate=0.4759
mean SCP projection norm ratio=0.5763
```

Unlike legacy EBST, v2 prevents client-level collapse. Final client deltas are
`+0.7100/-0.1375/+0.5125/+0.2200`; the old client-2 failure is removed. However,
individual class regressions remain, including client 3 class 3 (`-11.225`) and
client 2 class 7 (`-7.675`) points averaged across corruption groups.

Decision:

```text
EBST-v2 validates the safety correction and improves fairness-oriented metrics.
It does not yet validate a positive average-accuracy communication contribution.
Do not run full/PEW mode or claim EBST-v2 as the final communication method.
The next redesign must add recipient-class acceptance/trust-region protection,
not merely tune the communication lambda.
```

## Latest Implementation: EBST-v2 Corrective Communication Probe - 2026-07-20

The failed legacy EBST path remains available as `communication: ebst`. A new
switchable corrective path is implemented as `communication: ebst_v2`:

```text
pair-source eligibility requiring support for both c and competing class j
+ recipient-specific leave-one-out relation teachers
+ cross-environment and cross-client agreement gate
+ class-wise classifier SCP with communication-gradient norm cap
+ three-round relation warmup
```

Formal OpenI files:

```text
config: configs/openi_v100_fedease_ebst_v2_probe.yaml
entry:  scripts/openi_fedease_entry.py --mode=ebst_v2_probe
guide:  docs/experiments/archive/FEDEASE_EBST_V2_OPENI_RUN_ZH.md
```

Verification:

```text
24 targeted FedEASE tests passed
OpenI entry dry-run passed
two-round four-model real-data smoke passed without NaN
smoke EBST-v2 valid_pairs=0.770, mean_gate=0.698
smoke class-wise SCP conflict fraction=0.538, projection norm ratio=0.611
```

The debug smoke uses deliberately relaxed source thresholds only to execute the
path. The formal config requires at least two eligible leave-one-out sources and
uses the same 12-round budget as the stored Oracle BER+CDep local baseline.
Clients expose only thresholded support masks for source qualification; exact
class counts are removed before server aggregation.
The formal result is recorded above: safety improved, but average-accuracy gain
did not pass the predeclared gate.

## Latest Result: FedEASE Oracle EBST Probe Failed Its Communication Gate - 2026-07-20

Archive:

```text
outputs/fedease_ebst_probe_outputs.tar.gz
```

Matching 12-round comparison on CLE-HFL `alpha=0.5, gamma=0.9, seed=0`:

```text
Oracle BER+CDep local:       Avg=41.6206, Worst=35.5175, WCCA=14.000, CFG=6.155
Oracle BER+CDep+EBST+SCP:   Avg=38.7038, Worst=34.7225, WCCA=15.325, CFG=6.415
Communication delta:        Avg=-2.9169, Worst=-0.7950, WCCA=+1.325, CFG=+0.260
```

The EBST path was active rather than silently skipped:

```text
mean EBST loss=0.1392
mean gate=0.3905
valid environment fraction=1.0
mean SCP conflict rate=0.4531
mean SCP projection norm ratio=0.9808
```

The main failure is concentrated in client 2, whose final accuracy falls from
`45.2175` to `34.7225`. Its class 2/6/7/9 accuracies fall by roughly
`29.60/35.17/22.40/32.20` points. Final worst corruption-group accuracy also
falls from `40.4150` to `38.2700`.

Interpretation:

```text
Oracle BER+CDep remains a positive local mechanism.
Current EBST communication is a negative result and must not enter full mode.
The environment-stability gate does not measure recipient-specific or
cross-client relation conflict, and head-level SCP is too coarse to prevent
class-specific negative transfer.
Do not run PEW+EBST full training or tune lambda blindly.
```

Immediate research decision: freeze the current EBST implementation as a failed
probe. Diagnose/redefine source eligibility and recipient-level communication
safety before spending another full run; PEW can be tested independently only
if its deployability question is worth the remaining compute budget.

## Latest Result: FedEASE Oracle BER+CDep Probe Passed - 2026-07-20

Archive:

```text
outputs/fedease_oracle_probe_outputs.tar.gz
```

Matching 12-round local-only A/B on CLE-HFL `alpha=0.5, gamma=0.9, seed=0`:

```text
Oracle control final:  Avg=37.5813, Worst=30.1100, WCCA=13.70, CFG=10.855
Oracle BER+CDep final: Avg=41.6206, Worst=35.5175, WCCA=14.00, CFG= 6.155
Delta:                 Avg=+4.0394, Worst=+5.4075, WCCA=+0.30, CFG=-4.70
```

The result is broad rather than client-specific:

```text
all four clients improve final accuracy
worst corruption-group accuracy: 34.1575 -> 40.4150
worst client-corruption accuracy: 25.30 -> 34.78
last-5 WCCA mean: 8.91 -> 16.22
last-5 CFG mean: 10.948 -> 5.4805
```

Same-round RAHFL+AsymHFL at round 11 was `37.4575/30.695/8.15/9.725`, so the
Oracle local candidate is better at the same 12-round budget. This is not yet a
claim against the 40-round RAHFL final `46.72/38.1575/19.325/10.91`.

Interpretation boundary:

```text
BER+CDep jointly pass the mechanism Go/No-Go gate.
The experiment does not identify whether BER or CDep contributes more.
CDep is active on about three valid classes per minibatch, but its covariance
surrogate does not clearly decline, so attribution still needs later ablation.
```

The next run recorded at that point was:

```text
scripts/openi_fedease_entry.py --mode=ebst_probe
```

That EBST probe is now complete and failed the communication gate as recorded above.
Do not run `--mode=full`.

## Latest Override: FedEASE v2.1 Complete Candidate - 2026-07-19

Current research candidate:

```text
CLE-HFL + FedEASE v2.1
```

Read first:

```text
docs/archive/methods/FEDEASE_V2_1_FRAMEWORK_AND_IMPLEMENTATION_ZH.md
```

The full planned method is:

```text
RAHFL robust local base
+ BER/CDep class-conditional environment invariance
+ EBST environment-balanced structural communication
+ optional SCP negative-transfer protection
```

The complete switchable candidate is now implemented:

```text
Oracle or learned PEW environments
+ BER replacing clean CE
+ fixed-random-projection CDep
+ AugMix/JSD/DCL preserved
+ EBST environment-balanced relation communication
+ stability gate and classifier-head SCP
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
scripts/openi_fedease_entry.py
```

Formal configs:

```text
configs/openi_v100_fedease_oracle_control_probe.yaml
configs/openi_v100_fedease_oracle_ber_cdep_probe.yaml
configs/openi_v100_fedease_pew_probe.yaml
configs/openi_v100_fedease_ebst_probe.yaml
configs/openi_v100_fedease_full.yaml
```

Verification:

```text
compile check passed
19 targeted FedEASE tests passed
OpenI entry dry-run passed
two-round four-model real-data EBST smoke completed with finite losses/gradients
all five evaluation splits executed
```

The smoke result is interface validation only, not a research result. PEW/EBST/gate/SCP are
implemented but have no formal effectiveness result yet. The whole legacy test directory was
not completed because an unrelated Matplotlib/NumPy native crash occurs in
`tests/test_analyze_priors.py`; targeted FedEASE tests are green.

Prepared OpenI package and guide:

```text
local_runs/cle_hfl_prepared/fedease_cle_prepared_alpha05_gamma09_seed0.tar.gz
size: about 623.29 MiB
docs/experiments/archive/FEDEASE_OPENI_RUN_GUIDE_ZH.md
entry: scripts/openi_fedease_entry.py
first mode: --mode=oracle_probe
```

Immediate decision experiment:

```text
Oracle local control vs Oracle BER+CDep on the same gamma=0.9 data.
Only run PEW/EBST formal probes if WCCA improves, CFG falls, and Avg/Worst do not collapse.
Do not run the 40-round full mode first.
```

## Latest Override: FedCLEAR-PCCD - 2026-07-11

FedCLEAR v0.1 (`CCRE + IRD`) has completed a 40-round `gamma=0.9` run and is a
negative result:

```text
RAHFL:       avg=46.72, worst=38.16, WCCA=19.32, CFG=10.91
FedCLEAR v0.1 avg=45.41, worst=36.42, WCCA=17.80, CFG=11.42
```

CCRE reduced its surrogate risk, but private counterfactual views retained the
original corruption shortcut. IRD anchor disagreement remained high (last-10
mean about 0.891), so the cross-domain median teacher was not reliable.

The latest method is:

```text
FedCLEAR-PCCD
  fixed local base: AugMix + CE + JSD + DCL
  new communication: Paired Counterfactual Consensus Distillation
```

Read first:

```text
docs/archive/methods/FEDCLEAR_LATEST_THEORY_FRAMEWORK_ZH.md
```

PCCD implementation:

```text
fedprime/methods/pccd.py
fedprime/methods/fedclear_pccd.py
fedprime/methods/rahfl_asymhfl.py
scripts/prepare_cle_in_domain_public.py
scripts/import_cle_public_data.py
scripts/analyze_pccd_probe.py
scripts/openi_fedclear_pccd_entry.py
```

Disjoint public split verified locally:

```text
private=40000 unique CIFAR-10 train indices
public=5000 indices sampled only from the private complement
reserved=5000 remaining indices
package:
local_runs/cle_hfl_indomain_public/cle_hfl_indomain_public_alpha05_gamma09_seed0.tar.gz
```

Matching probe configs differ only in method identity and communication:

```text
configs/openi_v100_rahfl_cle_indomain_probe.yaml
configs/openi_v100_fedclear_pccd_probe.yaml
```

Verification completed:

```text
PCCD/FedCLEAR unit tests: 13 passed
config fairness test: passed
2-round four-model PCCD smoke: passed
legacy RAHFL CLE regression smoke: passed
OpenI entry dry-run and comparison analyzer: passed
```

Do not run PCCD for 40 rounds until its matching 12-round probe has:

```text
avg delta >= +1.5
worst delta >= +1.0
WCCA delta >= +4.0
CFG delta <= -1.5
```

## FedCLEAR Implementation Mainline - 2026-07-10

Current active research mainline:

```text
CLE-HFL problem + FedCLEAR method
FedCLEAR = CCRE local counterfactual risk learning + IRD invariant-residual distillation
```

The failure mode has already been validated on RAHFL. FedCLEAR v0.1 is now
implemented and locally tested; it has not yet produced a formal OpenI result.

Why the method targets CLE-HFL directly:

```text
CCRE:
  Generate explicit label-independent counterfactual views from a configurable
  operator bank. Compute classification risk for every present class and view,
  take a differentiable smooth maximum over views, then correct each class by
  its local batch-presence probability. Class counts stay local and are never
  uploaded.
  This targets the worst class-context risk instead of ordinary sample-average
  risk, so it is designed to improve WCCA and reduce CFG under label skew.

IRD:
  On public images, every client evaluates the same counterfactual views.
  Per-view logits are standardized across classes to remove heterogeneous model
  scale, then averaged into an invariant anchor. The server builds a leave-one-out
  coordinate-wise median teacher and each receiver minimizes its worst-view KL.
  Public data is only a response probe; it is not used to estimate private priors.
```

Privacy/fairness boundaries:

```text
FedCLEAR does not read train_corruption_ids or train_corruption_method_ids.
FedCLEAR does not upload private class counts.
FedCLEAR does not use test labels or test accuracy for teacher routing.
FedCLEAR does not aggregate model parameters or architecture-specific features.
RAHFL remains unchanged as AugMix/JSD + DCL + AsymHFL.
```

Core implementation files:

```text
fedprime/augmentations/counterfactual.py
fedprime/methods/ccre.py
fedprime/methods/ird.py
fedprime/methods/local_fedclear.py
fedprime/methods/fedclear.py
fedprime/methods/rahfl_asymhfl.py
scripts/run_experiment.py
```

Configs and OpenI entry:

```text
configs/debug_fedclear_cle.yaml
configs/openi_v100_fedclear_cle_gamma09_probe.yaml  # 12 rounds, first run
configs/openi_v100_fedclear_cle_gamma09_full.yaml   # 40 rounds, only after positive probe
scripts/openi_fedclear_entry.py
scripts/analyze_fedclear_probe.py
deliverables/baselines/rahfl_cle_alpha05_gamma09_seed0_round00_11.csv
```

OpenI startup file:

```text
scripts/openi_fedclear_entry.py
```

Recommended runtime parameter:

```text
--mode probe
```

The entry searches the mounted dataset for:

```text
cle_hfl_prepared_alpha05_gamma09_seed0.tar.gz
```

It imports the prepared data, checks the environment, runs unbuffered training,
summarizes outputs, packages them, copies them to `c2net_context.output_path`,
and calls `upload_output()`.

Local verification completed:

```text
13 unit/regression tests passed.
Two-round RTX 3050 smoke test passed:
  round 0: CCRE ran; IRD warmup correctly skipped communication
  round 1: IRD ran with finite loss/gradients and saved four checkpoints

round 1 diagnostic values:
  ccre_loss=2.9865
  ccre_worst_view_risk=2.4859
  ird_loss=0.7492
  ird_anchor_disagreement=0.9897
  ird_worst_view_kl=0.2087

RAHFL CLE debug regression also passed after the runner changes.
```

The debug run uses only two test batches, so its WCCA/CFG are not research
results. The 12-round OpenI probe uses the full counterfactual test set.

After probe training, the OpenI entry automatically compares FedCLEAR rounds
9-11 with the archived RAHFL rounds 9-11 and writes:

```text
probe_comparison.json
probe_comparison.md
```

RAHFL same-round reference:

```text
round 11: avg=37.4575, worst=30.6950, WCCA=8.1500, CFG=9.7250
rounds 9-11 mean: avg=36.6488, worst=30.4125, WCCA=8.1833, CFG=10.6558
```

Method document:

```text
docs/archive/methods/FEDCLEAR_METHOD_DESIGN_REVIEW_ZH.md
```

## CLE-HFL Diagnostic Route - 2026-07-08

New proposed paper direction:

```text
CLE-HFL = Corruption-Label Entanglement in Heterogeneous Federated Learning
FedCLEAR = current implemented method candidate, awaiting OpenI probe results
```

Core idea:

```text
Existing robust HFL studies corrupted clients.
CLE-HFL studies a finer failure mode: corruption-label shortcut.
Some classes are systematically tied to specific corruptions inside clients,
so models may learn "blur -> class A" or "clean -> class B" instead of semantics.
```

The prerequisite RAHFL diagnostic has completed and showed higher CFG and lower
WCCA as gamma increased. This justified implementing the current FedCLEAR v0.1.

Implemented for diagnostics:

```text
scripts/prepare_cle_data.py
scripts/import_cle_data.py
scripts/run_openi_cle_rahfl_diagnostic.sh
scripts/openi_cle_rahfl_diagnostic_entry.py
configs/debug_rahfl_cle.yaml
configs/diagnostic_rahfl_cle_alpha05_gamma00.yaml
configs/diagnostic_rahfl_cle_alpha05_gamma06.yaml
configs/diagnostic_rahfl_cle_alpha05_gamma09.yaml
docs/archive/methods/FEDCLEAR_CLE_HFL_PROPOSAL_ZH.md
```

OpenI training-task startup file:

```text
scripts/openi_cle_rahfl_diagnostic_entry.py
```

Use this Python entry when the OpenI UI requires a startup file. Runtime
parameters can be left empty for the default full diagnostic. The entry calls
`c2net.context.prepare()`, searches the mounted dataset path for the three
`cle_hfl_prepared_alpha05_gamma*_seed0.tar.gz` archives, runs the RAHFL
diagnostic configs, packages outputs, and uploads them through `upload_output()`.

OpenI diagnostic import fix:

```text
scripts/import_cle_data.py now extracts each tar.gz into a separate folder and
copies both cifar_10_cle and cifar_100 into RAHFL-master/Dataset.
```

This matters because OpenI mounts the uploaded data under paths such as
`/tmp/dataset/<dataset_name>/`. A previous import version extracted all gamma
archives into the same directory, which could accidentally match gamma00 when
importing gamma06/gamma09 and could miss the CIFAR-100 public tar.

Generated local CLE-HFL datasets:

```text
local_runs/cle_hfl_prepared/cle_hfl_prepared_alpha05_gamma00_seed0.tar.gz
local_runs/cle_hfl_prepared/cle_hfl_prepared_alpha05_gamma06_seed0.tar.gz
local_runs/cle_hfl_prepared/cle_hfl_prepared_alpha05_gamma09_seed0.tar.gz
```

Each archive is about 383 MB and contains:

```text
cifar_10_cle/<dataset_name>/
  client_i/train_images.npy
  client_i/train_labels.npy
  client_i/train_corruption_ids.npy
  client_i/train_corruption_method_ids.npy
  test_balanced/test_images.npy
  test_balanced/test_labels.npy
  test_balanced/test_corruption_ids.npy
  metadata.json
  audit/client_label_counts.csv
  audit/client_corruption_counts.csv
  audit/client_class_corruption_counts.csv
  audit/class_corruption_map.csv
cifar_100/cifar-100-python.tar.gz
```

Diagnostic metric meanings:

```text
WCCA = min accuracy over all class-corruption groups. Higher is better.
CFG  = average per-class gap between best and worst corruption context. Lower is better.
```

CLE-HFL RAHFL diagnostic result - 2026-07-10:

```text
Archive:
  outputs/cle_rahfl_diagnostic_outputs.tar.gz

Analysis directory:
  outputs/cle_rahfl_diagnostic_analysis/

Fixed conditions:
  alpha = 0.5
  seed = 0
  clients = 4
  samples_per_client = 10000
  model heterogeneity = ResNet10 / ResNet12 / ShuffleNet / MobileNetV2
  baseline = full RAHFL-style AugMix/JSD + DCL + AsymHFL

Varied factor:
  gamma = corruption-label entanglement strength
```

Main result:

```text
gamma=0.0:
  final avg_acc   = 52.17
  final worst_acc = 44.17
  final WCCA      = 35.35
  final CFG       = 2.54

gamma=0.6:
  final avg_acc   = 50.82
  final worst_acc = 42.83
  final WCCA      = 25.88
  final CFG       = 5.91

gamma=0.9:
  final avg_acc   = 46.72
  final worst_acc = 38.16
  final WCCA      = 19.32
  final CFG       = 10.91
```

Interpretation:

```text
As gamma increases from 0.0 to 0.9 while alpha and other settings are fixed:
  avg_acc drops by 5.45 points
  worst_acc drops by 6.02 points
  WCCA drops by 16.02 points
  CFG rises by 8.37 points

This is a strong initial signal that CLE-HFL exposes a RAHFL blind spot:
RAHFL may learn corruption-label shortcuts under entanglement, causing hidden
counterfactual class-corruption failures. The CLE-HFL scenario is therefore
initially supported as a benchmark/failure-mode direction.
```

Important caveat:

```text
This proves the problem/failure mode exists under seed0 alpha=0.5, not that our
future method has solved it. Next work must test/implement a method that improves
WCCA and reduces CFG under gamma=0.9, preferably without sacrificing avg_acc.
```

Local smoke test passed:

```text
python scripts/prepare_cle_data.py --output-root local_runs/cle_hfl_debug \
  --dataset-name alpha05_gamma09_seed0 --alpha 0.5 --gamma 0.9 \
  --seed 0 --num-clients 4 --samples-per-client 100 --max-test-images 200 \
  --include-public --make-tar

python scripts/import_cle_data.py \
  --source local_runs/cle_hfl_debug/cle_hfl_prepared_alpha05_gamma09_seed0 \
  --repo-root .

python scripts/run_experiment.py --config configs/debug_rahfl_cle.yaml
```

The debug run completed one round and wrote:

```text
outputs/debug_rahfl_cle_alpha05_gamma09/metrics.csv
outputs/debug_rahfl_cle_alpha05_gamma09/class_corruption_acc.csv
```

## FedSARA-CS New Scenario - 2026-07-08

New active scenario:

```text
model heterogeneity + label-skew Non-IID + corruption-skew Non-IID
```

This extends the previous RAHFL-style random corruption setting. Each client now
has both a different class distribution and a dominant corruption group:

```text
client 0: mainly noise
client 1: mainly blur
client 2: mainly weather
client 3: mainly digital
```

Protocol:

```text
alpha = 0.5
rho = 0.7
seed = 0
clients = 4
samples_per_client = 10000
test protocol = balanced noise / blur / weather / digital corruption groups
```

Generated prepared dataset:

```text
local_runs/fedsara_cs_prepared/fedsara_cs_prepared_alpha05_rho07_seed0.tar.gz
size: about 386 MB
```

Important files:

```text
fedprime/data/corruptions.py
scripts/prepare_corruption_skew_data.py
scripts/import_fedsara_cs_data.py
scripts/run_openi_fedsara_cs.sh
configs/openi_v100_rahfl_cs_alpha05_rho07.yaml
configs/openi_v100_fedsara_cs_alpha05_rho07.yaml
configs/debug_rahfl_cs.yaml
configs/debug_fedsara_cs.yaml
docs/experiments/archive/FEDSARA_CS_SCENARIO_OPENI_GUIDE_ZH.md
```

Both formal configs use:

```text
pretrain_epochs: 40
rounds: 40
batch_size: 64
public_batches_per_round: 4
```

The 40-epoch pretrain path uses a plain corruption-skew CE loader, not the
AugMix three-view loader. This avoids wasting compute while keeping RAHFL-CS and
FedSARA-CS fair. Formal communication rounds still use AugMix/JSD plus DCL or
SARA.

New metrics:

```text
worst_group_acc
worst_client_group_acc
corruption_group_acc.csv
client_group_acc.csv
```

Smoke tests passed locally:

```text
python scripts/run_experiment.py --config configs/debug_fedsara_cs.yaml
python scripts/run_experiment.py --config configs/debug_rahfl_cs.yaml
```

Both tests completed one round and wrote metrics/group metrics.

## Current Goal

Build a paper-worthy heterogeneous FL method for:

```text
model heterogeneity + data heterogeneity / Non-IID + data corruption robustness
```

The current baseline to beat is the unified-runner RAHFL baseline.

## Current Main Method

The current mainline is:

```text
SARA + AsymHFL = AugMix/JSD + Skew-Aware Robust Alignment + RAHFL AsymHFL
```

The project is no longer centered on D2C, FedPRIME-PAIR, PRAC-HFL, or FedCARA.
Those are historical diagnostic experiments. PRAC-HFL runner is still useful for
local-only controls because it has robust Kaggle heartbeat logging and can skip
communication with `warmup_rounds: 999`.

SARA is currently the best-performing mainline because SARA + original RAHFL
AsymHFL is the first setting that beats the fair RAHFL baseline on both final
average accuracy and final worst-client accuracy.

Key files/configs:

```text
fedprime/methods/sara.py
fedprime/methods/local_rahfl.py
fedprime/methods/rahfl_asymhfl.py
configs/kaggle_t4_sara_local_only.yaml
configs/kaggle_t4_sara_rahfl.yaml
configs/debug_sara_local_only.yaml
```

Latest pushed SARA commit:

```text
9df13a7 实现SARA偏斜感知鲁棒对齐
```

## SARA Design

SARA means:

```text
Skew-Aware Robust Alignment
```

It replaces RAHFL's DCL branch while keeping the RAHFL-style robust local base:

```text
CE(clean)
+ lambda_jsd * JSD(clean, aug1, aug2)
+ lambda_sara * SARA(clean_feature, weak_feature, strong_feature)
```

SARA contains:

```text
1. Skew-aware supervised contrastive alignment
   Uses client class counts to rebalance contrastive contributions from head and
   tail classes under label-skew Non-IID.

2. PRIME/AugMix-view reliability gate
   Uses strong-view true-class margin to down-weight unreliable augmented views.

3. Relation alignment
   Uses stable softmax(sim / T) relation matching instead of the more fragile
   softmax(exp(sim) / T) style relation.
```

Current implementation uses AugMix/JSD views from the RAHFL local base, not PRIME.
PRIME remains a historical route unless explicitly resumed.

## SARA Results - 2026-07-02

Result archive:

```text
outputs/sara_rahfl_results.tar.gz
```

Contained runs:

```text
SARA local-only:
  config: configs/kaggle_t4_sara_local_only.yaml
  method_name: prac_hfl
  communication disabled by warmup_rounds=999
  final avg_acc   = 54.10
  final worst_acc = 32.06
  best avg_acc    = 54.59 at round 38
  best worst_acc  = 33.96 at round 17

SARA + AsymHFL:
  config: configs/kaggle_t4_sara_rahfl.yaml
  method_name: rahfl
  cl_module: sara
  final avg_acc   = 57.83
  final worst_acc = 46.59
  best avg_acc    = 57.83 at round 39
  best worst_acc  = 46.59 at round 39
```

Main comparisons:

```text
RAHFL baseline:
  final avg_acc   = 56.41
  final worst_acc = 44.72

AugMix+DCL local-only:
  final avg_acc   = 56.11
  final worst_acc = 44.23

FedCARA v1:
  final avg_acc   = 55.88
  final worst_acc = 45.93
```

Interpretation:

```text
SARA local-only is not strong and should not be claimed as a standalone local
training improvement. It appears to over-regularize or hurt weak clients.

SARA + AsymHFL is currently the best mainline result:
  vs RAHFL: +1.42 avg_acc, +1.87 worst_acc
  vs AugMix+DCL local-only: +1.72 avg_acc, +2.36 worst_acc
  vs FedCARA: +1.95 avg_acc, +0.66 worst_acc

The key story is synergy:
  SARA alone may be too strict, but it changes local robust representations in
  a way that makes AsymHFL public-logit communication more effective under
  label-skew Non-IID.
```

## SARA Alpha=0.5 Seed Validation - 2026-07-05

New result archives:

```text
outputs/rahfl_seed1_results.tar.gz
outputs/sara_rahfl_seed12_results.tar.gz
```

Completed runs:

```text
RAHFL seed=1:
  config: configs/kaggle_t4_rahfl_seed1.yaml
  final avg/worst = 56.645 / 45.29
  best avg/worst  = 56.645 / 45.29

SARA + AsymHFL seed=1:
  config: configs/kaggle_t4_sara_rahfl_seed1.yaml
  cl_module: sara
  final avg/worst = 57.2975 / 46.23
  best avg/worst  = 57.2975 / 46.23
  paired gap vs RAHFL seed=1 = +0.6525 avg_acc, +0.94 worst_acc

SARA + AsymHFL seed=2:
  config: configs/kaggle_t4_sara_rahfl_seed2.yaml
  cl_module: sara
  final avg/worst = 58.0025 / 45.90
  best avg/worst  = 58.0025 / 45.90
```

SARA final results across alpha=0.5 seeds 0/1/2:

```text
seed0: 57.83   / 46.59
seed1: 57.2975 / 46.23
seed2: 58.0025 / 45.90

mean final avg_acc   = 57.71
mean final worst_acc = 46.24
population std avg   = 0.30
population std worst = 0.28
```

Available RAHFL final results across alpha=0.5 seeds 0/1:

```text
seed0: 56.41  / 44.72
seed1: 56.645 / 45.29

mean final avg_acc   = 56.5275
mean final worst_acc = 45.005
```

Important caveat:

```text
The archived partition files named seed0/seed1/seed2 have identical SHA-256
prefixes and identical client_class_counts in these runs. Therefore the current
alpha=0.5 seed validation is best interpreted as different training/randomness
seeds on the same fixed label-skew partition, not as different data partitions.

This is still useful for training stability, but formal paper claims should not
describe it as cross-partition validation unless new genuinely distinct
partition files are generated and audited.
```

Partition seed bug fix - 2026-07-06:

```text
Root cause:
  fedprime/data/loaders.py reused RAHFL-master/Dataset/sampling.py for IID and
  Dirichlet splits. That vendor file resets random.seed(0) and np.random.seed(0)
  at import time, so missing seed1/seed2 partition files could be generated with
  seed0 randomness despite different config seed names.

Fix:
  fedprime/data/loaders.py now implements local IID/Dirichlet partition
  generation with np.random.default_rng(partition_seed).
  All experiment runners and partition/audit/diagnostic scripts pass config.seed
  into partition_private_data().

Verification:
  A temporary alpha=0.5 seed0/1/2 generation produced distinct .npz SHA-256
  prefixes and each client had exactly 10000 samples.

Important:
  Existing historical archives are not changed. If a .npz already exists, the
  runner still loads it for reproducibility. To get genuinely different
  partitions, generate a new partition pack after this fix.
```

Interpretation:

```text
SARA + AsymHFL remains positive under the seed=1 matched comparison and SARA
seed=2 is also strong. The gain over RAHFL is smaller than the original seed0
gap but remains positive on both final average accuracy and final worst-client
accuracy for the completed matched seed=1 run.

The mainline claim is strengthened:
  SARA does not appear to be a seed0-only accident under the fixed alpha=0.5
  partition. It still needs RAHFL seed=2 and stronger/non-extreme alpha checks
  before final paper-level claims.
```

## SARA Alpha=0.3 Validation - 2026-07-06

Result archive:

```text
outputs/sara_vs_rahfl_alpha03_results.tar.gz
```

Setting:

```text
alpha=0.3
seed=0
corrupt_rate=1
rounds=40
same fixed partition for RAHFL and SARA
```

Results:

```text
RAHFL alpha=0.3:
  final avg/worst = 45.8425 / 41.9200
  best  avg/worst = 46.3825 / 43.1300

SARA + AsymHFL alpha=0.3:
  final avg/worst = 46.7325 / 42.7700
  best  avg/worst = 47.0825 / 44.1100
```

SARA gap:

```text
final avg_acc   +0.8900
final worst_acc +0.8500
best avg_acc    +0.7000
best worst_acc  +0.9800
```

Trend:

```text
SARA beats RAHFL in 36/40 rounds for avg_acc and 36/40 rounds for worst_acc.

Last-10-round mean gap:
  avg_acc   +0.6942
  worst_acc +0.5270
```

Partition audit:

```text
nonzero_classes_per_client = [7, 6, 7, 10]
max_client_class_proportion = [0.3625, 0.3669, 0.3583, 0.3716]
```

Interpretation:

```text
The alpha=0.3 split is clearly label-skewed and both methods use identical
client class counts. SARA still wins consistently, but the gain is modest and
smaller than the alpha=0.5 seed0 gain. Do not overclaim a large severe-Non-IID
advantage from this single alpha=0.3 run. Continue with alpha=0.1 and alpha=1.0.
```

Next required validation:

```text
1. Run RAHFL seed=2 under alpha=0.5 for the missing matched control.
2. Generate or verify genuinely distinct alpha=0.5 seed partitions if the paper
   needs cross-partition multi-seed claims.
3. Run alpha=0.1 to test stronger label skew.
4. Run alpha=1.0 to ensure normal/non-extreme Non-IID does not collapse.
5. Only redesign communication after these validations. Do not replace AsymHFL
   immediately, because SARA + AsymHFL is currently the strongest evidence.
```

Alpha validation preparation:

```text
New configs:
  configs/kaggle_t4_rahfl_alpha01.yaml
  configs/kaggle_t4_sara_rahfl_alpha01.yaml
  configs/kaggle_t4_sara_rahfl_alpha03.yaml
  configs/kaggle_t4_sara_rahfl_alpha10.yaml

New Kaggle launcher:
  scripts/run_kaggle_sara_vs_rahfl_alpha01.sh
  scripts/run_kaggle_sara_alpha0103.sh
  scripts/run_kaggle_sara_alpha0310.sh

New partition tools:
  scripts/build_partition_pack.py
  scripts/import_partition_pack.py

Local generated pack:
  local_runs/sara_partitions_alpha01_alpha03
  local_runs/sara_partitions_alpha01_alpha03.tar.gz
  local_runs/sara_partitions_alpha03_alpha10
  local_runs/sara_partitions_alpha03_alpha10.tar.gz

Suggested Kaggle dataset name:
  sara-partitions-alpha01-alpha03
  sara-partitions-alpha03-alpha10
```

The partition pack only contains fixed `.npz` partition files and audit metadata,
not CIFAR image data. It should be mounted together with the existing
`fedprime-data` dataset. This avoids re-uploading the large CIFAR-10-C/CIFAR-100
prepared data while keeping alpha=0.3 and alpha=1.0 partitions reproducible.

For the alpha=0.1 paired comparison, use:

```text
configs/kaggle_t4_rahfl_alpha01.yaml
configs/kaggle_t4_sara_rahfl_alpha01.yaml
scripts/run_kaggle_sara_vs_rahfl_alpha01.sh
```

Mount:

```text
/kaggle/input/fedprime-data
/kaggle/input/sara-partitions-alpha01-alpha03
```

If `PARTITION_SOURCE` is empty, the alpha=0.1 paired launcher will generate the
missing alpha=0.1 partition on the fly. It also performs a partition-only audit
for `configs/kaggle_t4_rahfl_alpha03.yaml`, so the final result archive includes
both:

```text
outputs/partitions/cifar10c_alpha01_seed0_clients4_samples10000.npz
outputs/partitions/cifar10c_alpha03_seed0_clients4_samples10000.npz
```

This lets the user download one archive and later reuse the alpha=0.3 partition
without another Kaggle data-generation pass.

Alpha=0.1 first Kaggle attempt interruption - 2026-07-06:

```text
RAHFL alpha=0.1 reached round 007 and interrupted with:
  FloatingPointError: RAHFL local phase: non-finite gradient at batch 26

The generated alpha=0.1 split was extremely skewed, for example:
  client0 mostly classes 2/8
  client1 mostly classes 3/4/5
  client2 mostly classes 7/9
  client3 mostly classes 0/1/6
```

Fix:

```text
configs/kaggle_t4_rahfl_alpha01.yaml now uses the same numerical safety settings
as SARA alpha=0.1:
  max_grad_norm: 5.0
  skip_nonfinite: true
  local_log_interval: 50
```

This is conservative for the baseline because it prevents RAHFL from crashing
and can only make the RAHFL comparison stronger/safer.

SARA Alpha=0.1 validation result - 2026-07-06:

```text
Result archive:
  outputs/sara_vs_rahfl_alpha01_results.tar.gz

Analysis deliverables:
  deliverables/sara_vs_rahfl_alpha01_analysis/

Setting:
  alpha=0.1, seed=0, corrupt_rate=1, rounds=40
  partition generated on Kaggle at run time
```

Results:

```text
RAHFL alpha=0.1:
  final avg/worst = 35.6825 / 29.3300
  best  avg/worst = 35.6825 / 29.3300

SARA + AsymHFL alpha=0.1:
  final avg/worst = 35.9625 / 29.1000
  best  avg/worst = 35.9625 / 29.3000
```

Gaps:

```text
final avg_acc   +0.2800
final worst_acc -0.2300
best avg_acc    +0.2800
best worst_acc  -0.0300
last10 avg gap  +0.1505
last10 worst gap -0.0060
```

Interpretation:

```text
This is essentially a tie, not a big win. SARA does not produce a strong
advantage at alpha=0.1. Both methods collapse to low absolute accuracy under the
extreme split. The earlier "larger gain under severe Non-IID" hypothesis is not
supported by this run.
```

Partition audit:

```text
alpha=0.1 nonzero_classes_per_client = [8, 6, 7, 8]
alpha=0.1 max_client_class_proportion = [0.4091, 0.3909, 0.4612, 0.3428]
```

The split has many effectively-missing classes with only a few samples, even if
the nonzero class count is not extremely small.

## SARA + Receiver-Side Class Residual - 2026-07-06

Motivation:

```text
SARA improves alpha=0.5 and modestly improves alpha=0.3, but alpha=0.1 is almost
tied with RAHFL. SARA alone cannot solve extreme missing-class transfer.
```

New method variant:

```text
SARA + AsymHFL + receiver-side class-aware residual KD
```

Key idea:

```text
Do not replace original AsymHFL.
Do not upload class counts.
The receiver computes a private class-need vector from its own local labels and
uses it only to reweight an auxiliary KD term on received public logits.
```

Communication loss:

```text
L_comm = L_AsymHFL + lambda_residual * L_private_class_residual
```

Implementation:

```text
fedprime/methods/rahfl_asymhfl.py
  _private_class_need_weights()
  method.class_residual switch

configs/kaggle_t4_sara_residual_rahfl.yaml
configs/debug_sara_residual_rahfl.yaml
scripts/run_kaggle_sara_residual_alpha05.sh
```

Default config:

```text
alpha=0.5, seed=0
lambda_residual=0.1
need_mode=inverse_count
need_power=0.5
smoothing=10
min_weight=0.5
max_weight=2.0
```

Privacy interpretation:

```text
Class counts are not uploaded. They are a receiver-local variable used only to
weight the local distillation objective. The server can still send ordinary
AsymHFL public logits.
```

First target comparison:

```text
RAHFL alpha=0.5 seed0:            56.41 / 44.72
SARA + AsymHFL alpha=0.5 seed0:   57.83 / 46.59
SARA residual target:             beat 57.83 / 46.59 or at least improve worst_acc
```

Result archive and analysis:

```text
Archive:
  outputs/sara_residual_alpha05_results.tar.gz

Analysis deliverables:
  deliverables/sara_residual_alpha05_analysis/
```

Final result:

```text
SARA + receiver-side residual AsymHFL:
  final avg/worst = 57.655 / 46.54
  best  avg/worst = 57.655 / 46.54

Gap vs RAHFL:
  +1.245 avg_acc
  +1.82  worst_acc

Gap vs SARA + AsymHFL:
  -0.17 avg_acc
  -0.05 worst_acc
```

Interpretation:

```text
The receiver-side residual is not a new breakthrough. It preserves most of
SARA + AsymHFL performance and still beats RAHFL, but it does not improve over
the simpler SARA + AsymHFL mainline. Do not spend scarce compute tuning this
residual unless future work specifically targets worst-client fairness.
```

## SARA + CCAD - 2026-07-07

Motivation:

```text
Small class-count or residual-KD changes are not enough as a paper-facing
communication innovation. CCAD keeps AsymHFL as the stable client-level route,
then adds public-sample corruption consistency as a sample-level communication
calibration signal.
```

Method name:

```text
CCAD = Corruption-Consistent Asymmetric Distillation
```

Core rule:

```text
For each public image u, each client predicts clean/augmented public views.
Teacher reliability is high when p(clean), p(aug1), p(aug2) are consistent and
the clean prediction is confident. Student need is high when the student is
uncertain or unstable under the same perturbations.

AsymHFL still provides the main client-level direction. CCAD adds a residual KD
term so that reliable teachers distill more strongly to needy students on
public samples where corruption consistency supports the transfer.
```

Implementation:

```text
fedprime/methods/rahfl_asymhfl.py
  method.communication: ccad
  _ccad_public_views()
  _ccad_collect_state()
  _ccad_pair_loss()

configs/kaggle_t4_sara_ccad.yaml
configs/debug_sara_ccad.yaml
scripts/run_kaggle_sara_ccad_alpha05.sh
```

Default first run:

```text
alpha=0.5, seed=0, corrupt_rate=1
SARA local module + AsymHFL + CCAD residual calibration
base_asymhfl_weight=1.0
lambda_ccad=0.2
max_pair_weight=2.0
```

Verification:

```text
python -m py_compile fedprime/methods/rahfl_asymhfl.py
config assertion passed for configs/kaggle_t4_sara_ccad.yaml and debug_sara_ccad.yaml
CCAD tensor unit smoke passed with tiny dummy models
local full debug could not run because local CIFAR-100 is not torchvision-valid
```

RAHFL multi-seed control preparation:

```text
New matching RAHFL seed configs:
  configs/kaggle_t4_rahfl_seed1.yaml
  configs/kaggle_t4_rahfl_seed2.yaml

New Kaggle launcher:
  scripts/run_kaggle_rahfl_seed12.sh
```

These configs are copied from the original unified-runner RAHFL baseline and
only change `seed`, `experiment_name`, and fixed partition path. They do not add
SARA-specific stability settings, so they remain an original RAHFL control.

Use the remaining RAHFL seed=2 control when preparing formal mean/std reporting:

```text
SARA seed=1 vs RAHFL seed=1: completed and positive
SARA seed=2 vs RAHFL seed=2: SARA completed, RAHFL seed=2 pending
```

Current alpha=0.5 validation status - 2026-07-05:

```text
Done:
  SARA + AsymHFL seed=1
  SARA + AsymHFL seed=2
  RAHFL seed=1

Pending:
  RAHFL seed=2
  alpha=0.3/0.1/1.0 SARA checks
  partition-generation audit because archived alpha=0.5 seed partition files
  are identical despite different seed names.
```

## CARA-L Design

CARA-L is the paper-facing name for the previously implemented NIR-DCL local
module. It modifies only the local DCL branch:

```text
CE(clean)
+ lambda_jsd * JSD(clean, aug1, aug2)
+ lambda_nir * NIR-DCL(clean_feature, weak_feature, strong_feature)
```

Implemented components:

```text
1. Class-balanced DCL
   Average per-class losses first, then average over classes present in the batch.

2. Client-local feature queue
   Each client keeps a private per-class feature queue to provide extra positives
   and negatives when a Non-IID mini-batch has too few tail-class samples.

3. Strong-view reliability gate
   Down-weights relation alignment when the strong AugMix view has a poor
   true-class margin.

4. Stable relation alignment
   Replaces the original DCL `softmax(exp(sim) / T)` style relation with a more
   stable `softmax(sim / T)` KL alignment.
```

Key files:

```text
fedprime/methods/nir_dcl.py
fedprime/methods/local_rahfl.py
configs/kaggle_t4_nir_dcl_local_only.yaml
configs/debug_nir_dcl_local_only.yaml
configs/kaggle_t4_nir_dcl_rahfl.yaml
```

## CARA-C Design

CARA-C is the FedCARA communication module. It replaces RAHFL AsymHFL's
overall-accuracy teacher routing with class-wise reliable teaching.

Original AsymHFL:

```text
If overall_acc(student) <= overall_acc(teacher),
the student learns the teacher's full public softmax distribution.
```

CARA-C:

```text
For receiver i, teacher j, class c:
  weight_{i,j,c} = reliability_{j,c} * need_{i,c}

where:
  reliability_{j,c} = per-class acc of teacher j on class c
  need_{i,c}        = 1 - per-class acc of receiver i on class c
```

The first implementation also uses:

```text
better_only: true
only teach class c if teacher_acc_{j,c} > student_acc_{i,c} + margin
```

The public-logit KL becomes:

```text
weighted_KL = sum_c weight_{i,j,c} * p_teacher,c * log(p_teacher,c / p_student,c)
```

Key files/configs:

```text
fedprime/methods/rahfl_asymhfl.py
configs/debug_fedcara_cifar10c.yaml
configs/kaggle_t4_fedcara.yaml
```

## CARA-L / NIR-DCL Results - 2026-07-01

Two Kaggle runs finished:

```text
outputs/nir_dcl_local_only_results.tar.gz
outputs/nir_dcl_rahfl_results.tar.gz
```

Results:

```text
NIR-DCL local-only:
  final avg_acc   = 53.30
  final worst_acc = 36.01
  best avg_acc    = 54.74 at round 37
  best worst_acc  = 37.37 at round 26

NIR-DCL + AsymHFL:
  final avg_acc   = 57.36
  final worst_acc = 46.23
  best avg_acc    = 57.89 at round 34
  best worst_acc  = 46.33 at round 34
```

Comparison:

```text
RAHFL baseline final:        avg_acc=56.41, worst_acc=44.72
AugMix+DCL local-only final: avg_acc=56.11, worst_acc=44.23

NIR-DCL local-only gap vs AugMix+DCL local-only:
  avg_acc=-2.81
  worst_acc=-8.22

NIR-DCL + AsymHFL gap vs RAHFL:
  avg_acc=+0.95
  worst_acc=+1.51
```

Interpretation:

```text
NIR-DCL alone hurts local-only performance, especially worst-client accuracy.
However, NIR-DCL combined with AsymHFL exceeds the RAHFL baseline on both
average accuracy and worst-client accuracy under the current alpha=0.5 setting.

This suggests NIR-DCL may improve the quality/compatibility of public-logit
communication even if it is too restrictive as a purely local objective.
The next research story should focus on synergy:
  RAHFL local DCL is strong by itself;
  CARA-L regularizes local representations so AsymHFL communication becomes
  more beneficial under Non-IID label skew.
```

## Next FedCARA Experiment

Run:

```text
configs/kaggle_t4_fedcara.yaml
```

Compare against:

```text
RAHFL baseline:        56.41 / 44.72
CARA-L + AsymHFL:      57.36 / 46.23
```

Goal:

```text
FedCARA should ideally match or exceed CARA-L + AsymHFL.
If it beats 57.36 / 46.23, the communication innovation is immediately useful.
If it stays above RAHFL but below CARA-L + AsymHFL, CARA-C still has a valid
class-aware communication story but needs tuning.
```

## FedCARA v1 Result - 2026-07-01

Result archive:

```text
outputs/fedcara_results.tar.gz
```

FedCARA v1:

```text
config: configs/kaggle_t4_fedcara.yaml
method_name: fedcara
local: CARA-L
communication: CARA-C class-weighted public-logit KD
```

Final and best metrics:

```text
FedCARA:
  final avg_acc   = 55.88
  final worst_acc = 45.93
  best avg_acc    = 56.86 at round 34
  best worst_acc  = 45.93 at round 39

RAHFL baseline:
  final avg_acc   = 56.41
  final worst_acc = 44.72

CARA-L + AsymHFL:
  final avg_acc   = 57.36
  final worst_acc = 46.23
```

Interpretation:

```text
FedCARA v1 does not beat RAHFL on final average accuracy:
  avg_acc gap vs RAHFL = -0.53

But it does beat RAHFL on final worst-client accuracy:
  worst_acc gap vs RAHFL = +1.21

It is also below CARA-L + original AsymHFL:
  avg_acc gap = -1.48
  worst_acc gap = -0.30
```

Current judgment:

```text
CARA-C v1 is not the final communication module yet.
It appears to bias learning toward weaker clients/classes, improving worst_acc
but sacrificing average accuracy. The class-aware communication direction is not
dead, but pure replacement of AsymHFL with hard class weighting is too conservative.

Best next version should be hybrid:
  keep part of original AsymHFL full-softmax KD
  add CARA-C class-aware weighted KD as an auxiliary or residual term
instead of fully replacing AsymHFL.
```

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

Safe PRAC-HFL public4 fair result from `outputs/prac_hfl_public4_results.tar.gz`:

```text
config used public_batches_per_round=4
final avg_acc=52.96, final worst_acc=43.27
best avg_acc=52.96 at round 39
best worst_acc=43.27 at round 39
mean accept_rate after warmup=25.5%
mean avg_delta after warmup=-0.0045
```

Comparison:

```text
RAHFL public4 final: avg_acc=56.41, worst_acc=44.72
PRAC public4 gap:    avg_acc=-3.45, worst_acc=-1.45
PRAC public1 final:  avg_acc=54.63, worst_acc=41.88
```

Interpretation:

```text
PRAC communication is not empty: accept_rate is nonzero and checkpoints change.
However, public4 did not improve over public1. More public batches lowered avg_acc
but improved final worst_acc over public1, suggesting PRAC may help weak clients
while causing average-performance negative transfer.
We still need AugMix+DCL local-only to decide whether PRAC adds real value over
local robust training alone.
```

Local-only control config:

```text
configs/kaggle_t4_augmix_dcl_local_only.yaml
method_name: prac_hfl
warmup_rounds: 999
meaning: AugMix + CE + JSD + DCL local training, no PRAC communication for all 40 rounds
```

AugMix+DCL local-only result from Kaggle log:

```text
final avg_acc=56.11, final worst_acc=44.23
best avg_acc=56.94 at round 38
best worst_acc=44.23 at round 39
prac_loss=0 and accept_rate=0 for all rounds
non-finite gradient warnings=822, all from client 2, skipped by skip_nonfinite=true
```

Comparison:

```text
RAHFL final:        avg_acc=56.41, worst_acc=44.72
PRAC public1 final: avg_acc=54.63, worst_acc=41.88
PRAC public4 final: avg_acc=52.96, worst_acc=43.27
Local-only final:   avg_acc=56.11, worst_acc=44.23
Local-only best avg exceeds RAHFL final avg by +0.53
```

Interpretation:

```text
Current PRAC communication does not add positive average-accuracy gain over
AugMix+DCL local robust training. The strongest evidence now is that most of the
performance comes from RAHFL-style local robust learning. Current PRAC should be
treated as weak/negative transfer unless redesigned. The main research direction
should shift toward Non-IID-aware robust DCL/local representation learning, with
communication as a secondary module.
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

1. Treat current PRAC communication as weak/negative transfer under the current design.
2. Shift main method design toward Non-IID-aware DCL/local robust representation learning.
3. If communication is kept, redesign it with held-out route/accept split and weaker aggregated updates.

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

## CLE-HFL Paper-Evidence Results - 2026-08-07

Two strict 12-round seed-0 screens completed on the fixed CLE-HFL v2
`alpha=0.5`, `gamma=0.9`, `seed0_split0` scenario. All comparisons used the
same heterogeneous models, initialization policy, fit/audit roles,
AsymHFL-val communication where applicable, and reporting-only final test.

### External-baseline screen

The completed arms were Local-only, FedMD, RHFL, native 1024-dimensional
FedProto, AugHFL, RAHFL, and the full candidate. Candidate-minus-RAHFL
last-five deltas were:

```text
Avg +3.9377, Worst +3.9040, WCCA +5.0500, CFG -6.3200
```

The candidate led this screen on both seen and unseen corruption operators.
This is a fixed-scenario, single-training-seed screening result, not a complete
SOTA claim. FedDF, KT-pFL, and FCCL were absent from this completed result.
Their core-mechanism adapters and matched five-arm entry were implemented on
2026-08-09. The formal 12-round screen later completed the same day. RAHFL and
PEW+BER exactly reproduced historical A0/A1. Last-five results were:

```text
method    Avg       Worst     WCCA    CFG
FedDF     23.6607   19.2507   0.35    38.395
KT-pFL    23.6587   19.5467   0.35    38.730
FCCL      23.3163   19.2280   0.70    37.400
RAHFL     30.0853   25.0427   0.85    30.440
PEW+BER   34.6320   29.4280   7.25    24.640
```

The three new core adaptations were 6.42--6.77 Avg points below RAHFL and
showed no rapid late catch-up, so none qualified for 40-round promotion. This
does not claim that every original-recipe implementation of those papers would
fail. Evidence:

```text
outputs/cle_remaining_baselines_seed0_12round_outputs.tar.gz
deliverables/cle_remaining_baselines_20260809/RESULT_SUMMARY_ZH.md
docs/experiments/archive/CLE_REMAINING_BASELINES_OPENI_RUN_ZH.md
```

Evidence:

```text
outputs/cle_external_baselines_seed0_12round_outputs.tar.gz
deliverables/cle_external_baselines_20260807/RESULT_SUMMARY_ZH.md
```

### A0--A6 local ablation

Last-five values and attribution relative to A0 RAHFL:

```text
arm                 Avg      Worst    WCCA    CFG
A0 RAHFL          30.0853   25.0427   0.85  30.440
A1 BER-only       34.6320   29.4280   7.25  24.640
A2 CDep-only      30.4070   24.7707   1.15  30.775
A3 full           34.0230   28.9467   5.90  24.120
A4 fixed PEW      33.5820   28.6040   5.05  27.070
A5 shuffled PEW   31.5437   26.0147   2.55  37.375
A6 oracle         35.1200   30.7253   7.70  20.690
```

```text
BER-only - RAHFL:  Avg +4.5467, Worst +4.3853, WCCA +6.4000, CFG -5.8000
CDep-only - RAHFL: Avg +0.3217, Worst -0.2720, WCCA +0.3000, CFG +0.3350
Full - RAHFL:      Avg +3.9377, Worst +3.9040, WCCA +5.0500, CFG -6.3200
Full - fixed PEW:  Avg +0.4410, Worst +0.3427, WCCA +0.8500, CFG -2.9500
Full - shuffled:   Avg +2.4793, Worst +2.9320, WCCA +3.3500, CFG -13.2550
Oracle - full:     Avg +1.0970, Worst +1.7787, WCCA +1.8000, CFG -3.4300
```

PEW group accuracy was 62.21% with calibration versus 39.99% with the fixed
0.55 threshold. BER is the dominant positive local component. Correct PEW
environment association and calibration contribute materially, with remaining
headroom to the oracle. CDep alone is neutral/slightly negative; full versus
BER-only loses `0.6090` Avg, `0.4813` Worst, and `1.3500` WCCA on last-five,
while improving CFG by only `0.5200`. Do not claim an independent stable CDep
benefit until the frozen lambda sensitivity is complete.

Evidence:

```text
outputs/cle_local_ablation_12round_seed0_outputs.tar.gz
deliverables/cle_local_ablation_20260807/RESULT_SUMMARY_ZH.md
deliverables/cle_local_ablation_20260807/independent_analysis.json
```

Next decision: run the focused CDep sensitivity on the same seed-0 scenario,
then freeze either calibrated PEW+BER+CDep or the simpler calibrated PEW+BER
before running the CIFAR-100-private second-dataset A/B.

## CDep Lambda Sensitivity - 2026-08-07

The matched 12-round seed-0 sensitivity compared CDep lambda 0.01, 0.05, and
0.10. All arms were complete; generated PEW annotations were byte-identical;
configs differed only in experiment name and lambda. Last-five results were:

```text
method            Avg       Worst     WCCA      CFG
BER-only A1       34.6320   29.4280   7.2500   24.6400
CDep lambda .01   34.1847   29.1053   6.0000   24.4350
CDep lambda .05   34.0230   28.9467   5.9000   24.1200
CDep lambda .10   33.9827   29.0373   5.9000   25.2500
```

Lambda 0.01 was the best CDep accuracy setting but still lost to BER-only by
`0.4473` Avg, `0.3227` Worst, and `1.2500` WCCA, with only `0.2050` lower CFG.
Increasing lambda reduced the measured dependence proxy monotonically but did
not improve robust classification. Current batch-local CDep is therefore not
a validated additive component. Do not continue lambda-only tuning.

Evidence:

```text
outputs/cle_sensitivity_12round_outputs.tar.gz
deliverables/cle_sensitivity_20260807/RESULT_SUMMARY_ZH.md
```

A separate protocol limitation was identified: seed-0 private-unseen operators
`impulse_noise`, `zoom_blur`, `fog`, and `pixelate` are excluded from client
fit data but are included in the public PEW augmentation library. Existing
unseen metrics establish private-fit holdout, not global operator holdout. A
strict PEW leave-one-operator-out experiment is required for a stronger claim.

## CDep-v2 Implementation Ready - 2026-08-07

After the lambda sensitivity failed, one final structural CDep revision was
implemented without changing PEW, BER, AsymHFL-val, audit routing, or legacy
CDep. CDep-v2 aligns confidence-weighted environment feature centroids within
each class using a bounded client-local cross-batch memory. It requires at
least two supported environments, warms up for two rounds, and ramps over
three rounds. Stored projected features are detached and never communicated.

Frozen single-arm entry and decision contract:

```text
entry: scripts/openi_cle_cdep_v2_entry.py
archive: cle_cdep_v2_12round_outputs.tar.gz
reference: matched calibrated PEW+BER A1
gates: ΔAvg >= 0, ΔWorst >= 0, ΔWCCA >= 0, ΔCFG <= -0.5
```

Focused tests passed (`24 passed`). A three-round local smoke confirmed buffer
growth `8 -> 20 -> 29`, ramp `0, 0, 0.33`, and nonzero round-2 CDep-v2 loss.
Smoke accuracy is not evidence.

## CDep-v2 Single-Arm Result - Attribution Inconclusive - 2026-08-08

The complete 12-round CDep-v2 run mechanically compared to historical PEW+BER
A1 as `Avg +0.3607`, `Worst +0.2000`, `WCCA -0.2500`, `CFG -0.7900`; three of
four frozen gates passed. CDep-v2 was active with last-five mean loss 0.04103,
7.0477 valid classes, 41.1299 valid groups, and a 2694.91 mean buffer size.

However, the CDep-v2 entry retrained PEW under a new checkpoint path. Its
private group accuracy was 68.075% versus 62.21% for historical A1, its
calibrated threshold was 0.22 versus 0.0, and all four client annotation hashes
differed. The historical automatic comparison is not a matched causal CDep
test. Preserve the mechanical `pass=false`, but scientific status is
`INCONCLUSIVE_FOR_ATTRIBUTION`.

Required next experiment: one paired task that prepares PEW once and runs
PEW+BER control and identical-PEW PEW+BER+CDep-v2 candidate. Keep the original
last-five gates and do not rerun CDep-v1.

The paired entry was implemented on 2026-08-08:

```text
scripts/openi_cle_cdep_v2_paired_entry.py
```

It runs control then candidate with `outputs/pew_checkpoints/
cle_cdep_v2_paired_seed0.pt`, requires all four PEW annotation NPZ files to be
byte-identical, and only then writes the unchanged four-gate decision. Focused
tests (`23 passed`) and an entry dry-run passed.

Evidence:

```text
outputs/cle_cdep_v2_12round_outputs.tar.gz
deliverables/cle_cdep_v2_20260808/RESULT_SUMMARY_ZH.md
```

## CDep-v2 Matched Shared-PEW Result - NO-GO - 2026-08-08

The required paired task completed with 12 rounds for PEW+BER control followed
by 12 rounds for the CDep-v2 candidate. Both arms contain exact rounds 0--11.
All four client PEW annotation NPZ files are byte-identical, and the resolved
configs differ only in experiment name and CDep-v2 settings. PEW diagnostics
also match exactly: private group accuracy 62.21%, threshold 0.0, validation
environment accuracy 57.4%, ECE 0.03412, and unknown AUROC 0.81671.

Independent last-five candidate-minus-control recomputation:

```text
Avg -0.1933, Worst -0.2280, WCCA -0.6000, CFG +0.4450
```

All four pre-registered gates failed. Seen and private-unseen metrics both
degraded. CDep-v2 was active rather than empty: last-five loss 0.04171, mean
valid groups 41.8807, and mean buffer size 2730.86. Mean round time increased
from 97.7426 to 99.2407 seconds (about 1.53%).

Decision: `NO-GO`. Freeze CDep-v1 and CDep-v2; do not continue structural,
lambda, buffer, or threshold tuning. The final local method is calibrated
PEW+BER. Before further paid experiments, propagate this frozen definition to
all prepared communication, cross-scenario, stress-grid, and second-dataset
candidate configs and re-run focused tests.

Evidence:

```text
outputs/cle_cdep_v2_paired_12round_outputs.tar.gz
outputs/cle_cdep_v2_paired_20260808/
deliverables/cle_cdep_v2_paired_20260808/RESULT_SUMMARY_ZH.md
docs/experiments/archive/CLE_CDEP_V2_PAIRED_OPENI_RUN_ZH.md
```

## Strict PEW Operator-LOO Result - GO - 2026-08-09

To test whether PEW+BER depends on public PEW exposure to the four seed-0
private-unseen concrete operators, an optional PEW public-operator exclusion
protocol was added. The default exclusion list is empty and preserves all old
behavior. Strict checkpoints persist their exclusion list and cannot be reused
under a mismatched protocol.

The new entry runs three sequential matched 12-round arms:

```text
RAHFL
standard calibrated PEW+BER
Strict-LOO calibrated PEW+BER
```

Strict LOO excludes `impulse_noise`, `zoom_blur`, `fog`, and `pixelate` from
both public PEW train and validation generation. It audits that the four have
zero occurrences in private fit, leaves AugMix/BER/AsymHFL unchanged, and
disables CDep. The standard and strict PEWs use separate checkpoints. Primary
last-five gates for Strict LOO minus same-task RAHFL are the original
`Avg >= +1.5`, `Worst >= +1.0`, `WCCA >= 0`, and `CFG <= -1.0`.

The formal three-arm result completed with all rounds 0--11. Independently
recomputed last-five values were:

```text
method             Avg       Worst     WCCA     CFG
RAHFL              30.0853   25.0427   0.8500   30.4400
standard PEW+BER   34.6320   29.4280   7.2500   24.6400
Strict-LOO PEW+BER 34.9880   31.2973   5.4500   24.3300
```

Strict-LOO minus RAHFL was `Avg +4.9027`, `Worst +6.2547`, `WCCA +4.6000`,
`CFG -6.1100`; all four frozen gates passed. Strict-LOO minus standard PEW was
`Avg +0.3560`, `Worst +1.8693`, `WCCA -1.8000`, `CFG -0.3100`.

All four held-out operators had zero private-fit counts and were absent from
Strict PEW public train/validation pools. This validates operator-level LOO
generalization within the known PEW families, not unseen-family or arbitrary
composite-corruption robustness.

```text
dataset: openi_cle_hfl_v2_alpha05_gamma09
entry: scripts/openi_cle_pew_loo_entry.py
args: none
archive: cle_pew_loo_12round_seed0_outputs.tar.gz
guide: docs/experiments/archive/CLE_PEW_LOO_OPENI_RUN_ZH.md
report: deliverables/cle_pew_loo_20260809/RESULT_SUMMARY_ZH.md
```

## CLE Local-First Shortcut -> PIDR -> PNCB-SCDW - 2026-08-30/31

Phase-A0 established a directional shortcut rather than generic degradation: historical RAHFL
`gamma=0.9` minus `gamma=0` pooled DSA was `+0.2016229552`, paired CI95
`[0.1964123272, 0.2072188988]`, with 4/4 clients positive and binding-permutation
`p=0.000999001`.

Phase-A1a then trained matched H0/H9/L0/L9 arms for 40 rounds. HFL and Local CLE effects were
`+0.2027476596` and `+0.2043658778`; communication difference-in-differences was
`-0.0016182182`, CI95 `[-0.0033365882, 0.0001283891]`, with only 1/4 positive clients. Verdict:
`NO_GO_FL_SPECIFIC_AMPLIFICATION`. The supported mechanism is local-first; do not revive bad-teacher,
routing-amplification, D2C/Oracle-D2C or communication-rescue narratives.

A paper-only identifiability audit showed that one already-corrupted image plus its task label and
i.i.d. AugMix views cannot distinguish a semantic predictor from a predictor using a persistent base
degradation. A zero-training oracle PIDR gate then showed that valid interventions make the hidden
direction observable: round-40 H0/H9 mAP was `0.441855/0.844847`, L0/L9 was
`0.430622/0.865557`, with 4/4 positive clients and null `p=0.000999`.

The conditional method is Public Nuisance Canonicalization Bridge (PNCB) plus Signed
Class-Directional Withdrawal (SCDW). PNCB learns public AugMix-to-source reconstruction on CIFAR-100
with labels ignored. Frozen PNCB gives paired endpoint `C(X)`. SCDW penalizes only statistically
credible positive wrong-class withdrawal `p(c|X)-p(c|C(X))`, with canonical probability and standard-
error threshold stop-gradient. The future local target keeps AugMix/JSD/DCL and adds canonical-view
CE plus SCDW; AsymHFL communication remains unchanged. Full classifier integration has not started.

Phase-B0 is bridge-only. The 535,256,689-byte input archive has SHA256
`DFB766F6494A5F61AA16F45666EC250A30501066AB54D89C984CD2324293B9BC` and contains 1,000 balanced
CIFAR-10 evaluation sources, CIFAR-100 public tar, and 16 final H0/H9/L0/L9 round-40 checkpoints.
The OpenI smoke completed on CUDA with 19/19 manifest files, 16/16 classifiers, one PNCB epoch/two
batches and 20x16 evaluation. Returned caches were finite with maximum probability-sum error
`2.38e-7`. Result archive bytes/SHA256 were `3344564` /
`C57D3A9BE84E04FDDBB35402DF011B59294D78B532951346476182A891E81E54`; verdict was
`SMOKE_ONLY_NO_SCIENTIFIC_DECISION`.

The smoke PNCB had poor bridge metrics, as expected after only two batches; these are explicitly
non-scientific. The frozen formal run then completed with 50,000 public images, 10 epochs, 1,000x16
evaluation and all 16 classifiers. Formal archive bytes/SHA256 are `31788115` /
`8F824A6EF21AFDF8E8CF089530786882FE684504079C17160DFD2205D140BE2C`.

Formal verdict: `NO_GO_PNCB_BRIDGE`. G1/G4/G6/G7 passed; G2/G3/G5 failed. Worst semantic delta was
only `-0.5875pp`, but within-source cross-operator variance contraction was `-12.2267%` versus the
required `+25%`, and family-separability relative reduction was only `9.4729%` versus `30%`. Local
gamma9 retrieval was mAP `0.593924`, hit `0.65`. PNCB loss decreased normally across all ten epochs,
so this is a scientific bridge failure, not an execution failure. Stop the current PNCB-SCDW route;
do not implement Phase-B1 and do not rescue it by SCDW-weight or epoch/channel/loss-only tuning.

Full external-discussion handoff:

```text
docs/research/status/CLE_PNCB_SCDW_CURRENT_RESEARCH_HANDOFF_FOR_GPTWEB_2026_08_31_ZH.md
deliverables/cle_public_canonicalization_phase_b0_20260831/RESULT_SUMMARY_ZH.md
```

## CLE Public-Carrier K0-A Transfer Oracle - 2026-09-01

PNCB bridge NO-GO 后，K0-A 不训练新模型，而是复用16个冻结 round-40 H0/H9/L0/L9
checkpoint，在1,000个固定 CIFAR-100 train 公共载体（标签未使用）上施加16个 severity-3
oracle operators，检验类别条件方向矩是否跨语义载体迁移。所有response在打开CLE binding
与operator-family真值前保存并哈希。

正式归档 `cle_public_carrier_k0a_seed0_formal_outputs.tar.gz` 为48,651,705 bytes，SHA256
`AA260672FED05C991DDEF2308342BD88150CA8A36FD8366EF9A9E85B2E523168`。16/16 response hash
匹配，1,000公共索引唯一；独立复算response mean、1,000 permutation与1,000 paired
bootstrap均与返回结果一致。

HFL H0/H9 mAP为`0.406176/0.796627`，delta `+0.390451`；Local L0/L9为
`0.411342/0.811235`，delta `+0.399894`。两者均4/4客户端同向，hit为`0.80/0.85`，
class-map和probe-identity null均为`p=0.000999001`。HFL directional-strength delta CI95
`[8.745423,9.236294]`、coherence delta CI95 `[0.310375,0.322847]`；Local分别为
`[10.606721,11.241361]`与`[0.254255,0.266638]`。HFL与Local各10/10冻结门槛通过，
判定`GO_TO_K0_B`。

解释边界：H0/L0 split cosine本身已高达`0.963/0.976`，所以普通corruption也能形成
跨载体稳定方向；CLE特有证据是方向强度/coherence大幅增加且能恢复隐藏binding。K0-A
仍使用真实operator bank，是oracle机制审计，不证明taxonomy-free probes、训练修复或跨场景
泛化。下一步仅设计K0-B generic-probe gate；不得直接进入DME/K1。

```text
deliverables/cle_public_carrier_k0a_20260901/RESULT_SUMMARY_ZH.md
```

## CLE K0-B v2 Taxonomy-Free Generic Probe Implementation - 2026-09-01

K0-B针对K0-A控制臂split cosine也很高的关键问题，将主对象收紧为
`carrier-stable + class-selective`方向响应。它复用同一16个冻结checkpoint和同一1,000张
CIFAR-100公共载体，不使用标签、真实operator/type/family/severity、CLE binding或private
corruption metadata。Ua/Ub固定为前后两个500-image disjoint halves。

新增冻结PRIME实现先采样完整recipe state再确定性应用，禁止逐carrier重采样。Bank A/B
各64 recipes，seeds为20260902/20260903，canonical hashes分别为
`6CAE529D4240715162B19B3968D47FA037A940B4D52D688FF52B859C5523DC01`与
`4A53497EC5DB6EC05C312E6166109FA4B52A5CC402CCE74E6EDB1253D913BF4E`。约5 MiB完整state与
manifest已版本化，包含composition/depth、谱系数、位移场、color coefficients、filter
kernel、strength和mixing参数及逐数组/逐recipe hash。

主统计为cross-fit kappa、class selectivity与`rho=kappa*[selectivity]+`；每个client仅保留
energy不低于该bank内median的active probes，R为active rho的top-20% CVaR。Combined bank
在128 probes上重新计算。正式门槛同时要求HFL/Local的Dcf/K/R增量、3/4客户端同向、两个
独立bank replication；若只有S增加而K/R失败，强制
`NO_GO_GENERIC_DIRECTIONAL_SIGNAL`。

聚焦K0-B及K0-A回归14/14通过。真实资产本地OpenI路径smoke验证19个输入文件，严格加载
H0/H9 client0，输出有限`(8,4,10)`响应；重复运行的blind和primary manifest hash完全一致。
Smoke archive为4,466,207 bytes，SHA256
`83442136ECC2D168F54E3E7283CA37135D084924632BE5B70C34A209B32DA543`，判定仅为
`SMOKE_ONLY_NO_SCIENTIFIC_DECISION`。下一步先OpenI smoke审计，再由用户批准formal；不训练。

## CLE K0-B Formal GO 与 K1-A Head-only SDMN - 2026-09-02

K0-B formal archive为234,888,047 bytes，SHA256
`1E02A16C765D8AB976A692D444FA9DAEBE38C30F8279CD6DCCFC49D1BFF88608`。16份response、primary
manifest、1,000公共index和两套frozen bank均通过完整性审计；从原始tensor独立重算S/Dcf/K/R
最大绝对误差`5.56e-17`，verdict一致为`GO_TO_K1_CHECKPOINT_SURGERY`。

HFL K delta `+0.252727`、R ratio `4.901569`、4/4客户端同向；Local K delta
`+0.232752`、R ratio `4.385780`、4/4同向。Bank A/B ratio分别为HFL
`5.739226/4.317300`、Local `5.166668/4.094945`；generic-fragility kill未触发。结论仅是
taxonomy-free detectability成立，不是纠正方法GO。

K1-A内部方法为CDR检测器加Selective Directional Moment Neutralization（SDMN）。它冻结
backbone/BN/dropout，只更新`model.linear`；在K0-B discover集上冻结high-rho probe方向，在
互斥surgery集上精确计算full-carrier directional moment，并受公共行为anchor KL约束。对照为
Direction-Sham、sensitivity-matched Random-Probe和Generic Invariance。Bank A/B做AB/BA
cross-fit，最终先看unseen-bank R，再封存primary后才允许DSA/WCCA/CFG oracle评价。

INSPECT确认16/16 checkpoint、两套bank、公共tar和CLE evaluator资产可复用；四种异构模型均有
`backbone + linear`接口。聚焦K1/K0回归20/20通过。本地H9 client0、A->B tiny smoke覆盖全部
五臂，所有2步objective下降、anchor KL低于0.003；第二次运行的split、probe selection、feature
hash、metrics和traces完全相同。Smoke只判
`SMOKE_ONLY_NO_SCIENTIFIC_DECISION`。

Public split在结果前固定：K0-B discover seed 20260901保持不变；排除完整1,000 discover后，
surgery/holdout使用seed 20260906各抽2,000张；sham base seed为20260907。Formal仍被代码锁定，
因为网页handoff未冻结maximum surgery steps及完整optimizer/backtracking contract。下一步先
OpenI smoke，再运行只看public surgery loss/anchor KL的calibration；审计后再冻结formal。

## CLE K1-A / K1-B0 / K1-C0 / K1-C Progress - 2026-09-02/04

K1-A head-only SDMN formal 已冻结为 `NO_GO_DIRECTIONAL_SURGERY`。它说明只修改 classifier
head 不能把 K0-B 的检测信号变成真实 CLE 改善；禁止靠学习率、步数或 head-only 变体复活。

K1-B0 随后检验 high-risk probes 是否定位一个跨 bank、相对 energy-matched low probes 特异的
共享表征 nuisance subspace。正式结果为 `NO_GO_SHARED_NUISANCE_ROUTING`。该失败不得通过降低
1.20 specificity gate 或重新选择 probe 挽救，但它暴露了另一条现象：强 CLE 模型的许多
generic intervention response 可能共同集中到少数表征方向。

K1-C0 因而被设计为零训练、零参数修改的观察性 response-spectrum gate。对冻结 H0/H9/L0/L9
round-40 checkpoint 和 2,000 张无标签 CIFAR-100 carriers，使用两套各 64 个冻结 PRIME banks，
计算归一化跨 carrier 表征响应矩阵的 Gram 谱集中度 `chi`，并以 clean-feature spectrum 排除一般
feature collapse。HFL 与 Local 的 10/10 预注册门槛全部通过，verdict 为
`GO_TO_K1_C_CRSF_SURGERY`。该 GO 只证明“存在值得干预的响应谱集中对象”，不预测压平它一定
降低 CLE，也不是训练方法 GO。

K1-C 的候选干预是 Cross-Carrier Response Spectrum Flattening (CRSF)。它只在 H9/L9 的
late backbone block 上做 checkpoint surgery，冻结 early backbone、全部 normalization、classifier
和 projector；比较 Frozen、CRSF、SharedMean、Generic Invariance 与 RawSpec。AB/BA 使用独立
PRIME correction/evaluation banks。当前真实模型 INSPECT、四类精确矩梯度对 direct full-graph
autograd 的数值等价检查、以及 H9 client0 AB tiny smoke 均通过；正式 CLE 结论尚不存在。

Exact K1-C calibration 规定 2,000 carriers x 64 probes、两遍 full-carrier sufficient-statistic
gradient、三个学习率和每个 candidate update 后不可删除的 exact objective + anchor-KL 复评。
第一次 OpenI 尝试约两小时仍主要停留在 transformed-input/cache 路径，accelerator utilization
接近零，未产生 calibration verdict。它是执行/工程 NO-GO，不是 `NO_GO_CRSF_INTERVENTION`。

成本定位现冻结如下：K0-B 当前作为训练前后可选的离线论文审计，不进入常规训练决策闭环；这
是当前设计选择而非永久禁止未来轻量 detector。Exact K1-C 只是一轮机制上界/因果验证器，绝不
允许原样放入每轮客户端训练。若 K1-C 最终通过，必须先经过 CRSF compression/efficiency gate，
验证随机小 carrier/probe 近似的方向与效果，并要求推理额外开销为零、通信近零、内存有界、训练
开销现实；压缩失败则 CRSF 不能成为最终主方法。

截至 2026-09-04，旧版 K1-C-FULL 已在任何 Formal 科学结果产生前冻结为
`SUPERSEDED_BEFORE_FORMAL`。其规范和实现只保留 provenance，禁止重启 calibration/formal；这不是
看结果改协议，因为没有产生 K1-C 科学结果。

新的 K1-C-Minimal Causal Intervention Gate 只检验 `chi_response下降是否导致DSA下降`。Formal
primary 冻结为 H9/L9、ResNet10(client0)/MobileNetV2(client3)、A-to-B、Frozen/CRSF/RawSpec。
Correction 从原 D_surgery 池预先固定512 carriers和Bank A的16 probes，固定初始LR 1e-4，运行5个
accepted steps，保留exact post-update objective、KL<=0.02、rollback与确定性LR halving；不再单独
做三学习率calibration。最终taxonomy-free评价不缩水，仍使用完整独立D_holdout 2,000 carriers ×
Bank B 64 probes，封存并hash后才允许读取CLE oracle并计算DSA。

实现入口为 `scripts/openi_cle_k1_c_minimal_entry.py`，预注册配置为
`configs/cle_k1_c_minimal_seed0.json`。聚焦测试14/14 PASS；真实H9/ResNet10 CUDA smoke完整通过，
确认选择hash、三arm路径、CRSF/RawSpec各一个accepted exact step、unseen流式评价、无磁盘变换缓存
及oracle隔离，verdict仅为`SMOKE_ONLY_NO_SCIENTIFIC_DECISION`。

下一步唯一允许的OpenI动作是Minimal `--mode=benchmark`：只测H9/ResNet10/Bank A，并外推Minimal
Formal的ETA/GPU-hours。用户审阅成本前不得运行Formal。Minimal失败即`NO_GO_CRSF_INTERVENTION`；
通过也只允许B-to-A与其余两个架构replication，不允许直接完整训练。K0-B在当前论文方案中固定为
离线审计，不进入训练决策闭环；K1-C-Minimal本身也不是最终训练算法。

K1-C-Minimal OpenI benchmark 随后完成并独立核验。原始包19,398 bytes，SHA256为
`D16E82F85FFA636DBEE50086BF6A083F932BB1F8833F3F7E366F5E90AF24F2D4`。实测H9/ResNet10的
512x16 prefix为6.9237秒，两条arm各一步总计11.7909秒，128x8 unseen三arm为1.3286秒；线性
外推四context Minimal Formal为894.39秒/0.2484单卡GPU-hours。考虑MobileNet未直接计时和oracle
阶段为代理估计，实际按30--45分钟预算。prefix常驻数组544 MiB、CUDA峰值341.9 MiB、磁盘变换
cache为0。所有artifact/source hash复核一致，evaluation未解压、标签/binding未加载。成本门通过，
但该结果仍为`BENCHMARK_ONLY_NO_SCIENTIFIC_DECISION`；必须由用户明确确认后才能运行Formal。

用户确认后，K1-C-Minimal Formal于2026-09-04完成。原始包58,124,491 bytes，SHA256为
`E07B9E75E2AEDDE0C1B3A4FF018CE0B4FD90EAA6CB88144D6E0D98588E43D4CA`。正式verdict为
`NO_GO_CRSF_INTERVENTION`：H9/L9 mean unseen-chi reduction仅`5.369%/5.452%`，低于15%；
CRSF相对RawSpec优势仅`3.838/3.959pp`，低于10pp。DSA reduction仅`0.005237/0.005193`
（相对`2.322%/2.308%`），低于absolute 0.05或relative 25%；相对RawSpec DSA优势仅
`0.005487/0.005903`，低于0.02。两个client方向均为正且energy retention约94--95%，但效果主要
来自ResNet10；MobileNetV2的chi reduction仅约2.2--2.4%，DSA reduction约0.001。

独立审计确认31项artifact hash、18项primary seal hash全部一致；72组moments/Gram重算chi/energy
误差为0；6组predictions重算DSA和任务指标误差为0；8条优化轨迹均完成5个accepted steps，目标
单调且最大最终KL为0.019646。失败不是代码或封存错误。按预注册规则，停止CRSF，不做B-to-A、
剩余架构、调参、replication或完整训练。K1-C0只保留“CLE模型存在响应谱集中”的观察性发现，
不能再解释为该集中度是足够强的因果杠杆。
