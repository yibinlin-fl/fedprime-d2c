# FedPRIME-D2C / PRAC-HFL Current Project Memory

Updated: 2026-07-19

## Latest Override: FedEASE v2.1 Complete Candidate - 2026-07-19

Current research candidate:

```text
CLE-HFL + FedEASE v2.1
```

Read first:

```text
FEDEASE_V2_1_FRAMEWORK_AND_IMPLEMENTATION_ZH.md
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
FEDEASE_OPENI_RUN_GUIDE_ZH.md
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
FEDCLEAR_LATEST_THEORY_FRAMEWORK_ZH.md
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
docs/rahfl_cle_alpha05_gamma09_seed0_round00_11.csv
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
FEDCLEAR_METHOD_DESIGN_REVIEW_ZH.md
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
FEDCLEAR_CLE_HFL_PROPOSAL_ZH.md
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
FEDSARA_CS_SCENARIO_OPENI_GUIDE_ZH.md
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
