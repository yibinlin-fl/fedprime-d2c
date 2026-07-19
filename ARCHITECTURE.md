# FedPRIME-D2C Architecture

This document is the long-term code map for the FedPRIME-D2C project.

Use it together with:

```text
PROJECT_STATE.md
EXPERIMENT_GUIDE_ZH.md
TODO_NEXT.md
```

When resuming work, read these three files first.

## 2026-07-19 Architecture Override

The current candidate is CLE-HFL + FedEASE v2.1. The detailed theory and exact
implementation boundary are documented in:

```text
FEDEASE_V2_1_FRAMEWORK_AND_IMPLEMENTATION_ZH.md
```

Current code architecture:

```text
CLE Oracle or PEW environment loader
  -> AugMix/JSD/DCL robust local path
  -> BER replaces clean CE using client-local class-environment counts
  -> fixed random projection or PEW environment embedding
  -> class-conditional environment dependence penalty
  -> local class x environment x class relation accumulation
  -> server environment-balanced structural aggregation
  -> cross-environment stability gate
  -> classifier-head EBST alignment with SCP
  -> clean/same/random/swapped/unseen evaluation
```

Formal entry and staged configs:

```text
scripts/openi_fedease_entry.py
configs/openi_v100_fedease_oracle_control_probe.yaml
configs/openi_v100_fedease_oracle_ber_cdep_probe.yaml
configs/openi_v100_fedease_pew_probe.yaml
configs/openi_v100_fedease_ebst_probe.yaml
configs/openi_v100_fedease_full.yaml
```

This is a complete candidate implementation, not a validated positive result.
The first formal run is `--mode=oracle_probe`.

The older D2C, PRAC-HFL, SARA, FedCLEAR, and PCCD sections below are historical
code maps and must not override the current FedEASE memory.

## Project Goal

FedPRIME-D2C targets robust heterogeneous federated learning under:

```text
model heterogeneity
data heterogeneity / Non-IID label skew
common corruption robustness
```

The main baseline is RAHFL.

The intended paper story is:

```text
RAHFL mainly handles unreliable or corrupted collaborators by deciding which
client should teach which other client.

FedPRIME-D2C instead targets public-logit communication under Non-IID data:
client public logits can be contaminated by local label priors, so D2C debiases
client logits, builds class-aware teachers, and applies personalized
complementary KD.
```

## Repository Layout

```text
FedPRIME-D2C/
  fedprime/
    augmentations/
    data/
    methods/
    models/
    engine/
    utils/
  configs/
    ablations/
  scripts/
  PRIME-augmentations-main/
  RAHFL-master/
  outputs/
  README.md
  PROJECT_STATE.md
  TODO_NEXT.md
  ARCHITECTURE.md
  requirements.txt
```

Important external code:

```text
PRIME-augmentations-main/  official PRIME implementation
RAHFL-master/             original RAHFL implementation
```

The `fedprime/` package wraps and reuses both codebases instead of rewriting
them from scratch.

## Core Files

### Experiment Entry

```text
scripts/run_experiment.py
```

Dispatches by `method_name`:

```text
fedprime_d2c -> FedPrimeD2CExperiment
fedprime_pair -> FedPrimePairExperiment
rahfl        -> AsymHFLExperiment
rahfl_prime  -> AsymHFLExperiment
```

Run multiple configs:

```text
scripts/run_grid.py
```

Kaggle launcher:

```text
scripts/run_kaggle.sh
```

Current Kaggle default:

```text
RAHFL vs FedPRIME-D2C
no DCL in FedPRIME-D2C
T4-safe batch sizes
configs/kaggle_t4_rahfl.yaml
configs/kaggle_t4_fedprime_d2c_warmup3.yaml
FedPRIME-D2C uses three local PRIME-only warmup rounds
```

Current experiment status on 2026-06-07:

```text
RAHFL final:          avg_acc=56.41, worst_acc=44.72
FedPRIME-D2C final:   avg_acc=52.31, worst_acc=39.78
LogitAvg+PRIME final: avg_acc=52.10, worst_acc=39.72

Current D2C is effectively tied with ordinary LogitAvg. The next diagnostic is
Oracle Prior D2C to test whether cross-domain predicted-prior estimation is the
main bottleneck.
```

### Data

```text
fedprime/data/loaders.py
```

Responsibilities:

```text
load RAHFL-style CIFAR-10-C npy caches
build private client loaders
build public CIFAR-100 loader
normalize CIFAR batches
call RAHFL Dirichlet/IID partitioning
save/load fixed partition indices
build PRIME-DCL two-view loaders
build AugMix loaders for RAHFL
```

Data preparation:

```text
scripts/prepare_data.py
```

Prepared mounted-data import:

```text
scripts/import_prepared_data.py
```

This helper searches a mounted prepared-data root, including nested Kaggle input
layouts, then copies and verifies:

```text
cifar_10_c -> RAHFL-master/Dataset/cifar_10_c
cifar_100 -> RAHFL-master/Dataset/cifar_100
outputs/partitions -> outputs/partitions
```

The current Kaggle prepared dataset is named:

```text
fedprime-data
```

This downloads CIFAR-10/CIFAR-100 through torchvision and creates
RAHFL-style random-corruption caches:

```text
RAHFL-master/Dataset/cifar_10_c/
  train/random_corrupt_0.npy
  train/random_corrupt_0.5.npy
  train/random_corrupt_1.npy
  train/labels.npy
  test/random_corrupt_0.npy
  test/random_corrupt_0.5.npy
  test/random_corrupt_1.npy
  test/labels.npy
```

This is not the official CIFAR-10-C layout. Official CIFAR-10-C files are only
needed later for per-corruption group evaluation.

### Model Factory

```text
fedprime/models/factory.py
```

This calls RAHFL's `Dataset.utils.init_nets` and builds heterogeneous clients:

```text
ResNet10
ResNet12
ShuffleNet
Mobilenetv2
```

The helper `forward_logits` handles models that return either logits directly
or `(logits, features)`.

### PRIME Reuse

```text
fedprime/augmentations/prime_adapter.py
```

This imports the official PRIME primitives from `PRIME-augmentations-main` and
builds a `GeneralizedPRIMEModule`.

Local PRIME training uses three views:

```text
clean + prime_aug1 + prime_aug2
```

Local PRIME loss:

```text
CE(clean logits, labels) + lambda_jsd * JSD(clean, aug1, aug2)
```

Implementation:

```text
fedprime/methods/local_prime.py
```

### RAHFL Runner

```text
fedprime/methods/rahfl_asymhfl.py
```

Supports:

```text
method_name: rahfl
  AugMix + DCL + AsymHFL

method_name: rahfl_prime
  PRIME + DCL + AsymHFL
```

RAHFL communication:

```text
evaluate each client
on public data, weaker clients learn from clients with no worse accuracy
loss is KL(student || selected teachers)
```

This runner keeps the RAHFL idea of asymmetric heterogeneous collaboration.

### FedPRIME-D2C Runner

```text
fedprime/methods/fedprime_d2c.py
```

Main flow:

```text
1. load private labels
2. build or load fixed client partition
3. build private loaders and public CIFAR-100 loader
4. build heterogeneous client models
5. local PRIME training
6. public-data communication
7. evaluate all clients
8. write metrics.csv
9. save final checkpoints
```

Supported communication modes:

```yaml
method:
  communication: d2c       # full D2C teacher
  communication: logit_avg # plain public-logit averaging baseline
```

Supported warmup:

```yaml
method:
  d2c_warmup_rounds: 0
```

If `d2c_warmup_rounds > 0`, early rounds run local PRIME only and skip public
D2C distillation.

Optional DCL local training:

```yaml
method:
  use_dcl: true
  cl_module: dcl
```

This gives:

```text
PRIME + DCL + D2C
```

It is intended as a stricter controlled comparison, not the first default claim.

### FedPRIME-PAIR Runner

```text
fedprime/methods/fedprime_pair.py
```

FedPRIME-PAIR is the new switchable implementation for:

```text
PRIME + CBCL + CPAD
```

It is designed after the D2C diagnostics showed that public-prior debiasing is
effectively tied with LogitAvg. The new communication unit is a directed class
pair boundary `c -> j`, not a full softmax vector or a client-level route.

Core files:

```text
fedprime/methods/cpad.py          CPAD tensor logic and expertise export
fedprime/methods/local_prime.py   PRIME+CBCL local training
scripts/analyze_pair_expertise.py class-pair expertise CSV/heatmap
```

Core switches:

```yaml
method:
  use_prime: true
  use_cbcl: true
  use_cpad: true
  lambda_jsd: 12.0
  cbcl:
    lambda_cbcl: 0.2
  cpad:
    warmup_rounds: 3
    lambda_cpad: 1.0
    leave_one_out: true
```

Implemented MVP flow:

```text
1. local PRIME training with optional CBCL
2. estimate directed client-class-pair expertise E_{k,c->j}
3. build leave-one-client-out class-pair teachers on public logits
4. optimize CPAD PairBCE on public data
5. write metrics, checkpoints, and pair-expertise snapshots
```

Code-level round flow:

```text
scripts/run_experiment.py
  method_name: fedprime_pair
    -> FedPrimePairExperiment.run()

FedPrimePairExperiment.run()
  1. load labels from RAHFL-style CIFAR-10-C caches
  2. load or create the fixed Dirichlet partition
  3. build private loaders and CIFAR-100 public loader
  4. build heterogeneous RAHFL models
  5. build the official PRIME module through prime_adapter.py
  6. for each round:
       a. _local_phase()
       b. if round >= cpad.warmup_rounds: _estimate_all_pair_expertise()
       c. if round >= cpad.warmup_rounds: _cpad_phase()
       d. _evaluate()
       e. append metrics.csv
  7. save final client checkpoints
```

CBCL implementation:

```text
fedprime/methods/local_prime.py
  train_local_prime_cbcl_epoch()
    views = prime_aug(images)
    output = model(views)
    if output is (logits, embedding):
      reuse both logits and embedding from the same forward pass
    loss = CE(clean/aug views) + lambda_jsd * JSD + lambda_cbcl * CBCL
```

CBCL loss:

```text
_class_balanced_cbcl_loss()
  - normalizes embeddings from clean + PRIME views
  - builds supervised positives by class label
  - gives each anchor a reliability weight based on true-class margin
  - averages loss per local class first, then averages classes
```

CPAD implementation:

```text
fedprime/methods/cpad.py
  normalize_logits()
  pair_margins()
  estimate_pair_expertise()
  cpad_pair_bce_loss()
```

Pair expertise:

```text
For a local labeled sample with class c:
  margin(c -> j) = normalized_logit_c - normalized_logit_j

Across clean/PRIME views:
  robust_margin = softmin(view margins)

Client expertise:
  E_raw[k,c,j] = mean sigmoid(robust_margin(c -> j) / expertise_tau)
  E_weighted[k,c,j] = E_raw[k,c,j] * support(count_k,c)
```

CPAD public distillation:

```text
public_logits_all: [K, B, C]
pair margins:      [K, B, C, C]

For every student client i:
  optionally remove client i from the teacher pool (leave-one-out)
  aggregate teacher pair margins by E_weighted[k,c,j]
  convert teacher/student margins to pair probabilities with sigmoid
  optimize pairwise BCE, weighted by:
    - gate: learn boundary only when global experts are stronger
    - confidence: ignore teacher boundaries near 0.5
    - agreement: downweight high-disagreement boundaries
```

Runtime logging:

```text
configs/kaggle_t4_fedprime_pair_full.yaml
  train.progress_every_batches: 50

FedPRIME-PAIR prints heartbeat logs for:
  round start
  each local client start/done
  every N local batches
  pair expertise estimation
  CPAD public batches
  evaluation and final round metrics
```

Configs:

```text
configs/debug_fedprime_pair_cifar10c.yaml
configs/kaggle_t4_fedprime_pair_full.yaml
```

## D2C Module

```text
fedprime/methods/d2c.py
```

Input public logits:

```text
logits_all shape = [K, B, C]
K = number of clients
B = public batch size
C = number of classes
```

For default configs:

```text
K = 4
B = 256
C = 10
```

Client softened public prediction:

```text
p_k(y|x) = softmax(z_k(x) / T)
```

Predictive prior:

```text
pi_k(y) = mean_x p_k(y|x)
```

Prior debias:

```text
z'_k(y|x) = z_k(y|x) - beta_k * log(pi_k(y) + eps)
```

Adaptive beta:

```text
beta_k = beta * (1 - H(pi_k) / log(C))
```

Class-balanced aggregation:

```text
a_k,c = pi_k(c)^eta / sum_j pi_j(c)^eta
```

Sample confidence:

```text
conf_k(x) = 1 - H(p'_k(.|x)) / log(C)
```

D2C teacher:

```text
q(y|x) = normalize_y sum_k a_k,y * conf_k(x) * p'_k(y|x)
```

Complementary KD:

```text
m_k(c) = (1 - pi_k(c))^rho

L_k = T^2 * mean_x sum_c m_k(c) * q(c|x)
      * [log q(c|x) - log p_k(c|x)]
```

Core switches:

```yaml
d2c:
  adaptive_beta: false
  ema_alpha:
  use_prior_debias: true
  use_class_balanced: true
  use_sample_confidence: true

method:
  use_self_gate: false
  use_complementary_kd: true
  prior_source: predicted # or oracle
```

### Oracle Prior Diagnostic

Oracle Prior is a diagnostic upper bound, not a deployable privacy-preserving
method. It replaces the predicted public-data prior with the true class
histogram of each client's fixed private partition.

The compatibility entry point remains:

```text
D2CServer.build_teacher() -> teacher, used_prior
```

Optional diagnostics use:

```text
D2CServer.build_teacher_with_diagnostics()
```

This keeps the original predicted-prior training path unchanged. Regression
tests verify its teacher and prior are exactly equal to the legacy formula.

T4-safe formal diagnostic:

```text
configs/kaggle_t4_fedprime_d2c_oracle_warmup3.yaml
```

It is identical to `configs/kaggle_t4_fedprime_d2c_warmup3.yaml` except for the
experiment name, `prior_source: oracle`, and diagnostic logging.

Optional prior logging:

```yaml
method:
  prior_diagnostics:
    enabled: true
    save_rounds: [3, 10, 20, 39]
```

Implementation:

```text
fedprime/engine/prior_diagnostics.py
scripts/analyze_priors.py
```

Outputs:

```text
outputs/<experiment>/
  prior_diagnostics.csv
  prior_summary.json
  priors/round_*.npz
  prior_analysis/
    prior_metrics_by_round.csv
    prior_error_by_round.png
    prior_heatmap_oracle.png
    prior_heatmap_predicted.png
    prior_heatmap_absolute_error.png
```

The CSV records every client/public-batch comparison, including complete
predicted, oracle, and actually-used prior vectors plus L1, KL, cosine,
entropy, and top-class-match metrics.

## Fixed Partition Fairness

All main comparison configs share fixed partition files through:

```yaml
data:
  partition_indices_path: outputs/partitions/cifar10c_alpha05_seed0_clients4_samples10000.npz
```

Purpose:

```text
RAHFL, RAHFL+PRIME, FedPRIME-D2C, and ablations use the same client data split.
```

This avoids reviewer criticism that methods saw different Non-IID partitions.

Audit script:

```text
scripts/audit_partition.py
```

Outputs:

```text
outputs/partition_audit/<experiment_name>/
  client_class_counts.csv
  client_class_proportions.csv
  client_class_counts.png
  partition_summary.json
```

## Experiment Config Matrix

### Main Comparison

```text
configs/cifar10c_rahfl.yaml
  RAHFL = AugMix + DCL + AsymHFL

configs/fedprime_d2c_cifar10c.yaml
  FedPRIME-D2C = PRIME + D2C
```

This is the urgent first comparison.

### Strong Controlled Comparison

```text
configs/cifar10c_rahfl_prime.yaml
  RAHFL+PRIME = PRIME + DCL + AsymHFL

configs/fedprime_d2c_dcl_cifar10c.yaml
  FedPRIME-D2C+DCL = PRIME + DCL + D2C
```

This isolates:

```text
AsymHFL vs D2C
```

### Severe Non-IID

```text
configs/fedprime_d2c_cifar10c_alpha01.yaml
configs/fedprime_d2c_dcl_cifar10c_alpha01.yaml
configs/logitavg_prime_cifar10c_alpha01.yaml
```

Alpha `0.1` is important because D2C should help more when label skew is
stronger.

### LogitAvg Baseline

```text
configs/logitavg_prime_cifar10c.yaml
configs/logitavg_prime_cifar10c_alpha01.yaml
```

Purpose:

```text
Check whether D2C beats plain public-logit averaging.
```

### Debug Configs

```text
configs/debug_fedprime_d2c_cifar10c.yaml
configs/debug_fedprime_d2c_dcl_cifar10c.yaml
configs/debug_logitavg_prime_cifar10c.yaml
```

These use tiny data and one round to verify that code paths run.

### Ablations

```text
configs/ablations/fedprime_d2c_no_prime.yaml
configs/ablations/fedprime_d2c_no_prior_debias.yaml
configs/ablations/fedprime_d2c_no_class_balanced.yaml
configs/ablations/fedprime_d2c_no_complementary_kd.yaml
configs/ablations/fedprime_d2c_oracle_prior.yaml
configs/ablations/fedprime_d2c_adaptive_ema_gate.yaml
```

Purpose:

```text
identify which D2C components actually contribute
```

## Metrics

Every main run writes:

```text
outputs/<experiment_name>/metrics.csv
```

Main columns:

```text
round
avg_acc
worst_acc
local_loss
d2c_loss or col_loss
```

Important interpretation:

```text
avg_acc   overall client performance
worst_acc weakest-client performance, very important for Non-IID
```

For RAHFL:

```text
col_loss = AsymHFL collaboration loss
```

For FedPRIME-D2C:

```text
d2c_loss = D2C or LogitAvg public KD loss
```

Summary script:

```text
scripts/summarize_results.py
```

Writes:

```text
outputs/summary.csv
outputs/summary.md
```

## Diagnostics

### Underrepresented Class Accuracy

```text
scripts/diagnose_underrepresented.py
```

Usage:

```bash
python scripts/diagnose_underrepresented.py \
  --config configs/fedprime_d2c_cifar10c.yaml \
  --checkpoint_dir outputs/fedprime_d2c_cifar10c_alpha05_cr1/checkpoints
```

Output columns:

```text
overall_acc
head_acc
tail_acc
missing_acc
head_classes
tail_classes
missing_classes
private_class_counts
```

Purpose:

```text
Check whether D2C helps classes that are rare or missing in a client's private data.
```

### Corruption Group Evaluation

```text
scripts/evaluate_corruptions.py
```

Requires official-style CIFAR-10-C per-corruption `.npy` files.

Purpose:

```text
evaluate noise / blur / weather / digital corruption groups
```

## Kaggle Commands

Current FedPRIME-PAIR runs should be launched from a **Python streaming
launcher cell** in Kaggle, not from a long `%%bash` cell. The streaming launcher
should clone/pull the repo, verify `git log -1 --oneline` is `8a4ee15` or
later, then call:

```bash
RUN_DEBUG=1 PYTHONUNBUFFERED=1 bash scripts/run_kaggle_pair.sh
```

Reason:

```text
Kaggle may buffer %%bash stdout until the command exits.
The Python launcher streams subprocess output line by line and prints a driver
heartbeat every 60 seconds.
Do not call sys.stdout.reconfigure in Kaggle notebooks.
```

The older commands below are kept for the historical FedPRIME-D2C comparison
route.

Default urgent comparison:

```bash
git clone https://github.com/yibinlin-fl/fedprime-d2c.git
cd fedprime-d2c
RUN_DEBUG=1 bash scripts/run_kaggle.sh
```

Default script runs:

```text
debug FedPRIME-D2C
RAHFL vs FedPRIME-D2C with T4-safe configs
summary
```

No manual CIFAR-10-C upload is needed for the first comparison. The script
generates RAHFL-style random corruption caches.

If data has already been downloaded in the active Kaggle session, rerun only the
training stage:

```bash
git pull
RUN_INSTALL=0 RUN_PREPARE_DATA=0 RUN_DEBUG=0 bash scripts/run_kaggle.sh
```

Full four-method comparison:

```bash
bash scripts/run_kaggle.sh \
  configs/cifar10c_rahfl.yaml \
  configs/cifar10c_rahfl_prime.yaml \
  configs/fedprime_d2c_cifar10c.yaml \
  configs/fedprime_d2c_dcl_cifar10c.yaml
```

Strict DCL-controlled comparison:

```bash
bash scripts/run_kaggle.sh \
  configs/cifar10c_rahfl_prime.yaml \
  configs/fedprime_d2c_dcl_cifar10c.yaml
```

## Quick Result Judgment

## 2026-06-29 Active Architecture: PRAC-HFL

The active architecture is:

```text
PRAC-HFL = RAHFL local robust training + receiver-adaptive safe communication
```

RAHFL local robust training is reused as the strong base:

```text
AugMix multi-view loader
CE classification loss
JSD consistency loss
DCLLoss from RAHFL-master/loss.py
```

PRAC-HFL changes only the communication phase. Instead of RAHFL AsymHFL selecting teachers by global performance, each receiver client privately tests whether a teacher helps its own route batch.

Communication phase:

```text
public logits -> candidate teacher set -> head-only virtual KD -> private route CE risk delta
-> classwise positive-effect weights -> mixed teacher -> accept-batch safety gate
```

Important implementation detail:

```text
PRAC-HFL currently updates only the classifier head during virtual teacher testing and accepted mixed-teacher KD.
This is intentional for stability and speed, and targets the Non-IID classifier-boundary bias.
```

Main files:

```text
fedprime/methods/prac_hfl.py
fedprime/methods/local_rahfl.py
configs/kaggle_t4_prac_hfl.yaml
scripts/run_kaggle_prac.sh
```

Historical modules:

```text
fedprime/methods/fedprime_d2c.py      # historical diagnostic, not active mainline
fedprime/methods/fedprime_pair.py     # historical diagnostic, not active mainline
fedprime/methods/cpad.py              # historical diagnostic, not active mainline
```

Current evidence:

```text
RAHFL unified runner final: avg_acc=56.41, worst_acc=44.72.
First unstable PRAC-HFL run reached round 028:
  PRAC avg_acc=53.86 vs same-round RAHFL avg_acc=53.21.
  PRAC worst_acc=39.52 vs same-round RAHFL worst_acc=41.64.
Round 029 produced NaN; safe PRAC-HFL config was introduced and must be rerun.
```

For the first short run, do not judge only by round-5 absolute accuracy.

Promising signs:

```text
FedPRIME-D2C avg_acc rises
FedPRIME-D2C worst_acc does not collapse
gap to RAHFL is small or shrinking
d2c_loss is finite and stable
local_loss is finite and stable
```

Warning signs:

```text
RAHFL rises but FedPRIME-D2C stays near random
FedPRIME-D2C is more than 8-10 points behind and gap widens
worst_acc is much worse than RAHFL
d2c_loss or local_loss becomes nan
```

Strong paper pattern:

```text
alpha=0.5: FedPRIME-D2C close to or better than RAHFL
alpha=0.1: FedPRIME-D2C advantage becomes larger
```

That supports:

```text
D2C is especially useful when Non-IID label skew is severe.
```

## Checkpoint Behavior

Current runners save only final client checkpoints:

```text
outputs/<experiment_name>/checkpoints/client_0.pt
outputs/<experiment_name>/checkpoints/client_1.pt
outputs/<experiment_name>/checkpoints/client_2.pt
outputs/<experiment_name>/checkpoints/client_3.pt
```

They do not save every round, so disk usage is controlled.

## GitHub Remote

Remote repository:

```text
git@github.com:yibinlin-fl/fedprime-d2c.git
```

Branch:

```text
main
```

The local repository is configured to push over SSH.

## Important Cautions

1. Do not claim D2C always beats RAHFL before full experiments.
2. RAHFL is strong because it includes AugMix + DCL + AsymHFL.
3. FedPRIME-D2C without DCL is the main-module claim.
4. FedPRIME-D2C + DCL is a stricter controlled comparison, not the first default claim.
5. LogitAvg+PRIME is important to prove D2C is better than plain public-logit averaging.
6. Official CIFAR-10-C is still needed for detailed per-corruption group analysis.
7. Use fixed partition files for fair method comparisons.
