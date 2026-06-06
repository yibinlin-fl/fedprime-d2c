# FedPRIME-D2C Architecture

This document is the long-term code map for the FedPRIME-D2C project.

Use it together with:

```text
PROJECT_STATE.md
EXPERIMENT_GUIDE_ZH.md
TODO_NEXT.md
```

When resuming work, read these three files first.

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

Current experiment status on 2026-06-06:

```text
The first complete Kaggle RAHFL vs FedPRIME-D2C warmup=3 comparison is running.
RAHFL training has been observed through round 5 and is healthy.
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
