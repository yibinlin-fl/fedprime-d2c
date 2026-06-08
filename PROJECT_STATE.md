# FedPRIME-D2C Project State

Last updated: 2026-06-07

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
`EXPERIMENT_GUIDE_ZH.md` for the complete comparison.

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
EXPERIMENT_GUIDE_ZH.md
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

## Resume Update - 2026-06-05

Completed since the previous state update:

- Added the Chinese experiment/configuration guide:

```text
EXPERIMENT_GUIDE_ZH.md
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

Use:

```text
读取 PROJECT_STATE.md 和 TODO_NEXT.md，继续推进 FedPRIME-D2C 项目。
```
