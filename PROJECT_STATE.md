# FedPRIME-D2C Project State

Last updated: 2026-06-06

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

All four clients completed a full local PRIME epoch without NaN/Inf. The
Kaggle warmup config now also sets:

```yaml
train:
  max_grad_norm: 5.0
```

The detailed Chinese experiment/configuration and metric guide is:

```text
EXPERIMENT_GUIDE_ZH.md
```

Current action:

```text
Push the numerical-stability fix.
Rerun only configs/kaggle_t4_fedprime_d2c_warmup3.yaml on Kaggle.
Do not rerun RAHFL; its completed 40-round result is already the baseline.
```

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
