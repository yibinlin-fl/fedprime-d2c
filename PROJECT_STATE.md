# FedPRIME-D2C Project State

Last updated: 2026-06-04

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
