# FedPRIME-D2C

Distribution-Decoupled Communication for PRIME-based Robust Heterogeneous Federated Learning.

This repository currently contains:

- `PRIME-augmentations-main/`: original PRIME implementation.
- `RAHFL-master/`: original RAHFL implementation.
- `fedprime/`: configuration-driven FedPRIME-D2C MVP code.
- `configs/`: experiment configs for Kaggle/server execution.
- `docs/experiment_plan.md`: full experiment plan.

## Install

```bash
pip install -r requirements.txt
```

Install a PyTorch build matching your CUDA version if needed.

## Data Layout

The default configs expect:

```text
RAHFL-master/Dataset/cifar_10_c/
  train/random_corrupt_1.npy
  train/labels.npy
  test/random_corrupt_1.npy
  test/labels.npy

RAHFL-master/Dataset/cifar_100/
```

If running on Kaggle or a school server, change paths in `configs/*.yaml`.

## Prepare Data

Prepare CIFAR-100 public data and RAHFL-style random-corrupted CIFAR-10 files:

```bash
python scripts/prepare_data.py \
  --config configs/fedprime_d2c_cifar10c.yaml \
  --download \
  --rates 0 0.5 1
```

This creates:

```text
RAHFL-master/Dataset/cifar_10_c/train/random_corrupt_1.npy
RAHFL-master/Dataset/cifar_10_c/test/random_corrupt_1.npy
RAHFL-master/Dataset/cifar_10_c/*/labels.npy
RAHFL-master/Dataset/cifar_100/cifar-100-python/
```

Audit the Dirichlet client split:

```bash
python scripts/audit_partition.py --config configs/fedprime_d2c_cifar10c.yaml
```

## Run FedPRIME-D2C MVP

Check dependencies and paths first:

```bash
python scripts/check_environment.py --config configs/fedprime_d2c_cifar10c.yaml
```

```bash
python scripts/run_experiment.py --config configs/fedprime_d2c_cifar10c.yaml
```

Severe Non-IID:

```bash
python scripts/run_experiment.py --config configs/fedprime_d2c_cifar10c_alpha01.yaml
```

Outputs are written to:

```text
outputs/<experiment_name>/
  config.resolved.json
  metrics.csv
  checkpoints/
```

Run multiple configs:

```bash
python scripts/run_grid.py \
  configs/fedprime_d2c_cifar10c.yaml \
  configs/fedprime_d2c_cifar10c_alpha01.yaml
```

Run the plain public-logit averaging baseline:

```bash
python scripts/run_experiment.py --config configs/logitavg_prime_cifar10c.yaml
python scripts/run_experiment.py --config configs/logitavg_prime_cifar10c_alpha01.yaml
```

Run multiple seeds:

```bash
python scripts/run_multiseed.py \
  --config configs/fedprime_d2c_cifar10c.yaml \
  --seeds 0 1 2
```

Summarize finished runs:

```bash
python scripts/summarize_results.py --outputs outputs
```

Evaluate corruption groups after a run:

```bash
python scripts/evaluate_corruptions.py \
  --config configs/fedprime_d2c_cifar10c.yaml \
  --checkpoint_dir outputs/fedprime_d2c_cifar10c_alpha05_cr1/checkpoints \
  --corruption_root /path/to/CIFAR-10-C \
  --out_csv outputs/fedprime_d2c_cifar10c_alpha05_cr1/corruption_eval.csv
```

Diagnose whether D2C helps underrepresented client classes:

```bash
python scripts/diagnose_underrepresented.py \
  --config configs/fedprime_d2c_cifar10c.yaml \
  --checkpoint_dir outputs/fedprime_d2c_cifar10c_alpha05_cr1/checkpoints
```

The output CSV reports each client's `head_acc`, `tail_acc`, and `missing_acc`
according to that client's private label distribution.

Load pre-trained client checkpoints:

```yaml
checkpoints:
  load_dir: /path/to/client_checkpoints
  resume: false
  resume_dir:
  strict: true
```

Resume from a previous run checkpoint directory:

```yaml
checkpoints:
  load_dir:
  resume: true
  resume_dir: outputs/fedprime_d2c_cifar10c_alpha05_cr1/checkpoints
  strict: true
```

## Run Core Comparison

```bash
bash scripts/run_core_comparison.sh
```

Equivalent Python command:

```bash
python scripts/run_grid.py \
  configs/cifar10c_rahfl.yaml \
  configs/cifar10c_rahfl_prime.yaml \
  configs/fedprime_d2c_cifar10c.yaml \
  configs/fedprime_d2c_dcl_cifar10c.yaml
```

## Run Ablations

```bash
python scripts/run_grid.py \
  configs/ablations/fedprime_d2c_no_prime.yaml \
  configs/ablations/fedprime_d2c_no_prior_debias.yaml \
  configs/ablations/fedprime_d2c_no_class_balanced.yaml \
  configs/ablations/fedprime_d2c_no_complementary_kd.yaml \
  configs/ablations/fedprime_d2c_oracle_prior.yaml \
  configs/ablations/fedprime_d2c_adaptive_ema_gate.yaml
```

## PRIME Reuse

FedPRIME-D2C does not rewrite PRIME. It imports the official code in `PRIME-augmentations-main` through thin adapters:

- `fedprime/augmentations/prime.py`
- `fedprime/augmentations/rand_filter.py`
- `fedprime/augmentations/diffeomorphism.py`
- `fedprime/augmentations/color_jitter.py`
- `fedprime/augmentations/prime_adapter.py`

`prime_adapter.py` builds the official `GeneralizedPRIMEModule` with CIFAR normalization and returns the three-view tensor used by local CE+JSD training.

## Kaggle / Server

On Kaggle, enable:

```text
Accelerator: GPU
Internet: On
```

If the repository is public:

```bash
git clone https://github.com/yibinlin-fl/fedprime-d2c.git
cd fedprime-d2c
bash scripts/run_kaggle.sh
```

By default this script runs the urgent head-to-head comparison:

```text
RAHFL = AugMix + DCL + AsymHFL
FedPRIME-D2C = PRIME + D2C
```

The Kaggle default uses T4-safe configs:

```text
configs/kaggle_t4_rahfl.yaml
configs/kaggle_t4_fedprime_d2c_warmup3.yaml
```

These keep the same shared Non-IID split and method settings as the core
comparison, but reduce batch sizes so the RAHFL AugMix+DCL multi-view training
does not run out of memory on a single 14 GB T4. The FedPRIME-D2C config uses
three local PRIME-only warmup rounds before D2C communication begins.

It also installs dependencies, downloads CIFAR-10/CIFAR-100 through torchvision,
creates the RAHFL-style random corrupted CIFAR-10 caches, audits the shared
Non-IID partition, runs the experiments, and writes `outputs/summary.csv`.

No manual CIFAR-10-C upload is required for this first comparison. The generated
data follows the RAHFL-style cached format:

```text
RAHFL-master/Dataset/cifar_10_c/train/random_corrupt_1.npy
RAHFL-master/Dataset/cifar_10_c/test/random_corrupt_1.npy
```

Official CIFAR-10-C files are only needed later for per-corruption group
evaluation.

To include smoke tests before full training:

```bash
RUN_DEBUG=1 bash scripts/run_kaggle.sh
```

This first runs `configs/debug_fedprime_d2c_cifar10c.yaml`, then runs the
default RAHFL vs FedPRIME-D2C comparison.

To run a custom config list:

```bash
bash scripts/run_kaggle.sh \
  configs/kaggle_t4_rahfl.yaml \
  configs/kaggle_t4_fedprime_d2c_warmup3.yaml
```

If the Kaggle session already downloaded data and only the training config needs
to be rerun:

```bash
git pull
RUN_INSTALL=0 RUN_PREPARE_DATA=0 RUN_DEBUG=0 bash scripts/run_kaggle.sh
```

To run the stricter later-stage controlled comparison where both PRIME methods
use DCL:

```bash
bash scripts/run_kaggle.sh \
  configs/cifar10c_rahfl_prime.yaml \
  configs/fedprime_d2c_dcl_cifar10c.yaml
```

## Generic Server / C2NET

For a normal Linux GPU server or C2NET-style task container, use:

```bash
git clone https://github.com/yibinlin-fl/fedprime-d2c.git
cd fedprime-d2c
RUN_DEBUG=1 bash scripts/run_server.sh
```

The server launcher defaults to:

```text
configs/server_safe_rahfl.yaml
configs/server_safe_fedprime_d2c.yaml
```

These configs use the same shared Non-IID split as the Kaggle-safe configs and
keep `batch_size=64`, which is suitable for V100/A10/T4-class GPUs.

If the data has already been downloaded in the same task container:

```bash
RUN_INSTALL=0 RUN_PREPARE_DATA=0 RUN_DEBUG=0 bash scripts/run_server.sh
```

On C2NET, if the platform expects outputs to be uploaded through its helper:

```bash
UPLOAD_C2NET=1 RUN_INSTALL=0 RUN_PREPARE_DATA=0 RUN_DEBUG=0 bash scripts/run_server.sh
```

With `UPLOAD_C2NET=1`, the launcher copies local `outputs/` into:

```text
<c2net output_path>/fedprime_outputs/
```

and then calls `upload_output()`, so completed training-task results survive
container teardown.

For a quick environment-only check:

```bash
RUN_TRAIN=0 RUN_SUMMARY=0 RUN_DEBUG=1 bash scripts/run_server.sh
```

## Recommended Comparison Order

1. Run original RAHFL from `RAHFL-master` as the first baseline.
2. Run `FedPRIME-D2C` with `alpha=0.5` and `alpha=0.1`.
3. Add `RAHFL + PRIME` as the strong baseline.
4. Run D2C ablations.

The key paper claim should be evaluated under model heterogeneity, Non-IID label skew, and corrupted private data.
