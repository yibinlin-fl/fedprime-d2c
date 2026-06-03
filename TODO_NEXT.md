# TODO Next

## Immediate Next Steps

1. Check Git status.

```powershell
git status --short
```

2. Prepare local data.

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

Recommended temporary edits or create a debug config:

```yaml
train:
  rounds: 1
  local_epochs: 1
  max_local_batches: 1
  public_batches_per_round: 1
```

Run:

```powershell
D:\anaconda3\Scripts\conda.exe run -n pytorch python scripts\run_experiment.py --config configs/fedprime_d2c_cifar10c.yaml
```

6. If smoke run passes, run core comparison.

```powershell
D:\anaconda3\Scripts\conda.exe run -n pytorch python scripts\run_grid.py configs\cifar10c_rahfl.yaml configs\cifar10c_rahfl_prime.yaml configs\fedprime_d2c_cifar10c.yaml
```

7. Summarize results.

```powershell
D:\anaconda3\Scripts\conda.exe run -n pytorch python scripts\summarize_results.py --outputs outputs
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

Tell Codex:

```text
读取 PROJECT_STATE.md 和 TODO_NEXT.md，继续推进 FedPRIME-D2C 项目。先检查 git 状态，然后准备数据和跑一个 debug smoke training。
```

