# TODO Next

## Current Authoritative Next Steps - 2026-06-07

### Now: diagnose why D2C collapses toward LogitAvg

The repaired FedPRIME-D2C warmup=3 experiment completed all 40 rounds without
NaN/Inf.

```text
RAHFL final:            avg_acc=56.41, worst_acc=44.72
FedPRIME-D2C final:     avg_acc=52.31, worst_acc=39.78
FedPRIME-D2C best avg:  avg_acc=52.83 at round 37
```

Conclusion:

```text
PRIME + D2C is numerically stable and learns, but the first valid run does not
beat RAHFL. The final gaps are -4.10 avg_acc and -4.94 worst_acc.
```

The strict LogitAvg+PRIME control has completed:

```text
LogitAvg+PRIME final: avg_acc=52.10, worst_acc=39.72
FedPRIME-D2C final:   avg_acc=52.31, worst_acc=39.78
D2C gain:             avg_acc=+0.21, worst_acc=+0.06
```

This is effectively a tie. Current D2C does not yet provide a meaningful gain
over ordinary public-logit averaging.

Run the underrepresented-class diagnosis before ending the Kaggle session:

```bash
python scripts/diagnose_underrepresented.py \
  --config configs/kaggle_t4_fedprime_d2c_warmup3.yaml \
  --checkpoint_dir outputs/fedprime_d2c_cifar10c_alpha05_cr1_t4_warmup3/checkpoints
```

Then summarize and preserve:

```bash
python scripts/summarize_results.py --outputs outputs
```

Collect:

```text
outputs/fedprime_d2c_cifar10c_alpha05_cr1_t4_warmup3/metrics.csv
outputs/fedprime_d2c_cifar10c_alpha05_cr1_t4_warmup3/underrepresented_accuracy.csv
outputs/fedprime_d2c_cifar10c_alpha05_cr1_t4_warmup3/checkpoints/
outputs/summary.csv
```

### Next experiments, in priority order

1. Run a T4-safe Oracle Prior D2C experiment. This is the highest-information
   next diagnostic:

```text
Implementation completed:
  configs/kaggle_t4_fedprime_d2c_oracle_warmup3.yaml
  fedprime/engine/prior_diagnostics.py
  scripts/analyze_priors.py

Formal Kaggle run is still pending.
```

Run:

```bash
python scripts/run_experiment.py \
  --config configs/kaggle_t4_fedprime_d2c_oracle_warmup3.yaml
python scripts/analyze_priors.py \
  --experiment_dir outputs/fedprime_d2c_oracle_cifar10c_alpha05_cr1_t4_warmup3
```

```text
If Oracle Prior improves substantially:
  predicted prior from cross-domain CIFAR-100 is the primary bottleneck.

If Oracle Prior remains near 52:
  class-balanced aggregation and/or complementary KD are the bottleneck.
```
2. Inspect `tail_acc` and `missing_acc` for both D2C and LogitAvg checkpoints.
3. Predicted-vs-true prior logging is implemented. After Oracle training, export
   and analyze `prior_diagnostics.csv`, `prior_summary.json`, and prior plots.
4. Inspect whether predicted priors are nearly uniform under temperature=3
   using normalized entropy, L1/KL, cosine similarity, and heatmaps.
5. The round-3 worst-client drop also suggests
   early D2C may be too aggressive.
6. Only after Oracle Prior, test targeted D2C stabilization:

```text
longer warmup
EMA prior
self-preserving gate
smaller lambda_d2c or beta
```

7. Run a T4-safe alpha=0.1 Severe Non-IID comparison after D2C is competitive.
8. After confirming the design is promising, add a strong RAHFL comparison:

```text
40 local pretraining epochs before communication
larger/full public communication budget per round
the same strengthened training budget for FedPRIME-D2C
```

9. Warmup ablation:

```text
configs/kaggle_t4_fedprime_d2c.yaml
configs/kaggle_t4_fedprime_d2c_warmup3.yaml
```

10. Create T4-safe controlled configs for:

```text
RAHFL+PRIME = PRIME + DCL + AsymHFL
FedPRIME-D2C+DCL = PRIME + DCL + D2C
```

11. Run D2C component ablations.
12. Run seeds 0, 1, 2 only after the design is stable.
13. Evaluate official CIFAR-10-C corruption groups later.

Full experiment descriptions and configuration paths:

```text
EXPERIMENT_GUIDE_ZH.md
```

### Resume prompt

```text
读取 ARCHITECTURE.md、PROJECT_STATE.md、EXPERIMENT_GUIDE_ZH.md 和 TODO_NEXT.md，
继续推进 FedPRIME-D2C。先检查当前 Kaggle 核心对比是否完成，并分析 summary.csv
以及两个 metrics.csv。
```

## Historical Next Steps

The section below records earlier plans and may be outdated. Use the
`Current Authoritative Next Steps - 2026-06-06` section above first.

## Immediate Next Steps

0. Current continuation checkpoint.

Done:

- local data prepared
- environment check passed
- partition audit generated
- debug FedPRIME-D2C smoke run passed

Output:

```text
outputs/debug_fedprime_d2c_cifar10c/metrics.csv
outputs/partition_audit/fedprime_d2c_cifar10c_alpha05_cr1/
```

1. Check Git status.

```powershell
git status --short
```

2. Prepare local data if cloning on a new machine.

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

Use the committed debug config:

Run:

```powershell
D:\anaconda3\Scripts\conda.exe run -n pytorch python scripts\run_experiment.py --config configs\debug_fedprime_d2c_cifar10c.yaml
```

6. Run core comparison.

```powershell
D:\anaconda3\Scripts\conda.exe run -n pytorch python scripts\run_grid.py configs\cifar10c_rahfl.yaml configs\cifar10c_rahfl_prime.yaml configs\fedprime_d2c_cifar10c.yaml
```

For the stricter controlled comparison with DCL on both PRIME methods:

```powershell
D:\anaconda3\Scripts\conda.exe run -n pytorch python scripts\run_grid.py configs\cifar10c_rahfl_prime.yaml configs\fedprime_d2c_dcl_cifar10c.yaml
```

7. Summarize results.

```powershell
D:\anaconda3\Scripts\conda.exe run -n pytorch python scripts\summarize_results.py --outputs outputs
```

8. Mechanism diagnostics after checkpoints exist.

Run LogitAvg+PRIME to check whether D2C beats plain public-logit averaging:

```powershell
D:\anaconda3\Scripts\conda.exe run -n pytorch python scripts\run_experiment.py --config configs\logitavg_prime_cifar10c.yaml
```

Diagnose whether weak / underrepresented client classes improved:

```powershell
D:\anaconda3\Scripts\conda.exe run -n pytorch python scripts\diagnose_underrepresented.py --config configs\fedprime_d2c_cifar10c.yaml --checkpoint_dir outputs\fedprime_d2c_cifar10c_alpha05_cr1\checkpoints
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
