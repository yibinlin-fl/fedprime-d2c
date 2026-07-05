# TODO Next

## 2026-07-02 Immediate Mainline: SARA + AsymHFL

Current best result:

```text
SARA + AsymHFL
config: configs/kaggle_t4_sara_rahfl.yaml
archive: outputs/sara_rahfl_results.tar.gz
alpha=0.5, seed=0, corrupt_rate=1, rounds=40
final avg_acc   = 57.83
final worst_acc = 46.59
```

Comparison:

```text
RAHFL baseline:            56.41 / 44.72
AugMix+DCL local-only:     56.11 / 44.23
FedCARA v1:                55.88 / 45.93
SARA local-only:           54.10 / 32.06
SARA + AsymHFL:            57.83 / 46.59
```

Interpretation:

```text
SARA local-only is not strong. It hurts weak-client performance.
SARA + AsymHFL is strong and currently beats RAHFL on both final average and
final worst-client accuracy.

Do not replace AsymHFL yet. The next priority is to verify whether this gain is
stable across seeds and Non-IID strengths.
```

Next experiments:

```text
1. Run SARA + AsymHFL with seed=1 and seed=2 at alpha=0.5.
2. Run SARA + AsymHFL at alpha=0.3 and alpha=0.1.
3. Run SARA + AsymHFL at alpha=1.0 to check non-extreme Non-IID.
4. If SARA remains positive, rerun RAHFL on matching seeds/settings.
5. Then decide whether a communication module replacement is necessary.
```

Matching RAHFL seed=1/2 controls are now prepared:

```text
configs/kaggle_t4_rahfl_seed1.yaml
configs/kaggle_t4_rahfl_seed2.yaml
scripts/run_kaggle_rahfl_seed12.sh
```

Recommended order:

```text
1. Let current SARA seed=1 finish.
2. Run SARA seed=2 if seed=1 is positive.
3. Then run RAHFL seed=1/2 with scripts/run_kaggle_rahfl_seed12.sh.
4. Report seed-matched mean/std, not seed0-only conclusions.
```

Prepared alpha partition pack:

```text
local_runs/sara_partitions_alpha01_alpha03
local_runs/sara_partitions_alpha01_alpha03.tar.gz

Contains:
  alpha=0.1 seeds 0/1/2
  alpha=0.3 seeds 0/1/2

Suggested Kaggle dataset name:
  sara-partitions-alpha01-alpha03

local_runs/sara_partitions_alpha03_alpha10
local_runs/sara_partitions_alpha03_alpha10.tar.gz

Contains:
  alpha=0.3 seeds 0/1/2
  alpha=1.0 seeds 0/1/2

Suggested Kaggle dataset name:
  sara-partitions-alpha03-alpha10
```

When running alpha=0.3/1.0 on Kaggle, mount both:

```text
/kaggle/input/fedprime-data
/kaggle/input/sara-partitions-alpha03-alpha10
```

When running alpha=0.3/0.1 on Kaggle, mount both:

```text
/kaggle/input/fedprime-data
/kaggle/input/sara-partitions-alpha01-alpha03
```

## 当前工作方向 - 2026-06-30

### 总方向

当前项目不要再定位成“简单替换 RAHFL 的通信模块”。

更合适的研究方向是：

```text
面向数据损坏 + 模型异构 + 数据异构的 Non-IID-aware 鲁棒异构联邦学习框架
```

核心问题不是“把 RAHFL 的粗粒度通信改成细粒度通信”这么简单，而是：

```text
RAHFL 主要解决了模型异构和数据损坏，
但在 label-skew Non-IID 场景下，
它的本地 DCL 和客户端级 AsymHFL 通信都没有充分考虑类别分布偏斜。
```

因此后续工作应围绕两条主线展开：

```text
1. Non-IID-aware robust local representation learning
   改进 AugMix + DCL，使其在类别不均衡、tail class、weak client 下更稳。

2. Receiver-safe heterogeneous communication
   改进 public-logit 通信，使客户端只吸收对自己本地分布真正有益的外部知识。
```

论文动机应表述为：

```text
RAHFL 使用强本地鲁棒增强和客户端级非对称通信，
但整体客户端准确率在 label-skew Non-IID 下不是可靠的知识可迁移性指标。
高准确率客户端可能只擅长自己的 head classes，
并不一定能为其他客户端的 tail/missing classes 提供可靠知识。

因此，我们研究如何在数据损坏和模型异构同时存在时，
进行 Non-IID-aware 的鲁棒表征学习和安全知识迁移。
```

### 当前实验事实

已有结果：

```text
RAHFL public4:
  final avg_acc   = 56.41
  final worst_acc = 44.72

PRAC-HFL public1:
  final avg_acc   = 54.63
  final worst_acc = 41.88
  best avg_acc    = 55.53

PRAC-HFL public4:
  final avg_acc   = 52.96
  final worst_acc = 43.27

AugMix + DCL local-only:
  final avg_acc   = 56.11
  final worst_acc = 44.23
  best avg_acc    = 56.94 at round 38
  best worst_acc  = 44.23 at round 39
```

当前结论：

```text
local-only 最终 avg_acc=56.11，几乎追平 RAHFL final 56.41；
local-only best avg_acc=56.94，已经超过 RAHFL final 56.41；
local-only final 同时高于 PRAC public1 和 PRAC public4。

这说明当前性能主要来自 RAHFL-style AugMix + DCL 本地鲁棒学习，
而当前 PRAC 通信没有提供稳定正增益，甚至可能引入负迁移。
后续不应继续盲目微调 PRAC 超参数。
```

### 立即工作重点

当前第一优先级不再是继续修 PRAC 通信，而是设计：

```text
Non-IID-aware robust DCL / local representation learning
```

可优先考虑：

```text
1. class-balanced DCL
   防止 head class 在 DCL 中支配特征空间。

2. tail-aware supervised contrastive learning
   提升少样本类和 weak client 的表征质量。

3. corruption-view reliability weighting
   对过强、语义可能被破坏的增强视图降低对比拉近强度。

4. client-adaptive contrastive loss strength
   根据客户端类别偏斜程度调整 DCL 权重。

5. communication as secondary
   只有在本地 Non-IID-aware DCL 稳定后，再考虑轻量安全通信。
```

PRAC 可以作为历史探索保留，但不能作为当前论文主线。

### 已实现的新版本 - NIR-DCL

2026-07-01 已实现：

```text
NIR-DCL = Non-IID-aware Robust DCL
```

实现文件：

```text
fedprime/methods/nir_dcl.py
fedprime/methods/local_rahfl.py
fedprime/methods/prac_hfl.py
fedprime/methods/rahfl_asymhfl.py
```

配置入口：

```text
configs/debug_nir_dcl_local_only.yaml
configs/kaggle_t4_nir_dcl_local_only.yaml
configs/kaggle_t4_nir_dcl_rahfl.yaml
```

当前最应该先跑：

```text
configs/kaggle_t4_nir_dcl_local_only.yaml
```

原因：

```text
先验证只改本地 DCL 是否能超过 AugMix+DCL local-only 和 RAHFL。
如果 NIR-DCL local-only 没有提升，就不要急着接通信。
如果 NIR-DCL local-only 已经明显提升，再跑 NIR-DCL + AsymHFL。
```

### CARA-L / NIR-DCL 首轮结果

已完成：

```text
NIR-DCL local-only:
  final avg_acc   = 53.30
  final worst_acc = 36.01
  best avg_acc    = 54.74

NIR-DCL + AsymHFL:
  final avg_acc   = 57.36
  final worst_acc = 46.23
  best avg_acc    = 57.89

RAHFL baseline:
  final avg_acc   = 56.41
  final worst_acc = 44.72
```

结论：

```text
NIR-DCL local-only 不成立，明显弱于 AugMix+DCL local-only。
但 NIR-DCL + AsymHFL 超过 RAHFL，说明 NIR-DCL 可能不是单独提升本地性能，
而是让本地表征更适合 AsymHFL public-logit 通信。
```

下一步不要马上大规模消融。更合理的下一步：

```text
1. 先复跑 seed=1 或 alpha=0.3 中的一个，确认这个提升不是 seed-0 偶然。
2. 补 tail_acc / per-client / per-class 指标，确认 worst_acc 提升来自哪里。
3. 如果还有算力，再跑 alpha=1.0，确认正常 Non-IID 不掉。
```

### 已实现 FedCARA / CARA-C

2026-07-01 新增：

```text
FedCARA = AugMix + CARA-L + CARA-C
```

其中：

```text
CARA-L: 原 NIR-DCL 的正式命名，负责类别自适应鲁棒本地对齐。
CARA-C: 新通信模块，负责类别自适应可靠教师蒸馏。
```

CARA-C v1 公式：

```text
w_{i,j,c} = acc_{j,c} * (1 - acc_{i,c})
```

默认还加一个安全门：

```text
only use class c if acc_{j,c} > acc_{i,c}
```

然后在 public logits 上做 class-weighted KL：

```text
L = sum_c w_{i,j,c} * p_j,c * log(p_j,c / p_i,c)
```

配置入口：

```text
configs/debug_fedcara_cifar10c.yaml
configs/kaggle_t4_fedcara.yaml
```

下一步先跑：

```text
configs/kaggle_t4_fedcara.yaml
```

比较：

```text
RAHFL baseline:   56.41 / 44.72
CARA-L+AsymHFL:   57.36 / 46.23
FedCARA v1:       55.88 / 45.93
```

结论：

```text
FedCARA v1 final avg_acc 没超过 RAHFL，但 worst_acc 超过 RAHFL。
说明当前 CARA-C 过于偏向弱类/弱客户端，提升公平性但牺牲平均精度。
```

下一步若继续改通信，建议不要纯替换 AsymHFL，而是做 hybrid：

```text
L_comm = L_AsymHFL + lambda_cara * L_CARA-C
```

或者：

```text
teacher selection 仍用 AsymHFL overall routing，
但 KD loss 中加入 class-aware residual weight。
```

### 必须补的严谨性

后续正式实验前，必须避免审稿人攻击：

```text
1. 不能只在 alpha=0.1 极端 Non-IID 上赢。
2. 至少覆盖 IID、alpha=1.0、0.5、0.3、0.1。
3. 正常场景不能明显低于 RAHFL。
4. severe Non-IID 下要重点看 avg_acc、worst_acc、tail_acc。
5. 通信路由不能使用最终测试集。
6. PRAC 的 route/accept 应使用本地 held-out validation split。
```

推荐最终实验定位：

```text
正常/IID 场景：保持 RAHFL 级别鲁棒性，不显著下降。
中重度 Non-IID：提升 worst-client / tail-class / average accuracy。
通信成本：如果 public1 接近 public4，应强调低通信开销。
```

### 当前不要做的事

暂时不要继续：

```text
1. 盲目设计新的 public-logit 通信模块。
2. 只在 PRAC 上反复调超参数。
3. 只追求 alpha=0.1 上超过 RAHFL。
4. 只报告 avg_acc，不分析 worst_acc / tail_acc。
5. 把工作讲成“RAHFL 粗粒度，我细粒度”。
```

更好的表述是：

```text
RAHFL-inspired but Non-IID-aware:
我们沿用强鲁棒本地增强思想作为公平基座，
但针对 RAHFL 在 label-skew 下的本地对比学习偏斜和客户端级通信偏差进行改进。
```

## Current Authoritative Next Steps - 2026-06-30

### Now: run AugMix + DCL local-only control

The key unanswered question is whether current PRAC communication adds value
beyond RAHFL-style local robust training.

Run:

```text
configs/kaggle_t4_augmix_dcl_local_only.yaml
```

Meaning:

```text
method_name: prac_hfl
warmup_rounds: 999
40 rounds of AugMix + CE + JSD + DCL local training
no PRAC communication in any round
```

Compare against:

```text
RAHFL public4 final:      avg_acc=56.41, worst_acc=44.72
PRAC public1 final:       avg_acc=54.63, worst_acc=41.88
PRAC public4 final:       avg_acc=52.96, worst_acc=43.27
```

Decision rule:

```text
If local-only >= PRAC public1/public4:
  current PRAC communication has weak or negative contribution.

If PRAC > local-only but still < RAHFL:
  PRAC has useful signal, but communication strength / accept policy needs tuning.

If local-only is much lower than PRAC:
  PRAC communication is useful and should be optimized rather than discarded.
```

Important current interpretation:

```text
PRAC is not empty: accept_rate is nonzero.
But public4 lowered avg_acc compared with public1 while improving final worst_acc.
This suggests weak-client help plus average-performance negative transfer.
```

## Current Authoritative Next Steps - 2026-06-25

### Now: run the new FedPRIME-PAIR full experiment

FedPRIME-PAIR has been implemented as a switchable method:

```text
method_name: fedprime_pair
FedPRIME-PAIR = PRIME + CBCL + CPAD
```

Smoke test passed:

```text
configs/debug_fedprime_pair_cifar10c.yaml
round 0: avg_acc=11.52, worst_acc=10.00, local_loss=5.1416, cpad_loss=0.7056
```

The next formal Kaggle run is:

```text
configs/kaggle_t4_fedprime_pair_full.yaml
```

Important runtime note:

```text
Use code at or after commit 8a4ee15.
The setup cell must show git log -1 containing 8a4ee15 or a later commit.
```

This version includes:

```text
1. FedPRIME-PAIR heartbeat logs in the full run.
2. CBCL forward-pass optimization that reuses model embeddings.
3. Kaggle data import compatibility for both --destination and --repo-root.
```

Expected full-run heartbeat:

```text
[heartbeat] round 000 start
[heartbeat] round 000 local client 0 start
[heartbeat] FedPRIME-PAIR local phase, client=0 batch=50 loss=...
```

Preferred Kaggle entry:

```text
Use the Python streaming launcher cell, not a long %%bash cell.
The Python cell should call:
  RUN_DEBUG=1 PYTHONUNBUFFERED=1 bash scripts/run_kaggle_pair.sh
```

This script performs data import, environment check, partition audit, optional
debug smoke, full FedPRIME-PAIR training, pair-expertise analysis, summary, and
output packaging in one uninterrupted background-safe command.

Reason:

```text
Kaggle/IPython may buffer %%bash stdout until the subprocess exits.
The Python streaming launcher uses subprocess.Popen and prints a driver
heartbeat every 60 seconds, so the run never appears silent.
Do not use sys.stdout.reconfigure in Kaggle; its OutStream has no reconfigure.
```

If a fresh full run reaches training but prints no heartbeat for about 10
minutes, stop it. Do not wait for hours. Inspect whether the notebook pulled
the latest commit and whether data import completed successfully.

It matches the previous T4 fair setting:

```text
rounds=40
local_epochs=1
batch_size=64
public_batch_size=128
public_batches_per_round=4
fixed partition: outputs/partitions/cifar10c_alpha05_seed0_clients4_samples10000.npz
```

The first comparison should be against the already reproduced baseline:

```text
RAHFL final: avg_acc=56.41, worst_acc=44.72
```

Primary decision criteria:

```text
1. final/best avg_acc vs 56.41
2. final/best worst_acc vs 44.72
3. cpad_loss finite and not exploding
4. pair_expertise heatmaps show client-class-pair differences
5. underrepresented diagnosis after checkpoints exist
```

If full FedPRIME-PAIR is below RAHFL, do not immediately redesign again. First
rerun with switches:

```yaml
method.use_cbcl: false   # PRIME + CPAD
method.use_cpad: false   # PRIME + CBCL local-only control
```

This identifies whether CBCL or CPAD is the bottleneck.

Kaggle prepared dataset remains:

```text
fedprime-data
```

Use `scripts/import_prepared_data.py` before training.

### Previous D2C diagnostic context

The repaired FedPRIME-D2C warmup=3 experiment completed all 40 rounds without
NaN/Inf, but did not beat RAHFL:

```text
RAHFL final:            avg_acc=56.41, worst_acc=44.72
FedPRIME-D2C final:     avg_acc=52.31, worst_acc=39.78
FedPRIME-D2C best avg:  avg_acc=52.83 at round 37
LogitAvg+PRIME final:   avg_acc=52.10, worst_acc=39.72
Oracle D2C final:       avg_acc=51.74, worst_acc=39.13
```

Conclusion:

```text
Old D2C public-prior debiasing is not the current main route.
FedPRIME-PAIR/CPAD is the new implementation to validate.
```

## Historical D2C Next Steps - 2026-06-07

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

0. Run the T4-safe PRIME local-backbone control before implementing another
   communication mechanism:

```text
configs/kaggle_t4_rahfl_prime.yaml
PRIME + DCL + original AsymHFL
```

This keeps all settings equal to `configs/kaggle_t4_rahfl.yaml` and changes
only `AugMix -> PRIME`. Compare its final/best `avg_acc`, `worst_acc`, and
underrepresented-class diagnosis against RAHFL. Do not attribute any later
PRIME-based communication gain to the communication module until this control
has been measured.

1. Run a T4-safe Oracle Prior D2C experiment. This is the highest-information
   next diagnostic:

```text
Implementation completed:
  configs/kaggle_t4_fedprime_d2c_oracle_warmup3.yaml
  fedprime/engine/prior_diagnostics.py
  scripts/analyze_priors.py

Formal Kaggle run is still pending.
```

Local end-to-end Oracle debug is complete on the RTX 3050. It produced finite
losses and all diagnostic outputs. The initial predicted prior had normalized
entropy `0.9999`, strongly indicating near-uniform prior collapse. The next
required experiment remains the full 40-round Kaggle Oracle run.

The full 40-round Oracle run is now complete:

```text
Oracle final:        avg_acc=51.74, worst_acc=39.13
Predicted D2C final: avg_acc=52.31, worst_acc=39.78
LogitAvg final:      avg_acc=52.10, worst_acc=39.72
```

Oracle is not better, so do not spend the next run only improving predicted
prior estimation. The next experiment priority is:

```text
1. T4-safe Oracle + no prior debias
2. Oracle with beta=0.1 or beta=0.2
3. Oracle + no class-balanced aggregation
4. Oracle + no complementary KD
5. smaller/ramped lambda_d2c
```

The first target is prior debias because missing classes receive up to about
`+3.45` logit under the current `beta=0.5, p_min=0.001` configuration.

The Oracle final checkpoints also show:

```text
client 2 missing_acc=0.00, tail_acc=4.63
client 3 missing_acc=0.00, tail_acc=0.00
```

Future D2C redesigns must be judged primarily by weak-client `tail_acc` and
`missing_acc`, not only average accuracy. A method that does not improve these
metrics does not validate the complementary-knowledge claim.

RAHFL-original missing/tail has now been measured:

```text
RAHFL final avg/worst: 56.41 / 44.72
client 2 missing_acc: 0.00
client 3 missing_acc: 0.00
```

So RAHFL's strong average performance does not demonstrate missing-class
transfer in this fixed alpha=0.5 split. The next research step is not another
RAHFL missing run. It is to design a public-logit communication module that can
explicitly improve missing/tail classes, or to test whether a same-domain
balanced public CIFAR-10 subset is required for that goal.

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

### Kaggle execution rule

Kaggle background `Save Version` runs cannot be modified or inspected with new
cells after starting. Prepare and validate the entire notebook before launch.
Future launch snippets must automatically perform setup, checks, training,
analysis, and result packaging. Never rely on adding a diagnostic cell during
an active background run.

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

## 2026-06-29 Next Actions

Current mainline:

```text
PRAC-HFL
```

D2C and FedPRIME-PAIR are now historical diagnostic results, not the active main method.

Immediate next task:

```text
Rerun safe PRAC-HFL on Kaggle from commit 5e476ea.
```

Before running, verify:

```text
git log -1 --oneline
5e476ea 增强PRAC-HFL数值稳定性
```

Use:

```text
configs/kaggle_t4_prac_hfl.yaml
scripts/run_kaggle_prac.sh
```

Safe PRAC-HFL has:

```text
warmup_rounds=3
CE-only route/accept risk
virtual_lr=0.005
head_max_grad_norm=1.0
train.max_grad_norm=5.0
skip_nonfinite=true
```

Judgment criteria:

```text
No NaN through round 039.
avg_acc should approach or exceed RAHFL 56.41.
worst_acc should approach or exceed RAHFL 44.72.
accept_rate should not remain zero forever.
avg_delta should become less negative than the first unstable run.
```

If safe PRAC-HFL works:

```text
1. Run PRAC-HFL multi-seed.
2. Run RAHFL local-only / Average-KD / AsymHFL / PRAC-HFL communication ablation.
3. Add underrepresented head/tail/missing diagnosis to PRAC-HFL checkpoints.
```

If safe PRAC-HFL still underperforms:

```text
1. Try accept gate with patience or EMA route risk.
2. Try classwise=false model-level PRAC to reduce noisy class routing.
3. Try full-model accepted KD only after head-only version is stable.
```

Tell Codex:

```text
读取 PROJECT_STATE.md 和 TODO_NEXT.md，继续推进 FedPRIME-D2C 项目。先检查 git 状态，然后准备数据和跑一个 debug smoke training。
```
