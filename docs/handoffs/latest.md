# FedPRIME-D2C Session Handoff

Updated: 2026-08-16

## Latest Paper-Claim Audit: PEW+BER Is Baseline GO, Core-Method NO-GO

The implementation, existing evidence and external literature were audited without
running experiments. The current exact method is a six-way public synthetic
corruption-family classifier followed by class x predicted-environment reweighted
ERM. It does not read private operator metadata for training, but it is not
taxonomy-free. BER is neither GroupDRO nor CVaR; it is a support-shrunk grouped
average risk and does not define a new robust-optimization principle.

External collision is material: Corrupted CIFAR-10 in Learning from Failure
(NeurIPS 2020) already couples labels with corruption types; SSA/BARACK already
use predicted spurious/group attributes for downstream robust training; CCDB and
FG-CCDB directly study class-conditional distribution balancing. CLE-HFL can be
positioned only as a controlled model-heterogeneous federated extension with
client-specific mappings and operator-cell evaluation, not as the first
class-corruption entanglement problem.

Verdict:

```text
PEW+BER empirical mechanism on fixed CLE:       GO
PEW+BER taxonomy-assisted diagnostic baseline:  GO
PEW+BER as the sole paper-level core method:     NO-GO
CLE-HFL as a federated benchmark extension:      CONDITIONAL GO
```

Do not spend the next stage merely adding exact PEW+BER seeds/rounds or renaming
the scenario. Preserve PEW+BER as the positive anchor; a new candidate must add
an FL-specific mathematical object and pass paper-level collision checks before
implementation. Full evidence:

```text
docs/research/status/PEW_BER_PAPER_CLAIM_AUDIT_2026_08_16_ZH.md
```

## Current Objective

Replace the reviewer-vulnerable five-family PEW assumption with a genuinely
taxonomy-free local risk mechanism that can retain BER's weak-cell benefit.
Do not spend the next stage ranking Non/HFL/AsymHFL or running the prepared
six-arm communication factorial. A candidate must first define a new
mathematical object, require no environment label, differ from PEW/PIE/C3R and
the frozen robust-risk baselines, and pass an isolated local promotion audit.

## Latest Theory Result: LCC - NO-GO Before Implementation

Latent Correction Conflict (LCC) was formalized as class-conditioned
per-sample last-layer gradient grouping followed by a minimum-norm common
descent update. It requires no environment labels and differs from the frozen
project methods, but it does not pass the external novelty gate:

```text
gradient clustering -> latent robust groups   GRASP collision
minimum-norm common descent                   MGDA/CAGrad collision
last-layer gradient KNN soft neighborhoods    GoG (KDD 2025) collision
```

Verdict: `THEORY NO-GO`. Do not implement LCC, change its clustering/graph, or
spend GPU/OpenI time on it. Evidence:

```text
docs/archive/methods/LCC_NOVELTY_AUDIT_ZH.md
```

## Taxonomy-Free Identifiability Boundary

Using client identity as an unlabeled mixture view was also checked before
turning it into a communication module. For class `c`, observable client risks
satisfy `r_c = Pi_c rho_c`; centered client contrasts can identify at most
`K-1` environment-risk directions. In the frozen four-client CLE mapping, the
effective family-contrast ranks by class are:

```text
class: 0 1 2 3 4 5 6 7 8 9
rank:  1 2 1 3 1 1 2 2 0 2
```

Only class 3 has full four-family contrast coverage; class 8 has none. Model
heterogeneity further confounds client-risk differences. Pure client-class
variance/DRO therefore cannot replace BER with a clean guarantee and must not
be promoted as the next communication innovation. Evidence:

```text
docs/research/status/TAXONOMY_FREE_IDENTIFIABILITY_2026_08_11_ZH.md
scripts/audit_mixture_contrast_identifiability.py
```

## Latest Result: CRSR Audit 0 - NO-GO

Class-conditional Residual Spectral Risk (CRSR) used only fit-internal labels
and predictions:

```text
r(x,c)   = softmax(f(x)) - one_hot(c)
Sigma_c  = Cov(r | y=c)
S_c      = sqrt(lambda_max(Sigma_c))
L_CRSR   = class-balanced CE + 2.0 * mean_c S_c
```

The frozen local Audit 0 completed on client 1/ResNet12 and client
3/MobileNetV2 without reading private audit or final test. Operator IDs were
used only for post-hoc cell evaluation. Independent recomputation matched the
script: G0--G3 passed; G4--G6 failed.

```text
median top share             0.752411  PASS
median direction cosine      0.975226  PASS
median transfer advantage    0.639379  PASS
median spectral cell rho     0.069658  FAIL (< 0.25)
median advantage vs baseline -0.903889 FAIL (< 0.02)
mean CE delta, clients 1/3   +0.006485 / +0.000360  FAIL
worst-cell CE delta          +0.101343 / -0.045068  FAIL
verdict                      NO-GO
```

Interpretation: the class-residual spectrum is active, stable across disjoint
splits, and nonredundant with sample CE/Brier, but it does not consistently
identify weak class-operator cells and its optimization is not mean-risk
noninferior. Freeze CRSR. Do not tune its weight, support thresholds, probe
size, power-iteration count, or gates; do not connect it to the runner or run
12/40 rounds.

Evidence and retained isolated implementation:

```text
docs/experiments/archive/CLASS_RESIDUAL_SPECTRAL_RISK_AUDIT_ZH.md
outputs/class_residual_spectral_risk_audit0/result.json
outputs/class_residual_spectral_risk_audit0/signals.npz
fedprime/methods/class_residual_spectral_risk.py
scripts/audit_class_residual_spectral_risk.py
tests/test_class_residual_spectral_risk.py
```

## Current Formal Positive Result

The selected local path remains calibrated hard PEW + hard BER; the legacy
strict three-seed positive package also included the then-active CDep term.
On fixed CLE-HFL v2 `seed0_split0`, that package's matched 12-round
training-seed 0/1/2 deltas versus AugMix/JSD/DCL control were:

```text
mean Avg +4.5880, Worst +4.2169, WCCA +5.5500, CFG -6.7150
```

The 40-round training-seed-0 durability result also passed all frozen gates.
These results establish the empirical target to preserve, not a defense of
PEW's five-family taxonomy.

## Other Frozen Recent Negatives

```text
Multi-label PEW + Soft-BER: NO-GO (0/4 matched last-five gates)
PIE/MPIE: NO-GO; do not implement PBR
C3R: NO-GO; do not implement its training loss
CRSR: NO-GO; stable geometry but invalid weak-cell surrogate
```

Also obey the permanent frozen-negative list in `AGENTS.md`; do not revive
FedCIS, continuous-witness, IRD/PCCD, or communication methods already archived
as negative. A new object must additionally be distinguished from
GroupDRO/CVaR and CCAD instead of merely renaming their objective.

## Next Action

Do not implement LCC or pure client-mixture contrast. The next candidate must
state what additional information makes hidden environments identifiable
(rather than silently assuming labels, clusters, high-loss tails, or enough
clients). First define that side information, its minimal assumption and a
cheap falsification audit; then distinguish it from PIE/MPIE, C3R, CRSR,
FedCIS, continuous witness, GRASP/GoG, MixStyle/Fourier mixing, and CVaR. Do not
run another full local training audit until this paper-level filter passes.
