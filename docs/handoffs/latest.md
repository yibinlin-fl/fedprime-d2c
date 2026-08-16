# FedPRIME-D2C Session Handoff

Updated: 2026-08-17

## Latest Candidate Audit: FCNT / FPER / FRT Do Not Enter Implementation

Three explicit side-information routes were formalized and checked against both frozen project
evidence and primary literature without changing training code or running experiments:

```text
FCNT: continuous nuisance coordinates + class-conditional federated OT
FPER: paired restoration intervention + degradation-effect risk
FRT:  public multi-view response tensor factorization
```

None passes the current core-method gate. FCNT is surrounded by CCDB/FG-CCDB,
class-conditioned Wasserstein DRO, FedWaD/FedDaDiL and SLOT-Align; a Wasserstein barycenter
also provides no lower mass bound for latent weak cells. FPER requires an unverifiable
label-preserving nuisance-removal oracle, does not prevent minority-cell dilution and collides with
counterfactual invariance/generation plus the frozen C3R/FedCISA reasoning. FRT lacks an identifiable
semantic/shortcut decomposition, cannot connect public responses to private weak-cell mass, and
repeats the public multi-view/shared-subspace risks already rejected by CCAD/IRD/FedCIS/EBST.

Verdict:

```text
FCNT current-protocol core:       NO-GO
FCNT with explicit real metadata: CONDITIONAL REFRAME ONLY
FPER observed-only core:          NO-GO; paired/clean ORACLE ONLY
FRT communication:               NO-GO
implementation / experiment:     NONE
```

Full report:

```text
docs/archive/methods/FCNT_FPER_FRT_THEORY_NOVELTY_AUDIT_2026_08_17_ZH.md
```

The next action is a strategic route choice, not a fourth taxonomy-free module: explicitly add a
realistically available side-information assumption, retain PEW+BER for a conservative empirical
paper, or stop CLE as the method-paper mainline. Do not implement or run any of these three candidates
before the user selects the route.

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

Await the user's strategic route choice after the paper-level rejection of the
three explicit side-information candidates. Do not implement another
taxonomy-free local or communication module by default. Preserve PEW+BER as the
strong supervised reference and the CLE-HFL scenario as a controlled benchmark
extension. The next authorized direction must explicitly choose between a new
realistic side-information assumption, a conservative empirical/benchmark
paper, or leaving CLE as the method-paper mainline.

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
