# Post-NO-GO Mechanism Audit P1: Class-Readout Routing

Date: 2026-09-04

## Verdict

The final zero-cost representation audit supports:

```text
READOUT_COUPLING_REMAINS_INTACT
+ GLOBAL_CHI_MISSES_CLASS_VISIBLE_GEOMETRY
+ READOUT_WEIGHTED_GEOMETRY_TRACKS_DSA
```

Status:

```text
CANDIDATE_MECHANISM_FOR_NEXT_AUDIT
```

This is not a method GO and does not revive CRSF. It says that the exploratory readout-weighted
response geometry tracks the direction of the already-observed DSA changes better than global chi,
while CRSF leaves the actual class routing almost unchanged. Because H0/L0 feature responses and
independent replications are unavailable, P1 does not establish CLE specificity, mediation or a
trainable objective.

## Execution and integrity

`scripts/analyze_post_no_go_class_readout_p1.py` performed CPU-only state-dict loading and linear
algebra. It did not construct a model, generate PRIME views, run forward/backward, update a tensor or
use CUDA/OpenI.

All four original checkpoint hashes and all four classifier hashes matched the sealed K1-C-Minimal
manifests. Both architectures expose exactly one linear classifier, with `10 x 512` and `10 x 1280`
weights matching their stored representation dimensions. CRSF/RawSpec deltas contain no classifier
parameter. All 24 normalized response matrices independently reproduced their saved Grams with
maximum error `9.55e-13`.

## Mathematical object

To remain consistent with K1-C, P1 reconstructs the normalized probe-response matrix

```text
S[:, q] = mu_q / sqrt(E_q),      S = U diag(sigma) V^T.
```

For the centered classifier `Wc = P_C W`, it computes

```text
readout_gain_j     = ||Wc u_j||^2
weighted_lambda_j  = sigma_j^2 * readout_gain_j
chi_RW             = sum(weighted_lambda_j^2) / sum(weighted_lambda_j)^2
R_normalized       = Wc S
R_raw              = Wc [mu_1,...,mu_64].
```

`weighted_lambda_j` decomposes total readout energy by the original response modes; it is not
claimed to be an eigenvalue of `R^T R`. The true singular spectrum of `R_normalized` is reported
separately.

## A. Readout coupling remains intact

CRSF does not modify `W`. Its small rotation of the dominant response mode also leaves the dominant
mode's class-readout vector almost unchanged:

| Architecture/context | Mode-1 response cosine | Mode-1 readout-coupling cosine | Routing-matrix cosine | Class-norm-vector cosine | Probe top-class retention |
| --- | ---: | ---: | ---: | ---: | ---: |
| ResNet10 H9/L9 | 0.997245 | 0.997850 | 0.995453 | 0.998742 | 90.625% |
| MobileNetV2 H9 | 0.999416 | 0.999786 | 0.999689 | 0.999818 | 100% |
| MobileNetV2 L9 | 0.999442 | 0.999817 | 0.999720 | 0.999838 | 100% |

For every CRSF context, Top-3 and Top-5 class-routing-norm sets are identical to Frozen. The average
row and column cosines are at least `0.99457` and `0.99558`, respectively. The dominant readout gain
does not decrease; it rises by approximately `0.82--1.70%`, and the response-energy-weighted mean
gain rises by `0.52--2.29%`.

The normalized routing magnitude falls only `0.46--1.32%`. The raw expected routing magnitude falls
more (`1.46--6.53%`) but retains almost the same direction, and the mean per-probe top1/top2 margin
does not contract. This directly supports `READOUT_COUPLING_REMAINS_INTACT`.

## B. Global chi versus class-visible geometry

| Context | Arm | Delta chi | Delta chi_RW | Relative delta DSA | chi_RW-to-DSA ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| ResNet10 H9/L9 | CRSF | +8.936% | +11.056% | +3.577% | 0.324 |
| MobileNetV2 H9 | CRSF | +2.199% | +1.733% | +0.550% | 0.317 |
| MobileNetV2 L9 | CRSF | +2.412% | +1.562% | +0.506% | 0.324 |
| ResNet10 H9/L9 | RawSpec | +2.409% | -0.996% | -0.269% | -- |
| MobileNetV2 H9 | RawSpec | +0.752% | +0.099% | +0.113% | -- |
| MobileNetV2 L9 | RawSpec | +0.694% | -0.606% | -0.382% | -- |

Positive deltas mean a reduction/improvement. RawSpec is the clearest counterexample to treating
global chi as class-visible geometry: global chi decreases in every context, but chi_RW and DSA both
worsen for ResNet10 and L9/MobileNetV2. The global spectrum can therefore flatten while the
classifier-visible outcome moves in the wrong direction. This supports
`GLOBAL_CHI_MISSES_CLASS_VISIBLE_GEOMETRY`.

## C. Does exploratory chi_RW track DSA?

Across all eight provenance rows, descriptive correlation with relative DSA change is:

| Predictor | Pearson | Spearman |
| --- | ---: | ---: |
| Delta global chi | 0.9684 | 0.7317 |
| Delta chi_RW | 0.9988 | 0.9268 |

After removing the bit-identical L9/ResNet duplicate, six functionally unique rows remain:

| Predictor | Pearson | Spearman |
| --- | ---: | ---: |
| Delta global chi | 0.9643 | 0.7143 |
| Delta chi_RW | 0.9980 | 0.9429 |

For the three functionally unique CRSF rows, chi_RW and DSA have the same ordering and nearly the
same conversion ratio (`0.317--0.324`). This is unusually consistent across the two architectures:
the MobileNet effect is small because CRSF barely changes chi_RW there, not because a large
class-visible change fails to reach DSA.

The sample is nevertheless tiny, shares one seed/intervention and contains no H0/L0 feature-space
counterpart. These correlations are descriptive and cannot establish significance or causal
mediation. The proper diagnosis is therefore `READOUT_WEIGHTED_GEOMETRY_TRACKS_DSA`, not a new
metric or method claim.

## D. Architecture interpretation

CRSF's controllability remains strongly architecture-dependent: chi_RW falls `11.06%` on ResNet10
but only `1.56--1.73%` on MobileNetV2; DSA follows at `3.58%` versus `0.51--0.55%`. However, the
chi_RW-to-DSA ratio is nearly constant and routing is retained in both architectures. Current data
therefore do not support the stronger allowed diagnosis `ARCHITECTURE_SPECIFIC_READOUT_MECHANISM`.
They support architecture-specific intervention leverage, already established by P0.

## What P1 resolves

P0 showed that global response concentration changed while final class-bound DSA routing stayed
nearly fixed. P1 identifies the internal reason:

```text
CRSF changes response-mode energy
-> classifier W remains fixed
-> response modes and Wc-mode coupling barely rotate
-> class/probe routing identity is retained
-> only a small amplitude/concentration change reaches DSA.
```

The readout-weighted statistic is a better retrospective descriptor than global chi in this frozen
artifact set, especially for RawSpec sign errors. It has not yet been shown to distinguish H9 from
H0, generalize across banks/seeds/all architectures, or be safely controllable.

## Stopping boundary

P1 does not authorize training, a new loss, another checkpoint surgery or CRSF tuning. CRSF remains
`NO_GO_CRSF_INTERVENTION`. The only permitted promotion is a paper-level next audit of whether the
readout-weighted geometry is genuinely CLE-specific using existing artifacts if possible. If that
cannot be established without new inference, a new experiment requires a separately frozen,
low-cost protocol and explicit user approval.

Generated files:

```text
classifier_manifest.json
readout_weighted_modes.csv
readout_weighted_summary_by_half.csv
readout_weighted_summary.csv
chi_chirw_dsa_changes.csv
chi_chirw_dsa_correlations.csv
class_routing_norms.csv
routing_singular_spectrum.csv
ARTIFACT_AVAILABILITY.md
```
