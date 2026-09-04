# Post-NO-GO Mechanism Audit P0

Date: 2026-09-04

## Conclusion

The P0 audit explains the K1-C-Minimal failure without changing its verdict:

```text
GLOBAL_SPECTRUM_IS_WEAK_PROXY
+ CLASS_ROUTING_REMAINS_INTACT
+ ARCHITECTURE_DEPENDENT_CONTROLLABILITY
+ SPECTRAL_REDUCTION_BY_TAIL_REDISTRIBUTION
```

CRSF did change the generic response spectrum, but that change was not a strong control variable for
the harmful CLE class routing. Its pooled unseen chi fell by `5.369%/5.452%` for H9/L9, while pooled
DSA fell by only `2.322%/2.308%`. The corresponding chi-to-DSA conversion efficiencies were only
`0.433/0.423`. This remains `NO_GO_CRSF_INTERVENTION`; it is not evidence for tuning, replication or
full training.

## Provenance and execution boundary

The analyzer read only the existing K0-B, K1-B0, K1-C0 and K1-C-Minimal artifacts. The numerical P0
tables were produced by `scripts/analyze_post_no_go_mechanism_p0.py` using NumPy on CPU. No
checkpoint was loaded, no PRIME view was generated, no model forward/backward pass ran, and no GPU
or OpenI job was used.

The K1-C-Minimal formal archive had already passed its independent integrity audit: all 31 artifact
hashes, 18 pre-oracle seal hashes, saved Gram/moment recomputations, six oracle prediction
recomputations and eight optimization traces matched. P0 analyzes that frozen evidence; it does not
redefine a gate.

## A. Class-routing retention

The frozen DSA was decomposed into additive class-by-bound-family terms. Two views were retained:

- a 20-dimensional client-binding vector, preserving each selected client's class/family pairing;
- a 10-dimensional semantic-class vector, averaging the same class across the two selected clients.

| System | Arm | Routing vector | Frozen/intervention cosine | Spearman | Magnitude ratio | Top-3 / Top-5 overlap |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| H9 | CRSF | client-binding 20d | 0.998579 | 1.000000 | 0.952435 | 1.00 / 1.00 |
| H9 | CRSF | semantic-class 10d | 0.999296 | 1.000000 | 0.956928 | 1.00 / 1.00 |
| L9 | CRSF | client-binding 20d | 0.998559 | 0.998496 | 0.952351 | 1.00 / 1.00 |
| L9 | CRSF | semantic-class 10d | 0.999251 | 1.000000 | 0.956967 | 1.00 / 1.00 |

The class ordering is therefore essentially unchanged. Across the 40 provenance-labelled class
terms, only 14 decreased and 26 increased; the positive pooled change is concentrated in a few large
components. For example, ResNet10 class 4 fell by `0.039694`, while class 2 increased by `0.012800`.
The same top shortcut classes remain top-ranked. RawSpec likewise retained cosine above `0.9987` and
all top-k sets.

Interpretation: CRSF slightly rescales the existing class-bound shortcut vector; it does not reroute
the corruption evidence away from its bound classes. This directly supports
`CLASS_ROUTING_REMAINS_INTACT`.

## B. Chi-to-DSA coupling

| System / scope | Arm | Relative chi reduction | Relative DSA reduction | Conversion efficiency |
| --- | --- | ---: | ---: | ---: |
| H9 pooled | CRSF | 5.369% | 2.322% | 0.433 |
| L9 pooled | CRSF | 5.452% | 2.308% | 0.423 |
| ResNet10 client0 | CRSF | 8.936% | 3.577% | 0.400 |
| H9 MobileNetV2 client3 | CRSF | 2.199% | 0.550% | 0.250 |
| L9 MobileNetV2 client3 | CRSF | 2.412% | 0.506% | 0.210 |

The eight client-arm points give descriptive Pearson `0.9684` and Spearman `0.7317`; CRSF-only gives
Pearson `0.9995` over four rows. These values must not be read as a reliable general law: the sample
is tiny, the two ResNet system rows are functionally duplicated, and the points are dominated by the
large ResNet-versus-MobileNet and CRSF-versus-RawSpec separation. RawSpec alone has Pearson
`-0.3370`, and its pooled DSA slightly worsens despite a small chi reduction.

Thus chi and DSA co-move under this one intervention, but with a shallow, architecture-dependent
conversion. A global chi reduction is not sufficient to predict useful CLE suppression. This
supports `GLOBAL_SPECTRUM_IS_WEAK_PROXY`, not a causal mediation claim.

## C. Spectral autopsy

All aligned 64-by-64 Bank-B Gram matrices were symmetrized and independently eigendecomposed.

| Scope | Arm | Trace change | Lambda-1 change | Tail change | Top-1 share change | Principal-vector cosine |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| All contexts | CRSF | -3.301% | -6.433% | +6.210% | -0.02452 | 0.999693 |
| ResNet10 | CRSF | -4.811% | -9.837% | +9.472% | -0.03905 | 0.999418 |
| MobileNetV2 | CRSF | -1.791% | -3.029% | +2.949% | -0.00999 | 0.999969 |
| All contexts | RawSpec | -0.221% | -1.070% | +2.384% | -0.00639 | 0.999228 |

The chi drop is not pure denominator inflation: CRSF does suppress the dominant eigenvalue. However,
it simultaneously increases the sum of eigenvalues 2--64, and the principal direction barely
rotates. In other words, the same dominant response axis remains, with reduced amplitude and more
energy redistributed into its tail. That flattening is real, but it is not equivalent to deleting
the class-bound shortcut pathway. This is the precise, qualified form of
`SPECTRAL_REDUCTION_BY_TAIL_REDISTRIBUTION`.

## D. Architecture leverage

ResNet10 carries nearly all of the observed intervention effect:

- CRSF chi reduction: `8.936%` on ResNet10 versus `2.199--2.412%` on MobileNetV2.
- Absolute DSA reduction: `0.009445` on ResNet10 versus `0.000940--0.001029` on MobileNetV2.
- CRSF-over-RawSpec chi advantage: `6.528 pp` on ResNet10 versus `1.447--1.717 pp` on MobileNetV2.
- CRSF-over-RawSpec DSA advantage: `0.010157` on ResNet10 versus `0.000817--0.001649` on
  MobileNetV2.
- WCCA is unchanged in every saved client/arm comparison; CFG improves by only `0.25--0.50 pp` under
  CRSF and worsens under RawSpec.

This supports `ARCHITECTURE_DEPENDENT_CONTROLLABILITY`: a statistic measured at the final
representation does not provide an architecture-invariant intervention handle. Architecture
dependence is an explanation of the failed gate, not a reason to rescue CRSF.

One further limitation is material: H9/client0 and L9/client0 have distinct checkpoint archive
hashes, but their K0-B response tensors and K1-C-Minimal Frozen/CRSF/RawSpec prediction tensors are
bit-identical. Those rows are retained for provenance but are not two independent ResNet
replications.

## Mechanism interpretation

The evidence now supports the more cautious chain:

```text
CLE -> harmful class-bound shortcut routing -> generic response-spectrum concentration
```

It does not support the stronger mediation assumption:

```text
CLE -> response-spectrum concentration -> DSA
```

K0-B remains a valid offline diagnostic and K1-C0 remains an observational mechanism result. P0
shows why flattening the observed global spectrum did not close the detect-to-intervene loop: the
class-bound routing direction and its ranking remained almost unchanged, the spectral change mixed
dominant-mode attenuation with tail redistribution, and the available control leverage differed
sharply by architecture.

## Scope and stopping rule

This P0 result proposes no new method and authorizes no new experiment. CRSF, K1-C-FULL, B-to-A,
additional architectures, tuning, replication and full training remain stopped. The existing
artifacts are sufficient for the requested four audits; the explicitly missing feature-level and
cross-stage sample-aligned analyses are documented separately and must not be filled by a rerun.

Generated tables:

```text
class_routing_retention.csv
class_routing_summary.csv
chi_dsa_coupling.csv
chi_dsa_correlations.csv
spectral_autopsy.csv
architecture_leverage.csv
ARTIFACT_AVAILABILITY.md
```
