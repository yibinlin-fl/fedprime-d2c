# P2: CLE-Specific Class-Visible Routing Audit

Date: 2026-09-04

## Verdict

Allowed diagnoses:

```text
CLE_SPECIFIC_CLASS_VISIBLE_ROUTING
CLASS_ROUTING_EXCEEDS_GENERIC_FRAGILITY
```

Status:

```text
CANDIDATE_MECHANISM_FOR_CAUSAL_AUDIT
```

This is a retrospective, zero-training audit. It does not revive CRSF and does not authorize a new
intervention unless the status is `CANDIDATE_MECHANISM_FOR_CAUSAL_AUDIT`.

## What was tested

For each of the 16 frozen round-40 H0/H9/L0/L9 client models, both independent PRIME banks and both
disjoint 500-carrier halves were analyzed from K0-B's already-saved centered class-logit responses.
For every probe q:

```text
mu_q = mean_u delta_q(u)
E_q  = mean_u ||delta_q(u)||^2
z_q  = mu_q / (sqrt(E_q) + eps)
Z    = [z_1,...,z_64] in R^(10 x 64)
```

`chi_out` is the participation concentration of `Z^T Z`. The positive class profile is
`g_c = mean_q relu(Z_cq)^2`. Because every column is normalized by its own raw response energy,
these objects ask whether nuisance responses acquire stable class-directed geometry, not merely
whether all output perturbations become larger.

## Taxonomy-free CLE contrasts

| System | Bank | Half | Mean client chi ratio | Mean client positive-routing ratio | Mean client class-concentration ratio | Positive chi clients | Mean client raw-energy ratio |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| hfl | bank_a | ua | 1.876x | 4.809x | 2.176x | 4/4 | 3.068x |
| hfl | bank_a | ub | 1.856x | 4.717x | 2.235x | 4/4 | 3.187x |
| hfl | bank_b | ua | 2.008x | 5.262x | 2.176x | 4/4 | 2.805x |
| hfl | bank_b | ub | 1.982x | 5.200x | 2.256x | 4/4 | 2.844x |
| local | bank_a | ua | 1.763x | 4.351x | 2.594x | 4/4 | 2.322x |
| local | bank_a | ub | 1.721x | 4.323x | 2.614x | 4/4 | 2.425x |
| local | bank_b | ua | 1.668x | 4.187x | 2.258x | 4/4 | 2.154x |
| local | bank_b | ub | 1.596x | 4.178x | 2.260x | 4/4 | 2.188x |

Across strong-CLE arms, the worst cross-half positive-profile cosine is
`0.997664` and the worst cross-bank cosine
is `0.975400`. The minimum pooled
`chi_out` ratio is `1.596x`; the minimum positive
routing-strength ratio is `4.178x`;
the minimum class-concentration ratio is
`2.176x`.

## Incremental value versus K0-B

The following associations use all 16 arm/client observations. They are descriptive only: there is
one matched seed and the observations share training/data ancestry.

| New object | Pearson with K0-B R | Spearman with K0-B R | Pearson with DSA | Spearman with DSA | Residual Pearson with DSA after K0-B R |
| --- | ---: | ---: | ---: | ---: | ---: |
| chi_out | 0.8740 | 0.8999 | 0.8394 | 0.7408 | 0.3423 |
| positive_routing_strength | 0.8483 | 0.9175 | 0.9810 | 0.8969 | 0.9305 |
| positive_class_concentration | 0.9655 | 0.9381 | 0.9026 | 0.8586 | 0.5169 |

For reference, original K0-B R has Pearson/Spearman association with DSA of
`0.8648/0.8616`.
The strongest new descriptive DSA association is
`positive_routing_strength` at
`0.9810/0.8969`.

The stricter matched CLE-effect analysis uses the eight `(H9-H0)` / `(L9-L0)` client contrasts:

| Effect predictor | Pearson with DSA effect | Spearman with DSA effect |
| --- | ---: | ---: |
| k0b_R | 0.5697 | 0.7381 |
| chi_out | -0.0030 | 0.1667 |
| positive_routing_strength | 0.9459 | 0.9524 |
| positive_class_concentration | 0.7273 | 0.7619 |

Here `chi_out` alone does not rank the size of client DSA effects, whereas normalized positive
routing strength does (`0.9459/0.9524`). This is why the promoted object is the explicit class-routing
profile/strength, not another spectral scalar. Its minimum pooled ratio (`4.178x`) also exceeds the
largest pooled raw centered-energy ratio (`3.187x`), while class concentration rises by at least
`2.176x`; the effect is not explained by a uniform enlargement of all responses.

## Interpretation boundary

- `CLE_SPECIFIC_CLASS_VISIBLE_ROUTING` requires a >=1.20x pooled increase in both `chi_out` and
  normalized positive routing strength in every bank/half/system slice, >=3/4 positive clients in
  every slice, and >=0.90 strong-CLE profile stability across halves and banks.
- `CLASS_ROUTING_EXCEEDS_GENERIC_FRAGILITY` additionally requires >=1.05x class-profile
  concentration and >=1.20x standardized routing energy in every pooled slice. This is a
  magnitude-normalized structural check, not a claim that its raw numerical ratio must exceed raw
  energy's ratio.
- `REDUCES_TO_K0B_DETECTOR` is assigned only when all three new summaries are highly redundant with
  K0-B R, have weak residual association after R, and none improves the descriptive DSA association
  by 0.05.
- No Pearson/Spearman value is a significance claim. No causal mediation is established here.

## Provenance and sealing

The four taxonomy-free tables were written and SHA256-sealed before the Phase-A1a round-40 DSA CSV
or K0-B risk table was opened. The source response hashes, public split hash and two probe-bank hashes
are recorded in `primary_taxonomy_free_manifest.json`; the final file hashes are in `manifest.json`.
No checkpoint, model, GPU, OpenI job, PRIME generator, corruption binding, family or severity was used.

All seven numerical CSV outputs were byte-identical across two complete reruns. A separate direct
NumPy recomputation of H9/ResNet12, Bank-B/Ub reproduced `chi_out=0.7131165927878319` to
`2.22e-16` absolute error and reproduced its standardized trace exactly at printed precision.

Generated taxonomy-free detail also includes `probe_routing_assignments.csv`, which records every
probe's top routed class, top1--top2 margin and cross-half retention.

## Scientific conclusion

The result supports only the diagnoses printed above. A positive CLE contrast alone is not a new
mechanism if it is merely a re-expression of K0-B's detector. Conversely, failure of the frozen
specificity/fragility gates stops this representation/readout intervention branch without another
GPU experiment.
