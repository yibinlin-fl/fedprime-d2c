# P4 Routing Targetability Gap Audit

Updated: 2026-09-04

## Verdict

```text
GENERIC_ROUTING_ALIGNS_WITH_HARMFUL_ROUTING_BUT_TARGET_RULE_FAILED
recommendation: INFORMATION_MAY_EXIST_IDENTIFIABILITY_AUDIT_REQUIRED
method_go: false
```

P4 used only already sealed K0-B, Phase-A1a, P3-A clean-base and P3-A permutation/null artifacts.
It ran no model inference, GPU/OpenI job, PRIME generation, optimization or training, and it did not
change or search the failed P3-A permutation.

## Objects

For class `a`, let `f_i(a)` be its historical CLE-bound family for client `i`. P4 defines:

```text
M_harm[a,b] = mean d_b
              over operators in f_i(a)
              and sources whose true class is not bound to f_i(a)

g_harm[a]   = mean relu(d_a) over the same support
```

The source exclusion exactly matches the original DSA valid-source rule. `g_generic` is reproduced
without oracle information from P2's K0-B Bank-A and carrier-half Ua definition. All 160 generic
class-profile entries exactly match the previously sealed P2 table.

## Main evidence

| arm | mean cosine | mean Spearman | Top-3 overlap | target signed-destructive percentile | target DSA percentile |
|---|---:|---:|---:|---:|---:|
| H0 | 0.7871 | 0.2970 | 0.3333 | 65.5% | 53.7% |
| H9 | 0.8496 | 0.7091 | 0.8333 | 42.5% | 60.1% |
| L0 | 0.8057 | 0.3030 | 0.5000 | 43.9% | 44.3% |
| L9 | 0.8459 | 0.4667 | 0.8333 | 100.0% | 0.0% |

The proposed HFL-decoupling explanation is rejected. H9 has strong generic-vs-harmful alignment and
actually higher mean Spearman than L9. Three of four matched clients have lower L9-minus-H9 Spearman,
so the Local result is not explained by Local having a uniformly better marginal profile.

The fixed rank-reversal rule instead acts very differently on the full harmful routing matrix. For
H9 its signed destructive score is only at the random-null 42.5th percentile (energy score 74.8th),
whereas L9 is at 100.0% (energy 99.7%). Across random controls, signed destructive score correlates
with DSA reduction at `0.721` for H9 and `0.677` for L9, supporting the matrix-action explanation.

## Meaning

Generic probes contain substantial information about which class marginals absorb nuisance evidence.
But a one-dimensional class-salience vector does not identify the pairwise action required to reroute
that evidence. Rank reversal discards the off-diagonal structure of `M_harm`; it happens to be highly
destructive for L9 but is ordinary under the H9 null.

This does not rescue P2 or P3-A. The harmful matrix itself uses the CLE binding oracle and cannot be
turned into a method. The only permitted next step, if pursued, is an independent identifiability
audit asking whether any taxonomy-free observable contains the missing pairwise information. No new
loss, permutation rule, training or GPU experiment is authorized.
