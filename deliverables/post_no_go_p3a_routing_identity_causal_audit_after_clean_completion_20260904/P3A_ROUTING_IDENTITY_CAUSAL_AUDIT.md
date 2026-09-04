# P3-A Matched Routing-Identity Causal Audit

Updated: 2026-09-04

## Outcome

```text
verdict: CLASS_IDENTITY_CAUSAL_BUT_GENERIC_PROFILE_NOT_TARGETING
status:  NO_GO_TO_METHOD
```

Clean-base completion first removed the only Stage-0 artifact gap. P3-A then created all targeted and
random class-coordinate permutations before reading CLE binding, corruption family or DSA. It used
no model inference beyond the separately sealed clean-only completion, no training and no OpenI job.

## Integrity and invariance

- identity reconstruction maximum probability error: `2.3315e-7`;
- maximum targeted invariance error: `3.6380e-10` (`<=1e-8` PASS);
- maximum random K0-B metric error: `1.6653e-16`;
- 1000 random permutations per arm/client were unique derangements from seed `20260904`;
- the project oracle independently reproduced all identity and targeted DSA values exactly.

The intervention therefore changes class-coordinate identity while preserving response norm,
pairwise sample Gram, singular spectrum, output chi, raw response energy and K0-B generic risk.

## Primary results

| arm | identity DSA | targeted DSA | reduction | positive clients | random q10 | target percentile |
|---|---:|---:|---:|---:|---:|---:|
| H0 | 0.001523 | 0.000151 | 0.001372 | 3/4 | -0.002443 | 53.7% |
| H9 | 0.204270 | -0.013220 | 0.217490 | 4/4 | -0.032512 | 60.1% |
| L0 | 0.000823 | -0.000788 | 0.001611 | 2/4 | -0.004907 | 44.3% |
| L9 | 0.205189 | -0.061513 | 0.266702 | 4/4 | -0.038680 | 0.0% |

Gates A, B, C, D and F pass. Gate E passes for L9 but fails for H9: the P2-derived HFL targeted
permutation is not better than a typical random derangement, much less the bottom-decile null required
by the frozen protocol.

## Interpretation

Changing class identity can strongly change the signed DSA while every geometry/magnitude statistic
is held fixed. However, random identity changes also collapse DSA, and the taxonomy-free P2 ranking
does not identify a specifically harmful HFL routing assignment. The Local-only L9 result is striking
but cannot rescue the method claim because the HFL H9 targeting gate was preregistered and failed.

Negative DSA after permutation means the response moves away from the historically bound class; it is
not negative accuracy and is not evidence that the model was repaired. P3-A is an output-space causal
audit, not a deployable mitigation.

Consequently P2 remains a strong observational descriptor of CLE-related class-visible routing, but
it is not validated as a targeting rule. Do not start P3-B, design a routing-strength loss, tune the
permutation, or run training from this result.
