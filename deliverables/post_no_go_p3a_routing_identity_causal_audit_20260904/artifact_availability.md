# P3-A Artifact Availability

Date: 2026-09-04

## Frozen requirement

P3-A requires the same model and CIFAR-10 source to have both clean/base output and all 16
real-corruption outputs, for H0/H9/L0/L9 and four clients. Only then is
`d = P_C(z_corrupt - z_clean)` identified without new inference.

## What exists

| Asset | Coverage | Why it is or is not sufficient |
| --- | --- | --- |
| Phase-A1a round-40 predictions | H0/H9/L0/L9 x 4 clients, corruptions only | Existing DSA source, but no clean/base output |
| K1-C-Minimal frozen oracle files | H9/L9, clients 0 and 3 | Valid paired clean/corrupt probabilities, but only 4/16 contexts and no H0/L0 controls |
| K0-A response files | Four arms x four clients, base + 16 real operators | CIFAR-100 public carriers, not the CIFAR-10 sources/labels used by DSA |
| K0-B response files | Four arms x four clients, base + 128 PRIME views | Valid routing-design source, but generic probes are not the real-corruption DSA grid |
| Phase-B0 PNCB predictions | Four arms x four clients, real corruptions | Original/overlay/canonical probabilities, but no clean/base output |

All saved probabilities inspected here are finite and strictly positive. Therefore pre-softmax
centered logits are recoverable as `log(p) - mean_c log(p)` where both clean and corrupted views
exist. This does not create a missing clean view. The maximum difference between K1-C-Minimal's
saved frozen corrupted probabilities and the aligned Phase-A1a entries is `7.153e-07`.

## Coverage decision

- Required matched contexts: 16.
- Available matched contexts: 4/16.
- Available no-CLE controls: 0/8.
- Functionally unique matched contexts: at most 3, because H9/L9 client0 is duplicated.

Verdict: `INSUFFICIENT_EXISTING_ARTIFACTS`.

No permutation, counterfactual output, DSA null, model inference or OpenI job was produced.
