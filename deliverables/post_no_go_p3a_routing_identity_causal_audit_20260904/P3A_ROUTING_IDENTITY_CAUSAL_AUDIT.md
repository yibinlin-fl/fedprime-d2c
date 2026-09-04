# P3-A Matched Routing-Identity Causal Audit

Date: 2026-09-04

## Verdict

```text
INSUFFICIENT_EXISTING_ARTIFACTS
status: STOP_BEFORE_PERMUTATION_OR_ORACLE_EVALUATION
```

The frozen Stage-0 availability gate failed. Existing artifacts do not contain a complete matched
clean-to-real-corruption response grid for H0/H9/L0/L9 and all four clients. Consequently P3-A did
not generate the targeted rank-reversal permutation, the 1,000 random derangements, counterfactual
logits, invariance tables or DSA nulls.

## Why the missing base matters

Phase-A1a stores strictly positive softmax probabilities for all real-corruption views, so each
view's centered logits can be recovered. It does not store the clean output for the same source.
Without it, `P_C(z_corrupt-z_clean)` is not identified. Permuting the full corrupted logit would also
permute semantic class evidence; using the mean corruption view as a surrogate base would define a
different intervention. Neither substitution is allowed after the P3-A contract was frozen.

K0-A cannot fill the gap: its base logits belong to different CIFAR-100 public carriers. K1-C-Minimal
does contain paired clean/corrupt probabilities, but only for H9/L9 clients 0 and 3, with no H0/L0
control and at most three functionally unique contexts. That cannot satisfy the 3/4 direction gate
or the no-CLE safety gate.

## Scientific consequence

P2 remains valid as observational evidence for CLE-specific stable class-visible routing. P3-A has
neither passed nor failed scientifically; its causal estimand is untestable from the complete current
artifact set. The frozen verdict explicitly forbids automatically launching new inference.

If the user later considers filling the gap, that must be a separately costed, inference-only data
export protocol that saves clean logits for the exact 1,000 CIFAR-10 sources and 16 existing
round-40 checkpoints. It is not authorized by this audit.
