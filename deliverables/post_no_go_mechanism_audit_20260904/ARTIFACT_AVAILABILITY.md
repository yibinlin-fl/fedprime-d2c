# Post-NO-GO P0 Artifact Availability

Date: 2026-09-04

This audit used only files already returned by completed experiments. It did not load a checkpoint,
generate a PRIME view, run a model forward/backward pass, use a GPU, or modify model parameters.

## Available artifacts

| Stage | Existing material | What P0 can recover | Important limitation |
| --- | --- | --- | --- |
| K0-B | 16 frozen-client response archives with base logits, 128 probe logits, class-vs-rest deltas and centered responses; both frozen PRIME bank states/manifests; bootstrap and per-client metrics | Provenance for the taxonomy-free detector and confirmation that raw response tensors exist | No intervention outputs; its 1,000-carrier pool is not sample-aligned to the K1-C-Minimal holdout |
| K1-B0 | `result.json`, per-client table, bootstrap arrays, selection/matching manifest and inspection records | The already-decided shared-routing NO-GO and frozen probe-selection provenance | No saved raw penultimate feature tensor or response Gram; a new subspace autopsy is `MISSING` and was not regenerated |
| K1-C0 | Response/clean Gram matrices, response/clean eigenvalues, spectrum tables, bootstrap arrays and the 10/10 gate result | Observational response-spectrum concentration and its cross-bank statistics | No intervention prediction cache or CLE DSA cache; it cannot by itself identify a causal shortcut reduction |
| K1-C-Minimal | Six frozen/CRSF/RawSpec prediction archives; labels, binding and operator-family ids; full Bank-B moments/Gram; taxonomy-free metrics; task metrics; eight optimization traces and block deltas | All requested P0 class-routing, chi--DSA, spectral and architecture analyses for H9/L9 clients 0 and 3 | Seed 0, A-to-B only, two architectures and two selected clients; it is not a multi-seed or all-client estimate |

## Coverage decisions

- Class-wise and corruption-family-binding-wise DSA are available from K1-C-Minimal and were
  reconstructed additively. Each family sum was checked against the frozen DSA implementation to
  `1e-12` tolerance.
- Operator-level probabilities are available, but explicit operator names are not stored in the
  prediction archives. P0 therefore reports class/family-binding components and does not infer names
  from code or regenerate predictions.
- Saved 64-by-64 Bank-B Gram matrices are aligned across Frozen/CRSF/RawSpec, so eigenvalues and
  principal-eigenvector cosines are available without features or inference.
- Per-client CFG and WCCA are recoverable from the saved prediction tensors. No accuracy or routing
  labels were used to select the original intervention.
- K1-B0 feature-level post-hoc localization and sample-wise K0-B-to-K1-C alignment remain
  `MISSING`. They are not required for the four completed P0 audits and do not authorize a rerun.

## Independence caveat

H9/client0 and L9/client0 have different checkpoint file hashes, but their saved K0-B responses and
all three K1-C-Minimal prediction tensors are exactly equal. Consequently, the two ResNet10 rows are
not independent functional replications. P0 keeps both provenance labels in the tables but does not
double-count them as independent scientific evidence.
