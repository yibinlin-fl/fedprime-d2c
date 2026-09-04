# P1 Class-Readout Audit Artifact Availability

Date: 2026-09-04

## Available and used

- K1-C-Minimal saved, per system/client/arm/half:
  - 64-by-feature-dimensional mean response matrices;
  - 64 response energies;
  - aligned 64-by-64 normalized response Gram matrices.
- The original Phase-B0 input contains the exact H9/L9 client0/client3 checkpoints used by
  K1-C-Minimal.
- Every checkpoint SHA256 matches `checkpoint_manifest.json`.
- Every extracted `linear.weight`/`linear.bias` hash matches the classifier hash sealed inside both
  CRSF and RawSpec block-delta artifacts.
- The intervention deltas contain no `linear.*` parameter. The classifier is therefore unchanged
  across Frozen/CRSF/RawSpec, while the saved representation responses change.
- ResNet10 uses a single `linear.weight` with shape `10 x 512`; MobileNetV2 uses a single
  `linear.weight` with shape `10 x 1280`. These dimensions exactly match the saved feature-response
  dimensions. No architectural approximation was needed.
- K1-C-Minimal saved per-client DSA values were used as the outcome. No prediction or model inference
  was repeated.

## Definitions recoverable from existing artifacts

The saved K1-C response Gram is generated from the normalized mean response

```text
S[:, q] = mean_response[q] / sqrt(mean_squared_response_energy[q]).
```

P1 reconstructed every saved Gram as `S.T @ S`; maximum absolute error was
`9.55e-13`. With the centered classifier `Wc = W - row_mean(W)`, the audit can therefore compute:

- mode/readout coupling `Wc @ U`;
- exploratory component energy `lambda_j * ||Wc u_j||^2`;
- normalized class/probe routing `Wc @ S`;
- raw expected class/probe routing `Wc @ mean_response.T`.

The classifier bias cancels from a response difference and is recorded but not used.

## Missing and non-identifiable from current files

- K1-C0 contains Gram/eigen information but not its feature-space mean response matrix. Its left
  singular vectors `U` and hence `Wc @ U` cannot be recovered from the Gram alone.
- K1-C-Minimal contains only H9/L9, client0/client3, A-to-B and seed 0. There is no H0/L0
  readout-weighted baseline, B-to-A replication, remaining-client coverage or multi-seed evidence.
- Only per-probe mean responses are saved. Per-carrier feature responses, earlier-layer responses and
  input-dependent routing cannot be reconstructed.
- `weighted_lambda_j` is a mode-additive readout energy, not generally an eigenvalue of the routing
  Gram because `Wc u_j` need not be mutually orthogonal. P1 separately reports the true singular
  spectrum of `Wc @ S`.
- H9/client0 and L9/client0 are functionally identical in the saved responses and predictions and
  share the same classifier hash. They remain separate provenance rows but are not independent
  ResNet10 evidence.

No missing item was regenerated. No OpenI job, GPU, PRIME generation, model forward/backward pass,
training or checkpoint update occurred.
