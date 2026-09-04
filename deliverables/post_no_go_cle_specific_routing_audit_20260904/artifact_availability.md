# P2 Artifact Availability

Date: 2026-09-04

## Available and used before sealing

- Complete K0-B response grid: H0/H9/L0/L9 x 4 clients = 16 files.
- Each file contains carrier-level `centered_response` with shape `1000 x 128 x 10`, plus base/probe logits.
- Probe order is Bank A recipes 0--63 followed by Bank B recipes 64--127.
- Carrier halves are Ua indices 0--499 and Ub indices 500--999, certified disjoint.
- All 16 response SHA256 values match the K0-B blind manifest.
- All checkpoint hashes match the Phase-B0 manifest, which identifies them as final round-40 only.
- The blind manifest certifies no public labels, corruption taxonomy, severity, binding, or private corruption metadata were used.

## Deferred until after taxonomy-free seal

- Phase-A1a round-40 per-client DSA for H0/H9/L0/L9.
- Original K0-B `R_i` from its saved per-client metrics.

The deferred files are not opened by `taxonomy_free_stage`; they are loaded only after the four
primary CSV files and `primary_taxonomy_free_manifest.json` have been written and hashed.

## Not used

- No checkpoint loading or model construction.
- No model forward/backward pass.
- No PRIME recipe generation.
- No GPU/OpenI/training/optimization.
- No corruption binding, family, severity, or CLE oracle is used to construct B--E statistics.
