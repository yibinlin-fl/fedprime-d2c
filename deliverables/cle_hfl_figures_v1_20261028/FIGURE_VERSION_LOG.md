# CLE-HFL figure version log

This workbench never overwrites the figures stored in an immutable manuscript
snapshot. The current paper V0.10 therefore retains
`fig1_cle_hfl_evidence_chain.*` and its original generator.

## Figure 1: CLE scene and paired DSA

- `fig_cle_paired_dsa_v1.*`: first rendered draft. It exposed a Matplotlib
  aspect-ratio failure and is retained only as a provenance snapshot.
- `fig_cle_paired_dsa_v2.*`: fixed the image-axis compression and established
  the three-panel scientific structure.
- `fig_cle_paired_dsa_v3.*`: repaired operator-label collisions, confined the
  central explanation to its panel, and separated the three information roles
  at final figure scale. This is the current acceptance candidate.
- `figures/generate_cle_paired_dsa.py`: deterministic Python source for the
  current `v3` candidate. It reads the frozen class--operator map and real
  source-paired images from the local evaluation package; it invents no
  prediction values.

Each revision retains PDF, SVG, and PNG artifacts. A later repair must write a
new suffix rather than replace `v3`. Once accepted, the PDF will be copied into
the next manuscript snapshot; V0.10 and all workbench revisions remain intact.

## Figure 2: PEW+BER execution

- `figures/fig_pew_ber_execution.pdf/.svg/.png` and its generator remain the
  existing first candidate. A future repair uses a new explicit suffix and
  preserves these original unsuffixed files.

## Figure 3: Local-first matched training seeds

- `figures/fig_local_first_multiseed_v1.pdf/.svg/.png`: current candidate,
  generated from the frozen three-seed table on 2026-10-28. First-pass visual
  review found no overlaps or clipping.
- Same-stem CSV retains the exact plotted seed rows and original gate status;
  JSON retains input/output hashes, runtime versions, and interpretation scope.
- `figures/generate_local_first_multiseed.py`: standalone generator, verifies
  communication differences and mean/sample SD, and refuses revision overwrite.
- `.vivid/template-sources/c81322a77c40cac07067eb77.py/.json`: original recipe
  and extraction record for `advanced.dumbbell`.
- `FIGURE_3_REVIEW_ZH.md`: data semantics, uncertainty scope, visual review,
  reproduction instructions, and suggested integration caption.

V0.10 and its old Figure 3 remain unchanged. Next are Figure 4 objective/utility
comparison and Figure 5 dose response.
