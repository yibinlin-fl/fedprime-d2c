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
