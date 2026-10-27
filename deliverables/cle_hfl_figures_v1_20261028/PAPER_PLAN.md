# CLE-HFL bounded five-figure redesign

This workbench redesigns the five manuscript figures already frozen by the
RAHFL visual audit. It does not expand the scientific claims or invent new
results. V0.10 remains the stable manuscript until a figure is visually
accepted.

<!-- BEGIN FIGURE_MANIFEST -->
## Figure manifest

**Deterministic technical diagrams:**

- `fig_pew_ber_execution` | claim=PEW supplies coarse public-taxonomy side information and BER changes the local class-conditional risk while the HFL communication protocol remains unchanged | source=V0.10 Method and frozen BER equations | section=Taxonomy-Assisted Local Mitigation | language=en | format=py,pdf,svg,png | layout=double-column landscape

**Data figures:**

- `fig_local_first_multiseed` | claim=the Local CLE effect tracks the HFL effect across three training seeds while the communication difference changes sign | source=deliverables/cle_v2_local_first_multiseed_20261026/PAPER_TABLE.csv | section=Local-versus-Communication Formation | language=en | format=py,pdf,svg,png | layout=double-column landscape
- `fig_objective_tradeoff` | claim=BER separates shortcut suppression from task utility relative to ERM, CVaR-DRO, and shared-PEW GroupDRO | source=deliverables/cle_v2_spurious_final_map2_20260913/RESULT_SUMMARY_ZH.md | section=Axis I | language=en | format=py,pdf,svg,png | layout=double-column landscape
- `fig_dose_response` | claim=DSA increases with controlled binding strength and PEW+BER reduces the gamma=.9 response with a nonuniform utility trade-off | source=deliverables/cle_v2_moderate_axis2_dose_formal_seed0_20261025/PAPER_TABLE.csv | section=Main Evaluation Protocol | language=en | format=py,pdf,svg,png | layout=double-column landscape

**Deterministic diagnosis diagram:**

- `fig_cle_paired_dsa_v3` | claim=client-specific class--operator bindings turn corruption into a directional class cue measurable by same-source paired DSA | source=V0.10 Problem Setup, frozen package metadata, and paired operator grid | section=Problem Setup and Diagnosis | language=en | format=py,pdf,svg,png | layout=double-column landscape

**Total:** DATA=3, DETERMINISTIC=2, ILLUSTRATION=0, ALL=5
<!-- END FIGURE_MANIFEST -->

## Visual contract

- White background; no gradients, shadows, glow, 3-D, cartoons, or decorative cards.
- Real CIFAR thumbnails may illustrate data flow, but never serve as evidence.
- Teal denotes the frozen public witness; rust denotes BER; gray denotes the
  unchanged HFL path and evaluation-only metadata.
- Every numerical plot reads frozen CSV/JSON/report artifacts.
- Each figure answers one research question and remains legible at two-column width.

## Versioning and integration

- Historical paper figures remain immutable. In particular, V0.10's
  `fig1_cle_hfl_evidence_chain.*` and its generator are never overwritten.
- Redesigned figures use semantic names plus revision suffixes, for example
  `fig_cle_paired_dsa_v1.*`; a repair becomes `v2`, then `v3`.
- Every figure keeps its Python generator beside PDF/SVG/PNG outputs. PDF is
  used by LaTeX, SVG remains editable, and PNG is the visual-review copy.
- The manuscript changes only after visual acceptance. The accepted PDF is
  copied into the next paper snapshot and one LaTeX include is updated; old
  paper snapshots and figure artifacts remain available for comparison.

