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

- `fig_local_first_multiseed_v1` | claim=the Local CLE effect tracks the HFL effect across three training seeds while the communication difference changes sign | source=deliverables/cle_v2_local_first_multiseed_20261026/PAPER_TABLE.csv | recipe=advanced.dumbbell | section=Local-versus-Communication Formation | language=en | format=py,pdf,svg,png,csv,json | layout=double-column landscape
- `fig_objective_tradeoff_v2` | claim=BER separates shortcut suppression from task utility relative to ERM, CVaR-DRO, and shared-PEW GroupDRO across three training seeds | source=three Formal seed summaries checked against deliverables/cle_hfl_map2_multiseed_20260922/MULTISEED_SUMMARY.json | recipe=academic.error_analysis | section=Axis I | language=en | format=py,pdf,svg,png,csv,json | layout=double-column landscape
- `fig_dose_response_v3` | claim=DSA increases with measured controlled binding strength and PEW+BER reduces the gamma=.9 response with a nonuniform utility trade-off | source=deliverables/cle_v2_moderate_axis2_dose_formal_seed0_20261025/PAPER_TABLE.csv | recipe=basic.line | section=Main Evaluation Protocol | language=en | format=py,pdf,svg,png,csv,json | layout=double-column landscape

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

## Figure 3 candidate, 2026-10-28

`fig_local_first_multiseed_v1` is generated and visually reviewed. It retains
the paired endpoints of recipe `advanced.dumbbell`; the right panel adds the
signed communication difference on a labelled magnified scale. The alternative
`advanced.paired_dot` preview was also inspected. Native size is 7.15 x 3.38
inches; expected 6.7-inch inclusion gives 0.94x scaling. Bottom statistics are
mean and sample SD across three training seeds. Data and review details are
in `FIGURE_3_REVIEW_ZH.md`.

## Figures 4 and 5 candidates, 2026-10-28

Figure 4 uses all 12 seed-level points plus four means and axis-aligned sample
SD bars. The scatter template was compared with the earlier separate-endpoint
plot: the joint axes directly expose shortcut--utility trade-offs. Unsupported
bubble-size, efficiency and interpolated Pareto layers are omitted. Current
candidate v2 resolves a label/seed-point proximity in v1.

Figure 5 combines the three measured ERM dose points and one matched BER
point with all six observed utility differences. This provides complementary
information that the former protocol flowchart could not show. No fabricated
uncertainty bands or unmeasured BER dose curve are added. Candidate v3 improves
point-label placement; v1/v2 remain available.

Their native widths are 7.15 inches, with heights 3.75/3.60 inches. Both were
visually reviewed and remain separate from V0.10. See
`FIGURES_4_5_REVIEW_ZH.md` for data checks, adaptations, and captions. The five
current candidates are indexed in `README_ZH.md`.

