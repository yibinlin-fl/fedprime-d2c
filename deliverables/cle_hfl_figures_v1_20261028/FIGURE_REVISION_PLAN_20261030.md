# Bounded revision: Figures 1--5

The initial diagram repair covered Figures 2 and 1. The author subsequently
requested completion of Figure 3 plus refinement of Figures 4 and 5. The five
entries below are independent candidates; the V0.11 paper remains unchanged.

Preserve V0.11 and every existing image. Use TikZ for formula-bearing technical
diagrams, Python only for extracting unchanged real thumbnails and data labels.
Reference review: RAHFL Fig.3 (p.5), AugHFL Fig.3 (p.4), FCCL Fig.2 (p.4).
No in-figure headline, decorative neural icon, gradient, or new empirical value.

<!-- BEGIN FIGURE_MANIFEST -->
## Figure manifest

**TikZ exact technical diagrams:**

- tikz_pew_ber_execution_v7 | claim=Frozen public PEW supplies groups while BER aggregates task-model losses and updates local parameters; only public predictions enter the existing communication interface | source=V0.11 Method and frozen PEW manifest; real public/private thumbnails | section=Taxonomy-Assisted Local Mitigation | language=en | format=tex,pdf,png,svg,json | layout=double-column landscape
- tikz_cle_paired_dsa_v7 | claim=Client-specific training bindings define the direction for same-source cross-operator evaluation with valid target pairs | source=V0.11 DSA definition and frozen paired grid; exact client binding metadata | section=Problem Setup and Diagnosis | language=en | format=tex,pdf,png,svg,json | layout=double-column landscape

**Data figures:**

- fig_local_first_multiseed_v2 | claim=Local CLE effects track HFL effects across matched training seeds while communication contrasts change sign | source=deliverables/cle_v2_local_first_multiseed_20261026/PAPER_TABLE.csv | section=Local-versus-Communication Formation | language=en | format=py,pdf,png,svg,csv,json | layout=double-column landscape
- fig_objective_tradeoff_v3 | claim=Four objectives show different DSA--utility trade-offs over three training seeds under the same communication | source=three frozen Formal seed summaries and MULTISEED_SUMMARY.json | section=Axis I | language=en | format=py,pdf,png,svg,csv,json | layout=double-column landscape
- fig_dose_response_v4 | claim=DSA rises across three measured ERM doses and matched BER changes shortcut use and all six utility endpoints | source=deliverables/cle_v2_moderate_axis2_dose_formal_seed0_20261025/PAPER_TABLE.csv | section=Main Evaluation Protocol | language=en | format=py,pdf,png,svg,csv,json | layout=double-column landscape

**Total:** TIKZ=2, DATA=3, ALL=5
<!-- END FIGURE_MANIFEST -->

Check native/page-scale render, square image geometry, arrow meaning, label
clearance, and all formula/data labels. If repaired, increment the suffix and
preserve the failed candidate as well. Integration into a manuscript is pending
author review; do not create a new paper version in this bounded revision.
