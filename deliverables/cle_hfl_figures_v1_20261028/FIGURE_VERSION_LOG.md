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

V0.10 and its old Figure 3 remain unchanged.

## Figure 4: Axis-I objective/utility comparison

- `fig_objective_tradeoff_v1.*`: first render, all four arms and three seeds.
- `fig_objective_tradeoff_v2.*`: current candidate; moved the BER label away
  from a nearby seed point. The data and summary statistics are unchanged.
- `generate_objective_tradeoff.py`: maintained generator, reads three Formal
  analysis JSON files (or its exported CSV) and cross-checks frozen aggregates.
- Each revision includes PDF/SVG/PNG, the 12 exact seed rows in CSV, a provenance
  JSON, and the exact generator snapshot in a same-stem `.py` file.
- Original recipe source/record: `.vivid/template-sources/0adda4e27f37b6544095adc9.py/.json`.

## Figure 5: Dose response and task-utility changes

- `fig_dose_response_v1.*`: three measured ERM doses, one BER point and six
  utility endpoints; two labels were close to the response line.
- `fig_dose_response_v2.*`: moved those labels clear of the line.
- `fig_dose_response_v3.*`: current candidate, brings the zero-dose label closer
  to its point while preserving clear separation. All numbers stay unchanged.
- `generate_dose_response.py`: maintained generator; reads the frozen main
  evaluation CSV and computes all contrasts, including clean/mixed declines.
- Each revision includes PDF/SVG/PNG/CSV/JSON and its generator `.py` snapshot.
- Original recipe source/record: `.vivid/template-sources/6af7ff40fd40ae7f1f50d7af.py/.json`.

Both use `figure_common.py`; its SHA256 is recorded in every revision manifest.
Existing revisions are protected against overwrite. Review and reproduction:
`FIGURES_4_5_REVIEW_ZH.md`; all-five index: `README_ZH.md`.
All five figures reached independent candidate status before integration.

## Five-figure integration review, 2026-10-28

- Figure 1 v4 explicitly conditions the training draw on corruption C=1 and
  identifies valid DSA target pairs y not in B. PDF/SVG/PNG/Python snapshot kept.
- Figure 2 v2 adds all six implemented pseudo-environments (clean, noise, blur,
  weather, digital, unknown), labels support cells schematic, and matches the
  valid-group/class BER rule. v3 routes the public-training/frozen-inference
  arrow clear of the private-panel heading. Original/v2/v3 all retained.
- Integration selection: Figure 1 v4; Figure 2 v3; Figure 3 v1; Figure 4 v2;
  Figure 5 v3. V0.10 was copied to a separate V0.11 snapshot before editing.
- Actual manuscript pages: 4, 6, 8, 9, 10. Five figures and all 16 pages were
  rendered and visually inspected. Captions distinguish schematic content,
  measured points, seed SD, and the two frozen experimental budgets.
- V0.11 remains an internal review draft with pending evidence tables, rather
  than a final page-limited conference submission. See its PDF_QA_ZH.md.

## Detailed re-audit, 2026-10-30 (no new figure revision yet)

User requested closer inspection of apparent skew and a less templated visual style.
All five standalone PNGs and the Figure-2 manuscript page were opened again.
Measured aspect distortion: public Figure-2 tiles +19.95/+20.78% width, private
tiles +3.53%, Figure-1 paired tiles -2.82%. No explicit image rotation found.
Figure-2 arrows also imply BER as pixel preprocessing; a future revision must
show task-model losses and the BER risk/update branch, and qualify the public
communication inputs. The earlier clipping check missed these issues.
Figures 3–5 need focused typography/marker/annotation refinements rather than
new data or elaborate chart types. See VISUAL_REAUDIT_20261030_ZH.md and the
geometry JSON/script. All current figure and manuscript artifacts are unchanged.
