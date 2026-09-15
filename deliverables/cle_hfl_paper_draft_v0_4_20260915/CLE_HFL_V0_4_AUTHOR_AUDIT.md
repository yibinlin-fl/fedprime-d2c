# CLE-HFL V0.4 Author Audit

> **Purpose.** This file is an internal author-side control document for Version 0.4. It is not part of the conference manuscript. It preserves evidence mapping, the frozen candidate experiment matrix, overclaim checks, and all unresolved evidence placeholders. A row appearing in the candidate matrix does **not** authorize, launch, or imply that an experiment is running.

## 1. Frozen Scientific Story

The manuscript must preserve one evidence chain:

**Problem discovery** → client-specific class–corruption binding can make corruption a directional class cue in model-heterogeneous FL.

**Shortcut diagnosis** → source-paired operator interventions and **Directional Shortcut Alignment (DSA)** test whether prediction probability moves toward classes associated with the applied operator during training.

**Formation attribution** → matched HFL-versus-Local factorial indicates predominantly local-first formation in the controlled setting.

**Targeted mitigation** → a coarse family-level, taxonomy-assisted **Public Environment Witness (PEW)** supplies pseudo-environment side information; **Balanced Environment Risk (BER)** compresses class-conditional pseudo-environment support imbalance.

**Controlled validation** → matched ERM, protocol-matched CVaR-DRO, shared-PEW GroupDRO, cross-binding-map replication, and a second protocol-matched communication base test alternative explanations.

**Theory boundary** → proxy non-identifiability shows why PEW/JSD/proxy improvement cannot substitute for target-aligned DSA evaluation without additional assumptions.

## 2. Frozen Fact Boundaries

| Topic | Frozen boundary |
|---|---|
| CLE-v2 granularity | Data generation, client/class binding, and DSA evaluation are operator-level. |
| PEW granularity | Final PEW is coarse family-level only. No operator-level PEW is proposed or implemented. |
| Oracle operator | Non-deployable granularity/correspondence audit only. |
| Taxonomy | PEW is taxonomy-assisted, not taxonomy-free. |
| DSA meaning | Binding-direction operator-response contrast under semantic preservation, valid intervention, pre-registered binding, source pairing, and evaluation isolation; not complete internal causal recovery. |
| FedDF-fidelity | Protocol-matched communication-base control; not a line-by-line official FedDF reproduction. |
| CVaR-DRO | Protocol-matched empirical tail-risk control; not claimed as a line-by-line reproduction of a cited algorithm. |
| JTT | 12-round screen only; no Formal win/loss claim. |
| Source bootstrap | Evaluation-source uncertainty conditional on fixed checkpoints; not training-seed uncertainty. |
| FedCD | arXiv preprint unless a later formal version is newly verified. |
| Method scope | No new loss, communication module, hierarchical PEW, or operator-level PEW. |
| Real-world scope | Controlled CIFAR-C-style evidence; no hospital, vehicle, factory, or industrial deployment claim. |

## 3. Contribution-to-Evidence Mapping

| Contribution | Evidence already available | Required boundary | Unresolved evidence |
|---|---|---|---|
| **C1. CLE-HFL formulation** | Client-specific class–operator binding; \(\gamma_{\mathrm{CLE}}=0\) and \(0.9\); operator-level paired grid; strict data-role isolation | Controlled synthetic CLE, not deployment evidence | **[EVIDENCE NEEDED]** second private dataset; dataset and protocol pending |
| **C2. Paired DSA + identification + proxy non-identifiability** | DSA definition; shuffled-binding null; exchangeable zero; affine property; paired cancellation; known-response recovery; JSD=0/DSA>0; source-level bootstrap; two-world non-identifiability theorem | DSA is target-aligned under assumptions, not a universal causal sufficient statistic | No new theory module required; proof polishing only |
| **C3. Local-first attribution** | Seed-0 HFL CLE effect \(0.119889\); Local effect \(0.107960\); communication add-on \(0.011929\); descriptive ratio \(90.05\%\) | Not per-example mediation; source CI does not cover training randomness | **[EVIDENCE NEEDED]** multi-training-seed stability for the same four-arm factorial |
| **C4. Taxonomy-assisted PEW+BER + controlled validation** | Original-map DSA reduction \(65.52\%\); map1 \(56.46\%\); FedDF-fidelity \(78.82\%\); held-out map2 matched ERM/CVaR/PEW+GroupDRO/PEW+BER; Oracle correspondence/granularity; BER effective-distribution audit | Coarse family PEW; mixed utility; FedDF utility gate fails; no per-client CVaR domination | **[EVIDENCE NEEDED]** multi-seed map2 control stability; second private dataset; bounded taxonomy stress |

## 4. Frozen Candidate Experiment Matrix

**Important:** The rows below are planning candidates only. They are not authorized runs, not in progress, and not evidence until a separate decision freezes and launches them.

| Research question | Experimental arms | Fixed conditions | Variable | Primary metrics | Candidate seeds | Current status | Can establish | Cannot establish | Priority |
|---|---|---|---|---|---|---|---|---|---|
| **RQ1 + RQ2: Does CLE form, and is it local-first?** | HFL \(\gamma=0\), HFL \(\gamma=.9\), Local \(\gamma=0\), Local \(\gamma=.9\) | CLE-v2 protocol, same architectures, same partition, same local recipe, same evaluation grid | Federation on/off × CLE strength | DSA; \(\Delta_{\mathrm{HFL}}\); \(\Delta_{\mathrm{Local}}\); \(\Delta_{\mathrm{comm}}\); operator-grid accuracy | S0 exists; S1/S2 are candidates | S0 Formal complete; S1/S2 **not authorized** | Whether strong CLE and local-first pattern are stable across training randomness if repeated | Per-example mediation; all communication protocols; new partitions/datasets | **P0 candidate** |
| **RQ3/RQ4: Is BER more than ERM/group/tail-risk reweighting?** | ERM, CVaR-DRO, PEW+GroupDRO, PEW+BER | held-out map2; same partition, initial states, batch trajectory, strict HFL protocol, 40 rounds | Risk objective | DSA; grid Acc; Avg; Worst; WCCA; CFG | S0 exists; S1/S2 are candidates | S0 Formal complete; S1/S2 **not authorized** | Stability of matched BER vs shared-PEW GroupDRO and pooled BER-vs-CVaR trade-off if repeated | Per-client uniform dominance; official CVaR algorithm superiority | **P0 candidate** |
| **RQ5a: Is the effect tied to one binding map?** | Base vs PEW+BER on map1 | partition, training/eval seed, initialization, public data, grid, frozen PEW | Binding map | DSA + utility | S0 | Complete | Replication on two controlled binding directions | Cross-seed, cross-partition, cross-domain generalization | Done |
| **RQ5b: Does suppression appear under another HFL communication base?** | Native CE FedDF-fidelity vs PEW+BER + same FedDF-fidelity | init, private batches, public distillation, budget | Local task risk within second communication base | DSA + utility gate | S0 | Complete | Suppression effect under a second protocol-matched communication base | Lossless plugin; consistent utility gain; official FedDF reproduction | Done |
| **RQ6: Does environment correspondence/granularity matter?** | Oracle family, Oracle operator, Random operator | same strong-CLE setting | grouping correspondence/granularity | DSA + utility | S0 | Complete | Correct correspondence matters; operator vs family gain is small in this audit | Universal uselessness of operator-level methods | Done |
| **External validity: second private dataset** | Protocol pending | **Not frozen** | private task/data family | DSA, local-first contrast, mitigation, utility | Not frozen | **[EVIDENCE NEEDED: second private dataset; dataset and protocol pending.]** **Not authorized** | Broader task-family evidence once a dataset/protocol is frozen and run | Real-world deployment or arbitrary dataset generalization | **P0 candidate** |
| **Taxonomy boundary** | Protocol pending | Method must remain coarse family-level PEW+BER | bounded mismatch between PEW training taxonomy/coverage and evaluation corruption | PEW family error, DSA, utility | Not frozen | **[EVIDENCE NEEDED: bounded taxonomy stress test; protocol pending.]** **Not authorized** | Sensitivity to bounded taxonomy mismatch after a protocol is frozen | Arbitrary open-world robustness | **P1 candidate** |
| **JTT Formal** | Only if future manuscript makes an empirical JTT superiority claim | Would require new frozen matched protocol | JTT vs selected controls | DSA + utility | Not frozen | Current JTT is screen only; **not authorized** | Formal JTT comparison if explicitly approved later | Nothing at present | Not required |

### 4.1 Multi-seed non-duplication rule

Do **not** create separate RQ1 and RQ2 multi-seed jobs. One candidate HFL-versus-Local four-arm repetition at S1/S2 would jointly cover:

- HFL \(\gamma=0\) vs. \(\gamma=.9\);
- Local \(\gamma=0\) vs. \(\gamma=.9\);
- stability of \(\Delta_{\mathrm{HFL}}\), \(\Delta_{\mathrm{Local}}\), and \(\Delta_{\mathrm{comm}}\);
- stability of the descriptive local-first interpretation.

This remains a candidate matrix only and does not authorize execution.

## 5. All Current [EVIDENCE NEEDED] Placeholders

The internal manuscript must preserve these placeholders until real frozen results exist:

1. **[EVIDENCE NEEDED: multi-training-seed stability for the matched HFL-versus-Local four-arm factorial.]**
   Candidate S1/S2 repetitions only; not authorized or running.

2. **[EVIDENCE NEEDED: multi-training-seed stability for the held-out map2 four-arm comparison.]**
   Candidate S1/S2 repetitions only; not authorized or running.

3. **[EVIDENCE NEEDED: second private dataset; dataset and protocol pending.]**
   No dataset is frozen. SVHN, CIFAR-family alternatives, or any other dataset must not be named in V0.4 as the chosen dataset.

4. **[EVIDENCE NEEDED: bounded taxonomy stress test; protocol pending.]**
   No held-out-operator, compound-corruption, or other stress protocol is frozen or authorized.

No old experiment containing CDep or another removed component may be used to fill a pure PEW+BER placeholder.

## 6. Overclaim Audit

| Avoid / attack surface | Why risky | Safe wording for V0.4 |
|---|---|---|
| “We are the first to study spurious correlation in FL.” | Prior federated shortcut work exists. | “We isolate a specific class–corruption directional shortcut in model-heterogeneous FL.” |
| “PEW discovers latent environments without supervision.” | PEW uses a predefined corruption-family taxonomy. | “PEW is a taxonomy-assisted coarse corruption-family witness trained on public synthetic corruptions.” |
| “DSA causally identifies the model’s shortcut mechanism.” | DSA identifies a narrower response contrast under assumptions. | “Under stated semantic-preservation, intervention, pre-registration, pairing, and isolation assumptions, DSA identifies a binding-direction operator-response contrast.” |
| “Counterfactual evaluation proves the causal mechanism.” | “Counterfactual” does not automatically imply unconditional causal identification. | “The source-paired operator intervention provides a controlled directional diagnostic under stated assumptions.” |
| “JSD invariance guarantees no shortcut.” | JSD can be zero while DSA is positive. | “Within-operator consistency does not imply cross-operator binding-direction invariance.” |
| “Proxy improvement proves true CLE improvement.” | Proxy-only non-identifiability. | “Without assumptions on the proxy error channel, proxy improvement alone cannot certify reduced true CLE dependence.” |
| “BER guarantees \(Y\perp E\).” | BER directly controls proxy effective distribution; current true-environment bound is trivial. | “BER compresses class–proxy-environment support imbalance under its effective distribution.” |
| “The shortcut is communication-induced.” | HFL and Local effects are close; communication add-on is smaller. | “The pooled shortcut is predominantly local-first in the evaluated controlled setting.” |
| “90.05% is a causal share.” | It is a ratio of matched contrasts, not mediation. | “At seed 0, the Local contrast is 90.05% of the HFL contrast descriptively.” |
| “This mechanism result changes the intervention target.” | Too causal/deterministic. | “This observation motivates targeting the local objective.” |
| “PEW side information explains BER.” | Shared-PEW GroupDRO does not reproduce BER, but optimizer/objective differs. | “Under the matched protocol, PEW side information alone is insufficient to reproduce BER’s lower pooled DSA.” |
| “BER uniformly beats CVaR-DRO.” | c0/c1 have lower DSA under CVaR. | “On held-out map2, BER has a better pooled shortcut–utility trade-off, without per-client DSA uniform dominance.” |
| “The method generalizes across HFL bases.” | FedDF utility gate fails; ‘generalizes’ can be read broadly. | “The DSA-suppression effect is also observed under a second protocol-matched HFL communication base, although its utility gate fails.” |
| “FedDF confirms cross-base success.” | Misstates failed utility gate and fidelity status. | “FedDF-fidelity provides supporting evidence for suppression under a second communication base, not consistent utility improvement.” |
| “FedDF-fidelity reproduces FedDF.” | It is only a protocol-matched adapter. | “protocol-matched FedDF-fidelity communication adapter.” |
| “Our CVaR-DRO reproduces a standard CVaR algorithm.” | It is a protocol-matched empirical control. | “protocol-matched CVaR-DRO tail-risk control.” |
| “We outperform JTT.” | Only a 12-round screen exists. | Do not make this claim. |
| “Cross-map proves scenario generalization.” | Only two controlled maps, same partition/seed. | “The reduction replicates across two controlled binding maps at seed 0.” |
| “Source CI shows training stability.” | Source bootstrap conditions on a fixed checkpoint. | “The source-bootstrap interval quantifies evaluation-source uncertainty only.” |
| “Operator-level PEW is unnecessary.” | Oracle result is local to current protocol. | “The current Oracle granularity audit does not justify developing operator-level PEW for this paper.” |
| “CLE is validated in hospitals/vehicles/factories.” | No real deployment data. | “CLE is a controlled abstraction; deployment effectiveness remains untested.” |
| “Second dataset will be SVHN.” | Dataset/protocol is not frozen. | **[EVIDENCE NEEDED: second private dataset; dataset and protocol pending.]** |
| “Multi-seed experiments are running.” | No authorization has been given. | “Multi-seed repetitions are candidate experiments in the frozen planning matrix only.” |
| “Taxonomy stress test is leave-one-operator-out.” | No protocol is frozen. | **[EVIDENCE NEEDED: bounded taxonomy stress test; protocol pending.]** |

## 7. Citation and Implementation-Fidelity Audit

Keep the current `\citep{key}` keys and adjacent `references.bib`.

- `ma2024fedcd`: treat FedCD as an **arXiv preprint** unless a new formal version is explicitly re-verified before submission.
- `lin2020feddf`: citation establishes FedDF background only. The project result must still be named **FedDF-fidelity**.
- `rockafellar2000cvar`, `duchi2021uniform`: foundational citations for CVaR/DRO only; the project arm remains a protocol-matched empirical tail-risk control.
- `sagawa2020groupdro`: GroupDRO is prior work; the project's **PEW+GroupDRO** is a matched baseline sharing PEW group assignments, not a new contribution.
- `liu2021jtt`: conceptual related work and screening history only; no Formal win/loss statement.
- If the paper later introduces specific factual claims about hospital, vehicle, industrial, scanner, or production-line prevalence, add direct primary citations before making those claims.

## 8. Main-Paper / Appendix Placement Freeze

### Main paper

Keep:

- CLE-HFL operator-level formulation and the family-level mitigation distinction.
- Data-role/evaluation isolation.
- DSA definition and one paired-cancellation result.
- Shuffled-binding evidence.
- Compact JSD insufficiency statement.
- Proxy non-identifiability theorem statement plus the conditional bridge.
- Table 1 HFL-vs-Local.
- Coarse PEW description and BER formula.
- Table 2 held-out map2 four-arm Formal.
- Table 3 cross-map/cross-base summary.
- Utility, taxonomy, and external-validity limitations.
- No more than two figure placeholders.

### Appendix

Move:

- affine-property proof/validation;
- injection recovery and cancellation numerical checks;
- full two-world non-identifiability construction;
- Hoeffding detail;
- PEW architecture/hyperparameters;
- BER CPU identity audit and TV table;
- original-map detailed per-client result;
- Oracle family/operator/random table;
- JTT screening-only status;
- any future extended seed/dataset/stress tables after they actually exist.

### Remove from manuscript entirely

Keep only in this author audit:

- contribution-to-evidence mapping;
- candidate experiment matrix;
- run authorization status;
- overclaim table;
- compression instructions;
- project-management instruction not to add modules.

## 9. Authorization Gate

As of V0.4 creation:

| Candidate experiment | Authorized? | Running? | Result may be written? |
|---|---:|---:|---:|
| HFL-vs-Local S1/S2 | **No** | **No** | **No** |
| held-out map2 S1/S2 | **No** | **No** | **No** |
| second private dataset | **No** | **No** | **No** |
| bounded taxonomy stress test | **No** | **No** | **No** |
| Formal JTT | **No** | **No** | **No** |

A future instruction must separately freeze each protocol and authorize execution. Until then, manuscript text must use `[EVIDENCE NEEDED]` rather than future-tense language such as “we run,” “we are evaluating,” or “experiments are underway.”

## 10. V0.4-to-Next-Version Checklist

Before converting V0.4 into a submission candidate:

- Verify every numeric value against the frozen Formal result files.
- Re-run citation audit if FedCD publication status changes.
- Resolve or explicitly retain every `[EVIDENCE NEEDED]`.
- Never replace missing seed evidence with source bootstrap.
- Preserve the FedDF utility failure and CVaR per-client exception.
- Preserve operator-level CLE/DSA versus family-level PEW+BER distinction.
- Keep CDep absent from all current-method text and equations.
- Do not add operator-level PEW, hierarchical PEW, a new loss, or a new communication method merely to address review-risk concerns.
