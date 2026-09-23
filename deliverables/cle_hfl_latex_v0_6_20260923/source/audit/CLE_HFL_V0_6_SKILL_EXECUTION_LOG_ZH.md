# CLE-HFL V0.6-SKILL 执行日志

**版本**：V0.6-SKILL  
**日期**：2026-09-23  
**性质**：conference-compressed internal manuscript；不是最终投稿稿。  

## 0. 本轮是否真正使用了上传的 Skill？

是。本轮不是按模型记忆“模拟”五个 Skill 的职责，而是实际解压用户上传的五个 ZIP，读取其中的 `SKILL.md`、被其路由要求引用的 supporting references，以及可运行的审计脚本，然后按各 Skill 规定的顺序执行。

读取的五个主 Skill 及 SHA-256：

| Pass | Skill | 实际读取的 `SKILL.md` | SHA-256 |
|---|---|---|---|
| 1 | Paper Review | `paper-review/.../skills/paper-review/SKILL.md` | `6432e3140c697a9e5e82fe8b5db3ab08739f9c1876b6666c66ae233023c7ae4b` |
| 2 | Academic Writing Skills | `academic-writing-skills/.../skills/academic-writing-skills/SKILL.md` | `58c06d7b359a8cd41aa7a0c75024b7bb1f7403c7778a7cdcb71b645c2c527ea9` |
| 3 | Scientific Writing | `scientific-writing-skill/.../SKILL.md` | `4030b029e5f77d33f389ae8566e4eefff1c074764dc432db2d10d9fb5c8c5462` |
| 4 | Academic Defensive Writing Auditor | `academic-defensive-writing-auditor/.../SKILL.md` | `f2bdb1c1b967bb703d452a96f90077c67d3fdb99ff2a5d765d3a57f0eb64e2de` |
| 5 | Humanizer | `humanizer/.../SKILL.md` | `f88e1e77d8056eab2457e459edc04b5245bdeec2e21f2247054fc6769e936379` |

同时实际读取并使用了以下 supporting material：

- Paper Review: `overlay-contract.md`, `ai-llm-computational.md`, `display-notation-provenance.md`；
- Academic Writing Skills: `state-and-authority.md`, `universal-integrity.md`, `prose-and-citation-editing.md`, `reviewer-red-team-and-release.md`, `banned_words.md`；
- Scientific Writing: `writing-principles.md`, `venue-conventions.md`；
- Defensive Auditor: `prompts/full-paper-cleanup.md`, `examples/before-after.md`；
- Humanizer: `measured-signals.md`, `before-after-examples.md`，并实际运行 `scripts/detect_ai.py`。

## 1. Authority hierarchy

按 Academic Writing Skills 的 managed-project 要求，本轮建立了 `manuscript_state.json`，并冻结如下权威层级：

1. **最新研究交接稿** `CLE_HFL_FULL_PAPER_WEB_HANDOFF_ZH(1).md`：事实、数字、证据状态、non-claims 的最高依据；
2. **实验表矩阵** `EXPERIMENT_TABLE_MATRIX_ZH.md`：表格位置、已完成/待补实验状态；
3. **引用审计** `CITATION_AUDIT_ZH.md`：文献发表状态和 baseline fidelity 边界；
4. **V0.5 LaTeX 源码包**：现有公式、label/ref、图表和 prose 的修改底本；
5. **RAHFL**：只用于学术表达、技术节奏、实验组织和图表叙事参考，不作为 CLE-HFL 新事实来源。

本轮没有使用已经过时的 V0.5 skeleton 去覆盖最新 handoff。例如，旧 skeleton 中曾提前写死的第二 private dataset 和 taxonomy stress 具体协议，均恢复为未冻结状态。

---

# Pass 1 — Paper Review

## Review basis

- **MODE**: authorized revision，但先执行 non-mutating reviewer pass；
- **STAGE**: integration / conference-compressed internal manuscript；
- **PROFILE**: general scientific review；
- **MODULES**: manuscript-integrity base + `ai-llm-computational` + `display-notation-provenance`；
- **SOURCE BASIS**: V0.5 源码、最新 handoff、实验矩阵、citation audit、RAHFL；
- **READINESS**: 核心证据链已成立，但仍有明确的未完成投稿证据。

## Reviewer-pass 主要发现

| Priority | Issue | Severity | Evidence status | Revision decision |
|---|---|---:|---|---|
| 1 | local-first 四臂 factorial 仍只有 training seed 0 | S3 | `[EVIDENCE NEEDED]` | 保留 local-first，但显式限定 seed 0；M2 不填造 |
| 2 | 第二 private dataset 未冻结 | S3 | `[EVIDENCE NEEDED]` | 删除旧 skeleton 中任何具体数据集决定；保持 dataset/protocol pending |
| 3 | Axis I map2 已经完成 seeds 0/1/2，但 V0.5 的叙事未充分提升这条证据 | S2 | verified Formal | 升级为主要 matched objective comparison，并传播到 Abstract/Intro/Results/Discussion/Conclusion |
| 4 | Axis II HFL mechanism-family context 仍无 Formal 科学数字 | S2 | pending | 保留 pending 表；benchmark/smoke 只作工程检查，不进入科学结论 |
| 5 | FedDF-fidelity 有强 DSA suppression，但 utility gate FAIL 且两臂低于 learning floor | S2 | verified Formal | 降级为 supporting cross-base evidence，完整保留负边界 |
| 6 | 现稿存在 reviewer-facing prebuttal、重复 non-claim 与 protocol-defense prose | S1–S2 | textual | 交给 Defensive Auditor 处理，必要 caveat 不删 |

Paper Review 后没有立即修改正文；先冻结上述 review ledger，再进入授权 revision。

---

# Pass 2 — Academic Writing Skills

## 执行方式

按 managed-project mode 执行：建立 authority state，进行 top-down argument audit 和 bottom-up evidence audit，并按变更类别做影响传播。

### Class A — semantic/argument changes

- 将 Experiments 的主逻辑冻结为两条轴：
  - **Axis I**: matched shortcut-objective attribution；
  - **Axis II**: protocol-matched HFL mechanism-family context；
- Shared validation 单独承载 local-first / cross-map / communication-base / future dataset-taxonomy boundaries；
- 该结构变化同步传播到 Introduction、Experiments、Discussion、Conclusion，避免各节使用不同论文故事。

### Class B — evidence/method propagation

- 把最新 handoff 中 **map2 seeds 0/1/2 Formal** 的事实升级为正文核心证据；
- 保留所有原有数值，使用 V0.5 已有三 seed mean±SD 表和冻结 contrasts；
- map1、FedDF-fidelity、Oracle 继续按各自协议解释，不做跨协议全局排行榜；
- local-first M2、second private dataset M3、taxonomy stress S1 和 Axis II Formal 均保持 pending；
- 删除旧 skeleton 对 M3/S1 的具体实验决定，防止旧草稿替代最新 authority source。

### Class C — metadata

无作者、机构、基金、匿名状态等元数据改动。

### Class D — surface/local clarity

- 缩短重复解释；
- 强化 claim-first results；
- 统一 DSA/PEW/BER/FedDF-fidelity/CVaR-DRO 术语；
- 保留全部 citation key、equation semantics、label/ref。

## 实际运行的 Academic Writing 审计脚本

- `audit_manuscript_state.py`
- `audit_candidate_text.py`
- `audit_prose_patterns.py`

最终 exact candidate SHA-256：

`e73227b0c38eefc7edb297cf5df9b2530e79cba9b2eb69591bd004c89ec73f98`

`audit_manuscript_state.py` 最终仍正确报告两个 S3 blocker：

- M2 local-first multi-training-seed stability；
- M3 second private dataset/protocol/result。

这些 blocker **没有被写作流程“润色掉”**。

---

# Pass 3 — Scientific Writing

目标不是 Nature 长文，而是紧凑 ML-conference prose。由于投稿 venue 尚未最终冻结，采用 NeurIPS/ICML/ICLR 类 ML-conference conventions 作为结构与 prose proxy，不虚构任何具体会议格式要求。

主要执行：

- Abstract 收紧为问题 → DSA → local-first → PEW+BER → 三 seed map2 headline → boundary；
- Introduction 保留四项贡献，删除结果提前解释和重复 novelty defense；
- Related Work 缩短为直接的 closest-work positioning；
- Results 改为 claim-first，每个结果先回答 RQ，再给数字和边界；
- 实验章节从“按运行历史”改为“按科学问题/比较轴”；
- Discussion 只集中 utility、proxy/taxonomy、statistical/external scope 三类边界。

RAHFL 仅用于学习：HFL context → corruption challenge → method role → experiment narrative 的节奏；未复制其句子、结论或期刊篇幅。

---

# Pass 4 — Academic Defensive Writing Auditor

按 D1–D14 taxonomy 做了 detection → necessity classification → reviewer reaction → rewrite → paper-level audit。

## 主要清理类型

- D1 reviewer-facing prebuttal；
- D2 repeated non-claim disclaimers；
- D3 caveat stacking；
- D6 “fair comparison”式自我辩护；
- D8 legalistic disclaimer lists；
- D10 promotional compensation；
- D11 AI-style automatic summary sentences；
- D12 evidence-boundary over-signaling。

V0.5 中代表性模式计数包括：`We do not claim`×2、`This does not`×1、`safe interpretation`×1、`not evidence`×1、`universal`×4、`lossless`×1。最终正文扫描中这些 reviewer-facing 模式均已清零；仅保留一次 `not uniformly`，用于 CVaR c0/c1 的真实 per-client 例外。

## 必须保留的 scientific caveats

以下内容被明确标为 NECESSARY_CAVEAT，因此不能由“去防御性”步骤删除：

- source bootstrap ≠ training-seed uncertainty；
- local-first factorial 仍为 seed 0；
- FedDF-fidelity utility gate FAIL；
- FedDF 两臂低于预注册 learning floor；
- CVaR-DRO 在 c0/c1 的 DSA 更低；
- PEW 是 taxonomy-assisted coarse family-level witness；
- JTT 仅 screen-only；
- single private task / fixed partition；
- second dataset、taxonomy stress、Axis II Formal 等 pending 状态。

---

# Pass 5 — Humanizer

严格按 Humanizer 的要求执行，而不是“降学术性”：

- 保留 citations、numbers、defined terms、equations、table/figure labels；
- 删除 Unicode em dash；
- 删除模板式总结、throat-clearing、过量 signposting；
- 恢复直接主语和自然学术动词；
- 保留必要的长句和技术词，不机械短句化；
- 最终逐节实际运行 `scripts/detect_ai.py` 作为诊断工具，而不是作者身份判定器。

最终诊断：

| Section | AI-likeness diagnostic | Skill verdict |
|---|---:|---|
| Abstract | 18.9 / 100 | reads human |
| Introduction | 22.8 / 100 | reads human |
| Experiments | 27.1 / 100 | reads human |
| Discussion | 17.4 / 100 | reads human |
| Conclusion | 24.3 / 100 | reads human |

这些分数只按 Humanizer Skill 自己的说明作为 pattern checklist，**不是 AI-authorship 证明或投稿指标**。

---

# Pass 6 — Final Fact Audit

最终进行了：

- V0.5 vs V0.6 citation-key diff；
- V0.5 vs V0.6 label/ref diff；
- 数字 token 来源检查；
- hard-boundary grep；
- bibliography resolution；
- draft/internal 与 clean/evidence-only 两种 LaTeX 编译；
- pending evidence blocker audit。

结果见 `CLE_HFL_V0_6_SKILL_FACT_AUDIT_ZH.md`。

## 编译状态

- internal `\draftskeletontrue`: 13 pages；
- clean `\draftskeletonfalse`: 11 pages；
- 两种模式均无 undefined reference/citation、无 overfull box、无 LaTeX error；
- BibTeX 仅有 FedProto/FedTGP 条目的 `volume` + `number` 格式 warning，属于 bibliography-formatting 清理项，不影响引用解析。

## 结论

本轮确实执行了上传 Skill 包内的工作流。Skill 的作用被限制在 review / argument / prose / defense / humanization 层；最新 handoff、实验矩阵和 citation audit 始终高于任何风格建议，因而 Skill 没有权限填造缺失实验或扩大科学结论。
