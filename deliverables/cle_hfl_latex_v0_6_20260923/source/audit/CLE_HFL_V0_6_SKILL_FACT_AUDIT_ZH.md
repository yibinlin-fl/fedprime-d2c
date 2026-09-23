# CLE-HFL V0.6-SKILL 最终事实审计

**Audit target**: V0.6-SKILL exact candidate  
**Candidate SHA-256**: `e73227b0c38eefc7edb297cf5df9b2530e79cba9b2eb69591bd004c89ec73f98`  

## 1. 结论

V0.6-SKILL 通过当前可执行的静态事实、引用、交叉引用和编译审计。它仍是 **internal manuscript**，不是 submission-ready artifact，因为 M2 local-first multi-seed 和 M3 second private dataset 仍是 S3 evidence blockers。

## 2. V0.5 → V0.6-SKILL 不变量

| Check | Result |
|---|---|
| Citation keys | **20 → 20；added 0 / removed 0** |
| Missing citation keys in `references.bib` | **0** |
| LaTeX labels | **15 → 15；added 0 / removed 0** |
| `\ref`/`\eqref` 指向不存在 label | **0** |
| 新出现、无法在 V0.5 或最新 handoff 追溯的 numeric token | **0** |
| `Directional Shortcut Attribution` | **0** |
| Active current-method `CDep` | **0** |
| PEW 被写成 taxonomy-free | **0** |
| Unicode em dash | **0** |

源码中有一次字符串 `operator-level PEW`，上下文是明确的**否定边界句**：当前 Oracle granularity audit “provides no empirical reason to introduce an operator-level PEW in the current paper”。它不是已实现方法或 proposal。

## 3. 核心事实边界逐项检查

| Boundary | Status | Evidence in manuscript |
|---|---|---|
| DSA 正式全称为 Directional Shortcut Alignment | PASS | Abstract/Method/Conclusion consistent |
| CLE-v2 construction 与 DSA evaluation 为 operator-level | PASS | Problem setup + information boundary |
| PEW+BER 为 coarse family-level taxonomy-assisted | PASS | Introduction/Method/Discussion |
| Oracle operator 仅为 non-deployable boundary audit | PASS | Experiments + Appendix |
| CDep 不进入 current method/formula/result claim | PASS | active TeX grep = 0 |
| source bootstrap ≠ training-seed uncertainty | PASS | Method/Discussion/Appendix |
| map2 四臂 seeds 0/1/2 Formal 已传播 | PASS | Abstract/Intro/Table/Experiments/Conclusion |
| PEW+BER vs CVaR 仅 pooled trade-off；c0/c1 exception 保留 | PASS | Experiments/Discussion |
| FedDF-fidelity 只支持 DSA suppression；utility gate FAIL | PASS | Abstract/Experiments/Discussion/Conclusion |
| FedDF 两臂 learning floor fail 保留 | PASS | Experiments |
| JTT 仍 screen-only | PASS | Experiments/Appendix/Table note |
| benchmark/smoke 不作为科学证据 | PASS | Axis II scientific cells pending |
| second private dataset 未被擅自指定 | PASS | generic pending only |
| taxonomy stress protocol 未被擅自指定 | PASS | generic pending only |
| 未新增 loss / communication module / hierarchical PEW / operator PEW | PASS | source diff + method inspection |

## 4. Multi-seed / source uncertainty 边界

当前证据层级在 V0.6-SKILL 中区分为：

- **map2 Axis I**: training seeds 0/1/2 Formal complete；
- **local-first four-arm factorial**: seed 0 only，M2 pending；
- **map1 replication**: seed 0；
- **FedDF-fidelity**: seed 0；
- **source-clustered bootstrap**: checkpoint-conditional evaluation-source uncertainty only。

因此没有把 map2 三 seed 稳定性错误外推到 local-first、cross-map 或 FedDF。

## 5. Pending evidence audit

`audit_manuscript_state.py` 最终报告两个 S3 blocker：

1. **M2**: Local-first multi-training-seed stability remains pending；
2. **M3**: Second private dataset and protocol/result remain pending。

此外，论文内部明确保留：

- **S1** bounded taxonomy stress: protocol pending；
- **S2-v2 Axis II**: engineering benchmark/pairing work exists, but Formal scientific metrics pending；
- **O1 JTT Formal**: optional, not authorized/completed；
- computation-cost unified timing: pending。

## 6. Academic Writing exact-candidate audit

`audit_candidate_text.py` 对最终 exact candidate 的自动 findings 仅剩：

- dense lexical hyphenation；
- 几个因定义需要重复的技术短语。

人工复核后保留 `source-paired`, `operator-grid`, `training-seed`, `family-level`, `protocol-matched`, `pseudo-environment` 等，因为它们是领域定义或精确复合词。没有为了消除机器 flag 而破坏技术术语。

## 7. Defensive-writing audit

最终正文 pattern scan：

- `We do not claim`: 0
- `This does not`: 0
- `safe interpretation`: 0
- `not evidence`: 0
- `We emphasize`: 0
- `should not be interpreted`: 0
- `universal`: 0
- `lossless`: 0
- `not uniformly`: 1（保留，因 CVaR c0/c1 exception 是必要 caveat）

## 8. Humanizer diagnostic

| Section | Score | Verdict |
|---|---:|---|
| Abstract | 18.9 / 100 | reads human |
| Introduction | 22.8 / 100 | reads human |
| Experiments | 27.1 / 100 | reads human |
| Discussion | 17.4 / 100 | reads human |
| Conclusion | 24.3 / 100 | reads human |

这些结果只用于检查 Humanizer Skill 定义的表面模式，不作为作者身份或审稿质量的证据。

## 9. LaTeX compile audit

| Mode | Pages | Error/undefined ref/citation/overfull |
|---|---:|---|
| internal `\draftskeletontrue` | 13 | none |
| clean `\draftskeletonfalse` | 11 | none |

BibTeX 解析成功；仅有两个非阻断格式 warning：

- `tan2022fedproto`: 同时存在 `volume` 与 `number`；
- `zhang2024fedtgp`: 同时存在 `volume` 与 `number`。

投稿模板冻结后再统一清理 bibliography formatting，不能因此静默修改 citation identity。

## 10. Release judgement

**V0.6-SKILL 可作为新的网页端/Overleaf 内部写作主稿。**  
**尚不应标记为最终 submission-ready**，原因是 M2/M3 仍是高优先级 evidence gaps，S1/S2 仍有未完成的边界/背景实验槽位。
