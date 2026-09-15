我完整审查了当前英文会议稿、`AUTHOR_AUDIT` 和 `FACT_AUDIT`。先给结论：**这已经是一篇结构成立、证据纪律明显强于普通“方法+结果”稿件的会议论文，但以严格 CCF-B / AISTATS 风格审稿标准看，当前最主要的问题已经不是故事是否成立，而是核心 mechanistic claim 与主要方法比较仍高度依赖 seed-0，以及外部有效性只有单一 private dataset。** `FACT_AUDIT` 也明确把 V0.4 定位为“会议压缩内部主稿而非最终投稿稿”，并确认当前所有待补实验仍是 `[EVIDENCE NEEDED]`。

# A. 我对论文主张的复述

本文研究模型异构联邦学习中的 **class–corruption entanglement (CLE-HFL)**：当客户端内部类别与具体 corruption operator 形成定向绑定时，corruption 可能从普通 nuisance 变成预测类别的 shortcut。论文使用同一 semantic source 的 paired operator intervention，并定义 **Directional Shortcut Alignment (DSA)**，检测预测概率是否沿预注册的训练 binding 方向移动；DSA 被严格限定为在 semantic preservation、valid intervention、pre-registered binding 和 evaluation isolation 等假设下的 binding-direction response，而非完整因果机制恢复。随后，seed-0 的 matched HFL-vs-Local factorial 显示 HFL CLE effect 为 0.119889、Local effect 为 0.107960、communication add-on 为 0.011929，因此提出 predominantly local-first 的机制解释。基于此，论文不改通信，而使用 coarse family-level、taxonomy-assisted PEW 提供伪环境，并用 BER 压缩类别内部 pseudo-environment support imbalance。held-out map2 上，PEW+BER 相对 ERM、共享 PEW 的 GroupDRO 和 pooled CVaR-DRO 获得更好的 shortcut–utility trade-off；cross-map 和 FedDF-fidelity 进一步测试作用边界，而 proxy non-identifiability 则解释为什么 proxy/JSD 的改善不能替代 target-aligned DSA。论文明确保留 mixed utility、taxonomy dependence、single-dataset、fixed-partition 和 seed-0 等边界。 

# B. 总体评分

这些是**审稿判断，不是项目事实**。

| 维度                            |                           评分 | 审稿意见                                                                                                                                                                                                  |
| ----------------------------- | ---------------------------: | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Novelty**                   |                   **7.0/10** | 单独看 PEW 或 BER 不新，但 “CLE-HFL specific failure mode + paired DSA + local-first attribution + structure-matched mitigation” 作为组合有辨识度。最大的风险是审稿人将其简化为“synthetic spurious correlation + group reweighting”。 |
| **Soundness**                 |                   **8.0/10** | 这是当前最强项。数据角色隔离、DSA 的识别边界、source-level bootstrap、proxy non-identifiability 和负结果边界写得比较严谨。尤其没有把 source CI 冒充 training stability。                                                                         |
| **Clarity**                   |                   **7.5/10** | 主线已经清楚，但 Section 4 理论密度偏高；Introduction、Section 4、Discussion 仍重复解释“proxy ≠ target harm”。图 1 很有帮助。                                                                                                      |
| **Experimental completeness** |                   **5.5/10** | 最大短板。关键 local-first 和 map2 Formal 仍是 seed-0，single private dataset，fixed partition；FedDF cross-base 又存在 learning-floor fail。                                                                          |
| **Overall 当前状态**              | **Borderline / Weak Reject** | 如果我是严格 CCF-B reviewer，我会认为“核心故事值得发表，但 stability / breadth 证据还没完全达到最终投稿成熟度”。                                                                                                                           |

AUTHOR_AUDIT 对未完成证据的状态也很明确：local-first multi-seed、map2 multi-seed、second private dataset、taxonomy stress 均尚未授权，更不能当已有证据。

# C. 最可能导致拒稿的 5 个问题

## 1. 核心 mechanistic claim 与主要 method comparison 都仍依赖 seed-0

这是当前最大的拒稿点。

Section 5 的 local-first 是论文第三项核心贡献，但当前四臂 factorial 明确只有 training seed 0。主文虽然诚实写了 `[EVIDENCE NEEDED]`，但这意味着目前 reviewer 可以直接说：

> The “local-first” conclusion may be training-randomness dependent.

更关键的是，Table 2 的 ERM/CVaR/PEW+GroupDRO/PEW+BER 也只有 seed-0。也就是说，C3 和 C4 的最关键 empirical claim 同时缺 training-seed stability。 

**严重程度：投稿前必须解决 `[EVIDENCE NEEDED]`。**

---

## 2. 外部有效性仍是 single private dataset + fixed partition

当前 CLE、DSA、local-first、BER 主要都围绕 CIFAR-10 controlled CLE-v2 展开。cross-map 只改变 binding direction，并没有改变 private task、partition 或 corruption family。论文自己已经正确承认这一点。

严格 reviewer 很可能问：

> Is this a general failure mode or an artifact of one CIFAR-10 partition and synthetic operator assignment?

两张 map 解决的是 **mapping sensitivity**，不是 dataset/generalization。

**严重程度：我会列为 CCF-B 投稿前必须补强 `[EVIDENCE NEEDED: second private dataset; dataset and protocol pending.]`。**

---

## 3. FedDF-fidelity 的 cross-base 证据有解释力，但不够“干净”

Section 7.3 已经非常诚实：FedDF-fidelity 的两个 arm 都低于预注册 20% operator-grid learning floor，并且 overall utility gate fail。

所以 reviewer 会攻击：

> Does DSA drop because BER truly mitigates CLE, or because both models are under-learned under this secondary protocol?

你目前通过“matched within-setting contrast”和四客户端同向 DSA reduction 来缓解这个质疑，但它仍然不是一个特别漂亮的 cross-base validation。

我的建议不是再做第三通信底座，而是**降低 FedDF 在主叙事中的权重**：

> supporting cross-base evidence

而不是：

> independent generalization validation.

**严重程度：最好解决不了也可以接受，但写作定位必须降级。**

---

## 4. 方法新颖性很容易被审稿人压缩成“public corruption classifier + reweighting”

这会是最典型的 novelty reject。

你已经做了最关键的防守：

* 不把 PEW 当架构创新；
* 共享 PEW 的 GroupDRO matched control；
* CVaR tail-risk control；
* BER effective-distribution derivation。

特别是 Table 2 中同 PEW side information 下 GroupDRO DSA 仍为 0.214575，而 BER 为 0.077741，这对“只是 PEW 分组”的解释是很好的反证。

但审稿人仍可能说：

> The method itself is technically simple; most novelty appears in problem construction and diagnostics.

这个攻击不是错误。你的回答应该是**承认并重新定义贡献单位**，而不是试图把 BER 包装成全新优化理论。

**严重程度：高，但主要是 framing，不应该通过新增模块解决。**

---

## 5. “local-first” 比其他结论更容易被读成因果机制结论

你目前已经加了很多防线：

* 90.05% 明确是 descriptive ratio；
* 不是 causal mediation；
* communication 不是 irrelevant。

这都很好。

但标题 **“Local-First Mechanism Attribution”** 和正文的 “mechanism” 一词还是略强，因为实验本质是 matched factorial contrast，而非完整 causal mediation / intervention decomposition。

这会产生 reviewer attack：

> The experiment supports relative formation under two training conditions, but does it justify “mechanism attribution”?

我建议标题稍降一级，例如：

**Local-First Formation Analysis**

或

**Local-versus-Communication Attribution**

不需要取消 local-first，只需避免让 “mechanism” 看起来比证据更强。

**严重程度：最好修改措辞；multi-seed 补完后会明显缓解。**

---

# D. 逐节审稿

## Abstract，p.1

当前 Abstract 很成熟，尤其已经写了 “Completed seed-0 controlled comparisons”，这正是 FACT_AUDIT 本轮修复的内容。

问题有两个。

第一，“A matched HFL-versus-Local factorial indicates that the observed shortcut is predominantly local-first” 在摘要里虽然事实没错，但没有 seed-0 qualification，而后面的 controls 才说 seed-0。建议把 qualification 提前统一。

更安全：

> In a seed-0 matched HFL-versus-Local factorial, the pooled shortcut effect is predominantly local-first in the controlled setting.

第二，Abstract 的理论部分略重，“identifies... complementary proxy non-identifiability...” 占了较多空间。若以后需要腾空间给 multi-seed/second dataset headline，可以压掉一句理论解释。

---

## Introduction，pp.1–2

总体逻辑是对的：HFL corruption → CLE → why accuracy/JSD insufficient → DSA → local-first → PEW+BER。

但有两处轻度重复。

第一，第 3 段已经解释一次 proxy improvement 不足；Section 4.3 又完整解释；Discussion 再解释一次。Introduction 保留一句即可，定理细节交 Section 4。

第二，第 5 段同时讲 map2、CVaR、FedDF、贡献列表，信息略拥挤。

建议 Introduction 只留下：

> We evaluate against matched group/tail-risk controls and under controlled changes of the binding map and communication base.

不要在那里提前告诉 reader 谁赢了谁。

---

## Related Work，pp.2–3

这是目前比较干净的一节。

优点是你没有犯最危险的 priority claim：

* 明确承认 federated spurious learning 已有工作；
* FedCD 只写 preprint；
* EIIL 被正确作为 taxonomy-free contrast；
* CVaR 被写成 protocol-matched empirical control。

这些和 AUTHOR_AUDIT 完全一致。

主要问题是 2.3 对 BER 的解释稍微开始进入 Method：

> “BER changes the total effective risk mass assigned to class–pseudo-environment supports.”

这句话可以保留，但不要再扩展公式性质，否则 Related Work 和 Section 6 重复。

---

## Problem Setup，p.3

这是目前论文最清楚的一节之一。

尤其这一组区别**必须保留**：

> CLE-v2 and DSA are operator-level; PEW+BER is coarse family-level.

这是 reviewer 最容易误解的方法边界，当前 p.3 写得非常好。

一个缺口是：\(\gamma_{\mathrm{CLE}}\) 的具体数据生成概率在主文没有公式，只描述 dominant operator。

如果篇幅允许，我反而建议把正式 sampling law 放回主文或 Appendix 立即引用，否则 reviewer 很难仅凭当前 Setup 精确理解 \(\gamma=0.9\) 到底如何生成 CLE。

这是**表达完整性问题，不是新增实验**。

---

## Section 4 — Paired Diagnosis and Identification，pp.3–4

这是论文最强、同时也是最容易写太重的一节。

### 4.1 DSA

定义清晰。

唯一需要进一步强调的是 pooled averaging hierarchy。现在写：

> equally averages valid operator-level values and then client-level values.

建议附录里给一个完整 pooled formula，避免 reviewer 质疑 client/operator weighting。

### 4.2 Paired Cancellation

数学是成立的，但目前这个 “decomposition” 很容易被 reviewer 读成一个模型假设：

$$
m=s+r.
$$

实际上任何 \(m\) 都可以形式上拆成 invariant + residual；关键是你对 \(s\) 的解释。

建议明确：

> This is an identification decomposition for the paired estimand, not a generative assumption about the neural network.

可以提前挡掉“theorem is tautological”攻击。

### 4.3 JSD + non-identifiability

逻辑成立，但**内容太密**：这里同时讲 JSD counterexample、TV 定义、two-world theorem、conditional bound、PEW error、DSA necessity。

这是会议稿里最可能让读者掉线的一段。

建议主文只保留：

1. JSD=0 does not imply DSA=0；
2. theorem statement；
3. current PEW bound is trivial；
4. implication: target-aligned DSA needed under this protocol。

0/0.8 construction 继续留 Appendix，很合适。

### 4.4 Statistical Unit

必须保留。当前最重要的一句话就是：

> source bootstrap does not quantify training-seed uncertainty.

不要再压缩掉。

---

## Section 5 — Local-First Mechanism Attribution，pp.4–5

这是最值得审稿人追问的一节。

Table 1 非常有价值，因为四个 arm 完整展示了 gamma0 / gamma.9 × HFL / Local，而不是只给三个 derived contrast。

三个问题：

**第一，标题略强。** 前面说过，推荐改 `Formation Analysis`。

**第二，12 rounds 的合理性需要一句说明。**

审稿人会问：

> Why is a 12-round experiment sufficient to support a mechanism claim when map2 uses 40 rounds?

不要拿 screen 做解释。需要明确这是冻结 Formal mechanism protocol；如果没有更多依据，不要增加结论，只说明 protocol role。

**第三，也是最大问题：multi-seed pending。**

当前 `[EVIDENCE NEEDED]` 放在正文是正确的。不要删除。

---

## Section 6 — PEW+BER，p.5

整体很好，尤其没有把 PEW 夸成 latent environment discovery。

PEW 段落最重要的是这一句：

> “does not define or motivate an operator-level PEW.”

这和 AUTHOR_AUDIT 的 frozen boundary 完全一致。

BER 部分则有一个潜在 reviewer misunderstanding：

$$
a_{k,c,e}\propto n_{k,c,e}^{0.5}
$$

表面看，大组得到的 **group mass** 仍比小组高，为什么叫 “balanced”？

你实际上平衡的是**每样本/组总风险质量的支持优势压缩**，不是 uniform group weighting。

当前有解释，但建议在公式之后加一句更直观的话：

> BER does not equalize environment groups; it sublinearly compresses their support-induced mass advantage.

这样 “Balanced Environment Risk” 这个名字不容易被攻击成名不副实。

---

## Section 7 — Experiments，pp.6–7

### 7.1 Protocol and Metrics

有点长。

p.6 从 “Absolute values across different protocols...” 到 “prevents protocol changes from being misinterpreted...” 是合理 audit prose，但对正式论文略显作者防御性。

可以压成两句。

### 7.2 map2 controls

这是目前最重要的主结果，位置和篇幅都应该保留。

表 2 强于单纯 Base vs BER，因为：

* ERM；
* tail-risk；
* same-PEW GroupDRO；
* BER。

而且你没有隐藏 c0/c1 的 CVaR 优势。

这里真正缺的只有：

**[EVIDENCE NEEDED: multi-training-seed stability]**

### 7.3 cross-map + FedDF

cross-map 很干净。

FedDF 则必须一直保留：

* utility gate fail；
* both below 20% floor；
* fidelity adapter。

你目前已经做对了。不要为了“结果好看”删掉 learning-floor sentence。

### 7.4 Oracle

内容正确，但主文价值有限。Appendix 已有完整表。

如果以后需要为 multi-seed 或 second dataset 腾空间，我建议**整节正文压成 2–3 句**。

### 7.5 Remaining Evidence Placeholders

作为 internal manuscript，保留是正确的。

最终提交版当然不能保留 project-management language：

> “not authorized or reported as running”

那是 AUTHOR_AUDIT 内容，而不是论文。

但在当前 V0.4 internal version 中没有问题。

---

## Figure 1，p.7

图是有价值的。

最好的地方是上下分成：

* TRAIN-TIME PATH；
* SEALED EVALUATION AND ATTRIBUTION。

这非常有效地说明 **PEW+BER 不读取 operator/binding，而 DSA 读取 sealed evaluation metadata**。这正是审稿人最容易误解的 leakage 问题。

但图中：

> “2 Local shortcut formation”

写得像 formation 已经直接观测到。

建议弱化为：

> **Shortcut-prone local training**

或

> **Local formation hypothesis**

然后第 4 步才显示 HFL-vs-Local attribution。

---

## Figure 2，p.8

图本身清楚，但和 Section 4.3 的文字重复度较高。

如果未来需要压页，我第一个移动到 Appendix 的就是 Figure 2，不是 Figure 1。

Figure 1 更重要，因为它同时解决：

* workflow；
* information boundary；
* method/evaluation distinction。

---

## Discussion，pp.7–8

这一节边界意识很好。

尤其三件事都明确：

* DSA ≠ utility；
* taxonomy-assisted；
* external validity limited。

一个小问题是 Discussion 又完整重复了一遍 non-identifiability 的 logic，可以压掉约三分之一。

---

## Conclusion，p.8

当前 Conclusion 很克制，没有 universal claim，整体通过。

但这句：

> “Completed experiments show large DSA reductions…”

“large” 虽然从数字上合理，但可以进一步中性：

> “Completed experiments show substantial pooled DSA reductions…”

或者直接给 relative numbers，避免主观 adjective。

---

## Appendix，pp.8–10

Appendix 内容选择总体合理。

A.1 的 numerical errors 很小，但这类 \(10^{-16}\) 数字是**实现 identity test**，不是 statistical evidence。建议明确叫：

> numerical identity checks

不要让 reviewer 误会成极高统计精度。

C.2 Oracle table 出现在 PDF 第 10 页，位置略奇怪——references 已经开始后才插表。最终排版要修复 float placement。

# E. 四项 contribution 是否真正独立

**总体独立，当前四条结构是合理的。**

| Contribution                           | 独立性 | 风险                                        |
| -------------------------------------- | --- | ----------------------------------------- |
| **C1 CLE-HFL formulation**             | 高   | 是研究对象贡献                                   |
| **C2 DSA + identification boundary**   | 高   | 是 measurement / theory contribution       |
| **C3 local-first attribution**         | 高   | 是 empirical mechanism finding             |
| **C4 PEW+BER + controlled validation** | 中高  | 是 intervention contribution，但依赖 C3 提供设计动机 |

唯一需要注意：

### C2 不要再拆成两个 contribution

`DSA` 和 `proxy non-identifiability` 本质上都回答：

> 为什么 CLE mitigation 需要 target-aligned evaluation？

所以它们属于同一个 contribution。当前处理正确。

### C4 也不要再拆

cross-map、FedDF、GroupDRO、CVaR、Oracle 都是 **C4 的 evidence**，不是独立贡献。

### C3 的独立性是存在的

即使 PEW+BER 最终不成功，“local-first” 仍然可以作为一个独立科学发现。

这是你这篇论文区别于普通算法论文的关键。

# F. 投稿前：必须补 / 最好补 / limitation 即可

## 投稿前必须补

**1. `[EVIDENCE NEEDED]` matched HFL-vs-Local multi-training-seed stability。**

这是 C3 的基础。

AUTHOR_AUDIT 已冻结规则：一个 S1/S2 四臂实验同时覆盖 RQ1 和 RQ2，不重复规划。

**2. `[EVIDENCE NEEDED]` held-out map2 four-arm multi-training-seed stability。**

这是 C4 中最强 empirical comparison 的稳定性证据。

**3. `[EVIDENCE NEEDED: second private dataset; dataset and protocol pending.]`**

如果目标是严格 CCF-B，我会把它列为必须。

当前不要提前指定数据集。AUTHOR_AUDIT 明确说明任何 dataset 均未冻结。

---

## 最好补

**4. `[EVIDENCE NEEDED: bounded taxonomy stress test; protocol pending.]`**

它对回应 taxonomy-assisted / closed-set criticism 很有价值，但我认为低于前 3 项。

目前未授权，也未冻结具体 stress protocol。

---

## 可以诚实放入 limitations

* 固定 non-IID partition；
* architecture 与 client data slice confounded；
* FedDF-fidelity learning-floor fail；
* PEW error bridge 当前是 trivial bound；
* real-world deployment 未验证；
* operator-specific semantic leakage 的潜在风险；
* taxonomy-free/open-world corruption 未解决；
* JTT 没有 Formal comparison。

这些都不需要为了“补齐”再发明新模块。

# G. 会议版进一步压缩建议

当前主文实际已经大约 8 页到 Conclusion，压缩程度不错。FACT_AUDIT 也确认已经从长稿压到约 5k 词并保留核心链。

如果以后加入 multi-seed / 第二数据集，需要腾约 **0.5–1 页**，我建议按以下顺序切：

1. **Figure 2 移 Appendix。**
2. Section 7.4 Oracle 正文从一整小节压成 2–3 句。
3. Section 7.1 的 protocol-defense paragraph 压缩 40–50%。
4. Section 4.3 non-identifiability 主文只留 theorem + implication；0.507–0.575 可以移 Appendix。
5. Related Work 2.3/2.4 各减 2–3 句，但保留四分组结构。
6. PEW 具体 taxonomy 列表可留，training details继续只在 Appendix。
7. Appendix 内 numerical identity checks 和 audit tables继续保留，不占主文。

**不要删：**

* Table 1；
* Table 2；
* Table 3；
* Figure 1；
* source-bootstrap ≠ training-seed 的声明；
* FedDF utility failure；
* CVaR c0/c1 exception；
* operator-level CLE/DSA vs family-level PEW 区别。

# H. 下一版不超过 10 条的具体修改清单

1. **将 Section 5 标题从 `Local-First Mechanism Attribution` 弱化为 `Local-First Formation Analysis` 或 `Local-versus-Communication Attribution`。**
2. **Abstract 的 local-first 句显式加入 `seed-0` qualification。**
3. **在 Section 3 补充或 appendix-reference 正式的 CLE-v2 operator sampling law，避免 \(\gamma_{\mathrm{CLE}}\) 只靠文字描述。**
4. **在 Section 4.2 明确 \(m=s+r\) 是 paired estimand 的 identification decomposition，不是 neural-network generative assumption。**
5. **Section 4.3 压缩：主文只保留 JSD counterexample、non-identifiability theorem、conditional bridge 和 DSA implication。**
6. **BER 公式后增加一句：`BER does not equalize groups; it sublinearly compresses support-induced risk-mass advantages.`**
7. **Figure 1 把 “Local shortcut formation” 改成更弱的 “Shortcut-prone local training” 或等价表述，避免图本身先验宣告机制结论。**
8. **FedDF-fidelity 始终降级为 supporting cross-base evidence；绝不删除 learning-floor fail 和 utility-gate fail。**
9. **继续保留三个核心 `[EVIDENCE NEEDED]`：local-first multi-seed、map2 multi-seed、second private dataset；taxonomy stress 保留为次优先级。**
10. **如果未来需要腾页面加入新证据，先移动 Figure 2 和 Oracle 正文，而不是删识别假设、负结果或 Table 1–3。**

最后，如果我是实际 reviewer，我现在最可能给出的 meta-summary 会是：

> **The paper presents an unusually coherent diagnosis–attribution–mitigation chain for a specific HFL shortcut failure mode, with careful identification and reporting boundaries. The main remaining concern is empirical stability and breadth: the strongest mechanistic and comparative results rely on a fixed partition and seed-0 controlled experiments on a single private dataset.**

换句话说，**现在不是“论文故事不成立”，而是“论文故事已经成立，但投稿级证据还差最后一层稳定性”。**
