# Version 0.2 — Fact-Audited English Long Draft

> **Working status.** This document is a long-form conference-paper draft assembled only from the uploaded CLE-HFL evidence package and theory notes, then fact-audited against the current repository on 2026-09-14. It is intentionally not page-limited. Statements that require experiments not yet completed are marked **[EVIDENCE NEEDED]**. Citations whose exact bibliographic entry or paper-level claim still requires paper-by-paper verification are marked **[REF TO VERIFY]**. The current evidence is controlled and synthetic; it does not establish deployment effectiveness in hospitals, vehicles, factories, or other real-world domains. The final method is frozen to a **coarse family-level PEW plus BER**; no operator-level PEW is proposed or implemented.

# 1. Title

**When Corruption Becomes a Label: Diagnosing and Mitigating Class–Corruption Shortcuts in Model-Heterogeneous Federated Learning**

Alternative working title for a more formal ML style:

**Class–Corruption Entanglement in Model-Heterogeneous Federated Learning: Paired Diagnosis, Local-First Attribution, and Taxonomy-Assisted Mitigation**

# 2. Abstract

Model-heterogeneous federated learning (HFL) allows clients with different model architectures to collaborate without sharing private training data, but robustness work in this setting typically treats data corruption as a nuisance that degrades input quality. We study a different failure mode, **class–corruption entanglement (CLE)**, in which client-specific associations between task classes and corruption operators can turn corruption into a directional class cue. Standard corrupted accuracy is insufficient to distinguish this behavior from ordinary corruption difficulty. We therefore introduce a source-paired operator counterfactual protocol and **Directional Shortcut Alignment (DSA)**, which measures whether changing the corruption operator for the same source systematically shifts predictive probability mass toward classes bound to that operator during training. Under explicit semantic-preservation, pre-registered-binding, and evaluation-isolation assumptions, DSA identifies a binding-direction operator-response contrast; we further establish its zero baseline, affine behavior, source-level inference unit, and a proxy non-identifiability result showing why generic proxy improvements cannot by themselves certify reduced CLE harm. A matched HFL-versus-Local analysis indicates that the observed shortcut is predominantly local-first in the controlled setting. Guided by this finding, we develop a taxonomy-assisted local intervention combining a **Public Environment Witness (PEW)** with **Balanced Environment Risk (BER)**, which compresses class-conditional pseudo-environment support imbalance without changing the communication protocol. Across controlled binding maps, PEW+BER substantially reduces DSA, and its shortcut-suppression effect also transfers to a second HFL communication base. On a held-out 40-round map, PEW+BER achieves a better pooled shortcut–utility trade-off than matched ERM, PEW+GroupDRO, and CVaR-DRO controls, while not uniformly dominating CVaR-DRO at every client. Utility gains are not consistent across HFL bases, and the method remains taxonomy-assisted. **[EVIDENCE NEEDED: pure PEW+BER multi-training-seed stability, a second private dataset, and a bounded taxonomy stress test.]**

# 3. Introduction

Federated learning enables distributed clients to collaboratively train predictive models while keeping private data local. In practical deployments, however, participating clients may differ not only in their data distributions but also in their model architectures, motivating **model-heterogeneous federated learning (HFL)**. Existing HFL approaches use public-data distillation, logit exchange, confidence reweighting, or asymmetric knowledge transfer to communicate across heterogeneous models. Robust HFL further considers clients whose private images are corrupted by noise, blur, weather effects, compression artifacts, or other common distortions. In particular, AugHFL and its later TPAMI extension RAHFL study model-heterogeneous collaboration under data corruption, using corruption-robust local learning and robust/asymmetric communication to preserve predictive performance. These works establish data corruption as an important robustness problem in HFL, but they primarily treat corruption as a nuisance that damages local learning or transmits low-quality knowledge rather than as a potentially predictive class code.

This paper studies a more specific failure mode that arises when corruption is statistically entangled with task labels. Consider a client for which most images of one class are observed with blur while most images of another class are observed with noise. In that client, the corruption environment is no longer approximately independent of the label: it can become a shortcut that predicts the class. Moreover, different HFL clients may possess different class–corruption bindings. We call this controlled failure mode **class–corruption entanglement in HFL (CLE-HFL)**. The key question is therefore not merely whether corrupted images reduce accuracy, but whether a model changes its prediction *in the direction of the class–corruption association it observed during training*. This distinction matters because low corrupted accuracy can arise from ordinary input difficulty, class imbalance, limited capacity, or non-IID training without implying that corruption itself is being used as a directional class cue.

Diagnosing such behavior requires an evaluation target that is aligned with the suspected shortcut. Within-operator consistency objectives such as AugMix-style Jensen–Shannon divergence can make multiple augmentations of the same corrupted view agree while leaving cross-operator directional behavior intact. Likewise, a public proxy score or a pseudo-environment statistic can improve without guaranteeing that a model reduces its reliance on the true class–corruption binding. We therefore construct **source-paired operator counterfactuals**: for the same semantic source, we preserve the task label and evaluation severity while intervening on the corruption operator. We then define **Directional Shortcut Alignment (DSA)** to quantify the change in probability mass assigned to the classes that were pre-registered as being bound to each operator during training. DSA is deliberately narrower than a general causal-representation claim. Under semantic preservation, valid operator intervention, pre-registered binding, source-level pairing, and strict separation of evaluation metadata from training and model selection, it identifies a binding-direction operator-response contrast. We further show that operator-invariant source terms cancel in the paired contrast, that DSA has an exchangeable zero baseline and affine behavior in predictive probabilities, and that within-operator consistency does not imply low DSA. A complementary non-identifiability result proves that without assumptions on the proxy-to-true-environment error channel, the same observed label–proxy distribution can correspond to either independent or strongly entangled latent true environments. Thus, proxy improvements alone cannot certify lower CLE harm; target-aligned behavioral evaluation remains necessary.

Using this diagnostic, we first ask *where* the shortcut forms. A matched HFL-versus-Local factorial analysis compares strong CLE with a no-entanglement control under the same local robust-learning recipe. The strong-CLE effect on DSA is 0.119889 in HFL and 0.107960 in Local training, with a pooled communication add-on of 0.011929. The descriptive Local/HFL ratio is 90.05%. These quantities do not constitute a per-example causal mediation decomposition, but they support a **local-first** interpretation in the controlled setting: most of the observed directional shortcut is already present without communication, while communication contributes a smaller additional effect on average. This mechanism result changes the intervention target. Rather than first redesigning the server protocol, we focus on the client-side risk. We train a **Public Environment Witness (PEW)** from public carrier images and an explicit corruption-family taxonomy, freeze it, and use its pseudo-environment predictions on private fit samples. **Balanced Environment Risk (BER)** then rebalances risk mass *within each task class* across pseudo-environment supports. BER is not claimed as a novel generic group-robustness principle: it is a CLE-structure-matched, taxonomy-assisted local intervention whose effective training distribution can be written explicitly. In controlled experiments, PEW+BER reduces DSA on the original binding map, replicates the reduction on another map, and transfers the shortcut-suppression effect to a FedDF-fidelity communication base. On a held-out map with a 40-round matched comparison, PEW+BER achieves lower pooled DSA and higher pooled utility than ERM, matched PEW+GroupDRO, and CVaR-DRO, although CVaR-DRO still attains lower DSA for some individual clients. We therefore separate **shortcut suppression** from **uniform utility improvement** rather than claiming a universal no-harm plugin.

Our contributions are fourfold:

1. **CLE-HFL problem formulation.** We formulate class–corruption entanglement in model-heterogeneous federated learning, where client-specific class–corruption associations can turn corruption operators into directional class cues.
2. **Paired diagnosis and identification boundary.** We introduce a source-paired operator counterfactual protocol and **Directional Shortcut Alignment (DSA)**, together with identification properties, source-level statistical inference, and a proxy non-identifiability result explaining why generic proxy improvements cannot certify reduced CLE harm.
3. **Local-first mechanism attribution.** Through a matched HFL-versus-Local factorial analysis, we show that the observed directional shortcut is predominantly local-first, while communication contributes only a smaller additional effect under the controlled setting.
4. **Taxonomy-assisted local mitigation and controlled validation.** Guided by this mechanism, we develop a taxonomy-assisted local intervention consisting of a **Public Environment Witness** and **Balanced Environment Risk**, and evaluate it against matched ERM, tail-risk, and group-robustness controls across binding maps and HFL communication bases, while explicitly reporting utility and taxonomy boundaries.

We do **not** claim to be the first work on spurious correlation in federated learning, the first to infer pseudo-environments, the first to reweight groups, or the first to use counterfactual shortcut evaluation. Our claim is narrower: CLE-HFL combines a specific client-dependent class–corruption failure mode with a binding-aligned paired diagnostic, a local-versus-communication formation analysis, and a structure-matched local intervention in model-heterogeneous federated learning.

# 4. Related Work

## 4.1 Model-Heterogeneous Federated Learning

Model heterogeneity breaks the parameter-averaging assumption of classical federated optimization. Knowledge-distillation approaches address this mismatch by communicating predictions or distilled knowledge instead of directly averaging all model parameters. FedDF, for example, performs ensemble distillation over public or unlabeled data to fuse heterogeneous client knowledge [REF TO VERIFY]. Other methods communicate class scores, prototypes, generated data, or shared representations under heterogeneous architectures [REF TO VERIFY].

Robust HFL has further considered unreliable or corrupted clients. RHFL studies model-heterogeneous FL with label noise and uses robust local objectives together with confidence-weighted communication. AugHFL, published at ICCV 2023, explicitly studies common data corruption in model-heterogeneous FL and combines local corruption-robust augmentation with robust reweighted communication. RAHFL is the IEEE TPAMI 2025 extension of AugHFL; it augments the local component with diversity-enhanced supervised contrastive learning and introduces asymmetric heterogeneous collaboration to avoid learning from lower-quality external feedback. These works are the closest technical basis for our training pipeline. Our focus is different: rather than asking only how to maintain performance when clients contain corrupted data, we ask whether corruption operators become *client-specific directional class cues* because of class–corruption binding. RAHFL and AugHFL do not provide the source-paired operator DSA diagnostic or the matched HFL-versus-Local attribution studied here.

## 4.2 Corruption Robustness and Consistency Learning

Common-corruption benchmarks such as CIFAR-C and ImageNet-C established standardized evaluations for model robustness to noise, blur, weather, and digital distortions [REF TO VERIFY]. AugMix improves corruption robustness by combining diverse stochastic augmentations with Jensen–Shannon consistency between multiple views [REF TO VERIFY].

This literature typically treats corruption as a nuisance variable to which predictions should be robust. CLE-HFL considers a different statistical structure: the environment itself is associated with task labels in a client-specific way. This distinction also separates DSA from within-view consistency. An AugMix-style objective can enforce

\[
p_\theta(y\mid x_o) \approx p_\theta(y\mid a_1(x_o)) \approx p_\theta(y\mid a_2(x_o)),
\]

while predictions across different operators can still move toward different classes according to the training binding. In our fixed prediction cache, an identical-view construction yields JSD equal to zero while DSA remains 0.119644. Hence, within-operator augmentation consistency does not imply cross-operator binding invariance.

## 4.3 Spurious Correlation, Group Robustness, and Environment Inference

A large literature studies learning under spurious correlations and group distribution shifts. GroupDRO optimizes worst-group risk when group labels are known [REF TO VERIFY]. GEORGE clusters learned representations to recover hidden subclasses before applying group-robust learning [REF TO VERIFY]. EIIL infers environments from violations of invariance without requiring a predefined environment taxonomy [REF TO VERIFY]. JTT identifies misclassified or high-loss examples after an initial ERM model and upweights them during retraining [REF TO VERIFY]. CVaR and related tail-risk objectives emphasize high-loss subsets without explicitly modeling environment support [REF TO VERIFY].

These methods constrain the novelty claim of PEW+BER. Pseudo-environment discovery, group reweighting, worst-group optimization, and difficult-example reweighting are not new ideas. PEW is explicitly **taxonomy-assisted**, not taxonomy-free: it learns a coarse corruption-family witness from public, programmatically labeled carrier images. BER also differs from standard GroupDRO in optimization target. GroupDRO dynamically emphasizes groups with high current loss, whereas BER redistributes *total risk mass within each task class* according to class–pseudo-environment support counts. This difference is empirically testable because PEW+GroupDRO and PEW+BER can share the same pseudo-environment assignments. In the held-out map2 Formal, the shared-PEW GroupDRO control retains substantially higher DSA than BER, indicating that the effect cannot be attributed only to having PEW side information. JTT is included in our screening history only; because the current JTT run is a 12-round screen rather than final scientific evidence, we do not claim a formal empirical victory over JTT in this draft.

## 4.4 Federated Spurious Learning and Counterfactual Diagnosis

Spurious features in federated learning have already been studied. Personalized federated learning can reintroduce or amplify client-specific spurious correlations during local personalization [REF TO VERIFY]. FedPIN studies shortcut-averse invariant learning in personalized FL using causally motivated and information-theoretic regularization [REF TO VERIFY]. FedCD explicitly targets spurious correlation in federated domain generalization [REF TO VERIFY]. These works rule out any broad claim that we are the first to study federated spurious correlation.

Counterfactual shortcut evaluation also has precedents outside our setting. For example, counterfactual analyses in medical imaging investigate whether attributes such as acquisition site or sex are actually used by predictive models rather than merely correlated with outcomes [REF TO VERIFY]. Our contribution is therefore not “counterfactual evaluation” in the abstract. We specialize it to **source-paired corruption-operator interventions**, use a **pre-registered client-specific binding direction**, measure **full predictive probability-mass alignment**, and connect the diagnostic to a matched local-versus-federated formation analysis. To the best of the currently verified publication landscape, we have not identified a formal prior work that jointly studies model-heterogeneous HFL, client-specific class–corruption directional binding, source-paired operator diagnosis, local-first attribution, and a taxonomy-assisted structural mitigation. This sentence should remain conservative and be rechecked in the final literature audit.

# 5. Problem Setup: Class–Corruption Entanglement in HFL

## 5.1 Model-Heterogeneous Federated Learning

We consider a federated system with clients indexed by \(k\in\{1,\dots,K\}\). Client \(k\) maintains a private dataset and a local classifier \(f_k\) whose architecture need not match other clients. In the primary controlled setup, four clients use heterogeneous architectures: ResNet10, ResNet12, ShuffleNet, and MobileNetV2. The private task is CIFAR-10, and client label distributions are non-IID using a Dirichlet partition with concentration \(\alpha=0.5\). The current formal evidence uses one fixed private-data partition unless otherwise stated.

The federated protocol provides a communication mechanism through public information rather than assuming direct parameter compatibility. We use two controlled HFL communication bases in the evidence package: a strict asymmetric HFL route derived from the RAHFL-style pipeline, and a protocol-matched FedDF-fidelity adapter. The second should not be interpreted as a line-by-line reproduction of an official FedDF recipe; it is used as a controlled communication-base change while holding the local intervention contrast fixed.

## 5.2 Class–Corruption Entanglement

Let \(Y\) denote the task class, \(O\) the concrete corruption operator, and \(F=\pi(O)\) its coarse corruption family. Standard corruption-robust learning often implicitly treats corruption and class as approximately independent, e.g.,

\[
P(O=o\mid Y=c)\approx P(O=o).
\]

CLE-HFL instead considers a client-specific dependence

\[
P_k(O=o\mid Y=c)\neq P_k(O=o),
\]

with bindings that can vary across clients. The current CLE-v2 protocol pre-registers a client- and class-specific concrete-operator map

\[
b_k:\mathcal C\rightarrow\mathcal O.
\]

The control setting \(\gamma_{\mathrm{CLE}}=0\) removes class-dependent operator dominance, whereas \(\gamma_{\mathrm{CLE}}=0.9\) makes \(b_k(c)\) a strong but non-deterministic dominant operator for class \(c\) at client \(k\). Both seen and unseen operator roles are retained in evaluation. The earlier four-family CLE construction is a historical precursor and is not the formal problem definition used by the current CLE-v2 experiments.

This operator-level problem definition must be distinguished from the mitigation proxy. The deployed PEW does **not** predict \(O\) or \(b_k(c)\). It predicts only the coarse family proxy \(Z=\hat F\) (plus clean/unknown categories), and BER groups private fit samples by \((Y,Z)\). Fine-grained operator metadata remain unavailable to the training method and are used only by the sealed evaluator.

The central failure hypothesis is directional: if operator \(o\) is associated with particular classes during client training, then applying \(o\) to a source whose true class is different may increase the model’s predictive mass assigned to those bound classes. This behavior cannot be inferred from corrupted accuracy alone.

## 5.3 Controlled CLE-v2 Protocol

The main fixed CLE-v2 scenario uses the following configuration:

- private task: CIFAR-10;
- four heterogeneous clients;
- 40,000 private samples available before protocol-specific sampling;
- Dirichlet \(\alpha=0.5\);
- strong CLE \(\gamma_{\mathrm{CLE}}=0.9\) and control \(\gamma_{\mathrm{CLE}}=0\);
- paired evaluation over 1,000 sources, 15 operators, and four clients;
- evaluation corruption severity 3.

The 40,000 available samples should not be interpreted as meaning that each local model exhaustively traverses all 40,000 distinct images in a finite-round run.

## 5.4 Information and Data-Role Separation

We separate data by role:

- **private fit:** generates local training gradients; PEW+BER group counts are computed only from this fit subset;
- **private audit:** may support strict communication routing but does not produce local-training gradients;
- **final test:** used only for reporting frozen task-level metrics;
- **paired operator grid:** used only after training for operator-level counterfactual evaluation and DSA;
- **public data:** used for PEW training or pre-existing public communication, without private-task ground-truth leakage.

The train-time method does not read private true operator IDs, true corruption family, evaluation severity, seen/unseen flags, or final paired-grid labels for optimization, routing, hyperparameter tuning, or model selection. Binding and operator metadata are read only by the sealed evaluation procedure after the trained model is fixed. This separation is essential because DSA deliberately uses the pre-registered true evaluation binding; without role separation, the diagnostic could be contaminated by target leakage.

# 6. Paired Counterfactual Diagnosis and DSA

## 6.1 Why Accuracy Is Not a Shortcut Diagnostic

A reduction in accuracy under corruption is compatible with many explanations: a corruption may simply destroy task-relevant information; one client may have fewer examples for a class; a smaller architecture may have lower capacity; or non-IID optimization may be unstable. None of these implies that the model has learned a mapping such as “blur implies class A.”

To isolate directional dependence on the corruption operator, we hold the semantic source fixed and intervene only on the operator. For source \(z=(x,y)\) and semantic-preserving transform \(T_o\), the paired evaluation creates

\[
x\longrightarrow \{T_o(x):o\in\mathcal O\}.
\]

This is a controlled counterfactual *evaluation protocol*, not an unconditional claim that all causal properties of the internal model are identified. Its validity depends on the source semantics being preserved, the operator manipulation being well-defined, and the class–operator binding being fixed before evaluation predictions are inspected.

## 6.2 Directional Shortcut Alignment

For client \(k\), let the pre-registered training binding be \(b_k(c)\). For operator \(o\), define the set of classes associated with \(o\) during training as

\[
B_{k,o}=\{c:b_k(c)=o\}.
\]

To avoid a trivial contribution when the source’s true class already belongs to that set, DSA for operator \(o\) uses sources with \(y\notin B_{k,o}\). Define the bound probability mass

\[
m_{k,o}(z,o')=
\sum_{c\in B_{k,o}} p_k(c\mid T_{o'}(x)).
\]

The source-level directional contrast is

\[
d_{k,o}(z)=m_{k,o}(z,o)
-\frac{1}{|\mathcal O|-1}\sum_{o'\neq o}m_{k,o}(z,o').
\]

We define

\[
\operatorname{DSA}_{k,o}
=\mathbb E_z[d_{k,o}(z)],
\]

and compute pooled DSA by equally averaging valid operator-level values and then client-level values. DSA lies on a probability scale. For example, DSA \(=0.12\) corresponds to roughly a 12-percentage-point difference in predictive mass toward the bound class set, not a 0.12-percentage-point accuracy difference.

The interpretation is intentionally narrow:

> For the same semantic source, does applying operator \(o\) shift prediction probability toward classes that were bound to \(o\) in this client’s training data, relative to applying other operators?

## 6.3 Shuffled-Binding Null

Certain corruptions might systematically change predictions toward particular classes even if those directions have nothing to do with the training binding. To test binding specificity, we preserve the number of classes assigned to each operator but randomly permute the binding map and recompute DSA on the same fixed prediction cache. Under the primary strong-CLE HFL checkpoint, true-binding DSA is 0.119644, while the shuffled-binding null has p95 = 0.029559, with permutation \(p=0.000999\). This comparison helps rule out the alternative explanation that the observed DSA merely reflects generic operator-to-class response patterns unrelated to the pre-registered training association.

# 7. DSA Identification and Proxy Non-identifiability

## 7.1 Identification Target Under Paired Cancellation

Assume that for client \(k\), target operator \(o\), source \(z\), and applied operator \(o'\), the bound predictive mass can be decomposed as

\[
m_{k,o}(z,o')=s_{k,o}(z)+r_{k,o}(z,o'),
\]

where \(s_{k,o}(z)\) is an operator-invariant source term and \(r_{k,o}(z,o')\) is the operator-dependent response. Substituting into the source-level contrast yields

\[
d_{k,o}(z)=
 r_{k,o}(z,o)
-\frac{1}{|\mathcal O|-1}\sum_{o'\neq o}r_{k,o}(z,o').
\]

The source term cancels exactly. Under this decomposition, DSA identifies a **binding-direction operator-response contrast**. This result does not prove that the model relies only on corruption, that no operator changes any task semantics, or that the identified response is harmful under every deployment distribution. Those stronger claims require assumptions or evidence beyond the paired algebra.

## 7.2 Zero Baseline and Affine Property

If, for the same source, predictions are exchangeable across operators,

\[
p(\cdot\mid T_o(x))=p(\cdot\mid T_{o'}(x)),\quad \forall o,o',
\]

then the DSA contrast is exactly zero. The fixed-cache exchangeable projection yields \(-2.50\times10^{-20}\), while the largest empirical absolute DSA among the \(\gamma_{\mathrm{CLE}}=0\) controls is 0.001914.

DSA is also affine in predictive probabilities. For

\[
p_\lambda=(1-\lambda)p_0+\lambda p_1,
\]

we have

\[
\operatorname{DSA}(p_\lambda)
=(1-\lambda)\operatorname{DSA}(p_0)
+\lambda\operatorname{DSA}(p_1).
\]

When \(p_1\) introduces additional binding-aligned response relative to \(p_0\), DSA increases monotonically with the mixture strength. An 11-point fixed-cache validation has maximum affine numerical error \(2.78\times10^{-17}\).

## 7.3 Controlled Recovery and Paired Cancellation

In a controlled probability-simplex construction, we inject a binding-aligned predictive response of known strength. Across 11 injection levels, DSA recovers the target directional strength with maximum numerical error \(6.66\times10^{-16}\). Adding the same probability displacement to every operator view of the same source changes DSA by only \(8.33\times10^{-17}\), numerically confirming cancellation of operator-invariant response.

For the same controlled response, true-binding DSA is 0.241071, compared with shuffled-binding null p95 of 0.026786 and permutation \(p=0.000999\). These tests validate the implementation’s ability to distinguish a response aligned with the frozen binding from an equal-scale response with randomized binding direction.

## 7.4 Why JSD Cannot Replace DSA

A model can be perfectly consistent among augmentations around one corrupted observation while still using the corruption operator as a class cue. Construct three identical predictive views around each operator. The within-operator JSD is then exactly zero, yet the fixed strong-CLE HFL cache still has DSA 0.119644. Therefore,

\[
\text{within-operator consistency}
\not\Rightarrow
\text{cross-operator directional invariance}.
\]

This does not imply that JSD is ineffective for corruption robustness. It establishes that JSD and DSA diagnose different objects: local augmentation consistency versus binding-direction cross-operator response.

## 7.5 Source-Level Statistical Inference

The statistical unit is the **source**, not each corrupted image. The 15 operator views of one source share content and therefore are not independent observations. We first compute the directional contrast within a source and then resample across sources. If source-level contrasts are independent and bounded in \([-1,1]\), a conservative Hoeffding radius is

\[
\epsilon(n,\delta)
=\sqrt{\frac{2\log(2/\delta)}{n}}.
\]

For \(n=1000\) and \(\delta=0.05\), this radius is 0.085894. Main DSA uncertainty intervals use source-clustered bootstrap. Crucially, these intervals condition on the trained checkpoint: they cover evaluation-source sampling uncertainty, **not** training-seed uncertainty, partition uncertainty, binding-map uncertainty, model-selection uncertainty, or cross-dataset uncertainty.

## 7.6 Proxy-Only Non-identifiability

BER acts on a proxy environment \(Z=\hat E\) predicted by PEW, whereas CLE ultimately concerns a latent true corruption environment \(E\) and the resulting model behavior. We therefore ask whether a reduction in the observable dependence \(P(Y,Z)\) is sufficient to certify a reduction in true dependence \(P(Y,E)\).

For discrete variables \(A,B\), define dependence by total variation from independence,

\[
\mathcal D(A,B)
=TV(P_{A,B},P_AP_B)
=\frac12\sum_{a,b}|P(a,b)-P(a)P(b)|.
\]

**Theorem 1 (proxy-only non-identifiability).** Fix any observable joint distribution \(P(Y,Z)\), with at least two positive-probability values of \(Y\), and place no constraint on the unknown error channel \(P(Z\mid E,Y)\). Then there exist two complete latent data-generating distributions with exactly the same observable \(P(Y,Z)\) but different true environment dependence: one with \(\mathcal D(Y,E)=0\), and another with

\[
\mathcal D(Y,E)=1-\sum_eP(E=e)^2>0.
\]

The constructive proof keeps \(P(Z\mid Y)\) unchanged. In one world, \(E\) is chosen independent of \((Y,Z)\); in the other, \(E=g(Y)\) for a nonconstant mapping \(g\). Because the observable distribution is identical in both worlds, no statistic that reads only \(P(Y,Z)\) can unconditionally identify the true environment dependence.

For the balanced ten-class, five-environment construction used in our cache validation, the latent dependent world has \(\mathcal D(Y,E)=0.8\) while the independent world has 0, despite identical observable \(P(Y,Z)\). This construction proves non-identifiability; it is not intended to simulate a natural corruption-generation process.

## 7.7 Conditional Bridge and the Need for Target-Aligned Evaluation

A conditional transfer bound remains valid. Under a sample-wise coupling of true and proxy environments,

\[
\mathcal D_Q(Y,E)
\le
\mathcal D_Q(Y,Z)+2\epsilon_Q,
\qquad
\epsilon_Q=P_Q(Z\neq E).
\]

Thus, a useful true-environment guarantee requires both low observable dependence and a sufficiently controlled proxy error under the BER effective distribution. In the current experiment, the BER-weighted PEW family error lies in 0.507–0.575, causing the bound to truncate at the trivial value 1.0. We therefore do not claim a nontrivial unconditional theoretical guarantee that BER makes the true environment independent of the class.

The non-identifiability theorem has an important evaluation consequence. If a training method optimizes only a proxy objective that does not access the true frozen evaluation binding, then proxy improvement alone cannot certify lower CLE directional harm without additional identification assumptions. A CLE-mitigation claim therefore needs a **target-aligned behavioral estimand** that reads the frozen evaluation binding and compares same-source responses across operators. Our paper uses paired DSA for this purpose. This “necessity” is specific to the type of evidence required by our protocol; we do not claim DSA is the only possible shortcut metric in general.

# 8. Local-First Mechanism Attribution

## 8.1 Matched HFL-versus-Local Factorial

To determine whether the shortcut is mainly created by federation or by client-side training, we compare four matched arms under the CLE-v2 controlled scenario:

- HFL with \(\gamma_{\mathrm{CLE}}=0\),
- HFL with \(\gamma_{\mathrm{CLE}}=0.9\),
- Local training with \(\gamma_{\mathrm{CLE}}=0\),
- Local training with \(\gamma_{\mathrm{CLE}}=0.9\).

All four arms use the same AugMix/JSD/DCL local baseline and differ only in federation condition and CLE strength. The Formal mechanism stage uses training seed 0, 12 rounds, and at most 16 local batches per client per round.

| Arm | Scope | \(\gamma_{\mathrm{CLE}}\) | Operator-grid Acc. (%) | Pooled DSA |
|---|---|---:|---:|---:|
| HFL control | HFL | 0.0 | 24.9300 | -0.000245 |
| HFL strong CLE | HFL | 0.9 | 21.4367 | 0.119644 |
| Local control | Local | 0.0 | 25.5683 | -0.001914 |
| Local strong CLE | Local | 0.9 | 22.4233 | 0.106046 |

The corresponding matched contrasts are

\[
\Delta_{\mathrm{HFL}}
=0.119644-(-0.000245)=0.119889,
\]

\[
\Delta_{\mathrm{Local}}
=0.106046-(-0.001914)=0.107960,
\]

and

\[
\Delta_{\mathrm{comm}}
=\Delta_{\mathrm{HFL}}-\Delta_{\mathrm{Local}}
=0.011929.
\]

The source-bootstrap 95% intervals are [0.118050, 0.121562] for the HFL CLE effect, [0.106145, 0.109678] for the Local CLE effect, and [0.011115, 0.012712] for the communication add-on.

## 8.2 Interpretation: Predominantly Local-First

The descriptive ratio

\[
\frac{\Delta_{\mathrm{Local}}}{\Delta_{\mathrm{HFL}}}=90.05\%
\]

indicates that most of the pooled CLE-associated DSA observed under HFL is already present in matched Local training. We call this **local-first formation**. The term is deliberately descriptive: 90.05% is not a per-example causal mediation share, and the factorial does not prove that communication is irrelevant. Federation still defines the multi-client, non-IID, model-heterogeneous learning system and can change client-specific outcomes. The finding is narrower: under this controlled scenario and matched recipe, the pooled communication contribution is much smaller than the local CLE effect.

This result motivates a local intervention. If the dominant shortcut emerges during local optimization, then redesigning the communication mechanism alone is not the most direct response to this failure mode.

# 9. Taxonomy-Assisted PEW and Balanced Environment Risk

## 9.1 Design Principle

The mitigation is intentionally local. It does not claim a new universal communication protocol. Instead, we seek side information about coarse corruption environments without reading private true corruption metadata, then use that information to reduce the class-conditional support imbalance that generates CLE.

## 9.2 Public Environment Witness

The **Public Environment Witness (PEW)** is trained only on public CIFAR-100 carrier images with programmatically generated corruption supervision. It is frozen as a **coarse family-level witness**, not an operator classifier. The predefined environment taxonomy is

\[
\{\text{clean},\text{noise},\text{blur},\text{weather},\text{digital},\text{unknown}\}.
\]

The unknown class is constructed by sequentially composing two distinct base corruption families. Severity ranges from 1 to 5. A small CNN predicts the six environment classes, five severity levels, and a 32-dimensional embedding, and is trained with environment cross-entropy plus 0.25 times severity cross-entropy using Adam with learning rate \(10^{-3}\) for five epochs. The best checkpoint and unknown threshold are selected using public validation only.

After training, PEW is frozen. For each private fit sample \(i\), it produces a hard pseudo-environment \(\hat e_i\). PEW never reads private true operator IDs or true family labels during train-time deployment. It is therefore private-label-free with respect to the corruption metadata, but it is **not taxonomy-free**: its meaning and supervision are explicitly defined by a hand-specified corruption-family taxonomy. We treat PEW as side-information infrastructure rather than a standalone architecture contribution.

The granularity mismatch is deliberate: **CLE-v2 and DSA are operator-level, whereas PEW and BER are family-level.** DSA measures the fine-grained target harm that the training method is not allowed to read; PEW supplies only a coarser deployable proxy. The later Oracle family/operator comparison is a non-deployable boundary audit, not an alternative operator-level PEW implementation.

## 9.3 Balanced Environment Risk

For client \(k\), task class \(c\), and pseudo-environment \(e\), let

\[
n_{k,c,e}=\#\{i:y_i=c,\hat e_i=e\},
\]

and

\[
R_{k,c,e}
=\frac{1}{n_{k,c,e}}
\sum_{i:y_i=c,\hat e_i=e}\ell_i.
\]

The hard BER weight is

\[
a^{(\gamma_{\mathrm{BER}})}_{k,c,e}
=\frac{
\mathbf 1[n_{k,c,e}\ge m]\min(n_{k,c,e},K_{\mathrm{cap}})^{\gamma_{\mathrm{BER}}}
}{
\sum_{e'}\mathbf 1[n_{k,c,e'}\ge m]
\min(n_{k,c,e'},K_{\mathrm{cap}})^{\gamma_{\mathrm{BER}}}
},
\]

with current Formal configuration

\[
\gamma_{\mathrm{BER}}=0.5,\qquad K_{\mathrm{cap}}=32,\qquad m=2.
\]

The client objective is

\[
R_{\mathrm{BER},k}
=\frac{1}{|\mathcal C_k^{\mathrm{valid}}|}
\sum_{c\in\mathcal C_k^{\mathrm{valid}}}
\sum_e a^{(\gamma_{\mathrm{BER}})}_{k,c,e}R_{k,c,e}.
\]

BER should not be interpreted as “optimize the worst environment.” Rather, it reduces the total-risk dominance created by large pseudo-environment support within each task class.

## 9.4 Effective-Distribution Equivalence

BER is exactly equivalent to empirical risk minimization under an effective sample distribution with mass

\[
q_i
=\frac{a^{(\gamma_{\mathrm{BER}})}_{k,y_i,\hat e_i}}
{|\mathcal C_k^{\mathrm{valid}}|n_{k,y_i,\hat e_i}}.
\]

A fixed-data CPU audit reproduces the class–environment joint mass of the implemented training objective with maximum error \(7.32\times10^{-16}\).

For two valid environments within a class,

\[
\frac{Q_{\gamma}(e_1\mid c)}{Q_{\gamma}(e_2\mid c)}
=\left(
\frac{\min(n_{c,e_1},K_{\mathrm{cap}})}
{\min(n_{c,e_2},K_{\mathrm{cap}})}
\right)^{\gamma_{\mathrm{BER}}}.
\]

In the uncapped region,

\[
\log\frac{Q_{\gamma}(e_1\mid c)}{Q_{\gamma}(e_2\mid c)}
=\gamma_{\mathrm{BER}}
\log\frac{n_{c,e_1}}{n_{c,e_2}}.
\]

Hence \(0\le\gamma_{\mathrm{BER}}<1\) compresses the class-conditional environment log-support advantage. With \(\gamma_{\mathrm{BER}}=0.5\), \(K_{\mathrm{cap}}=32\), and \(m=2\), the maximum theoretical effective-mass ratio between valid pseudo-environment groups is \(\sqrt{32/2}=4\). In the four-client audit, pre-reweighting support ratios range from 116.5 to 205.67, while the BER effective ratio is 4.

Under the idealized special case \(\gamma_{\mathrm{BER}}=0\) and common environment support across classes,

\[
Q_0(\hat E=e\mid Y=c)=Q_0(\hat E=e),
\]

so \(Y\perp\hat E\) under the effective distribution. This is an ideal corollary, not the current operating point. BER cannot create missing class–environment groups, and the current \(\gamma_{\mathrm{BER}}=0.5\) configuration trades off support correction, variance, and pseudo-label noise.

## 9.5 Fixed-Data Mechanism Audit

On the strict-fit data used by the current Formal protocol, BER changes the equal-client dependence statistics as follows:

| Object | Before | BER effective | Relative drop |
|---|---:|---:|---:|
| \(TV(Y,\hat E)\) | 0.487505 | 0.199744 | 59.03% |
| \(TV(Y,E_{\text{true-family}})\) | 0.634914 | 0.433513 | 31.72% |
| True-family environment-only Bayes advantage | 0.313840 | 0.240863 | 23.25% |

The true-family quantities are reporting-only audits; true corruption metadata do not enter the deployable training rule. These results explain why BER is structurally aligned with CLE support imbalance, but they do **not** imply that post-training DSA must decrease or that task accuracy must improve. Model behavior is verified separately through the Formal training experiments and paired DSA.

## 9.6 Relation to GroupDRO, CVaR, and Consistency Learning

BER, GroupDRO, CVaR, and JSD operate on distinct objects:

- **JSD:** prediction consistency among nearby augmentations of the same corrupted observation;
- **CVaR-DRO:** high-loss tail risk without an explicit class–environment support model;
- **GroupDRO:** dynamic emphasis on the currently high-risk groups under a defined grouping;
- **BER:** class-conditional redistribution of total risk mass according to pseudo-environment support;
- **PEW:** a taxonomy-assisted estimator of coarse environment side information.

These conceptual distinctions are complemented by matched empirical controls in Section 10.

# 10. Experiments

## 10.1 Research Questions

We organize experiments by scientific question rather than run chronology:

- **RQ1:** Does strong CLE induce a binding-specific directional shortcut?
- **RQ2:** Is the shortcut mainly formed during local optimization or by federated communication?
- **RQ3:** Can taxonomy-assisted PEW+BER reduce DSA, and what happens to task utility?
- **RQ4:** Is the observed benefit explained by generic difficult-sample, tail-risk, or group-robustness reweighting?
- **RQ5:** Does shortcut suppression replicate across binding maps and HFL communication bases?
- **RQ6:** How important are correct environment correspondence and environment granularity?

Pure-method multi-training-seed stability, a second private dataset, and a bounded taxonomy stress test remain pending and are explicitly separated from completed evidence.

## 10.2 Metrics and Uncertainty

The primary shortcut metric is DSA. We also report operator-grid accuracy and frozen task-level utility metrics. Let \(A_{c,o}\) be the accuracy in a valid class–operator cell, aggregated under the evaluation protocol. Then

\[
\mathrm{WCCA}=\min_{(c,o):n_{c,o}>0} A_{c,o},
\]

and

\[
\mathrm{CFG}=\frac{1}{|\mathcal C_{\mathrm{valid}}|}
\sum_{c\in\mathcal C_{\mathrm{valid}}}
\left(\max_o A_{c,o}-\min_o A_{c,o}\right).
\]

WCCA is the worst valid class–operator-cell accuracy and is higher-is-better. CFG is the mean within-class accuracy range across operators and is lower-is-better. Avg is the mean client accuracy, and Worst is the minimum client accuracy.

All DSA bootstrap intervals described below resample at the source level. They should not be interpreted as uncertainty over training seeds or partitions.

## 10.3 RQ1–RQ2: Strong CLE and Local-First Formation

The four-arm mechanism experiment in Section 8 provides the primary evidence. HFL DSA changes from \(-0.000245\) at \(\gamma_{\mathrm{CLE}}=0\) to 0.119644 at \(\gamma_{\mathrm{CLE}}=0.9\), while Local DSA changes from \(-0.001914\) to 0.106046. The HFL CLE effect is 0.119889, Local CLE effect is 0.107960, and the communication add-on is 0.011929. The true-binding strong-CLE HFL DSA also exceeds the shuffled-binding null p95 of 0.029559 with permutation \(p=0.000999\).

These results support two conclusions within the fixed controlled scenario: (i) strong CLE produces a binding-specific directional probability response, and (ii) the pooled effect is predominantly local-first rather than primarily manufactured by communication.

## 10.4 RQ3: PEW+BER on the Asymmetric HFL Base

Under the original strong-CLE map, the matched baseline and PEW+BER arms use the same strict HFL communication base and matched local data trajectory. The baseline uses the existing robust local recipe, while the intervention replaces the standard task risk with the PEW-grouped BER risk while preserving the other frozen local components.

The primary DSA result is

\[
0.119644\rightarrow0.041252,
\]

an absolute reduction of 0.078392 and a relative reduction of 65.52%, with source-bootstrap 95% interval [0.077191, 0.079601] for the DSA difference. Per-client DSA reductions are [0.111312, 0.011317, 0.115762, 0.075178].

The last-five-round intervention-minus-baseline task deltas are:

- Avg: +1.7323 percentage points;
- Worst: -0.3760 points;
- WCCA: +0.3500 points;
- CFG: -9.2600 points.

The shortcut-mitigation criterion passes, while the frozen full-utility criterion does not pass because the Worst metric fails its pre-registered threshold. We therefore interpret this experiment as **strong shortcut suppression with non-uniform utility**, not as a universal no-harm improvement.

A client-level analysis further illustrates why DSA and utility must remain separate. One client shows a 5.1733-point operator-grid accuracy decrease while its DSA still falls by 0.115762. Because model architecture and non-IID client data are confounded by design, we do not attribute this effect causally to model capacity or to the specific architecture.

## 10.5 RQ4: Held-Out Map2 Matched Controls

A five-arm 12-round screen was used only for method selection and protocol design; it is not treated as final scientific evidence. In particular, its JTT result is not used to claim a formal victory over JTT.

After screening, a held-out map2 was kept unseen and evaluated under a frozen 40-round four-arm protocol with ERM, CVaR-DRO, PEW+GroupDRO, and PEW+BER. The four arms share the partition, training/evaluation seed, initial states, private batch trajectory, and strict HFL communication protocol. The resulting pooled metrics are:

| Arm | DSA ↓ | Operator-grid Acc. ↑ | Last-10 Avg ↑ | Last-10 Worst ↑ | WCCA ↑ | CFG ↓ |
|---|---:|---:|---:|---:|---:|---:|
| ERM | 0.260308 | 20.3067 | 19.8578 | 16.3007 | 0.000 | 32.8325 |
| CVaR-DRO | 0.088982 | 20.4567 | 18.6700 | 14.0380 | 0.125 | 27.7775 |
| PEW+GroupDRO | 0.214575 | 20.7883 | 20.9562 | 17.8140 | 0.025 | 31.5275 |
| **PEW+BER** | **0.077741** | **23.7433** | **24.4518** | **20.4727** | **2.650** | **22.8025** |

Source-paired bootstrap comparisons are:

\[
\text{DSA(ERM)}-\text{DSA(BER)}=0.182567,
\]

95% interval [0.180596, 0.184407];

\[
\text{DSA(GroupDRO)}-\text{DSA(BER)}=0.136834,
\]

95% interval [0.134952, 0.138706]; and

\[
\text{DSA(BER)}-\text{DSA(CVaR)}=-0.011240,
\]

95% interval [-0.012355, -0.010092].

The matched PEW+GroupDRO comparison is important because the two methods receive the same PEW grouping information. The large DSA gap indicates that the BER result cannot be explained only by access to PEW side information. Relative to CVaR-DRO, BER has lower pooled DSA and substantially higher pooled task utility in this held-out Formal.

However, the conclusion must remain at the pooled level. BER reduces DSA relative to ERM and PEW+GroupDRO on all four clients, but relative to CVaR-DRO, clients c0 and c1 still have lower DSA under CVaR-DRO, whereas c2 and c3 favor BER. Operator-grid accuracy is higher for BER on all four clients. We therefore state:

> **PEW+BER achieves a better pooled shortcut–utility trade-off than CVaR-DRO on held-out map2, without uniformly dominating CVaR-DRO in per-client DSA.**

## 10.6 RQ5: Cross-Binding-Map Replication

To test whether the result depends on one particular class–operator assignment, we change only the client-specific binding map while fixing the data partition, evaluation seed, training seed, initial weights, public data, evaluation grid, and frozen PEW.

On map1,

\[
\text{Base DSA}=0.113761,
\qquad
\text{PEW+BER DSA}=0.049531,
\]

for an absolute reduction of 0.064230 (56.46%) with source-bootstrap 95% interval [0.063105, 0.065328]. All four clients reduce DSA, with reductions [0.064195, 0.047682, 0.085614, 0.059428]. Operator-grid pooled accuracy changes from 20.9450% to 21.4583%. Last-five-round deltas are Avg +0.7040, Worst +1.0760, WCCA +0.3500, and CFG -9.8550.

This result supports replication across two different controlled binding directions. It does **not** establish generalization across new partitions, training seeds, corruption libraries, severities, datasets, or real deployment environments.

## 10.7 RQ5: Shortcut Suppression on a Second Communication Base

We next test whether the DSA reduction depends on the RAHFL-style local/communication stack. The second experiment uses a native single-view CE baseline and a protocol-matched **FedDF-fidelity** communication adapter. The matched intervention differs in the local task risk by using PEW-grouped hard BER weighting; initialization, private batches, public distillation, and training budget are otherwise held fixed.

The pooled DSA changes from

\[
0.136989\rightarrow0.029012,
\]

an absolute reduction of 0.107977 and a relative reduction of 78.82%, with source-bootstrap 95% interval [0.106498, 0.109417]. All four clients reduce DSA, with reductions [0.150575, 0.074155, 0.053119, 0.154059].

Task utility is mixed. Operator-grid pooled accuracy changes from 18.4917% to 18.4517%, a -0.0400-point difference. Last-five-round deltas are Avg -0.6660, Worst +0.2147, WCCA -0.1000, and CFG -13.2350. Both arms are below the pre-registered 20% operator-grid learning floor, and the overall frozen utility gate does not pass.

The correct conclusion is therefore:

> **DSA suppression generalizes across two HFL communication bases, but consistent utility improvement does not.**

We do not describe the FedDF-fidelity result as a lossless or uniformly successful plugin experiment. Absolute metric values across the original-map, map1, map2, and FedDF-fidelity experiments are not directly comparable because their controlled recipes differ; only the matched within-setting contrasts support attribution.

## 10.8 RQ6: Environment Correspondence and Granularity

We compare three non-deployable or control grouping schemes under the same strong-CLE setting. None of them changes the frozen family-level PEW used by the final method:

- **Oracle family:** use private true corruption family for BER grouping;
- **Oracle operator:** use private true corruption operator for BER grouping;
- **Random operator:** permute operator correspondence within class before grouping.

| Grouping | Pooled DSA | Operator-grid Avg | Last-5 Avg | Last-5 Worst |
|---|---:|---:|---:|---:|
| Oracle family | 0.017232 | 21.9333 | 21.0960 | 17.8400 |
| Oracle operator | 0.015783 | 21.5200 | 20.0273 | 15.8827 |
| Random operator | 0.067895 | 18.7817 | 18.3933 | 14.7867 |

The DSA gap between Random and Oracle operator is 0.052112 with 95% interval [0.051231, 0.052983], showing that meaningful environment correspondence matters. The Oracle family–operator DSA gap is only 0.001449 with 95% interval [0.001086, 0.001790] in this controlled setup. This result does **not** prove that operator-level environment models are universally useless. It means that the current Oracle granularity experiment does not provide sufficient evidence to justify further development of an operator-level PEW for the present paper. Accordingly, operator-level PEW is outside the final method and outside the remaining experiment plan.

## 10.9 Pending Training-Seed Stability

**[EVIDENCE NEEDED: Pure PEW+BER multi-training-seed stability.]**

The current narrow source-bootstrap intervals condition on a fixed trained checkpoint and do not quantify training randomness. Before final submission, the main matched PEW+BER comparisons should be repeated across independently initialized training seeds under a frozen protocol. The final paper should report seed-level DSA and task utility, together with mean/dispersion across seeds. No values should be inserted until those runs are complete.

## 10.10 Pending Second Private Dataset

**[EVIDENCE NEEDED: Second private dataset.]**

The current core evidence is built on a controlled CIFAR-10 private task with CIFAR-style common corruptions. A second private dataset is required to test whether CLE formation, DSA diagnosis, local-first attribution, and PEW+BER mitigation persist beyond this single task family. The exact dataset, corruption construction, and matched baseline protocol are not yet frozen and must not be precommitted in this draft.

## 10.11 Pending Taxonomy Stress Test

**[EVIDENCE NEEDED: bounded taxonomy stress test.]**

The present method is closed-set and taxonomy-assisted. A bounded stress test involving held-out, compound, or otherwise partially unmatched corruptions would help characterize degradation when PEW’s public taxonomy is incomplete. The protocol is not yet frozen. This experiment is a scope/boundary test rather than a requirement to claim arbitrary open-world robustness.

# 11. Discussion and Limitations

## 11.1 The Contribution Is an Evidence Chain, Not a New Backbone

PEW is a small public corruption witness, and BER belongs to the broad family of non-uniform risk weighting. We therefore do not position the work as a new network-architecture paper. The main contribution is the closed scientific chain: a concrete CLE-HFL failure mode, a target-aligned paired diagnostic, an analysis of where the shortcut forms, a local intervention derived from the support structure, and matched validation against alternative explanations.

## 11.2 Shortcut Suppression and Task Utility Are Distinct

Removing a shortcut can reduce performance on the same biased distribution if the shortcut is predictive there. For this reason, DSA reduction and in-distribution utility should not be collapsed into one criterion. The Asymmetric-HFL result shows strong DSA reduction with mixed per-client utility, and the FedDF-fidelity result shows that large DSA suppression can transfer while the aggregate utility gate still fails. The held-out map2 Formal provides a more favorable shortcut–utility trade-off, but even there BER does not uniformly dominate CVaR-DRO for every client’s DSA.

This distinction prevents two opposite overclaims: low DSA alone does not prove a universally better task model, and high task accuracy alone does not prove that the model has stopped exploiting the CLE binding.

## 11.3 Taxonomy-Assisted Rather Than Taxonomy-Free

PEW requires a predefined corruption-family taxonomy and public programmatic supervision. This makes the intervention interpretable and keeps private true environment labels out of training, but it also limits scope. Taxonomy-free environment inference methods such as EIIL target a different setting [REF TO VERIFY]. Our paper does not claim to solve arbitrary latent-environment discovery.

The current proxy-error transfer bound is also intentionally weak: under the BER effective distribution, PEW family error is large enough that the bound becomes trivial. The proxy non-identifiability theorem explains why this limitation cannot be removed by wording alone. Without additional assumptions on the proxy error channel, proxy support statistics cannot certify the true environment dependence. This is why we retain paired DSA as a target-aligned evaluation endpoint.

## 11.4 Synthetic CLE and External Validity

The controlled class–corruption map is designed to isolate a mechanism. Similar associations can plausibly arise when task labels co-occur with acquisition devices, sites, weather, compression pipelines, scanners, or production conditions, but the current experiments do **not** demonstrate effectiveness in real hospitals, autonomous vehicles, factories, or other operational systems. Such domains may contain continuous, compound, semantic-changing, or previously unseen environmental variation that is not represented by the current CIFAR-C-style taxonomy.

A second private dataset and a bounded taxonomy stress test remain the most direct additions for broadening external validity without changing the core method. Real-world deployment validation remains future work.

## 11.5 Training Randomness and Partition Scope

The strongest Formal comparisons still use a fixed private partition and key runs at training seed 0. Source-level bootstrap quantifies evaluation-source uncertainty only. It does not establish stability across random initialization, stochastic optimization, alternative non-IID partitions, or new datasets. **[EVIDENCE NEEDED: pure PEW+BER multi-training-seed Formal.]**

Cross-binding-map replication already shows that the conclusion does not depend on one specific class–operator map, but this should not be generalized to cross-partition or cross-domain robustness.

## 11.6 Architecture–Client Confounding

Each client is associated with both a particular non-IID data slice and a specific architecture. Consequently, client-level differences cannot be causally attributed to architecture capacity alone. In particular, a poor utility outcome on the ShuffleNet client does not establish that smaller architectures are intrinsically harmed by BER. Answering that question would require an architecture-by-data crossover design, which is outside the current submission scope.

## 11.7 Counterfactual Language and Causal Scope

Our use of “counterfactual” refers to a controlled same-source operator intervention. DSA identifies a binding-direction response under semantic-preservation, valid-intervention, pre-registration, and evaluation-isolation assumptions. We do not claim that it recovers the model’s complete internal causal mechanism or that it is a sufficient statistic for all forms of spurious correlation. Operator-specific semantic leakage would violate the intended interpretation and must be controlled by the transformation design and sensitivity analysis rather than assumed away.

## 11.8 Environment Granularity

The Oracle family/operator experiment indicates a very small additional DSA reduction from family-level to operator-level grouping under the current controlled protocol. The appropriate conclusion is that **the present evidence does not justify further operator-level PEW development for this paper**. This is a local design decision under the observed setting, not a universal statement that operator-level environment modeling can never be useful.

# 12. Conclusion

We study class–corruption entanglement in model-heterogeneous federated learning, a controlled failure mode in which client-specific class–corruption associations can turn corruption operators into directional class cues. To separate this behavior from ordinary corruption difficulty, we introduce a source-paired operator counterfactual protocol and Directional Shortcut Alignment, a target-aligned probability-mass contrast with explicit identification assumptions and source-level inference. A proxy non-identifiability result further shows why improvements in pseudo-environment or generic public-response statistics cannot, without additional assumptions, certify reduced true CLE harm.

A matched HFL-versus-Local factorial indicates that the observed directional shortcut is predominantly local-first in the controlled setup, motivating intervention at the client objective rather than the communication protocol. We therefore combine a taxonomy-assisted Public Environment Witness with Balanced Environment Risk, which compresses class-conditional pseudo-environment support imbalance. Controlled experiments show substantial DSA reduction across binding maps and transfer of shortcut suppression to two HFL communication bases. On a held-out 40-round map, PEW+BER yields a favorable pooled shortcut–utility trade-off relative to ERM, matched PEW+GroupDRO, and CVaR-DRO, while not uniformly dominating CVaR-DRO at every client. The method remains taxonomy-assisted, and utility gains are not universal.

The remaining evidence gaps are explicit: **[EVIDENCE NEEDED]** multi-training-seed stability for the pure PEW+BER method, a second private dataset, and a bounded taxonomy stress test. These gaps affect external validity and stability claims, but they do not motivate adding a new loss, a new communication module, or a more complex PEW architecture to the current paper.

---

# A. Contribution-to-Evidence Mapping

| Contribution | Core evidence already available | Boundary that must remain explicit | Pending evidence |
|---|---|---|---|
| **C1. CLE-HFL formulation** | Client-specific class–operator binding, \(\gamma_{\mathrm{CLE}}=0\) vs 0.9 control, strict data-role isolation, paired operator grid | Controlled synthetic CLE; not a real-world deployment claim | Second private dataset; optional bounded taxonomy stress test |
| **C2. Paired DSA + identification + proxy non-identifiability** | DSA definition; exchangeable zero; affine mixture; paired cancellation; controlled injection recovery; shuffled-binding null; source-level bootstrap; JSD=0/DSA>0; proxy two-world theorem; CVRS proxy-down/DSA-up counterexample | DSA identifies binding-direction response under explicit assumptions, not full internal causality or a universal shortcut statistic | No new theory module needed; tighten theorem statements and proof appendix |
| **C3. Local-first attribution** | HFL CLE effect 0.119889; Local CLE effect 0.107960; communication add-on 0.011929; descriptive ratio 90.05% | Not per-example causal mediation; communication still matters to the HFL system | Multi-seed repetition would improve stability, but do not reinterpret source CI |
| **C4. Taxonomy-assisted PEW+BER and controlled validation** | Asymmetric-HFL DSA reduction 65.52%; map1 reduction 56.46%; FedDF-fidelity reduction 78.82%; held-out map2 four-arm Formal; matched PEW+GroupDRO; pooled CVaR comparison; Oracle correspondence/granularity; BER effective-distribution audit | PEW is taxonomy-assisted; FedDF utility gate fails; BER does not uniformly dominate CVaR per client; Oracle is non-deployable | Pure-method multi-seed; second private dataset; bounded taxonomy stress test |

# B. Figure/Table Placement Plan

## Figure 1 — Problem and evidence-chain overview

**Placement:** end of Introduction or beginning of Problem Setup.

**Content:**

```text
client-specific class ↔ corruption binding
        ↓
source-paired operator intervention
        ↓
Directional Shortcut Alignment (DSA)
        ↓
HFL vs Local matched attribution
        ↓
taxonomy-assisted PEW
        ↓
BER class-conditional support balancing
        ↓
original map / map1 / held-out map2 / second HFL base
        ↓
explicit utility + taxonomy + external-validity boundaries
```

The figure should visually separate *diagnosis* from *training*: DSA reads the frozen evaluation binding after model sealing, whereas PEW+BER does not read private true corruption metadata during training.

## Figure 2 — DSA identification schematic

**Placement:** Section 6 or 7.

**Panels:**

1. same source under multiple operators;
2. predictive mass moving toward operator-bound classes;
3. true binding versus shuffled binding;
4. operator-invariant source term canceling in the paired contrast.

Optional inset: JSD = 0 but DSA = 0.119644 fixed-cache counterexample.

## Figure 3 — Proxy non-identifiability

**Placement:** Section 7.

Show identical observable \(P(Y,Z)\) feeding two latent worlds:

- World A: \(\mathcal D(Y,E)=0\);
- World B: \(\mathcal D(Y,E)=0.8\).

Then show the implication: proxy-only improvement cannot certify true CLE harm; paired DSA is used as the target-aligned behavioral evaluation.

## Table 1 — CLE mechanism and local-first attribution

**Placement:** Section 8.

Use the four-arm HFL/Local \(\times\) \(\gamma\) results plus the three contrasts and shuffled-binding reference.

## Table 2 — Cross-map / cross-base mitigation summary

**Placement:** Section 10.4–10.6.

Suggested columns:

| Setting | Base DSA | PEW+BER DSA | Relative reduction | Utility summary | Interpretation |
|---|---:|---:|---:|---|---|
| Asymmetric HFL, original map | 0.119644 | 0.041252 | 65.52% | Avg +1.7323, Worst -0.3760 | mitigation GO; uniform utility not established |
| Asymmetric HFL, map1 | 0.113761 | 0.049531 | 56.46% | Avg +0.7040, Worst +1.0760 | cross-map replication |
| FedDF-fidelity | 0.136989 | 0.029012 | 78.82% | grid -0.04, Avg -0.6660 | shortcut suppression transfers; overall utility gate fails |

## Table 3 — Held-out map2 matched control comparison

**Placement:** Section 10.7.

Use ERM / CVaR-DRO / PEW+GroupDRO / PEW+BER 40-round Formal table. Include an explicit table note:

> Pooled BER DSA is lower than pooled CVaR-DRO DSA, but CVaR-DRO attains lower DSA for clients c0 and c1; therefore no per-client uniform dominance is claimed.

## Table 4 — Oracle correspondence and granularity

**Placement:** Section 10.8 or appendix if page-limited.

Use Oracle family / Oracle operator / Random operator. The table note should state that Oracle labels are unavailable to the deployable method.

# C. Missing Citation List

The following citations or claim-level descriptions should be checked against the original paper before the submission bibliography is frozen:

1. **FedDF** — exact title, author list, NeurIPS 2020 bibliographic entry, and the precise scope of heterogeneous-model distillation. **[REF TO VERIFY]**
2. **GroupDRO** — exact theorem/method wording and ICLR 2020 bibliographic entry. **[REF TO VERIFY]**
3. **GEORGE / No Subclass Left Behind** — exact pseudo-group recovery description and NeurIPS 2020 entry. **[REF TO VERIFY]**
4. **EIIL** — exact objective for environment inference and ICML 2021 entry. **[REF TO VERIFY]**
5. **JTT** — exact two-stage upweighting protocol and ICML 2021 entry. **[REF TO VERIFY]**
6. **CVaR / joint-DRO family** — choose and verify the exact citation(s); do not present the project baseline as a line-by-line reproduction of one specific paper unless that is true. **[REF TO VERIFY]**
7. **Personalized Federated Learning with Spurious Features** — verify the TMLR 2024 formal version and distinguish it from the earlier workshop version. **[REF TO VERIFY]**
8. **FedPIN** — verify the ICML 2024 original paper’s assumptions, objective, and experiment setting before using stronger comparison language. **[REF TO VERIFY]**
9. **FedCD** — verify whether a formal venue appeared after the arXiv 2024 version; current project evidence only confirms the preprint status at the time of the literature audit. **[REF TO VERIFY]**
10. **MIDL 2026 counterfactual shortcut-utilization paper** — verify exact intervention variables and estimator language before drawing the final distinction. **[REF TO VERIFY]**
11. **Common corruption benchmark** — exact ICLR 2019 citation. **[REF TO VERIFY]**
12. **General shortcut-learning overview** — exact Nature Machine Intelligence 2020 citation if used in the final Introduction. **[REF TO VERIFY]**
13. **Medical shortcut case study (MLHC 2020)** — verify exact real-world shortcut examples before using them as motivation. **[REF TO VERIFY]**
14. **AugMix** — direct PDF is available in the project, but final bibliography metadata should still be checked when generating BibTeX.
15. **RAHFL / AugHFL / RHFL** — direct paper files are available and the publication relationship is already documented; final BibTeX/DOI formatting should still be checked mechanically.

# D. Remaining Experiment Placeholders

## D.1 Pure PEW+BER Multi-Training-Seed Stability

**[EVIDENCE NEEDED]**

Minimum insertion template after completion:

```text
Across N independent training seeds under the frozen [setting],
PEW+BER changed pooled DSA from [BASE mean ± dispersion] to
[BER mean ± dispersion]. Seed-level task metrics were [...].
These seed-level statistics quantify training randomness and are
reported separately from source-clustered bootstrap intervals.
```

Do not use old runs that include components outside the current PEW+BER method as evidence for this placeholder.

## D.2 Second Private Dataset

**[EVIDENCE NEEDED]**

Protocol not yet frozen. The final experiment should answer only the minimum external-validity question: does the CLE → DSA → local-first → mitigation chain persist on another private task/data family? Avoid expanding into a broad benchmark suite unless scientifically necessary.

## D.3 Bounded Taxonomy Stress Test

**[EVIDENCE NEEDED]**

Protocol not yet frozen. Suggested scientific question only:

> How does shortcut suppression degrade when the true corruption does not map cleanly to PEW’s training taxonomy?

Do not claim arbitrary open-world robustness from a single stress test.

## D.4 Optional Formal JTT Comparison

Not required unless the paper intends to claim empirical superiority over JTT. The current 12-round JTT result is a screen and should remain out of the final scientific win/loss table.

# E. Claims Requiring Weaker Wording

Use the right-hand wording in the final manuscript.

| Avoid | Use instead |
|---|---|
| “We are the first to study spurious correlation in federated learning.” | “We isolate a specific class–corruption directional shortcut in model-heterogeneous federated learning.” |
| “We propose a taxonomy-free environment discovery method.” | “PEW is a taxonomy-assisted public environment witness.” |
| “DSA causally identifies the model’s shortcut mechanism.” | “Under semantic-preservation, pre-registration, valid intervention, and evaluation-isolation assumptions, DSA identifies a binding-direction operator-response contrast.” |
| “Counterfactual evaluation proves the causal mechanism.” | “The source-paired operator intervention provides a controlled directional diagnostic under stated assumptions.” |
| “BER guarantees true environment–label independence.” | “BER exactly compresses class–proxy-environment support under its effective distribution; transfer to true environment dependence requires additional proxy-error assumptions.” |
| “PEW+BER is a universal lossless HFL plugin.” | “PEW+BER consistently suppresses DSA in the completed controlled settings, while task utility is base- and client-dependent.” |
| “PEW+BER improves robustness and utility across two HFL bases.” | “DSA suppression transfers across two HFL communication bases; utility improvement is not consistent across bases.” |
| “BER uniformly outperforms CVaR-DRO.” | “On held-out map2, BER achieves a better pooled shortcut–utility trade-off than CVaR-DRO; CVaR-DRO still has lower DSA on clients c0 and c1.” |
| “The result generalizes across scenarios.” | “The result replicates across two controlled binding maps; cross-partition, cross-seed, cross-dataset, and real-world generalization are not yet established.” |
| “The source-bootstrap CI shows the training result is stable.” | “The source-bootstrap CI quantifies evaluation-source uncertainty conditional on a fixed trained checkpoint.” |
| “Operator-level PEW is unnecessary.” | “The current Oracle granularity experiment does not provide sufficient evidence to justify further operator-level PEW development for this paper.” |
| “CIFAR-C experiments demonstrate hospital/vehicle/industrial deployment robustness.” | “The controlled construction abstracts a plausible class–acquisition correlation mechanism; real deployment effectiveness remains untested.” |
| “We outperform JTT.” | Do not make this claim unless a final matched JTT Formal is completed. |
| “FedDF confirms a second-base success.” | “FedDF-fidelity provides supporting evidence that shortcut suppression transfers to a second communication base, despite a failed overall utility gate.” |

# F. Suggested Compression Plan for a Conference Page Limit

If the long draft must be reduced to an AISTATS-/CCF-B-style main paper, preserve the **problem → diagnosis → mechanism → intervention → controlled validation → boundary** chain and move implementation/audit detail to the appendix.

## Keep in the main paper

1. **Introduction** — five compact paragraphs plus four contributions.
2. **Related Work** — four short subsections, approximately one paragraph each.
3. **Problem Setup** — CLE definition, data-role isolation, one compact protocol paragraph.
4. **DSA** — definition, one identification proposition, shuffled-binding rationale, source-level inference note.
5. **Proxy non-identifiability** — theorem statement plus one-sentence construction and implication; move full proof to appendix.
6. **Local-first** — one four-arm table and three contrasts.
7. **PEW+BER** — PEW in one paragraph; BER formula; effective-distribution/support-compression proposition.
8. **Experiments** — three core tables:
   - mechanism/local-first,
   - held-out map2 matched controls,
   - cross-map/cross-base summary.
9. **Limitations** — taxonomy-assisted scope, synthetic CLE, training-seed/second-dataset boundary, mixed utility.

## Move to appendix

- full DSA affine proof and injection-validation details;
- full proxy non-identifiability constructive proof;
- Hoeffding derivation;
- PEW architecture/training hyperparameters;
- BER CPU identity audit and all TV tables;
- full per-client DSA values;
- Oracle granularity table if space is tight;
- all frozen gate definitions;
- all screening runs, including JTT screen;
- implementation audit and data-role checks;
- any future multi-seed, second-dataset, or taxonomy-stress extended tables.

## Candidate main-paper narrative after compression

```text
Motivation: corruption can become a class code, not only a nuisance.
    ↓
Diagnosis: same-source operator intervention + DSA.
    ↓
Theory: paired cancellation + proxy non-identifiability.
    ↓
Mechanism: matched HFL/Local shows local-first formation.
    ↓
Intervention: taxonomy-assisted PEW + class-conditional BER.
    ↓
Evidence: held-out matched controls + cross-map + second HFL base.
    ↓
Boundary: utility is not uniform; taxonomy and external validity remain limited.
```

The first cuts should be implementation prose and historical screening detail, **not** the negative results or the identification assumptions. Those boundaries are part of the paper’s credibility and should survive page compression.
