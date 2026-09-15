# Version 0.4 — Conference-Compressed Internal Manuscript

> **Internal status.** This is a conference-compressed internal manuscript, not a final submission. It is grounded in the current CLE-HFL evidence package, V0.3 citation audit, and frozen method definition. Incomplete empirical claims remain marked **[EVIDENCE NEEDED]**. The final method uses a **coarse family-level, taxonomy-assisted Public Environment Witness (PEW)** with **Balanced Environment Risk (BER)**. CLE-v2 data generation and DSA evaluation remain **operator-level**. No operator-level PEW, hierarchical PEW, new loss, or new communication mechanism is proposed.

# When Corruption Becomes a Label: Diagnosing and Mitigating Class–Corruption Shortcuts in Model-Heterogeneous Federated Learning

## Abstract

Model-heterogeneous federated learning (HFL) enables clients with different architectures to collaborate, but existing corruption-robust HFL methods mainly treat corruption as a nuisance that degrades input quality. We study a controlled failure mode, **class–corruption entanglement (CLE)**, in which client-specific associations between task classes and corruption operators can turn corruption into a directional class cue. To distinguish this behavior from ordinary corruption difficulty, we construct source-paired operator interventions and introduce **Directional Shortcut Alignment (DSA)**, which measures whether predictive probability mass moves toward classes associated with the applied operator during training. Under explicit semantic-preservation, pre-registered-binding, and evaluation-isolation assumptions, DSA identifies a binding-direction operator-response contrast. A complementary proxy non-identifiability result shows that improvements in pseudo-environment statistics or generic proxy objectives cannot, without additional assumptions, certify reduced true CLE dependence. A matched HFL-versus-Local factorial indicates that the observed shortcut is predominantly local-first in the controlled setting, motivating a local intervention. We therefore combine a taxonomy-assisted coarse **Public Environment Witness (PEW)** with **Balanced Environment Risk (BER)**, which compresses class-conditional pseudo-environment support imbalance without changing the communication protocol. Completed seed-0 controlled comparisons against ERM, a protocol-matched CVaR-DRO control, and PEW+GroupDRO show substantial pooled DSA reduction, with replication across two controlled binding maps and a similar suppression effect under a second protocol-matched HFL communication base. Utility improvements are not uniform across bases or clients, and PEW remains taxonomy-assisted.

## 1. Introduction

Federated learning allows multiple clients to collaboratively learn predictive models while keeping private training data local. In realistic systems, clients may differ not only in data distribution but also in model architecture, motivating **model-heterogeneous federated learning (HFL)**. Knowledge-distillation approaches such as FedDF communicate predictions rather than assuming parameter compatibility \citep{lin2020feddf}. Robust HFL further considers noisy or corrupted clients: RHFL studies noisy labels under model heterogeneity \citep{fang2022rhfl}, while AugHFL and its TPAMI extension RAHFL study model-heterogeneous collaboration under common data corruption \citep{fang2023aughfl,fang2025rahfl}. These works establish corruption as an important HFL robustness problem, but common corruption benchmarks and robust-learning pipelines typically evaluate corruption as a label-preserving nuisance rather than as a class-dependent training cue \citep{hendrycks2019corruptions,cubuk2020augmix}.

We study a more specific failure mode. Suppose one client observes most images of one class with blur and most images of another class with noise, while another client has a different class–corruption association. Corruption can then become predictive of the task label instead of acting only as nuisance variation. We call this controlled setting **class–corruption entanglement in HFL (CLE-HFL)**. The scientific question is not merely whether corruption lowers accuracy, but whether changing only the corruption operator moves predictions *toward the classes associated with that operator during training*. Corrupted accuracy alone cannot distinguish this directional shortcut from ordinary image degradation, label imbalance, model capacity limits, or non-IID optimization.

This motivates a target-aligned diagnostic. Within-view consistency, including AugMix-style Jensen–Shannon consistency, can make multiple augmentations around the same corrupted observation agree while leaving cross-operator class routing intact \citep{cubuk2020augmix}. Likewise, a pseudo-environment statistic can improve without identifying whether the model has reduced reliance on the true class–corruption association. We therefore evaluate the same semantic source under multiple corruption operators and define **Directional Shortcut Alignment (DSA)** as a probability-mass contrast toward the classes pre-registered as bound to the applied operator. DSA is intentionally narrower than a general causal representation claim: under semantic preservation, a valid operator intervention, pre-registered binding, source-level pairing, and strict evaluation isolation, it identifies a binding-direction operator-response contrast. We further show that proxy-only improvements cannot generally certify reduced true CLE dependence when the proxy error channel is unconstrained, making a target-aligned behavioral endpoint necessary under our evidence protocol.

Using DSA, we next ask where the shortcut forms. In a matched HFL-versus-Local factorial, the pooled strong-CLE effect is close between HFL and Local training, while the additional communication contrast is much smaller. We therefore describe the observed mechanism as **predominantly local-first** in this controlled setting and use that result to motivate intervention at the client objective rather than redesigning communication. The intervention combines a **coarse family-level Public Environment Witness (PEW)**, trained only on public synthetic corruptions, with **Balanced Environment Risk (BER)**, which reduces the effective risk-mass advantage of large pseudo-environment supports within each task class. PEW is explicitly **taxonomy-assisted**, not taxonomy-free, and the training method never reads private true corruption operators or families.

We evaluate the resulting evidence chain rather than presenting PEW or BER as isolated architectural novelties. A held-out 40-round comparison uses matched ERM, a protocol-matched CVaR-DRO tail-risk control, PEW+GroupDRO with the same PEW grouping information, and PEW+BER. Additional controlled experiments change only the class–operator binding map or the HFL communication base. The completed evidence supports strong shortcut suppression, but not a universal no-harm claim: the FedDF-fidelity utility gate fails, and CVaR-DRO retains lower DSA for some individual clients on the held-out map. Our contributions are therefore fourfold:

1. **CLE-HFL problem formulation.** We formulate class–corruption entanglement in model-heterogeneous federated learning, where client-specific class–corruption associations can turn corruption operators into directional class cues.
2. **Paired diagnosis and identification boundary.** We introduce source-paired operator counterfactual evaluation and **Directional Shortcut Alignment (DSA)**, characterize its identification assumptions and source-level inference, and establish a proxy non-identifiability result showing why generic proxy improvements cannot certify reduced CLE harm without additional assumptions.
3. **Local-first mechanism attribution.** Using a matched HFL-versus-Local factorial, we find that the observed directional shortcut is predominantly local-first in the controlled setting, with communication contributing a smaller pooled additional effect.
4. **Taxonomy-assisted local mitigation and controlled validation.** Guided by this finding, we develop a coarse family-level taxonomy-assisted local intervention combining PEW and BER, and evaluate it against matched ERM, tail-risk, and group-robustness controls across controlled binding maps and HFL communication bases while explicitly reporting utility and taxonomy boundaries.

We do not claim to be the first to study spurious correlation in federated learning, environment inference, group reweighting, or counterfactual shortcut evaluation \citep{wang2024pflspurious,tang2024fedpin,ma2024fedcd,creager2021eiil,sagawa2020groupdro,vigneshwaran2026counterfactual}. Our focus is the joint instantiation of a client-specific class–corruption directional failure mode, a binding-aligned paired diagnostic, local-versus-communication attribution, and a structure-matched local intervention in model-heterogeneous FL.

## 2. Related Work

### 2.1 Model-Heterogeneous Federated Learning

Model heterogeneity breaks the direct parameter-averaging assumption used by standard homogeneous FL. FedDF addresses this mismatch through ensemble distillation over unlabeled data \citep{lin2020feddf}. RHFL considers model-heterogeneous clients with noisy labels and confidence-weighted collaboration \citep{fang2022rhfl}. AugHFL explicitly studies common corruption under model heterogeneity, while RAHFL extends this line with diversity-enhanced local representation learning and asymmetric heterogeneous collaboration \citep{fang2023aughfl,fang2025rahfl}. These methods are the closest technical background for our HFL training pipeline. Our question is different: whether client-specific class–corruption associations make corruption operators directional class cues, and where that shortcut forms.

### 2.2 Corruption Robustness and Consistency Learning

CIFAR-C and ImageNet-C established standard common-corruption evaluations across noise, blur, weather, and digital distortions \citep{hendrycks2019corruptions}. AugMix combines diverse stochastic augmentations with Jensen–Shannon consistency to improve corruption robustness and uncertainty \citep{cubuk2020augmix}. CLE-HFL differs in statistical structure: corruption is deliberately associated with class during client training. Accordingly, our diagnosis compares *different operators for the same semantic source* rather than only checking consistency around one corrupted view.

### 2.3 Spurious Correlation, Group Robustness, and Environment Inference

Shortcut learning and group distribution shift have a broad literature \citep{geirhos2020shortcut}. GroupDRO optimizes worst-group risk with known groups \citep{sagawa2020groupdro}; GEORGE recovers hidden subclasses before group-robust learning \citep{sohoni2020george}; EIIL infers environments without a predefined taxonomy \citep{creager2021eiil}; and JTT upweights examples misclassified by an initial ERM model \citep{liu2021jtt}. CVaR and related DRO objectives emphasize upper-tail loss rather than an explicit class–environment support model \citep{rockafellar2000cvar,duchi2021uniform}. These works constrain our novelty claim: pseudo-environment discovery, difficult-example reweighting, and group robustness are not new. PEW is instead a taxonomy-assisted corruption-family witness, while BER changes the total effective risk mass assigned to class–pseudo-environment supports. Our CVaR-DRO implementation is a protocol-matched empirical tail-risk control, not a line-by-line reproduction of either cited method.

### 2.4 Federated Spurious Learning and Counterfactual Diagnosis

Federated spurious features have already been studied. Personalized federated learning can exploit client-specific spurious features \citep{wang2024pflspurious}; FedPIN targets shortcut-averse personalized invariant learning \citep{tang2024fedpin}; and FedCD studies spurious correlation in federated domain generalization \citep{ma2024fedcd}. In the current citation audit, FedCD is treated only as an arXiv preprint. Counterfactual shortcut evaluation also has precedents, including medical-imaging analyses that test whether acquisition or demographic attributes are actually used by a model \citep{vigneshwaran2026counterfactual,jabbour2020chestxray}. Our contribution is not counterfactual analysis in the abstract. We specialize it to source-paired corruption-operator interventions, a pre-registered client-specific binding direction, probability-mass alignment, and a matched local-versus-federated formation analysis.

## 3. Problem Setup

### 3.1 Model-Heterogeneous CLE

We consider clients \(k\in\{1,\ldots,K\}\), each with a private dataset and classifier \(f_k\); architectures need not match across clients. The primary controlled setup uses four heterogeneous clients with ResNet10, ResNet12, ShuffleNet, and MobileNetV2 on a CIFAR-10 private task. Client label distributions are non-IID under a Dirichlet partition with concentration \(\alpha=0.5\).

Let \(Y\) denote the task class, \(O\) a concrete corruption operator, and \(F=\pi(O)\) its coarse corruption family. CLE-HFL introduces client-specific class dependence,
\[
P_k(O=o\mid Y=c)\neq P_k(O=o),
\]
with a pre-registered client-specific binding
\[
b_k:\mathcal C\rightarrow\mathcal O.
\]
The control setting uses \(\gamma_{\mathrm{CLE}}=0\), while strong CLE uses \(\gamma_{\mathrm{CLE}}=0.9\), making \(b_k(c)\) a strong but non-deterministic dominant operator for class \(c\). The formal paired evaluation uses 1,000 semantic sources, 15 operators, four clients, and corruption severity 3.

A crucial distinction is granularity. **CLE-v2 data generation and DSA evaluation are operator-level. The final PEW+BER method is not.** PEW predicts only a coarse corruption-family proxy and BER groups private fit samples by class and that coarse proxy. Operator metadata remain unavailable to the deployable training method. Operator-level labels are used only for CLE construction, sealed DSA evaluation, and non-deployable Oracle boundary audits.

### 3.2 Information Separation

Private data are separated by role. The **private fit** subset produces local gradients and BER group counts. A **private audit** subset may support strict communication routing but does not produce local-training gradients. The **final task test** is used only for frozen utility reporting. The **paired operator grid** is used only after training for DSA. Public data support PEW training or pre-existing communication without private-task ground-truth leakage.

The train-time method does not read private true operator IDs, true corruption families, evaluation severity, seen/unseen flags, or the paired evaluation binding for optimization, routing, hyperparameter tuning, or model selection. DSA reads the frozen binding only after model sealing. This isolation is required because the diagnostic is deliberately target-aligned.

## 4. Paired Diagnosis and Identification

### 4.1 Directional Shortcut Alignment

For source \(z=(x,y)\), let \(T_o(x)\) be a semantic-preserving corruption transformation. For client \(k\) and operator \(o\), define the classes bound to that operator during training:
\[
B_{k,o}=\{c:b_k(c)=o\}.
\]
We evaluate only source–operator pairs for which \(y\notin B_{k,o}\), avoiding a trivial contribution from the source's true class. Define bound probability mass
\[
m_{k,o}(z,o')=\sum_{c\in B_{k,o}}p_k(c\mid T_{o'}(x)).
\]
The source-level directional contrast is
\[
d_{k,o}(z)=m_{k,o}(z,o)
-\frac{1}{|\mathcal O|-1}\sum_{o'\neq o}m_{k,o}(z,o'),
\]
and
\[
\operatorname{DSA}_{k,o}=\mathbb E_z[d_{k,o}(z)].
\]
Pooled DSA equally averages valid operator-level values and then client-level values. DSA lies on a probability scale: a value of \(0.12\) means roughly a 12-percentage-point directional difference in predictive mass, not a \(0.12\)-point accuracy difference.

The interpretation is narrow: *for the same semantic source, does applying operator \(o\) move probability toward the classes that were associated with \(o\) during this client's training, relative to applying other operators?* The source-paired intervention is a controlled diagnostic under stated assumptions, not an unconditional recovery of the model's full causal mechanism.

### 4.2 Paired Cancellation and Binding Specificity

Suppose
\[
m_{k,o}(z,o')=s_{k,o}(z)+r_{k,o}(z,o'),
\]
where \(s_{k,o}(z)\) is an operator-invariant source term and \(r_{k,o}\) is an operator-dependent response. Then
\[
d_{k,o}(z)=r_{k,o}(z,o)
-\frac{1}{|\mathcal O|-1}\sum_{o'\neq o}r_{k,o}(z,o'),
\]
so the operator-invariant source term cancels exactly. Under this decomposition, DSA identifies a **binding-direction operator-response contrast**. It does not prove that the model uses only corruption, that every operator preserves all task semantics, or that the same response is harmful in every deployment distribution.

To test binding specificity, we preserve the number of classes assigned to each operator but randomly permute the binding and recompute DSA on the same prediction cache. Under the primary strong-CLE HFL checkpoint, true-binding DSA is \(0.119644\), whereas the shuffled-binding null has p95 \(=0.029559\) with permutation \(p=0.000999\). This provides evidence against the explanation that the effect is driven only by generic operator-to-class response patterns unrelated to the training association.

### 4.3 Why Consistency and Proxy Scores Are Insufficient

Within-operator consistency and cross-operator directional invariance are different objects. If multiple predictive views around the same corrupted observation are identical, their JSD is zero; predictions can nevertheless differ systematically across corruption operators. In a fixed-cache construction, within-operator JSD is zero while DSA remains \(0.119644\). Thus,
\[
\text{within-operator consistency}\not\Rightarrow
\text{cross-operator directional invariance}.
\]
This does not make JSD ineffective for corruption robustness; it shows that consistency and DSA answer different questions.

The same separation motivates a proxy non-identifiability result. Let \(Z\) be a proxy environment predicted by PEW and \(E\) the latent true corruption environment. Define
\[
\mathcal D(A,B)=TV(P_{A,B},P_AP_B).
\]
With no constraint on the error channel \(P(Z\mid E,Y)\), a fixed observable \(P(Y,Z)\) can correspond to one latent world with \(\mathcal D(Y,E)=0\) and another with
\[
\mathcal D(Y,E)=1-\sum_eP(E=e)^2>0.
\]
Therefore \(P(Y,Z)\) alone does not identify true environment dependence. A conditional bridge remains valid under sample-wise coupling:
\[
\mathcal D_Q(Y,E)\le
\mathcal D_Q(Y,Z)+2\epsilon_Q,
\qquad
\epsilon_Q=P_Q(Z\neq E).
\]
In the current BER effective distribution, the PEW family error is \(0.507\)--\(0.575\), making this bound trivial after truncation at 1.0. We therefore do not claim an unconditional theoretical guarantee that BER makes the true environment independent of class. Under our evidence protocol, mitigation claims require an additional target-aligned behavioral endpoint; we use paired DSA. Full constructive proofs and algebraic validations are deferred to Appendix A.

### 4.4 Statistical Unit

The independent evaluation unit is the **semantic source**, not each corrupted image. Multiple operator views of one source share content and must not be treated as independent observations. Main DSA intervals use source-clustered bootstrap after computing source-level contrasts. These intervals condition on the trained checkpoint and quantify evaluation-source uncertainty only; they do **not** quantify training-seed, partition, binding-map, model-selection, or cross-dataset uncertainty.

## 5. Local-First Mechanism Attribution

We compare four matched arms under CLE-v2: HFL with \(\gamma_{\mathrm{CLE}}=0\), HFL with \(\gamma_{\mathrm{CLE}}=0.9\), Local with \(\gamma_{\mathrm{CLE}}=0\), and Local with \(\gamma_{\mathrm{CLE}}=0.9\). All arms share the same AugMix/JSD/DCL local baseline, use training seed 0, run for 12 rounds, and use at most 16 local batches per client per round.

**Table 1. CLE formation and matched local-first attribution. Source-bootstrap intervals condition on the trained checkpoints.**

| Arm | Scope | \(\gamma_{\mathrm{CLE}}\) | Operator-grid Acc. (%) | Pooled DSA |
|---|---|---:|---:|---:|
| HFL control | HFL | 0.0 | 24.9300 | -0.000245 |
| HFL strong CLE | HFL | 0.9 | 21.4367 | 0.119644 |
| Local control | Local | 0.0 | 25.5683 | -0.001914 |
| Local strong CLE | Local | 0.9 | 22.4233 | 0.106046 |

The matched contrasts are
\[
\Delta_{\mathrm{HFL}}=0.119889,\qquad
\Delta_{\mathrm{Local}}=0.107960,
\]
and
\[
\Delta_{\mathrm{comm}}
=\Delta_{\mathrm{HFL}}-\Delta_{\mathrm{Local}}
=0.011929.
\]
Their source-bootstrap 95% intervals are \([0.118050,0.121562]\), \([0.106145,0.109678]\), and \([0.011115,0.012712]\), respectively. At this evaluated seed and fixed scenario, the descriptive ratio
\[
\Delta_{\mathrm{Local}}/\Delta_{\mathrm{HFL}}=90.05\%
\]
shows that the Local contrast is close to the HFL contrast. We call this **predominantly local-first formation**. The ratio is not a per-example causal mediation share, and the result does not imply that communication is irrelevant. It motivates targeting the local objective because the matched Local effect already accounts for most of the pooled HFL effect in this setting.

**[EVIDENCE NEEDED: multi-training-seed stability for the matched HFL-versus-Local four-arm factorial. Candidate S1/S2 repetitions would jointly cover HFL \(\gamma=0\) vs. \(.9\), Local \(\gamma=0\) vs. \(.9\), and local-first stability; these candidate runs are not authorized or reported here.]**

## 6. Taxonomy-Assisted Local Mitigation

### 6.1 Public Environment Witness

The **Public Environment Witness (PEW)** is trained only on public carrier images with programmatically generated corruption supervision. It is frozen as a **coarse family-level witness** over
\[
\{\text{clean},\text{noise},\text{blur},\text{weather},\text{digital},\text{unknown}\}.
\]
For each private fit sample, PEW produces a hard pseudo-environment \(\hat e_i\). PEW does not read private true operators or private true corruption families during training. It is therefore private-environment-metadata-free at deployment but explicitly **taxonomy-assisted**: its semantics come from a predefined corruption-family taxonomy. Architecture and optimizer details are deferred to Appendix B because PEW is treated as side-information infrastructure rather than the paper's architectural contribution.

The granularity mismatch is deliberate. CLE-v2 construction and DSA evaluation remain operator-level, while PEW and BER use only coarse family-level pseudo-environments. The Oracle operator analysis in Appendix C is a non-deployable boundary audit and does not define or motivate an operator-level PEW.

### 6.2 Balanced Environment Risk

For client \(k\), task class \(c\), and pseudo-environment \(e\), define
\[
n_{k,c,e}=\#\{i:y_i=c,\hat e_i=e\},
\qquad
R_{k,c,e}=\frac1{n_{k,c,e}}\sum_{i:y_i=c,\hat e_i=e}\ell_i.
\]
BER assigns class-conditional environment weight
\[
a^{(\gamma_{\mathrm{BER}})}_{k,c,e}
=
\frac{
\mathbf 1[n_{k,c,e}\ge m]
\min(n_{k,c,e},K_{\mathrm{cap}})^{\gamma_{\mathrm{BER}}}
}{
\sum_{e'}
\mathbf 1[n_{k,c,e'}\ge m]
\min(n_{k,c,e'},K_{\mathrm{cap}})^{\gamma_{\mathrm{BER}}}
},
\]
with the frozen Formal configuration
\[
\gamma_{\mathrm{BER}}=0.5,\qquad K_{\mathrm{cap}}=32,\qquad m=2.
\]
The client task risk is
\[
R_{\mathrm{BER},k}
=
\frac1{|\mathcal C_k^{\mathrm{valid}}|}
\sum_{c\in\mathcal C_k^{\mathrm{valid}}}
\sum_e a^{(\gamma_{\mathrm{BER}})}_{k,c,e}R_{k,c,e}.
\]

BER does not directly optimize the worst environment. It reduces the effective risk-mass advantage of large pseudo-environment supports *within each class*. Equivalently, it performs empirical risk minimization under effective sample mass
\[
q_i=
\frac{a^{(\gamma_{\mathrm{BER}})}_{k,y_i,\hat e_i}}
{|\mathcal C_k^{\mathrm{valid}}|n_{k,y_i,\hat e_i}}.
\]
For two valid environments within a class,
\[
\frac{Q_\gamma(e_1\mid c)}{Q_\gamma(e_2\mid c)}
=
\left(
\frac{\min(n_{c,e_1},K_{\mathrm{cap}})}
{\min(n_{c,e_2},K_{\mathrm{cap}})}
\right)^{\gamma_{\mathrm{BER}}}.
\]
Hence \(0\le\gamma_{\mathrm{BER}}<1\) compresses the class-conditional log-support advantage. At the current configuration, the maximum effective mass ratio between valid pseudo-environment groups is \(4\). A fixed-data audit verifies that the implemented objective matches this effective-distribution formula; detailed audit statistics are in Appendix B. These statements characterize the *training distribution induced by BER* and do not imply that DSA must decrease or that task accuracy must improve.

## 7. Experiments

### 7.1 Protocol and Metrics

We organize the main evidence around four questions: whether CLE induces binding-aligned behavior; whether formation is predominantly local-first; whether PEW+BER suppresses DSA beyond generic ERM/group/tail-risk alternatives; and whether suppression persists under controlled changes of the binding map or communication base.

DSA is the primary shortcut metric. We also report operator-grid accuracy and frozen task utility metrics: client-average accuracy (Avg), minimum client accuracy (Worst), worst valid class–operator-cell accuracy (WCCA), and mean within-class accuracy range across operators (CFG, lower is better). Absolute values across different protocols are not compared directly; attribution relies on matched within-setting contrasts. This distinction is especially important because the original-map intervention experiment, the held-out map2 control experiment, and the FedDF-fidelity experiment intentionally use different controlled recipes. We therefore use each experiment to answer a specific scientific question rather than ranking all checkpoints in a single global leaderboard. The held-out map2 table is the primary comparison for alternative risk objectives; map1 tests dependence on one particular binding assignment; and FedDF-fidelity tests whether the shortcut-suppression effect is tied to the primary HFL communication stack. This organization prevents protocol changes from being misinterpreted as treatment effects.

> **Figure 1 placeholder — CLE-HFL evidence chain.**
> Client-specific class–operator binding \(\rightarrow\) source-paired operator intervention \(\rightarrow\) DSA \(\rightarrow\) HFL-vs-Local attribution \(\rightarrow\) coarse taxonomy-assisted PEW \(\rightarrow\) BER class-conditional support balancing \(\rightarrow\) matched controls and controlled replication. The figure must visually separate sealed DSA evaluation from train-time PEW+BER.

### 7.2 Held-Out Matched Controls

A five-arm 12-round screen was used only for design and method selection; it is not treated as final scientific evidence. In particular, JTT remains a screening-only comparison and we do not claim a Formal empirical victory over JTT.

The final held-out map2 comparison uses a frozen 40-round protocol with ERM, protocol-matched CVaR-DRO, PEW+GroupDRO, and PEW+BER. The four arms share the same partition, training/evaluation seed, initial states, private batch trajectory, and strict HFL communication protocol. PEW+GroupDRO and PEW+BER receive the same PEW grouping information.

**Table 2. Held-out map2 matched control comparison.**

| Arm | DSA ↓ | Operator-grid Acc. ↑ | Last-10 Avg ↑ | Last-10 Worst ↑ | WCCA ↑ | CFG ↓ |
|---|---:|---:|---:|---:|---:|---:|
| ERM | 0.260308 | 20.3067 | 19.8578 | 16.3007 | 0.000 | 32.8325 |
| CVaR-DRO | 0.088982 | 20.4567 | 18.6700 | 14.0380 | 0.125 | 27.7775 |
| PEW+GroupDRO | 0.214575 | 20.7883 | 20.9562 | 17.8140 | 0.025 | 31.5275 |
| **PEW+BER** | **0.077741** | **23.7433** | **24.4518** | **20.4727** | **2.650** | **22.8025** |

Source-paired bootstrap differences are
\[
\mathrm{DSA(ERM)}-\mathrm{DSA(BER)}=0.182567
\]
with 95% interval \([0.180596,0.184407]\),
\[
\mathrm{DSA(GroupDRO)}-\mathrm{DSA(BER)}=0.136834
\]
with \([0.134952,0.138706]\), and
\[
\mathrm{DSA(BER)}-\mathrm{DSA(CVaR)}=-0.011240
\]
with \([-0.012355,-0.010092]\).

The matched PEW+GroupDRO result shows that shared PEW grouping alone is insufficient to reproduce BER's lower pooled DSA under this protocol. Relative to CVaR-DRO, BER has lower pooled DSA and substantially higher pooled task utility on the held-out map2 Formal. The conclusion is not uniform at client level: CVaR-DRO still has lower DSA on clients c0 and c1, while BER has lower DSA on c2 and c3; BER's operator-grid accuracy is higher on all four clients. We therefore state only that **PEW+BER achieves a better pooled shortcut–utility trade-off than CVaR-DRO on held-out map2, without uniformly dominating CVaR-DRO in per-client DSA.**

**[EVIDENCE NEEDED: multi-training-seed stability for the held-out map2 four-arm comparison. Candidate S1/S2 repetitions are part of the frozen candidate experiment matrix only; they are not authorized or reported as running.]**

### 7.3 Controlled Replication and Communication-Base Boundary

We next summarize matched PEW+BER comparisons under three completed settings.

**Table 3. Cross-setting mitigation summary. Utility values are matched intervention-minus-baseline deltas within each setting.**

| Setting | Base DSA | PEW+BER DSA | Relative DSA reduction | Utility summary | Supported interpretation |
|---|---:|---:|---:|---|---|
| Asymmetric HFL, original map | 0.119644 | 0.041252 | 65.52% | Avg +1.7323; Worst -0.3760 | DSA mitigation; uniform utility not established |
| Asymmetric HFL, map1 | 0.113761 | 0.049531 | 56.46% | Avg +0.7040; Worst +1.0760 | replication on a second controlled binding map |
| FedDF-fidelity | 0.136989 | 0.029012 | 78.82% | grid -0.0400; Avg -0.6660 | suppression also observed under a second communication base; utility gate fails |

On map1, only the client-specific class–operator binding map is changed while the partition, evaluation seed, training seed, initial weights, public data, evaluation grid, and frozen PEW remain fixed. All four clients reduce DSA, and pooled operator-grid accuracy changes from \(20.9450\%\) to \(21.4583\%\). This supports replication across two controlled binding directions at seed 0; it does not establish cross-partition, cross-seed, cross-dataset, or real-world generalization.

The second communication-base experiment uses native single-view CE with a protocol-matched **FedDF-fidelity** adapter. Base and BER arms share initialization, private batches, public distillation, and training budget. DSA changes from \(0.136989\) to \(0.029012\), but utility is mixed: operator-grid accuracy changes by \(-0.0400\) points, last-five Avg by \(-0.6660\), and the frozen overall utility gate fails; both arms also fall below the pre-registered 20% operator-grid learning floor. The safe interpretation is therefore: **the DSA-suppression effect is also observed under a second protocol-matched HFL communication base, although its utility gate fails.** This is not evidence for a lossless universal plugin or consistent cross-base utility improvement.

> **Figure 2 placeholder — DSA and proxy-identifiability schematic.**
> Left: same source under several operators, with probability mass evaluated toward pre-registered operator-bound classes and a shuffled-binding null. Right: identical observable \(P(Y,Z)\) can correspond to latent worlds with different \(\mathcal D(Y,E)\), motivating target-aligned DSA rather than proxy-only certification.

### 7.4 Environment Correspondence and Granularity

A non-deployable Oracle audit compares true family grouping, true operator grouping, and a random operator correspondence. Correct environment correspondence materially affects DSA in this controlled audit: Random operator DSA exceeds Oracle operator DSA by \(0.052112\), with source-bootstrap 95% interval \([0.051231,0.052983]\). The Oracle family–operator DSA gap is only \(0.001449\), with \([0.001086,0.001790]\). This does **not** establish that operator-level environment models are universally unhelpful. It only means that the present Oracle granularity experiment does not provide sufficient evidence to justify developing an operator-level PEW for this paper. The complete Oracle table is moved to Appendix C.

### 7.5 Remaining Evidence Placeholders

The following are internal evidence placeholders, not completed or authorized runs:

**[EVIDENCE NEEDED: second private dataset; dataset and protocol pending.]**

**[EVIDENCE NEEDED: bounded taxonomy stress test; protocol pending.]**

The frozen candidate matrix also contains the multi-seed repetitions noted in Sections 5 and 7.2. None of these candidate experiments should be described as authorized, launched, in progress, or successful until actual runs and frozen results exist. The current paper does not add a new loss, communication mechanism, hierarchical PEW, or operator-level PEW in response to these gaps.

## 8. Discussion and Limitations

**Shortcut suppression and utility are distinct.** A shortcut can be predictive on the same biased distribution that created it, so reducing DSA need not increase in-distribution task accuracy. The original Asymmetric-HFL result shows a large DSA reduction with mixed client utility, and the FedDF-fidelity result shows similar shortcut suppression while the overall utility gate fails. Even the favorable map2 result is a pooled trade-off rather than per-client uniform dominance over CVaR-DRO. We therefore report DSA and utility separately.

**The method is taxonomy-assisted and proxy-limited.** PEW requires a predefined coarse corruption-family taxonomy and public programmatic supervision. This differs from taxonomy-free environment inference such as EIIL \citep{creager2021eiil}. BER has an exact effective-distribution interpretation for the proxy groups, but the current proxy-error transfer bound is trivial; the non-identifiability theorem explains why stronger true-environment claims cannot be obtained from proxy statistics alone without additional assumptions. The final method remains coarse family-level. Operator-level information is reserved for CLE-v2 construction, DSA evaluation, and Oracle boundary analysis.

**External validity and statistical scope remain limited.** The core evidence uses controlled CIFAR-10 CLE with a fixed private partition, and key Formal runs use training seed 0. Source bootstrap does not establish training-seed stability. Cross-map replication shows that the result is not tied to one class–operator assignment, but does not imply cross-partition, cross-dataset, or deployment robustness. **[EVIDENCE NEEDED: second private dataset; dataset and protocol pending.]** A bounded taxonomy stress test is also **[EVIDENCE NEEDED]** and would characterize incomplete PEW coverage rather than prove arbitrary open-world robustness. Because each client is paired with both a specific architecture and a specific non-IID data slice, client differences also cannot be causally attributed to architecture capacity alone.

## 9. Conclusion

We study class–corruption entanglement in model-heterogeneous federated learning, where client-specific class–operator associations can turn corruption into directional class cues. Source-paired operator interventions and Directional Shortcut Alignment provide a target-aligned diagnostic under explicit identification assumptions, while proxy non-identifiability explains why pseudo-environment improvements alone cannot certify reduced CLE harm. A matched HFL-versus-Local analysis indicates predominantly local-first formation in the controlled setting, motivating a taxonomy-assisted local intervention that combines coarse family-level PEW with BER. Completed experiments show large DSA reductions under matched controls, on two controlled binding maps, and under a second protocol-matched HFL communication base, while also exposing mixed utility and taxonomy boundaries. The current claims remain limited by training-seed, single-private-dataset, and controlled-corruption scope.

# Appendix A. DSA Properties and Proxy Non-identifiability

## A.1 Exchangeable Zero and Affine Property

If predictions for the same source are exchangeable across all operators,
\[
p(\cdot\mid T_o(x))=p(\cdot\mid T_{o'}(x)),\quad\forall o,o',
\]
then every source-level DSA contrast is zero. The fixed-cache exchangeable projection yields \(-2.50\times10^{-20}\); the largest empirical absolute DSA under the \(\gamma_{\mathrm{CLE}}=0\) controls is \(0.001914\).

DSA is affine in predictive probabilities. For
\[
p_\lambda=(1-\lambda)p_0+\lambda p_1,
\]
\[
\operatorname{DSA}(p_\lambda)
=(1-\lambda)\operatorname{DSA}(p_0)
+\lambda\operatorname{DSA}(p_1).
\]
An 11-point fixed-cache validation has maximum numerical error \(2.78\times10^{-17}\). In a controlled probability-simplex injection, DSA recovers a known binding-aligned response with maximum numerical error \(6.66\times10^{-16}\). Adding an identical displacement to every operator view of the same source changes DSA by only \(8.33\times10^{-17}\), numerically validating paired cancellation.

## A.2 Constructive Proxy Non-identifiability

Fix an observable \(P(Y,Z)\), with at least two positive-probability values of \(Y\), and leave \(P(Z\mid E,Y)\) unconstrained. One latent world can choose \(E\) independent of \((Y,Z)\), yielding \(\mathcal D(Y,E)=0\). Another can set \(E=g(Y)\) for a nonconstant mapping while retaining the same \(P(Z\mid Y)\), yielding
\[
\mathcal D(Y,E)=1-\sum_eP(E=e)^2.
\]
The two worlds have identical observable \(P(Y,Z)\) but different true dependence. In the balanced ten-class, five-environment cache construction, these values are 0 and 0.8. The construction proves non-identifiability; it is not intended to model a natural corruption-generating process.

## A.3 Source-Level Inference

The main bootstrap first forms source-level directional contrasts and then resamples sources. A conservative bounded-variable calculation for \(n=1000\), \(\delta=0.05\) gives Hoeffding radius \(0.085894\), but the paper uses source-clustered bootstrap intervals. Neither calculation covers training randomness.

# Appendix B. PEW and BER Implementation/Audit Details

PEW uses public CIFAR-100 carrier images with programmatically generated coarse labels over clean, noise, blur, weather, digital, and unknown environments. Unknown is generated by sequentially composing two distinct base families. Severity ranges from 1 to 5. The frozen witness is a small CNN predicting six environments, five severity levels, and a 32-dimensional embedding, trained with environment cross-entropy plus \(0.25\) times severity cross-entropy for five epochs using Adam at learning rate \(10^{-3}\). Checkpoint and unknown-threshold selection use public validation only.

A fixed-data CPU audit verifies BER's effective empirical distribution with maximum implementation/formula mismatch \(7.32\times10^{-16}\). On the strict-fit audit distribution, equal-client dependence changes from \(TV(Y,\hat E)=0.487505\) to \(0.199744\) (59.03% relative drop). Reporting-only true-family dependence changes from \(0.634914\) to \(0.433513\) (31.72% relative drop), and true-family environment-only Bayes advantage changes from \(0.313840\) to \(0.240863\) (23.25% relative drop). These quantities audit the effective training distribution; true environment metadata are not used by the deployable training rule, and the audit does not imply that trained-model DSA or accuracy must improve.

# Appendix C. Additional Experimental Evidence

## C.1 Original-Map PEW+BER

On the original strong-CLE map, pooled DSA changes from \(0.119644\) to \(0.041252\), an absolute reduction of \(0.078392\) and relative reduction of 65.52%; the source-bootstrap 95% interval for the DSA difference is \([0.077191,0.079601]\). Per-client DSA reductions are \([0.111312,0.011317,0.115762,0.075178]\). Last-five intervention-minus-baseline deltas are Avg \(+1.7323\), Worst \(-0.3760\), WCCA \(+0.3500\), and CFG \(-9.2600\) percentage points. The shortcut-mitigation criterion passes, but the frozen full-utility criterion fails because Worst does not meet its pre-registered threshold.

## C.2 Oracle Correspondence and Granularity

| Grouping | Pooled DSA | Operator-grid Avg | Last-5 Avg | Last-5 Worst |
|---|---:|---:|---:|---:|
| Oracle family | 0.017232 | 21.9333 | 21.0960 | 17.8400 |
| Oracle operator | 0.015783 | 21.5200 | 20.0273 | 15.8827 |
| Random operator | 0.067895 | 18.7817 | 18.3933 | 14.7867 |

Oracle labels are unavailable to the deployable method. The Random-minus-Oracle-operator DSA difference is \(0.052112\), 95% interval \([0.051231,0.052983]\). The Oracle-family-minus-Oracle-operator difference is \(0.001449\), \([0.001086,0.001790]\). These results motivate keeping the final PEW coarse rather than establishing a universal statement about operator-level environment models.

## C.3 Screening-Only JTT Boundary

JTT was evaluated only in the 12-round screening stage used for method selection. Those screening values are not used as final scientific win/loss evidence. The main text therefore discusses JTT conceptually \citep{liu2021jtt} but makes no claim that PEW+BER formally outperforms JTT.

# References

Bibliographic records are maintained in the adjacent `references.bib` and cited with natbib-style commands. FedCD remains cited as an arXiv preprint unless a later formal version is independently verified before submission.
