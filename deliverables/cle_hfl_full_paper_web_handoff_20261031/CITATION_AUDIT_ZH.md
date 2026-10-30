# CLE-HFL V0.8 文献与引用审计

初次审计：2026-09-14。增量复核：2026-10-14。

## 结论

当前写作版沿用可追踪的 BibTeX 键和`references.bib`，并于2026-10-14重新核验最容易影响创新边界的近邻工作。论文相关工作仍应保持四条边界：

1. 不宣称首次研究联邦学习中的 spurious correlation；
2. 不宣称首次进行环境推断、伪组发现、group reweighting 或 counterfactual shortcut evaluation；
3. PEW 明确定位为 taxonomy-assisted corruption-family witness，而不是 taxonomy-free latent-environment discovery；
4. 项目中的 CVaR-DRO 是协议匹配的尾部风险对照，不宣称逐行复现某一篇论文。

## 已核验的正式发表工作

| 方向 | 工作 | 正式状态 | 在本文中的用途 |
|---|---|---|---|
| 异构联邦蒸馏 | FedDF | NeurIPS 2020 | 说明公共/无标签数据上的 ensemble distillation 可融合异构客户端知识 |
| 鲁棒异构联邦 | RHFL | CVPR 2022 | 噪声标签与异构客户端背景 |
| corruption-robust HFL | AugHFL | ICCV 2023 | 当前训练底座的最近技术前身 |
| corruption-robust HFL | RAHFL | IEEE TPAMI 2025 | AugHFL 的期刊扩展及非对称协作背景 |
| common corruptions | CIFAR-C/ImageNet-C | ICLR 2019 | corruption benchmark 与 taxonomy 背景 |
| corruption robustness | AugMix | ICLR 2020 | 多视图增强与 JSD 一致性背景 |
| group robustness | GroupDRO | ICLR 2020 | 已知组标签下的 worst-group 优化 |
| hidden groups | GEORGE | NeurIPS 2020 | 学习表示聚类后进行 group-robust learning |
| environment inference | EIIL | ICML 2021 | 不依赖预定义 taxonomy 的环境推断挑战 |
| hard-example reweighting | JTT | ICML 2021 | 两阶段错误样本上采样对照概念 |
| tail risk | CVaR / DRO | Journal of Risk 2000；Annals of Statistics 2021 | CVaR 定义与机器学习 DRO 基础 |
| federated shortcuts | Personalized FL with Spurious Features | TMLR 2024 | 个性化阶段重新利用 spurious features 的相邻工作 |
| federated shortcuts | FedPIN | ICML 2024 | 因果/信息论的个性化联邦不变学习 |
| counterfactual diagnosis | Vigneshwaran et al. | MIDL 2026 | 医学影像中反事实 shortcut 利用评价的先例 |
| general shortcuts | Geirhos et al. | Nature Machine Intelligence 2020 | shortcut learning 总体背景 |
| medical shortcuts | Jabbour et al. | MLHC 2020 | 胸片采集来源等 shortcut 案例 |

## 仅预印本的工作

- **FedCD**：截至2026-10-14，只确认到 arXiv:2407.19174 的 v1（2024-07-27）。当前稿明确写为 arXiv preprint，不升级为正式会议或期刊论文。若投稿前发现正式版本，应重新核验标题、作者、方法和发表信息，而不是只改 venue 字段。

## 2026-10-14增量核验记录

- **FedPIN**：PMLR v235 正式页面确认 Tang 等人的 *Causally Motivated Personalized Federated Invariant Learning with Shortcut-Averse Information-Theoretic Regularization* 发表于ICML 2024；当前`references.bib`的标题、作者、卷号、页码和落地页相互一致。正文只采用“相邻的个性化联邦不变学习”表述，不以其支撑本文首创性。
- **Counterfactual shortcut evaluation**：PMLR v301 正式页面确认 Vigneshwaran 等工作发表于 MIDL 2026。它说明“使用反事实干预检查 shortcut”不是本文首创；本文差异必须落在同源跨operator网格、预注册binding方向、完整概率质量DSA和HFL归因上。
- **FedCD**：arXiv页面仍只显示2024年v1；继续按预印本处理。
- **MIDL venue修正**：PMLR正式页面写作“8th Conference on Medical Imaging with Deep Learning”，原BibTeX中的“9th”已修正。
- **FedProto/FedTGP**：AAAI正式页面确认标题、作者、卷期、页码与DOI；已补入官方URL。
- **Personalized FL with Spurious Features**：OpenReview正式PDF标明“Published in Transactions on Machine Learning Research (03/2024)”，与当前TMLR 2024记录一致。

## 原文核验覆盖表

| 键 | 官方/作者侧落地页 | 核验结果 | 本文仅允许支撑 |
|---|---|---|---|
| `lin2020feddf` | NeurIPS proceedings | 题名、作者、venue一致 | 异构模型的公共数据ensemble distillation |
| `fang2022rhfl` | CVF Open Access | 页码和venue一致 | noisy-label异构联邦与公共反馈重加权 |
| `fang2023aughfl` | CVF Open Access | 题名和ICCV 2023一致 | corruption-robust HFL与PubAug |
| `fang2025rahfl` | IEEE Xplore/DOI | TPAMI 47(4):2693--2705一致 | DCL与asymmetric collaboration |
| `hendrycks2019corruptions` | ICLR/OpenReview | 题名与venue一致 | common-corruption benchmark背景 |
| `cubuk2020augmix` | ICLR/OpenReview | 题名与venue一致 | AugMix与JSD一致性 |
| `sagawa2020groupdro` | ICLR/OpenReview | 题名与venue一致 | 已知group下worst-group优化 |
| `sohoni2020george` | NeurIPS proceedings | 题名、作者与方法摘要一致 | hidden subclass聚类加group robustness |
| `creager2021eiil` | PMLR v139 | 页码2189--2200一致 | 无环境标签时的环境推断 |
| `liu2021jtt` | PMLR v139 | 页码6781--6792一致 | 两阶段错误/高损失样本重采样 |
| `rockafellar2000cvar` | DOI落地页 | 书目信息保留 | CVaR定义背景 |
| `duchi2021uniform` | DOI/Annals of Statistics | 书目信息保留 | 通用DRO背景 |
| `wang2024pflspurious` | OpenReview正式PDF | TMLR 03/2024一致 | personalization可能重新利用spurious features |
| `tang2024fedpin` | PMLR v235 | 题名、作者、页码一致 | PFL中个性化与spurious feature纠缠 |
| `ma2024fedcd` | arXiv v1 | 仅预印本，无正式venue | 联邦域泛化中的spurious correlation |
| `vigneshwaran2026counterfactual` | PMLR v301 | 页码一致；conference序号已修正 | 医学图像反事实shortcut评价先例 |
| `geirhos2020shortcut` | Nature DOI | 书目信息保留 | shortcut learning总体背景 |
| `jabbour2020chestxray` | PMLR v126 | 题名、页码一致 | 胸片采集来源shortcut实例 |
| `zhang2024fedtgp` | AAAI官方页面 | 38(15):16768--16776一致 | trainable global prototypes背景 |
| `tan2022fedproto` | AAAI官方页面 | 36(8):8432--8440一致 | prototype通信的异构FL背景 |

正式投稿前仍需按目标模板统一大小写、作者名、会议简称、DOI和URL格式；这属于格式终审，不再是相关工作事实缺口。

## workshop 与正式版本的区分

- **Personalized Federated Learning with Spurious Features**：存在 NeurIPS 2021 workshop 早期版本；正文引用使用 TMLR 2024 正式版本。两者不得被当成两篇独立正式证据重复计数。

## 对照实现忠实度边界

- **FedDF-fidelity**：项目实现用于受控地更换 HFL 通信底座，不宣称是 FedDF 官方 recipe 的逐行复现。
- **CVaR-DRO**：项目实现是与 ERM/PEW+BER 匹配的 top-tail empirical-risk control。引用 Rockafellar--Uryasev 和 Duchi--Namkoong是为说明 CVaR/DRO 基础，不代表实现逐行对应其中任一算法。
- **PEW+GroupDRO**：GroupDRO 是既有方法；“共享 PEW 分组 + GroupDRO 优化器”是本项目构造的 matched baseline，不是本文原创方法贡献。
- **JTT**：现有结果仍为 12-round screen；正文可讨论概念差异，但不能宣称 Formal 胜出。

## 投稿前仍需做的引用工作

1. 根据目标会议模板统一作者名、会议简称、DOI 与 URL 格式；
2. 检查目标模板使用`natbib`还是`biblatex`，再决定保留`\citep{}`或机械替换为会议宏；
3. 投稿前再次检索 FedCD 是否出现正式版本；
4. 若 Introduction 加入医院、汽车或工业的具体事实性案例，必须逐条增加直接支持该案例的原始来源；
5. 不把“尚未检索到联合覆盖全部要素的先行工作”改写成绝对的“首次”。
