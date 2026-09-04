# FedPRIME-D2C 文档总索引

Updated: 2026-09-04

2026-09-04 K1-C-FULL 在没有任何 Formal 科学结果前冻结为
`SUPERSEDED_BEFORE_FORMAL`，禁止重启旧 calibration/formal。K1-C0 的 10/10 PASS 只证明强
CLE checkpoint 存在可测的 response-spectrum concentration，不预测 CRSF 一定成功。

当前唯一活动阶段是低成本 `K1-C-Minimal Causal Intervention Gate`：H9/L9、ResNet10 与
MobileNetV2、A→B、Frozen/CRSF/RawSpec；用预先冻结的 512 carriers × 16 Bank-A probes 做5个
accepted steps，仍以完整独立 2,000 × 64 Bank-B 评价，并在 taxonomy-free 结果封存后才读取
CLE oracle/DSA。其配置、实现和 OpenI 入口为：

```text
docs/experiments/current/CLE_K1_C_MINIMAL_CAUSAL_GATE_OPENI_ZH.md
configs/cle_k1_c_minimal_seed0.json
fedprime/engine/cle_crsf_surgery.py
scripts/run_cle_k1_c_minimal.py
scripts/openi_cle_k1_c_minimal_entry.py
tests/test_cle_k1_c_minimal.py
```

聚焦回归 14/14 PASS；真实 checkpoint CUDA smoke 已完整通过，且没有加载 oracle/evaluation。
OpenI benchmark 已独立审计：Minimal Formal 线性外推约14.91分钟/0.2484单卡GPU-hours，保守
预留30--45分钟；成本门通过，但不构成科学结果。报告位于：

```text
deliverables/cle_k1_c_minimal_benchmark_20260904/RESULT_SUMMARY_ZH.md
```

K1-C-Minimal Formal 随后完成并独立审计，正式 verdict 为 `NO_GO_CRSF_INTERVENTION`。H9/L9
未见chi仅下降`5.369%/5.452%`，DSA仅下降`0.005237/0.005193`，远低于冻结门槛；效果主要来自
ResNet10，MobileNetV2近乎无效。全部artifact/seal hash、moments/Gram、DSA与任务指标均独立
重算一致，8条轨迹均完成5个accepted steps且KL合规。因此停止CRSF，不做B→A、剩余架构、调参、
replication或完整训练。正式报告：

```text
deliverables/cle_k1_c_minimal_formal_20260904/RESULT_SUMMARY_ZH.md
```

K1-C0仅保留为观察性机制发现，K0-B仅保留为离线审计。

2026-09-02 K1-A head-only SDMN formal 已冻结为 `NO_GO_DIRECTIONAL_SURGERY`。当前唯一活动阶段是
零训练 K1-B0 CDR-SNR shared-representation localization：复用 K0-B high-risk probes，检查
penultimate representation 中的 carrier-stable、matched-low-specific、cross-bank transferable
nuisance subspace。入口与冻结 20-gate 协议：

```text
docs/experiments/current/CLE_K1_B0_CDR_SNR_OPENI_ZH.md
scripts/openi_cle_k1_b0_cdr_snr_entry.py --mode=formal
```

本地 INSPECT/tiny smoke 已通过；继续使用原 Phase-B0 510.46 MiB OpenI 数据集，不重新上传，
不读取标签/evaluation assets，不训练、不写 checkpoint。正式结果出来后停止并独立复算。

2026-09-02 K0-B v2 formal已强通过，并启动K1-A head-only SDMN checkpoint surgery：

```text
docs/experiments/current/CLE_K1_SDMN_HEADONLY_OPENI_ZH.md
fedprime/engine/cle_sdmn_headonly.py
scripts/run_cle_k1_sdmn_headonly.py
scripts/openi_cle_k1_sdmn_headonly_entry.py
tests/test_cle_sdmn_headonly.py
```

K1-A复用冻结checkpoint、PRIME Bank A/B和K0-B风险，在互斥discover/surgery/holdout公共
载体上只修改分类头，并包含Direction-Sham、sensitivity-matched Random-Probe与Generic
Invariance对照。INSPECT通过，聚焦测试20/20通过，本地tiny smoke两次确定性一致，判定仅为
`SMOKE_ONLY_NO_SCIENTIFIC_DECISION`。Formal入口保持锁定；下一步先OpenI smoke，再运行
不读取DSA/WCCA/CFG的数值calibration。

K0-B v2入口与结果：

```text
docs/experiments/current/CLE_GENERIC_PROBE_K0B_OPENI_ZH.md
fedprime/augmentations/frozen_prime.py
fedprime/engine/cle_generic_probe_gate.py
scripts/analyze_cle_generic_probe_k0b.py
scripts/openi_cle_generic_probe_k0b_entry.py
tests/test_cle_generic_probe_k0b.py
```

K0-B 冻结两套各64个 PRIME recipe 的完整state，检测跨500/500公共载体稳定且类别选择性的
方向风险；primary不读取corruption taxonomy、severity、family、binding或公共标签。聚焦
K0-B+K0-A regression为14/14 PASS。正式OpenI及独立response重算均为
`GO_TO_K1_CHECKPOINT_SURGERY`：HFL/Local R ratio为`4.90/4.39`，各4/4客户端同向，两套bank
均独立复现。该结果只授权K1-A最小机制手术，不授权完整训练。

2026-09-01 CLE-HFL 的当前唯一活动门槛是零训练 K0-A public-carrier transfer oracle：

```text
docs/experiments/current/CLE_PUBLIC_CARRIER_K0A_OPENI_ZH.md
```

它复用 H0/H9/L0/L9 的16个 round-40 checkpoint，在不使用 CIFAR-100 标签的1,000个固定
公共载体上施加16个 oracle operators，检验跨载体类别方向矩能否恢复隐藏binding。正式
结果为 `GO_TO_K0_B`：HFL 与 Local 各10/10冻结门槛通过，报告位于：

```text
deliverables/cle_public_carrier_k0a_20260901/RESULT_SUMMARY_ZH.md
```

K0-A 不训练模型、不修改通信，也不复活已失败的 PNCB bridge；它只允许下一步设计
taxonomy-free K0-B generic probes，不允许直接进入DME/K1训练。

2026-08-31 Phase-B0 formal 已完成，当前 PNCB bridge 正式 `NO-GO`：

```text
deliverables/cle_public_canonicalization_phase_b0_20260831/RESULT_SUMMARY_ZH.md
```

G1/G4/G6/G7通过，G2/G3/G5失败。PNCB保住语义，但同源跨operator方差反而增加12.23%，
family可分性仅下降9.47%，Local retrieval也未达门槛。禁止进入Phase-B1分类器/SCDW训练，
禁止用SCDW权重或仅调epoch/channel/loss救当前bridge。

2026-08-31 生成供 GPT Web 讨论的 CLE-PNCB-SCDW 当前课题完整交接稿：

```text
docs/research/status/CLE_PNCB_SCDW_CURRENT_RESEARCH_HANDOFF_FOR_GPTWEB_2026_08_31_ZH.md
```

该文档自包含地记录了课题定义、RAHFL/PEW转向原因、Phase-A0/A1a/PIDR证据、PNCB与SCDW
数学设计、最弱假设、Phase-B0数据与七项冻结门槛、Smoke结果、Phase-B1条件计划、论文边界
和建议外部AI重点攻击的八个问题。正式Phase-B0结果返回前，当前状态仍为
`PNCB-SCDW CONDITIONAL GO / NOT YET EVALUATED`。

2026-08-30 完成 PIDR 之后的最小干预桥与训练目标纸面设计：

```text
docs/research/status/CLE_PUBLIC_CANONICALIZATION_DIRECTIONAL_WITHDRAWAL_DESIGN_2026_08_30_ZH.md
```

当前条件候选为 Public Nuisance Canonicalization Bridge（PNCB）+ Signed
Class-Directional Withdrawal（SCDW）。它复用 HFL 已有公共无标签图像学习退化中和器，客户端
只惩罚原图相对中和图中统计可信的错误类别正向证据，不读取 corruption/family/binding 标签，
不修改通信。该方案仍是 `CONDITIONAL GO`：下一步必须先做 Identity/AugMix-overlay/public-
canonicalizer 三桥对照，验证语义保持、旧退化收缩、隐藏绑定恢复与 clean artifact null；
bridge gate 通过前不得实现12轮训练。

Phase-B0 冻结规范与实现入口：

```text
docs/experiments/current/CLE_PUBLIC_CANONICALIZATION_PHASE_B0_ZH.md
fedprime/models/public_canonicalizer.py
fedprime/engine/cle_directional_withdrawal.py
scripts/train_cle_public_canonicalizer_phase_b0.py
scripts/analyze_cle_public_canonicalization_phase_b0.py
scripts/prepare_cle_public_canonicalization_phase_b0_input.py
scripts/openi_cle_public_canonicalization_phase_b0_entry.py
```

Phase-B0 当前 focused tests 为 `7/7`，4图/1-batch CPU 训练器运行 smoke 已通过。Phase-A1a
的 H0/H9/L0/L9 四臂16个 round-40 最终 checkpoint 已严格加载，并连同1000张冻结评估源图
和 CIFAR-100 公共 tar 打成单一瘦身输入包。OpenI 端到端 smoke 已于2026-08-30通过，判定
`SMOKE_ONLY_NO_SCIENTIFIC_DECISION`；正式 bridge-only 运行仍未启动，等待用户批准。

2026-08-30 完成 CLE local-first directional shortcut 的定义、不可识别性反例、最弱干预
假设与 2024--2026 方法碰撞审计：

```text
docs/research/status/CLE_LOCAL_FIRST_DIRECTIONAL_SHORTCUT_IDENTIFIABILITY_AUDIT_2026_08_30_ZH.md
```

结论是：CLE 现实动机和当前受控诊断成立，DSA 可作为论文中的评价贡献，但不能声称首次发现
类别—损坏 spurious correlation。只用已损坏图像、标签和同分布 AugMix views 无法识别真实
directional shortcut；两世界反例给出严格 NO-GO。唯一条件候选 PIDR 必须增加语义保持、能
overwrite 退化且 probe identity 可区分的主动干预桥，并与 FedPIN、FedCD、ShortcutProbe、
FedDDL/FedCAug 等直接邻接方法区分。

同日完成 PIDR 零训练隐藏绑定恢复门槛。分析器只复用现有 round12/round40 prediction cache；
promotion matrix 估计阶段不读取 family id 或 binding，真值只用于最终评分。round40 的
HFL/Local gamma0.9 mAP 为 `0.844847/0.865557`，相对 gamma0 增量
`+0.402993/+0.434935`，class-to-family hit 为 `0.85/0.875`，4/4客户端同向，两类置换
`p=0.000999`，全部门槛通过，判定 `GO_TO_INTERVENTION_BRIDGE_DESIGN`。这只允许继续设计
能真实改写基础退化的 probe bridge，不允许直接实现训练 loss 或启动12轮实验。结果位于：

```text
deliverables/cle_pidr_zero_training_gate_20260830/RESULT_SUMMARY_ZH.md
```

2026-08-30 完成 Phase-A1a 四臂 OpenI 通信放大归因：

```text
docs/experiments/current/CLE_SHORTCUT_COMMUNICATION_AMPLIFICATION_PHASE_A1A_ZH.md
```

严格匹配的 `H0/H9/L0/L9` 均完成40轮，数据、初始化、fit/audit 与本地增强轨迹审计全部
通过。round-40 的 HFL/Local CLE 效应分别为 `0.202748/0.204366`，通信差分中的差分
`A_pool=-0.001618`，CI95 `[-0.003337,0.000128]`，仅1/4客户端为正；正式结论为
`NO_GO_FL_SPECIFIC_AMPLIFICATION`。CLE directional shortcut 仍然强且可复现，但机制是
local-first，不能继续讲坏教师或通信放大。用户明确继续 CLE；下一步只做无环境标签本地
shortcut-suppression 数学对象与查重，不先实现或训练。

2026-08-30 完成 CLE directional shortcut 的零训练 Phase-A0 OpenI 审计：

```text
docs/experiments/current/CLE_SHORTCUT_ALIGNMENT_PHASE_A0_OPENI_ZH.md
```

该审计只复用历史 `gamma=0/0.9` checkpoint，在同一批 clean source 上生成固定 severity 的
16-operator 配对干预网格，并以 Directional Shortcut Alignment、gamma 差分及保持 group-size
的随机绑定检验判断模型是否真的把 corruption 当成类别线索。OpenI 正式结果
`delta DSA=0.201623`、CI95 `[0.196412,0.207219]`，4/4 客户端方向一致，G1--G5 全部通过；
它只允许进入 HFL-vs-Local 归因设计，不等于论文或方法 GO。

2026-08-17 启动离开损坏主线后的 CCF-B 联邦学习选题重置：

```text
docs/research/status/CCF_B_FEDERATED_TOPIC_RESET_SCREEN_2026_08_17_ZH.md
```

异构 LoRA、联邦 calibration/conformal、开放世界类别发现、缺失模态多模态 FL、跨站因果
overlap 恢复和通用 pairwise risk 因 2024--2026 直接工作在第一轮淘汰。当前唯一进入严格
理论门槛的主候选是“客户端专有且部分重叠标注源下的联邦规则可靠性补全”：客户端只上传
规则对的一致性充分统计量，不传样本、规则代码或逐样本弱标签矩阵；研究每个局部规则图均
不可识别、但全局共现图在连通、非二分和方向锚条件下可识别的边界。该方向必须与 WSHFL、
FlyingSquid、Snorkel、联邦弱监督专利和 truth inference 明确区分，当前仅为
`CONDITIONAL GO FOR THEORY GATE`，不实现、不实验、不提交。联邦全局生存排序只保留为
`HIGH-RISK BACKUP`。

2026-08-17 完成“客户端严重度区间缺失下的联邦序数边界补全”纸面审计：

```text
docs/archive/methods/FEDERATED_ORDINAL_BOUNDARY_COMPLETION_AUDIT_2026_08_17_ZH.md
```

序数任务与局部缺类破坏边界的现象真实，但直接联邦序数论文已经明确报告该问题。同构模型下
累计切点风险仍按客户端可加，一步 FedSGD 等于集中式梯度；所谓边界互补量又直接化为
`Var_w(p_i,k)`。异构模型在无公共输入/共享 proxy 时因分数尺度与输入—标签映射不对齐而不
可识别；加入桥梁后分别退化为 FedMD/FedH2L 公共 posterior 蒸馏或 FedeKD 共享 proxy 与可
靠性 gating。判定 `PAPER NO-GO`，不实现、不实验、不提交。

2026-08-17 在用户允许离开模型异构 HFL 核心、并明确拒绝 PEW/BER benchmark 退路后，完成
最后一次联邦学习新选题收敛：

```text
docs/research/status/FEDERATED_NEW_TOPIC_FINAL_SCREEN_2026_08_17_ZH.md
```

联邦表示版本兼容、客户端异构优化器、跨客户端去重和客户端标签成熟度均因直接文献或退化
为现有模块判定方法 `NO-GO`。最后的“客户端私有非 IID holdout 联邦自适应验证复用”问题
真实，逐客户端 `FVI_i(T)` 也能刻画验证乐观偏差，但未通过非可分解门槛：Reusable
Holdout/Thresholdout 已允许基于完整历史记录的任意自适应查询，逐客户端独立机制足以处理
服务器混合其他客户端回答后提出的模型；DP-Hype 又覆盖了本地评价、加噪、安全聚合与私有
联邦选择。因此本轮没有论文 `GO` 候选，暂不实现、不实验、不提交；下一步须确定是否允许
进一步离开 FL，或留在 FL 但引入新的真实数据/可观测侧信息。

2026-08-17 完成“同一客户端架构切换后的个性知识连续性”最终纸面门槛：

```text
docs/archive/methods/CLIENT_ARCHITECTURE_TRANSITION_IDENTIFIABILITY_AUDIT_2026_08_17_ZH.md
```

完整旧函数可查询时，本地 old-to-new KD 已是在新架构函数类中保存旧行为的最优投影，
联邦记录不能降低纯连续性风险；旧输出只在有限迁移集可见或完全不可见时，又存在相同通信
记录、未查询区域旧函数相反的两世界不可识别性。加入当前联邦知识后，目标退化为历史本地
教师 + 当前全局教师的双/多教师蒸馏，与 pFedSD、pFedKT、FedPSD 直接碰撞；AdaptFL、
FedKDNAS 和 cross-architecture KD 已覆盖动态架构及迁移机制。最终判定
`ATR/FTG EVALUATION GO / CCF-B METHOD CORE NO-GO / DO NOT IMPLEMENT OR RUN`。

2026-08-17 完成非损坏方向的 CCF-B 选题收敛矩阵：

```text
docs/research/status/CCF_B_TOPIC_CONVERGENCE_MATRIX_2026_08_17_ZH.md
```

统一比较了六个候选。公共代理覆盖、接收者容量/可学习性、异构标签空间和模型异构联邦
遗忘均因直接工作或当前实现约束判定 `NO-GO`。主候选仅保留为“同一客户端架构切换后的
个性知识连续性”，但 AdaptFL/FedKDNAS 已覆盖实时资源变化、逐轮架构选择与联邦 KD，所以下
一步必须先证明它超过普通本地 old-to-new KD 且不退化为双教师蒸馏。后备仅为 matched
backbone 下的架构条件协作收益诊断/benchmark，不是已通过的方法论文。当前仍不实现、不实验。

2026-08-17 完成重加入客户端“遗漏知识路径”的最终理论门槛：

```text
docs/archive/methods/MISSED_KNOWLEDGE_PATH_THEORY_GATE_2026_08_17_ZH.md
```

构造了非单射异构学生的严格最小反例：最终教师和时间均值可停在信息退化点，正序与倒序
回放会进入相反分支，因此“顺序可能携带最终快照没有的信息”在数学上成立；同时给出了初始
陈旧误差、加权路径压缩误差和局部求解误差的标准收缩跟踪界。但 Pro-KD、progressive
distillation implicit curriculum、Continuation-KD、FAPD、FedGKD、FedLFH 以及既有
dynamic-regret path-variation 理论已经分别覆盖核心机制和理论。最终判定
`MATHEMATICAL PHENOMENON GO / CCF-B CORE NO-GO / DO NOT IMPLEMENT`。

2026-08-17 完成“模型异构联邦蒸馏中重加入客户端知识恢复”的三个问题精确审计：

```text
docs/research/status/REJOINING_HETERO_LOGIT_RECOVERY_AUDIT_2026_08_17_ZH.md
```

泛化的客户端快速恢复已与 DFedCAD、FedAPP、BRIDGE-T、动态到离 FL、FARe-DUST 和
FedRevive 实质碰撞；旧本地锚点 + 当前联邦教师的双蒸馏也与 FCCL、pFedDB 及历史教师
蒸馏碰撞，二者均 `NO-GO`。仅保留一个窄缺口：返回客户端在任意模型架构和公共-logit-only
协议下，有序补学离线期间遗漏的聚合联邦知识路径。该候选为 `CONDITIONAL GO FOR THEORY
ONLY`，必须先证明它不能退化为双 KL / 无序多教师 KD，并闭合 endpoint-only 反例、路径变化
与压缩误差跟踪界，暂不实现、不实验。

2026-08-17 完成“模型异构公共 Logit 隐私泄漏与防护”的新课题纸面审计：

```text
docs/archive/methods/MODEL_HETERO_PUBLIC_LOGIT_PRIVACY_AUDIT_2026_08_17_ZH.md
```

当前通信确实逐客户端、逐公共样本、跨轮暴露完整 softmax，存在标签分布与成员泄漏风险；
但 PDA-FD 攻击、架构相关隐私差异、异构/协同隐私防御和安全 HFL-FD 聚合均已有强先例。
严格 DP 的最坏概率输出敏感度不依赖 backbone，经验架构感知扰动又不能提供最坏保证。
判定 `PAPER NO-GO / DO NOT IMPLEMENT / NO EXPERIMENT`。隐私预算 teacher-query 调度仅保留
为理论未闭合的 conditional idea，不进入 runner。

2026-08-17 完成“复合退化不变知识＋联邦复合交互风险”的单路线纸面审计：

```text
docs/archive/methods/COMPOUND_DEGRADATION_INVARIANT_KNOWLEDGE_AUDIT_2026_08_17_ZH.md
```

复合退化作为更严格评价协议成立，但不是新问题贡献。唯一候选 FCIR 在训练增强库内可由
边际损害与正交互风险得到望远镜组合界；该界不能无条件推广到未知真实退化。本地行为与
AugMix/AugMax/CoCor/straightening 相邻，联邦交互表退化为 FedAvP 式共享增强策略，且
没有 BER 弱单元质量链条。判定 `PAPER NO-GO / DO NOT IMPLEMENT / NO EXPERIMENT`。

2026-08-17 完成“类别条件隐藏退化风险可识别性＋异构蒸馏捷径传播”的正式数学与查重审计：

```text
docs/archive/methods/LATENT_DEGRADATION_RISK_IDENTIFIABILITY_AUDIT_2026_08_17_ZH.md
```

`r_c = Pi_c rho_c`、两世界不可识别反例、秩条件、通信 transcript 不增信息以及模型异构
混淆均成立，但核心分别与 fairness without demographics、Federated Fairness without
Sensitive Groups、aggregate-statistic bounds、mutual-contamination identifiability 和 KD
mechanism transfer 发生实质碰撞。判定 `P0/P1 PAPER NO-GO AS STANDALONE CORE`，不实现
identified-set 审计、不做通信 DiD、不跑实验。

2026-08-17 完成“模型异构联邦安全持续测试时适应”的不可识别性、最弱假设与外部碰撞审计：

```text
docs/archive/methods/SAFE_MODEL_HETERO_FTTA_IDENTIFIABILITY_AUDIT_2026_08_17_ZH.md
```

fully-unlabeled 目标流下，协作收益符号不可识别；source audit、公共响应和多模型一致性不能
自动提供 target no-harm 保证。候选同时与 FedTHE/ATP/FedTSA/CoLA/FedCTTA/Latte 以及
AETTA/TTA risk monitoring 高度相邻，判定 `PAPER NO-GO`，不进入实现或实验。稀疏无偏
延迟任务标签仅为 `CONDITIONAL REFRAME`，属于新的 online/continual learning 问题。

2026-08-17 完成三个额外信息候选的数学设计与原始论文查重：

```text
docs/archive/methods/FCNT_FPER_FRT_THEORY_NOVELTY_AUDIT_2026_08_17_ZH.md
```

FCNT、FPER、FRT 均未通过当前协议下的论文核心门槛，不进入实现或实验。FCNT 只在显式增加
可信连续设备元数据的新问题设定下 `CONDITIONAL REFRAME`；FPER 只允许作为 paired/clean-source
oracle；FRT 因不可识别性、BER 链条缺失及 CCAD/IRD/FedCIS/FCCL+ 碰撞判定 `NO-GO`。

2026-08-16 完成 PEW+BER 论文主张、实现、证据与外部新颖性审计：

```text
docs/research/status/PEW_BER_PAPER_CLAIM_AUDIT_2026_08_16_ZH.md
```

判决为 `CORE-METHOD NO-GO / BASELINE GO / BENCHMARK CONDITIONAL GO`。PEW+BER
保留为强 taxonomy-assisted 诊断基线和后续方法必须保住的 empirical target，不能
作为唯一论文核心创新；CLE-HFL 是 centralized class-corruption spurious correlation
在模型异构联邦下的受控扩展，不能声称首次提出类别-损坏纠缠。

已完成的 LCC 数学定义、外部查重与 taxonomy-free 可识别性边界：

```text
docs/archive/methods/LCC_NOVELTY_AUDIT_ZH.md                 THEORY NO-GO；GRASP/GoG/MGDA 碰撞
docs/research/status/TAXONOMY_FREE_IDENTIFIABILITY_2026_08_11_ZH.md  客户端混合对比的秩与覆盖边界
scripts/audit_mixture_contrast_identifiability.py            纯 metadata 秩审计，无训练
```

已完成并冻结的 CRSR 类条件预测残差谱风险审计（`NO-GO`；G0--G3 通过，
G4--G6 失败）：

```text
docs/experiments/archive/CLASS_RESIDUAL_SPECTRAL_RISK_AUDIT_ZH.md
fedprime/methods/class_residual_spectral_risk.py
scripts/audit_class_residual_spectral_risk.py
```

已完成并冻结的 taxonomy-free 本地信号审计：

```text
docs/experiments/archive/CLASS_CONDITIONAL_COUNTERFACTUAL_REGRET_AUDIT_ZH.md
```

已完成并冻结的 taxonomy-free 本地表征审计：

```text
docs/experiments/archive/FEDLENS_PIE_AUDIT_ZH.md
docs/experiments/archive/FEDLENS_MPIE_CONFIRMATORY_AUDIT_ZH.md
scripts/audit_fedlens_pie.py
scripts/audit_fedlens_mpie_confirmatory.py
```

当前项目框架一屏说明：

```text
docs/research/status/CURRENT_FRAMEWORK_2026_08_10_ZH.md
```

已完成且判定为 `NO-GO (0/4 gates)` 的 Multi-label PEW + Soft-BER 配对筛选：

```text
docs/experiments/current/CLE_MULTILABEL_PEW_SOFTBER_OPENI_RUN_ZH.md
scripts/openi_cle_multilabel_softber_entry.py
```

## 2026-08-09 基线公平性与忠实度修复

```text
docs/research/baselines/BASELINE_FIDELITY_REPAIR_ZH.md
docs/experiments/current/CLE_BASELINE_FIDELITY_OPENI_RUN_ZH.md
deliverables/baseline_fairness_audit_20260809/BASELINE_FAIRNESS_AUDIT_ZH.md
```

## 新会话最小读取顺序

```text
AGENTS.md                  执行约束、科研纪律和文档放置规则
docs/handoffs/latest.md    当前目标、正式实验、待决策事项与下一入口
docs/README_ZH.md          按任务定位其余文档（本文件）
```

默认到此停止。只有当前任务需要精确历史、实现位置或旧实验依据时，才继续读取对应文件；禁止为了“全面理解项目”一次性加载全部长文档。

## 事实来源优先级

```text
相关代码/配置与当前 Git diff
> docs/handoffs/latest.md
> 当前实验指南
> docs/project/ 下的长期日志
> docs/archive/ 与历史 deliverables
```

## 最新正式实验

当前论文证据补全总入口：

```text
docs/experiments/current/CLE_HFL_PAPER_EVIDENCE_OPENI_RUN_ZH.md
docs/experiments/archive/CLE_PEW_LOO_OPENI_RUN_ZH.md               已完成 Strict PEW operator-LOO（GO）
docs/experiments/archive/CLE_REMAINING_BASELINES_OPENI_RUN_ZH.md   已完成 FedDF/KT-pFL/FCCL 匹配筛选
docs/experiments/archive/CLE_CDEP_V2_PAIRED_OPENI_RUN_ZH.md      已完成的 CDep-v2 共享PEW配对实验（NO-GO）
docs/experiments/archive/CLE_CDEP_V2_SINGLE_ARM_OPENI_RUN_ZH.md   已完成但归因不匹配的单臂实验
```

```text
docs/experiments/archive/STRICT_PEW_ASYMHFL_VAL_OPENI_RUN_ZH.md  已完成的 seed-0 strict A/B 说明（GO）
docs/experiments/archive/STRICT_PEW_ASYMHFL_VAL_MULTISEED_OPENI_RUN_ZH.md  已完成的 training-seed 0/1/2 复验（GO）
docs/experiments/current/STRICT_PEW_ASYMHFL_VAL_40ROUND_OPENI_RUN_ZH.md  当前 40 轮 training-seed 1/2 复验说明
docs/experiments/guides/CLE_HFL_V2_OPENI_RUN_GUIDE_ZH.md        CLE-HFL v2 通用平台说明
```

## 长期记忆与代码地图

这些文件很大，只在需要追溯精确历史时按需读取：

```text
docs/project/CURRENT_PROJECT_MEMORY.md  按时间记录的重要实验决策与结果
docs/project/PROJECT_STATE.md           已实现代码和实验状态的长日志
docs/project/TODO_NEXT.md               历史与当前待办日志
docs/project/ARCHITECTURE.md            代码模块和 runner 架构
docs/experiments/guides/EXPERIMENT_GUIDE_ZH.md      配置、指标与实验运行指南
docs/project/RESEARCH_CODE_PRACTICES_ZH.md  科研代码实践
```

## 基线理解

```text
docs/research/baselines/RAHFL_IMPLEMENTATION_READING_ZH.md  RAHFL 源码与训练流程精读
docs/archive/legacy/AGENT.md                            早期 D2C 约束，已经过时，禁止作为当前指令
docs/archive/legacy/experiment_plan.md                  最早期 D2C 实验计划，仅作历史背景
docs/archive/legacy/ARCHITECTURE_PRE_CLEANUP_2026_08_09.md  清理前的完整历史架构快照
```

## CLE-HFL 与当前研究问题

```text
docs/archive/methods/FEDCLEAR_CLE_HFL_PROPOSAL_ZH.md
docs/archive/methods/CLE_HFL_V2_FEDFALSIFY_FRAMEWORK_ZH.md
docs/research/status/CURRENT_RESEARCH_STATUS_RAHFL_AND_COMMUNICATION_REVIEW_20260727_ZH.md
docs/research/status/CURRENT_RESEARCH_STATUS_FOR_EXTERNAL_AI_ZH.md
```

## 已归档的候选方法与负结果

```text
docs/archive/methods/NEGATIVE_CODE_REMOVAL_INDEX_ZH.md
docs/archive/methods/CONTINUOUS_WITNESS_OFFLINE_AUDIT_ZH.md
docs/archive/methods/FEDCIS_FRAMEWORK_AND_OFFLINE_AUDIT_ZH.md
docs/archive/methods/FEDCISA_FRAMEWORK_REVIEW_AND_MODULE_SPEC_ZH.md
docs/experiments/archive/FEDFALSIFY_AUDIT_GUIDE_ZH.md
docs/experiments/archive/FEDFALSIFY_OPENI_RUN_GUIDE_ZH.md
docs/archive/methods/FEDFALSIFY_LATEST_EXTERNAL_AI_DISCUSSION_BRIEF_ZH.md
docs/archive/methods/FEDCLEAR_LATEST_THEORY_FRAMEWORK_ZH.md
docs/archive/methods/FEDCLEAR_METHOD_DESIGN_REVIEW_ZH.md
docs/archive/methods/FEDCLEAR_V2_REVISION_PLAN_ZH.md
docs/archive/methods/FEDCLEAR_EXTERNAL_AI_REVIEW_BRIEF_ZH.md
docs/archive/methods/FEDCARA_CURRENT_FRAMEWORK_EXPERIMENTS_AND_NEXT_PLAN_ZH.md
docs/experiments/archive/FEDSARA_CS_SCENARIO_OPENI_GUIDE_ZH.md
docs/archive/methods/FRAMEWORK_REVIEW_C3LP_D2CCR_ZH.md
docs/archive/methods/PROTOGRAPH_LITERATURE_REVIEW_AND_CTPG_FRAMEWORK_ZH.md
docs/archive/methods/C3LP_D2C_LITERATURE_AND_METHOD_ZH.md
```

这些文件用于保留研究证据，不代表仍应继续实现其中的方法。

## 历史 FedEASE 运行资料

```text
docs/archive/methods/FEDEASE_V2_1_FRAMEWORK_AND_IMPLEMENTATION_ZH.md
docs/experiments/archive/FEDEASE_OPENI_RUN_GUIDE_ZH.md
docs/experiments/archive/FEDEASE_CALIBRATED_PEW_LOCAL_OPENI_RUN_ZH.md
docs/experiments/archive/FEDEASE_EBST_V2_OPENI_RUN_ZH.md
docs/experiments/archive/FEDEASE_PEW_EBST_V2_OPENI_RUN_ZH.md
```

当前只复用其中已经验证过的 calibrated PEW + BER+CDep 本地机制；EBST
与 EBST-v2 已经冻结为负结果。

## 汇报与外部讨论稿

```text
docs/research/status/RECENT_PROGRESS_REPORT_2026_07_20_ZH.md
docs/research/status/EXTERNAL_AI_RESEARCH_REVIEW_BRIEF_2026_07_22_ZH.md
docs/research/status/CURRENT_RESEARCH_STATUS_FOR_EXTERNAL_AI_ZH.md
docs/archive/methods/FEDCLEAR_EXTERNAL_AI_REVIEW_BRIEF_ZH.md
docs/archive/methods/FEDFALSIFY_LATEST_EXTERNAL_AI_DISCUSSION_BRIEF_ZH.md
```

## 结果产物的位置

```text
outputs/       原始压缩包和运行输出，默认不提交 Git
deliverables/  已解析的表格、图和报告
local_runs/    本地数据和临时运行，默认不提交 Git
```

紧凑的 RAHFL CLE 基线轮次表位于：

```text
deliverables/baselines/rahfl_cle_alpha05_gamma09_seed0_round00_11.csv
deliverables/strict_pew_asymhfl_val_probe_20260804/RESULT_SUMMARY_ZH.md
deliverables/strict_pew_asymhfl_val_multiseed_20260804/RESULT_SUMMARY_ZH.md
deliverables/strict_pew_asymhfl_val_40round_seed0_20260805/RESULT_SUMMARY_ZH.md
```

## 文档目录职责

```text
docs/handoffs/             当前交接，只保留 latest.md
docs/project/              架构、长期记忆、实现状态、TODO、科研代码规范
docs/experiments/current/  正在运行或等待正式结果的实验
docs/experiments/guides/   可复用的基准与平台指南
docs/experiments/archive/  已完成、已替代或已冻结实验的运行资料
docs/research/status/      当前研究综述与带日期的阶段报告
docs/research/baselines/   基线实现精读
docs/archive/methods/      失败、冻结或被替代的方法证据
docs/archive/legacy/       仅保留来源的过时说明
```

## 后续维护规则

1. 根目录只保留 `README.md` 和 `AGENTS.md`，不再新增研究 Markdown。
2. 新正式结果先更新 `docs/handoffs/latest.md`，再按需追加长期记忆或产出分析报告。
3. 新建、移动、归档文档时同步更新本索引和有效交叉引用。
4. 失败方法保留证据并移入归档，不把它重新写成活动 TODO。
5. 原始日志、压缩包、检查点和数据不放入 `docs/`；分别使用 `outputs/`、`deliverables/`、`local_runs/`。
6. 历史 `outputs/` 与 `deliverables/` 中的快照不因目录整理而重写。
