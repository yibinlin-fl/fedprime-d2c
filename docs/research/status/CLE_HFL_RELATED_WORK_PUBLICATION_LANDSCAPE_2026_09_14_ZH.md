# CLE-HFL 相关工作发表格局与创新边界

Updated: 2026-09-14

> 用途：供导师快速了解本课题相关工作的发表情况，也可直接交给网页端 GPT，作为英文初稿中
> `Introduction`、`Related Work`、贡献定位和审稿风险分析的依据。
>
> 范围说明：本文是围绕 CLE-HFL 当前论文叙事进行的代表性文献核验，不是完整系统综述或
> bibliometric survey。发表状态以论文主页、正式 proceedings、期刊 DOI/作者主页等可核验来源为准；
> 仅在层级明确时标注 CCF 类别。CCF 类别用于国内汇报，不等于论文质量或录用难度的绝对排序。

---

## 1. 给导师的结论摘要

1. 本课题位于三个已经得到高水平发表认可的方向交叉处：`模型异构联邦学习`、`corruption
   robustness`、`spurious/shortcut learning`。相邻工作已发表于 NeurIPS、ICML、ICLR、CVPR、
   ICCV、TPAMI、TMLR 等会议或期刊，因此选题不是孤立或过时的问题。本次表格收录的代表性
   工作中，15篇已经正式发表，其中11篇位于明确的CCF-A会议或期刊；另单列FedCD这一篇当前只
   核验到arXiv的相邻工作，以及CVaR/joint-DRO这一类方法家族。
2. 最接近的技术底座是 AugHFL（ICCV 2023）及其扩展 RAHFL（IEEE TPAMI 2025）。它们研究
   模型异构客户端的数据损坏和鲁棒协作，但没有把客户端特定的 `class-corruption binding`
   作为定向捷径来识别，也没有 paired operator DSA 与 HFL-vs-Local 归因。
3. 联邦学习中的 spurious correlation 已有正式先例：个性化联邦模型的 spurious-feature 问题
   发表于 TMLR 2024；FedPIN 发表于 ICML 2024。论文不能声称“首次研究联邦学习中的虚假相关”。
4. GroupDRO（ICLR 2020）、GEORGE（NeurIPS 2020）、EIIL（ICML 2021）和 JTT（ICML 2021）
   已覆盖已知组鲁棒、伪组发现和无训练组标签的困难样本重加权。PEW 或“伪环境 + 重加权”本身
   不能作为首创点。
5. 反事实检测 shortcut 也不是空白；医学影像已有正式工作通过 counterfactual analysis 判断
   site/sex 等属性是否被模型实际利用。本文的差异应落在“同源跨 corruption operator、沿预先固定
   binding 方向的概率迁移、模型异构 HFL”这一更具体的识别对象上。
6. 目前未发现一篇正式工作同时覆盖：`模型异构 HFL + 客户端特定 class-corruption 定向绑定 +
   paired operator 反事实 + DSA 识别/不可识别理论 + local-first 归因 + 本地结构性缓解`。这是
   当前最强且最诚实的差异化空间，但在投稿前仍需做一次可复现的系统检索，不能直接写成绝对“首次”。
7. 当前论文最合理的贡献单位不是 PEW 网络结构，而是完整证据链：

   ```text
   CLE-HFL具体问题
   + paired DSA诊断及理论
   + local-first机制归因
   + taxonomy-assisted PEW/BER干预
   + 跨binding map/通信底座的受控验证与边界
   ```

8. 发表格局说明该课题具备 CCF-B 论文的合理基础；能否达到最终投稿标准，主要取决于第二数据集、
   training-seed 稳定性、相关工作对照的忠实度和对 taxonomy-assisted/合成场景边界的诚实处理，
   而不是继续向 PEW+BER 任意堆叠模块。

---

## 2. 相关研究的发表时间线

```text
2019  ICLR       common-corruption benchmark（ImageNet-C/CIFAR-C）
2020  ICLR       AugMix；GroupDRO
2020  NeurIPS    FedDF；GEORGE
2020  NatMI      shortcut learning统一视角
2020  MLHC       医疗图像真实shortcut案例
2021  ICML       EIIL；JTT
2021  NeurIPS-W  个性化FL中的spurious feature早期版本
2022  CVPR       RHFL：模型异构+标签噪声
2023  ICCV       AugHFL：模型异构+数据corruption
2024  ICML       FedPIN：个性化FL中的shortcut-averse invariant learning
2024  TMLR       Personalized FL with Spurious Features正式期刊版
2024  arXiv      FedCD：联邦域泛化中的spurious correlation（未核实到正式venue）
2025  TPAMI      RAHFL：AugHFL扩展版，DCL+非对称协作
2026  MIDL       医学影像counterfactual shortcut utilization
```

这条时间线反映出：通用 shortcut/group robustness 的方法基础主要形成于 2020--2021 年；
2022--2025 年开始出现明确结合模型异构联邦、数据损坏或联邦 spurious correlation 的工作。
CLE-HFL 的机会不在于重新发明“spurious correlation”或“group reweighting”，而在于指出并识别
一个现有 HFL corruption 研究未单独处理的定向机制。

---

## 3. 第一组：corruption robustness 与一致性学习

| 工作 | 正式发表情况 | 核心对象 | 与 CLE-HFL 的关系 |
|---|---|---|---|
| Hendrycks & Dietterich, *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations* | ICLR 2019，正式主会，CCF-A | 建立 ImageNet-C、CIFAR-10-C、CIFAR-100-C 等 common-corruption benchmark | 为本文 operator 和常见损坏提供标准来源；它把 corruption 当作 nuisance，不研究其与类别形成客户端特定编码 |
| Hendrycks et al., *AugMix* | ICLR 2020，正式主会，CCF-A | 混合随机增强，并用 JSD 约束同一样本多视图预测一致 | RAHFL 本地鲁棒训练的重要基础；同一 operator 邻域的一致性不推出跨 operator 的 binding-independent |

### 对本文最重要的区分

AugMix/JSD 回答的是：

\[
p_\theta(y\mid x_o)\approx p_\theta(y\mid a_1(x_o))
\approx p_\theta(y\mid a_2(x_o)).
\]

DSA 回答的是：当同一个 clean source 在不同 operator 下变化时，预测概率是否沿客户端训练时的
类别—operator 绑定方向系统迁移。前者可以为零，而后者仍显著非零。因此本文不能把 PEW+BER
写成“比 AugMix 更强的普通 corruption robustness”，而应写成二者处理不同统计对象。

### 主要来源

- [ICLR 2019 common-corruption benchmark与官方代码](https://github.com/hendrycks/robustness)
- [AugMix，ICLR 2020正式论文](https://openreview.net/pdf?id=S1gmrxHFvB)

---

## 4. 第二组：模型异构联邦学习与损坏鲁棒性

| 工作 | 正式发表情况 | 核心对象 | 与 CLE-HFL 的重合和差异 |
|---|---|---|---|
| Lin et al., *FedDF: Ensemble Distillation for Robust Model Fusion in Federated Learning* | NeurIPS 2020，正式主会，CCF-A | 用无标签公共数据上的集成蒸馏融合异构客户端模型 | 提供另一种异构通信底座；不研究 corruption shortcut。本文已做 native-CE FedDF-fidelity 的支持性跨底座验证 |
| Fang & Ye, *RHFL: Robust Federated Learning With Noisy and Heterogeneous Clients* | CVPR 2022，正式主会，CCF-A | 模型异构、标签噪声、公共 logits 对齐及客户端置信重加权 | 处理 label noise 和坏客户端反馈，不等于输入 corruption 与类别之间的定向捷径 |
| Fang et al., *AugHFL: Robust Heterogeneous Federated Learning under Data Corruption* | ICCV 2023，正式主会，CCF-A | 模型异构客户端、common corruption、本地增强鲁棒学习与通信重加权 | 当前 CLE-HFL 最直接的前身；其目标是总体 corruption robustness，没有 paired binding-specific shortcut 识别 |
| Fang, Ye & Du, *RAHFL: Robust Asymmetric Heterogeneous Federated Learning With Corrupted Clients* | IEEE TPAMI 2025，正式期刊论文，CCF-A；DOI `10.1109/TPAMI.2025.3527137` | AugHFL 的扩展版；DCL 增强本地表示，AsymHFL 抑制低质量外部反馈 | 当前主要代码和训练链路底座。本文发现 strong CLE 主要 local-first，因此“坏教师/通信污染”不是 CLE 的主故事 |

### 必须向导师说明的发表关系

RAHFL 不是一篇尚未发表的普通 arXiv 工作。可核实的正式关系是：

```text
ICCV 2023 AugHFL
        ↓ 扩展
IEEE TPAMI 2025 RAHFL
```

RAHFL 的正式期刊版本使本课题的对照标准更高：论文必须明确说明自己不是重新解决“异构客户端有
corruption 时如何保持准确率”，而是揭示该设定中 corruption 可能被利用为类别代码，并给出
binding-specific 诊断、形成位置归因和针对性缓解。

### 主要来源

- [FedDF，NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/18df51b97ccd68128e994804f3eccc87-Abstract.html)
- [RHFL，CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Fang_Robust_Federated_Learning_With_Noisy_and_Heterogeneous_Clients_CVPR_2022_paper.html)
- [AugHFL，ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Fang_Robust_Heterogeneous_Federated_Learning_under_Data_Corruption_ICCV_2023_paper.html)
- [RAHFL正式版本 DOI，IEEE TPAMI 2025](https://doi.org/10.1109/TPAMI.2025.3527137)
- [RAHFL作者代码库及与AugHFL的版本关系](https://github.com/FangXiuwen/RAHFL)

---

## 5. 第三组：spurious correlation、group robustness 与伪环境发现

| 工作 | 正式发表情况 | 核心机制 | 对本文创新性的约束 |
|---|---|---|---|
| Sagawa et al., *Distributionally Robust Neural Networks for Group Shifts* | ICLR 2020，正式主会，CCF-A | 已知组标签时优化最坏组风险，强调正则化和模型选择 | “按环境组做鲁棒优化”不是本文首创；PEW+GroupDRO必须作为同分组机制对照 |
| Sohoni et al., *No Subclass Left Behind*（GEORGE） | NeurIPS 2020，正式主会，CCF-A | 用表征聚类恢复未知 subclass，再做 group DRO | “伪组 + DRO”已有强先例；PEW 的优势只能是与 corruption 机制对齐、可解释且部署边界明确 |
| Creager et al., *Environment Inference for Invariant Learning*（EIIL） | ICML 2021，正式主会，CCF-A | 不给训练环境标签，寻找最能暴露参考模型不变性违背的环境划分 | EIIL 已是 taxonomy-free 环境推断；本文不能声称首次无环境真值学习，但可比较其潜在二分环境与 CLE operator/family 的差异 |
| Liu et al., *Just Train Twice*（JTT） | ICML 2021，正式主会，CCF-A | 先训练 ERM，选出错误/高损失样本，再上采样重训 | 检验 PEW+BER 的收益是否只是困难样本重加权；JTT 不识别 corruption binding |
| CVaR / joint-DRO 类尾部风险优化 | 不是本文对应某一篇论文的逐行复现；属于成熟 DRO 家族 | 优化高损失尾部样本，而非显式类别×环境支持结构 | 本项目的 CVaR-DRO 是协议对齐的强 tail-risk baseline，不应写成“复现某篇 CVaR 论文” |

### PEW+BER 能够保留的差异

PEW+BER 与上述方法共享“非均匀重加权”这一大类思想，但操作对象不同：

- JTT/CVaR：根据错误或高损失选择困难样本；
- GroupDRO：动态追逐当前最差的已定义组；
- EIIL/GEORGE：从模型不变性违背或表征结构推断潜在环境/子类；
- PEW：用公共合成数据学习一个明确的 corruption-family witness，不读取私有真实环境标签；
- BER：在每个类别内部，根据预测环境支持量压缩环境不平衡，而不是直接优化最高损失组。

因此安全表述是“由 CLE 支持结构导出的 taxonomy-assisted 类别内环境平衡”，而不是“首次伪环境
发现”或“首次环境重加权”。

### 主要来源

- [GroupDRO，ICLR 2020](https://openreview.net/forum?id=ryxGuJrFvS)
- [GEORGE，NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/e0688d13958a19e087e123148555e4b4-Abstract.html)
- [EIIL，ICML 2021](https://proceedings.mlr.press/v139/creager21a.html)
- [JTT，ICML 2021](https://proceedings.mlr.press/v139/liu21f.html)

---

## 6. 第四组：联邦学习中的 spurious correlation 与不变学习

| 工作 | 正式发表情况 | 研究设定 | 与 CLE-HFL 的关键差异 |
|---|---|---|---|
| Wang et al., *Personalized Federated Learning with Spurious Features: An Adversarial Approach* | NeurIPS 2021 FL workshop 有早期版本；TMLR 2024 为正式期刊版 | 全局混合环境较少偏置，但本地 personalization/fine-tuning 重新利用各客户端 spurious feature | 与本文 local-first 观察最接近；但不研究模型异构公共-logit协作、class-corruption operator binding或paired DSA |
| Tang et al., *FedPIN: Causally Motivated Personalized Federated Invariant Learning with Shortcut-Averse Information-Theoretic Regularization* | ICML 2024，正式主会，CCF-A | 在个性化与去除 spurious feature 的冲突下，用因果签名和信息论正则学习 invariant/personalized 表示 | 理论与方法都比单纯PEW结构更强，是必须正面讨论的相邻工作；目标仍是 PFL OOD generalization，而非 CLE-HFL 的定向诊断和local-vs-communication归因 |
| Ma et al., *FedCD: Reducing Spurious Correlation for Federated Domain Generalization* | arXiv:2407.19174（2024）；截至本次核验未找到可确认的正式会议/期刊版本 | 本地 spurious correlation intervener + 全局 risk extrapolation，用于联邦域泛化 | 已明确研究联邦 spurious correlation；不涉及模型异构、客户端class-corruption binding、公共logit通信或paired DSA |

### 这组文献对“首次性”的影响

以下说法不可使用：

> 我们首次发现联邦学习存在 shortcut/spurious correlation。

可以使用的、更精确的表述是：

> Prior work studies generic or personalized federated spurious correlations. We isolate a distinct
> failure mode in model-heterogeneous federated learning, where client-specific class--corruption
> associations turn corruption operators into directional class codes, and we identify it with
> source-paired operator counterfactuals.

### 主要来源

- [NeurIPS 2021 workshop早期版本](https://neurips.cc/virtual/2021/35197)
- [Personalized FL with Spurious Features，TMLR 2024版本入口](https://openreview.net/pdf?id=N2wx9UVHkH)
- [FedPIN，ICML 2024](https://proceedings.mlr.press/v235/tang24a.html)
- [FedCD，arXiv 2024](https://arxiv.org/abs/2407.19174)

---

## 7. 第五组：shortcut 诊断、反事实证据与现实案例

| 工作 | 正式发表情况 | 主要结论 | 与本文关系 |
|---|---|---|---|
| Geirhos et al., *Shortcut Learning in Deep Neural Networks* | Nature Machine Intelligence 2020，正式期刊观点/综述 | 将“标准测试有效但在更困难条件下失效的决策规则”统一为 shortcut learning | 提供概念基础，但不是 CLE-HFL 的诊断方法 |
| Jabbour et al., *Deep Learning Applied to Chest X-Rays: Exploiting and Preventing Shortcuts* | MLHC 2020，PMLR 正式会议论文 | 医疗数据中患者属性、临床协议或设备相关因素可与诊断标签形成偏置并被模型利用 | 支撑 CLE 现实机制合理性：医院/设备/协议可成为类别捷径；不能替代本文在真实医院数据上的外部验证 |
| Vigneshwaran et al., *Evaluating Shortcut Utilization ... through Counterfactual Analysis* | MIDL 2026，PMLR 正式会议论文 | 通过反事实去除 site/sex 等属性，检查它们是否真正影响疾病预测 | 说明 counterfactual shortcut evaluation 已有先例；本文差异是同源图像跨operator、预固定binding方向、完整概率DSA及HFL机制归因 |

### 现实对应关系

本文的合成 class-corruption binding 可以被解释为下列现实机制的可控抽象：

```text
医院：疾病类别 ↔ 扫描仪/采集协议/文字标记/对比度
工厂：缺陷类别 ↔ 生产线相机/照明/压缩/运动模糊
车辆：目标类别 ↔ 天气/时段/镜头污渍/传感器型号
遥感：地物类别 ↔ 卫星/季节/地域/大气条件
```

但是“现实中可能存在这种机制”不等于“当前 CIFAR-C 受控实验已经证明真实部署有效”。论文仍应
将真实、复合、连续、未知环境下的适用性列为外部有效性边界。

### 主要来源

- [Shortcut Learning in Deep Neural Networks，Nature Machine Intelligence 2020](https://doi.org/10.1038/s42256-020-00257-z)
- [胸片中的shortcut，MLHC 2020](https://proceedings.mlr.press/v126/jabbour20a.html)
- [Counterfactual shortcut utilization，MIDL 2026](https://proceedings.mlr.press/v301/vigneshwaran26a.html)

---

## 8. 最近邻工作矩阵：谁最可能被审稿人拿来质疑

| 审稿人可能引用 | 可能的攻击 | 当前回答 | 投稿前仍需补强 |
|---|---|---|---|
| RAHFL / AugHFL | “异构FL的数据corruption早已解决” | 它们优化普通corruption robustness；本文诊断的是corruption沿客户端binding方向充当class code。Phase-A0/Stage-1与DSA提供直接证据 | 在Related Work和实验中同时报告普通效用与DSA，避免把二者混写 |
| FedPIN / PFL with Spurious Features | “联邦shortcut并非新问题” | 承认大问题已有先例；本文聚焦model-HFL、class-corruption定向绑定、paired operator估计量和local-first归因 | 精读其假设、定理和实验，逐项制作setting matrix |
| EIIL / GEORGE | “无需人工taxonomy也能发现环境” | 承认PEW是taxonomy-assisted；PEW追求可解释的corruption witness，而不是宣称通用latent environment discovery | 至少在讨论中给出taxonomy-free基线结果/限制，不能回避 |
| GroupDRO | “BER只是group reweighting” | 相同PEW分组下，GroupDRO追逐高损失组，BER平衡类别内支持结构；held-out map2 Formal中PEW+BER优于PEW+GroupDRO | 多training seed复现，确保不是seed0偶然性 |
| JTT / CVaR-DRO | “收益只是困难样本或tail-risk重加权” | map2四臂以相同base和训练协议比较，PEW+BER同时取得更低DSA和更高效用；但客户端级CVaR优势仍需诚实报告 | 多seed和第二数据集；不要把protocol-matched CVaR写成某篇论文的完全复现 |
| Counterfactual shortcut evaluation | “反事实诊断不是首创” | 承认反事实思想已有；本文的新对象是source-paired operator response及binding-direction estimand DSA | 说明干预有效性、同源语义保持与binding预注册假设 |

---

## 9. 本文在发表版图中的准确位置

### 9.1 不应声称的内容

- 不声称首次研究 shortcut、spurious correlation 或 counterfactual evaluation；
- 不声称首次在联邦学习中研究 spurious features；
- 不声称首次使用伪环境、分组鲁棒优化或样本重加权；
- 不声称 PEW 是 taxonomy-free；
- 不声称 AugMix/JSD、DCL、RAHFL 已被本文替代；
- 不声称当前合成 CIFAR-C 证据已经覆盖真实医院、车辆或工业部署。

### 9.2 可以主张的组合贡献

在现有证据和本次代表性检索下，最强的可防守定位是：

1. 定义模型异构联邦学习中的客户端特定 class-corruption directional shortcut；
2. 用同源 paired operator counterfactual 和 DSA 直接测量沿预注册 binding 方向的概率迁移；
3. 给出 DSA 零基准、混合单调性、JSD不充分性、统计推断及proxy不可识别性边界；
4. 用 HFL-vs-Local factorial 证明该机制主要 local-first，而非由通信单独制造；
5. 从类别内环境支持结构导出 taxonomy-assisted PEW+BER，并与 ERM、JTT、CVaR-DRO、
   PEW+GroupDRO 及多个 HFL 通信底座进行受控比较；
6. 同时报告该方法对 taxonomy、PEW误差、合成场景和外部有效性的限制。

### 9.3 一句话定位

> 本文不是一篇“发明新重加权器”的论文，而是一篇围绕 CLE-HFL 完成问题发现、定向识别、
> 形成机制归因、结构性干预和适用边界验证的完整研究。

---

## 10. 建议写入英文 Related Work 的四段结构

### 10.1 Model-Heterogeneous Federated Learning

从 FedDF 的公共数据蒸馏切入，再写 RHFL、AugHFL 和 RAHFL 的噪声/损坏鲁棒性。段末指出：
这些工作没有单独识别客户端特定 class-corruption association 被异构模型用作定向类别代码。

### 10.2 Corruption Robustness and Consistency Learning

介绍 CIFAR-C/ImageNet-C、AugMix/JSD 与 DCL。段末用形式化区别收束：同一 corruption 附近的
augmentation consistency 不蕴含跨 operator 的 binding invariance。

### 10.3 Spurious Correlation, Group Robustness, and Environment Inference

覆盖 GroupDRO、GEORGE、EIIL、JTT 和 CVaR/joint-DRO。主动承认伪组与重加权已有成熟先例，
然后突出 BER 的类别内支持对象、PEW 的 taxonomy-assisted 属性和 DSA 的 target-aligned 评价。

### 10.4 Federated Spurious Learning and Counterfactual Diagnosis

覆盖 PFL with Spurious Features、FedPIN、FedCD，以及医学领域 counterfactual shortcut work。
段末指出本文把问题收窄到模型异构HFL中的class-corruption directional binding，并补上过去工作
没有联合给出的 paired diagnosis、local-first attribution 与可行动缓解。

---

## 11. 给网页端 GPT 的使用说明

把本文与主交接文档一起上传/粘贴：

```text
docs/research/status/CLE_HFL_RELATED_WORK_PUBLICATION_LANDSCAPE_2026_09_14_ZH.md
deliverables/cle_hfl_full_paper_web_handoff_20260911/CLE_HFL_FULL_PAPER_WEB_HANDOFF_ZH.md
```

建议给网页端 GPT 的指令：

```text
请先核验“相关工作发表格局”中每篇论文的正式发表状态和原文，再基于主交接文档写英文初稿。
Related Work必须分成四组，并在每组末尾说明CLE-HFL的精确缺口。不要声称首次研究联邦
spurious correlation、首次伪环境发现、首次group reweighting或首次counterfactual shortcut
evaluation。把RAHFL写成TPAMI 2025正式论文及ICCV 2023 AugHFL的扩展，把FedCD标为当前
只核验到arXiv版本。任何未由正式来源支持的SOTA、首次性或现实部署结论标为[待核验]。
先给出：标题、摘要、Introduction、Related Work、Method和Experiment的英文骨架；等我确认后
再扩写全文。
```

网页端具备更长的论文阅读上下文时，应优先精读以下六篇：RAHFL、FedPIN、PFL with Spurious
Features、EIIL、GroupDRO、MIDL 2026 counterfactual shortcut analysis。它们分别代表最接近的
HFL底座、联邦shortcut方法、local-first相邻结论、taxonomy-free挑战、group reweighting挑战和
反事实诊断先例。

---

## 12. 参考文献核验清单

初稿形成前，网页端或人工应对每篇论文记录：

- 正式题名、作者、venue、年份、页码/卷号、DOI或proceedings链接；
- 研究设定是否模型异构、是否个性化FL、是否需要环境/组标签；
- corruption是普通性能退化，还是与类别形成spurious association；
- 评价是accuracy/worst-group/OOD，还是直接测量shortcut utilization；
- counterfactual是否同源、干预什么变量、估计量是否有方向；
- 方法是否依赖taxonomy、公共数据、真实环境标签或验证组标签；
- 理论保证的前提和结论，不能只依据摘要概括；
- 代码是否官方、本文基线是faithful reproduction还是protocol-matched adapter。

在上述逐篇核验完成前，论文中的“首次”“最接近”“优于现有方法”“现实可部署”等句子都应保留
`[待核验]`标记。
