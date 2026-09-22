# CLE-HFL 投稿版完整实验矩阵

> 2026-09-17 implementation update：M1、M2、M3、S1、S2及条件性JTT Formal的runner、
> 审计、分析与OpenI入口已完成；M3新输入包已生成并通过数据审计。执行参数、文件hash与结果
> 落盘位置见`CLE_HFL_SUBMISSION_EXECUTION_READY_2026_09_17_ZH.md`。这不代表任何新
> benchmark或Formal已获授权或已经运行。

Updated: 2026-09-15

## 0. 用途与纪律

本文档把当前论文所需实验分为“已完成证据、投稿前必须补、强烈建议补、条件性可选、明确不做”。
它是规划文件，不构成任何 OpenI、付费、Formal、多种子或长任务授权。所有新 Formal 必须先完成
协议冻结、输入审计、本地 smoke 和成本 benchmark，并由用户明确同意启动。

当前论文的四项贡献保持不变：

1. CLE-HFL 受控问题定义；
2. source-paired operator counterfactual、DSA 及识别边界；
3. HFL-vs-Local 的 local-first 机制归因；
4. taxonomy-assisted coarse PEW+BER 及受控验证。

不增加新 loss、通信模块、层次化 PEW 或 operator-level PEW。CDep 不进入当前方法。

## 1. 论文研究问题与证据映射

| RQ | 科学问题 | 当前状态 | 论文位置 |
|---|---|---|---|
| RQ1 | strong CLE 是否产生沿 binding 方向的 shortcut？ | 已完成 | 主文机制表、DSA 图 |
| RQ2 | DSA 是否识别 binding-specific response，而非普通损坏难度？ | 已完成 | 理论、null、缓存验证 |
| RQ3 | shortcut 主要由本地训练还是通信形成？ | seed 0 已完成；训练种子稳定性待补 | HFL-vs-Local factorial |
| RQ4 | PEW+BER 是否降低 CLE shortcut？ | map0/map1/map2 与第二通信底座已有证据 | 方法主表 |
| RQ5 | 收益是否只是一般 tail/group reweighting？ | map2 四臂 seed 0 已完成；训练种子稳定性待补 | ERM/CVaR/GroupDRO/BER 表 |
| RQ6 | coarse family 分组是否足够？ | Oracle family/operator/random 已完成 | 粒度边界消融 |
| RQ7 | 结论是否跨 private dataset？ | 未完成 | 外部有效性表 |
| RQ8 | taxonomy 覆盖不完整时如何退化？ | PEW operator-LOO 审计已有；端到端 stress 待补 | stress/limitation |

## 2. 已完成、可直接进入论文的实验

### E1. Phase-A0/A1a：CLE 形成与 paired DSA

- 对比：`gamma=0` 与 strong CLE `gamma=0.9`。
- 评价：同一 semantic source 跨 concrete corruption operators 的完整预测概率。
- 主指标：pooled/client-level DSA、operator-grid accuracy、shuffled-binding null。
- 结论：strong CLE directional shortcut 存在；普通 accuracy 不能替代 DSA。
- 边界：受控 CIFAR-10 CLE，不外推到真实部署。

### E2. DSA 理论与 CPU 缓存验证

- paired cancellation、exchangeable zero point、mixture linearity；
- operator-invariant shift 消除；
- shuffled-binding specificity 与 source-level bootstrap；
- `JSD=0` 但 `DSA>0` 构造；
- proxy non-identifiability 与 CVRS 数值反例。

这些是诊断识别证据，不是训练性能实验。

### E3. HFL-vs-Local seed-0 四臂归因

- 四臂：HFL/Local × `gamma in {0,0.9}`；
- 共同 local objective、初始化、partition、batch budget 和评价；
- 当前结论：pooled shortcut predominantly local-first；通信 add-on 较小但不为零。
- 尚缺：training seeds 1/2 的同协议复验。

### E4. PEW+BER 受控有效性

- 原 binding map：PEW+BER 显著降低 DSA；
- map1：只改变 client-specific binding map，DSA 下降 56.46%，4/4 clients 同向；
- held-out map2：40 轮 ERM/CVaR-DRO/PEW+GroupDRO/PEW+BER 四臂 Formal；
- FedDF-fidelity：第二通信底座出现 DSA suppression，但 frozen utility gate 失败。

允许结论是“受控 CLE shortcut suppression”；不允许结论是“通用无损插件”。

### E5. matched spurious baselines

held-out map2 的四臂共享 partition、初始化、batch trajectory、通信协议和训练预算：

- ERM：普通经验风险；
- CVaR-DRO：不使用 PEW 的 upper-tail loss 对照；
- PEW+GroupDRO：与 BER 使用相同 PEW 伪组，动态追逐高损失组；
- PEW+BER：按 class × pseudo-environment support 压缩多数支持优势。

seed 0 下 PEW+BER 取得最低 pooled DSA、最高 grid/Avg/Worst 和最低 CFG；但不能宣称逐客户端
统一胜过 CVaR。

### E6. Oracle 粒度边界

- Oracle family、Oracle operator、类别内 random grouping；
- Oracle 只用于离线边界审计，不能部署；
- operator 相对 family 的 DSA 增益仅 0.001449，未过 0.02 门槛；
- 结论：保留 coarse family PEW，不开发 operator PEW。

## 3. 投稿前必须补的实验

### M1. held-out map2 四臂 40 轮 training-seed stability（2026-09-22完成）

本实验已解决“最强方法结果主要来自 training seed 0”的问题，正式verdict为
`GO_MAP2_TRAINING_SEED_STABILITY`。完整结果：

```text
deliverables/cle_hfl_map2_multiseed_20260922/RESULT_SUMMARY_ZH.md
```

冻结对象：

- arms：ERM、CVaR-DRO、PEW+GroupDRO、PEW+BER；
- 新增 training seeds：1、2；与已完成 seed 0 合并报告；
- rounds：40；local batches：保持 seed-0 Formal 完全一致；
- dataset、partition、binding map2、PEW checkpoint、public batches、evaluation sources、通信协议不变；
- 禁用 AugMix/JSD/DCL/CDep；
- 不根据 seed 1 结果调整 seed 2 或任何超参数。

主要 estimand：

```text
Delta_DSA(ERM, BER)
Delta_DSA(GroupDRO, BER)
Delta_DSA(CVaR, BER)
```

报告：每 seed pooled/client DSA、operator-grid accuracy、last-10 Avg/Worst/WCCA/CFG、三 seed
mean ± std；source-bootstrap 与 seed variation 分开报告。

候选成功规则必须在运行前另行冻结，建议至少要求：

1. BER-vs-ERM 与 BER-vs-GroupDRO 的三 seed 平均 DSA 改善为正；
2. 不出现多数 seed 方向反转；
3. BER-vs-CVaR 按“pooled shortcut--utility trade-off”判断，不要求逐客户端 DSA 全胜；
4. utility 单独报告，不用事后放宽门槛换取 GO。

### M2. HFL-vs-Local 四臂 training-seed stability

2026-09-22执行更新：seeds 1/2任务因完整fit epoch导致预计50--70小时成本而由用户主动停止。
日志在8.5小时处仅完成seed 1首臂`h0_b`的round 11附近；没有形成四臂配对结果，不得作为证据，
也不构成NO-GO。M2暂缓。未来若恢复，必须给所有arms和seeds设置相同显式local-batch预算、
重新运行seed 0/1/2，并与历史full-fit seed-0分开报告。

目的：把 local-first 从 seed-0 机制发现提升为训练随机性下可复现结论。

协议：

- 重复现有 12-round 四臂 factorial；
- training seeds：1、2；
- HFL `gamma=0/0.9`，Local `gamma=0/0.9`；
- 固定 partition、CLE map、模型集合、初始协议、local objective、batch budget 与评价 sources；
- 不把通信 add-on 强行解释成样本级 causal mediation share。

每 seed 计算：

```text
delta_HFL   = DSA(HFL, gamma=.9)   - DSA(HFL, gamma=0)
delta_Local = DSA(Local, gamma=.9) - DSA(Local, gamma=0)
delta_Comm  = delta_HFL - delta_Local
```

报告三 seed 的三个 contrast、mean ± std，以及 `delta_Local/delta_HFL` 的描述性范围。不要只报告
平均后的“90%”。最终措辞应由方向稳定性决定：predominantly local-first、mixed，或 seed-specific。

### M3. 第二 private dataset

目的：回答 CLE、DSA 和 PEW+BER 是否只存在于当前 CIFAR-10 private task。

推荐优先候选：CIFAR-100 private，继续使用与 private task 标签无关的公共合成 corruption 数据
训练/复用 PEW。该选择必须先完成独立协议审计，不能直接复活旧准备包并当作当前 Formal。

最小闭环：

- architectures：与主实验相同，或在文中明确缩减并保持两臂一致；
- 构造 client-specific class--operator maps，预注册后冻结；
- 先验证 `gamma=0` 的 DSA 零基准和 `gamma=0.9` 的 directional shortcut；
- 主 arms：ERM、CVaR-DRO、PEW+BER；若算力允许再加入 PEW+GroupDRO；
- 至少 3 个 matched training seeds；
- 最终轮数由独立 benchmark 和学习曲线决定，不能拿 smoke/12-round screen 当论文 Formal；
- PEW 优先复用冻结 checkpoint，以检验跨 private semantic dataset 的 side-information transfer；
- 如果必须重训 PEW，应把“复用”和“重训”分成不同科学问题，不可混写。

主要指标：DSA、operator-grid accuracy、Avg/Worst/WCCA/CFG、逐客户端方向、seed mean ± std。

最低可写结论仅是“在第二个受控图像 private task 上复现”；仍不能声称真实医院/车辆部署有效。

## 4. 强烈建议补的实验

### S1. bounded taxonomy stress

目的不是证明开放世界鲁棒，而是量化 PEW taxonomy 覆盖不完整时方法如何退化。

推荐只冻结一个端到端压力协议，避免扩成新课题。两个候选中二选一：

1. held-out operator：PEW 公共训练不看某一 concrete operator，但该 operator 出现在 private CLE；
2. compound corruption：private CLE 使用预注册的两种 corruption 复合，PEW 仍只能输出现有 coarse family。

最小 arms：ERM 与 PEW+BER；必要时加入 CVaR-DRO。报告 PEW coverage/calibration、DSA 与 utility。
现有 PEW operator-LOO 只能证明 witness 层面的泛化，不能代替端到端 shortcut mitigation stress。

无论结果好坏都可进入论文：成功说明有限覆盖鲁棒性，失败则定量确定 applicability boundary。

### S2. 十臂protocol-matched HFL context table

目的：说明 CLE 不是只在一个自定义通信实现上出现，并把工作放回 HFL 文献坐标。

当前冻结的机制家族代表为：

- Local/ERM；
- FedMD protocol-matched adapter；
- FedProto protocol-matched adapter；
- FedTGP protocol-matched core adapter；
- FedDF-fidelity；
- KT-pFL-fidelity；
- FCCL protocol-matched adapter；
- RHFL protocol-matched adapter；
- AugHFL-fidelity；
- RAHFL anchor。

先审计算法忠实度，再在相同 CLE scenario 下报告 standalone baseline 的 DSA 与 utility。早期 12-round
适配器结果只作筛选，不能进入最终胜负表。该表用于“背景比较”，不承担 BER 因果归因；BER 的
归因仍由 matched ERM/CVaR/PEW+GroupDRO 四臂提供。

## 5. 条件性可选实验

### O1. JTT Formal

只有正文要声称“PEW+BER 正式胜过 JTT”时才补。否则保留 12-round screening-only 边界，并在
Related Work 做概念比较即可。

### O2. partition stability

在 M1--M3 完成后若算力仍允许，再增加 1--2 个 private partition seeds。它覆盖的是数据划分
不确定性，不能被 training seed 替代。

### O3. 更多通信底座的插件两臂

KT-pFL/FCCL 的独立 Base-vs-PEW+BER runner 已 smoke，但只有论文仍要使用“跨多底座插件”主张时
才值得 Formal。当前已有 AsymHFL 与 FedDF-fidelity 机制迁移证据，因此优先级低于第二数据集。

### O4. alpha/gamma severity sweep 与成本表

可作为附录压力分析，报告不同 non-IID/CLE 强度、训练时间、显存和 PEW 额外成本。不得从 sweep
中选择最有利点替代预注册主结果。

## 6. 明确不做

- 不恢复 CDep；
- 不开发 operator-level、hierarchical 或 multi-label PEW；
- 不通过调 BER 权重、cap 或门槛复活冻结路线；
- 不把 Oracle metadata 用于可部署训练；
- 不把 JSD、PEW accuracy 或 proxy TV 当成 DSA 的替代指标；
- 不要求每个 HFL baseline 都安装 PEW+BER；
- 不把 smoke、benchmark、12-round screen 当 Formal 证据；
- 不声称无条件真实环境去相关、准确率必升或开放世界泛化。

## 7. 推荐执行顺序与停止点

```text
P0  冻结 M1 协议、门槛和成本
 -> M1 map2 40-round seeds 1/2
 -> 若主要 DSA 方向不稳定，先重新评估论文主张，不补更多实验掩盖

P1  M2 local-first seeds 1/2
 -> 若 local-first 不稳定，把结论降级为 seed-0 controlled finding

P2  设计、审计并 smoke M3 第二 private dataset
 -> benchmark 后由用户决定是否启动多 seed Formal

P3  S1 bounded taxonomy stress
 -> 成功或失败都报告边界，不据此开发 PEW-v2

P4  根据目标会议页数和审稿意见决定 S2/O1/O2/O3/O4
```

每一步完成后立即更新主表、limitations、source-of-uncertainty 描述和当前 handoff；不要等全部跑完
再追溯配置。

## 8. 最小投稿版与理想投稿版

### 最小可信投稿版

- 当前全部已完成证据；
- M1 map2 方法多训练种子；
- M2 local-first 多训练种子；
- M3 第二 private dataset；
- 诚实保留 taxonomy-assisted 与 mixed utility 限制。

### 更强投稿版

- 最小可信投稿版；
- S1 bounded taxonomy stress；
- S2 精简 faithful HFL context table；
- 若需要明确比较，再选择性加入 JTT Formal 或 partition stability。

对当前论文而言，增加新方法的边际价值低于补齐训练随机性、第二数据集和 taxonomy 边界。
