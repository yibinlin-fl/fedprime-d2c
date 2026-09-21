# CLE-HFL相关论文数据集与实验图审计

Updated: 2026-09-22

## 1. 审计目标

本文档逐篇核对以下五份用户提供的PDF，重点不是复述方法，而是回答三个投稿问题：

1. 相邻工作实际使用了多少private/public/evaluation datasets；
2. 它们用哪些表和图建立实验说服力；
3. CLE-HFL V0.4还缺哪些实验图，每张图依赖什么已完成或待补实验。

来源：

```text
D:\googleDownload\01_联邦学习研究\抗数据损坏与频域论文\RAHFL增强流派\2503.09206v1.pdf
D:\googleDownload\01_联邦学习研究\抗数据损坏与频域论文\RAHFL增强流派\AugHFL.pdf
D:\googleDownload\01_联邦学习研究\抗数据损坏与频域论文\RAHFL增强流派\Fang_Robust_Federated_Learning_With_Noisy_and_Heterogeneous_Clients_CVPR_2022_paper.pdf
D:\googleDownload\01_联邦学习研究\抗数据损坏与频域论文\RAHFL增强流派\FedERL_Federated Efficient and Robust Learning for Common Corruptions.pdf
D:\googleDownload\01_联邦学习研究\抗数据损坏与频域论文\最新的各个论文\DART_A Server-side Plug-in for Resource-efficient Robust Federated Learning.pdf
```

页数分别为15、11、10、12和18页。FedERL是arXiv:2508.17381v1，DART PDF是同一arXiv条目的
2026年v2扩展/改名版本，因此二者不是两篇完全独立的证据来源，不能在相关工作中当作两个独立
方法重复计数。

## 2. 五篇论文的数据集与实验呈现

### 2.1 RHFL，CVPR 2022

任务是模型异构FL中的label noise，而非common corruption。

```text
private task:       CIFAR-10
public dataset:     CIFAR-100 subset
noise types:        symmetric flip / pair flip
noise rates:        0.1 / 0.2，另含无噪声对照
heterogeneous:      ResNet10 / ResNet12 / ShuffleNet / MobileNetV2
homogeneous:        four ResNet12 clients
communication:      40 collaborative epochs
```

论文实验广度主要来自噪声类型、噪声率、异构/同构设置和逐客户端结果，并不是来自很多数据集。
Figure 1是场景示意，Figure 2是完整方法框架，Figure 3是流程图；最终结果主要放在Table 1--6，
没有大规模结果曲线。对CLE-HFL的启示是：一个private task加一个public dataset也可以支撑正式
论文，但必须用受控因素、消融和逐客户端结果把证据做完整。

### 2.2 AugHFL，ICCV 2023

```text
private task:       CIFAR-10-C sampled into clients
public dataset:     CIFAR-100 subset
evaluation:         clean CIFAR-10 / randomly corrupted CIFAR-10-C
corruption rates:   0 / 0.5 / 1.0
corruption type and severity: sampled randomly
models:             ResNet10 / ResNet12 / ShuffleNet / MobileNetV2
training budget:    40 local pre-training epochs + 40 collaborative rounds
```

Figure 1是HFL corrupted-client场景；Figure 2用clean/corrupted预测差异做动机；Figure 3是方法
总图。实验结果几乎全部由Table 1--3组件消融和Table 4--6 SOTA比较承担，没有单独结果曲线。
所以AugHFL“实验够”的关键不是图多，而是一个数据生成轴覆盖三档corruption rate，并同时报告
clean和corrupted evaluation、四个客户端和多个HFL baseline。

### 2.3 RAHFL，TPAMI 2025版本

```text
private task:       CIFAR-10-C sampled into clients
public dataset:     CIFAR-100 subset
evaluation:         clean CIFAR-10 / randomly corrupted CIFAR-10-C
corruption rates:   0 / 0.5 / 1.0
models:             ResNet10 / ResNet12 / ShuffleNet / MobileNetV2
homogeneous extra:  four ResNet12 clients
training budget:    40 local pre-training epochs + 40 collaborative rounds
```

核心数据集仍只有一个private task和一个public dataset。相比AugHFL，TPAMI扩展主要增加DCL、
AsymHFL、更多组件消融、同构FL结果和训练曲线，而不是增加多个private datasets。

```text
Fig.1  场景示意
Fig.2  clean vs corrupted预测动机
Fig.3  RAHFL完整方法框架
Fig.4  40轮clean/corrupted accuracy曲线（同构模型）
Table 1--4  三档corruption rate及DCL消融
Table 5--7  三档corruption rate的SOTA比较
```

正文真正的结果图只有Figure 4，其余证据主要是表格。该论文说明：结果图不需要很多，但至少应有
一张让读者直接看见训练过程或主要趋势的图。

### 2.4 FedERL，arXiv v1

FedERL开始把实验论证从单纯accuracy表扩展到resource-efficiency。

```text
main private task:     CIFAR-10
main public dataset:   CIFAR-100
alternative public:    Tiny ImageNet
evaluation:             CIFAR-10 / CIFAR-10-C
appendix private task:  CIFAR-100，public改为CIFAR-10
models:                 ResNet-18 / MobileNet / VGG-16
```

主要图承担明确问题：

```text
Fig.1   clean vs robust训练的时间/能耗代价
Fig.2   CleanFL / RobustFL / FedERL系统差异
Fig.3   DART方法流程
Fig.4-5 ResNet-18的accuracy-time-energy曲线
Fig.6   DART更新频率
Fig.7-10 MobileNet/VGG-16跨架构曲线
Fig.11-12 CIFAR-100第二client dataset曲线
```

FedERL的图看起来更丰富，是因为论文核心claim本身包含time/energy/resource efficiency，需要曲线
展示Pareto关系。它不是所有CLE-HFL论文都必须照搬的图量标准。

### 2.5 DART，arXiv v2

DART明显扩大了v1的实验覆盖：

```text
main private task:      CIFAR-10
second private task:    CIFAR-100（附录）
server/public datasets: CIFAR-100 / Tiny ImageNet / BigGAN synthetic data
robust benchmarks:      CIFAR-10-C / CIFAR-10-bar-C / CIFAR-10-P
second robust task:     CIFAR-100-C
models:                 ResNet-18 / MobileNet / VGG-16
FL methods:             FedAvg / FedProx / FedNova / FedDyn及其+DART
robust controls:        FedAugMix等
heterogeneity:          multiple Dirichlet alpha values
```

```text
Fig.1  Clean FL / Robust FL / DART-enhanced FL系统差异
Fig.2  DART方法图
Fig.3  8-panel robustness-time-energy Pareto图（正文核心结果图）
Fig.4-7 多架构及第二client dataset的time/energy曲线
Fig.8  DART更新次数
Fig.9  alpha与augmentation width敏感性
Fig.10 多GPU/并行成本
```

DART最值得借鉴的是Figure 3：同一颜色/点形体系、多个受控panel、明确Pareto边界，一张图回答
主claim。它不依赖装饰性图标，而是把真实结果可视化。CLE-HFL不需要复制time/energy坐标，但应
学习“每张结果图只回答一个可检验问题”和“小多图共享视觉编码”的方式。

## 3. 数据集数量的客观判断

| 论文 | private task | public/server data | 额外鲁棒评价 | 主要扩展轴 |
|---|---|---|---|---|
| RHFL | CIFAR-10 | CIFAR-100 | 两类label noise | noise rate/type、异构/同构 |
| AugHFL | CIFAR-10-C | CIFAR-100 | clean与random corruption | corruption rate、逐客户端 |
| RAHFL | CIFAR-10-C | CIFAR-100 | clean与random corruption | corruption rate、消融、同构/异构 |
| FedERL v1 | CIFAR-10；附录CIFAR-100 | CIFAR-100/Tiny ImageNet | CIFAR-C | 时间、能耗、架构、public data |
| DART v2 | CIFAR-10；附录CIFAR-100 | CIFAR-100/Tiny ImageNet/合成数据 | C、bar-C、P、CIFAR-100-C | 多FL算法、异质性、Pareto、成本 |

结论：RAHFL/AugHFL的private dataset确实不多，主线只有CIFAR-10/CIFAR-10-C；但2026年较新
工作已经提高实验覆盖预期。CLE-HFL若只用CIFAR-10受控任务，仍可能形成完整论文，但容易受到
“仅为一个synthetic benchmark现象”的攻击。因此当前冻结的M3 CIFAR-100第二private task不是
为了机械追求数据集数量，而是为了覆盖最关键的外部有效性缺口。完成M3后，CLE-HFL在private
task数量上已不弱于RAHFL主实验，并接近FedERL/DART的基本双任务结构；没有必要为了凑数量再
添加第三个图像数据集。

## 4. 当前V0.4实验图缺口

V0.4目前只有两张双栏图：

```text
Figure 1  CLE-HFL evidence chain and information boundary
Figure 2  paired DSA and proxy non-identifiability boundary
```

两张都是概念/理论图。主结果全部放在表格中，因此读者无法通过任何图片快速看到：

- local-first的效应大小；
- PEW+BER相对ERM/CVaR/GroupDRO的DSA--utility折中；
- 结果跨binding map、training seed、dataset和communication base的稳定性。

这比“图的绝对数量少”更重要。推荐最终正文保持4张核心图，而不是堆很多装饰图。

## 5. 建议的正文核心图

### Figure 1：CLE-HFL setting与信息边界（重画现有Figure 1）

唯一问题：不同客户端的class--operator binding为什么会让corruption成为directional class cue？

三段式：client-specific bindings -> heterogeneous local models -> public-response communication。只展示
两个类别、两个operator和两个客户端的冲突binding，不再把全部论文证据链塞进一张图。图底部用
细线区分train-time可见的fit label/PEW proxy和sealed evaluation-only operator/binding信息。

数据依赖：无，新图可以现在完成。

### Figure 2：Paired DSA estimand与null（改造现有Figure 2）

唯一问题：DSA测的是什么，为什么普通accuracy/JSD不能替代？

```text
(a) same source across operators
(b) bound-class probability-mass contrast
(c) true binding DSA vs shuffled-binding null
```

将proxy non-identifiability theorem保留为小型理论panel或移到附录，避免Figure 2同时解释两个过于
抽象的故事。`true DSA=0.119644`、shuffled p95和permutation p可使用已有缓存。

数据依赖：已有，当前即可绘制。

### Figure 3：Local-first formation effect plot

唯一问题：shortcut在联邦管线的哪个阶段形成？

推荐两个panel：

```text
(a) HFL/Local x gamma=0/0.9 的pooled DSA点图或区间图
(b) delta_Local、delta_HFL、delta_Comm的matched effect plot
```

禁止用90.05%饼图，因为该比例不是严格causal mediation share。M2 seeds 1/2完成后，按seed显示
点并报告mean/std；完成前只能做seed-0临时图，不能作为投稿终稿。

数据依赖：M2未完成。

### Figure 4：Held-out map2 shortcut--utility trade-off

唯一问题：PEW+BER是否只是在牺牲任务性能换取更低DSA，或只是普通group/tail reweighting？

推荐三panel小多图：

```text
(a) x=DSA, y=operator-grid accuracy，标记ERM/CVaR/PEW+GroupDRO/PEW+BER
(b) 四方法逐客户端DSA的paired dot/slope plot
(c) last-10 Avg/Worst与DSA的紧凑trade-off plot
```

颜色固定：ERM灰、CVaR蓝、PEW+GroupDRO橙、PEW+BER深红；同时使用点形避免仅靠颜色。M1
seeds 1/2完成后显示seed-level点和均值区间，不能用source bootstrap冒充training-seed CI。

数据依赖：M1未完成；seed-0数据可用于布局原型，不能作为最终稳定性图。

## 6. 建议的第五张图与附录图

正文是否加入第五张图由页数和结果决定：

### Candidate Figure 5：跨场景DSA reduction forest plot

每一行是一个matched setting：original map、map1、map2 seeds、FedDF-fidelity、CIFAR-100 seeds。
横轴只画`DSA(Base)-DSA(BER)`及相应不确定性。不同protocol的绝对accuracy不得混成全局排行榜。

数据依赖：M1和M3完成后才能形成完整主图。若页数有限，放附录。

### Appendix A：taxonomy stress

PEW排除motion_blur后的held-out witness accuracy/unknown rate、整体DSA、motion-blur-specific DSA
和utility。数据依赖S1。

### Appendix B：per-client/operator DSA heatmap

展示Base与BER的operator-level DSA以及差值，回答改善是否只来自少数operator。已有map2 seed-0
缓存可先画；最终版应加入多seed汇总或明确seed-0。

### Appendix C：BER support compression

以class x pseudo-environment有效质量的before/after heatmap展示BER如何压缩支持优势；使用已有CPU
审计结果，不把真实family用于训练。

### Appendix D：PEW witness audit

family confusion matrix、unknown detection和环境支持分布。该图用于诚实展示taxonomy-assisted
proxy的错误结构，不能用PEW accuracy代替DSA。

### Appendix E：training curves与成本

只在正式长轮数结果到齐后画round--Avg/DSA曲线，并报告PEW训练、annotation和BER计算开销。
不需要复制DART的大量time/energy图，因为resource efficiency不是本文主claim。

## 7. 图与待补实验的依赖关系

| 图 | 当前能否做终稿 | 依赖 |
|---|---|---|
| Fig.1 CLE setting | 可以 | 无新实验 |
| Fig.2 paired DSA/null | 可以 | 已有缓存与理论验证 |
| Fig.3 local-first | 不可以 | M2 seeds 1/2 |
| Fig.4 map2 trade-off | 不可以 | M1 seeds 1/2 |
| Fig.5 cross-setting forest | 不可以 | M1 + M3，视页数 |
| App. taxonomy stress | 不可以 | S1 |
| App. operator heatmap | seed-0可做，终稿待定 | M1更完整 |
| App. BER compression | 可以 | 已有CPU审计 |
| App. PEW audit | 可以 | 已有PEW audit/annotation |
| App. HFL context | 不可以 | S2 |

因此实验图不足不能通过“现在先画更多概念图”解决。最有效顺序是M1 -> M2 -> M3 -> S1 -> S2，
并让每个实验直接解锁一张预注册结果图。

## 8. 最终建议的图表配额

对当前约11页venue-neutral双栏稿，建议目标为：

```text
正文：4张图 + 3至4张核心表
附录：3至5张诊断/边界图 + 完整逐客户端/逐operator表
```

正文四图分别承担setting、estimand、formation和mitigation，不重复。若目标会议正文页数较紧，
Figure 5及PEW/taxonomy图进入supplement。RAHFL与AugHFL证明“结果表多、结果图少”本身并不致命；
但CLE-HFL至少需要Figure 3和Figure 4两张结果图，否则V0.4视觉上只展示概念，没有展示论文最强
的机制和方法证据。

## 9. 视觉风格边界

- 采用白底、无阴影、无渐变、无3D、无卡通机器人；
- 所有结果图从冻结CSV/JSON/预测缓存生成，不手工抄数；
- 同一方法跨所有图保持相同颜色、点形和顺序；
- 每个panel只回答一个问题，共享坐标和图例；
- 不把不同protocol的绝对性能直接连线或排名；
- CI必须标明是source-bootstrap还是training-seed variation；
- 图的SVG/PDF及生成脚本同时归档，禁止只保留不可编辑位图。

这些要求比“让生成式AI自由画得漂亮”更接近DART结果图的优点：信息密度高、比较关系清楚、视觉
编码稳定，并且每个点都能回到实验记录。
