# PEW + BER 论文主张审计

日期：2026-08-16
范围：CLE-HFL v2、calibrated hard PEW + hard BER、strict AsymHFL-val
纪律：本审计不修改训练代码、不启动实验，只核对实现、现有证据与公开原始论文。

## 1. 结论先行

```text
PEW+BER 作为当前固定 CLE 场景的强经验机制：      GO
PEW+BER 作为 taxonomy-assisted 诊断基线/上界锚点：GO
PEW 或 BER 作为单独的新数学对象：                NO-GO
PEW+BER 作为论文唯一核心方法贡献：               NO-GO
CLE-HFL 作为首次提出“类别-损坏纠缠”：            NO-GO
CLE-HFL 作为模型异构联邦中的受控扩展基准：        CONDITIONAL GO
```

这里的 `NO-GO` 不是否定现有准确率，也不是删除 PEW/BER。它表示：按照当前
定义，不能把它们单独包装成足以支撑方法论文的原创核心。现有结果应冻结为强
positive anchor，后续新方法必须解释如何保留这部分 weak-cell 收益。

最关键的外部事实是：NeurIPS 2020 的 Learning from Failure 已使用
Corrupted CIFAR-10，将 CIFAR-10 类别与 corruption type 高度相关，并在测试时
解除该相关性。2025 的 CCDB 与 ICLR 2026 的 FG-CCDB 又直接从
class-conditional distribution imbalance/balancing 角度研究同一类 spurious
correlation。因此，本项目不能声称首次发现“类别与损坏类型相关会产生弱组”。

本项目仍有一个明确但较窄的差异空间：把 class-corruption spurious correlation
放进四个不同 backbone、client-specific 映射、label skew、私有 fit/audit 隔离和
公共 logits 异构通信中，并做具体 operator 的 counterfactual seen/unseen 评价。
它可以构成受控 benchmark extension；若没有新的 FL-specific 方法对象，仅靠这个
交叉组合，论文方法贡献仍偏弱。

## 2. 冻结待审计主张

| ID | 待审计主张 | 判决 |
|---|---|---|
| C1 | 当前场景存在类别-损坏统计纠缠 | `SUPPORTED BY CONSTRUCTION` |
| C2 | 当前场景不是“每类永远只有一种损坏” | `SUPPORTED` |
| C3 | PEW 训练不读取私有 operator/family 真值 | `SUPPORTED, NARROW` |
| C4 | PEW 不依赖环境 taxonomy | `FALSE` |
| C5 | BER 是最坏组优化或新的 DRO | `FALSE` |
| C6 | PEW+BER 的增益依赖有意义的环境对应 | `SUPPORTED, FIXED SCENARIO` |
| C7 | PEW 可泛化到任意未知损坏 | `UNSUPPORTED` |
| C8 | PEW+BER 可处理复合基础损坏 | `UNSUPPORTED` |
| C9 | exact PEW+BER 已有三训练种子和 40 轮证据 | `FALSE` |
| C10 | CLE-HFL 首次提出类别-损坏纠缠 | `COLLISION` |
| C11 | PEW+BER 的两阶段结构是新的 | `COLLISION` |
| C12 | AsymHFL-val 是项目原创通信 | `FALSE` |

## 3. 代码事实

### 3.1 CLE-HFL v2 生成过程

对客户端 `k` 和类别 `c`，生成器先冻结一个具体 dominant seen operator
`d(k,c)`。训练样本的具体 operator 按下式采样：

\[
P(E=e\mid K=k,Y=c)
=
\frac{1-\gamma}{11}+\gamma\,\mathbf 1[e=d(k,c)].
\]

当前 `gamma=0.9`，所以 dominant operator 的理论概率为
`0.9 + 0.1/11 = 0.90909`，其他十个 seen operator 各约 `0.00909`。随后随机采样
severity，并对基础图像只施加一个具体 operator。

因此：

```text
真实代码语义：client/class-conditional corruption distribution
不是：        一个类别确定性等于一个 corruption
也是：        每张持久化基础输入只施加一个 base operator
```

测试集对每个选中的干净图像逐个施加具体 operator，覆盖 11 seen 和 4 unseen，
用于 class x operator counterfactual reporting。

### 3.2 Hard PEW 的真实监督

`PublicEnvironmentWitness` 使用 5,000 张公共 CIFAR-100 图片在线合成监督。其六个
输出为：

```text
clean / noise / blur / weather / digital / unknown
```

四个 corruption family 由人工 `CORRUPTION_GROUPS` 定义。`unknown` 公共训练样本
由两个不同 family 的操作依次合成。PEW 同时预测五级 severity，正式配置使用：

```text
5 epochs
environment CE + 0.25 * severity CE
public validation best epoch
auto unknown threshold
```

然后 PEW 给所有私有训练图像推断 hard environment ID。私有保存的真实 operator ID
只在 `_diagnostic_environment_ids` 中计算 family accuracy 和 oracle upper bound，
不参与 learned PEW 训练。

所以准确表述只能是：

> PEW 不读取私有环境真值，但使用人工 corruption taxonomy 生成公共环境监督。

不能表述成 taxonomy-free 或 environment-free。

### 3.3 Hard BER 的精确目标

设 PEW 给出的组为 `e`，客户端本地完整 fit 数据计数为 `n_{c,e}`。正式配置：

```text
support_gamma = 0.5
count_cap = 32
min_group_count = 2
```

对有效组定义：

\[
s_{c,e}=\min(n_{c,e},32)^{1/2},\qquad
q_{c,e}=\frac{s_{c,e}}{\sum_{e'}s_{c,e'}}.
\]

BER 等价于：

\[
\widehat L_{BER}
=
\frac{1}{|\mathcal C_{valid}|}
\sum_c\sum_{e:n_{c,e}\ge 2}
q_{c,e}
\left(\frac{1}{n_{c,e}}\sum_{i:y_i=c,\hat e_i=e}\ell_i\right).
\]

它统一平均类别，并在类内以截断后的平方根支持度加权环境平均风险。它不是：

- GroupDRO 的 `max_g R_g`；
- CVaR 的高损失尾部均值；
- IRM 的跨环境最优性约束；
- 新的可识别潜在环境目标。

它属于已知 `class x predicted-group` 重加权/平衡 ERM 家族。与普通完全均衡相比，
当前实现用 `sqrt(capped support)` 在方差与少数组强调之间折中。

### 3.4 与 AugMix/JSD/DCL 的关系

BER 的逐样本 CE 只使用基础 corruption 视图 `logits_clean`。JSD 使用基础视图和
两个 strong AugMix 视图；DCL 使用 clean/strong/weak 特征。最终本地目标为：

\[
L_{local}=L_{BER}+12L_{JSD}+L_{DCL}.
\]

因此 AugMix 的复合增强不能被当成“基础数据已经验证复合 corruption”的证据。

## 4. 内部证据边界

### 4.1 Exact PEW+BER 的 matched 消融

`cle_local_ablation_20260807` 固定 scenario seed 0、training seed 0、strict
fit/audit 和 AsymHFL-val。最后五轮：

| Arm | Avg | Worst | WCCA | CFG |
|---|---:|---:|---:|---:|
| RAHFL | 30.0853 | 25.0427 | 0.8500 | 30.4400 |
| PEW+BER | 34.6320 | 29.4280 | 7.2500 | 24.6400 |
| shuffled PEW+BER | 31.5437 | 26.0147 | 2.5500 | 37.3750 |
| oracle-family BER | 35.1200 | 30.7253 | 7.7000 | 20.6900 |

PEW+BER 相对 RAHFL：

```text
Avg +4.5467, Worst +4.3853, WCCA +6.4000, CFG -5.8000
```

有效归因：

- 打乱 PEW 对应后显著退化，说明收益不是任意随机分组正则；
- oracle family 更强，说明 environment estimation 仍是瓶颈；
- PEW 私有 family 诊断准确率只有 `62.21%`，但仍保留强收益；
- calibration 的自动阈值为 `0.0`，即没有额外的低置信度强制拒绝，不能把收益
  归因成复杂 unknown rejection。

这支持“固定场景下，有信息的粗环境分组重加权有效”，不支持 taxonomy 的普适性。

### 4.2 Operator-level LOO

Strict PEW-LOO 从公共 PEW 训练和验证中各留出一个 family 内具体 operator：

```text
impulse_noise / zoom_blur / fog / pixelate
```

相对 matched RAHFL 的最后五轮增量为：

```text
Avg +4.9027, Worst +6.2547, WCCA +4.6000, CFG -6.1100
```

可声称：同一已知 family 内，PEW 未见具体 operator 时仍保留收益。

不可声称：全新 family、任意真实退化或复合基础退化泛化。

### 4.3 多种子与 40 轮的归属边界

三训练种子和 40 轮 seed-0 的历史正式 positive package 是：

```text
calibrated PEW + BER + CDep + strict AsymHFL-val
```

后续 matched 消融已经判定 CDep 无稳定正因果贡献并移除。但这不允许把旧 package
的三 seed/40-round 证据逐字改写成 exact current `PEW+BER` 的证据。

当前 exact PEW+BER 最强直接证据仍是：

- scenario seed 0 / training seed 0 的 A1 matched 12-round；
- 同一 seed 的 Strict operator-LOO 12-round。

由于 seed-0 中 CDep full 的最后五轮还弱于 PEW+BER，这提供了支持性迹象，但不是
exact current method 的多种子或长期稳定证明。

### 4.4 尚未被支持的主张

```text
不同 scenario seed / class-operator map 泛化
不同 fit/audit partition 泛化
复合 base corruption
新 corruption family
真实世界 client metadata 可用性
exact PEW+BER 三训练种子和 40 轮
相对 spurious-correlation 专用基线的优势
```

## 5. 外部碰撞

### 5.1 场景碰撞：Corrupted CIFAR-10 已先存在

NeurIPS 2020 的 [Learning from Failure](https://papers.nips.cc/paper/2020/hash/eddc3427c5d77843c2253f1e799fe933-Abstract.html)
使用 Corrupted CIFAR-10，把 object label 与 corruption type 设置成强 spurious
correlation，并研究 bias-conflicting 样本。后续大量 group-robustness 工作沿用该
基准。

CLE-HFL v2 的差异是：

```text
centralized -> federated
single global class-corruption map -> client-specific maps
one model -> four heterogeneous backbones
ordinary training -> public-logit heterogeneous communication
10-way bias groups -> 11 seen / 4 unseen operator counterfactual audit
```

因此 CLE-HFL 可以称为 federated/model-heterogeneous extension，不能称为首次提出
class-corruption entanglement。

### 5.2 联邦场景碰撞

[FedDiverse](https://arxiv.org/abs/2504.11216) 已在 FL 中联合刻画 class
imbalance、attribute imbalance 和 spurious correlation，给出 global/client 指标，
构造七个联邦视觉数据集，并通过 client selection 处理互补分布。

[Personalized FL with Spurious Features](https://openreview.net/forum?id=N2wx9UVHkH)
也研究每客户端环境内标签与 spurious feature 相关的问题。

RAHFL 本身则已覆盖模型异构 FL + 数据 corruption + robust local learning +
asymmetric public-logit learning。CLE-HFL 的剩余场景差异是把 corruption 进一步设为
client/class-conditional，并强化 weak class x operator 评价；这是真差异，但属于
交叉扩展，不是空白领域。

### 5.3 PEW 两阶段结构碰撞

PEW+BER 的抽象结构是：

```text
训练 spurious/group attribute classifier
-> 给目标训练样本生成 pseudo group labels
-> 在 class x pseudo-group 上做 robust/balanced training
```

这条结构已有直接先例：

- [Spread Spurious Attribute, ICLR 2022](https://openreview.net/pdf?id=_F9xpOrqyX9)：
  用少量 spurious attribute 标注训练 attribute predictor，再用 pseudo attributes
  训练 worst-group robust model；
- [BARACK](https://arxiv.org/abs/2201.00072)：预测缺失 group labels，再把预测组送入
  robust objective；
- [GIC, ICML 2024](https://openreview.net/forum?id=KycvgOCBBR)：强调更准确的 group
  inference 才能改善下游 group robustness；
- [EIIL, ICML 2021](https://proceedings.mlr.press/v139/creager21a.html) 和
  [XRM, ICML 2024](https://openreview.net/forum?id=gPStP3FSY9)：在没有人工完整组标注时
  推断环境供下游 invariant/group-robust 学习。

PEW 的具体差异是：组分类器不从目标私有类别或小规模目标 group labels 学习，而从
异域公共图片上的人工 corruption generator 学习；随后在客户端本地推断。这个监督
来源有工程特点，但不足以使“两阶段伪组 + 分组风险”成为新范式。

### 5.4 BER 目标碰撞

[GroupDRO](https://arxiv.org/abs/1911.08731) 在已知组上优化最坏组风险；
[Simple data balancing](https://proceedings.mlr.press/v177/idrissi22a.html) 已表明简单
class/group 重采样或重加权可获得有竞争力的 worst-group accuracy。

BER 不是 GroupDRO，因此不能说公式完全相同；但它更接近后者的 group-balanced
reweighted ERM，而不是一个新的 robust optimization principle。

[JTT](https://proceedings.mlr.press/v139/liu21f.html)、
[AFR](https://proceedings.mlr.press/v202/qiu23c.html) 等方法还说明，在没有 group
标签时，通过失败/置信度重加权 minority samples 也是成熟路线。

### 5.5 “类条件分布平衡”表述碰撞

[CCDB](https://arxiv.org/abs/2504.17314) 已将 spurious correlation 明确表述为
class-conditional distribution mismatch，并在类内学习样本权重，使每类条件分布
靠近边际分布；[FG-CCDB, ICLR 2026](https://openreview.net/forum?id=NEFldJX4zb)
进一步用 bias exploration 和 confusion-cell-wise reweighting 做细粒度平衡。

因此不能把 “class-conditional degradation distribution balancing” 本身作为
PEW+BER 的新理论贡献。PEW+BER 反而比 CCDB/FG-CCDB 使用更强的预定义 taxonomy。

### 5.6 与 GroupDRO/CVaR/内部冻结方法的准确区分

| 方法 | 分组/尾部来源 | 优化对象 | 与 PEW+BER 的关系 |
|---|---|---|---|
| PEW+BER | 公共 taxonomy 训练的 PEW hard group | 类均匀、类内组加权平均风险 | 当前对象 |
| GroupDRO | 已知 group | 最大组风险 | 目标不同，但同属 group-aware risk |
| CVaR | loss quantile | 高损失尾均值 | 不需要环境组；BER 不是 CVaR |
| SSA/BARACK | 少量 group 标注预测 pseudo group | pseudo-group robust loss | 两阶段结构直接相邻 |
| EIIL/XRM/GIC | 训练行为/比较分布推断环境 | 下游 invariant/group robust | PEW 的 taxonomy-free 竞争者 |
| CCDB/FG-CCDB | 类条件特征/偏差分布 | 分布匹配与样本重加权 | 问题表述和类条件平衡直接相邻 |
| PIE/MPIE | 公共合成干预学习潜在环境 | 连续/序数代理 | 项目冻结负结果 |
| C3R/CRSR/LCC | 本地反事实、谱或梯度几何 | taxonomy-free 本地风险代理 | 项目冻结负结果 |
| CCAD/IRD | 公共多视图一致/残差蒸馏 | 通信正则 | 项目冻结负结果；非 PEW 新颖性来源 |

“不同于 GroupDRO/CVaR”只说明 BER 不是它们的逐字复制，不能自动推出 BER 新颖。

## 6. 论文主张许可表

### 6.1 当前可以写

1. CLE-HFL v2 是 centralized class-corruption spurious-correlation benchmark 在
   model-heterogeneous FL 下的受控扩展；
2. learned PEW 不读取私有 operator metadata，而用公共合成 corruption taxonomy
   进行环境伪标注；
3. 在固定 `seed0_split0` 场景，PEW+BER 相对 matched RAHFL 显著改善 Avg、Worst、
   WCCA 和 CFG；
4. shuffled control 说明正确环境对应很重要；oracle gap 说明 group inference 仍是
   瓶颈；
5. operator-level family-internal LOO 后收益仍存在。

### 6.2 当前禁止写

1. 首次提出类别-损坏纠缠或 corruption spurious correlation；
2. PEW taxonomy-free、无环境监督或适用于任意未知 corruption；
3. BER 是新的 DRO/最坏组风险；
4. PEW+BER 已证明复合 corruption 或真实世界泛化；
5. exact current PEW+BER 已完成三 seed/40-round；
6. AsymHFL 是本项目原创通信；
7. 仅凭超过 RAHFL 就证明方法新颖。

## 7. 最强审稿攻击与当前回答

| 攻击 | 当前回答 | 风险 |
|---|---|---|
| 场景早已有 Corrupted CIFAR-10 | 承认；只主张 model-heterogeneous FL extension | 高 |
| 为什么需要五/六类 taxonomy | 公共 generator 提供弱监督，但部署来源未证 | 高 |
| PEW+BER 不就是预测组后重加权 | 结构上确实接近 SSA/BARACK + balancing | 致命于方法首创 |
| BER 不就是简单 group balancing | 是带 support shrinkage 的 group-weighted ERM | 致命于 BER 首创 |
| 为什么不比 LfF/JTT/SSA/EIIL/XRM/CCDB | 当前 paper baseline 包未覆盖 | 高 |
| 只在一张 client/class map 上有效 | exact method 尚无跨 scenario 证据 | 高 |
| multi-seed/40-round 属于 exact 方法吗 | 不是，历史 package 含 CDep | 中高 |
| 新 family/复合 corruption 呢 | 当前不支持 | 高（若作普适主张） |
| 联邦部分是否原创 | RAHFL/AsymHFL 是采用的骨架 | 高 |

## 8. 最终判决与优先级

### P0：立即冻结的科学地位

```text
hard PEW + hard BER
= strong taxonomy-assisted diagnostic baseline
= empirical target to preserve
!= final paper-level novel method
```

不要删除实现或否认正结果，也不要继续靠换名、软标签、阈值或权重把它包装成新对象。

### P1：原场景的可继续方式

CLE-HFL 可以继续作为研究 testbed，但必须改成以下诚实定位：

> A model-heterogeneous federated extension of class-corruption spurious
> correlation, with client-specific mappings and counterfactual operator-cell
> evaluation.

它不应被宣传为首次提出损坏纠缠。若最终方法仍只是 PEW+BER，场景贡献与方法贡献
都不足以形成强方法论文。

### P2：下一方法进入实现前的硬条件

新候选必须至少满足一个当前工作没有的 FL-specific 数学增量：

1. 不使用固定 corruption taxonomy，却有明确可识别 side information；或
2. 利用模型异构/客户端交互产生中央 group-robustness 方法没有的对象；或
3. 定义新的通信风险并证明它不是 AsymHFL/CCAD/IRD/普通 teacher weighting；
4. 能解释为何保留 PEW+BER 的 weak-cell 收益，而不是只优化 Avg。

同时必须在纸面上逐项区别于 LfF、SSA/BARACK、EIIL/XRM/GIC、CCDB/FG-CCDB、
GroupDRO/CVaR 和项目冻结负结果，之后才值得实现。

### P3：暂缓的算力投入

在新核心贡献确定前，不建议优先补 exact PEW+BER 多 seed、40 轮或大规模
cross-scenario。那些实验只能加固一个当前判定为“强基线”的对象，不能解决论文
首要的新颖性缺口。后续若选择 benchmark/empirical-paper 路线，再预注册这些证据。

## 9. 审计后项目状态

```text
场景：      有效受控 benchmark extension；非首次问题
PEW：       可部署式私有伪标注路径，但依赖公共人工 taxonomy
BER：       强有效的 class x predicted-group reweighted ERM；非新风险原则
通信：      RAHFL/AsymHFL 骨架；非原创
实证：      固定场景强；exact final method 的跨 seed/scenario/compound 证据不足
论文价值：  baseline/diagnostic 高，唯一核心方法贡献不足
总判决：    CORE-METHOD NO-GO / BASELINE GO / BENCHMARK CONDITIONAL GO
```
