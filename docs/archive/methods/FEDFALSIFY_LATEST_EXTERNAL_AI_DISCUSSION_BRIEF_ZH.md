# CLE-HFL v2 与 FedFalsify 最新实验：外部 AI 讨论材料

更新时间：2026-07-24

## 一、研究目标

我的目标是在视觉联邦学习中同时研究：

```text
模型异构
+ label-skew 数据异构
+ 数据损坏
+ corruption-label shortcut
```

目标投稿为一篇 CCF B 类会议小论文。当前不满足于只对 RAHFL 的一个损失项
做微小修改，希望形成一个有清晰问题定义、通信机制和实验闭环的工作。

## 二、新问题与 CLE-HFL v2 协议

我们提出的问题是 corruption-label entanglement：

> 在同一客户端内部，不同类别与不同损坏算子形成统计关联，模型可能把损坏
> 模式当作预测类别的捷径；联邦通信还可能把这种 shortcut 传播给其他客户端。

CLE-HFL v2 不把 `noise/blur/weather/digital` 四个大类提供给训练算法，而是
使用 15 个具体 CIFAR-C 风格算子：

```text
11 个 seen operators：可出现在客户端训练集
4 个 unseen operators：不出现在任何客户端训练集，只用于测试
```

对客户端 `k` 和类别 `c`，随机指定一个主导训练算子 `phi_k(c)`：

```text
P_k(o | y=c)
= gamma * 1[o=phi_k(c)]
+ (1-gamma) / |O_seen|
```

本次正式实验：

```text
clients = 4
models = ResNet10 / ResNet12 / ShuffleNet / MobileNetV2
alpha = 0.5
gamma = 0.9
seed = 0
rounds = 12
samples/client = 10,000
realized dominant-operator rate = 0.91015
```

训练代码不能看到 operator ID、operator name、family、seen/unseen 或 severity。
这些信息只用于离线评价。

指标：

- Avg：所有客户端平均准确率，越高越好。
- Worst：最差客户端准确率，越高越好。
- WCCA：最差 class-operator 单元格准确率，越高越好。
- CFG：同类样本在不同损坏算子下的最大准确率差距，越低越好。

## 三、当前方法 FedFalsify v0.3

### 1. 本地基座

直接使用 RAHFL 的强本地训练思路：

```text
AugMix + JSD + DCL
```

### 2. 严格数据隔离

每个客户端的本地训练集固定拆分为：

```text
D_fit   ≈ 85%，用于梯度训练
D_audit ≈ 15%，用于通信证据
D_test  独立最终测试，绝不参与通信或训练
```

### 3. 通信证据

对 receiver `i`、类别 `c`、source `j`：

1. FRA 使用 `D_audit` 上的配对正确性差值估计 source 是否优于 receiver。
2. 计算 paired advantage、标准误和单侧上置信界：

```text
UCB = paired_advantage + kappa * SE
eligible iff UCB >= 0
```

3. 对通过非劣否决的 source，使用一次虚拟 classifier-head 更新计算 head-TAU，
   选择 TAU 最高的 Top-1 source。
4. 使用 Conservative Margin Transfer 对接收客户端的分类头做保守迁移。

核心假设是：先用统计证据剔除明确更差的 source，再用 head 梯度兼容性选择
最适合 receiver 的 source。

## 四、CLE-HFL v2 最新结果

三组实验均完整运行 12 轮：

| 方法 | Final Avg | Worst | WCCA | CFG |
|---|---:|---:|---:|---:|
| RAHFL | 33.8267 | 27.0400 | 0.250 | 30.050 |
| Strict fit-only control | 30.7550 | 24.9800 | 0.250 | 30.225 |
| FedFalsify v0.3 | 31.0733 | 24.5733 | 0.500 | 31.825 |

FedFalsify 相对严格 control：

```text
final:
  Avg   +0.3183
  Worst -0.4067
  WCCA  +0.250
  CFG   +1.600  (退化)

last-five:
  Avg   +0.1180
  Worst -0.4373
  WCCA  +0.850
  CFG   +2.185  (退化)
```

通信不是完全无效。warmup 后第 3 至 11 轮，FedFalsify 相对 control 的 Avg
差值为：

```text
+0.755, +1.135, +0.485, +1.368, +0.720,
-0.453, -0.088, +0.093, +0.318
```

它前期有效，后期收益衰减。

通信统计：

```text
平均激活路由             19.56 / 40
平均覆盖率               48.89%
selected head-TAU        0.8825
每轮平均否决劣质候选     62.78
selected advantage <= 0  30.68%
selected FRA strength=0  45.45%
相邻轮 source 切换率     19.60%
```

类别-算子层面存在明显收益不均：

```text
class 0 / shot_noise      +11.25
class 0 / gaussian_noise  +11.00
class 0 / fog             +10.50

class 3 / contrast         -9.00
class 1 / shot_noise       -7.50
class 1 / zoom_blur        -6.50
```

所以 FedFalsify 只是把收益集中到少数单元格，同时伤害另一些单元格，最终
Avg 微升而 Worst/CFG 退化。

### RAHFL 数字的限制

当前 RAHFL 数字不能视为完全公平的正式结论：

```text
RAHFL：使用全部 10,000 个本地训练样本，并使用 final-test accuracy 路由
FedFalsify：只在约 8,500 个 D_fit 样本训练，D_audit 用于通信，不看测试标签
```

因此 `RAHFL 33.83 vs FedFalsify 31.07` 只能作为诊断参考。严格可信的机制
结论是：FedFalsify 相对自己的 fit-only control 没有通过 Worst/CFG 门槛。

目前不准备继续消耗算力补跑 RAHFL-val，希望先从理论层面重新审查方法。

## 五、容易记混的历史结果

历史上确实出现过“提升约 5 点”，但不是相对 RAHFL：

| 方法 | 对照 | Avg 增益 | Worst 增益 | 备注 |
|---|---|---:|---:|---|
| Oracle BER+CDep | local control | +4.04 | +5.41 | 使用真实环境信息，是机制上界 |
| Learned PEW | local control | +2.79 | +5.31 | 依赖固定环境类别 |
| SARA + AsymHFL | RAHFL | +1.42 | +1.87 | 旧协议、40 轮 |
| NIR-DCL + AsymHFL | RAHFL | +0.95 | +1.51 | 旧协议、40 轮 |

Oracle/PEW 的结果说明 corruption 条件信息可能有价值，但固定损坏 taxonomy
容易受到审稿质疑，不能直接作为最终主方法。

## 六、当前已经明确的失败机制

1. **TAU 不是 expertise。**  
   一次 head 更新的梯度兼容性高，不代表 source 拥有稳定、可迁移的类别知识。

2. **非劣不等于有益。**  
   `UCB >= 0` 只能说明尚不能证明 source 更差，无法证明它会提升 receiver。

3. **类别级路由仍然太粗。**  
   同一个 source 对某类别的不同损坏算子可能既有大幅正迁移，也有严重负迁移。

4. **audit 证据覆盖有限。**  
   严重 label skew 下，receiver/source 对部分类别缺乏足够审计样本。

5. **强本地基座已经很强。**  
   AugMix/JSD/DCL 对 seen 和 unseen operator 都具有较强泛化，通信模块必须提供
   超越普通鲁棒训练的新增信息，而不能只是重新加权已有预测。

## 七、希望外部 AI 回答的问题

请不要直接给出更多“稳定性加权、置信度加权、EMA、阈值门控”的排列组合。
请以审稿人和方法设计者的双重视角回答：

1. CLE-HFL v2 的问题设定是否值得保留？它是否足以构成 CCF B 论文的问题贡献？
2. 当前 FedFalsify 的失败是实现细节问题，还是通信信息本身不足？
3. 在完全不知道 operator ID/family/severity、不能使用最终测试标签的约束下，
   联邦通信到底应该传什么，才能抑制 corruption-label shortcut？
4. 是否应该保留 `D_fit/D_audit` 的接收端反事实审计思想，但彻底替换
   TAU Top-1 和 CMT？
5. 请提出一个最多包含两个核心创新模块的新框架。每个模块必须说明：
   - 输入与输出；
   - 客户端和服务器分别做什么；
   - 具体公式；
   - 为什么对模型异构、label skew、corruption-label shortcut 有效；
   - 为什么不会重演 FedFalsify 的局部负迁移；
   - 隐私、计算和通信开销；
   - 与 RAHFL/FedProto/FedDF/普通 robust FL 的本质区别。
6. 必须给出最小验证实验和明确 Go/No-Go 标准，不能一开始就建议多 seed、
   40 轮和大量网格调参。
7. 如果你认为不存在仅靠现有客户端预测/梯度就可靠补充缺失知识的方法，请明确
   说明，并提出更合理的问题收缩方案，而不是为了完整而继续堆模块。

## 八、可直接复制给 Web 端 GPT 的提示词

```text
你现在是一名严格的联邦学习与鲁棒视觉研究者。请阅读我下面提供的完整研究
事实，不要先迎合我，也不要直接排列组合已有模块。

我的目标是做一篇 CCF B 类视觉联邦学习小论文，场景同时包含模型异构、
label-skew、数据损坏，以及 corruption-label shortcut。我们设计了 CLE-HFL v2：
4 个异构客户端模型，Dirichlet alpha=0.5；对每个 client-class 随机绑定一个
主导具体损坏算子，gamma=0.9；11 个 seen operators 用于训练，4 个 unseen
operators 完全不进入训练。训练算法看不到 operator ID、family、severity 或
seen/unseen 信息。

当前方法 FedFalsify v0.3 使用 RAHFL 的 AugMix+JSD+DCL 本地基座，把每个
客户端训练数据拆为 85% D_fit 和 15% D_audit。通信时，先用 D_audit 上的
paired correctness advantage 及 UCB 剔除统计上明确更差的 source，再用一次
虚拟 classifier-head 更新得到 head-TAU，从候选 source 中选 Top-1，最后执行
Conservative Margin Transfer。最终测试集完全不参与 FedFalsify 路由。

12 轮结果：
RAHFL = Avg 33.8267 / Worst 27.0400 / WCCA 0.250 / CFG 30.050；
strict fit-only control = 30.7550 / 24.9800 / 0.250 / 30.225；
FedFalsify = 31.0733 / 24.5733 / 0.500 / 31.825。
FedFalsify 相对 control 最终为 Avg +0.3183、Worst -0.4067、WCCA +0.250、
CFG +1.600；最后五轮为 Avg +0.1180、Worst -0.4373、WCCA +0.850、
CFG +2.185。通信前期 Avg 可提升 0.5 到 1.4 点，后期衰减。

路由中平均覆盖率 48.89%，selected TAU 0.8825，但 30.68% 被选 source 的
paired advantage <= 0，45.45% 的 FRA strength=0。类别-算子结果显示部分
单元格提升 10 点以上，另一些下降 5 到 9 点。因此 TAU 只表示梯度兼容性，
不等于稳定 expertise；非劣 UCB 也不等于确定有益。

注意：当前 RAHFL 使用全部本地数据并用最终测试准确率路由，所以其绝对领先
只能作为诊断参考；当前真正可信的失败结论是 FedFalsify 没有战胜自己的严格
fit-only control。历史上 Oracle BER+CDep 相对 local control 的 Worst 曾提升
5.41 点，但它使用真实环境标签；Learned PEW Worst 提升 5.31 点，但依赖固定
损坏 taxonomy，不能作为最终方案。

请完成以下工作：
1. 从审稿人视角判断 CLE-HFL v2 是否值得作为论文新问题，并指出最致命攻击点；
2. 判断 FedFalsify 的失败来自哪些信息论或优化层面的根因；
3. 不使用固定损坏类别、不使用测试泄漏、不假设同构特征空间，提出最多两个
   核心创新模块的新通信框架；
4. 给出客户端/服务器完整流程、公式、通信内容、理论直觉与复杂度；
5. 解释它为什么不会再次出现“Avg 微升但 Worst/CFG 退化”；
6. 给出只需一次短实验即可判断方向的最小验证方案和 Go/No-Go 门槛；
7. 不要只提出稳定性加权、置信度加权、EMA、阈值门控或现有方法的表面拼接；
8. 如果理论上无法仅靠客户端输出/梯度实现，请诚实指出，并提出可发表的问题
   收缩方案。
```

## 九、相关本地文档

```text
docs/archive/methods/CLE_HFL_V2_FEDFALSIFY_FRAMEWORK_ZH.md
deliverables/cle_hfl_v2_probe_analysis_20260724/
  CLE_HFL_V2_PROBE_ANALYSIS_ZH.md
docs/project/CURRENT_PROJECT_MEMORY.md
docs/project/PROJECT_STATE.md
docs/project/TODO_NEXT.md
```

