# CLE-HFL 外部基线公平性审计（2026-08-09）

## 总结论

当前 12 轮外部基线结果可以保留，但必须准确命名为：

> 在统一 CLE-HFL 数据、模型、随机种子和名义训练预算下，对各方法核心机制适配版进行的筛选实验。

它还不能支持以下强结论：

- “CLE-HFL 在完全公平条件下击败所有原论文方法”；
- “PEW/BER 的通信策略优于 FedMD、FedDF、KT-pFL、FCCL 等通信方法”；
- “已经完成官方端到端复现或 SOTA 对比”。

最可信、最接近受控 A/B 的外部比较是 `PEW+BER` 对 `RAHFL`：二者都采用
AugMix/JSD/DCL 和 strict AsymHFL-val，主要差异是 PEW/BER。其他方法之间还混入了
不同本地损失、不同额外遍历次数和不同原论文训练配方。

## 审计判定

| 审计维度 | 判定 | 说明 |
|---|---|---|
| 数据与客户端划分 | 通过（有一项可追溯性缺口） | CIFAR-10 私有数据、CIFAR-100 公共数据、CLE 场景和 seed 均一致；两批实验的 split 文件名不同，尚未在归档中记录内容哈希 |
| 模型与初始化条件 | 通过 | 四个异构模型一致，seed=0；跨任务重复的 RAHFL/PEW 锚点逐数值一致，支持初始化和数据顺序可复现 |
| 轮数、batch、优化器 | 通过 | 12 轮、1 local epoch、batch=64、Adam、lr=0.001、pretrain=0 一致 |
| 公共蒸馏名义预算 | 基本通过 | 需要公共数据的方法统一为每轮 4×128；FedProto/Local 不需要公共数据 |
| 实际计算预算 | 不通过 | RHFL/FedProto 每轮约 177–185 秒，其他方法约 93–98 秒；PEW 还有未计入 round time 的 5 epoch 预训练和全私有集推理 |
| 本地训练目标一致性 | 不通过 | PEW+BER/RAHFL 使用 AugMix+JSD+DCL；FedMD/FedDF/KT-pFL/FCCL/FedProto 多为普通 CE；RHFL 使用 SCE |
| 训练期私有/测试标签隔离 | 通过 | fit 负责梯度，AsymHFL 用 audit 路由，final-test 只在运行内报告；非路由基线不读取 audit/test 进行通信 |
| 方法间信息条件一致 | 不通过但可披露 | PEW 额外获得由已知合成变换产生的环境/严重度监督，并用合成验证标签选择 epoch 和 unknown 阈值 |
| 调参预算一致 | 不通过 | 候选方法已在 seed0 上做多轮模块和超参数筛选；多数外部基线只跑单一默认参数 |
| 原论文实现忠实度 | 部分通过 | 多数是核心机制适配，不是原仓库完整训练流程 |
| 未触碰最终评测集 | 运行内通过、研究流程层面不通过 | 代码没有用 final-test 标签训练；但研究过程中已经多次根据 seed0 final 指标选模块和超参数 |
| 统计与泛化结论 | 不通过 | 外部基线表目前只有单 scenario、单 training seed、12 轮筛选 |

## 已统一的实验条件

两批结果包中的 resolved/generated config 显示所有 arm 共享：

- `scenario = cle_hfl_v2`；
- 私有场景 `alpha05_gamma09_seed0_split0`；
- `seed = 0`，strict fit/audit split seed 也为 0；
- 模型为 ResNet10、ResNet12、ShuffleNet、MobileNetV2；
- 12 rounds、1 local epoch、batch 64；
- Adam，lr 0.001，weight decay 0；
- public size 5000，public batch 128，每轮最多 4 个公共 batch；
- 同一 final-test 协议及 Avg、Worst、WCCA、CFG 等指标。

两批归档中的 RAHFL 最后五轮结果完全一致；标准 PEW/候选锚点也完全一致。这是跨 OpenI
任务复现性很强的正面证据。不过旧归档使用
`strict_cle_v2_alpha05_gamma09_seed0_split0.npz`，新归档使用
`strict_cle_v2_seed0_split0.npz`。未来需要将 split SHA-256 写入结果清单，不能只依赖文件名或结果吻合。

## 最大的不公平来源：本地训练骨架不同

| Arm | 本地训练 | 通信/协作 |
|---|---|---|
| Local-only | AugMix + JSD + DCL | 无 |
| FedMD | 普通 CE | 对称公共 logit KD |
| RHFL | SCE | 置信度加权公共 KD |
| FedProto | 普通 CE + prototype MSE | 类别 prototype 聚合 |
| AugHFL | AugMix + JSD（无 DCL） | 公共增强一致性教师加权 |
| RAHFL | AugMix + JSD + DCL | strict AsymHFL-val |
| FedDF | 普通 CE | 平均 logits 教师蒸馏 |
| KT-pFL | 普通 CE | 个性化可学习教师系数 |
| FCCL | 普通 CE | 公共 logits 互相关 |
| PEW+BER | AugMix + JSD + DCL + PEW/BER | strict AsymHFL-val |

因此，现有表回答的是“整套适配方案在相同名义训练预算下谁更好”，而不是“只更换通信方法
后谁更好”。尤其不能把 PEW+BER 对 FedDF/FedMD 的差距全部归因于 PEW/BER。

## 信息条件与 PEW 的额外资源

PEW 不使用 CIFAR-100 原类别标签，也不使用私有类别标签来训练环境识别器；但是它并非与
其他方法处于完全相同的无监督信息条件：

1. 对 5000 张公共图像施加代码已知的 corruption；
2. 由生成过程自动得到 6 类环境标签和 5 级严重度标签；
3. 训练 5 epoch 的 witness；
4. 用合成公共验证标签选择最佳 epoch；
5. 扫描阈值并用合成验证准确率选择 unknown threshold；
6. 对约 4 万张私有 fit 图像离线推理环境和 embedding。

这是一种“无需人工标注类别、但使用已知变换监督”的额外假设。论文可以使用它，但必须明确
披露，不能写成所有方法拥有完全相同监督。Strict PEW-LOO 能检验未见 operator 泛化，却不会
消除 PEW 拥有 corruption-family 生成知识和额外预处理计算这一事实。

## 基线实现忠实度分级

| 基线 | 忠实度 | 可采用的准确表述 | 主要偏差/风险 |
|---|---|---|---|
| RAHFL | 较高的核心适配 | RAHFL-style robust local training under strict CLE protocol | 原论文更长预训练/轮数；这里把路由改为严格 audit，采用统一 4 public batches |
| RHFL | 较高的核心适配 | RHFL confidence-weighted KD adaptation | 使用 fit-only SCE 估计质量；统一调度不同于完整原配方；每轮额外遍历 fit |
| FedProto | 较高的核心适配 | native 1024-d prototype aggregation adaptation | 为异构模型统一到现有 1024 维 embedding；每轮额外遍历 fit，非未经修改的官方实现 |
| AugHFL | 中等 | AugHFL core-mechanism adaptation | 原实现为每客户端生成独立公共 AugMix view；当前实现给各客户端共享同一组三视图，增强细节也不完全相同 |
| FCCL | 中等 | FCCL public cross-correlation core | 保留互相关核心及 0.0051 系数，但没有完整 federated continual/history 部分 |
| FedDF | 中等 | FedDF-style heterogeneous logit ensemble adaptation | 每个持久客户端都作为 student；不是原论文完整 server fusion/长蒸馏流程 |
| FedMD | 中低 | FedMD-style symmetric public-logit KD | 只保留 consensus/public KD 核心，没有完整 transfer-set revisit/pretraining 流程 |
| KT-pFL | 中低 | KT-pFL equation-based adaptation | 依据论文公式实现，缺少可信官方代码逐数值对照；正式结果也缺少教师系数诊断 |

这些实现不是“伪基线”，但论文中必须统一称作 core-mechanism adaptation；若写成“official
implementation”或“完整复现”，会被审稿人抓住。

## 实际预算证据

归档 `metrics.csv` 的 12 轮平均耗时（秒/轮）为：

| 方法 | 秒/轮 | 峰值显存 MB |
|---|---:|---:|
| FedDF | 93.97 | 2873.8 |
| KT-pFL | 93.88 | 2873.8 |
| FCCL | 93.18 | 2873.8 |
| FedMD | 95.75 | 3767.3 |
| AugHFL | 95.79 | 3766.0 |
| RAHFL | 94.67–96.24 | 5394.9 |
| PEW+BER | 97.71 | 5394.4 |
| RHFL | 185.21 | 3766.4 |
| FedProto | 177.11 | 3941.5 |

PEW 的 witness 训练和私有集标注发生在 round 计时之前，所以 97.71 秒不能代表其完整端到端
代价。论文效率表应同时报告 setup time、训练总时长、额外 fit/public forward 次数和显存。

## final-test 隔离的两层结论

### 算法运行层面

通过。strict runner 中：fit-only gradients、audit-only AsymHFL routing、final-test reporting-only。
FedMD、FedDF、KT-pFL、FCCL、RHFL、AugHFL、FedProto 的通信策略声明不使用 accuracy routing，
因此不会读取 audit 或 final-test 来选择教师。

### 整个研究流程层面

不再是 untouched test。seed0 的 final 指标已经被用于决定是否保留 CDep、选择 BER、比较
PEW 版本和推进实验。即使代码没有反向传播测试标签，研究者层面也发生了对 seed0 benchmark
的自适应选择。

因此应立即把 scenario seed0 定义为 development scenario。之后冻结方法和超参数，只把从未
看过的 scenario seed1/2 与第二数据集作为 confirmatory evaluation；确认集出结果后不能继续
针对它调参再重跑并只报告最好结果。

## 当前论文允许和禁止的表述

允许：

- 所有方法在统一 CLE-HFL 数据、异构模型和 12 轮名义预算下评估；
- PEW+BER 在当前 seed0 筛选中优于这些 core-mechanism adaptations；
- PEW+BER 相对共享 robust local backbone 的 RAHFL 有明确正增益；
- 结果是 screening evidence，尚待未见场景和第二数据集确认。

禁止：

- “公平复现了所有原论文并取得 SOTA”；
- “PEW/BER 的通信优于 FedDF/FedMD/KT-pFL/FCCL”；
- “所有方法监督完全相同”；
- “final test 从未参与开发决策”；
- 仅凭单 seed、12 轮给出显著性或跨场景泛化结论。

## 修复优先级

### P0：在继续大规模实验前完成

1. 冻结 seed0 为开发场景，冻结 PEW+BER 当前配置。
2. 新增 machine-readable fairness manifest：记录数据目录、split SHA-256、seed、模型、初始化
   hash、每轮 public batch 索引、optimizer、训练预算和代码 commit。
3. 把结果分为两张表：
   - 表 A：现有 method-native/core-adaptation 的统一预算筛选；
   - 表 B：完全相同本地 backbone 下，仅替换通信策略的受控比较。
4. 给外部基线相同的小规模调参预算，或明确预注册“全部使用论文默认参数且不调参”；不能让候选
   方法经历多轮搜索，而基线只跑一个任意配置。

### P1：形成可投稿证据

5. 为 FedMD/FedDF/KT-pFL/FCCL/AugHFL 建立同一 robust local backbone 的 communication-only
   版本；不要覆盖现有 native/core 版本。
6. 给 KT-pFL 保存系数矩阵轨迹，给 FCCL 保存互相关诊断，给 AugHFL 保存教师权重和 view
   consistency，证明机制确实工作而不只是“代码能跑”。
7. 对原论文完整配方与统一预算适配分别说明；至少对最强和最相关基线做一次原配方 sanity run。
8. 在 untouched scenario seed1/2 上做冻结后的 confirmatory comparison，再上第二数据集。

### P2：报告质量

9. 报告 setup time、总 GPU 时间、显存、公共/私有额外 forward passes 和通信量。
10. 所有表格脚注注明 `core-mechanism adaptation`、公开数据监督差异和 12-round screening 属性。

## 最终判断

现有结果不是应当推翻的“假阳性”，因为最关键的 PEW+BER 对 RAHFL 比较具有较好的受控性，
并且数据、模型、seed 和运行内测试隔离都做得较严谨。但“我们的完整方案显著优于一批外部
方法”的幅度中，确实混入了更强本地训练、额外 PEW 监督/计算以及更充分的候选调参。

在完成同本地骨架通信对照、等额调参规则和 untouched scenario 验证前，应把外部基线表定位为
筛选证据，而不是最终公平 SOTA 表。

## 验证记录

使用项目 PyTorch 环境执行以下聚焦测试：

```text
tests/test_communication_strategies.py
tests/test_cle_external_baselines.py
tests/test_cle_remaining_baselines.py
tests/test_cle_communication_factorial.py
```

结果为 `16 passed`。这证明当前适配器的接口、配置生成和已登记机制测试通过，但不能替代与
原作者代码的端到端数值复现，也不能消除上述实验设计层面的公平性问题。
