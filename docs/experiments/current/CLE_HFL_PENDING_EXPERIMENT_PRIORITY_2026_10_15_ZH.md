# CLE-HFL待补实验优先级（2026-10-15）

## 1. 目的

本文档把V0.7-writing PDF中的13处`[EVIDENCE NEEDED]`归并为真正独立的实验缺口，并按论文
价值、算力成本和可由收缩论断替代的程度排序。13处红字不是13组实验；同一个缺口会在Abstract、
Introduction、Experiments、Discussion和Appendix重复出现。

本优先级不授权任何付费、Formal、多种子或长任务。每个实验仍须在启动前完成入口、输入、失败
安全、成本和冻结门槛检查。

## 2. 13处红字对应的8类缺口

| 唯一缺口 | 在论文中的重复位置 | 是否必须新训练 | 处理原则 |
|---|---|---:|---|
| A. Axis I更高本地预算复验 | Abstract、Introduction、Experiments、Main Table 3 | 否 | 已有40轮三seed M1可作主证据；不再默认把昂贵复验列为投稿前必须 |
| B. moderate剂量响应 | Experiments、dose-response表 | 是 | 高价值；补`gamma=0/0.5`，复用`gamma=0.9`方法锚点 |
| C. moderate Axis II剩余行 | Abstract、Introduction、Experiments、Discussion、HFL表 | 部分 | 先补AsymHFL成对锚点；八行全满不是前置条件 |
| D. local-first训练种子1/2 | Method、Experiments、Discussion、主表与seed附录 | 是 | 冲CCF-B的重要稳定性证据；成本过高时必须降低结论强度 |
| E. 第二private dataset | Introduction、Experiments、Discussion、跨设置表与seed附录 | 是 | 当前最关键的外部有效性缺口 |
| F. bounded taxonomy stress | Introduction、Experiments、Discussion、附录 | 是 | 强烈建议，但可诚实写作limitation替代 |
| G. JTT Formal | Axis I表 | 否（可删除行） | 仅当正文声称正式胜过JTT时才运行；默认删除pending行 |
| H. computation cost | 成本附录 | 否 | 从已有benchmark/Formal manifest回填；缺项可删表，不估算 |

## 3. 零GPU先完成的冻结决策（P0）

### P0-1：保留现有M1作为Axis I正式主证据

现有held-out map2四臂结果已经是40轮、training seeds 0/1/2、matched协议：

```text
ERM / CVaR-DRO / PEW+GroupDRO / PEW+BER
```

它已同时覆盖方法归因和训练随机性。后来启动但中断的full-local-budget seed-0复验只增加预算
耐久性信息，不改变研究问题，也不应倒逼论文把已完成M1写成“证据不足”。最终正文应准确披露
其每轮local-batch预算；除非目标会议审稿意见明确要求，否则不再优先重跑四臂昂贵复验。

### P0-2：缩减Axis II，而不是强行填满八行

Axis II的任务是提供HFL领域坐标，不承担BER的因果归因。最小有意义表为：

```text
Local/ERM
FedMD
FedProto
AsymHFL--ERM
AsymHFL+PEW+BER
```

前三行已经完成。两条AsymHFL行是必须补的moderate方法锚点。FedDF、RAHFL是高价值扩展；
AugHFL可选。若算力不足，正文使用5--7行精简表比保留八行pending更严谨。

### P0-3：删除不打算证明的pending主张

- JTT只留Related Work概念比较，不保留Formal胜负行；
- 成本表仅回填有manifest的实测值，否则移至补充材料或删除；
- 不把八行领域表写成完整排名或SOTA榜单；
- 不把local-first seed-0写成训练随机性下的普遍结论。

## 4. 付费实验的冻结优先级

### P1：moderate AsymHFL成对方法锚点——投稿主线必须

```text
AsymHFL--ERM, gamma=0.9
AsymHFL+PEW+BER, gamma=0.9
```

作用：在utility更可解释的moderate协议中直接回答“PEW+BER是否降低DSA，以及代价是什么”。
两臂必须共享初始化、partition、batch trace、通信、轮数、评价source和PEW checkpoint。旧任务因
入口回传故障而没有形成结果，修复后必须单臂运行并在训练结束立即上传recovery包。

若两臂没有完成，moderate Axis II只能说明CLE存在于Local/FedMD/FedProto，不能说明本方法在
该协议有效。

### P2：moderate剂量响应——投稿主线强烈建议

最终四行应为：

```text
ERM, gamma=0
ERM, gamma=0.5
ERM, gamma=0.9       # 复用P1的AsymHFL--ERM
PEW+BER, gamma=0.9   # 复用P1的方法臂
```

因此P1完成后，P2只新增`gamma=0`和`gamma=0.5`两次ERM训练，而不是再跑四次。它回答：无
定向绑定时DSA是否接近零、CLE增强时DSA是否上升，以及低准确率是否只是strong协议难度造成。
这是回应“场景过于人工”和“只挑gamma=0.9”的最直接证据。

### P3：M2 local-first多训练种子——冲CCF-B的重要项

目的：把local-first从seed-0受控机制发现提升为训练随机性下可复现发现。旧full-fit协议成本
过高且中断，不能直接恢复。再次运行前必须冻结统一显式local-batch预算，并在同一新预算下重跑
seed 0/1/2四臂，不能把历史full-fit seed 0与新budget seeds 1/2拼表。

若不运行，论文仍可成立，但只能写：

```text
predominantly local-first in one matched controlled factorial
```

不能写成稳定的一般机制。若目标明确为CCF-B，建议保留P3；若算力无法承受，则降低论断并把它
列为限制，不得用source-bootstrap冒充training-seed稳定性。

### P4：第二private dataset——冲CCF-B最关键的外部复现

当前所有主实验仍属于同一CIFAR-10 private semantic task。第二数据集要回答CLE、DSA和缓解
是否只存在于该任务。推荐CIFAR-100 private，但必须重新完成协议和类别--operator map审计。

分阶段执行：

1. seed 0 gate：`gamma=0`零基准与`gamma=0.9`directional shortcut；
2. seed 0方法：ERM、CVaR-DRO、PEW+BER；
3. 只有方向成立且学习水平合格，才追加matched training seeds 1/2；
4. PEW优先复用冻结checkpoint，检验跨private semantic task迁移。

最终论文若要认真冲CCF-B，P4的重要性高于再加第三通信底座、JTT或新BER模块。

### P5：bounded taxonomy stress——高价值边界实验

只做一个预注册压力协议：held-out operator或compound corruption二选一。最小arms为ERM与
PEW+BER，必要时加入CVaR-DRO。无论成功或失败都可报告：成功支持有限coverage robustness，
失败则定量界定taxonomy-assisted适用边界。

若不运行，必须在Limitations明确保留closed-set taxonomy、未知/复合corruption未验证；不能
通过提高PEW训练准确率或理论上界替代。

### P6：Axis II领域扩展——按预算选择

在P1之后，优先顺序为：

```text
FedDF-fidelity > RAHFL anchor > AugHFL-fidelity
```

- FedDF补充另一种public-distillation通信机制；
- RAHFL是最接近的corruption-robust HFL anchor；
- AugHFL与RAHFL同属鲁棒流派，预算不足时不必同时跑。

这些行提升领域定位，但不应优先于P3/P4，也不负责证明BER区别于CVaR/GroupDRO。

## 5. 两个投稿包

### 最小诚实成稿包

```text
现有问题、DSA理论、strong existence、seed-0 local-first
+ 已完成M1四臂40轮三seed
+ P1 moderate AsymHFL成对锚点
+ P2 moderate剂量响应
+ Axis II精简为已完成行
+ taxonomy与单数据集限制
```

该版本可以形成完整论文，但对CCF-B仍可能因单private dataset和local-first单seed而处于边缘。

### 认真冲CCF-B包

```text
最小诚实成稿包
+ P3 local-first三seed
+ P4第二private dataset三seed
+ P5 bounded taxonomy stress
+ P6至少一个FedDF或RAHFL领域锚点
```

这套扩展解决的是证据广度和不确定性覆盖，不是增加新模块。

## 6. 推荐实际执行顺序

```text
0. 不再重跑Axis I昂贵full-budget四臂；使用现有M1并准确披露预算
1. 重跑moderate AsymHFL--ERM与AsymHFL+PEW+BER（P1）
2. 复用P1的gamma=.9结果，只补ERM gamma=0/.5（P2）
3. 根据目标会议和积分决定P3与P4；若冲CCF-B，两者都做
4. 做一个bounded taxonomy stress（P5）
5. 有余力再补FedDF或RAHFL；AugHFL、JTT和更多底座停止在可选层
6. 每项结果到达后立即回填表格、Claim--Evidence台账和limitations
```

## 7. 当前最具体的下一项

下一项不是一次启动多个实验，而是先对修复后的moderate入口做一次最终失败安全复核，然后分别
启动P1的两个单臂Formal。只有这两个结果完整回传后，才启动P2的两个新增gamma点。

