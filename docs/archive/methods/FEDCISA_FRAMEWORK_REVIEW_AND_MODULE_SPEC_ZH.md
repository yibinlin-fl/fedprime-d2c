# FedCISA 框架审查与模块规范

状态：候选方案，尚未实现

更新日期：2026-07-18

本文用于审查 `CLE-HFL_FedCISA_框架与Codex实现规范.pdf`，并在编码前冻结问题定义、模块边界、公平性假设和最小验证顺序。本文不是实验结果，也不代表 FedCISA 已经能够稳定超过 RAHFL。

## 1. 项目当前事实

### 1.1 研究目标

当前目标是在以下联合场景中建立可发表的方法：

```text
模型异构 + 标签分布异构 + 数据损坏 + corruption-label shortcut
```

CLE-HFL 使用 `alpha` 控制标签异构，使用 `gamma/rho` 控制类别与损坏环境的绑定强度。

### 1.2 已验证的问题信号

固定 `alpha=0.5`、`seed=0` 和四种异构模型，RAHFL 的结果为：

| gamma | Avg | Worst | WCCA | CFG |
| ---: | ---: | ---: | ---: | ---: |
| 0.0 | 52.17 | 44.17 | 35.35 | 2.54 |
| 0.6 | 50.82 | 42.83 | 25.88 | 5.91 |
| 0.9 | 46.72 | 38.16 | 19.32 | 10.91 |

当 `gamma` 从 0 增加到 0.9 时：

```text
Avg   -5.45
Worst -6.01
WCCA  -16.03
CFG   +8.37（越低越好）
```

这说明 CLE-HFL 在当前受控协议下能够暴露 RAHFL 的系统性短板，但尚不能单独证明该问题在所有现实联邦场景中普遍存在。

### 1.3 已完成的负结果

FedCLEAR v0.1（CCRE + IRD）在 `gamma=0.9` 下没有超过 RAHFL：

| 方法 | Avg | Worst | WCCA | CFG |
| --- | ---: | ---: | ---: | ---: |
| RAHFL | 46.72 | 38.16 | 19.32 | 10.91 |
| FedCLEAR v0.1 | 45.41 | 36.42 | 17.80 | 11.42 |

主要失败原因：

1. CCRE 在已经损坏的私有图像上继续叠加新损坏，没有删除原始 corruption-label shortcut。
2. IRD 使用跨域 CIFAR-100 public logits，客户端容易形成共享无知而非可靠语义共识。
3. 复杂损失下降不等于 WCCA、CFG 或最终准确率改善。

FedCISA 必须避免重复以上失败机制。

## 2. 总体判断

### 2.1 结论

```text
研究方向：有条件保留
PDF 原版：不建议直接编码
修正版：解决两个阻塞问题后进入分阶段实现
```

FedCISA 的优点是首次把当前问题组织成完整闭环：

```text
协议暴露 shortcut
  -> 本地干预减少 shortcut
  -> 架构无关统计传递可靠知识
  -> 梯度投影减少通信负迁移
```

但 PDF 原版存在两个实现前阻塞项：

1. **反事实来源不成立**：当前 CLE 私有训练图像已经带有 base corruption。直接计算 `a_e(x)` 得到的是 `a_e(a_base(x_clean))`，仍然保留原 shortcut，不能直接称为 `do(E=e)`。
2. **异构统计不可直接比较**：`S_k[c]=||Cov(z,w|Y=c)||_F` 会随特征维度、尺度和架构变化；未经归一化的 logit margin 也会受不同模型 logit 温度影响。

在这两个问题解决前直接实现五个模块，很可能再次得到一个理论上解释漂亮、实际重复旧失败的系统。

### 2.2 建议的论文核心

不要把五个模块分别包装成五项创新。建议最终只保留两项方法贡献：

1. **Conditional counterfactual invariance**：针对 CLE-HFL 的本地类条件去耦。
2. **Invariance-gated safe structural transfer**：面向异构模型的可靠关系通信与负迁移保护。

GCW、GroupDRO 和梯度投影应被描述为实现上述核心机制的组件，而非分别声称首创。

## 3. FedCISA 总体结构

建议将框架写为：

```text
FedCISA
  = 固定 RAHFL local base
  + Conditional Counterfactual Invariance (CCI)
  + Invariance-Gated Safe Structural Transfer (IGSST)
```

其中：

```text
固定基座：AugMix + CE + JSD + DCL
CCI：CCR + 条件去耦 + margin stability
IGSST：归一化关系统计 + 不变性门控 + support mask + SPCG
```

这样可以把论文从“五个模块排列组合”压缩为“一个本地因果机制 + 一个联邦通信机制”。

## 4. 模块 A：CLE-HFL 协议与五类评测分布

### 4.1 保留内容

客户端联合分布为：

\[
P_k(Y,E)=P_k(Y)P_k(E\mid Y),
\qquad
P_k(E\mid Y)\neq P_j(E\mid Y).
\]

参数含义：

- `alpha`：客户端标签分布异构程度。
- `rho/gamma`：类别与主导损坏环境的绑定强度。
- `beta`：不同客户端类别-损坏映射的冲突程度。
- `xi`：训练样本被损坏的比例。
- `severity`：损坏强度分布。

### 4.2 必须实现的评测集

1. `clean`：基础语义能力。
2. `same`：延续训练映射，只作为诊断，不能单独作为主结果。
3. `random`：损坏与类别独立，是主要鲁棒泛化结果。
4. `swapped`：交换类别-损坏映射，直接检测 shortcut。
5. `unseen`：使用训练未见损坏类型或组合。

### 4.3 指标

- `Avg`：客户端平均准确率。
- `Worst`：最差客户端准确率。
- `WCCA`：最差 class-corruption 单元准确率，越高越好。
- `CFG`：同一类别在不同损坏组上的准确率跨度均值，越低越好。
- `ERS = Acc_same - Acc_swapped`：越低越不依赖训练映射。
- `NTR`：加入通信后性能下降的客户端比例。

### 4.4 审查意见

该模块可以优先实现，因为它不依赖 FedCISA 方法是否成功，并且能够补强 CLE-HFL 的协议贡献。

当前 `prepare_cle_data.py` 只有 balanced test，需要扩展为五个明确 split，并保存映射、种子、样本索引和 SHA256 审计。

## 5. 模块 B：Global Corruption Witness（GCW）

### 5.1 原始目标

使用共享的小型辅助模型 `psi` 预测人工 corruption type/severity，并输出统一 nuisance embedding：

\[
w=\psi(x),
\qquad
L_{wit}=CE(\hat e,e)+\lambda_s CE(\hat s,s).
\]

主模型特征只读取 `stopgrad(w)`，不能反向操纵 witness。

### 5.2 价值

1. 主分类模型可以保持 ResNet、ShuffleNet、MobileNet 等架构异构。
2. witness 为 nuisance 提供统一坐标。
3. 人工 intervention ID 不需要人工标注。

### 5.3 风险

如果 witness 在已经损坏的图像上学习“后来叠加的操作”，它未必能识别原始 base corruption。此时 `w` 不能代表真正造成 CLE 的 nuisance。

### 5.4 修正版决策

第一阶段不直接训练 GCW，而采用两级验证：

```text
Oracle witness：使用协议中已知的 base corruption ID，只做机制上界诊断
Learned witness：Oracle 有效后，再验证能否从图像预测 base nuisance
```

Oracle 结果只能作为机制实验，不能作为无额外标签的最终主方法结果。

GCW 的 Go 条件：

- base corruption 分类准确率显著高于随机水平；
- embedding 与 corruption 的关联高于与 class label 的关联；
- unknown/combined corruption 上的距离仍具有排序意义。

## 6. 模块 C：Counterfactual Corruption Randomization（CCR）

### 6.1 原始目标

从类别无关分布采样环境：

\[
(e_j,s_j)\sim q_t(e,s),
\qquad
x_{cf}^{(j)}=a_{e_j,s_j}(x),
\qquad
q_t(e,s)\perp Y.
\]

GroupDRO 更新环境采样权重：

\[
q_{t+1}(e)\propto q_t(e)\exp(\eta L_e).
\]

### 6.2 阻塞问题：叠加不等于替换

当前私有图像满足：

\[
x^{obs}=T_{e_{base}(y,k)}(x^{clean}).
\]

继续应用 `a_e` 得到：

\[
x'=a_e(x^{obs})=a_e(T_{e_{base}}(x^{clean})).
\]

原始 `e_base` 仍然存在。因此这只能叫 nuisance diversification，不能无条件声称执行了 `do(E=e)`。

### 6.3 编码前必须选择的模式

#### 模式 A：Oracle rerender（只用于机制验证）

从同一 clean source 重新渲染不同 corruption，而不是叠加：

\[
x_{cf}^{(j)}=T_{e_j}(x^{clean}).
\]

优点：反事实定义最干净。

缺点：训练方法获得 clean source 或 paired source，属于额外信息假设，不能直接和只见 corrupted image 的 RAHFL 宣称完全公平。

#### 模式 B：Observed-only composition（可部署但理论较弱）

只访问 `x_obs`，继续做类别无关损坏扩增。

优点：与现实输入条件一致。

缺点：不能保证删除原 shortcut，已经被 CCRE 负结果警告。

### 6.4 当前建议

先实现 Oracle rerender 作为机制审计，并同时运行两个公平控制：

```text
RAHFL on observed corrupted data
RAHFL + same oracle rerender views
FedCISA-local + same oracle rerender views
```

如果第二组已经获得全部增益，说明贡献来自额外反事实数据，而非条件去耦方法。

若希望形成最终可部署主方法，必须进一步提出不依赖 clean source 的 nuisance replacement 或明确把“可获得采集环境元数据/重渲染器”写成任务假设。

## 7. 模块 D：Conditional Semantic Decoupling（CSD）

### 7.1 反事实分类

\[
L_{cf}=\frac{1}{J}\sum_j CE(f(x_{cf}^{(j)}),y).
\]

该项保证同一语义样本在多个环境中仍然可分类。

### 7.2 条件交叉协方差

PDF 使用：

\[
L_{cind}=\sum_c \|Cov(z,w\mid Y=c)\|_F^2.
\]

条件化比无条件去相关更合理，因为它只抑制同一类别内部随 nuisance 改变的表示成分。

### 7.3 必须弱化的理论表述

简单线性交叉协方差为零，一般不能推出：

\[
Z\perp E\mid Y.
\]

它只能说明当前特征与 witness 在二阶线性统计上的条件相关性降低。若需要更强的条件独立主张，应使用 CIRCE、conditional HSIC 或给出受限分布假设。

第一版建议使用归一化条件交叉协方差，并将其称为 surrogate：

\[
\widetilde S_{k,c}
=
\frac{\|Cov(\bar z,\bar w\mid Y=c)\|_F^2}
{d_zd_w+\epsilon}.
\]

### 7.4 Margin variance

\[
m_y(x)=z_y-\log\sum_{c\neq y}\exp z_c,
\qquad
L_{mvar}=Var_j[m_y(x_{cf}^{(j)})].
\]

它比只约束 softmax 输出更直接地度量正确类别相对竞争类别的稳定性。

### 7.5 第一版范围

第一版只启用：

```text
L_RAHFL-local + lambda_cf L_cf + lambda_cind L_cind + lambda_mvar L_mvar
```

暂缓 GroupDRO 和 cross-environment SupCon，避免一次加入过多变量。

## 8. 模块 E：Invariance-Gated Structural Relation Transfer（IGSRT）

### 8.1 原始关系统计

对客户端 `k`：

\[
R_k[c,c']
=
E_{x:y=c,j}[z_c(x_{cf}^{(j)})-z_{c'}(x_{cf}^{(j)})].
\]

关系敏感度：

\[
V_k[c,c']
=
Var_j[z_c(x_{cf}^{(j)})-z_{c'}(x_{cf}^{(j)})].
\]

### 8.2 优点

- `C x C` 关系矩阵不依赖 feature dimension。
- 类别缺失可通过 support mask 跳过。
- 不需要共享样本级 public logits。

### 8.3 与已有工作的重叠

FedSAF 已明确提出在异构联邦学习中从坐标对齐转向类间结构对齐。因此 FedCISA 不能把“结构关系通信”本身写成首创。

FedCISA 能保留的差异是：

```text
关系来自 counterfactual corruption views
+ 用跨环境敏感度过滤 shortcut-heavy 关系
+ 使用安全梯度限制关系通信负迁移
```

### 8.4 异构 logit 尺度校准

不同架构的 margin 绝对值不可直接聚合。上传前必须进行架构内归一化，例如：

\[
\bar R_k[c,:]
=
\frac{R_k[c,:]-mean(R_k[c,:])}
{std(R_k[c,:])+\epsilon}.
\]

也可以比较 row-wise rank 或 normalized log-odds。第一版建议使用 row-wise z-score，并保留 raw margin 仅做诊断。

### 8.5 门控修正

PDF 中的 `S_k[c]` 不能直接跨架构比较。第一版门控只使用公共输出空间中的量：

\[
\omega_{k,cc'}
=
support_{k,c}
\cdot
\exp(-\bar V_k[c,c']/\tau_v).
\]

其中 support 使用归一化或分桶后的类别支持度，不上传精确计数。

只有 learned witness 和归一化 `S` 经验证可比较后，才加入第二个门控项。

### 8.6 聚合与本地对齐

服务器只在有效 support 上聚合：

\[
R^*[c,c']
=
RobustAggregate_k(\bar R_k[c,c'];\omega_{k,cc'}).
\]

本地结构损失：

\[
L_{rel}
=
Huber(M\odot(\bar R_{batch}-stopgrad(R^*))).
\]

第一版先用 weighted mean 验证信号，之后再引入 Huber/trimmed mean，避免稳健聚合器本身成为额外变量。

## 9. 模块 F：Safety-Protected Communication Gradient（SPCG）

### 9.1 定义

\[
g_p=\nabla L_{primary},
\qquad
g_r=\nabla L_{rel}.
\]

当 `dot(g_p,g_r)<0` 时：

\[
g_r^{safe}
=
g_r-
\frac{\langle g_r,g_p\rangle}
{\|g_p\|^2+\epsilon}g_p.
\]

更新梯度：

\[
g_{total}=g_p+\lambda_{rel}g_r^{safe}.
\]

### 9.2 能保证什么

在欧氏梯度、小步长的一阶近似下，投影后的通信梯度不会与主目标梯度形成负内积。

### 9.3 不能保证什么

- 不能保证有限步长下准确率单调提升。
- 不能保证 Adam、动量和权重衰减下仍严格满足同一结论。
- 不能保证测试风险、WCCA 或 CFG 不下降。
- PCGrad/梯度投影本身不是新算法。

### 9.4 实现建议

第一版只对 classifier head 做投影，记录：

```text
gradient_dot
gradient_cosine
conflict_rate
projection_norm_ratio
NTR
```

如果 head-only 有效，再扩展到最后一个 feature block。

## 10. 修正版训练流程

### 阶段 0：协议与基线

1. 生成 `clean/same/random/swapped/unseen` 五种评测集。
2. 在同一配置下重新评估 RAHFL。
3. 检查 `rho` 增大时 ERS/CFG 上升，random/swapped/WCCA 下降。

### 阶段 1：Oracle 本地机制验证

1. 使用已知 base environment ID 作为 oracle witness。
2. 明确区分 `oracle_rerender` 与 `observed_compose`。
3. 只启用 `L_cf + L_cind + L_mvar`。
4. 对比 RAHFL、RAHFL+同反事实数据、FedCISA-local。

Go 条件：

```text
random/swapped WCCA 明显提高
ERS/CFG 明显下降
clean 降幅 <= 1 point
FedCISA-local 超过 RAHFL + same views
```

### 阶段 2：Learned witness

只有 Oracle 机制成立后才训练 GCW，并比较：

```text
oracle witness
learned witness
no witness
```

### 阶段 3：结构通信

1. 先实现归一化 `R/V`、support mask 和 weighted mean。
2. 与 local-only 比较，确认通信收益。
3. 再加入 invariance gate。
4. 最后加入 SPCG，验证 conflict rate 和 NTR 是否下降。

### 阶段 4：正式实验

通过 12-round probe 后再跑 40 rounds、多 seed、多 `rho/alpha/beta`。

## 11. 最小消融矩阵

| 编号 | 本地模块 | 通信模块 | 回答问题 |
| --- | --- | --- | --- |
| A | RAHFL local | AsymHFL | 匹配基线 |
| B | RAHFL + same counterfactual views | AsymHFL | 额外视图本身的收益 |
| C | CCI local | 无 | 条件去耦是否有效 |
| D | CCI local | 普通 normalized relation | 结构通信是否有信息 |
| E | CCI local | invariance-gated relation | 门控是否过滤 shortcut |
| F | CCI local | gate + SPCG | 安全投影是否降低负迁移 |

必须同时报告 `Avg/Worst/WCCA/CFG/ERS/clean`，不能只挑对方法有利的指标。

## 12. 代码模块规划（审批后执行）

不新建一套与当前项目平行的 `project/` 目录，而是沿用现有 `fedprime/` 结构。

### 12.1 数据与指标

```text
fedprime/data/cle_protocol.py
fedprime/data/cle_eval.py
fedprime/engine/cle_metrics.py
scripts/prepare_cle_data.py              # 扩展，不破坏旧参数
scripts/audit_cle_protocol.py
```

### 12.2 方法

```text
fedprime/methods/fedcisa.py               # runner/orchestrator
fedprime/methods/fedcisa_local.py         # CCI local objective
fedprime/methods/corruption_witness.py    # oracle/learned witness
fedprime/methods/structural_transfer.py   # normalized R/V + masks + aggregation
fedprime/methods/safe_gradient.py         # SPCG
```

### 12.3 配置

```text
configs/debug_fedcisa_oracle_local.yaml
configs/debug_fedcisa_relation.yaml
configs/openi_v100_fedcisa_probe.yaml
configs/ablations/fedcisa_*.yaml
```

### 12.4 测试

```text
tests/test_cle_protocol_splits.py
tests/test_fedcisa_conditional_cov.py
tests/test_fedcisa_relation_scale.py
tests/test_fedcisa_relation_mask.py
tests/test_fedcisa_safe_gradient.py
tests/test_fedcisa_config_fairness.py
```

## 13. 实现前审批项

开始编码前必须确认以下决策：

1. **反事实访问假设**：最终主方法是否允许访问 clean source/paired rerender？
2. **Oracle 定位**：是否同意先把 corruption ID 作为机制上界，而不是最终主结果？
3. **关系通信范围**：第一版是否只用 normalized `R/V`，暂不使用跨架构不可比的 raw `S`？
4. **梯度投影范围**：第一版是否只投影 classifier head？
5. **实验预算**：是否同意先跑协议 + oracle local probe，未通过则不实现完整通信？

## 14. 文献边界

FedCISA 所用思想均有相关基础，投稿时必须准确定位：

1. NURD 已研究 nuisance-label spurious correlation 和 nuisance-randomized distribution：<https://arxiv.org/abs/2107.00520>
2. CIRCE 已研究条件不变表示与条件独立度量：<https://arxiv.org/abs/2212.08645>
3. PCGrad 已提出冲突梯度投影：<https://arxiv.org/abs/2001.06782>
4. FedSAF 已在异构联邦学习中提出类间结构对齐：<https://arxiv.org/abs/2605.05959>
5. FedDiverse 已研究联邦类别不平衡、属性不平衡和虚假相关：<https://arxiv.org/abs/2504.11216>

因此不能将“条件去相关”“结构关系”“梯度投影”分别宣称为首创。可能成立的创新边界是：

> 在 CLE-HFL 的模型异构场景中，以受控 corruption intervention 估计关系可靠性，并将不变性门控的架构无关结构通信与本地主目标安全约束组合成可验证闭环。

该表述仍需正式系统文献排重和实验支持。

## 15. 最终审查结论

### 可以保留

- CLE-HFL 的 `rho/beta` 问题定义和五种评测 split。
- 条件化而非无条件去耦的思想。
- 架构无关的关系通信。
- 缺失类 support mask。
- 通信负迁移的显式梯度诊断与安全投影。

### 必须修正

- 不得把已损坏图像上的二次增强直接称为反事实环境替换。
- 关系 margin 必须做模型内尺度归一化。
- raw feature covariance norm 不得直接作为异构客户端权重。
- 简单 covariance penalty 不得声称严格实现条件独立。
- SPCG 只能声称一阶梯度冲突消除，不能声称准确率安全。

### 暂缓

- GroupDRO 自适应环境采样。
- cross-environment SupCon。
- learned GCW 联邦预训练。
- weighted Huber/trimmed mean。
- last-block/full-model 梯度投影。

### 当前决策

FedCISA 不是可以“一次性全实现再赌一次结果”的冻结方案。它是一个值得继续的候选框架，但应先解决反事实来源，并按照：

```text
协议 -> Oracle 本地机制 -> Learned witness -> 关系通信 -> 安全投影
```

逐层通过 Go/No-Go。这样每一步都有明确的理论问题和可证伪指标，也能避免继续试盲盒。
