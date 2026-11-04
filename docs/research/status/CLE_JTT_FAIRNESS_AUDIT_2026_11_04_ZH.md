# JTT与已完成Axis I公平性审计

日期：2026-11-04。结论：**旧JTT入口不能直接用于补入当前Axis I正式表。**
这是工程与协议就绪性结论，不是JTT方法的科学NO-GO。现有四臂结果保持不变。
本轮没有启动训练、上传数据、推送或修改任何训练实现。

## 1. 审计基准是实际完成的配置

已读取并核验12个冻结配置（4 arms×3 training seeds），以及三个seed下四臂的local trace哈希。
它们来自已回传的原始configs/outputs，而不是今天的通用runner默认值：

```text
private/public: CIFAR-10 / CIFAR-100
scenario: cle_hfl_v2_cross_map2_seed0_split0
partition seed: 0; binding-map seed: 2; evaluation seed: 20260909
training seeds: 0/1/2
models: ResNet10 / ResNet12 / ShuffleNet / Mobilenetv2
rounds: 40; local_epochs: 1; local_batches/client/round: 16
batch_size: 64; public_batch_size: 128; public_batches_per_round: 4
pretrain_epochs: 0; num_workers: 0
Adam: lr=.001, weight_decay=0; max_grad_norm: 5
communication: strict AsymHFL-val
standard single-view; lambda_jsd=0; no DCL/CDep
strict fit/audit split; loader_seed=20260909+train_seed
```

standard单视图仍沿用基础ColorJitter/随机裁剪/翻转及归一化；不能把“无AugMix/JSD/DCL”
误写成完全没有数据增强。fit-only仅指错误集合；AsymHFL仍按原协议使用audit任务标签路由。

冻结配置还有`skip_nonfinite=true`。当前通用`run_cle_v2_spurious_final.py`相比它们变成了
`max_local_batches=64`和`skip_nonfinite=false`。这不是本轮新改动，也不能反向改写已完成实验。
补JTT时必须以真实冻结配置为基准，不能只改当前runner的arm列表。非有限处理策略与实际跳步
需记录，不能仅凭最终metrics有限就声称每步均未跳过。

## 2. 已通过的实现检查

| 检查 | 结果与证据 |
|---|---|
| 错误集合来源 | `generate_jtt_error_masks`只遍历持久化fit indices，读取任务标签，未将audit/test样本用于错误集合 |
| 索引对齐 | CPU mock验证非连续fit索引[4,1,2]，其中index1误判；最终mask准确落回完整dataset的index1，audit中的误判index5未进入 |
| 错误集合冻结 | 第一阶段结束后生成一次mask，第二阶段不动态重算错误样本 |
| 第二阶段初始化 | 配置继续指向initial_states；新进程创建模型、严格载入初始权重后新建Adam，不沿用识别模型/优化器 |
| 额外模块 | JTT配置删除fedease，不使用PEW/BER；单视图，lambda_jsd=0 |
| 旧独立JTT预算 | Formal第二阶段为40×16；不能误称旧JTT代码完全未实现 |
| 旧四臂配对 | 三个seed各自四臂的trace哈希完全一致；12个配置的预算和seed身份核验通过 |

这些局部通过项不等于新JTT Formal已可运行。初始化的检查范围是配置/调用路径；本轮没有加载
真实四模型执行完整训练。mask检查是小型CPU单元核验，不是私有数据上的科学结果。

## 3. 必须修正/补齐的项目

### P1-A：旧入口默认原map，未绑定map2与三seed

`openi_cle_jtt_formal_entry.py`继承原factorial输入搜索，默认寻找694118746-byte输入包；
没有cross-map2解包与身份审计。独立runner只用train_seed0，没有可选1/2，也不写scenario_id。
手动给runner传map2目录时可以构造路径，但旧OpenI启动流程并不会因此自动选对map2。

修复要求：明确复用1385820059-byte cross-maps包、选出map2；检查manifest、fit/audit split、
初始权重、public archive、paired evaluation grid与同seed冻结配置一致。禁止用主评价893MB包，
也禁止拿原map的错误mask直接放到map2。

### P1-B：两阶段长度不独立，旧实现为40+40

旧`ROUND_BUDGET[formal]=40`同时用于识别ERM与最终JTT，识别用的是round40最后checkpoint。
原论文强调较早的识别模型与独立超参数T；旧入口没有T，不能选择事前冻结的早期识别点。
40轮识别并非数学上绝对禁止，但不能未经检验就当作原论文的合适实现，也增加额外完整训练。

修复要求：识别时长和最终训练时长分别记录；不能把“第二阶段40轮”说成“总共40轮”。
更不能把第二阶段接着第一阶段继续训练当作JTT。

### P1-C：batch自归一化不等价于全数据重采样

旧函数计算：

\[
\widehat L_B=\frac{\sum_{i\in B}w_i\ell_i}{\sum_{i\in B}w_i}.
\]

原论文固定错误集后按权重重复样本，对应的归一化全数据目标为
\(L_w=\sum_iw_i\ell_i/\sum_iw_i\)。一般情况下
\(E_B[\widehat L_B]\ne L_w\)。已有测试只证明一个batch内部显式重复的算术一致，
没有验证全数据minibatch期望。

实际CPU数值反例：四个样本loss=[1,4,4,4]，只有第一个属于错误集合，lambda=20，batch=2，
均匀遍历全部六种batch。全数据目标为32/23=`1.3913043478`，旧batch目标均值为18/7=
`2.5714285714`。用全fit固定均值权重归一化的uniform-batch估计则得到`1.3913043478`。
这说明全局等价声明不成立；不代表真实batch64的实际偏差已经测量，也不证明JTT效果差。

优先方案：使用固定错误集的重采样分布，再做普通mean CE，明确为JTT的HFL适配。
备选方案：保留原batch轨迹，用固定全fit归一化权重；必须命名为JTT-style weighted adaptation，
不能声称与原文的minibatch、BatchNorm和优化轨迹逐步相同。不要改旧函数以改写历史screen实现。

### P1-D：旧JTT入口仍有分析失败丢结果风险

旧入口无训练前output context强制检查；等两阶段训练和CUDA-only分析全部结束后才上传。
成功包还使用排除checkpoint的light打包函数。它没有本项目后来补给其他入口的分析前恢复包，
也没有第一阶段独立回传。

修复要求：识别训练完成后回传checkpoint/config，mask生成后回传mask和provenance；
第二阶段完成先回传checkpoint，再用auto设备分析；失败保持非零状态。平台硬停止仍不保证
finally能执行，不能承诺绝对不丢失。

### P2：mask与分析的证据合同不足

mask manifest只有记录数/错误率/输出mask哈希，没有绑定输入、split、识别checkpoint、
train seed/map seed或归一化规则；加载器仅检查0/1形状，不能排除同长度但来源不同的mask。
生成函数允许部分fit扫描，Formal没有强制fit_evaluated==fit_total。需新增完整覆盖审计。

旧分析器硬编码seed0，未对训练合同、完成标记、全量预测有限性或配对失败执行科学fail-closed；
`mode=formal`本身即可使scientific_evidence=true。需与现有正式评价身份、source级bootstrap、
shuffled null、DSA/grid accuracy和完成审计统一；不得拿source CI冒充三seed稳定性。

## 4. 建议协议：草案，尚未实施/启动

详细比较矩阵见`CLE_JTT_MATCHED_PROTOCOL_DRAFT_2026_11_04_ZH.md`。

- 主比较继承原Axis I map2、40轮×16的最终训练预算与三training-seed。
- 识别阶段建议候选为12轮×16、lambda=20，继承旧screen设置，不依据已见map2 DSA重调。
  这只是待确认的HFL适配起点，不是已验证最优超参，也不是原论文指定的通用值。
- 首选固定错误集重采样；样本顺序变化属于方法，不再要求与ERM私有trace逐字节相同。
  要检查同fit来源、相同最终更新预算、初始权重与公共batch流，以及独立可复现的采样seed。
- 先seed0小型smoke与benchmark；若正式只跑seed0，只能与原四臂seed0作补充比较。
  三seed主表中增加JTT均值/SD必须实际完成对应三seed，不混单seed与三seed均值。
- 新增baseline为已见场景的post-hoc扩展，不再称map2对本次新增设计完全未见。

## 5. 成本公平的准确表述

最终阶段40×16与已有四臂相同，但T=12识别阶段仍增加30%的本地更新预算：

```text
普通单阶段：每client 40×16=640次local updates
JTT候选：每client (12+40)×16=832次local updates
若识别也用AsymHFL：共52轮通信预算，而不是40轮
额外：fit全量预测、mask生成、回传和最终评价
```

因此可声称“最终训练阶段预算匹配”，不能声称“总算力/总通信相同”。若需要总预算40的敏感性，
可另设12+28，但必须单列，不能假装第二阶段也有40轮。无须因此重跑全部原四臂；现有结果可复用。
要做总算力优越主张时，则另需合适的同总预算对照。实际积分在修复后benchmark前不作保证。

可复用的是原四臂Formal结果、输入包、split、initial states与评价数据。冻结配置的
`save_rounds=[]`，不能假定已有round12识别checkpoint，更不能从round40权重或日志倒推round12。
若选择T=40且确实找回对应权重、核验来源，也可复用其fit预测来构建错误集；但必须事前冻结这一
设计并将识别训练计入JTT总成本，不能为了复用而声称它符合原论文的早期识别设定。

## 6. 核验与复现

CPU审计脚本/机器证据：

```text
deliverables/cle_jtt_fairness_audit_20261104/audit_jtt_fairness.py
deliverables/cle_jtt_fairness_audit_20261104/AUDIT_EVIDENCE.json
```

审计脚本保留12个输入配置SHA256、三seed冻结trace哈希、最新runner差分、数值反例与fit-only
mask核验结果。它不训练模型、不上传、不调用旧OpenI入口，输出明确为NOT_READY。
现有spurious-baseline五项局部测试另外通过（4.61秒）；其中单batch重复测试通过，仍不足以
证明P1-C中的全局等价。这也是本轮新增数值审计的原因。
旧结果和源码未覆盖。后续实现应新建独立map2/16-batch入口，避免影响当前64-batch实验。

## 7. 原始来源

- [ICML 2021 JTT论文，Section 4 / Algorithm 1](https://proceedings.mlr.press/v139/liu21f/liu21f.pdf)：
  识别模型、固定错误集、第二阶段上采样、验证集组信息用于超参选择。
- [作者官方代码与运行说明](https://github.com/anniesch/jtt)：区分识别checkpoint与第二阶段运行。
- 既有四臂事实：`deliverables/cle_hfl_map2_multiseed_20260922/RESULT_SUMMARY_ZH.md`及本轮读取的冻结配置。

当前论文不能把旧12轮screen适配的负结果推广为“JTT本身无效”，也不能声称最终正式击败JTT。
