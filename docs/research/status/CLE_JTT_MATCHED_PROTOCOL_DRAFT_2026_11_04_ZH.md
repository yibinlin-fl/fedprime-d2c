# JTT补充比较的公平协议草案

日期：2026-11-04。状态：审计后候选；不是已实现启动协议，不授权训练或上传。
配套审计：`CLE_JTT_FAIRNESS_AUDIT_2026_11_04_ZH.md`。

## 1. 为什么需要这一补充

检验BER的收益能否由先识别ERM错误、再增加错误样本训练贡献来解释。
JTT与CVaR的区别在于错误集固定于一个早期参考模型，不随最终模型每步重新选择。
当前四臂主表保持原样；这一补充使用相同机制协议，不与主评价64-batch领域表混合。

## 2. 必须继承的条件

| 项目 | 与原四臂对齐的条件 |
|---|---|
| 数据 | 相同cross-map2 input、private/public source、alpha=.5/rho1/gamma.9 |
| 划分 | partition0、完全相同fit/audit/test及私有样本顺序身份 |
| 模型 | 原四个架构，同一initial_states的逐文件hash |
| 随机性 | 同seed配对0/1/2；相同loader/private RNG规则；重采样独立generator |
| 最终训练 | 40轮、local_epochs1、每client每轮16个local batches、batch64 |
| 优化器 | Adam lr.001、wd0；全新优化器，grad clip5 |
| 公共通信 | AsymHFL-val，public batch128，每轮4 batches；同一公共batch流 |
| 增强/模块 | standard单视图，lambda_jsd0，无DCL/CDep/PEW/BER |
| 评价 | 同1000-source×15-operator网格、labels、binding、推理预处理及统一汇总 |
| 条件性范围 | 同固定partition、map2，不能称第二数据集或新的held-out选择验证 |

保留冻结配置的非有限处理身份并审计实际处理；不要默认为当前runner的64-batch/false设置。
standard单视图仍保留原基础增强，不能擅自删掉ColorJitter/裁剪/翻转。fit-only错误识别不等于
整个AsymHFL不使用audit标签；路由仍遵循原协议，真实环境标签不参与错误集合/超参选择。

## 3. 两阶段与信息边界

1. 识别阶段：相同data/模型/初始化，普通ERM；建议候选T=12轮×16。
   HFL适配建议保留相同AsymHFL通信，不把无通信识别与有通信识别混成一组。
2. 在完整fit上用确定性推理生成二元错误集合。audit/test不参与集合构造。
3. 独立重建最终模型，重新载入与原四臂相同的初始权重，新建优化器并重置随机流。
4. 最终40轮使用固定错误集，候选lambda=20；不动态更新集合。

T=12和lambda20只是从旧screen迁移的候选固定设置。不得根据map2最终DSA、准确率、
未来JTT结果或真实operator标签挑超参。若采用验证集选择，需要先另行记录允许的数据角色、
选择指标、候选集合和各方法选择预算；不能冒称复现原论文的group-labelled validation流程。
错误集为空/全体时需明确记录退化，不伪造环境分组或事后调T到有利结果。

## 4. 首选：原理对应的重采样适配

对每个client的fit集合固定权重：错误样本lambda，其余1。按照归一化权重分布采样，
用普通mean CE更新。每轮保持原预算的16个batch；公开说明固定步数HFL适配与原论文完整
upsampled epoch的差别，不宣称逐行官方复现。

采样造成私有batch不同是算法的一部分。因此验收应从“图像trace与ERM完全相同”改为：

- sample IDs全部属于同一fit，无audit/test；
- stage2四client更新数量与baseline相同；
- 同seed重跑的采样序列可复现，采样generator不改变public batch流；
- 记录每样本权重、全fit权重和、总draw数、错误/非错误draw统计与实际采样hash；
- 不能用一个固定小样本误差阈值强制采样频率精确等于期望，随机抽样仍有波动。

如改用相同batch的固定全fit归一化加权方案，应写JTT-style reweighted adaptation，并单独说明
它不改变BatchNorm的输入组成；不可把旧batch自归一化保留而宣称已解决全局目标差异。

## 5. 两种公平性要分清

主建议是匹配最终阶段：识别T+最终40，因此候选12+40共52轮预算。原四臂无需重跑，
但JTT的识别成本和额外通信必须报告，不能声称相同总compute。
若另做同总40预算敏感性，可采用12+28并独立列明，不能与40轮最终阶段混写。

三seed主表只能追加三个实际JTT seed结果。先seed0时只与已有四臂seed0列作比较，
保留单seed/source-bootstrap边界，不能与原mean±SD直接作胜负判断。

## 6. 实现和晋级门槛

```text
独立map2/16-batch配置生成与identity audit
→ 错误集全fit覆盖与来源hash核验
→ 二阶段初始化/optimizer/RNG重置检查
→ resampler与公共流隔离检查
→ 分阶段recovery回传/失败不冒充科学证据
→ 小型本地smoke（只验证执行）
→ 单seed benchmark（只验证成本）
→ 用户明确授权后才可Formal
```

最终分析至少核对：输入与配置身份、两阶段完成标记、4模型权重可用、预测有限、评价source/binding
一致、pooled/client DSA、grid accuracy、同seed对照与source级区间。保留现有其他统一指标。
不得为JTT事后发明有利的DSA胜负门槛，也不继承BER专属promotion gate给JTT定科学成败。

旧`openi_cle_jtt_formal_entry.py`目前不满足本草案，暂不提供Formal启动参数。
