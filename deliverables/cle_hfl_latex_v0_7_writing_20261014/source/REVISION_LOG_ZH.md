# V0.7-writing修改日志

日期：2026-10-14。

## 本轮完成

1. 重写Abstract，删除把pending结果写成既成事实的句子。
2. 重构Introduction为“问题—诊断—归因—缓解—边界”四项贡献，不把PEW分类器单独包装为创新。
3. 重写Related Work，主动承认EIIL、GroupDRO、JTT、FedPIN、FedCD及反事实shortcut评价的先例，并收窄差异点。
4. 在Method中写清strong/moderate双职责协议、PEW监督来源、DSA假设与proxy不可识别边界。
5. 在Experiments中区分已完成固定预算Axis I与pending submission-budget Axis I；回填moderate Axis II前三行Formal。
6. 重写Discussion和Conclusion，明确local-first仍属于HFL机制问题，同时保留taxonomy、外部有效性与utility边界。
7. 新增`CLAIM_EVIDENCE_AUDIT_ZH.md`，锁定允许、条件性、pending和禁止表述。
8. 更新引用审计与实验表矩阵。

## 未做

- 未改动冻结V0.7源码与zip。
- 未伪造或推断任何未完成实验结果。
- 未进行目标会议模板压缩。
- 未用语言风格修改覆盖科学边界。

## 下一轮写作入口

等待新Formal结果时，可继续做三项低风险工作：

1. 逐段删除重复解释和模板化过渡句；
2. 核对每个citation key与原文；
3. 根据目标会议页数压缩正文，并把证明、审计和细粒度表移到附录。

新结果到达后，应先更新论断台账和实验矩阵，再改正文；不要直接在Abstract里填数字。
