# V0.10修改日志

## 2026-10-27 V0.10协议命名收敛

- 从V0.9不可变快照派生。
- 正文统一使用`mechanism protocol`与`main evaluation protocol`。
- `alpha/rho/gamma`差异只在Problem Setup与Experiments协议段集中解释。
- 历史脚本、目录、文件名中的`strong`/`moderate`工程标识不改。
- 未改变任何实验数字、公式、引用键、表格行或科学结论。

## 2026-10-26 Local-first Multi-seed回填

- 从V0.8不可变快照派生V0.9；未覆盖V0.8。
- 回填training seeds 1/2四臂Formal与三seed mean/sample-SD。
- 正文按DSA主终点报告3/3机制门槛复现；辅助学习下限保留在内部正式审计报告。
- 更新Abstract、Introduction、Method、Experiments、Discussion与Conclusion，禁止“三seed完整门槛全过”。
- 将local-first图由seed-0堆叠分解改为三seed effect与communication-difference图。
- 未改动CLE问题定义、DSA/BER公式、PEW粒度、moderate Formal数字或引用键。

基线写作审计：2026-10-14；Moderate Formal回填：2026-10-25。

## 2026-10-25 Moderate Formal回填

- 回填gamma 0/.5/.9 ERM与gamma .9 PEW+BER的seed-0 dose-response表。
- 回填Axis II两条AsymHFL行，完成行由3/8增至5/8。
- 正文加入source-paired bootstrap、4/4客户端DSA下降及clean/mixed utility代价。
- 未修改问题定义、DSA/BER公式、理论命题或引用键；FedDF/AugHFL/RAHFL继续保留pending。
- 原始数字来源：`deliverables/cle_v2_moderate_axis2_dose_formal_seed0_20261025/RESULT_SUMMARY_ZH.md`。

## 本轮完成

1. 重写Abstract，删除把pending结果写成既成事实的句子。
2. 重构Introduction为“问题—诊断—归因—缓解—边界”四项贡献，不把PEW分类器单独包装为创新。
3. 重写Related Work，主动承认EIIL、GroupDRO、JTT、FedPIN、FedCD及反事实shortcut评价的先例，并收窄差异点。
4. 在Method中写清strong/moderate双职责协议、PEW监督来源、DSA假设与proxy不可识别边界。
5. 在Experiments中区分已完成固定预算Axis I与moderate Axis II；Axis II目前回填五行Formal。
6. 重写Discussion和Conclusion，明确local-first仍属于HFL机制问题，同时保留taxonomy、外部有效性与utility边界。
7. 新增`CLAIM_EVIDENCE_AUDIT_ZH.md`，锁定允许、条件性、pending和禁止表述。
8. 更新引用审计与实验表矩阵。
9. 重绘五张可复现科研图：证据链、DSA/代理不可识别边界、local-first归因、Axis-I双端点比较和双协议职责图。
10. 新增RQ1--RQ7实验问题表，使每个问题、主要端点和证据状态一一对应。
11. 完成第二轮语言审计，去除模板化转折、重复总结和不必要的防御性措辞，同时保持科学限制不变。
12. 逐篇核对20个引用键的官方或一手来源，并修正反事实医学论文的会议届次。
13. 对编译图执行静态完整性检查：输入文件、引用键、图片、标签、交叉引用和环境配对均无缺失。

## 编译与外部验证

- 未改动冻结V0.7源码与zip。
- 未伪造或推断任何未完成实验结果。
- 未进行目标会议模板压缩。
- 2026-10-25内置LaTeX编译已成功；仍需在Overleaf检查浮动体、跨栏图、断行、引用和最终页数。
- 未用语言风格修改覆盖科学边界。

## 下一轮写作入口

等待新Formal结果时，可继续做两项低风险工作：

1. 在Overleaf完成整稿编译与逐页视觉检查；
2. 根据目标会议页数压缩正文，并把证明、审计和细粒度表移到附录。

新结果到达后，应先更新论断台账和实验矩阵，再改正文；不要直接在Abstract里填数字。
