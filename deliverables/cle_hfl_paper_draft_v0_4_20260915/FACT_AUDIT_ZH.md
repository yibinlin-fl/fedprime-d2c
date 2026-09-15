# CLE-HFL英文会议压缩稿V0.4事实审计

审计日期：2026-09-15。

## 总结

网页端返回的V0.4通过本轮仓库事实审计，可以作为新的**会议压缩内部主稿**，但还不是最终投稿稿。它把V0.3从约9500词压缩到约5165词，同时保留了CLE-HFL问题、paired DSA、local-first归因、family-level PEW+BER、matched controls及理论边界。

## 已通过检查

1. CLE-v2构造和DSA评价仍为operator-level；PEW+BER仍为coarse family-level。
2. Oracle operator只被描述为不可部署边界消融，没有被重新包装成候选PEW。
3. CDep没有进入当前方法、公式或结果主张。
4. FedDF-fidelity保留“协议匹配通信底座对照”边界，且明确记录utility gate失败与learning floor问题。
5. CVaR-DRO保留“协议匹配尾部风险对照”边界；正文只声称map2 pooled trade-off更好，没有声称逐客户端统一支配。
6. JTT仍为12轮screen-only，没有虚构Formal胜负。
7. source bootstrap与training-seed uncertainty严格分开。
8. map2主表、local-first表、map1、FedDF、Oracle、BER审计及proxy误差数字均能在仓库冻结证据中定位。
9. 18个真实citation key均能在相邻`references.bib`解析；没有`[REF TO VERIFY]`。
10. 待补实验均保留为`[EVIDENCE NEEDED]`，且明确未授权、未运行。

## 本轮直接修复

1. Abstract增加`Completed seed-0 controlled comparisons`，避免把现有受控结果误读为已经完成multi-training-seed验证。
2. 将`private-metadata-free`收窄为`private-environment-metadata-free`。PEW+BER仍使用private task labels，因此前一种写法过宽。
3. 删除文末会被自动引用检查误识别为真实key的示例`\citep{key}`，保留natbib说明。

## 当前仍不能宣称

- 不能宣称training-seed stability已经建立；
- 不能宣称跨partition或第二private dataset泛化；
- 不能宣称taxonomy-free或arbitrary open-world robustness；
- 不能宣称BER保证真实环境与类别独立；
- 不能宣称FedDF上的utility成功；
- 不能宣称逐客户端全面战胜CVaR-DRO；
- 不能宣称正式战胜JTT；
- 不能把90.05%写成因果中介比例。

## 下一步冻结顺序

1. 先将V0.4作为写作主稿，V0.3继续保留为完整证据长稿；
2. 单独冻结map2四臂40轮S1/S2协议并完成成本审计，获得用户明确授权后才能运行；
3. 之后再评估HFL-vs-Local S1/S2；
4. 第二private dataset与bounded taxonomy stress test仍需先设计协议，不能直接冻结为SVHN或leave-one-operator-out；
5. 不增加新loss、通信模块、hierarchical PEW或operator-level PEW。
