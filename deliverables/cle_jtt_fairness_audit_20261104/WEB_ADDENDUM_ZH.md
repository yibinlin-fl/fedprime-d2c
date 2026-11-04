# 网页讨论增量：JTT审计结论（2026-11-04）

本文件补充2026-10-31审稿包，不改其论文PDF、源码或历史结果。

- 当前Axis I正式主表仍为四目标×三training seeds、40轮×16 local batches。
- 仓库确实有旧JTT独立Formal入口；不能说完全没有实现。但它默认原map与seed0、识别阶段和
  最终阶段均40轮，尚未满足当前map2三seed比较及失败恢复要求。
- 旧batch自归一化损失只与batch内部重复的算术一致，一般不等于全数据重采样目标。
  CPU反例已验证；不能将旧12轮screen适配结果推广为JTT本体无效。
- 当前通用Axis I runner已是64-batch版本，不可用它悄悄替换历史16-batch四臂协议。
- 已完成四臂结果不受本次JTT问题影响；本文仍不能声称在最终Formal中战胜JTT。
- 新JTT方案仅为候选：同map2、同最终40×16；早期识别T及lambda须独立冻结，首选固定错误集
  重采样。候选12+40有30%额外local update预算，不可称同总算力/总通信。
- 本轮只做审计、CPU数值/索引验证与文档，没有修改训练实现、没有启动付费任务。

详细报告：docs/research/status/CLE_JTT_FAIRNESS_AUDIT_2026_11_04_ZH.md。
协议草案：docs/research/status/CLE_JTT_MATCHED_PROTOCOL_DRAFT_2026_11_04_ZH.md。
