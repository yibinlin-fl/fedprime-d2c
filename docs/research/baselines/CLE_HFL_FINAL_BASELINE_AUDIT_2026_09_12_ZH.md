# CLE-HFL最终基线审计

Updated: 2026-09-12

## 忠实度修复的含义

“fidelity repair”不是增强基线，也不是给基线调参。它表示早期统一runner只复现了方法的大致
接口，后来依据原论文或发布代码修正了核心时序/目标偏差，同时保留旧适配器和旧结果以便追溯。
当前明确修复的只有：

```text
AugHFL: 独立participant公共增强、发布归一化、协作Adam时序
FedDF: 先local后server fusion、冻结教师快照、avg-logit forward-KL
KT-pFL: 先模型蒸馏后系数更新、行随机系数、正确的正则与样本量权重
```

修复版仍然运行在统一CLE-HFL数据、四架构和公共预算下，必须称为
`protocol-matched fidelity adapter`，不能称为逐行复现原论文完整recipe。

## 九种HFL基线的最终口径

| 基线 | 仓库状态 | 最终表口径 | 是否需要最终重跑 |
|---|---|---|---|
| Local/ERM | 已实现 | 直接基准/归因 | 是 |
| FedMD | 已实现核心机制 | protocol-matched | 是 |
| RHFL | 已实现核心机制 | protocol-matched | 是 |
| FedProto | 已实现核心机制 | protocol-matched | 是 |
| AugHFL | 有fidelity版 | 只使用`aughfl_fidelity` | 是 |
| FedDF | 有fidelity版 | 只使用`feddf_fidelity` | 是 |
| KT-pFL | 有fidelity版 | 只使用`kt_pfl_fidelity` | 是 |
| FCCL | 已实现公开互相关核心 | core adapter | 是 |
| RAHFL | strict AsymHFL-val已实现 | matched CLE protocol | 是 |

历史12轮表只能用于筛选和恢复旧结论，不能替代最终长程主表。早期`aughfl/feddf/kt_pfl`适配器
的数字不得与修复版混合。

## 两类表必须分开

主比较表比较上述九个基线与最终方法。插件迁移表只选通信机制差异明显的代表性底座：RAHFL-like、
FedDF-fidelity、KT-pFL-fidelity、FCCL-core。无需把插件机械安装到所有九个方法上。

## corruption/spurious相关对照

本地目标层新增五臂统一协议：

```text
ERM
JTT：stage-1 ERM的fit错误样本，stage-2从相同初始状态重训并上权
CVaR-DRO：不使用环境标签，只优化batch损失上尾
PEW+GroupDRO：使用冻结PEW组，指数梯度更新class x environment最坏组权重
PEW+BER：使用相同冻结PEW组与当前BER
```

JTT和CVaR-DRO配置不加载PEW。PEW+GroupDRO不调用BER公式。所有臂使用相同数据、初始模型、
single-view输入、strict fit/audit/test角色和AsymHFL-val通信；AugMix/JSD/DCL/CDep均禁用。

## 轮数纪律

12轮肯定不足以作为最终论文性能。它只负责筛选明显失败的方法、测算JTT双阶段成本并观察是否
形成可识别DSA。最终协议必须在筛选结果与benchmark之后另行冻结：至少匹配40轮通信；若主张
RAHFL完整训练性能，则应使用共享的40轮预训练起点再做40轮通信，或者把“统一40轮matched
budget”和“各论文原recipe”分表报告。不得用不同预训练预算制造不公平优势。

JTT天然包含stage-1和stage-2两次训练，最终表必须额外报告总训练量；不能把它的双倍计算隐藏在
同一个“40轮”标签里。
