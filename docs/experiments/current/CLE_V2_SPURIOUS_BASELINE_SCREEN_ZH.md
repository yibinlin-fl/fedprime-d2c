# CLE-HFL v2 Spurious Baseline Screen

Updated: 2026-09-12

状态：实现、本地真实CUDA smoke与20-source分析通过；OpenI benchmark和12轮screen均未授权。

## 目标

回答PEW+BER是否优于通用困难样本/最坏组训练，而不再增加通信模块。五臂固定
`seed0_split0/gamma0.9`、training seed 0、相同初始模型和strict AsymHFL-val通信：

```text
erm
jtt
cvar_dro
pew_groupdro
pew_ber
```

JTT第一阶段就是`erm`臂；错误集合只从四客户端fit样本的最终ERM checkpoint推理产生。stage-2
从共同initial states重新训练，默认错误样本权重20。CVaR默认上尾比例0.2。PEW+GroupDRO默认
指数更新步长0.01，持久维护class x PEW-environment权重。三项超参数在screen前冻结，screen
之后不得根据最终测试结果调参。

## 预算与权限

```text
smoke:     1 round/stage, 1 local batch/client
benchmark: 1 round/stage, 8 local batches/client
screen:    12 rounds/stage, 16 local batches/client/round
```

JTT为两阶段，因此其screen总计12轮ERM发现阶段+12轮重训阶段。Smoke和benchmark不是证据；
12轮screen只决定哪些方法进入最终长程比较，也不是最终论文主表。

## 已验证

```text
JTT/CVaR数学目标单元测试: PASS
PEW+GroupDRO状态更新测试: PASS
五臂真实CUDA训练/checkpoint: PASS
JTT fit-only错误集合: 4/4客户端完整，audit/test标签未读取
paired DSA 20-source分析: PASS
五臂fresh-output local batch trace: MATCHED
```

本地smoke中模型未收敛、DSA接近零，所有数值均禁止作为方法优劣证据。首次smoke因通用日志将
CVaR诊断误读为BER字段而中止；修复为独立`spurious_*`诊断命名空间后通过。随后在全新本地
目录完整重跑，五臂训练、checkpoint、JTT错误集合、20-source DSA和逐batch配对全部通过。

## 入口

```text
scripts/run_cle_v2_spurious_baselines.py
scripts/analyze_cle_v2_spurious_baselines.py
scripts/openi_cle_v2_spurious_baselines_entry.py
tests/test_cle_v2_spurious_baselines.py
```

OpenI的`screen`具有显式`confirm_screen=true`锁。当前没有最终Formal入口；只有screen结果和成本
通过后，才允许单独设计40-pretrain+40-communication或统一40-round最终协议。
