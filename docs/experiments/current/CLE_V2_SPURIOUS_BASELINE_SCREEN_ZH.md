# CLE-HFL v2 Spurious Baseline Screen

Updated: 2026-09-13

状态：OpenI 12轮五臂screen已完成并独立读取结果；它只用于选择最终四臂，不是最终论文证据。

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

## 12轮Screen结果

五臂输入、初始化、strict AsymHFL-val通信、single-view训练、fit/audit/test角色和逐batch轨迹
匹配。完整评价结果为：

| Arm | DSA | Operator-grid Acc | Last-5 Avg | Last-5 Worst | WCCA | CFG |
|---|---:|---:|---:|---:|---:|---:|
| ERM | 0.214812 | 18.4183 | 17.8240 | 15.5213 | 0.0000 | 35.5800 |
| JTT | 0.046833 | 15.2617 | 16.0470 | 12.2240 | 0.4000 | 19.3500 |
| CVaR-DRO | **0.034980** | 16.5917 | 16.6053 | 12.9427 | 0.1000 | 25.6350 |
| PEW+GroupDRO | 0.177739 | 19.5450 | 19.2727 | 14.9640 | 0.6000 | 32.1150 |
| PEW+BER | 0.037889 | **21.1267** | **19.3430** | **16.4133** | **0.7000** | **19.5450** |

关键解释：

- `PEW+BER vs ERM`：DSA降低`0.176922`（82.36%），operator-grid accuracy提高`2.7083pp`；
- `CVaR vs PEW+BER`：CVaR的DSA低`0.002909`，但PEW+BER的operator-grid accuracy高
  `4.5350pp`、Avg高`2.7377pp`、Worst高`3.4707pp`、CFG低`6.0900pp`，两者是
  shortcut—utility取舍，不能写成任一方全面支配；
- `PEW+BER vs PEW+GroupDRO`：共享相同PEW分组时，BER的DSA低`0.139850`且
  operator-grid accuracy高`1.5817pp`，说明收益不只是“有了伪环境组”；
- JTT显著降低DSA但任务效用最弱，未晋级最终长程比较。

最终晋级：`ERM/CVaR-DRO/PEW+GroupDRO/PEW+BER`。最终协议改用筛选未见的binding map2并
训练40轮；见`CLE_V2_SPURIOUS_FINAL_MAP2_ZH.md`。本表必须标为selection screen，不进入论文
最终主结果表冒充Formal。

实测V100训练`3559.602s`、完整分析`88.887s`，总计约60.81分钟。

## 入口

```text
scripts/run_cle_v2_spurious_baselines.py
scripts/analyze_cle_v2_spurious_baselines.py
scripts/openi_cle_v2_spurious_baselines_entry.py
tests/test_cle_v2_spurious_baselines.py
```

OpenI的`screen`具有显式`confirm_screen=true`锁。screen已完成，不得根据结果修改五臂超参数或
把12轮结果升级为论文Formal。40轮held-out map2四臂协议已在结果前冻结并于2026-09-13启动。
