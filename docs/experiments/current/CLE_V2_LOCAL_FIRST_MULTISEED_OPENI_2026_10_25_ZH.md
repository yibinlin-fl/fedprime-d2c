# CLE-v2 Local-first Training-seed 1/2 OpenI指南

状态：2026-10-25实现、聚焦测试与本地四臂真实smoke通过；入口已随`a3b7841`推送至
`origin/main`，尚未确认OpenI任务已启动。

## 1. 科学问题

本实验只检验seed-0的local-first归因是否跨训练随机性稳定，不检验PEW+BER效果。每个seed包含：

```text
h0_b = gamma0   + HFL
h9_b = gamma0.9 + HFL
l0_b = gamma0   + Local
l9_b = gamma0.9 + Local
```

计算`Delta_HFL`、`Delta_Local`与`Delta_comm=Delta_HFL-Delta_Local`。`Local/HFL`只能作为描述性
比例，不能解释为逐样本因果中介份额。

## 2. 与seed-0完全匹配的冻结预算

seed-0已于2026-09-10完成，不需要重跑：12 rounds、16 local batches/client/round、batch size 64、
4 public batches/round、相同partition、initial states、fit/audit/test角色和paired operator grid。
正式结果为：

```text
Delta_HFL   = 0.119889
Delta_Local = 0.107960
Delta_comm  = 0.011929
Local/HFL descriptive ratio = 90.05%
```

旧通用`openi_cle_v2_factorial_entry.py`的Formal没有冻结16-batch上限，曾触发full-fit高成本，不得再
用于M2。唯一有效入口是本指南的专用multiseed入口。

## 3. 输入与代码

```text
Git：a3b7841（已推送至origin/main）
复用数据集：是
OpenI显示名：CLE_v2_Factorial_Seed0_PEW_20260909
本地输入包：C:\Users\asus\Desktop\FedPRIME-D2C\local_runs\cle_v2_factorial\cle_hfl_v2_paired_factorial_seed0_split0_with_pew.tar.gz
大小：694118746 bytes
SHA256：B6C802DA0BC2A95183DB91D2855E35D9A807B39441F2A24CF88514EB1CAD69AA
启动文件：scripts/openi_cle_v2_local_first_multiseed_entry.py
GPU：1 x V100 32GB
```

数据包虽含PEW文件，但四臂配置均强制`pew_used=false, ber_used=false`。

## 4. 推荐运行方式

若有两个账号，分别运行seed1和seed2，速度更快且每个方法完整留在一个任务内：

```text
账号A：
mode = formal
train_seed = 1
confirm_formal = true
data_source = ""
skip_install = false

账号B：
mode = formal
train_seed = 2
confirm_formal = true
data_source = ""
skip_install = false
```

若只有一个账号，可运行：

```text
mode = formal
train_seed = all
confirm_formal = true
data_source = ""
skip_install = false
```

`all`只运行待补seed 1/2，不会重跑seed 0；seed1完成后立即上传checkpoint-bearing recovery包和
最终轻量包，再开始seed2。Formal失败会回传已有recovery/status并保持非零退出。

## 5. 成本

已完成seed-0四臂实测总时间约4小时22分，按6积分/小时约26.2积分。因此：

```text
单个新增seed：预计4.4--5.0 V100小时，约27--30积分
seed1+seed2：预计8.8--10.0 V100小时，约53--60积分
建议安全余额：单seed 35积分；两seed同账号65积分
```

该估计来自完全同预算seed-0 Formal，不需要再次付费benchmark；平台排队、安装和上传不计入训练
计时，仍应保留安全余量。

## 6. 下载

每个seed下载三个文件：

```text
cle_v2_local_first_trainseed{seed}_formal_outputs.tar.gz
cle_v2_local_first_trainseed{seed}_formal_recovery_outputs.tar.gz
RUN_TIMING.json
```

已预建目录：

```text
C:\Users\asus\Desktop\FedPRIME-D2C\outputs\openi_downloads\m2_local_first_multiseed\trainseed1
C:\Users\asus\Desktop\FedPRIME-D2C\outputs\openi_downloads\m2_local_first_multiseed\trainseed2
```

## 7. 允许证明与禁止表述

若seeds 1/2方向一致，可支持“local-first attribution is stable across three matched training seeds”。
它仍不覆盖partition、binding-map population、数据集或现实部署不确定性。不得把source bootstrap
当作training-seed CI，也不得把90.05%写成严格因果中介比例。
