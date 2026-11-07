# 主评价八行领域表：剩余三行OpenI启动卡

日期：2026-11-04。当前表已完成5/8行。本文件准备FedDF、AugHFL和RAHFL三个独立单臂Formal，
不更改算法、冻结协议或科学结果；不代用户创建付费任务。

2026-11-07增量：FedDF已验收，表完成6/8行，实测训练与分析合计约5h23m，按此前费率折算32.29积分。
AugHFL/RAHFL将使用20/30/40轮中途回传的新版本，启动请改用
`CLE_ROBUST_ROUND_RECOVERY_OPENI_2026_11_07_ZH.md`；该新代码本地已验证，提交d51e438已推送。

## 1. 八行表的准确状态

| 行 | 方法 | 主评价Formal状态 |
|---|---|---|
| 1 | Local/ERM | 已完成 |
| 2 | FedMD | 已完成 |
| 3 | FedProto | 已完成 |
| 4 | FedDF | 未完成可用结果；旧任务训练40轮完成，但旧分析/回传失败，权重未取回 |
| 5 | AugHFL | benchmark通过，Formal未完成 |
| 6 | RAHFL | benchmark通过，Formal未完成 |
| 7 | AsymHFL--ERM | 已完成 |
| 8 | AsymHFL+PEW+BER | 已完成 |

三项待跑都是基线自身，不叠加PEW+BER。它们提供HFL领域位置，结合第7/8行比较DSA与任务效用。
BER的单因素归因仍由同通信四目标Axis I与匹配两臂承担。旧机制协议上的FedDF插件A/B已经完成，
不能当作本表的主评价FedDF行；两者数据协议不同。

## 2. 代码与验证（实际核验）

- GitHub `refs/heads/main`已通过`git ls-remote`只读确认：
  `32f4050c61861ae072ab00f725f5b40ecf88bbee`。
- OpenI项目选择现有`fedprime-d2c`、分支`main`；本卡的实验代码已在远端，无须为了运行而推送论文。
- 当前本地较新提交主要为图/文档；`scripts/`、`fedprime/`、`configs/`和`requirements.txt`
  相对于远端实验版本无差异。入口已包含d8974be的Formal单臂及上传context检查。
- 11项配置/回传/失败标记检查本轮通过，耗时13.56秒；没有运行本地训练。
- 既有真实V100 benchmark：普通四基线及AugHFL/RAHFL均已通过。此次不用重复相同benchmark。
- 冻结本地输入包本轮重新计算bytes和SHA256通过。

启动文件（这三项完全相同）：

```text
scripts/openi_cle_v2_moderate_entry.py
```

禁止使用FedMD四目标入口、旧`openi_cle_hfl_context_entry.py`或CIFAR-100入口启动本表。

## 3. 输入数据（这三项完全相同）

复用已上传数据集，不需要新建或上传。

```text
显示名称：CLE_v2_Moderate_Alpha10_Rho05_Map2_Seed0_PEW_20261013
owner/repo_id：chujiu/CLE_v2_Moderate_Alpha10_Rho05_Map2_Seed0_PEW_20261013
远端文件：cle_hfl_v2_moderate_alpha10_rho05_map2_seed0_split0_with_pew.tar.gz
Windows绝对路径：C:\Users\asus\Desktop\FedPRIME-D2C\local_runs\cle_v2_moderate_alpha10_rho05\cle_hfl_v2_moderate_alpha10_rho05_map2_seed0_split0_with_pew.tar.gz
精确大小：893421220 bytes
SHA256：DEA800EC1465C45037E74E4B948DFB8D778F1354E8522B551438C6A515C9E588
包内根目录：cle_hfl_v2_moderate_alpha10_rho05_map2_seed0_split0
```

本表固定CIFAR-10 private、CIFAR-100 public、map2、alpha1、rho.5、gamma.9、partition/train seed0。
不要使用1.3GB cross-binding maps旧包、694118746-byte factorial机制包或681493067-byte
CIFAR-100-private第二任务包。数据集名称中的Moderate是历史工程名；论文称main evaluation protocol。

## 4. 环境与共同训练预算

```text
GPU：1×V100 32GB（CPU 8、内存50GB，与已完成任务规格一致）
镜像：ubuntu22.04-cuda11.8-py310-torch2.1.0-tf2.14.0
访问Internet：是（安装已有requirements）
任务性质：Formal

pretrain_epochs = 0
rounds = 40
local_epochs = 1
max_local_batches = 64
batch_size = 64
public_batch_size = 128
public_batches_per_round = 4
train_seed = 0
skip_nonfinite = false
```

这些是内部冻结配置，不要作为额外网页参数添加。该统一预算表不是RAHFL官方40预训练+40通信榜单。
FedDF使用standard单视图；AugHFL/RAHFL保留本身的AugMix/JSD等路径，二者不是一个任务。
入口没有`train_seed`、`arm`、`rounds`、`batch_size`网页参数；不要另填，以免unrecognized arguments。

## 5. 三个任务的照填参数

| 参数名 | FedDF | AugHFL | RAHFL |
|---|---|---|---|
| `mode` | `formal` | `formal` | `formal` |
| `shard` | `feddf_fidelity` | `aughfl_fidelity` | `rahfl_fidelity` |
| `confirm_formal` | `true` | `true` | `true` |
| `skip_install` | `false` | `false` | `false` |
| `data_source` | 空字符串 | 空字符串 | 空字符串 |

`data_source`建议直接不添加这一行，精确生效值为`""`；入口通过c2net自动定位挂载。
不要填写Windows路径、数据集显示名称、`None`、`null`或带引号的`""`字样。
`skip_install=false`为冻结默认值，可不添加；显式添加时填`false`。

任务名及完整产物、下载目录：

### FedDF单臂

```text
任务名：cle-main-feddf-s0-20261104
shard：feddf_fidelity
正式包：cle_v2_moderate_feddf_fidelity_formal_outputs.tar.gz
恢复包：cle_v2_moderate_feddf_fidelity_formal_recovery_outputs.tar.gz
时间/状态：RUN_TIMING.json
日志：下载该任务完整日志.txt
下载目录：C:\Users\asus\Desktop\FedPRIME-D2C\outputs\openi_downloads\main_axis2_feddf_formal_s0_20261104
```

### AugHFL单臂

```text
任务名：cle-main-aughfl-s0-20261104
shard：aughfl_fidelity
正式包：cle_v2_moderate_aughfl_fidelity_formal_outputs.tar.gz
恢复包：cle_v2_moderate_aughfl_fidelity_formal_recovery_outputs.tar.gz
时间/状态：RUN_TIMING.json
日志：下载该任务完整日志.txt
下载目录：C:\Users\asus\Desktop\FedPRIME-D2C\outputs\openi_downloads\main_axis2_aughfl_formal_s0_20261104
```

### RAHFL单臂

```text
任务名：cle-main-rahfl-s0-20261104
shard：rahfl_fidelity
正式包：cle_v2_moderate_rahfl_fidelity_formal_outputs.tar.gz
恢复包：cle_v2_moderate_rahfl_fidelity_formal_recovery_outputs.tar.gz
时间/状态：RUN_TIMING.json
日志：下载该任务完整日志.txt
下载目录：C:\Users\asus\Desktop\FedPRIME-D2C\outputs\openi_downloads\main_axis2_rahfl_formal_s0_20261104
```

以上三个本地目录均已创建。不同任务的RUN_TIMING.json分别保存，不相互覆盖。

## 6. 成本与当前账号分配

按用户此前V100费率6积分/小时，100积分对应16.67小时；实际费率与余额以创建页为准。

| 单臂 | 预算依据 | 当前建议 |
|---|---|---|
| FedDF | 历史失败训练与保守启动记录，约100积分规划额；无新成功端到端实测耗时 | 最多余额账号优先；100积分刚到保守下限，不承诺一定足够 |
| AugHFL | 既有robust benchmark机械估算约17小时/102积分，尚非Formal实测 | 至少110--120积分，建议120再开 |
| RAHFL | 同一robust benchmark平均外推及后续保守规划 | 至少110--120积分，建议120再开 |

不使用已经被Formal否定的普通基线“每臂3.3小时”乐观外推。AugHFL/RAHFL约17小时也只是
历史benchmark规划估计，包含全部分析/上传的实际耗时仍需回传后核验。若担心余额不足，可等到
建议余量再开；不能靠修改batch/轮数或拼接两个账号轮次去挤预算。第四号暂作预算备用，
不是这三项必须新增的第四个任务。用户如果改变启动日期，任务名可改日期，下载目录保持一一对应。

## 7. 结果合同与科学用途

入口在训练后、正式分析前先上传含checkpoint的恢复包；之后分析用`auto`选择设备。
这保护已完成训练免于分析失败丢失，但平台若在训练阶段硬销毁容器，Python finally不能保证回传。
每臂单独运行，Formal不得选择`axis2_all/axis2_robust/axis2_plain`。

收到结果后核对：40轮×64预算、seed0/map2/输入身份、4客户端checkpoint、完成状态、
训练loss有限、batch trace、统一预测缓存/分析与既有五行的协议匹配。
主要指标为pooled/client DSA、operator-grid/clean/corrupted-balanced/seen/unseen/mixed accuracy，
并报告shuffled-binding null及source级区间。无新增的事后DSA胜负或accuracy下限门槛。
科学结论由完整结果决定，不以任务SUCCEEDED或benchmark数值替代。

该表能比较固定主评价协议下HFL方法的shortcut行为与效用；不能单凭seed0领域表证明
跨数据集泛化、任意HFL通用插件、官方原文性能、训练seed稳定性或BER单因素因果优势。
