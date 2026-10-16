# FedPRIME-D2C：OpenI实验操作与跨窗口完整交接

更新：2026-10-17（Asia/Shanghai）。仓库：`C:\Users\asus\Desktop\FedPRIME-D2C`。

本文集中记录本项目已经使用的OpenI数据上传、任务配置、输入选择、恢复包、结果下载与审计流程。
接手窗口先读根目录`AGENTS.md`、`docs/handoffs/latest.md`和`docs/README_ZH.md`，再读本文。
不需要为了恢复OpenI操作而通读大型历史日志。本文不是新实验授权，也不是可以直接执行的Formal启动卡。

跨项目通用Skill已安装为`C:\Users\asus\.codex\skills\openi-experiments`，可说“我要在OpenI跑实验”
或显式调用`$openi-experiments`。可维护源码：`deliverables/openi_experiments_skill_20261017/openi-experiments/`。
新课题使用其通用规范，不复制本文的FedPRIME数据包、预算与科学结论。

## 1. 接手时先恢复的最新决策

1. 论文主线是CLE-HFL的问题发现、paired DSA、local-first归因、taxonomy-assisted PEW+BER缓解、
   跨通信验证、领域比较及边界。不能写成通用无损插件或taxonomy-free方法。
2. 2026-10-15优先级已调整：已有M1是40轮、四臂、training seeds 0/1/2的matched正式证据，
   其原协议为16 local batches/client/round，必须如实披露。后来中断的64-batch高预算复验是
   可选耐久性检查，不再默认优先重跑。两套预算不得混表或互相冒充。
3. 下一科学优先项是moderate `asymhfl_erm`与`asymhfl_pew_ber`成对锚点；随后复用gamma=.9
   结果，只新增`erm_g00`与`erm_g05`完成剂量响应。复用前仍须核对实际配置、轨迹及评价合同。
4. moderate Local/FedMD/FedProto的40轮seed-0 Formal已完成。FedDF、两条AsymHFL先前的
   失败任务没有形成可用正式结果；不能把“完成训练的日志”当成完整科学证据。
5. 八行Axis II全满、JTT Formal及高预算Axis I复验都不是当前强制前置项。完整八行表未完成，
   不得提前写完整排名。第二private dataset、local-first多seed和taxonomy stress按预算与授权决定。
6. 原冻结V0.7不覆盖；继续写作使用独立V0.7-writing目录。实验补齐后再决定V0.8。
7. 没有在本文中授权付费、长时、Formal、多种子任务。接手窗口只能先检查、准备和给出完整启动卡。

详细优先级：`docs/experiments/current/CLE_HFL_PENDING_EXPERIMENT_PRIORITY_2026_10_15_ZH.md`。
若后续handoff有更新，以更晚的明确决策为准。

## 2. 本地代码与远端代码状态

编写本文前的本地基线：`32c7040552b6ad16647e06effa4b5b551a67f4bc`，分支`main`。
本地记录的`origin/main`为`0ed99a66fbf3258db9d599ffb04ab43ffed84233`，当时本地领先10个提交。
这是本地tracking ref快照，本轮没有fetch或push，不能代替远端实时检查；本文后续提交也不计入该数字。

关键历史提交：

| 提交 | 内容 | 编写前本地记录的推送状态 |
|---|---|---|
| `0ab080c` | moderate训练后恢复包、auto分析及失败状态 | 已包含在本地origin/main记录中 |
| `3ed6210` | Axis I中断事故记录 | 尚未包含在本地origin/main记录中 |
| `c083aab` | Axis I逐臂恢复包、Formal单arm单seed保护 | 同上 |
| `2a6414b` | Axis I上传上下文启动前检查及故障回归测试 | 同上 |

启动前重新检查：

```powershell
Set-Location 'C:\Users\asus\Desktop\FedPRIME-D2C'
git status --short
git rev-parse HEAD
git log --oneline '@{upstream}..HEAD'
```

只能提交当前任务拥有的文件，不能`git add .`把无关dirty修改带入提交。不得覆盖、回退或清理
其他窗口的修改。推送需要用户明确要求；“提交”默认不等于“推送”。OpenI选中的远端commit必须
包含本次准备使用的入口修复。不能只改本机代码，却让OpenI继续拉旧代码。

## 3. 两种上传上下文：不能混淆

| 场景 | 凭据/上下文 | 用途 |
|---|---|---|
| Windows本机上传输入数据 | OpenI CLI及本机登录凭据 | 把tar.gz上传到OpenI数据集 |
| OpenI训练容器回传结果 | `c2net.context.prepare()`得到的context | 找到数据挂载目录、输出目录并调用`upload_output()` |

Windows已安装的CLI路径：

```text
D:\anaconda3\envs\pytorch\Scripts\openi.exe
```

本项目已使用版本为`openi==3.0.1`。凭据位置只记录路径：

```text
C:\Users\asus\.openi\token.json
```

2026-10-14本窗口只读认证检查返回账号`chujiu`且认证成功；2026-10-17编写本文时再次确认CLI与
凭据文件存在，但没有重新进行在线认证。后续窗口按需运行`whoami`，不能把旧认证结果当成永久有效。
不要输出或复制token.json内容。不要把Token放入聊天、启动卡、仓库、配置、日志或Git历史。

```powershell
& 'D:\anaconda3\envs\pytorch\Scripts\openi.exe' whoami
```

若确实失效，由用户在`https://openi.pcl.ac.cn/user/settings/applications`管理凭据，并在本机交互登录。
历史曾出现过Token明文暴露，不能找回或复用聊天里的旧Token。已有有效凭据时不重复登录、不升级CLI。

本机Token不能生成训练容器的c2net上下文。缺少c2net时，不能通过删除上传检查或伪造context让
长训练继续运行。`requirements.txt`没有显式安装c2net；平台镜像必须实际提供可用的c2net环境。

## 4. 输入包：moderate与strong严格分开

### 4.1 当前优先：moderate

```text
repo_id: chujiu/CLE_v2_Moderate_Alpha10_Rho05_Map2_Seed0_PEW_20261013
display name: CLE_v2_Moderate_Alpha10_Rho05_Map2_Seed0_PEW_20261013
remote filename: cle_hfl_v2_moderate_alpha10_rho05_map2_seed0_split0_with_pew.tar.gz
local file: C:\Users\asus\Desktop\FedPRIME-D2C\local_runs\cle_v2_moderate_alpha10_rho05\cle_hfl_v2_moderate_alpha10_rho05_map2_seed0_split0_with_pew.tar.gz
bytes: 893421220
SHA256: DEA800EC1465C45037E74E4B948DFB8D778F1354E8522B551438C6A515C9E588
extracted package: cle_hfl_v2_moderate_alpha10_rho05_map2_seed0_split0
```

该包已于2026-10-13由本机CLI上传成功，记录为100%、852.0 MiB、exit code 0。
2026-10-17再次实测本地bytes与SHA256，均与上面一致；本轮未重新审计远端文件内容。
优先复用现有公开数据集，无需因为改入口代码而重新上传未变化的数据。

冻结场景：`alpha=1.0 / rho=0.5 / map2 / partition seed0 / split0 / training seed0`，
包含`gamma=0/0.5/0.9`、四客户端私有数据、strict fit/audit、公共CIFAR-100、初始化、PEW
annotations与1000-source × 15-operator DSA网格。moderate仍是CIFAR-10 private，不算第二数据集。

### 4.2 strong / Axis I高预算可选复验

```text
historically used display name: CLE_v2_CrossBindingMaps_Seed0_PEW_20260911
local file: C:\Users\asus\Desktop\FedPRIME-D2C\local_runs\cle_v2_cross_scenario\cle_hfl_v2_cross_maps1_2_seed0_split0_with_pew.tar.gz
bytes: 1385820059
SHA256: BEA8E98737BF881C701DCFFF05F4E04C3A1E6095B7CF7702A5177260C2F186F5
Axis I extracted subpackage: cle_hfl_v2_cross_map2_seed0_split0
```

2026-10-17本地bytes与SHA256再次核对一致。上述是旧指南记录的数据集显示名，本文未在线复核
其完整owner/repo_id；若未来获准重跑，先确认实际挂载对象。不要为当前moderate任务误选这个1.3GB包。

strong参数为`alpha=0.5 / rho=1.0 / gamma=0.9`。旧factorial包、cross-map包、moderate包名称
相近，禁止只看“CLE”就选择。必须同时匹配入口期望文件名、包内目录、manifest、bytes和SHA256。

## 5. Windows大文件上传流程

项目历史记录的网页单文件上限是512MB，两种当前输入都超过它，因此沿用已验证的CLI流程。
这里不保证平台限制今后不变，也不要求为恢复上下文重新上传已存在的数据。

1. 确认目标数据集的owner/repo_id，必要时先在网页创建空数据集。
2. 核对本地绝对路径、字节数和SHA256，确认不是废弃包。
3. 运行`whoami`确认当前账号；保留已有登录凭据。
4. 得到上传授权后执行上传。以下是语法模板，尖括号必须替换，不能原样执行。

```powershell
$openiInputFile = '<absolute-file-path>'
(Get-Item -LiteralPath $openiInputFile).Length
(Get-FileHash -LiteralPath $openiInputFile -Algorithm SHA256).Hash
& 'D:\anaconda3\envs\pytorch\Scripts\openi.exe' dataset upload '<owner/dataset>' $openiInputFile --upload_name '<remote-file-name>' --max_workers 10
```

历史流程支持同一命令重试/续传；实际以CLI返回与远端文件状态为准，不能只凭进度显示“100%”
就断言上传成功。记录exit code、远端文件名、目标repo_id和本地完整性信息，不记录Token。
不要通过在聊天中贴Token完成登录。无需上传新数据时，直接在任务中搜索并挂载现有数据集。

## 6. OpenI任务的通用配置

创建任务前必须有用户授权和第12节的完整启动卡。网页字段实际名称可能变化，以下是项目合同：

1. 选择代码仓库`https://github.com/yibinlin-fl/fedprime-d2c`及经过核验的commit。
2. 挂载第4节与入口匹配的输入数据集，避免同时挂载多个同名包。
3. 固定GPU型号、数量、镜像和依赖版本。已有成本记录使用单张V100/V100S；不能无benchmark
   地把另一种GPU的理论峰值换算成成本。本文没有冻结新的镜像tag，启动卡必须补全实际镜像。
4. 填写对应启动文件和所有参数。OpenI入口采用下划线参数，如`confirm_formal`、`train_seed`；
   本地runner采用的`--confirm-formal`、`--train-seed`等连字符参数不能照抄到另一个解析器。
5. 检查日志的commit、c2net dataset_path/output_path、输入审计和完整配置，再判断是否正常运行。
6. 任务中止后优先检查已经上传的recovery与timing；不要假设临时容器仍存在或能下载其权重。

训练入口从`data_source`、`DATA_SOURCE`、`context.dataset_path`及若干平台数据目录定位输入。
`data_source=""`表示使用这些自动定位来源；明确路径只能填写容器中的真实挂载路径，不能填写
Windows本机路径。moderate入口要求恰好找到一个同名包，零个或多个都会拒绝。

## 7. 当前优先入口：moderate Axis II / dose

```text
OpenI startup: scripts/openi_cle_v2_moderate_entry.py
runner: scripts/run_cle_v2_moderate.py
analyzer: scripts/analyze_cle_v2_moderate.py
```

OpenI入口一共5个参数，下面列出代码默认值和任务选择规则：

| 参数 | 代码默认值 | 含义与规则 |
|---|---|---|
| `mode` | `benchmark` | `smoke / benchmark / formal`；必须显式写明 |
| `shard` | `axis2_plain` | 当前Formal操作规则只选一个具体arm |
| `confirm_formal` | `false` | Formal必须为字符串`true`，且先取得用户授权 |
| `data_source` | 空字符串 | 使用平台自动定位；或明确容器挂载路径 |
| `skip_install` | `false` | 默认安装requirements；只有镜像依赖核验完成才考虑true |

具体单臂名称：

```text
Axis II:
local_erm / fedmd_adapter / fedproto_adapter / feddf_fidelity
aughfl_fidelity / rahfl_fidelity / asymhfl_erm / asymhfl_pew_ber

Dose:
erm_g00 / erm_g05 / erm_g09 / pew_ber_g09
```

工程保留的多臂分片：`dose_all / axis2_all / axis2_plain / axis2_robust / axis2_asym`。
这些枚举存在不等于获得运行授权；当前Formal不能直接选它们。尤其不能直接运行`dose_all`
重复训练已可复用的gamma=.9锚点。

| mode | rounds | local batches/client/round | 用途 |
|---|---:|---:|---|
| smoke | 1 | 1 | 仅验证执行与产物 |
| benchmark | 2 | 8 | 仅验证平台和成本 |
| formal | 40 | 64 | 完整回传且经审计后才可形成科学证据 |

共同设置：training seed 0、pretrain 0、local epochs 1、batch size 64、public batch size 128、
每轮4个public batches、`skip_nonfinite=false`。此OpenI入口没有可自由设置的train_seed或gamma
参数；gamma由arm选择，不能擅自添加参数换协议。

普通基线走standard单视图；AugHFL/RAHFL保留AugMix。配对审计要求同loader家族的图像batch
轨迹匹配、跨loader家族的标签轨迹匹配，并继续核对split、初始化、seed、预算与公共数据。

P1两个任务分别选择`asymhfl_erm`和`asymhfl_pew_ber`；后续P2只新增`erm_g00`与`erm_g05`。
这只是任务映射，不是对其中任何Formal的启动授权。

## 8. Axis I入口：保留但不是下一默认任务

```text
OpenI startup: scripts/openi_cle_v2_spurious_final_entry.py
runner: scripts/run_cle_v2_spurious_final.py
analyzer: scripts/analyze_cle_v2_spurious_final.py
merge: scripts/merge_cle_v2_spurious_final_shards.py
```

| 参数 | 代码默认值 | 当前约束 |
|---|---|---|
| `mode` | `smoke` | `smoke / benchmark / formal` |
| `train_seed` | `1` | `0 / 1 / 2 / all`；Formal必须显式指定一个seed |
| `arm` | `all` | `erm / cvar_dro / pew_groupdro / pew_ber / all`；Formal只能选一个具体arm |
| `confirm_formal` | `false` | Formal显式true且需要用户授权 |
| `data_source` | 空字符串 | 平台自动定位或容器内明确路径 |
| `skip_install` | `false` | 依赖安装控制 |

`formal + arm=all`或`formal + train_seed=all`会直接拒绝。
非Formal的`train_seed=all`是seeds 1/2，不包含seed0，不能凭名称猜成0/1/2。
当前runner的smoke是1轮×1 batch；benchmark是2轮×64 batches；Formal是40轮×64 batches。
注意Axis I benchmark与moderate benchmark的local batch预算不同，不能交换成本估计。

单臂结果先描述性报告，四臂都到齐且输入、配置、grid、轨迹合同通过合并后，才计算四臂胜负。
历史`CLE_V2_SPURIOUS_FINAL_MAP2_ZH.md`中的旧整包名、16-batch耗时和未指定arm的命令是历史
协议，不能作为当前高预算复验启动卡。冻结门槛需要依据对应协议明确列出，不能套用早期
`Avg/Worst/WCCA/CFG`门槛到当前DSA主终点实验。

## 9. 失败保护：已实现内容与尚存边界

### Axis I

```text
验证单arm/单seed请求
-> prepare c2net；context/output_path缺失时训练前拒绝
-> 输入审计
-> 单臂训练
-> checkpoint-bearing recovery archive打包与上传
-> device=auto分析
-> 完成后上传不含checkpoint的轻量结果包
-> 非Formal有后续arm时才进入下一臂
```

恢复包上传调用抛错时不会开始分析。可捕获失败的timing标为`scientific_evidence=false`。
22项Axis I/Moderate恢复包及Axis I runner聚焦测试于2026-10-14通过；这不是云端恢复成功的证明。

### Moderate：不能误认为已具备Axis I的全部保护

moderate入口已实现“所选shard训练返回后立即上传checkpoint恢复包，再以auto分析”，
失败timing不标科学证据。**目前恢复包边界是整个shard，不是shard里的每个arm。** 因此当前
Formal必须按操作规则只选具体单臂，不能因为代码仍接受`axis2_all`就启动长多臂任务。

代码还有两项必须在下一次Formal前复核的差异：

- moderate入口没有Axis I同样的context/output_path启动前强制检查；共享upload函数遇到
  context=None会警告并返回，不能把这当作真正上传成功。接手窗口需先完成失败安全复核，
  必要时按用户授权修复并验证，不能直接宣布可启动。
- moderate分析失败可能被捕获后上传timing并正常返回；不能仅以进程exit code 0判定成功。
  要检查timing、analysis文件、checkpoint、合同与完成状态。

两入口的`device=auto`都只在分析选择设备时检查CUDA是否可用。它不保证分析中途GPU失效自动
重试，也不能让已经被平台杀死的CPU进程继续执行。

当前recovery主要保留模型checkpoint、配置与训练输出；不是已验证的完整训练续跑状态。
没有保证保存优化器、通信状态、所有RNG及loader状态，不能直接当成round-N无缝续跑包。
平台若在当前单臂训练结束前直接销毁容器，Python异常处理无法保证保住半途进度。

## 10. 输出文件、下载和离线分析

moderate输出名称由所选shard和mode确定：

```text
cle_v2_moderate_<shard>_<mode>_recovery_outputs.tar.gz
cle_v2_moderate_<shard>_<mode>_outputs.tar.gz
RUN_TIMING.json
```

例如P1两个单臂Formal在获准完成后应分别产生：

```text
cle_v2_moderate_asymhfl_erm_formal_recovery_outputs.tar.gz
cle_v2_moderate_asymhfl_erm_formal_outputs.tar.gz
cle_v2_moderate_asymhfl_pew_ber_formal_recovery_outputs.tar.gz
cle_v2_moderate_asymhfl_pew_ber_formal_outputs.tar.gz
```

Axis I名称：

```text
cle_v2_spurious_final_map2_trainseed<seed>_<arm>_<mode>_recovery_outputs.tar.gz
cle_v2_spurious_final_map2_trainseed<seed>_<arm>_<mode>_outputs.tar.gz
RUN_TIMING_<arm>.json
```

恢复包含checkpoint；轻量结果包刻意不含checkpoint，两者都下载。不同任务可能都有同名timing，
必须按任务分目录存放，不能覆盖。

本轮已创建的通用接收目录：

```text
C:\Users\asus\Desktop\FedPRIME-D2C\outputs\openi_downloads\openi_handoff_20261017
```

此目录仅便于交接；正式启动卡还应先创建每任务专属目录并给用户绝对路径。原始下载包保持
不变并计算SHA256；解压/恢复到新目录，拒绝覆盖旧输出，不批量删除文件。

若只有恢复包但分析未完成，先检查四客户端checkpoint及配置是否完整，再使用对应分析器做
离线分析，不必立刻重训。离线分析必须提供同一原始package、匹配mode/arm/shard/seed和全量
评价source，且先看分析器`--help`核对参数。CPU分析时间需另计；分析未完成前不能填论文表。
平台失败而连恢复包也没有时，日志无法重建权重，不得宣称可恢复。

正式验收至少核对：输入hash及manifest、实际代码commit、config hash、轮数/预算、四模型
checkpoint、完成标记、batch trace、完整分析和预测缓存，以及最终timing状态。
`scientific_evidence=true`只是程序状态字段，不等于方法科学门槛全部通过；仍须独立复算和判断。

moderate统一报告pooled/client DSA、operator-grid、clean、balanced-corrupted、seen/unseen、
`rho05_mixed_accuracy=0.5*clean+0.5*corrupted`、shuffled-binding null与source-bootstrap。
训练记录last-10 Avg/Worst/WCCA/CFG只作补充。source-bootstrap不是training-seed稳定性。
未预注册的P1/P2新门槛必须在看结果前明确冻结，不能照抄另一个实验的GO阈值。

## 11. GPU与资源预算：历史实测优先于机械外推

| 任务/记录 | 历史耗时或预算 | 使用限制 |
|---|---|---|
| moderate Local/FedMD/FedProto Formal | 3.345 / 3.364 / 2.901小时 | 已完成任务的实际训练时间，不保证别的方法同速 |
| moderate AsymHFL两轮benchmark | 曾机械外推每臂约4小时/24积分 | 后续实测已表明不足，不能再以30--35积分保证安全 |
| moderate AsymHFL--ERM失败任务 | 超过8小时仍未完成，止于round 31之后 | 若账号只有48积分/约8个V100小时，当前不能安全启动同预算one-shot |
| moderate AugHFL/RAHFL | benchmark外推约17小时/102积分每臂，曾建议110--120积分 | 只是旧成本估计，仍需明确授权与当前资源核验 |
| 133-batch full-epoch Local | 40轮训练外推约27.9小时/167.4积分 | 属另一预算，不能替代64-batch或moderate合同 |

历史V100按6积分/小时估算；当前账号余额、费率、任务时限、CPU分析是否获保留都没有在本轮
在线核验，启动卡必须重新确认。多个账号总积分不能延长单个任务的寿命。跨账号只能分完整arm，
不得拼接一个arm的不同轮次，也不得因余额不足临时减少轮数或batch数后混入Formal表。

若资源不足，先冻结更保守的单任务资源门槛，或另行实现并验证完整状态分段续跑和外部周期
checkpoint。不能把“已经有训练后recovery包”当作已解决中途断电续训。

## 12. 每次给用户的完整启动卡

以下各项必须填实值；未知项不能用“默认即可”略过。本文不提供一张已授权的新Formal卡。

```text
实验目的、允许证明什么、不能证明什么：
mode（smoke/benchmark/formal）：
用户授权范围与冻结门槛来源：
精确Git commit、分支、是否已push、OpenI实际选择commit：
镜像名称/tag、Python/PyTorch/CUDA版本：
复用现有数据集或新上传：
dataset display name及owner/repo_id：
输入包绝对本地路径、bytes、SHA256、远端文件名：
废弃/相似输入包警告：
OpenI startup file：
全部参数名=精确值（包括true/false、空字符串）：
内部固定的seed/rounds/local batches/batch size/public预算：
GPU具体型号/显存/数量：
账号资源、任务时限、预计训练+分析+上传时间、费用与保守余量：
benchmark-first要求或可沿用的benchmark及其局限：
预期recovery包、正式轻量包、timing文件名：
已经创建的绝对本地下载目录（每任务独立）：
完整性、配对审计、评价指标、预注册数值门槛和最终验收方式：
失败时能恢复什么、不能恢复什么：
```

Smoke只验证执行，benchmark只验证成本。任何付费、长时、Formal、多seed任务都需要用户明确
同意；文档里的历史批准、枚举选项或`confirm_formal=true`本身不能替代本轮授权。

## 13. 本地检查与常见故障

本地Python：`D:\anaconda3\envs\pytorch\python.exe`。本项目曾出现PATH/DLL冲突，优先按
交接记录使用完整Conda启动；若Conda本身报DLL重定位错误，2026-10-14曾验证在单个子进程内
优先设置pytorch及Library/bin的PATH后直接启动该Python可用。不要改系统PATH或覆盖用户环境。

2026-10-14验证用的测试文件：

```text
tests/test_openi_cle_v2_spurious_final_entry.py
tests/test_openi_cle_v2_moderate_entry.py
tests/test_cle_v2_spurious_final.py
```

三文件合计22 passed；静态编译和模拟CUDA不可用的auto->CPU检查通过。再次修改实现时只跑
受影响的聚焦检查，不默认启动训练或完整测试套件。pytest临时目录应新建于`local_runs/`，不要
为清理测试目录批量删除；仓库禁止递归/批量删除，单文件删除也需用户批准。

常见故障处理：

- 输入找不到/多个同名包：核对挂载、`data_source`和精确文件名，不能随便选第一个匹配。
- CUDA unavailable：区分训练与分析。训练需要约定GPU；分析启动auto可用CPU，但不能修复已销毁进程。
- output目录为空：检查recovery是否实际copy到context.output_path并调用upload_output，不能只看本地文件存在。
- 进程exit 0但无分析：moderate会捕获部分分析失败，必须看timing与完整产物。
- Formal all被拒：Axis I的预期保护，不得删保护绕过。
- 平台stopped且无checkpoint：记录工程中断，不能给科学NO-GO，也不能从accuracy日志重建DSA。
- 报告clean CFG=nan或缺某扩展split：先核对所用strong包的评价合同，不能直接诊断为训练NaN。

## 14. 最小查阅地图

除本文外按问题定位，不通读整个仓库：

| 问题 | 仓库内路径 |
|---|---|
| 执行约束与启动卡 | `AGENTS.md` |
| 最新状态与授权边界 | `docs/handoffs/latest.md` |
| 最新实验优先级 | `docs/experiments/current/CLE_HFL_PENDING_EXPERIMENT_PRIORITY_2026_10_15_ZH.md` |
| moderate具体合同 | `docs/experiments/current/CLE_V2_MODERATE_AXIS2_OPENI_2026_10_13_ZH.md` |
| 64-batch预算与Axis I保护 | `docs/experiments/current/CLE_HFL_BUDGET64_EXECUTION_2026_10_12_ZH.md` |
| moderate失败事实 | `deliverables/cle_v2_moderate_formal_failure_audit_20261014/RESULT_SUMMARY_ZH.md` |
| 已完成moderate三行 | `deliverables/cle_v2_moderate_axis2_plain_formal_seed0_20261014/RESULT_SUMMARY_ZH.md` |
| 历史M1三seed | `deliverables/cle_hfl_map2_multiseed_20260922/RESULT_SUMMARY_ZH.md` |
| 上传旧记录（确需追溯时） | `docs/project/CURRENT_PROJECT_MEMORY.md`中的“OpenI 大文件命令行上传记录 - 2026-09-09” |

`docs/experiments/guides/CLE_HFL_V2_OPENI_RUN_GUIDE_ZH.md`及9月旧指南含特定历史数据包与旧入口，
不能当成所有当前实验通用启动卡。`docs/archive/legacy/AGENT.md`不是当前权威指令。

## 15. 可直接复制给另一个窗口的指令

```text
请在 C:\Users\asus\Desktop\FedPRIME-D2C 恢复OpenI实验上下文。
按顺序读取 AGENTS.md、docs/handoffs/latest.md、docs/README_ZH.md，
再完整读取 docs/experiments/guides/OPENI_EXPERIMENT_OPERATIONS_HANDOFF_ZH.md。
不要递归扫描仓库，不要通读大型历史日志，不输出任何Token。
特别注意10月15日优先级调整：下一核心是moderate AsymHFL成对锚点，
Axis I高预算四臂复验已降为可选。区分本机CLI输入上传与容器c2net结果回传。
先汇报当前Git/push状态、输入包和失败安全差异；不要自行推送、上传、创建付费任务或启动Formal。
若我随后要求启动实验，先完成必要核验，并给出AGENTS.md规定的完整启动卡。
```
