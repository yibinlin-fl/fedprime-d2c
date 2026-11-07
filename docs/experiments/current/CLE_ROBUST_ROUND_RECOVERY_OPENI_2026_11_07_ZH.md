# AugHFL / RAHFL 第20、30、40轮回传与续训

2026-11-07。本指南替代旧卡中这两个方法的启动说明。主评价八行表已完成6/8行，待补这两行。

## 已实现并验证的行为

两项单臂Formal自动启用恢复保护，科学配置仍为主评价map2、training seed 0、α=1.0、ρ=0.5、γ=0.9、40轮×每客户端每轮最多64个batch，batch size 64、公共batch size 128、每轮4个公共batch、预训练0轮。

1. 验证输入并生成配置后，训练前上传 `RUN_STARTED.json` 和对应 `CONTRACT_*.json`。启动记录包含代码、输入身份和计划保护轮次。
2. 第20、30、40轮完成后分别上传独立的progress包。每包含完整模型/Adam状态、模型模式、Python/NumPy/Torch/CUDA RNG、私有loader generator、公共迭代位置、通信状态、已有CSV及首batch轨迹、配置和输入/代码/运行环境身份。上传在独立进程执行，父进程保留原训练RNG。
3. 训练完成后继续上传原有checkpoint recovery包，再做正式分析，最后上传正式结果包与 `RUN_TIMING.json`。
4. progress包使用各自轮次文件名。后一次上传失败会使任务失败并尽力回传失败状态，先前的成功包保持独立。

保护覆盖已完成且已成功上传的阶段。第20轮首次进度包上传前，中断仍可能损失尚未回传的训练进度。平台硬销毁或正在进行的上传无法由Python保证完成。首次新云端任务还需核对上传日志及平台产物实际可下载性。

## 本地验证与边界

22项聚焦检查通过。CPU小规模smoke使用两个客户端、小模型和合成输入，执行真实的AugHFL公共多视图通信/本地JSD路径，以及RAHFL的audit路由/本地JSD+DCL路径。分别比较原运行、开启保护后连续运行、完成2轮后模拟停止并恢复到4轮。最终权重逐张量一致，训练指标和首batch轨迹一致，CSV没有重复轮次。还覆盖公共iterator未启动/部分消耗/耗尽、Adam状态、输入身份不匹配、上传失败、危险解压条目和训练前先回传身份。

此验证证明恢复实现的小规模执行正确性。新上传路径尚未在OpenI真实任务中实测。GPU下的浮点执行仍遵循原PyTorch/CUDA行为，恢复要求同版本、GPU型号/数量和后端设置。半程包标为 `scientific_evidence=false`，最终40轮和正式分析完成后才进入论文验收。

## 首次启动卡

代码已创建本地提交，推送后以本次交付给出的commit核对OpenI选择。旧远端 `32f4050` 没有本指南的训练中回传功能。

| 字段 | AugHFL | RAHFL |
|---|---|---|
| 任务名 | `cle-main-aughfl-recovery-s0-20261107` | `cle-main-rahfl-recovery-s0-20261107` |
| 项目/分支 | `fedprime-d2c / main` | 相同 |
| 启动文件 | `scripts/openi_cle_v2_moderate_entry.py` | 相同 |
| `mode` | `formal` | `formal` |
| `shard` | `aughfl_fidelity` | `rahfl_fidelity` |
| `confirm_formal` | `true` | `true` |
| `skip_install` | `false` | `false` |
| `data_source` | 空字符串，可不添加 | 相同 |
| `resume_source` | 空字符串，可不添加 | 相同 |

两账号各130积分。按此前6积分/小时费率，各可覆盖21小时40分；历史benchmark外推训练约16–17小时，按约120积分预留，实际耗时及费率以平台记录为准。GPU各1×V100 32GB、CPU8、内存50GB，镜像 `ubuntu22.04-cuda11.8-py310-torch2.1.0-tf2.14.0`，访问Internet选是。

复用数据集 `chujiu/CLE_v2_Moderate_Alpha10_Rho05_Map2_Seed0_PEW_20261013`。本机输入文件为 `C:\Users\asus\Desktop\FedPRIME-D2C\local_runs\cle_v2_moderate_alpha10_rho05\cle_hfl_v2_moderate_alpha10_rho05_map2_seed0_split0_with_pew.tar.gz`，893421220 bytes，SHA256 `DEA800EC1465C45037E74E4B948DFB8D778F1354E8522B551438C6A515C9E588`。禁用旧cross-binding/factorial机制包以及CIFAR-100-private包。

## 下载文件与现成目录

各任务的 `<shard>` 分别替换为 `aughfl_fidelity` / `rahfl_fidelity`。

- 中途保存包为 `cle_v2_moderate_<shard>_formal_progress_round_020_outputs.tar.gz`、`..._round_030_outputs.tar.gz`、`..._round_040_outputs.tar.gz`。中断时下载已成功上传的最高轮次包，保留原字节。
- 成功后下载 `cle_v2_moderate_<shard>_formal_outputs.tar.gz`、`cle_v2_moderate_<shard>_formal_recovery_outputs.tar.gz`、`RUN_TIMING.json` 和完整日志。可另存最高轮次progress包作为续训来源记录。
- AugHFL目标目录为 `C:\Users\asus\Desktop\FedPRIME-D2C\outputs\openi_downloads\main_axis2_aughfl_formal_s0_20261104`。
- RAHFL目标目录为 `C:\Users\asus\Desktop\FedPRIME-D2C\outputs\openi_downloads\main_axis2_rahfl_formal_s0_20261104`。

这些目录已创建，任务之间的同名计时/状态文件各放各的目录。

## 中断后恢复

下载最高成功轮次progress包后，将其作为附加数据集上传并挂载到新任务，同时挂载原893421220-byte输入包。使用同一代码commit、镜像、GPU型号/数量和原参数，额外填写 `resume_source`，值为该progress包的完整文件名。入口在数据挂载中要求只找到一个同名包，也支持明确的容器绝对路径。

例如AugHFL从第30轮续跑，填写 `resume_source=cle_v2_moderate_aughfl_fidelity_formal_progress_round_030_outputs.tar.gz`。恢复日志应出现 `restored 30 completed rounds including optimizer/RNG`，从零基编号 `round 030` 开始继续，即第31轮。原40轮总目标保持不变。已完成轮次的CSV与trace保留，之后追加。

入口校验代码commit、输入SHA256、科学配置、PyTorch/CUDA/GPU与后端设置、权重与优化器库存、状态文件SHA256和历史轮次。身份冲突会停止。旧的纯权重recovery包不接受作为中途续训包。第40轮progress包可以跳过训练轮次，直接保存最终权重并进入分析。续跑任务的计时记录覆盖本次执行，累计算力要合并前序任务账单。

## 科学验收

核对完整40轮、seed0/map2/输入身份、四客户端权重、有限训练loss、配对trace、预测缓存和正式分析状态。报告pooled/client DSA、grid/clean/corrupted/seen/unseen/mixed accuracy及shuffled-binding null。没有新增事后胜负门槛。两条基线补齐固定主评价协议下的领域比较，训练seed稳定性仍由对应多seed实验承担。
