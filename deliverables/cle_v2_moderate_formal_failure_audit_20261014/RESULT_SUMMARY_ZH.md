# 2026-10-14 Moderate Formal失败审计与恢复边界

## 1. 直接结论

三个OpenI任务不是同一种算法失败：

| 任务 | 训练状态 | 失败位置 | 当前科学状态 |
|---|---|---|---|
| FedDF | 40/40轮完成，4个checkpoint已写入临时容器 | 统一分析启动时CUDA不可用 | 训练完成，但没有DSA统一推理包，不能填主表 |
| AsymHFL+PEW+BER | 40/40轮完成，4个checkpoint已写入临时容器 | 统一分析启动时CUDA不可用 | 训练完成，但没有DSA统一推理包，不能填主表 |
| AsymHFL--ERM | 仅完成0--31轮 | 平台写入`stopped`，无Python异常 | 不完整Formal，永久禁止填表 |

FedDF和PEW+BER的直接故障是旧入口把分析设备硬编码成`cuda`；长训练结束后新分析子进程观察到
`torch.cuda.is_available()==False`而终止。两份日志都没有训练NaN、non-finite或OOM。至于平台
为什么撤掉GPU，日志没有给出精确原因；最可能是账号可用时长/积分或任务资源租期结束，不能把
这一推断写成已证实事实。AsymHFL--ERM由平台外部停止，也没有算法异常证据。

## 2. 为什么旧任务没有回传

旧入口的顺序是：训练 -> CUDA分析 -> 写timing -> 打包 -> `upload_output()`。分析抛异常后，打包
和上传均未执行。因此日志证明临时容器内曾保存checkpoint，但日志本身不包含模型权重；若失败
任务页面没有单独保存容器文件的能力，这些checkpoint不能从文本日志重建。

## 3. 日志可恢复的有限数字

下列数字仅来自训练日志，可用于事故审计，不能替代统一checkpoint推理结果：

| 任务 | 完成轮数 | last-10 Avg | last-10 Worst | last-10 WCCA | last-10 CFG |
|---|---:|---:|---:|---:|---:|
| FedDF | 40 | 38.013 | 34.746 | 2.175 | 36.185 |
| AsymHFL+PEW+BER | 40 | 36.858 | 32.316 | 4.825 | 27.933 |
| AsymHFL--ERM | 32 | 36.140 | 34.015 | 3.500 | 34.879 |

第三行是未完成任务最后十个已完成轮次的均值，不是Formal结果。日志无法恢复DSA、paired
operator-grid accuracy、统一clean/corrupted/mixed accuracy、shuffled-binding null或source
bootstrap，因此第4、7、8行仍不能正式填入Axis II表。

## 4. 已完成且允许填表的结果

Local/ERM、FedMD和FedProto的正式结果包在本次事故前已经成功上传并完成独立审计，可继续填入
Axis II第1--3行。详见：

```text
deliverables/cle_v2_moderate_axis2_plain_formal_seed0_20261014/RESULT_SUMMARY_ZH.md
```

## 5. 入口修复

`scripts/openi_cle_v2_moderate_entry.py`现在执行：

1. 训练结束后先写入非科学的`training_complete_analysis_pending`状态；
2. 立即打包包含checkpoint和config的`*_recovery_outputs.tar.gz`；
3. 在分析前立即上传恢复包，抵抗随后发生的平台硬停止；
4. 分析设备改为`auto`，CUDA不可用时回退CPU；
5. 若分析失败，回传`failed_with_recovery_artifact` timing，且绝不标记为科学证据；
6. 若训练本身失败，尽力上传partial recovery后仍以失败退出。

聚焦验证：Python静态编译通过；moderate配置测试与恢复包/timing测试共`6 passed`。

## 6. 下一步约束

- 修复提交推送前禁止重跑三个Formal。
- 先在失败任务页面检查是否能直接下载临时checkpoint；如果不能，FedDF与PEW+BER才需要重训。
- 重训时账号必须预留训练、恢复包上传和CPU/GPU分析余量，不能再按纯训练机械估计卡余额。
- 后续任何Formal即使分析失败，也应至少返回可离线分析的checkpoint恢复包。
