# 平台官方事实、用户预算与工程约定

核验日期：2026-10-17。下面严格区分平台资料、账号实时状态与本Skill建议。

| 项目 | 已核验说明 | 启动时处理 |
|---|---|---|
| 免费GPU周配额 | 官方写30小时，有时随需求与资源增加 | 保守按用户30小时预算；查当前total/remaining/reset，不承诺固定额度或重置时刻 |
| 当前常见T4选项 | 官方技术表/CLI标为T4×2 | 用nvidia-smi及torch实测型号、数量、每卡显存 |
| 会话时限 | 官方Notebook文档当前写CPU/GPU 12小时 | 以实际界面/运行类型当前限制为准，不顶着12小时安排训练 |
| 可保存输出 | 官方文档写/kaggle/working最多20GB | 同时预算原始checkpoint+压缩包+分析，不能只算最终tar |
| 临时空间 | working之外临时空间不随Notebook输出保存 | 临时区不能是唯一checkpoint存放处 |
| 保存版本 | Save & Run All启新会话，从头运行 | 任何依赖先前手动执行、变量或临时文件的Notebook都需修复 |

GPU周配额依据：[Kaggle Efficient GPU Usage](https://www.kaggle.com/docs/efficient-gpu-usage)。
T4、时限和输出的规格依据：[Kaggle Notebooks](https://www.kaggle.com/docs/notebooks)。
T4请求名依据：[官方kernel metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md)。

官方Notebook部分旧段落还只提P100，不能因此声称T4不存在；采用更具体的当前规格并实测。
这些公开页面不是该账号可用硬件、剩余额度或已登录状态的证明。本轮未登录任何Kaggle账号。

## 账号与网络

需要可正常使用Kaggle的账号。GPU/Internet灰色时先按界面检查账号验证、当前quota、资源排队
及Notebook/竞赛限制；不替用户完成手机号/身份验证，也不声称某一旧教程的验证步骤永远适用。

联网拉GitHub、在线pip需要Internet可用；纯离线任务提前挂载代码包、数据、权重和依赖。
代码竞赛的联网与输出要求由具体竞赛决定，不能把一般科研Notebook规则直接当竞赛提交规则。
联网与依赖说明：[官方Notebook环境说明](https://www.kaggle.com/docs/notebooks)、
[官方Packages/依赖说明](https://www.kaggle.com/docs/packages)。

## 每周30小时的计划方式

这是工程建议，不是Kaggle官方分配规则：

- CPU做配置/输入/语法检查；只在需要时启动GPU会话，用完手动停掉。
- 先实测一个代表性单元，统计加载、训练、评价、写盘和压缩，不按T4理论峰值猜耗时。
- 举例：若本周剩30小时，可先预算约1小时诊断/benchmark、24小时已选实验、5小时故障/回传
  余量。它只是示例，不能当新课题固定配方；训练总时长还要切成各自满足会话限制的任务。
- T4×2不自动把30小时变成60小时。配额计量以平台返回为准，不臆测按卡倍增/折半。
- 双卡各自显存独立；DDP/数据并行需要现有代码支持与配对验证，不能为“免费双卡”改变科学预算。
- 不在后台运行的同时保留无用GPU编辑会话；不要用保活脚本或反复账号切换规避平台限制。

实际每卡显存、资源余量和耗时是启动卡必填实值。若单任务超时风险高，先实现并验证分段续跑，
不能把一次完整实验随意截成若干段后只保留好看的部分。
