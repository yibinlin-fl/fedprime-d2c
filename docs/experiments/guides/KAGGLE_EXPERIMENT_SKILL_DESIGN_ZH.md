# Kaggle免费T4实验Skill：设计与安装记录

日期：2026-10-17。状态：**用户已确认，已完成本机全局安装**。
此前先设计并修正为默认网页Notebook拉仓库；本轮获得安装确认后完成安装。未登录Kaggle、
未上传、未创建Notebook远端版本、未消耗GPU。

## 1. 名称、目的与适用范围

名称：`kaggle-experiments`。
源码：`deliverables/kaggle_experiments_skill_design_20261017/kaggle-experiments/`。
已安装到`C:\Users\asus\.codex\skills\kaggle-experiments`，下一轮起可通过自然语言或
`$kaggle-experiments`调用。源码目录保留原名，供后续维护。

用户确认的默认方式：Kaggle网页新建Notebook，填仓库地址，由Notebook自己拉取代码。
它为任意新课题交付：可直接复制的Notebook代码单元、完整中文启动卡、代码与数据来源合同、预算和
checkpoint/结果保存方式、每任务本机下载目录，以及结果交回后独立审计流程。
不要求本机先生成/上传.ipynb，不要求配置Kaggle CLI；文件导入只在用户明确需要时提供。
不要求新课题有FedPRIME文件，不复制其数据、rounds、seed、DSA或GO阈值。

## 2. 已设计的七份配套文档

| 文档 | 解决的问题 |
|---|---|
| [平台要求](../../../deliverables/kaggle_experiments_skill_design_20261017/kaggle-experiments/references/platform-requirements.md) | 周配额、T4×2、会话时限、输出容量、验证/联网、30小时规划 |
| [Notebook完整操作](../../../deliverables/kaggle_experiments_skill_design_20261017/kaggle-experiments/references/notebook-workflow.md) | 网页新建Notebook、粘贴代码单元、填仓库地址拉代码、Input与版本执行 |
| [代码与数据](../../../deliverables/kaggle_experiments_skill_design_20261017/kaggle-experiments/references/code-and-data.md) | 固定commit拉GitHub、私有源码包、数据路径、依赖与Secrets |
| [输出、回传与续跑](../../../deliverables/kaggle_experiments_skill_design_20261017/kaggle-experiments/references/outputs-and-resume.md) | checkpoint放哪、恢复/正式包、保存版本、Output下载、跨会话恢复 |
| [完整启动卡](../../../deliverables/kaggle_experiments_skill_design_20261017/kaggle-experiments/references/launch-card.md) | 用户网页怎么选、Notebook参数怎么填、时长/版本/下载目录怎么记 |
| [CLI与凭据](../../../deliverables/kaggle_experiments_skill_design_20261017/kaggle-experiments/references/cli-and-auth.md) | 可选OAuth/token配置、quota、push/pull/status/output及版本差异 |
| [官方来源核验表](../../../deliverables/kaggle_experiments_skill_design_20261017/kaggle-experiments/references/official-sources.md) | 哪些是官方事实、哪些是工程建议、何时要重新核验 |

此外有SKILL.md路由入口、UI元数据、可复制代码单元文档与本地生成器；.ipynb/metadata保留为可选格式。
这些文件同时保留在仓库交付目录；安装副本的13个文件SHA256与源码逐项一致。

## 3. 已核验的关键平台事实

- 官方配额说明为每周30小时，有时更高；本Skill按用户30小时保守规划，实际仍需读账号当前
  remaining/total/reset，不保证固定周额度或固定重置时间。
  [官方GPU说明](https://www.kaggle.com/docs/efficient-gpu-usage)。
- 官方T4配置当前为T4×2；代码不支持多卡时仍可能只用一张，各卡显存不能简单合并。
  [官方Notebook规格](https://www.kaggle.com/docs/notebooks)、
  [官方加速器字段](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md)。
- 官方Notebook文档当前写CPU/GPU会话12小时、working可保存输出20GB；运行前还需对照账号
  UI，预算不能顶着上限。不同历史帖子里的9/12小时不混成保证。
  [Notebook规格](https://www.kaggle.com/docs/notebooks)。
- Save & Run All会启动新会话从头执行；它不是保存当前训练进度。working写盘、保存版本Output、
  下载到本机是三步，需要分别确认。
  [官方版本说明](https://www.kaggle.com/docs/notebooks)。
- 官方CLI同时支持Notebook和Script；用户默认网页新建Notebook粘贴拉仓库等代码，不上传.ipynb。
  文件导入和CLI都是可选路线；普通kernels push通常还会运行，只有适用版本的--no-run才不执行。
  [官方CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)。

这里的公开资料不是用户账号额度、验证状态或T4当前可用性的证明，本轮没有登录核对账号。

## 4. 完整工作流与路径

```text
本机：读新项目约束/README/入口，检查Git
-> 固定源码commit或私有源码包，核对数据版本/hash
-> 给可复制代码单元与启动卡，CPU准备/小smoke
-> 用户核对启动卡及GPU预算
-> Kaggle网页Create Notebook，粘贴代码、填写仓库地址，添加Input，配置T4/Internet/环境
-> Notebook拉固定代码并调用新项目真实runner
-> checkpoints先写working，分析前整理恢复包
-> 分析结束输出正式包、status、manifest
-> 保存版本，核对该版本Output
-> 下载到该新项目的预建绝对目录
-> 本机审计并更新该项目自己的记录
```

默认路径分工：

```text
/kaggle/input/<实际资源路径>/          只读输入
/tmp/kaggle-project/                 临时代码，需固定身份
/kaggle/working/<run_id>/             需要回传的模型/日志/分析/包
<当前项目>/outputs/kaggle_downloads/<run_id>/  本机接收目录，先创建再交用户
```

Kaggle结果不调用OpenI的c2net。不能只把文件保存在clone仓库、/tmp或当前Python变量里。
20GB预算要包括解压/原始checkpoint与压缩包同时占用的情况，不能只看最终包大小。

## 5. Notebook骨架当前包含什么

位置：
`deliverables/kaggle_experiments_skill_design_20261017/kaggle-experiments/assets/notebook-template/`。

- notebook-cells.md：默认使用，按顺序复制进网页Notebook；含说明、配置、环境/GPU检查、固定commit代码来源、输入hash检查、真实runner
  调用、期望checkpoint检查、分析前恢复包、分析、最终包与输出清单，以及版本保存/下载提示。
- experiment.ipynb与kernel-metadata.json：仅作为用户明确需要文件导入时的可选格式；metadata
  默认Private、CPU、Internet关闭，owner/slug和输入资源需按项目填。
- scripts/create_notebook.py：默认只生成notebook-cells.md；加--with-ipynb才额外生成.ipynb和
  metadata。目录已存在则拒绝，不登录或调用Kaggle，也可以直接在回复中给代码而不运行生成器。

模板run_enabled=false；从头执行默认不联网、不训练。配置没有虚构项目入口、数据hash或轮数。
公有GitHub代码可固定commit克隆；私有挂载源码需要项目额外提供源码清单，模板不声称已验证。
依赖、周期性checkpoint、完整续跑、项目科学门槛与GPU多卡由真实项目适配。
模板的scientific_evidence始终为false，独立审计前不自动晋级。

它是经本地静态/安全空运行验证的骨架，不是已经在用户Kaggle账号验证成功的训练Notebook。

## 6. 示例调用与预期交付

确认安装后，你可在新课题窗口说：

```text
使用 $kaggle-experiments。
我想用Kaggle每周约30小时免费T4跑当前项目，代码在GitHub。
请给我在网页新建Notebook后可直接复制的代码单元，填入仓库地址让它自己拉项目，
并准备数据挂载方案、完整启动卡和结果下载目录，不用生成或上传.ipynb。
先做成本benchmark，未经我同意不要启动正式长任务。
```

窗口应先看该项目，交付可复制代码单元、实际commit/数据hash、可照填设置、Notebook命令、GPU/时长
预算、明确保存/下载步骤和失败范围。若你随后明确授权运行，它才执行授权范围内动作。

若资源不够，提出事前冻结的分段续跑方案或缩小已授权执行范围，不擅自改科学预算；多个账号
也不能被当成绕过平台配额的办法。新项目的结果不会下载回FedPRIME目录。

## 7. 验证及安装状态

本轮验证范围为Skill结构、引用/模板结构、代码单元可编译、默认关闭时无外部副作用、生成器
默认不产生.ipynb、显式导出才产生.ipynb以及拒绝覆盖已有目录。没有启动真实Kaggle会话、测试T4速度或验证云端保存成功。

用户确认后已安装13文件，安装副本通过Skill格式校验、引用与隐式调用元数据检查；安装后的
生成器实际验证默认只输出notebook-cells.md，不产生.ipynb。安装未改全局配置或其他Skill。
本轮没有创建远端Notebook、登录账号、上传或启动GPU任务。使用时仍需按当前项目适配和授权。
