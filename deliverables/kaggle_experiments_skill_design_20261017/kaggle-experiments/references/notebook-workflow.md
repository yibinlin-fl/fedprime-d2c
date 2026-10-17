# Notebook从创建到保存的完整操作

## 1. 默认准备可复制代码单元

读取当前项目约束与入口，固定commit、数据版本/hash、mode、实验单元、预算和指标。
按顺序给用户代码单元，让用户在Kaggle网页新建Notebook并粘贴；Notebook自己根据仓库地址
拉项目。不要求本机生成/上传.ipynb或配置CLI。仅用户明确需要导入或API管理时，才额外交
experiment.ipynb与kernel-metadata.json。Notebook从头运行应能重建必要环境，不依赖先前
编辑会话。官方也支持Script，不能把.ipynb上传说成平台要求。

## 2. 在网页建立Notebook

1. 登录Kaggle，Create → New Notebook，先保持Private；把交付代码单元按顺序粘贴进去。
2. 在Input/Add Input添加数据集/模型/已保存Notebook输出，记录资源版本及实际挂载路径。
3. 找到Settings/Session options中的Language、Accelerator、Internet、环境选项；菜单可能移动。
4. 准备代码/数据时先CPU。正式GPU benchmark前按实际可用选GPU T4×2；检查账号配额及会话时限。
5. GitHub/pip在线方案打开Internet；离线方案关闭，确认所有依赖已挂载。Secrets只按所需标签授权。

配置cell填写repo_url和固定commit；拉取cell执行clone/checkout，之后调用仓库中的实际入口。
不需要用户先将整个项目上传到Notebook。只有私有/离线源码方案才另外准备所需挂载材料。

Notebook创建、保存与环境行为依据：[官方Notebook文档](https://www.kaggle.com/docs/notebooks)。

## 3. Notebook单元职责

| 单元 | 内容 | 可核验产物 |
|---|---|---|
| 实验说明与配置 | 目的、run_id、mode、commit、输入及命令 | 用户能审阅的配置 |
| 环境与GPU检查 | Python/torch/CUDA、卡数、每卡显存、磁盘 | environment.json |
| 代码来源 | 公有GitHub固定commit或私有挂载源码包 | source.json、实际HEAD/hash |
| 输入与依赖 | 版本/hash、挂载、必要的精确依赖 | input_manifest/config |
| 训练调用 | 调当前项目runner，不在Notebook重写算法 | metrics、日志、checkpoints |
| 恢复包 | 分析前包含checkpoint与未完成状态 | recovery_outputs.tar.gz |
| 分析与最终产物 | 完整评价、打包、hash、失败标记 | results、status、manifest |
| 保存与下载提示 | 版本ID、Output验收、下载命令 | 可交回本机的输出列表 |

Notebook参数是普通Python配置，不是OpenI网页的命令行参数栏。具体argv仍要与项目argparse一致。
本Skill的模板默认不拉代码、不启训练，适配后才可开启run_enabled。

## 4. 两种运行路线，不能混用含义

路线A（推荐长任务）：CPU准备与小smoke → 固定Notebook及输入 → Save Version → Save & Run All
启动干净后台执行 → 记录生成的版本 → 检查日志、完成状态和该版本Output → 下载。
这次后台执行本身会跑训练，不要先把相同长训练在编辑会话完整跑完再误点此按钮。

路线B（交互调试）：按单元执行 → 对当前会话产物及时下载，或核对当前Quick Save/输出保存选项
后保存并检查版本Output。Quick Save保存代码快照不能一概当成checkpoint保存保证。
Persistence方便恢复编辑会话，不等于已验证的完整训练恢复；实际选项与能力以当前UI为准。

官方说明Save & Run All使用新会话：[Notebook版本说明](https://www.kaggle.com/docs/notebooks)。
Kaggle工作人员也明确区分版本、Persistence与无状态重跑：
[官方工作人员说明](https://www.kaggle.com/product-feedback/532308)。

## 5. 完成后的操作

进入Notebook viewer对应版本，打开Output，核对恢复包、正式结果、status和manifest确实存在。
关闭编辑器页面不等于完成这些步骤；文件在工作目录存在也不等于已被版本持久保存。
下载到启动卡预建的本地目录，再停掉不用的会话；重要结果在停会话前先确认已有可下载副本。
保存失败或超时后逐项看实际留存产物，不承诺所有失败会话都能取回工作文件。
