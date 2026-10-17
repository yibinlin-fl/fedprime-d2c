---
name: kaggle-experiments
description: "为跨项目科研任务准备Kaggle免费GPU实验：Jupyter Notebook、固定版本拉取代码、数据集挂载、T4和周配额预算、启动卡、保存版本、checkpoint和结果下载审计。用户说“我要在Kaggle跑实验”“用每周30小时T4”“生成Kaggle notebook”“Kaggle输出放哪里/怎么回传”时使用。"
---

# Kaggle 免费 GPU 实验助手

平台资料核验日期：2026-10-17。配额和UI可能变化，按当前账号核验。

为当前项目交付可复现的`.ipynb`、完整中文启动卡和可核验的结果交回流程。默认用用户的免费
GPU资源，按每周30小时保守规划，但实际配额、剩余时长、硬件及单会话时限以账号当前显示为准。
不继承FedPRIME或其他课题的数据、轮数、模型、seed与科学门槛。

## 先识别当前项目

读AGENTS及其指定handoff/索引，检查相关Git状态和一个实际入口/配置。没有这些文件时从README
和训练命令开始，不扫描全仓库。确认目标是准备、smoke、benchmark、Formal、排错或结果接收。
“我要跑Kaggle”先准备具体可审阅结果，不擅自上传私有数据、公开仓库或消耗GPU额度。
已有明确授权就继续，不重复询问；关键科学参数未知时只补最少必要问题。

## 按任务读取文档

- 配额、T4、时限、路径、官方事实与更新时间：读[平台要求](references/platform-requirements.md)。
- 第一次操作或生成Notebook：读[Notebook完整流程](references/notebook-workflow.md)。
- GitHub拉项目、私有代码、离线数据与依赖：读[代码和数据](references/code-and-data.md)。
- checkpoint放哪、保存、下载、续跑：读[输出和恢复](references/outputs-and-resume.md)。
- 给用户照填网页：读[完整启动卡](references/launch-card.md)。
- 本地API登录、数据上传和Notebook CLI：读[CLI与凭据](references/cli-and-auth.md)。
- 需要更新平台事实时：读[官方来源与核验表](references/official-sources.md)，优先当前官方资料。

## 交付流程

1. 确定一个明确实验单元、准确代码commit/源码hash、数据版本与输入hash、预算和评价合同。
   云端不能拉到本机未推送代码；不自动push。私有代码优先私有Dataset源码包或安全凭据方案。
2. 为该项目生成/适配`.ipynb`；Notebook负责环境/代码/数据/调用与产物整理，训练逻辑复用项目
   runner，避免把算法重写进Notebook。Notebook是本Skill约定，Kaggle官方也支持Script。
3. 默认先CPU准备/小smoke，再在授权后跑代表性T4 benchmark；按实际剩余额度与单会话上限
   给训练、分析、打包留余量。免费也不能无授权反复长跑；不把T4×2当自动两倍速度或合并显存。
4. 所有需交回产物明确写到`/kaggle/working/<run_id>/`；输入从挂载路径读，源码/大缓存放
   临时区，防止把整个Git仓库、依赖和原始数据一起保存为输出。
5. 把恢复checkpoint与状态提前落盘，完成后保存Notebook版本并查该版本Output。
   不把工作目录写入说成云端持久保存；不把OpenI的c2net.upload_output移植进Kaggle。
6. 给完整启动卡与已创建的本机下载目录。收到结果后核对Notebook版本、commit、输入/配置hash、
   checkpoint、预算、日志和分析，更新当前项目记录。smoke/benchmark不作科学证据。

## 高风险误解

- Save & Run All从干净的新会话运行，不是保存当前进程内存/继续当前轮次。避免交互已跑一遍
  又误触同一长训练后台重跑，两个会话都要检查并在不用时停止。
- Quick Save/Persistence不统一保证输出与状态保存；按实际保存选项并检查版本Output。
- 本地checkpoint不等于完整断点续跑。硬超时可能在打包前杀进程，需要提前分段和外部已保存产物。
- `kaggle kernels push`通常触发运行。只准备文件时不执行push；若当前CLI支持--no-run，也
  只能在用户已授权上传该Notebook的范围内使用。
- 运行成功、输出已保存、下载完成、科学验收分别核对；失败/部分结果保持scientific_evidence=false。

## 资源与模板

- `scripts/create_notebook.py`只在本地新目录生成Notebook和metadata，不联网、不登录、不训练。
- `assets/notebook-template/experiment.ipynb`是可审阅骨架；所有昂贵/联网操作默认关闭，配置和项目命令需适配。
- `assets/notebook-template/kernel-metadata.json`默认私有、CPU、关闭Internet，避免设计稿意外消耗GPU。

用生成器向新目录输出骨架，再按当前项目适配；不覆盖已有Notebook。
