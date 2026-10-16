---
name: openi-experiments
description: "为任意研究项目在OpenI/启智社区准备和接收实验：数据集CLI上传、训练入口、完整启动卡与参数表、c2net结果回传、checkpoint恢复包及下载审计。用户说“我要在OpenI跑实验”“帮我上传OpenI数据”“OpenI怎么填参数/回传结果”或处理已有OpenI任务时使用。"
---

# OpenI 实验助手

把当前项目准备成可核验、可回传的OpenI实验，交付可照填网页的完整中文启动卡。
支持新课题，不依赖FedPRIME代码或文档存在。

## 触发与范围

“我要在OpenI跑实验”默认开始准备当前项目：检查约束、入口、输入、输出，完成必要工程并给卡。
不要自动扩展为上传私有数据、公开数据、推送代码或启动付费长任务。用户已明确授权相应范围时
继续完成，不重复索要许可。缺少科学参数时先推进无依赖工作，再集中询问；不要发明轮数或门槛。

## 识别当前项目

- 先读项目AGENTS.md及其指定handoff/索引，检查相关Git状态；没有时读取README和一个直接
  相关入口/配置。不递归扫描全部代码或通读历史日志。
- 确认实验目标、smoke/benchmark/formal角色、数据、预算/seed、评价和已有入口。
  复用正确实现，只做必要修改；遵守项目的提交、删除、数据可见性与授权规则。
- 不把本Skill当上传、推送、付费或删除授权。没有确认的字段标待定；不擅自公开新课题数据。

## 按任务读取参考材料

- 新建/修复入口、恢复包或接收结果：读[入口与结果合同](references/entry-and-results.md)。
- 网页怎么填写、完整启动卡或资源分配：读[启动卡格式](references/launch-card.md)。
- Windows登录、数据上传或SDK问题：读[本机上传上下文](references/windows-local.md)。
- 仅当前仓库为FedPRIME-D2C时读[历史示例](references/fedprime-example.md)，并以最新handoff
  覆盖示例；其他项目不加载，不套用其数据、轮数、DSA或GO阈值。

## 准备到交付

1. 核对代码commit、远端实际可用版本、输入绝对路径/bytes/SHA256和dataset repo_id。
   未变输入优先复用；提示相似废弃包。未fetch时把origin tracking状态标本地快照。
2. 对照真实argparse填写全部参数，区分下划线/连字符、字符串布尔值与空参数。
   确认镜像提供c2net、dataset/output挂载与项目依赖，不偷偷升级研究协议。
3. 确定run_id和产物合同：输入/config/Git/环境、训练记录、checkpoint、分析、manifest与status。
   在昂贵分析前打包并上传含checkpoint的恢复包，不能等整个多臂任务结束才上传。
4. 按改动做聚焦验证：参数解析、输入审计、成功输出及故障路径。smoke只验证执行，benchmark
   只验证成本。文档更新或恢复上下文不自动启动训练。
5. 给一张自包含启动卡，未完成核验则标待验证，不宣称可启动。对缺少授权的外部动作，在具体
   结果可审阅后请求最后一步授权并说明原因；已有授权不反复确认。
6. 先建每任务下载目录再给用户绝对路径。收到原始包后保留并算hash，核对合同再分析。
   状态更新写入当前项目自己的handoff/索引，不回写其他课题或本Skill。

## 关键区别

- 本机CLI登录用于上传输入，容器c2net用于挂载和回传；Token不能替代c2net，不能输出Token值。
- 默认一个昂贵任务运行一个明确单元，如arm×seed。用户明确选择多单元时逐单元上传并核定
  整个任务资源，不采用全有或全无的末尾上传。
- 模型恢复包不是完整续跑状态。auto只保证设备选择时可用CPU；被平台销毁的容器不能靠finally挽救。
- exit 0、上传100%、status complete与科学GO分别核验。失败/部分任务保持scientific_evidence=false；
  非假设检验项目用自己的验收标准，不强造GO阈值。

## 可复用工具

`scripts/artifact_manifest.py`仅流式读取明确指定文件，生成或核验bytes/SHA256清单；不登录、
联网、上传、训练或删除，输出已存在时拒绝覆盖。

```text
python <skill-dir>/scripts/artifact_manifest.py create --file <absolute-file> --output <new-manifest.json>
python <skill-dir>/scripts/artifact_manifest.py verify --manifest <manifest.json>
```

多文件重复--file；不要把凭据文件交给该工具。分享清单前检查路径信息是否适合分享。
