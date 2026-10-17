---
name: openi-dataset-upload
description: "专门处理OpenI/启智社区输入数据集的创建与上传：先在平台建立/确认数据集名称、可见性和owner/repo_id，再选择网页手动上传或OpenI CLI，复用本机Token、生成命令并核验大小/SHA256。用户说“创建OpenI数据集”“上传OpenI数据”“OpenI CLI和Token怎么用”时使用；训练入口及c2net结果回传属于openi-experiments。"
---

# OpenI 数据集上传助手

为任意项目准备或执行“先有目标数据集，再上传输入文件”的完整流程，交付可照填网页字段、
准确repo_id、手动/CLI操作、文件身份与验收记录。平台资料与本机CLI核验日期：2026-10-17。

## 先恢复最少上下文

遵守当前项目AGENTS及其handoff/索引；检查拟上传文件和相关Git状态，不扫描整个仓库或历史。
明确本次是教程/准备命令、创建数据集、手动上传指导、执行CLI，还是失败排查。用户已授权的
动作继续完成；只要求创建Skill或教程不代表授权上传任何现有数据、公开数据或启动训练。

## 工作流

1. 确认明确文件/包、绝对路径、用途、版本、包内结构，计算精确bytes和SHA256。
   改代码不要求重传未变数据；避免误选同名旧包或把凭据/缓存打进数据集。
2. **先在OpenI网页新建空数据集，或确认现有目标可复用**。记录显示名、拥有者、平台提供的
   owner/repo_id、页面URL、可见性及写权限；不能只凭显示名猜CLI地址。
3. 在目标建立后选择网页手动上传或CLI。不要发明openi dataset create命令；本机3.0.1的
   dataset子命令仅upload/download。若平台以后变化，先查实际--help和官方说明。
4. CLI优先复用现有登录；只有缺失/失效且用户选择本机登录时才交互输入Token。
   不索要Token明文，不读出token.json、不把Token写进上传命令、Skill或日志。
5. 给一张完整上传卡：文件身份、目标/可见性、方法、精确命令或网页步骤、验收方法和授权状态。
   外部创建/上传尚未授权时，先完成准备再请求具体授权；不要对已有授权重复确认。
6. 验收远端文件可见、路径/大小正确及处理完成；如需要强完整性证明再核对平台hash或独立下载
   重算。不把进度100%、exit 0或本地hash冒充远端SHA256已验证。上传记录写入当前项目。

## 按需读取

- 新建目标和手动上传：读[网页创建与手动上传](references/web-create-and-manual.md)。
- CLI安装/参数/单文件与文件夹：读[CLI上传步骤](references/cli-upload.md)。
- Token、本机路径、账号切换：读[Token与本机上下文](references/token-and-local.md)。
- 上传后验证、错误、重试：读[验收与排错](references/verification-and-errors.md)。
- 平台/SDK更新：读[核验来源](references/source-notes.md)，不把旧网页限制当永久规则。

## 工具与边界

`scripts/prepare_upload.py`仅为一个明确普通文件生成本地upload-plan.json和upload-command.ps1，
包含bytes/SHA256与正确PowerShell引号，不登录、不联网、不创建远端数据集、不执行上传。

```text
python <skill-dir>/scripts/prepare_upload.py --repo-id OWNER/DATASET --file ABSOLUTE_FILE --output-dir NEW_FOLDER
```

目标是否已创建、远端是否验收都保留未核验状态，不能因生成命令就标完成。输出目录存在时拒绝覆盖。
Skill本身不保存Token，不调用OpenI的c2net，不默认切账号、升级SDK、重传或删除远端文件。
