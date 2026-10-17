# OpenI 数据集创建、手动上传与 CLI 完整指南

Updated: 2026-10-17

本指南配套独立跨项目 Skill `openi-dataset-upload`。维护源码：
`deliverables/openi_dataset_upload_skill_20261017/openi-dataset-upload/`；安装位置：
`C:\Users\asus\.codex\skills\openi-dataset-upload`。本轮只创建和安装工具，没有创建远端数据集、
登录、读取 Token 明文或上传数据。训练启动和 c2net 结果回传仍使用 `openi-experiments`。

## 1. 先准备文件，再在平台创建目标

确认本地输入包的版本、用途、绝对路径及包内根目录。不要选择旧版同名包，也不要包含
Token、`.env`、`.git`、缓存或其他账号凭据。上传前记录精确大小和 SHA256：

```powershell
$openiInputFile = 'C:\absolute\path\myproject_input_v1.tar.gz'
(Get-Item -LiteralPath $openiInputFile).Length
(Get-FileHash -LiteralPath $openiInputFile -Algorithm SHA256).Hash
```

以上路径是示例，执行前换成当前项目真实文件。文件变更后重新计算。

登录 [OpenI 启智社区](https://openi.pcl.ac.cn)，确认个人或组织身份，进入实际数据集管理入口，
选择新建/创建数据集。若当前页面把数据集挂在项目内，就从对应项目的数据集页进入。
先给用户完整建集卡，再创建；已有可写目标可直接复用，不必重复建同名数据集：

| 字段 | 新项目示例与说明 |
|---|---|
| 显示名 | `MyProject-Input-v1`，按当前项目替换 |
| 所属个人/组织 | 当前用户或具有写权限的组织 |
| 简介 | 输入数据内容、版本、训练/评价用途与入口所需目录结构 |
| 可见性 | 按用户要求；未明确时待定，不能默认公开 |
| 关联项目/存储位置 | 仅按实际页面的必填项确认，保证后续任务能挂载 |
| 文件 | 明确绝对路径、bytes、SHA256 |
| 创建后记录 | 页面 URL、显示名、拥有者、准确 `owner/repo_id`、可见性与写权限 |

先创建空目标再传文件；若当前 UI 要求创建时同时选文件，则按实际页面完成。
**CLI 使用的 `OWNER/DATASET_PATH` 不保证等于中文显示名，也不能凭任意 URL 片段猜测。**
从平台上传示例或对应配置确认准确值。本机 `openi 3.0.1` 没有 `dataset create` 命令，
不能用上传命令代替建集。

## 2. 路线 A：网页手动上传

进入目标数据集 → 上传/文件区域 → 选择或拖入文件 → 核对目标及远端名称 → 开始上传 →
等待处理完成 → 刷新文件列表，核对名称、路径和大小。

网页文件选择器填写本机路径；这不是训练容器挂载路径。平台接受压缩包也不保证按训练入口
期望自动解压。当前页面不接受文件大小、上传不稳定或用户偏好终端时改用 CLI，复用同一目标。
历史项目曾记录网页单文件 512MB 限制，本轮未核实当前上限，不将其作为永久平台规则。

## 3. 路线 B：CLI 上传

本机已安装并核对 `openi 3.0.1`：

```powershell
$openiCli = 'D:\anaconda3\envs\pytorch\Scripts\openi.exe'
& $openiCli --help
& $openiCli dataset upload --help
& $openiCli whoami
```

已有可用环境就复用。新机器确实未安装时，在选定 Python 环境按官方发行安装
`python -m pip install openi==3.0.1`，随后核对当前 `--help`；不要无条件升级旧项目环境。
[官方发行页](https://pypi.org/project/openi/3.0.1/)列出版本、Python 要求和官方帮助链接。

### Token 获取与本机登录

1. 登录正确账号，进入个人设置 → 应用/访问令牌；入口为
   [OpenI 应用设置](https://openi.pcl.ac.cn/user/settings/applications)，按钮以实际页面为准。
2. 创建本机上传用途的令牌，按当前页面支持范围选择满足目标写权限的权限；不虚构 scope 名称。
   组织数据还需账号本身具备对应写权限。
3. 在用户自己的交互终端运行 `& $openiCli login`，将 Token 粘贴到屏蔽输入的提示中。
4. 再执行 `& $openiCli whoami`，确认用户名。

本机凭据位置：`C:\Users\asus\.openi\token.json`。其他窗口使用现有 CLI 即可复用，
无需把 Token 发到聊天。历史验证账号是 `chujiu`（2026-10-14）；2026-10-17 只确认凭据文件
存在，没有读取值或重新验证在线身份。已有有效登录不重复 login，重新登录会覆盖现有凭据。

本机 3.0.1 的无效 Token/401 分支可能回显 Token。因此失败终端截图、原始日志不要直接分享；
登录由用户在本地完成。Skill 不把 Token 写进参数、脚本、Git、上传包或清单。

### 单文件准确命令

先在网页建好目标，替换下面全部占位值：

```powershell
$openiRepoId = 'OWNER/DATASET_PATH'
$openiInputFile = 'C:\absolute\path\myproject_input_v1.tar.gz'
$openiRemoteName = 'myproject_input_v1.tar.gz'
& $openiCli dataset upload $openiRepoId $openiInputFile --upload_name $openiRemoteName --max_workers 10
if ($LASTEXITCODE -ne 0) { throw 'OpenI upload failed; inspect a redacted error.' }
```

`--upload_name`（`-n`）是远端名称/相对路径；`--max_workers`（`-w`）是最大并行文件数，默认 10，
不是一个大文件必然加速十倍。普通上传直接复用本机凭据，不需要 `--token`。
目录输入虽受支持，但保留内部子目录、不含最外层目录名，且此时 `upload_name` 无效。
只上传用户明确授权的范围，不默认把整个项目目录传上去。

## 4. 上传后验收与常见失败

上传进度 100% 或 CLI 返回 0 后，还要进入正确数据集核对远端路径、文件名、大小、处理状态。
本地 SHA256 只证明源文件身份；需要远端内容完整性证明时，核对平台可信 hash，或独立下载到
新建目录再重算 hash，避免覆盖本地源文件。本机 3.0.1 下载语法：

```text
openi dataset download OWNER/DATASET --filename REMOTE_FILENAME --local_dir NEW_LOCAL_DIRECTORY
```

401 先查登录状态；403 先查目标写权限；404 先查是否已建集及 repo_id；进度完成但无文件先查
合并处理状态、目标或子目录；DLL/日志目录错误先排环境，不直接判定 Token 失效。
同名旧包先对比身份，未经授权不删除或覆盖；网络故障有限次重试，不盲目循环上传。

将目标、远端名称、本地 hash、CLI 版本和实际验收层级写进**当前项目**的数据记录。
只确认可见/大小时注明“远端内容 hash 未独立验证”。上传完成不等于代码已推送或实验可启动。

## 5. Skill 用法与本地辅助工具

新项目/新窗口可以说：

```text
使用 $openi-dataset-upload。
我要把 C:\MyProject\local_runs\myproject_input_v1.tar.gz 上传到 OpenI。
先给我建数据集要填的名称、简介、可见性和字段，再给网页及 CLI 两种上传步骤。
复用本机已有 Token，不输出明文；我建好后提供数据集页面和准确 repo_id。
```

提供准确目标后，可要求生成上传卡和命令；明确要求执行上传，代理才执行授权范围内的上传。
只请求教程/Skill 不代表授权上传现有项目数据。

`scripts/prepare_upload.py` 为一个明确文件计算 hash 并生成 `upload-plan.json` 和
`upload-command.ps1`，不登录、不联网、不上传。输出目录必须不存在，避免覆盖：

```powershell
& 'D:\anaconda3\envs\pytorch\python.exe' -X utf8 `
  'C:\Users\asus\.codex\skills\openi-dataset-upload\scripts\prepare_upload.py' `
  --repo-id 'OWNER/DATASET_PATH' `
  --file 'C:\absolute\path\myproject_input_v1.tar.gz' `
  --cli 'D:\anaconda3\envs\pytorch\Scripts\openi.exe' `
  --visibility private `
  --output-dir 'C:\MyProject\local_runs\openi_upload_plan_v1'
```

真实调用应使用当前项目已有父目录和新输出目录。脚本会创建该目录；它没有证明目标已创建、
写权限有效或远端内容已验证。生成的 `.ps1` 需核对目标及用户授权后再执行。

Skill 附带五份按需文档：网页创建与手动上传、CLI 上传、Token 与本机上下文、验收与排错、
核验来源。官方帮助入口本轮返回临时维护提示，源码页面抓取 403；CLI 语法已通过本机实现与
实际 `--help` 复核。未宣称核实当前网页布局/上限，不混用其他同名平台或旧版 beta 的语法。

安装验证：8 个明确文件 SHA256 与维护源码一致，Skill 格式校验通过；生成器通过正常准备、
禁止覆盖及 12 类错误输入检查（共 14 项），生成的 PowerShell 通过语法和字面路径核验。
检查仅使用本地合成小文件，没有执行生成的上传命令。
