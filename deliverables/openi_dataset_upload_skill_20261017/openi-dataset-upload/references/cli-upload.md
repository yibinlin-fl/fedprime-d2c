# CLI完整流程（以本机openi 3.0.1为核验基线）

用户确认的默认分流：单文件超过500MB使用本CLI流程，不超过500MB（含500MB）默认网页
手动上传。小文件网页上传失败或用户指定CLI时也可使用；该阈值是用户操作规则。

## 1. 先确认数据集已存在

网页创建目标数据集或确认可复用的已有目标，取得准确owner/repo_id和写权限。
CLI文件上传不是创建数据集。本机3.0.1没有公开的dataset create/list命令，不发明命令补步骤。

## 2. 检查/准备本机CLI

已有可用SDK就复用，不无条件升级。当前机器已知位置：

```powershell
$openiCli = 'D:\anaconda3\envs\pytorch\Scripts\openi.exe'
& $openiCli --help
& $openiCli dataset upload --help
& 'D:\anaconda3\envs\pytorch\python.exe' -c 'from importlib.metadata import version; print(version("openi"))'
```

另一台机器用实际Python环境发现路径。确实未安装且获得环境变更授权后，可在选定环境安装
`python -m pip install openi==3.0.1`作为已核验版本；不要把旧beta或1.x教程的参数套进3.x。
官方发行记录：[openi 3.0.1](https://pypi.org/project/openi/3.0.1/)。

## 3. 登录

```powershell
& $openiCli whoami
# 只有确需登录时，由用户在自己的交互终端运行：
& $openiCli login
& $openiCli whoami
```

login会提示输入Token并保存到本机；不要把真实Token放进命令参数或上传脚本。
已有有效登录不重复login，后者会覆盖原有凭据。Token的获取和风险见token-and-local.md。

## 4. 单文件上传

以下全部占位值都要换成当前项目实际值，不原样执行：

```powershell
$openiRepoId = 'OWNER/DATASET_PATH'
$openiInputFile = 'C:\absolute\path\project_input_v1.tar.gz'
$openiRemoteName = 'project_input_v1.tar.gz'
(Get-Item -LiteralPath $openiInputFile).Length
(Get-FileHash -LiteralPath $openiInputFile -Algorithm SHA256).Hash
& $openiCli dataset upload $openiRepoId $openiInputFile --upload_name $openiRemoteName --max_workers 10
if ($LASTEXITCODE -ne 0) { throw 'OpenI CLI returned a nonzero exit code; inspect a redacted error.' }
```

| 字段 | 意义 |
|---|---|
| repo_id | 已创建目标的拥有者/数据集路径；非本机路径、不是随意显示名 |
| file_or_folder_path | 本机文件或明确目录路径，含空格必须正确引用 |
| --upload_name / -n | 单文件的远端路径及文件名，可含子目录；按入口需要保持精确名称 |
| --max_workers / -w | 最大并行文件数，3.0.1默认10，不保证一个大文件提速10倍 |

普通上传从本机凭据读取Token，不需要每次带--token。本机上传命令没有c2net或GPU参数。
不能通过把文件改名或拆分来绕开平台限制后，又让入口误读成原输入包；如需分片，先适配审计/解包。

## 5. 可选目录上传

3.0.1支持目录输入，保留内部子目录，但不含目录最外层名称；此时upload_name无效。
只在用户明确确认目录和文件范围后上传，避免递归带上凭据/缓存。科研可复现输入优先选择
明确归档文件及hash，不因CLI支持目录就默认上传整个项目目录。

## 6. 重试与验收

上传中断先核对账号、目标、文件未变和CLI版本。3.0.1历史流程支持重试/续传，但是否续上及
同名文件处理以当前SDK/远端状态为准，不盲目循环或自动删远端旧文件。网络重试有限次；
认证、权限、repo_id错误先修正原因。上传完刷新平台文件列表并验收，不能只看进度100%。
