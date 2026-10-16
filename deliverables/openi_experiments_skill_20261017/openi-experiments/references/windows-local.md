# Windows本机上传上下文

以下是用户当前Windows机器的历史配置，不是所有机器的默认；先查存在性再用。

```text
Python: D:\anaconda3\envs\pytorch\python.exe
CLI: D:\anaconda3\envs\pytorch\Scripts\openi.exe
已使用SDK: openi 3.0.1
凭据位置: C:\Users\asus\.openi\token.json
历史账号: chujiu（2026-10-14只读认证成功）
```

文件存在不等于登录有效，旧认证不是当前认证。其他机器从实际环境/命令发现定位。
不要输出token.json内容或环境中的Token，不覆盖凭据。

## 上传流程模板

新课题先确认数据允许上传及数据集可见性，不因历史用了公开数据就默认公开。
未变输入复用已上传数据；改代码不要求重传数据。先确定目标owner/repo_id。

```powershell
& 'D:\anaconda3\envs\pytorch\Scripts\openi.exe' whoami
$openiInputPath = '<absolute-file-path>'
(Get-Item -LiteralPath $openiInputPath).Length
(Get-FileHash -LiteralPath $openiInputPath -Algorithm SHA256).Hash
& 'D:\anaconda3\envs\pytorch\Scripts\openi.exe' dataset upload '<owner/dataset>' $openiInputPath --upload_name '<remote-file-name>' --max_workers 10
```

不要原样运行尖括号。上传需用户授权，参数以本机CLI的--help核对。
3.0.1历史支持同命令重试/续传；100%不替代exit code和远端验收，失败不无限重试。
历史网页单文件上限约512MB，项目因此用CLI；不将旧限制当永久平台事实。

SDK正常不主动升级。确需重新登录，由用户在OpenI官方账号设置管理凭据并在本机交互输入，
不要让用户贴Token到聊天。完整认证URL、headers或Token也不写日志。

## DLL与SDK日志

本机曾出现PATH/DLL冲突。优先使用项目已有完整启动方式；必要时仅在子进程内优先放对应env、
Library/bin、Scripts路径，退出恢复，不改系统PATH，不覆盖HOME/CODEX_HOME。
openi 3.0.1导入会配置~/.openi/logs，沙箱拒绝写日志导致的初始化错误不能判断成Token失效。
用只读诊断定位，或在授权范围处理日志初始化；不为此升级SDK、轮换凭据或输出认证信息。

本机whoami成功只证明上传输入的认证。云端仍需要c2net的dataset_path/output_path及真实
upload_output；Windows Token不能补出这些上下文，也不能延长GPU/CPU租约。
