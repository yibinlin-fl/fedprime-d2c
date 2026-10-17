# Token获取、复用及本机上下文

## 本机事实

```text
Python: D:\anaconda3\envs\pytorch\python.exe
CLI: D:\anaconda3\envs\pytorch\Scripts\openi.exe
核验版本: openi 3.0.1
凭据文件: C:\Users\asus\.openi\token.json
历史账号: chujiu（2026-10-14只读认证成功）
```

2026-10-17创建本Skill时再次确认CLI和凭据文件存在，未读取/输出Token值，未重新登录或在线
验证账号。存在不等于当前有效；在实际上传前按需要运行whoami。其他机器从实际用户环境定位。
不把以上路径、用户名硬套到所有机器，更不把旧项目数据集当新项目默认目标。

## 没有Token时怎么获取

1. 用户登录正确的`https://openi.pcl.ac.cn`账号。
2. 打开个人设置 → 应用/访问令牌，官方入口为
   `https://openi.pcl.ac.cn/user/settings/applications`；以当前界面为准。
3. 按页面要求创建用于本机上传的令牌，取容易识别的用途名称；权限按实际支持范围满足目标
   写权限即可，不臆造固定scope名称。若是组织数据，账号也必须有相应组织/数据集权限。
4. 用户仅在自己的终端执行openi login，并把Token粘贴到交互提示框。正常输入被屏蔽，登录
   成功会持久保存；接着用whoami确认用户名。

不要让用户把Token发到聊天，不把令牌写进Skill、README、实验配置、Git、上传计划或.ps1。
不把Token放入数据集文件，也不打印token.json、环境变量或带认证URL/headers。

## 已有Token怎么复用

普通whoami/upload通过SDK读取本机已保存凭据，无需把Token复制给另一个窗口。新窗口只需知道
凭据位置、实际CLI路径和验证命令。不要主动logout、撤销、轮换或覆盖登录状态。
如需切换账号，先确认目标身份和用户意图，再由用户交互登录，避免覆盖当前可用账号。

## 本机3.0.1的具体注意事项

已检查本机openi/_login.py：login不带token参数时调用屏蔽输入；成功保存凭据；重新登录会
覆盖原文件。但是401分支可能将无效Token打印到终端。因此不要将登录失败的原始输出、截图
或终端日志发到聊天/提交仓库。登录由用户在本机进行；需要代理执行时应先实现输出去敏流程，
不直接让工具回传原始login输出。本Skill不提供把真实Token写在--token参数里的示例。

日志权限、DLL冲突或代理失败不等于Token失效：本机openi导入时会初始化~/.openi/logs，沙箱
拒写可能先触发日志配置错误。先定位错误类型，不为此重新登录/升级SDK；诊断不输出敏感错误全文。
环境PATH若需调整只在当前子进程中做，不改系统PATH或覆盖HOME/CODEX_HOME。

## 作用范围

Token用于本机访问OpenI API，不是云端数据路径，不延长算力租约，也不能替代训练容器的c2net
上下文。数据集上传完成不等于训练代码已推送、任务已挂载或结果可以回传。
