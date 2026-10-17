# 来源与版本边界（2026-10-17）

## 已核验依据

- [官方发行的openi 3.0.1](https://pypi.org/project/openi/3.0.1/)：版本、Python要求及官方帮助入口。
- 本机安装包`D:\anaconda3\envs\pytorch\Lib\site-packages\openi\commands\dataset.py`：
  upload/download参数、目录结构、upload_name与max_workers含义；实际运行upload --help复核。
- 同包commands/openi_cli.py、commands/user.py：命令分组、login/whoami；没有dataset create。
- 同包_login.py：屏蔽交互输入、成功写凭据、覆盖登录以及无效Token可能回显的行为。
- 当前项目已验证的历史CLI上传流程，仅用于说明本机路径和经验，不作为新课题的输入/授权。

官方PyPI指向的[帮助入口](https://openi.pcl.ac.cn/docs/index.html)本轮返回临时提示页面，
[源码入口](https://openi.pcl.ac.cn/OpenIOSSG/openi-pypi)抓取返回403，因此没有声称在线核实
当前数据集创建页布局、文件大小上限或各存储集群选项；实际操作按用户当前页面确认。
PyPI正文中的示例也可能落后于实际3.0.1 argparse，操作语法优先核对已安装版本--help。

## 不混用的平台

目标是openi.pcl.ac.cn启智社区及其openi SDK。不要把OpenXLab/OpenDataLab、其他OpenI同名
站点或OpenAI的CLI/Token指令拼进来。旧openi 1.x、openi-beta示例的参数也不混用。

## 本Skill的工程约定

先网页创建/确认数据集、记录显示名与repo_id区别、明确可见性、计算hash、给上传卡、按证据
层级验收，是为了可靠复现。用户2026-10-17明确指定默认按单文件500MB分流：不超过500MB
网页手动，超过500MB用CLI。这是用户操作规则，不是本轮在线核验的官方硬限制；历史512MB
记录不用于默认分流。没有保证CLI能自动建集。实际SDK/网页变更时先查官方信息和本机帮助。
