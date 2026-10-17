# 拉取项目、挂载输入、依赖与Secrets

## 公有GitHub仓库

联网Notebook中把代码拉到临时目录，如/tmp/kaggle-project；所需最终产物始终另写working。
下面是工程示例，参数是当前项目实值；目标已存在时检查身份或选新目录，不删除旧目录重来。

```python
from pathlib import Path
import subprocess

repo_url = "https://github.com/OWNER/REPO.git"
commit = "填写完整40位commit"
repo_dir = Path("/tmp/kaggle-project")
subprocess.run(["git", "clone", "--no-checkout", repo_url, str(repo_dir)], check=True)
subprocess.run(["git", "-C", str(repo_dir), "checkout", "--detach", commit], check=True)
actual = subprocess.check_output(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True).strip()
assert actual == commit
```

不能只git pull main就把它称为冻结版本。本机未提交/未推送的代码无法从GitHub取得；需要用户
明确授权push，或在授权范围内将明确源码文件打包为私有Dataset，并记录源码包SHA256。
Git LFS/子模块另核对所需内容和凭据，不能只看到clone成功就宣布源码完整。

## 私有仓库与离线源码

私有代码不因跑Kaggle改公开。优先上传经核对的私有源码包并挂载；拷贝/解压到临时可写目录，
核对source manifest与hash。数据集压缩包先安全检查成员路径和链接，不覆盖旧目录。

若必须私有Git认证，使用最小权限Kaggle Secrets与临时认证机制，不把Token放在URL、argv、
Notebook源码、git remote、输出包或打印内容中。凭据细节单独适配，模板不包含带Token的clone。
Secrets接口来源：[官方CLI Secrets章节](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md#using-secrets-in-kernels)。

## 数据输入与输出目录分工

| 路径 | 用途 | 约定 |
|---|---|---|
| /kaggle/input/... | 已挂载Dataset/模型/Notebook输出 | 只读，实际路径从Input面板确认 |
| /kaggle/working/<run_id>/ | 需要保存和交回的产物 | checkpoints/configs/logs/analysis/archives |
| /tmp/...等临时区 | Git源码、下载中间件、大缓存 | 会话结束可丢失，不是结果存放处 |

只读输入与输出来源：[Kaggle工作人员说明](https://www.kaggle.com/general/223073)。
同一Dataset更新后要确认Notebook已挂到所需版本，记录版本/hash；不凭slug猜包内目录。
挂载已有数据不要求在Notebook中用API重新下载一遍，也不要求把API凭据上传进Dataset。

## 环境与依赖

先记录平台预装Python/torch/CUDA及镜像相关信息，再安装确实缺少的依赖。不要无条件pip install
--upgrade torch破坏平台GPU兼容性。设置可写cache路径，避免库向只读input写缓存。
在线pip需要Internet；离线提前准备兼容wheel或按平台Dependency Manager方式保存依赖。
平台镜像来源：[Kaggle官方docker-python](https://github.com/Kaggle/docker-python)。
依赖机制：[官方Packages说明](https://www.kaggle.com/docs/packages)。

平台支持持久输入不代表新项目数据允许公开，创建Dataset与Notebook时都检查可见性、许可及用户授权。
