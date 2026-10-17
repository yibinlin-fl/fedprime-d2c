# 可选CLI与认证（不要求用户先交API密钥）

网页Notebook流程可以独立使用。只有用户要本地上传/管理/下载时才检查CLI版本与凭据存在性。
不读取或打印密钥值，不要求把kaggle.json上传进数据集或Notebook。

官方当前CLI支持OAuth、API token及旧式kaggle.json；已有正确配置不主动轮换。新CLI例子：

```text
kaggle --version
kaggle --help
kaggle auth login
kaggle quota
```

登录由用户在本机浏览器完成，已有登录时无需重复。旧CLI可能没有auth/quota或新版flags，先
检查--help；此时用网页查看配额，不凭旧教程编造命令。新CLI有Python版本要求，不擅自替换项目环境。

认证依据：[官方Authentication](https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/auth.md)。
quota报告used/remaining/total和可用refresh时间：[官方Quota](https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/quota.md)。

## Notebook本地文件

同目录放.ipynb和kernel-metadata.json。应核对id、code_file、language、kernel_type、私有性、
GPU/Internet开关及dataset_sources/kernel_sources。默认生成私有CPU、Internet关闭的设计稿，
真实T4任务再按当前CLI/UI验证设置。metadata支持Notebook与Script，语法不是OpenI启动卡。
字段依据：[官方Kernel metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md)。

```text
kaggle kernels init -p NOTEBOOK_FOLDER
kaggle kernels pull OWNER/SLUG -p LOCAL_SOURCE_FOLDER -m
kaggle kernels push -p NOTEBOOK_FOLDER --no-run
kaggle kernels push -p NOTEBOOK_FOLDER --accelerator NvidiaTeslaT4
```

以上仅语法，不是本轮执行指令。**push不带--no-run通常会上传并运行**；--no-run只在当前
CLI确认支持时使用，而且仍是外部上传，需授权。设计Skill不执行任何push。
T4 accelerator ID、push行为、output和status参数依据：[官方Kernels命令](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)。

数据上传可用网页Create Dataset，或先kaggle datasets init准备metadata，再在授权后create/version。
资源本身默认保持私有；license与内容权限由当前课题核对，不擅自填公开许可。
[官方Datasets CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md)。

不得把c2net/OpenI Token搬来Kaggle；两平台认证与结果保存机制完全不同。
