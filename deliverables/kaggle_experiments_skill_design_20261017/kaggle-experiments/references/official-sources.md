# 官方来源与事实核验记录

查询日期：2026-10-17。只将官方文档/官方仓库和明确Kaggle Staff说明作为平台依据。
没有登录用户账号；当前剩余配额、实际硬件、验证状态和会话选项仍待运行前确认。

| 来源 | 核验范围 |
|---|---|
| [Notebooks](https://www.kaggle.com/docs/notebooks) | Notebook/Script、版本、GPU技术表、会话与输出规格 |
| [Efficient GPU usage](https://www.kaggle.com/docs/efficient-gpu-usage) | 周配额30小时或更高，按需求/资源变化，使用监测 |
| [Official Kernels CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md) | push通常运行、--no-run、T4请求、版本output/status、Secrets |
| [Kernel metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md) | code_file、kernel_type、private/GPU/Internet、数据源 |
| [Official quota](https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/quota.md) | 账号总量、已用、剩余、refreshAt |
| [Official auth](https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/auth.md) | OAuth/token/legacy配置与版本差异 |
| [Official datasets](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md) | 数据集创建/版本/下载命令 |
| [Official docker-python](https://github.com/Kaggle/docker-python) | Kaggle Python环境镜像来源 |
| [Packages](https://www.kaggle.com/docs/packages) | 依赖管理、联网和资源依赖的背景 |
| [Staff: read-only inputs](https://www.kaggle.com/general/223073) | input只读、working输出、后台独立会话；较早说明，结合当前文档 |
| [Staff: version vs persistence](https://www.kaggle.com/product-feedback/532308) | 无状态从头运行与交互Persistence的区别 |

## 以后使用时重新核验

平台可能调整硬件、额度、时限、存储和UI菜单；准备长任务时优先查看当前官方页面及账号UI。
本地已安装CLI的--help可能与main分支文档不同，记录版本而不是为套命令无条件升级。
Notebook文档含较早P100段落，使用更具体的当前T4技术表/CLI信息并实测，不混用矛盾旧数字。
非Staff社区评论、第三方博客中的9/12小时时限、自动保存、保活教程都不当成官方保证。

## 工程约定，不冒充官方要求

完整中文启动卡、每任务独立run_id、固定Git commit/hash、恢复包先于分析、分段续跑审计、
失败科学标记和本机下载目录是本Skill的科研工程约定。
用户默认网页新建Notebook，按仓库地址自行拉取代码；本Skill交可复制单元，不强制上传.ipynb。
Kaggle官方同时允许Script；.ipynb导入只是可选交付方式。
T4双卡显存不合并、多卡需代码支持是GPU编程事实，不是“每周额外60小时”的配额承诺。
