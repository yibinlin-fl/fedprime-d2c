# 可照填的Kaggle启动卡

每个任务给一张自包含中文卡，未知值标待核验，不把下面示例参数当项目合同。

## 身份与平台设置

```text
实验目的、run_id、允许证明什么、mode及授权范围：
Notebook本地.ipynb绝对路径：
Notebook owner/slug、Private/Public、目标版本：
代码仓库、分支、完整commit、是否push；或私有源码包版本/hash：
数据集owner/slug/版本、输入包绝对本地路径/bytes/SHA256：
实际/kaggle/input挂载路径：
Language：Python
Accelerator：实际可用T4配置（先实测）
Internet：On/Off及原因
Environment/依赖：实际平台设置和锁定依赖
Secrets标签：只列标签，不列值
GPU本周total/remaining/reset、当前session上限：
```

Notebook配置表逐项列出真实入口需要的字段：mode、seed、预算、数据/输出路径、run_id、
准确runner命令与分析命令；无该字段则不虚构。与OpenI不同，这些通常填写在Notebook配置cell，
不是网页启动参数栏。必须同时交付对应.ipynb，不只给一条Linux命令。

## 时长、保存与交回

```text
benchmark依据、训练/分析/打包预计耗时与余量：
单GPU/双GPU实际使用方式、每卡显存、batch/AMP等协议：
运行方式：交互调试或Save & Run All后台执行
是否会从头重跑、哪些手动状态不能复用：
GPU会话何时开启、何时停止：
云端产物目录：/kaggle/working/<run_id>/
精确恢复包/正式包/status/manifest文件名：
保存版本步骤与Output文件验收：
已预建的本机绝对下载目录：
数据/配置/版本/配对校验与指标、冻结科学门槛或项目验收：
中断时能恢复什么、是否已验证完整续跑：
```

先CPU预检，免费GPUbenchmark也按用户授权范围启动。不因额度充足就自动扩大到全部seed/arm。
Save & Run All会新启动执行；CLI push通常也启动执行。已有明确授权不重复请求，但设计/安装Skill
的授权不等于上传Notebook或启动任务。
