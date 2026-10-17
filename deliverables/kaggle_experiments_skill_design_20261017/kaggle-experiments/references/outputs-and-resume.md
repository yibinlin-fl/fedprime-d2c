# checkpoint、结果保存、下载及跨会话续跑

## 建议的结果树（工程约定）

```text
/kaggle/working/<project>_<run_id>/
  metadata/run.json        代码/input/config hash、mode、环境、实际GPU、版本身份
  configs/resolved.json    实际配置
  logs/train.log
  metrics/...
  checkpoints/...          权重；若要续跑，增加完整训练状态
  analysis/...
  status.json
  <run_id>_recovery_outputs.tar.gz
  <run_id>_outputs.tar.gz
  artifact_manifest.json
```

路径树是通用设计，不强制模型checkpoint格式或科研指标。源码、.git、整个数据包、pip cache
不放在最终输出根目录，避免容量浪费或误带凭据。原始文件与两份压缩包共存时要预算总空间。

训练按合理间隔保存checkpoint；训练返回后立即整理恢复包，再做昂贵分析。恢复包含模型、配置、
训练日志与未完成分析的status。最终包包含分析、必要预测缓存、训练身份和最终status；可不重复
checkpoint，但恢复包必须保留并下载。打包明确目录，不能把正在生成的tar包自身递归打进去。

## Kaggle持久保存与OpenI不同

Kaggle不调用c2net上传结果。写入working是第一步，下一步是成功保存Notebook版本并检查该版本
Output，最后才是下载或挂入下一Notebook。不要只保存代码、只打印FileLink或只在/tmp留权重。

已保存输出可作为下一Notebook输入：[官方Notebook输出复用说明](https://www.kaggle.com/docs/notebooks)。
未成功保存版本时，只能按当前交互会话实际能力手动下载，不保证进程硬停止后还能取回全部文件。

## 下载回当前项目

先创建用户本机每任务专属目录，再给绝对路径，例如当前项目约定的outputs/kaggle_downloads/<run_id>。
不要因为通用Skill源于FedPRIME就默认下载回FedPRIME。

网页：对应Notebook版本 → Output → 下载恢复包、正式包、status、manifest及必要日志。
CLI：先查当前版本/文件列表，再下载明确版本到新目录；不使用强制覆盖旧结果。

```text
kaggle kernels status OWNER/NOTEBOOK/VERSION
kaggle kernels files OWNER/NOTEBOOK/VERSION
kaggle kernels output OWNER/NOTEBOOK/VERSION -p LOCAL_DOWNLOAD_DIRECTORY
```

命令和版本寻址依据：[官方Kernels CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)。
实际flags以已安装版本--help为准，记录Notebook版本号，不能把latest悄悄替换成另一次任务。
CLI鉴权失败不等于云端结果丢失，可先通过网页核验。

本地验收核对：原始包bytes/hash、run_id、Notebook版本、commit、输入/配置hash、预算与seed、
checkpoint数/可加载性、日志完整性、指标/分析与配对合同。科学判定按当前课题预注册协议；
失败、部分训练、未完成分析保持scientific_evidence=false。平台status complete也不能替代科学GO。

## 跨会话续跑

单会话时限与每周额度是不同约束。预计超时的训练要提前冻结分段协议，主动在安全点停下并保存
可下载版本，不等平台硬杀。下次把已保存结果挂载为input，再恢复到新的working目录。

完整续跑通常需要模型、优化器、scheduler、AMP scaler、随机状态、sampler/loader状态、epoch/step、
算法的额外服务器/通信状态（如适用），并核对配置/input/source身份。只有模型权重时应称继续
微调或重新初始化优化器，不能宣称无缝续跑。需验证恢复前后轨迹，而不只验证能load权重。

auto仅指分析开始选GPU/CPU，不挽救被销毁的Notebook会话。必须有已经保存到外部的产物才谈
跨会话恢复；本地checkpoint、atomic写盘或finally都不是云端持久性保证。
