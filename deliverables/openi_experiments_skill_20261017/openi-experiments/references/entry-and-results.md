# 云端入口、失败恢复与结果回传合同

## 适配与训练前检查

读一个实际入口及直接调用的runner/分析器，确认参数、工作目录、依赖、输入定位、输出位置。
这些是设计合同，不是假定新项目已有所有接口；复用训练逻辑，不重写算法。
以下c2net API来自本项目已验证流程；新镜像/SDK变化时查已安装实现、官方说明或最小smoke。

训练前确认：

- 实验单元与授权一致；缺失/冲突输入、重复run_id报错，不随便选首个匹配。
- c2net真实上下文、output_path及上传函数可用；缺失时停止昂贵训练。
- 包的hash/manifest与协议相符；安全解压到新目录，拒绝路径穿越、链接逃逸及覆盖旧结果。
- 记录run_id、mode、arm/seed（如适用）、代码/input/config hash、环境/GPU、预算、split、时间；
  配置和日志不含凭据。

## c2net基本连接

```python
from pathlib import Path
import shutil
from c2net.context import prepare, upload_output

context = prepare()
if context is None or not getattr(context, "output_path", None):
    raise RuntimeError("OpenI output context unavailable; do not start training")

def publish(paths):
    output_dir = Path(context.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in paths:
        source = Path(source)
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = output_dir / source.name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
    upload_output()  # 上传错误必须传播，不能无context时跳过并假称成功。
```

片段仅展示API连接，不包含校验、发布确认、重试和同名覆盖防护。真实入口使用唯一run_id/版本名，
核对SDK返回状态或远端确认能力。调用正常返回不替代最终可下载产物验收。
网络/上传重试须有限次、有界退避，不无限循环；用户未授权上传时不要执行publish。

## 每个独立实验单元的顺序

```text
训练 -> 保存checkpoint和合同
     -> 写training_complete_analysis_pending且scientific_evidence=false
     -> 打包、上传含checkpoint的recovery
     -> 确认上传调用成功，再做昂贵分析
     -> 完成全量分析与所需审计
     -> 写完整status并上传正式结果包
     -> 有用户授权时才进入下一个单元
```

默认按arm×seed等独立单元分任务。明确选择多单元时逐单元回传，不让最后一臂失败吞掉此前结果；
同时核定整个任务资源。不要误把一个单臂入口的保护当成多臂runner自动具备同样保护。

## 产物命名与内容

```text
<project>_<run_id>_recovery_outputs.tar.gz
<project>_<run_id>_outputs.tar.gz
<project>_<run_id>_status.json
```

恢复包建议包含metadata/run.json、实际configs、训练metrics/traces、checkpoints和status。
权重核对数量、架构及可加载性；不要在checkpoint中夹带凭据。
正式包包含相同任务身份、analysis、必要预测/指标缓存、审计及最终manifest/status。
可以不重复checkpoint，但必须保留并下载恢复包。

status记录run_id、mode、代码/输入/config身份、阶段、训练/分析完成与否、错误类型和简要信息、
产物名、耗时及恢复包发布状态。错误先去凭据，不把认证URL/headers原样写入报告。

区分execution_status、analysis_status、scientific_evidence与scientific_verdict：完整阴性
实验可以是有效证据但不是方法GO。没有科学GO门槛的项目用自己的验收字段。
失败/部分任务的scientific_evidence=false要进入可下载status与失败包，避免把先前构造但未
发布成功、仍标成功的包误当失败报告上传。已有完整结果后出现单独发布错误也要单列说明。

## 设备、硬中断与续跑

- auto可在分析启动时无CUDA选择CPU；未实现异常后重新建模/重试时，不承诺分析中途GPU故障自动切换。
- 容器硬销毁时except/finally不保证执行。先做分析前外部回传；长训练中途保护另做周期性外部checkpoint。
- 权重恢复包不等于可审计续跑。按算法保存模型、优化器、scheduler、AMP scaler、RNG、loader/
  sampler、轮次/步、通信/服务器状态（如适用），并验证恢复后的协议/轨迹一致。
- 可捕获异常尽力上传现存状态但保留失败标记；上传也失败时报告两者，不伪称恢复成功。
- 不自动清理路径，遵守项目删除规则，不批量删除用户数据。

## 有价值的聚焦验证

按当前改动选择，不要求每次文档更新都跑：

- 无c2net/output_path时训练前停；缺失/重复输入fail-closed。
- 恢复包含真实可用checkpoint、配置与训练身份。
- 上传失败不开始分析；分析失败仍有失败status和恢复包。
- 多单元获准时，后一单元失败不影响前一单元外部结果。
- smoke/benchmark不标科学证据；exit 0不能掩盖analysis failed。
- 模拟CUDA不可用能选择CPU；mock测试不能冒充真实云端上传验收。

## 结果交回当前项目

先建下载目录，保留原始包并生成bytes/SHA256清单；查预算、seed、模型数、配置/input身份、
完成标记、预测与指标，matched研究另查split和随机轨迹。
只有完整训练恢复包时可先做对应全量离线分析；日志无法重建缺失权重。
把执行失败、部分结果、完整阴性结果与科学GO分开，更新当前课题handoff及证据台账。
