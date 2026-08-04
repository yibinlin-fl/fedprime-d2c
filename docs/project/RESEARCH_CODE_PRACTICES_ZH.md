# 研究生实验代码最佳实践

本文档面向本项目这类机器学习 / 联邦学习实验代码。目标不是把代码写得“花”，而是让实验具备：

```text
可复现
可扩展
模块清晰
容易定位问题
容易跑消融
容易整理论文结果
```

## 1. 总原则

不要把研究代码写成一次性脚本。一个长期实验项目最好像一个小型实验平台：

```text
配置决定实验差异
模块决定算法能力
日志决定可诊断性
结果文件决定可复现性
```

每次新增方法时，尽量只新增一个 `method` 文件和一个 `config` 文件，而不是复制一整套训练脚本。

## 2. 推荐目录结构

```text
fedprime/
  data/            # 数据加载、划分、损坏数据处理
  models/          # 模型构造、统一 forward 接口
  methods/         # 具体算法：RAHFL、D2C、PAIR、PRAC-HFL
  engine/          # 通用训练、评估、保存逻辑
  utils/           # seed、device、config、logging 等工具

configs/           # 所有实验配置
scripts/           # 命令入口：run_experiment、run_grid、summary、diagnosis
deliverables/      # 手动整理给老师或论文看的图表和文档
outputs/           # 实验自动输出，通常不进 Git
```

核心思想：

```text
代码负责能力
配置负责差异
outputs 负责结果
deliverables 负责汇报
```

## 3. 配置驱动实验

每个实验都应该由一个 YAML 配置完整描述。配置中至少包含：

```yaml
experiment_name:
method_name:
seed:
device:
data:
models:
train:
method:
checkpoints:
```

这样做的好处：

```text
1. Kaggle / 服务器 / 本地可以共用同一套入口。
2. 消融实验只需要改配置，不需要复制代码。
3. 写论文表格时，可以回溯每个结果对应的配置。
```

关键超参数不要硬编码在方法文件里。如果某个值会影响实验结论，就应该出现在 config 里。

## 4. 方法模块边界

每个算法最好有一个清晰的 Experiment 类：

```python
class PRACHFLExperiment:
    def run(self):
        ...
```

内部建议拆成：

```text
_build_optimizer()
_local_phase()
_communication_phase()
_evaluate()
_save_models()
_load_models_if_configured()
```

这样定位问题时，可以快速判断问题发生在：

```text
数据加载
本地训练
通信模块
评估模块
保存和恢复
```

## 5. 统一 Runner

不要给每个方法写一套启动脚本。推荐统一入口：

```text
scripts/run_experiment.py --config configs/xxx.yaml
```

Runner 只负责：

```text
1. 读取配置
2. 设置随机种子
3. 构造数据、模型、方法类
4. 调用 experiment.run()
5. 汇总结果
```

方法差异由 `method_name` 分发，例如：

```text
rahfl
fedprime_d2c
fedprime_pair
prac_hfl
```

## 6. 日志必须能诊断问题

机器学习实验经常一跑就是几小时。日志不能只在最后输出结果，尤其 Kaggle 后台跑时更需要心跳日志。

每轮至少打印：

```text
round
avg_acc
worst_acc
local_loss
method_loss
elapsed
```

长循环中建议打印：

```text
[heartbeat] round 000 local client 0 start
[heartbeat] local phase batch=50 loss=...
[heartbeat] communication phase start
[heartbeat] evaluating clients
```

这样一旦 30 分钟没有准确率输出，也能判断是在：

```text
下载数据
复制数据
本地训练
通信
评估
还是日志缓冲
```

## 7. Kaggle / 服务器实践

Kaggle 一次性后台运行时，最容易出问题的是“没有任何输出”。启动脚本应做到：

```text
1. clone 后立刻打印 git log -1 --oneline。
2. 立刻检查 CUDA、GPU 名称。
3. 立刻检查数据路径。
4. 训练长循环中打印 heartbeat。
5. 使用已挂载数据时，不再重新下载数据。
```

本项目的 Kaggle 数据集通常挂载为：

```text
/kaggle/input/fedprime-data
```

应复制到：

```text
RAHFL-master/Dataset/cifar_10_c
RAHFL-master/Dataset/cifar_100
outputs/partitions
```

## 8. 可复现性

正式实验必须固定：

```text
seed
数据划分文件
客户端模型列表
训练轮数
local_epochs
batch_size
public_batches_per_round
public_batch_size
学习率
损坏率
测试集
```

本项目已经支持保存 / 复用 partition 文件。公平比较时，RAHFL、PRAC-HFL、其他消融都应该读取同一个 partition。

## 9. 数值稳定性

研究代码不要等 NaN 出现后才排查。推荐常规保护：

```text
1. 对关键 loss 检查 torch.isfinite。
2. 对关键梯度检查非有限值。
3. 必要时跳过坏 batch。
4. 大步长蒸馏或虚拟更新要有 max_grad_norm。
5. 先 warmup 再通信，避免随机初始化阶段互相污染。
```

但也要注意公平性：如果给自己的方法加了梯度裁剪，而 baseline 没有，就需要在论文里说明，或者给 baseline 也使用同样的数值稳定策略。

## 10. 消融实验写法

一个新方法不要只跑最终版本。至少保留这些开关：

```text
use_module_a: true / false
use_module_b: true / false
warmup_rounds: 0 / 3
lambda_x: 0.1 / 0.5 / 1.0
```

开关放在 config，而不是改代码。这样同一套代码可以跑：

```text
完整方法
去掉通信模块
去掉本地增强
去掉门控
只保留基础蒸馏
```

## 11. 输出和汇报分离

建议约定：

```text
outputs/       自动实验输出，不上传 Git
deliverables/  人工整理后的图表、Excel、PDF、Markdown，可以按需上传
```

`outputs/` 往往包含大量日志、checkpoint、npy、npz，不适合进 Git。需要给老师看的内容应该整理成 `deliverables/`。

## 12. Git 实践

推荐做法：

```text
1. 一组相关代码改完并测试后再提交。
2. 提交信息说明“做了什么”，不要只写 update。
3. 大文件、数据集、checkpoint、outputs 不进 Git。
4. 论文图表可以放 deliverables，但要确认体积合理。
5. 每次跑 Kaggle 前确认远程最新 commit。
```

实验代码可以多次提交，但不要每改一行就提交。比较好的粒度是：

```text
一个完整模块
一个可运行实验入口
一次数值稳定性修复
一次文档/记忆同步
```

## 13. 定位问题的顺序

实验效果不好时，不要马上改方法。建议按这个顺序查：

```text
1. 数据是否正确加载？
2. 数据划分是否符合预期？
3. baseline 是否能复现？
4. 本地训练是否正常下降？
5. 通信 loss 是否数值合理？
6. 指标是否在同一测试集上算？
7. 是否存在测试集泄漏？
8. 是否有 NaN / 梯度爆炸？
9. 是否和 baseline 使用同一配置？
```

## 14. 本项目当前经验

本项目已经验证出几个重要经验：

```text
D2C 的 predicted prior 在跨域 CIFAR-100 public logits 上不可靠。
FedPRIME-PAIR 的类别对 public-logit 蒸馏效果弱于预期。
RAHFL 的强度很大程度来自 AugMix + DCL 的本地鲁棒训练。
PRAC-HFL 目前是最值得继续推进的方向。
Kaggle 必须有 heartbeat，否则长时间无输出会严重影响判断。
```

后续最重要的是：

```text
稳定跑完 safe PRAC-HFL。
如果超过或接近 RAHFL，再做多 seed。
如果仍低于 RAHFL，优先分析通信 accept_rate、avg_delta、worst_acc，而不是盲目堆模块。
```
