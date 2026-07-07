# FedSARA-CS 场景、实现与启智运行指南

更新时间：2026-07-08

## 1. 新场景：Corruption-Skew Heterogeneous FL

原来的 RAHFL-style 数据协议主要是：

```text
模型异构 + label-skew Non-IID + 全局随机 corruption
```

新的 FedSARA-CS 场景进一步强调：

```text
模型异构 + label-skew Non-IID + corruption-skew Non-IID
```

也就是说，每个客户端不仅类别分布不同，而且遭遇的数据损坏类型也不同：

```text
client 0: 主要 noise
client 1: 主要 blur
client 2: 主要 weather
client 3: 主要 digital
```

当前协议参数：

```text
alpha = 0.5     # label-skew Dirichlet 非独立同分布强度
rho   = 0.7     # 每个客户端主导 corruption group 的比例
seed  = 0
clients = 4
samples_per_client = 10000
```

测试集是 corruption-group balanced：

```text
noise / blur / weather / digital 各覆盖一遍 CIFAR-10 test
```

因此除了平均准确率和最差客户端准确率，还能看：

```text
worst_group_acc
worst_client_group_acc
```

这两个指标更贴合新场景：方法不能只在某一类 corruption 上强，而要在最差损坏组、最差客户端-损坏组上也稳。

## 2. 当前方法对比

本轮正式要跑两个主实验：

```text
RAHFL-CS:
  AugMix/JSD + DCL + AsymHFL
  config: configs/openi_v100_rahfl_cs_alpha05_rho07.yaml

FedSARA-CS:
  AugMix/JSD + SARA + CS-AsymHFL
  config: configs/openi_v100_fedsara_cs_alpha05_rho07.yaml
```

两个配置都开启：

```text
pretrain_epochs: 40
rounds: 40
batch_size: 64
public_batches_per_round: 4
public_batch_size: 128
```

预训练阶段使用同一套 plain local CE loader，不使用 AugMix 三视图，避免不必要的计算浪费。正式通信轮仍然使用 AugMix/JSD + DCL 或 SARA。

## 3. FedSARA-CS 的核心思想

FedSARA-CS 的本地部分仍然承认 RAHFL 是强基座：

```text
AugMix/JSD 提供抗数据损坏的强增强约束。
SARA 替换 DCL，让 robust contrastive alignment 感知 label-skew。
```

通信部分不是完全替换 AsymHFL，而是在 AsymHFL 上加入 corruption-skew 感知项：

```text
CS-AsymHFL = base AsymHFL + corruption-consistency residual KD
```

服务器公共数据仍然是 CIFAR-100 public images。通信时对同一 public image 构造多种 corruption probe：

```text
clean
noise
blur
weather
digital
```

如果一个教师客户端在某个 corruption probe 下对 public image 的输出更稳定、更一致，则它对该 corruption context 的蒸馏权重更高。接收端如果在该 probe 下更不稳定，则学习需求更高。

当前实现文件：

```text
fedprime/methods/rahfl_asymhfl.py
fedprime/methods/sara.py
fedprime/data/corruptions.py
fedprime/data/loaders.py
```

入口：

```text
method_name: fedsara_cs
```

## 4. 数据包

正式数据包已生成：

```text
local_runs/fedsara_cs_prepared/fedsara_cs_prepared_alpha05_rho07_seed0.tar.gz
```

大小约：

```text
386 MB
```

包内结构：

```text
cifar_10_cs/
  alpha05_rho07_seed0/
    client_0/
      train_images.npy
      train_labels.npy
      train_corruption_ids.npy
      train_corruption_method_ids.npy
    ...
    test_balanced/
      test_images.npy
      test_labels.npy
      test_corruption_ids.npy
    metadata.json
    audit/
      client_label_counts.csv
      client_corruption_counts.csv

cifar_100/
  cifar-100-python.tar.gz
```

注意：`.npy` 是 numpy 数组文件，不是一张张图片。这样读写速度更快，也更容易保证每个客户端的划分和 corruption 协议完全固定。

## 5. 本地验证状态

已通过 smoke test：

```text
configs/debug_fedsara_cs.yaml
configs/debug_rahfl_cs.yaml
```

验证内容：

```text
1. corruption-skew 数据可读取
2. CIFAR-100 public loader 可在 torchvision 失败时直接读取 tar
3. RAHFL-CS 能完整跑 1 轮
4. FedSARA-CS 能完整跑 1 轮
5. metrics.csv / corruption_group_acc.csv / client_group_acc.csv 可输出
```

## 6. 启智社区运行步骤

### 6.1 上传数据集

在启智社区左侧进入：

```text
数据集 -> 新建数据集
```

上传本地文件：

```text
C:\Users\asus\Desktop\FedPRIME-D2C\local_runs\fedsara_cs_prepared\fedsara_cs_prepared_alpha05_rho07_seed0.tar.gz
```

建议数据集名称：

```text
fedsara-cs-alpha05-rho07-seed0
```

### 6.2 创建计算任务

推荐先用：

```text
调试任务
```

资源：

```text
1 x V100 32GB
```

镜像建议：

```text
PyTorch 2.1.x / Python 3.10 / CUDA 11.8
```

参数区挂载：

```text
数据集：选择 fedsara-cs-alpha05-rho07-seed0
项目：选择/关联当前 GitHub 仓库
```

### 6.3 进入 Jupyter 后的命令

先 clone：

```bash
git clone https://github.com/yibinlin-fl/fedprime-d2c.git
cd fedprime-d2c
```

先做 debug，不训练正式 40 轮：

```bash
RUN_INSTALL=1 RUN_IMPORT_DATA=1 RUN_DEBUG=1 RUN_TRAIN=0 RUN_SUMMARY=0 \
DATA_SOURCE=/dataset \
bash scripts/run_openi_fedsara_cs.sh
```

如果 `/dataset` 找不到，脚本会继续在常见挂载目录里搜索；如果平台实际路径不同，可以把 `DATA_SOURCE` 改成右侧数据集挂载路径。

正式跑 RAHFL-CS 和 FedSARA-CS 对比：

```bash
RUN_INSTALL=1 RUN_IMPORT_DATA=1 RUN_DEBUG=0 RUN_TRAIN=1 RUN_SUMMARY=1 UPLOAD_C2NET=1 \
DATA_SOURCE=/dataset \
bash scripts/run_openi_fedsara_cs.sh
```

如果只想跑 FedSARA-CS：

```bash
RUN_INSTALL=1 RUN_IMPORT_DATA=1 RUN_DEBUG=0 RUN_TRAIN=1 RUN_SUMMARY=1 UPLOAD_C2NET=1 \
DATA_SOURCE=/dataset \
bash scripts/run_openi_fedsara_cs.sh configs/openi_v100_fedsara_cs_alpha05_rho07.yaml
```

如果只想跑 RAHFL-CS：

```bash
RUN_INSTALL=1 RUN_IMPORT_DATA=1 RUN_DEBUG=0 RUN_TRAIN=1 RUN_SUMMARY=1 UPLOAD_C2NET=1 \
DATA_SOURCE=/dataset \
bash scripts/run_openi_fedsara_cs.sh configs/openi_v100_rahfl_cs_alpha05_rho07.yaml
```

## 7. 结果在哪里

训练结束后看：

```text
outputs/<experiment_name>/metrics.csv
outputs/<experiment_name>/corruption_group_acc.csv
outputs/<experiment_name>/client_group_acc.csv
outputs/summary.csv
fedsara_cs_openi_outputs.tar.gz
```

启智计算任务会自动销毁环境，因此推荐开启：

```text
UPLOAD_C2NET=1
```

脚本会把 `outputs/` 和 `fedsara_cs_openi_outputs.tar.gz` 复制到平台输出目录并调用 `upload_output()`。

