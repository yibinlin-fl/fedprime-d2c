# CLE-HFL v2 OpenI 启动指南

## 1. 上传数据集

上传本地文件：

```text
local_runs/cle_hfl_v2_prepared/
cle_hfl_v2_prepared_alpha05_gamma09_seed0_split0.tar.gz
```

压缩包约 346 MiB，低于 OpenI 网页端单文件 512 MiB 限制。

OpenI 数据集名称可以自定义，例如：

```text
openi_cle_hfl_v2_alpha05_gamma09
```

代码按压缩包文件名递归搜索，不依赖 OpenI 数据集名称。

## 2. 任务设置

```text
镜像：ubuntu22.04-cuda11.8.0-py310-torch2.1.0-tf2.14.0
资源：1 x V100 32GB
项目：fedprime-d2c
分支：main
启动文件：scripts/openi_cle_v2_entry.py
```

## 3. 运行参数

最推荐先跑严格 A/B：

```text
--method=both
```

它依次运行：

```text
strict fit-only control
FedFalsify v0.3
```

单独运行：

```text
--method=control
--method=fedfalsify
--method=rahfl
```

三者一次跑完：

```text
--method=all
```

不建议第一次使用 `--method=all`，因为 12 轮 RAHFL 还需要 public communication，
总时间和积分都会增加。

## 4. 自动流程

入口会自动执行：

```text
1. c2net prepare()
2. 搜索挂载的数据压缩包
3. 解压并导入 cifar_10_cle_v2 与 cifar_100
4. 安装 requirements.txt
5. 检查配置和 CUDA
6. 运行所选实验
7. 汇总 outputs
8. 打包 cle_hfl_v2_probe_outputs.tar.gz
9. 回传到 OpenI output_path
```

日志使用无缓冲输出，并在候选路由、本地客户端训练和每轮评价阶段打印 heartbeat。

## 5. 结果文件

下载：

```text
cle_hfl_v2_probe_outputs.tar.gz
```

重点文件：

```text
metrics.csv
operator_split_metrics.csv
client_operator_accuracy.csv
client_class_operator_accuracy.csv
route_candidates.csv
```

其中：

```text
operator_split_metrics.csv
```

直接给出 all/seen/unseen 的 Avg、Worst、WCCA 和 CFG。
