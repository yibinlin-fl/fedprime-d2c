# CLE-HFL CDep-v2 共享PEW两臂 OpenI 实验

更新日期：2026-08-08

## 唯一目的

单臂 CDep-v2 与历史 PEW+BER 使用了不同 PEW 注释，不能完成因果归因。本入口
在同一个 OpenI 任务中只训练一次 PEW，并依次运行：

```text
control:   同一PEW + BER，CDep关闭
candidate: 同一PEW + BER + CDep-v2
```

候选运行加载 control 保存的同一个 PEW checkpoint。打包前脚本强制验证四个
客户端的 PEW 注释文件逐字节一致；任何哈希不一致都会使任务报错，不输出有效
决策。

## OpenI 填写

```text
代码分支：main
数据集：openi_cle_hfl_v2_alpha05_gamma09
启动文件：scripts/openi_cle_cdep_v2_paired_entry.py
运行参数：留空
```

不要填写 `arms`，也不要添加 `--skip_train` 或 `--no_upload`。

## 运行规模

```text
control 12轮
candidate 12轮
总计24轮，在一个任务中依次执行
scenario seed=0
training seed=0
fit/audit split=seed0_split0
communication=AsymHFL-val
```

只有 CDep 开关不同。无需运行 CDep-v1。

## 正常日志

先看到 control：

```text
[setup] training PEW from unlabeled CIFAR-100 corruptions
...
cle_cdep_v2_paired_control_seed0_12round
```

之后 candidate 必须看到：

```text
[setup] loading PEW checkpoint: outputs/pew_checkpoints/cle_cdep_v2_paired_seed0.pt
...
cle_cdep_v2_paired_candidate_seed0_12round
```

candidate 的第0/1轮 ramp 为0，第2轮为0.33，第4轮起为1.0。

## 输出与冻结门槛

自动上传：

```text
cle_cdep_v2_paired_12round_outputs.tar.gz
```

核心自动文件：

```text
outputs/cle_cdep_v2_paired_comparison.json
```

其中必须包含：

```text
pew_annotations.byte_identical = true
```

候选减 control 的最后5轮四门槛保持不变：

```text
ΔAvg   >= 0
ΔWorst >= 0
ΔWCCA  >= 0
ΔCFG   <= -0.5
```

全部通过才保留 CDep-v2；任一失败则最终本地方法冻结为 calibrated PEW+BER。
