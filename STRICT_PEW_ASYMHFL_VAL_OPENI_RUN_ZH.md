# Strict PEW + AsymHFL-val 运行说明

## 目的

本实验只回答一个问题：在完全相同且无测试泄漏的通信条件下，已经验证有效的
`calibrated PEW + BER+CDep` 本地机制能否稳定优于 `AugMix + JSD + DCL`。

两组实验都使用 CLE-HFL v2、同一组异构模型、同一个随机种子、同一个持久化
`fit/audit` 划分、同样的 CIFAR-100 public batches 和 audit-only AsymHFL 路由。

```text
control   = AugMix + JSD + DCL + strict AsymHFL-val
candidate = AugMix + JSD + DCL + calibrated PEW + BER+CDep + strict AsymHFL-val
```

`fit` 用于本地梯度训练，`audit` 只用于每轮 AsymHFL 教师排序，最终测试集只记分。

## OpenI 设置

数据集继续使用：

```text
cle_hfl_v2_prepared_alpha05_gamma09_seed0_split0.tar.gz
```

启动文件：

```text
scripts/openi_strict_pew_asymhfl_entry.py
```

一次性运行 A/B 时增加运行参数：

```text
mode=both
```

如果平台参数框要求完整命令行形式，则填写：

```text
--mode=both
```

## 冻结判据

看最后五轮 candidate 减 control：

```text
Avg   >= +1.5
Worst >= +1.0
WCCA  >=  0.0
CFG   <= -1.0
```

四项全部通过才进入 40 轮；否则停止该组合，不靠盲调损失权重挽救。
