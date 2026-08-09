# CLE-HFL 基线忠实度修复 OpenI 运行说明

Updated: 2026-08-09

## 目的

重新评估修复后的 AugHFL、FedDF 和 KT-pFL 核心实现。旧实现及旧结果不覆盖；RAHFL和
PEW+BER作为同任务锚点重新运行，用于检查任务环境和合并有效性。

这仍是 12 轮统一预算筛选，不是各论文完整原始训练日程。

## OpenI填写

```text
数据集: openi_cle_hfl_v2_alpha05_gamma09
代码分支: main（必须先由用户明确要求提交并推送本次修改）
启动文件: scripts/openi_cle_baseline_fidelity_entry.py
```

建议一次运行全部五臂：

```text
参数名: arms
参数值: all
```

如果平台只能分批：

```text
第一批: aughfl_fidelity,feddf_fidelity,kt_pfl_fidelity
第二批: rahfl,pew_ber
```

锚点最好与新实现同任务运行。不要用本地 smoke 结果代替正式锚点。

## 预期产物

```text
cle_baseline_fidelity_seed0_12round_outputs.tar.gz
```

压缩包包含：

- 每个已完成arm的resolved output；
- 每个generated config；
- `fidelity_manifest.json`；
- 在包含RAHFL时生成的汇总JSON。

## 结果解释

必须同时检查：

- Avg、Worst、WCCA、CFG；
- 三个新实现的 `col_loss` 非零且有限；
- AugHFL teacher weight/view consistency诊断；
- FedDF teacher entropy/disagreement/server updates；
- KT-pFL coefficient entropy/diagonal/off-diagonal/drift；
- RAHFL、PEW+BER锚点是否复现历史值。

只有12轮明显优于旧适配或接近RAHFL，才考虑40轮。不得因为某臂烟雾测试准确率低而淘汰。
