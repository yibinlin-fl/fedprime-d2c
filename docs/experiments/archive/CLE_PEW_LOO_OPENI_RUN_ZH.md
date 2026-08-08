# CLE-HFL Strict PEW-LOO OpenI 实验

更新日期：2026-08-08

## 实验目的

检验当前 PEW+BER 的收益是否依赖 PEW 在公共合成训练阶段见过
private-unseen 的具体 corruption operator。

原始 PEW 的每个 family 使用四个 operator：

```text
noise:   gaussian_noise, shot_noise, impulse_noise, speckle_noise
blur:    defocus_blur, glass_blur, motion_blur, zoom_blur
weather: snow, frost, fog, spatter
digital: contrast, brightness, jpeg_compression, pixelate
```

Strict PEW-LOO 从 PEW 的公共训练和公共验证生成器中同时排除：

```text
impulse_noise
zoom_blur
fog
pixelate
```

排除后每个 family 仍保留三个 operator。旧 PEW 默认配置和历史 checkpoint
行为保持不变；Strict LOO 使用独立 checkpoint，禁止与原版混用。

## 三臂设计

一个 OpenI 任务顺序运行三臂，每臂 12 轮：

```text
rahfl:                AugMix/JSD/DCL + strict AsymHFL-val
standard_pew_ber:     原始 calibrated PEW + BER + 同一 AsymHFL-val
strict_loo_pew_ber:   Strict-LOO calibrated PEW + BER + 同一 AsymHFL-val
```

CDep-v1/v2 均关闭。三臂固定 scenario seed 0、training seed 0、模型集合、
fit/audit 划分、通信和评估协议。FedEASE 在 PEW 准备后重置 RNG，以保持模型
初始化和联邦训练随机性匹配。

## OpenI 填写

```text
代码分支：main
数据集：openi_cle_hfl_v2_alpha05_gamma09
启动文件：scripts/openi_cle_pew_loo_entry.py
运行参数：留空
```

不要添加 `arms`，不要添加 `--skip_train` 或 `--no_upload`。

## 自动防泄漏审计

入口在训练前检查四个留出 operator 在所有客户端 private fit 中计数均为 0。
训练后继续检查：

```text
standard PEW exclusion = []
strict PEW exclusion = [impulse_noise, zoom_blur, fog, pixelate]
strict public operator pools 不包含上述四项
checkpoint 记录的 exclusion protocol 必须与配置一致
```

任何不一致都会报错，不产生有效科学决策。

## 预注册决策

主判据是 Strict-LOO PEW+BER 相对本任务 RAHFL 的 last-five 增量：

```text
Delta Avg   >= +1.5
Delta Worst >= +1.0
Delta WCCA  >=  0.0
Delta CFG   <= -1.0
```

四项全部通过才认为 PEW+BER 在 PEW 未见具体 operator 时仍保留有效收益。
同时报告 Strict LOO 相对原始 PEW+BER 的退化或提升，但不在看到结果后修改
上述主门槛。

这里验证的是“PEW 公共监督的 operator LOO”，不是声称整个训练管线从未使用
任何相似的通用增强；AugMix 保持不变并在三臂中匹配。

## 返回文件

```text
cle_pew_loo_12round_seed0_outputs.tar.gz
```

核心自动结果：

```text
outputs/cle_pew_loo_protocol_audit.json
outputs/cle_pew_loo_report.json
outputs/cle_pew_loo_decision.json
```
