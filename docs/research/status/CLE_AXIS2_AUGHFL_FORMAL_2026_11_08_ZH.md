# 2026-11-08 Axis II AugHFL 单臂 Formal 验收

AugHFL-fidelity 可填主评价八行表第5行。表现在已验收7/8行，RAHFL待恢复分析。AugHFL为training seed 0的一次40轮正式运行，20/30/40轮包属于同一次运行。

## 身份与验收

- OpenI回传合同确认map2、partition seed 0、training seed 0、α=1.0、ρ=0.5、γ=0.9、40轮、每客户端每轮最多64个本地batch、batch size 64、每轮4个公共batch。与已验收六行的预训练、本地轮次、公共batch size、架构和初始权重加载配置匹配。
- `RUN_STARTED.json` 的输入SHA256为 `dea800ec1465c45037e74e4b948dfb8d778f1354e8522b551438c6a515c9e588`。源码指纹 `34a587e4e59f7fc4833538691dceb3abd5a35969d80ce78f4e2a53a46e20c978` 与提交 `0323f1b4bce13c8f0ff94d142ae2b63cc86098f5` 的Linux LF源码一致。Windows Git导出的换行格式差异已核对。
- 40轮连续、训练loss均为有限值、配置SHA256与合同一致、完成清单及正式包内外计时均确认完成。OpenI日志记录保存四个最终权重及第20/30/40轮上传成功。本次验收使用轻量包，未逐个加载权重文件。
- AugHFL使用自身的公共多视图协作和本地AugMix/JSD，关闭DCL。与已有六行的160条本地首batch标签哈希一致，预测缓存中的源图标签、绑定矩阵及算子顺序一致。不同方法的增强视图按各自训练配置产生。
- 从4客户端 × 1000源图 × 15算子预测缓存独立复算七行pooled/client DSA及Grid accuracy，均与正式摘要一致。AugHFL的1000次shuffled-binding null复算为95%分位0.018024、p=0.000999。

## 结果与解释

| 方法 | DSA ↓ | Grid (%) | Clean (%) | Corrupt (%) | Mixed (%) |
|---|---:|---:|---:|---:|---:|
| Local/ERM | 0.168374 | 39.830 | 51.425 | 39.508 | 45.467 |
| FedMD | 0.167835 | 38.732 | 50.400 | 38.378 | 44.389 |
| FedProto | 0.160788 | 39.250 | 50.800 | 38.937 | 44.868 |
| FedDF-fidelity | 0.149599 | 40.557 | 53.250 | 40.057 | 46.653 |
| AugHFL-fidelity | 0.084157 | 44.765 | 50.500 | 43.808 | 47.154 |
| AsymHFL–ERM | 0.184514 | 37.193 | 49.700 | 37.008 | 43.354 |
| AsymHFL+PEW+BER | 0.090024 | 38.160 | 45.350 | 37.773 | 41.562 |

AugHFL的seen/unseen operator accuracy为43.623%/44.319%。四客户端DSA依次为0.047598、0.099470、0.094594、0.094966。

截至七行，AugHFL的pooled DSA最低，Grid、Corrupt与Mixed准确率最高，FedDF的Clean准确率最高。相对AsymHFL+PEW+BER，AugHFL的DSA低0.005867，Grid/Corrupt/Mixed分别高6.605/6.035/5.593个百分点。5000次同源配对bootstrap（seed 20261025）的AugHFL−AsymHFL+PEW+BER DSA区间为[-0.007588, -0.004246]。该区间只涵盖固定训练种子下的测试源图变异。两行同时改变本地目标与通信，不能单独归因给AugMix/JSD或PEW+BER。PEW+BER的效果继续由匹配的AsymHFL两臂检验。

## 耗时与归档

训练阶段83576.735秒（23h12m57s，包含中途回传），分析4757.220秒（1h19m17s），合计88333.955秒（24h32m14s）。按此前6积分/小时估算147.22积分，平台账单另核。正式用时高于16–17小时的benchmark外推。40轮实测平均34m48s/轮。

原始三件套保存在 `outputs/openi_downloads/main_axis2_aughfl_formal_s0_20261104/`。正式包 `cle_v2_moderate_aughfl_fidelity_formal_outputs.tar.gz` 为2385020 bytes，SHA256 `90CC037260CDBA5EF11AE5A2F0D9BC05486AFD2428A4A08FA799DC8C8624DB9A`。另有 `RUN_TIMING.json` 与 `cle-aughfl-recot153075117-log.txt`。当前V0.11留作阅读基线，待RAHFL补分析并验收后集中回填三行。
