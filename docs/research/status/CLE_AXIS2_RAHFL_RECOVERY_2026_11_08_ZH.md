# 2026-11-08 Axis II RAHFL 正式结果恢复与验收

RAHFL-fidelity 的 OpenI 任务完成全部40轮和最终权重回传，随后在分析阶段停止。用原四个 checkpoint 在本机完成同协议评估，第6行通过验收。主评价表已取得8/8行 seed-0 结果，包含作者希望进一步复核的 AugHFL 行。

## 恢复来源与配对核验

- 原始资料保存在 `outputs/openi_downloads/main_axis2_rahfl_formal_s0_20261104/`。训练恢复包 `cle_v2_moderate_rahfl_fidelity_formal_recovery_outputs.tar.gz` 为277205826 bytes，SHA256 `820BFDB53343781528CD67A81B98B1924E9A541BB580EAEC2701731FF0DC7C17`。云端日志为 `cle-rahfl-recovt160583832-log.txt`。
- 包内外 `RUN_TIMING.json` 一致，原始状态为 `training_complete_analysis_pending`。完成清单确认训练完成，日志显示第20、30、40轮上传，最终保存四个客户端权重，并开始分析推理。云端日志末尾为 `stopped`，没有完成摘要或正式结果包。
- 恢复包的配置SHA256及40轮状态文件SHA256与各自记录一致，40轮连续且训练loss均为有限值。训练是 map2、partition seed 0、training seed 0、α=1.0、ρ=0.5、γ=0.9、40轮×每客户端每轮最多64个本地batch。RAHFL使用本地AugMix/JSD/DCL及audit路由的AsymHFL通信。
- 启动记录的源码指纹与已验收AugHFL一致，对应提交 `0323f1b4bce13c8f0ff94d142ae2b63cc86098f5`。本机分析器及相关模型源码与该提交一致。输入包SHA256为 `dea800ec1465c45037e74e4b948dfb8d778f1354e8522b551438c6a515c9e588`。15份本机评估相关文件与该包逐字节一致。
- 用 `scripts/analyze_cle_v2_moderate.py --mode formal --shard rahfl_fidelity --device cuda --confirm-formal` 补评估。RTX 3050 Laptop GPU，PyTorch 2.8.0+cu126，严格加载四个最终权重，只执行推理与统计。分析退出码0，耗时85.680秒。云端训练阶段计时83638.448秒（23h13m58s）。本机平衡损坏准确率与OpenI第40轮日志相差0.0067个百分点。
- 八行的架构、初始权重加载配置、配对预算、fit/audit设置和160条本地首batch标签哈希匹配。预测缓存的1000张源图标签、15种算子顺序及绑定矩阵一致。八行DSA和Grid从预测缓存复算一致。

## 同协议八行结果

| 方法 | DSA ↓ | Grid (%) | Clean (%) | Corrupt (%) | Mixed (%) |
|---|---:|---:|---:|---:|---:|
| Local/ERM | 0.168374 | 39.830 | 51.425 | 39.508 | 45.467 |
| FedMD | 0.167835 | 38.732 | 50.400 | 38.378 | 44.389 |
| FedProto | 0.160788 | 39.250 | 50.800 | 38.937 | 44.868 |
| FedDF-fidelity | 0.149599 | 40.557 | 53.250 | 40.057 | 46.653 |
| AugHFL-fidelity | 0.084157 | 44.765 | 50.500 | 43.808 | 47.154 |
| RAHFL-fidelity | 0.074984 | 45.882 | 51.225 | 45.277 | 48.251 |
| AsymHFL–ERM | 0.184514 | 37.193 | 49.700 | 37.008 | 43.354 |
| AsymHFL+PEW+BER | 0.090024 | 38.160 | 45.350 | 37.773 | 41.562 |

RAHFL的seen/unseen operator accuracy为45.470%/44.744%。四客户端DSA依次为0.041520、0.095121、0.080126、0.083170。Shuffled-binding null的95%分位为0.015528，1000次置换p=0.000999。

在固定seed 0的八行中，RAHFL的DSA最低，Grid、Corrupt、Mixed最高。FedDF的Clean最高。相对AugHFL，RAHFL的DSA低0.009173，Mixed高1.097个百分点。相对AsymHFL+PEW+BER，RAHFL的DSA低0.015040，Mixed高6.689个百分点。5000次同源配对bootstrap（seed 20261025）的RAHFL−AugHFL DSA区间为[-0.010340, -0.008000]，RAHFL−AsymHFL+PEW+BER为[-0.016888, -0.013266]。区间反映固定训练模型下的源图变异。

两条鲁棒基线在当前主评价协议下取得了比AsymHFL+PEW+BER更低的DSA与更高的Mixed。各行采用各自的本地目标及通信配置，不能把跨方法差异归因给某个单独模块。PEW+BER在匹配AsymHFL两臂中的结论仍为DSA下降51.2%、Mixed下降1.793个百分点。轴二整表为固定map2、seed 0的领域对比。原V0.11留作作者阅读基线，待确定表格叙事后集中回填。

## 本机派生文件

位于原RAHFL结果目录下的 `local_recovery_20261108/`。保留 `LOCAL_ANALYSIS.log`、`LOCAL_ANALYSIS_TIMING.txt`、四个解出的最终权重及原始元数据。

| 文件 | SHA256 |
|---|---|
| `analysis/RESULT_SUMMARY.json` | `96789c7c4eaf444f36abe3d257bb2799056f0274ddf52977ca8308a081dbedc3` |
| `analysis/MODERATE_PREDICTIONS.npz` | `e907c8d33847894fc38734b4ec2275d7a5de59a9a66b203b61f46b3b790a582c` |
