# 2026-11-07 Axis II FedDF 单臂 Formal 验收

FedDF-fidelity 通过结果验收，可填主评价八行表第 4 行。当前已完成 6/8 行，AugHFL 和 RAHFL 的 Formal 待补。

## 实验身份与审计

- OpenI 入口为 `scripts/openi_cle_v2_moderate_entry.py`，`mode=formal`、`shard=feddf_fidelity`。启动前核验的远端提交为 `32f4050c61861ae072ab00f725f5b40ecf88bbee`。
- 复用 CIFAR-10 private、CIFAR-100 public 的主评价包（893421220 bytes，SHA256 `DEA800EC1465C45037E74E4B948DFB8D778F1354E8522B551438C6A515C9E588`）。回传合同和配置确认 map2、partition seed 0、training seed 0、α=1.0、ρ=0.5、γ=0.9。
- 0 轮预训练、40 轮、每轮每客户端最多 64 个本地 batch，batch size 64。每轮 4 个公共 batch，public batch size 128。FedDF 配置使用单视图和 public distillation，公共阶段每轮实际记录 4 次服务器更新。
- 正式包、恢复包和日志齐全。40 轮记录连续，两项训练 loss 均为有限值，配置哈希符合合同，恢复包含 4 个 checkpoint。两个包的 10 项共享配置与训练文件逐字节一致。恢复包状态为 `training_complete_analysis_pending`，正式包和外部计时文件均为 `complete`。
- FedDF 与已有五行的首 batch 图像和标签哈希均匹配 160/160 条。该检查覆盖每轮每客户端的首个本地 batch。六行缓存的标签、绑定矩阵和 operator 顺序逐项一致，预算、模型架构和初始权重配置匹配。
- 从 4 客户端 × 1000 源图 × 15 operator 的预测缓存重新计算六行 DSA 与 grid accuracy，与正式摘要一致。FedDF 的 1000 次 shuffled-binding null 复算一致（95% 分位 0.026066，p=0.000999）。FedDF 的 5000 次 source bootstrap 95% 区间为 [0.146253, 0.153076]，反映固定模型下的测试源图变异。

## 论文主表数值

| 方法 | DSA ↓ | Grid (%) | Clean (%) | Corrupt (%) | Mixed (%) |
|---|---:|---:|---:|---:|---:|
| Local/ERM | 0.168374 | 39.830 | 51.425 | 39.508 | 45.467 |
| FedMD | 0.167835 | 38.732 | 50.400 | 38.378 | 44.389 |
| FedProto | 0.160788 | 39.250 | 50.800 | 38.937 | 44.868 |
| FedDF-fidelity | 0.149599 | 40.557 | 53.250 | 40.057 | 46.653 |
| AsymHFL–ERM | 0.184514 | 37.193 | 49.700 | 37.008 | 43.354 |
| AsymHFL+PEW+BER | 0.090024 | 38.160 | 45.350 | 37.773 | 41.562 |

FedDF 的 seen / unseen accuracy 为 39.966% / 40.306%，四客户端 DSA 依次为 0.067916、0.130718、0.209312 和 0.190450。相对 Local，FedDF 的 DSA 低 0.018775，Mixed 高 1.187 个百分点。与 AsymHFL+PEW+BER 比，FedDF 的 Mixed 高 5.092 个百分点，DSA 高 0.059575。当前六行中 FedDF 的各项准确率最高，AsymHFL+PEW+BER 的 DSA 最低。两者的通信和本地目标均不同，这一对比用于描述主评价表中的效用与捷径表现。PEW+BER 的干预效应由匹配的 AsymHFL 两臂检验。

## 耗时与回传

训练 19256.144 秒（5 小时 20 分 56 秒），分析 119.110 秒（1 分 59 秒），两段合计 19375.254 秒（5 小时 22 分 55 秒）。按该账号此前 V100 的 6 积分/小时费率折算约 32.29 积分。实际扣分以 OpenI 账单为准，脚本计时不覆盖全部平台开销。FedDF 实测明显低于之前预留的 100 积分。AugHFL / RAHFL 的预算仍依据各自鲁棒训练 benchmark。

原始文件保存在 `outputs/openi_downloads/main_axis2_feddf_formal_s0_20261104/`，包括 `cle-feddf-format363424855-log.txt` 和 `RUN_TIMING.json`。

| 包 | Bytes | SHA256 |
|---|---:|---|
| `cle_v2_moderate_feddf_fidelity_formal_outputs.tar.gz` | 2395216 | `591CDEE31A91CEF5901A3B2C52629AA25B3C81C954F62B3763B622261881EA52` |
| `cle_v2_moderate_feddf_fidelity_formal_recovery_outputs.tar.gz` | 90906714 | `4DC6390556F1675684ACBB42929B9E51ECADF295108810B778B680EC62CB9034` |
