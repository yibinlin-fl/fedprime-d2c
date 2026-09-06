# CLE LCRE M0 低成本方法门槛

Updated: 2026-09-06

## 1. 状态与目标

LCRE（Label-Conditioned Response Equalization，标签条件响应均衡）是工作名，不是最终论文名。
本阶段只做两个历史 H9 round-40 checkpoint 上的 matched short adaptation kill test，判断是否值得
设计完整四客户端 HFL 集成。它不是 round-41 continuation，因为历史产物没有 Adam state。

当前协议在 Formal 前冻结。Smoke 与 benchmark 只验证执行和成本，不是科学证据。不得声称
first、novel、SOTA 或已经消除 CLE。

截至 2026-09-06，13/13 聚焦测试和两架构六臂真实 checkpoint smoke 已通过；private/AugMix
trace、JSD/LCRE probe trace、BN buffers、taxonomy/public/oracle 隔离及 checkpoint 重载全部
合规。Smoke LCRE active classes 为 ResNet10 6、MobileNetV2 7，skip rate 均为0。修正后的本地
8-step benchmark 也通过完整性检查，LCRE active-class 分布为 ResNet10 `{5:1,6:1}`、
MobileNetV2 `{7:2}`，skip rate 0。RTX 3050 对完整六臂三 epoch 训练的 wall-clock 外推为
`8150.01 s / 2.2639 GPU-hours`，不含最终 oracle。这不能代替冻结的 V100 成本门；OpenI V100
benchmark 和 Formal 均未启动。

## 2. CLE 背景与当前证据边界

CLE（Corruption–Label Entanglement）使 corruption 与 task label 形成虚假相关。Phase-A0 的配对
干预和 DSA 已证明 strong CLE directional shortcut 存在；HFL-vs-Local 归因表明它主要
local-first，不支持通信放大或坏教师主故事。PEW/BER 是有效但依赖人工 corruption taxonomy 的
正基线。K0-B 只保留为离线 taxonomy-free 检测器。

## 3. CVRS NO-GO 与新假设

CVRS 压制公共载体上跨样本持久的全局类别方向 `||E_x delta||^2`。MobileNetV2 上 CVRS 的 generic
proxy 比 Public-JSD 更低，但真实 DSA 更高，故该 proxy 不是架构稳定的治疗靶点。

LCRE 不再先定位一个 harmful routing map，而直接检验以下方法假设：对同一个 taxonomy-free
intervention，不同 task labels 不应产生系统不同的平均 class-visible response。

## 4. 完整数学定义

模型 logits 为 `z_theta(x) in R^C`，类别中心化算子为：

```text
P_C = I - (1/C) 11^T
delta_iq = P_C [z_theta(A_q(x_i)) - z_theta(x_i)]
```

对 minibatch 中计数 `n_c >= 2` 的 active classes `C_B`：

```text
mu_qc     = (1/n_c) sum_{i:y_i=c} delta_iq
bar_mu_q  = (1/|C_B|) sum_c mu_qc
B_q       = (1/|C_B|) sum_c ||mu_qc - bar_mu_q||_2^2
E_q_bal   = (1/|C_B|) sum_c [(1/n_c) sum_{i:y_i=c} ||delta_iq||_2^2]
L_LCRE    = (1/|Q_t|) sum_q B_q / (stopgrad(E_q_bal) + 1e-8)
```

若 `|C_B| < 2`，该次 LCRE 为可微分的零并记录 skip。singleton 不进入 class statistic。只对
denominator stop-gradient；不得 detach response、class mean 或 numerator。

## 5. Cross-covariance 解释

在 active classes 的 class-balanced empirical measure 下，令标签 one-hot 为 `e_Y`，则：

```text
B_q = |C_B| * ||Cov_bal(e_Y, delta_q)||_F^2
```

因此 LCRE 抑制 label 与 intervention response 的一阶统计依赖。`L_LCRE=0` 只表示 active classes
的 conditional first moments 相等；不表示 `delta` 与 `Y` 独立，不表示互信息为零。

## 6. 与 JSD、CRSF 和 CVRS 的区别

- Private-PRIME-JSD 惩罚逐样本预测分布变化，是 total perturbation consistency。
- LCRE 惩罚 `Var_Y(E[delta|Y])`；非零但类别间均值相同的 response 不被惩罚。
- CRSF 修改 representation response spectrum；其谱变化没有有效改变 class routing。
- CVRS 压制 `||E delta||^2`，会遗漏全局均值抵消但各 label 条件响应不同的情况。

Private-PRIME-JSD 是必须的 compute-matched control，用于判断 LCRE 是否超出“增加 PRIME
augmentation/consistency”的收益。

## 7. 数据限制与 taxonomy-free 边界

优先复用唯一有效输入包 `cle_cvrs_m0_seed0_inputs.tar.gz`：

```text
bytes:  109142359
sha256: E9427A55DBE2545AF9D5A1EBD8BEA5B18C41C84D7FE89D06674165F4109E3818
```

禁止旧 164 MiB/172255488-byte 包。训练 loader 只可打开 private image、task label 和固定 fit
split；不得打开 corruption id/type/family、severity、CLE binding、source/environment metadata 或
DSA output。CIFAR-100 public carriers 即使存在于继承包中也不得由训练 runner 打开；OpenI 入口
对所有模式均不提取 public 目录。

LCRE 不识别 corruption type，不恢复真实 harmful routing map，也不保证 generic PRIME response
等于真实 corruption response。它只是 taxonomy-free method hypothesis 加 cheap causal-performance
test。

## 8. 三臂与模型

每个 architecture 从同一 checkpoint 克隆三份：

```text
client0 / ResNet10:    Baseline, Private-PRIME-JSD, LCRE
client3 / MobileNetV2: Baseline, Private-PRIME-JSD, LCRE
```

Baseline 保持原 AugMix/JSD/DCL。JSD 与 LCRE 每第 4 个 private step 使用同一 batch、同一 Bank-A
probe IDs、同一 recipe 及同一 auxiliary forward 数量。所有臂都只有一次 backward 和一次
optimizer.step；不得先 task step 再 regularizer step。

## 9. PRIME 与 BN 协议

Bank A hash 固定为：

```text
6CAE529D4240715162B19B3968D47FA037A940B4D52D688FF52B859C5523DC01
```

每个 regularized step 取 4 个 probes；seed `20260905`，shuffle without replacement，16 次覆盖
全部 64 recipes 后 reshuffle。同一个 recipe 应用于整批样本。不得按 DSA、taxonomy 或 K0-B
risk 选 probe。

正常 task forward 保留 baseline BN 行为。PRIME auxiliary forward 暂停所有 BN running-stat
tracking；running_mean、running_var、num_batches_tracked 必须完全不变，BN affine 和其他参数仍
参与梯度。随后恢复原状态，并对 total loss 做一次 backward。

## 10. 优化器与 lambda 校准

全部三臂使用 fresh Adam、LR `0.001`、weight decay `0`、batch size `64`、epochs `3`。首次
regularized step 在更新前，分别对所有 trainable parameters 计算：

```text
g_task = ||grad L_base||_2
g_reg  = ||grad L_reg||_2
lambda = 0.1 * g_task / (g_reg + 1e-8)
```

每个 architecture/regularized arm 独立校准一次并永久冻结。记录 calibration batch hash、probe
IDs、两个梯度范数、lambda、参数数和范数算法。`g_reg` 非 finite 或 `<1e-12` 时判
`CALIBRATION_INVALID`，不得人工指定 lambda。

## 11. RNG 与完整性匹配

同一 architecture 三臂必须具有相同 starting checkpoint、fit split、private batch order、baseline
augmentation/AugMix/DCL trace 和 optimizer step 数。JSD 与 LCRE 还必须具有相同 probe schedule、
transformation output 和 auxiliary batch identities。runner 输出 private/AugMix/probe hashes 并在
不一致时直接终止。

## 12. Active-class 可行性

LCRE 每个 regularized step 写出 batch class histogram、active-class 数、singleton 数、skip、每个
probe 的 `B_q`、`E_q_bal` 与 normalized loss。Smoke/benchmark 汇总 active-class distribution 与
skip rate。若大量 batch 只有不足两个 active classes，停止并报告，不得修改规则。

## 13. Smoke

真实 ResNet10/MobileNetV2 checkpoint 的 tiny smoke 各跑三个臂和四个 private steps，确保命中
一次 regularized step。检查有限 forward/backward/lambda、BN buffers、trace matching、无 taxonomy
或 public-carrier 读取，以及输出 checkpoint 严格重载。唯一 verdict：

```text
SMOKE_ONLY_NO_SCIENTIFIC_DECISION
```

## 14. Benchmark

Smoke PASS 后跑每臂 8 private steps，记录 task forward、PRIME generation、auxiliary forward、
backward/optimizer、peak VRAM、process RSS，并外推六臂三 epoch GPU-hours。不得读取 DSA evaluator、
CLE binding 或 taxonomy。唯一 verdict：

```text
BENCHMARK_ONLY_NO_SCIENTIFIC_DECISION
```

若预计 V100 总训练成本超过 1 GPU-hour，停止并报告；不得自行缩短协议。任何 Formal 都需要用户
随后明确批准 `--mode=formal --confirm-formal`。

## 15. Formal、pre-oracle seal 与冻结门槛

训练完成后先保存 checkpoint/hash、lambda、logs、trace、BN audit、config 和 code commit，生成并
hash `pre_oracle_manifest.json`。只有 seal 完成后才可打开冻结 Phase-A0 evaluator：1000 balanced
clean sources × 16 frozen corruptions。不得改变 operator、severity、source、binding 或 aggregation。

每个 architecture 分别要求四项全过：

```text
(DSA_B - DSA_L) / DSA_B >= 20%
DSA_J - DSA_L             >= 0.02
Avg_L - Avg_B             >= -1.0 pp
Worst_L - Worst_B         >= -1.0 pp
```

两个 architecture 都通过才输出 `GO_TO_FULL_HFL_INTEGRATION`；任一失败即 `NO_GO_LCRE`。GO 也不
自动授权完整 HFL，只允许先做 literature collision audit、集成设计、效率审计和多 seed 计划。
artifact/hash、taxonomy/RNG、NaN、evaluator、checkpoint 或 calibration 问题输出
`INVALID_EXPERIMENT`，不算科学 NO-GO。

## 16. 禁止救援与声明边界

禁止改通信、使用 taxonomy/family/severity/binding、online detector、K0-B targeting、CRSF、
CVRS-v2、BNR、MixStyle/Fourier swapping、PEW/BER、lambda grid、补 epoch/seed、换 probes、改
active-class rule、平均架构结果掩盖失败或事后修改 gate。

即使 M0 GO，也只允许表述：LCRE penalizes the between-class component of taxonomy-free
perturbation responses，并将 CLE operationalize 为 label-conditioned interventional response
dependence。不得声称识别真实 nuisance、恢复因果 corruption mechanism、保证 nuisance
independence 或具有已确认的新颖性。
