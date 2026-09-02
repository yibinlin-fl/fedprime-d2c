# CLE K1-B0：CDR-SNR 共享表征定位门

更新时间：2026-09-02

## 1. 当前阶段回答什么

K0-B 已经证明：不使用真实 corruption taxonomy，仅用冻结的 PRIME generic probes，能够在
H9/L9 中检测到比 H0/L0 更强的 `carrier-stable + class-selective` 方向。K1-A 随后的
head-only SDMN 手术正式失败，说明只改最后分类头不能可靠消除该方向。

K1-B0 不训练、不修模型。它只定位这个信号是否已经进入共享的 penultimate
representation nuisance subspace：

```text
K0-B high-risk probes
        ↓
penultimate response Δh_q(u)=h(A_q(u))-h(u)
        ↓
跨公共载体稳定性 C_h
        ↓
Bank A 建子空间、Bank B 验证（以及 B→A）
        ↓
若强 CLE 模型明显更强且超过随机同秩子空间，才授权后续 SNR surgery
```

它不是完整方法训练，也不是论文最终结果。正式 verdict 只允许二选一：

```text
GO_TO_K1_B_SNR_SURGERY
NO_GO_SHARED_NUISANCE_ROUTING
```

## 2. 冻结输入

继续使用 OpenI 上 2026-08-30 创建的同一个 Phase-B0 数据集：

```text
数据集英文名：cle-pncb-phase-b0-seed0
数据集中文名：CLE公共规范化Phase-B0输入
文件：cle_public_canonicalization_phase_b0_seed0_inputs.tar.gz
大小：535256689 bytes（约 510.46 MiB）
SHA256：DFB766F6494A5F61AA16F45666EC250A30501066AB54D89C984CD2324293B9BC
```

实际读取内容只有：

- CIFAR-100 public image tar（只读取 image matrix，禁止读取 label）；
- H0/H9/L0/L9 的 16 个 round-40 checkpoint。

`evaluation/` 虽然仍在旧输入包内，但 K1-B0 入口不会把它传给 analyzer，也不会读取。
因此不需要重新上传数据集。

## 3. 冻结数据角色

公共图像拆分完全复用既有冻结索引：

- `D_select`：K0-B 原来的 1000 张，hash
  `731B8CFF...F6CA`，只负责复用 high-risk probe 和计算 representation-energy matched low；
- `D_rep`：K1-A 留出的 2000 张 holdout，hash
  `321C0910...40EE`，只负责正式表征定位；
- `U_a`：`D_rep` 前 1000 张；
- `U_b`：`D_rep` 后 1000 张。

HFL 的 probe selection、权重和 low matching 只由 H9 决定，并原样用于 H0；Local 同理由
L9 决定并原样用于 L0。这样 9/0 比较不会因为各自重新选 probe 而失配。

## 4. 核心统计量

对每个 client、probe 和 carrier：

```text
Δh_q(u) = h(A_q(u)) - h(u)
```

在 `U_a/U_b` 分别计算均值与能量，得到表征共享性：

```text
C_h(q) = relu(<μ_a, μ_b>) / (sqrt(E_a E_b) + 1e-12)
```

high-risk probe 使用 K0-B 冻结 rho 归一化权重。matched-low 从 active 且非 high 的 probe 中，
按 `D_select` 上 log representation energy 最近原则一对一匹配，probe id 是确定性 tie-break。

跨 bank 子空间迁移：

```text
M_A = [sqrt(w_q) μ_q]          # Bank A + U_a
N_A = span(M_A)                # float64 SVD
G_A→B(q) = ||N_A^T μ_q^B||² / (||μ_q^B||² + eps)   # Bank B + U_b
```

同时计算 B→A。数值秩阈值为 `σ > 1e-6 σ_max`，不强迫 H9/H0 或 L9/L0 秩相同。

随机基线为 100 个 Gaussian-QR 同秩子空间，seed `20260911`；bootstrap 为 carrier-level
2000 次，seed `20260910`，每次重新计算均值、SVD、子空间和 transfer，9/0 使用同一组重采样。

## 5. 冻结 20 gates

HFL 和 Local 分别要求：

1. Bank A `C9/C0 >= 1.25`；
2. Bank B `C9/C0 >= 1.25`；
3. combined `C9/C0 >= 1.50`；
4. A/B 平均后 positive clients `>=3/4`；
5. combined paired-bootstrap `CI95(C9-C0).lower > 0`；
6. strong arm 的 Bank A high/matched `>=1.10`；
7. strong arm 的 Bank B high/matched `>=1.10`；
8. strong arm 的 combined high/matched `>=1.20`；
9. A→B 与 B→A 都至少 `3/4` strong clients 的 true G 超过 random Q95；
10. combined `G9-G0` paired-bootstrap CI95 lower `>0`。

两套系统共 20 gates，必须全部通过才返回 `GO_TO_K1_B_SNR_SURGERY`。

## 6. OpenI 正式运行

本地 INSPECT 和 tiny smoke 已通过，因此 OpenI 不需要重复 smoke。选择原 Phase-B0 数据集，
代码仓库更新后，启动文件填写：

```text
scripts/openi_cle_k1_b0_cdr_snr_entry.py
```

运行参数：

```text
--mode=formal
```

完整命令等价于：

```bash
python scripts/openi_cle_k1_b0_cdr_snr_entry.py --mode=formal
```

它虽然零训练，但需要对 16 个异构模型、两套 bank 和两个公共集合做大量 forward，并做
2000 次 carrier bootstrap，所以不会像普通 analyzer 一样瞬间结束。

## 7. 下载与放置

正式结束后优先下载：

```text
cle_k1_b0_cdr_snr_seed0_formal_outputs.tar.gz
```

该包只含 JSON/CSV/bootstrap 小数组和审计 manifest，不保存新 checkpoint，预期远小于
K1-A 的 2.1 GiB 包。放入：

```text
outputs/openi_downloads/cle_k1_b0_cdr_snr_seed0/
```

然后再进行独立 hash、20-gate 和 bootstrap 复算。结果出来后必须停止；即便 GO，也不能
自动开始 SNR surgery 或完整 FL 训练。

## 8. 明确禁止

- 不训练、fine-tune 或更新模型；
- 不构造 optimizer，不调用 backward；
- 不写 checkpoint；
- 不读取 CIFAR-100 label、CLE binding、corruption type/family/severity 或私有标签；
- 不读取 DSA/WCCA/CFG evaluator；
- 不使用 classifier weight `W` 作为 primary localization 对象；
- 不根据结果改 probe、阈值、rank 或随机种子；
- 不复活 K1-A、PNCB、PEW/BER 或通信改造。
