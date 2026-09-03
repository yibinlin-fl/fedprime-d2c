# CLE K1-C CRSF Checkpoint Surgery（冻结协议）

## 目的

K1-C0 已观察到：强 CLE 的 H9/L9 checkpoint 对 64 个通用 PRIME 扰动的表征响应，
比 H0/L0 更集中在少数共享方向。K1-C 不再训练 RAHFL，而是在既有强 CLE round-40
checkpoint 上做最小因果干预，检验降低这种响应谱集中度是否会同时降低真实 CLE shortcut。

候选方法是 Cross-Carrier Response Spectrum Flattening（CRSF）：

```text
delta_h_q(u) = h(A_q(u)) - h(u)
mu_q         = mean_u delta_h_q(u)
E_q          = mean_u ||delta_h_q(u)||^2
r_q          = mu_q / (sqrt(E_q) + 1e-12)
S            = [r_1,...,r_64]
K            = S^T S
L_CRSF       = ||K||_F^2 / (trace(K)^2 + 1e-12)
```

它优化方向集中度而非响应幅度。Formal 禁止 mini-batch 近似；每个参数步用全 2000 张
公共图像的两遍 sufficient-statistic VJP 得到精确矩梯度。

工程实现允许复用数学上完全相同的中间量，但不得改变上述精确协议。特别是：每个 Adam
candidate update 后，必须在完整 `D_surgery x 64` 上重新计算 updated exact objective，
并在完整 `D_surgery` 上重新计算相对原始 checkpoint 的 anchor KL；只有二者都满足门槛
才 accept，否则恢复参数与 Adam state 后回溯。上一 accepted step 的 post-update exact
moments 可作为下一步的 pre-update moments；同一初始 checkpoint 的三个 LR 可共享 step-1
更新前 moments、anchor reference 和 exact gradient。

## 冻结输入与拆分

- 输入包：`cle_public_canonicalization_phase_b0_seed0_inputs.tar.gz`
- 大小：`535256689` bytes
- SHA256：`DFB766F6494A5F61AA16F45666EC250A30501066AB54D89C984CD2324293B9BC`
- checkpoint：只用 H9/L9，各 4 个异构客户端；H0/L0 仅保留为 K1-C0 观察对照
- `D_surgery`：2000 张无标签 CIFAR-100，SHA256 `B5441E50...5334A`
- `D_holdout`：另一组 2000 张无标签 CIFAR-100，SHA256 `321C0910...40EE`
- AB：Bank A 校正，Bank B 未见评估；BA 反向，且均从原 checkpoint 独立开始

Formal Stage 1 不解读 CIFAR-100 标签，也不得加载 CLE binding、真实损坏算子、
CIFAR-10 标签、DSA/WCCA/CFG。Stage 1 文件全部写出并生成
`primary_taxonomy_free_manifest.json` 后，Stage 2 才能读取 oracle 资产。

## 可训练范围

代码按仓库真实计算图核对并冻结：

| 架构 | 唯一可训练 stage | 其余部分 |
|---|---|---|
| ResNet10/12 | `layer4` 中非归一化参数 | 冻结 |
| ShuffleNet | `layer3` 中非归一化参数 | 冻结 |
| MobileNetV2 | `layers.16` 最后 inverted-residual 中非归一化参数 | 冻结 |

分类头、早期 backbone、全部 BN 参数与 running statistics、projector 均冻结。只保存
late-block 参数 delta，不复制完整 checkpoint。frozen-prefix float32 cache 只存在 OpenI 临时盘，
不进入结果包。

PRIME transformed public inputs 与模型无关，因此 calibration/formal 先各自冻结一次 input-level
float32 cache，随后跨 architecture/system 复用。prefix cache 仍由每个具体 checkpoint 计算，
但只使用一个覆盖式 architecture workspace，避免四种架构缓存累计约 30 GiB。input/source/
checkpoint/PRIME manifest hash 必须保持冻结；浮点输出文件本身不要求 SHA256 与旧实现相同。

## 五个实验臂

每个 `system × client × fold` 同时运行：

1. Frozen：原 checkpoint；
2. CRSF：优化归一化响应谱集中度；
3. SharedMean：优化 `mean_q ||mu_q||^2`；
4. Generic Invariance：优化 `mean_q E_q`；
5. RawSpec：只压平干净公共表征协方差谱。

四个非 Frozen loss 都除以各自初始 loss，使初始归一化目标为 1。Adam 使用默认参数、
`weight_decay=0`；Formal 恰好 10 个 accepted steps。每次尝试同时要求目标不增且原始
`D_surgery` 上 anchor KL 不超过 0.02，否则恢复参数和 Adam 状态、LR 减半后重试。

## 执行顺序

```bash
# 1. OpenI 可不重复 inspect；本地已完成
python scripts/openi_cle_k1_c_crsf_surgery_entry.py --mode=inspect

# 2. 执行性 smoke，不作科学判断
python scripts/openi_cle_k1_c_crsf_surgery_entry.py --mode=smoke

# 3. 盲数值校准，只读 D_surgery + correction bank
python scripts/openi_cle_k1_c_crsf_surgery_entry.py --mode=calibration

# 4. 只有校准结果独立审计并冻结进仓库后才能执行
python scripts/openi_cle_k1_c_crsf_surgery_entry.py --mode=formal
```

学习率候选固定为 `1e-5 / 3e-5 / 1e-4`。每个架构的候选要同时通过 H9-AB、H9-BA、
L9-AB、L9-BA 三步校准；选择全部通过的最大值，并原样复用于全部四个非 Frozen 对照。
校准不得查看 holdout、unseen chi、DSA、WCCA、CFG 或真实损坏元数据。

## Formal 门控

Stage 1 要求 H9/L9 combined unseen chi 各下降至少 25%，每个 AB/BA 至少 15%，各系统
至少 3/4 客户端为正，CRSF 每个 system/fold 的未见响应能量保留至少 50%，且 CRSF
相对 RawSpec 的 chi 降幅优势各至少 10pp。

Stage 2 要求 H9/L9 的 DSA 绝对下降至少 0.05 或相对下降至少 25%，各至少 3/4 客户端
为正，CRSF 相对 RawSpec 的 DSA 降幅优势各至少 0.02；同时 WCCA 至少 +1pp、CFG 至少
改善 1pp，Avg/Worst/Clean 各不下降超过 1pp。若 SharedMean 或 GI 在 H9 和 L9 上都按
冻结六项容差完全支配 CRSF，则不能给方法 GO。

唯一 verdict：

```text
GO_TO_TRAINING_INTEGRATION
MECHANISM_PASS_INTEGRATION_NEEDS_REDESIGN
NO_GO_CRSF_INTERVENTION
```

即便 GO 也必须停止，不得自动实现 40-round CRSF 训练。

## 当前验证状态

- 模型图 INSPECT：PASS，四个 adapter 表征与原 `model.backbone` 最大误差均为 0；
- 精确梯度：CRSF/SharedMean/GI/RawSpec 均通过 `relative error <=1e-5`、
  `cosine >=0.99999`；
- 本地 tiny smoke：17/17 检查通过，verdict 为
  `SMOKE_ONLY_NO_SCIENTIFIC_DECISION`；
- blind calibration：尚未运行；
- Formal：被校准 manifest 锁住，尚未运行。

下一步只运行 OpenI `--mode=calibration`。不要跳过校准直接 formal，也不要依据 smoke
效果量改学习率、步数、开放层或门槛。

## 等价优化验收与运行可观测性

- old-vs-new objective/moments 在冻结数值容差内一致；
- gradient relative error `<=1e-5`，cosine `>=0.99999`；
- accepted/rejected 决策必须完全一致；
- heartbeat 覆盖 architecture/system/fold/LR/step/pass/probe；
- 每个 LR 完成后原子写入 `calibration_progress.json`，同一容器内可按完整输入签名 resume；
- calibration manifest 记录 PRIME preprocessing、各 architecture/context prefix cache、单个 exact
  gradient、post-update exact objective、单 context 与总耗时；
- 优化期间仍禁止读取 unseen bank、CLE binding、DSA/WCCA/CFG 或任何任务标签。
