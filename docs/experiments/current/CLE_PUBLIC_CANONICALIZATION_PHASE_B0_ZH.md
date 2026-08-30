# CLE Public Canonicalization Phase-B0：Bridge-Only Kill Test

Updated: 2026-08-30

## 1. 任务边界

Phase-B0 只判断 Public Nuisance Canonicalization Bridge（PNCB）是否满足进入分类训练的最低
条件。它允许训练一个公共无标签重建器，并复用已经训练完成的 H0/H9/L0/L9 checkpoint 做
推理；不允许更新四个分类器、不允许实现 SCDW 训练 loss、不允许启动12轮 A/B/C。

```text
public canonicalizer training: 允许
classifier training:           禁止
communication change:          禁止
private corruption labels:     canonicalizer/withdrawal estimator 禁止读取
family/binding truth:          仅最终 oracle scoring 可读取
final test:                    禁止用于训练、选择和调参
```

## 2. 冻结资产

```text
private/evaluation package:
local_runs/cle_shortcut_amplification_phase_a1a/
  cle_shortcut_amplification_phase_a1a_seed0/

public data:
public/cifar-100-python.tar.gz

evaluation sources:
1000 balanced clean CIFAR-10 sources x 16 frozen operators, severity=3

classifier arms:
h0 = HFL gamma0
h9 = HFL gamma0.9
l0 = Local gamma0
l9 = Local gamma0.9

client models:
ResNet10 / ResNet12 / ShuffleNet / MobileNetV2
```

正式运行必须提供四臂共16个最终 checkpoint，目录固定为：

```text
checkpoint_root/
  h0/client_0.pt ... client_3.pt
  h9/client_0.pt ... client_3.pt
  l0/client_0.pt ... client_3.pt
  l9/client_0.pt ... client_3.pt
```

本地目前只有四个共享初始 checkpoint，不能替代这些最终权重，也不能用于科学判定。

## 3. 公共 canonicalizer

入口：

```text
scripts/train_cle_public_canonicalizer_phase_b0.py
```

输入是 CIFAR-100 public image 的 RAHFL AugMix views：

```text
clean public image -> target
AugMix view 1/2    -> canonicalizer input
public class label -> loader 可以返回，但训练代码明确忽略
```

冻结正式默认值：

```text
model:             PublicNuisanceCanonicalizer, base_channels=24
residual_scale:    0.5
epochs:            10
public_size:       50000
batch_size:        128
optimizer:         AdamW
learning_rate:     2e-4
loss:              L1 + 0.2 * local-SSIM-loss
seed:              20260830
```

Phase-B0 不以 public reconstruction loss 决定 GO；真正门槛是后续 private paired grid 上的
semantic preservation 和 original-nuisance contraction。

## 4. 三个 bridge

```text
Identity:       bridge(X)=X
AugMix overlay: 对已经损坏的 X 再调用发布版 RAHFL AugMix
PNCB:           bridge(X)=C_phi(X)
```

AugMix overlay 使用原发布实现和逐图固定 seed，只作为“继续叠加增强”的负对照。它不是新的
方法，也不能被报告为 canonicalization。

## 5. Withdrawal 估计与 oracle scoring

估计器先在不知道 family/binding 的情况下计算：

\[
M_{i,o,c}=\mathbb E_{Y\ne c}
[p_i(c\mid X_o)-p_i(c\mid bridge(X_o))].
\]

SCDW observable 对 operator 轴平均后再取正部平方。只有 `M` 全部生成后，分析器才按真实
family 聚合 `M` 并计算 binding retrieval mAP/hit。后者是 oracle 归因，不是训练输入，也不
表示客户端知道 corruption family。

## 6. 正式门槛

```text
G1 semantic preservation:
   四臂所有客户端中最差 canonical accuracy delta >= -1.0pp

G2 old-nuisance contraction:
   within-source operator variance contraction >= 25%

G3 family separability:
   source-folded residual family separability relative reduction >= 30%

G4 HFL directional retrieval:
   h9 mAP >= 0.65, h9-h0 mAP >= 0.20, hit >= 0.70, positive clients >= 3/4

G5 Local directional retrieval:
   l9 mAP >= 0.65, l9-l0 mAP >= 0.20, hit >= 0.70, positive clients >= 3/4

G6 overlay attribution:
   PNCB variance contraction - overlay contraction >= 0.10

G7 clean artifact null:
   max-arm clean-vs-canonical-clean SCDW <= 0.03
```

全部通过：`GO_TO_12ROUND_ABC`。任一失败：`NO_GO_PNCB_BRIDGE`。不得在看到正式结果后修改
阈值，也不得通过调 SCDW 权重救 bridge。

## 7. 分析入口与产物

入口：

```text
scripts/analyze_cle_public_canonicalization_phase_b0.py \
  --phase-a1a-root <prepared-package-root> \
  --checkpoint-root <four-arm-checkpoint-root> \
  --canonicalizer-checkpoint <public_nuisance_canonicalizer.pt> \
  --output-dir outputs/cle_public_canonicalization_phase_b0_seed0
```

输出：

```text
cle_public_canonicalization_phase_b0_summary.json
cle_public_canonicalization_phase_b0_per_client.csv
cle_public_canonicalization_phase_b0_predictions.npz
```

概率 cache 不保存三套图像数组，以控制 OpenI 回传大小；bridge quality 已固化在 summary。

## 8. Smoke 规则

`--smoke --max-sources <small N>` 至少需要20个样本；分析器按类别分层抽取，确保每类至少
两个样本。Smoke 只检查：

```text
canonicalizer checkpoint 可加载；
16个分类 checkpoint strict load；
三个 bridge 形状/范围正确；
概率、CSV、JSON、NPZ 能生成；
family/binding 只在 oracle scoring 打开。
```

Smoke 必须输出 `SMOKE_ONLY_NO_SCIENTIFIC_DECISION`，任何 smoke accuracy 都不能成为证据。

## 9. 当前状态

```text
spec:                       FROZEN BEFORE FORMAL RESULT
canonicalizer model:        IMPLEMENTED
canonicalizer train entry:  IMPLEMENTED
bridge analyzer:            IMPLEMENTED
OpenI entry:                IMPLEMENTED; default mode is smoke
slim OpenI input:           BUILT AND HASHED
focused tests:              7 PASSED in the current focused test file
CPU public-train smoke:      PASSED (4 images, 1 batch; execution only)
OpenI end-to-end smoke:      NOT STARTED
OpenI formal run:           NOT STARTED
```

## 10. OpenI 输入与入口

本地打包器：

```text
scripts/prepare_cle_public_canonicalization_phase_b0_input.py
```

生成物：

```text
local_runs/cle_public_canonicalization_phase_b0/
  cle_public_canonicalization_phase_b0_seed0_inputs.tar.gz

bytes:   535256689
sha256:  DFB766F6494A5F61AA16F45666EC250A30501066AB54D89C984CD2324293B9BC
```

该包只包含：1000 张冻结评估源图及其标签、CIFAR-100 公共 tar、四臂共16个 round-40
最终 checkpoint 和逐文件哈希 manifest。它明确排除私有训练数组及重复的 round-12
checkpoint；因此不需要重新上传原 Phase-A1a 的 408 MB 训练输入包。

OpenI 入口：

```text
scripts/openi_cle_public_canonicalization_phase_b0_entry.py --mode=smoke
```

入口默认也是 `smoke`，会训练1 epoch/2 batch 的临时公共桥，并只取20个平衡源样本跑三桥
和16个分类 checkpoint 的端到端推理。它只验证执行，输出必须标记
`SMOKE_ONLY_NO_SCIENTIFIC_DECISION`。Smoke 全通过并由用户确认以后，正式入口才允许改为：

```text
scripts/openi_cle_public_canonicalization_phase_b0_entry.py --mode=formal
```

正式模式固定使用第3节的10 epoch公共训练和1000源图完整审计；不得在 smoke 后改门槛。
