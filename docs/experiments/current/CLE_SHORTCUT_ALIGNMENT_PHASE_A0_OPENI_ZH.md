# CLE Shortcut Alignment Phase-A0 OpenI 审计规范（待确认）

Updated: 2026-08-30

## 0. 当前状态与边界

```text
status: FROZEN / implementation ready
training: NONE
formal inference: OpenI only
local RTX 3050: forbidden for formal inference
implementation: COMPLETE
OpenI task: NOT STARTED
```

本阶段只判断历史 CLE-HFL 模型是否真的把 corruption 当成类别线索。它不研究坏教师、
teacher selection、知识传播、PEW/BER、标签噪声、攻击或新方法。通过本阶段只允许进入
matched-global partition 的 Local-D/Local-E 设计，不等于论文课题已经 GO。

## 1. 已核对资产

### 1.1 历史模型

本地归档：

```text
outputs/cle_rahfl_diagnostic_outputs.tar.gz
bytes: 272728582
```

归档包含 `gamma=0.0/0.6/0.9` 各四个最终 checkpoint：

```text
ResNet10     about 30.20 MB
ResNet12     about 30.50 MB
ShuffleNet   about 15.48 MB
MobileNetV2  about 23.16 MB
```

Phase-A0 只复用 `gamma=0.0` 和 `gamma=0.9`，共八个 checkpoint；不重新训练，也不使用
`gamma=0.6`。两组均来自 seed0、40-round RAHFL、相同四种架构和相同 Dirichlet
`alpha=0.5` 协议。

### 1.2 现有测试数据的限制

旧 CLE v1 和 CLE-HFL v2 生成器都按同一批原始测试 index 依次生成多个 corruption，因而
配对身份可以由数组顺序恢复。但是不同 corruption 的 severity 独立随机采样，且没有显式
保存 `source_id`。因此旧 `test_balanced` 适合 Avg/WCCA/CFG 报告，却不满足这次更严格的：

```text
same clean image
+ same true label
+ same severity
+ only corruption operator changes
```

Phase-A0 将复用 CLE-HFL v2 seed0 中类别均衡的 1,000 张 `test_clean` 图像（每类 100 张），
在 OpenI 任务内生成新的确定性评价网格；不修改任何训练数据。

## 2. 冻结评价网格

旧 CLE v1 的四个 family 及 operator 为：

```text
noise:    gaussian_noise, shot_noise, impulse_noise, speckle_noise
blur:     defocus_blur, glass_blur, motion_blur, zoom_blur
weather:  snow, frost, fog, spatter
digital:  contrast, brightness, jpeg_compression, pixelate
```

冻结设置：

```text
clean source images: 1,000 (100 per CIFAR-10 class)
operators:           16 (all four operators in every old CLE family)
severity:            exactly 3 for every source/operator pair
generation seed:     deterministic hash of (source_id, operator_id, 20260830)
evaluation images:   1,000 x 16 = 16,000
models:              2 gamma conditions x 4 clients = 8
total forward items: 128,000
```

每条评价记录必须显式保存：

```text
source_id, true_class, operator_id, operator_name, family_id, severity
```

并审计每个 `source_id` 恰好出现 16 次、标签完全一致、severity 全为 3。

## 3. 训练绑定图

为了与历史 checkpoint 的真实 CLE v1 协议匹配，使用原生成器的固定 family 顺序：

```text
noise, blur, weather, digital
```

对客户端 `k` 和类别 `c`，其历史 dominant family 为：

\[
d(k,c)=g_{(c+k)\bmod 4}.
\]

令：

\[
\mathcal C_{k,g}=\{c:d(k,c)=g\}
\]

表示客户端 `k` 中训练时主要与 family `g` 绑定的类别集合。该绑定图只用于离线评价，
不得进入训练或重新路由。

## 4. Primary：Directional Shortcut Alignment

对 gamma 条件 `a`、客户端模型 `k`、family `g` 和输入 `z`，定义该 family 所绑定类别的
总预测概率：

\[
Q_{a,k,g}(z)=\sum_{c\in\mathcal C_{k,g}}p_{a,k}(c\mid z).
\]

对真实类别不属于 `C_{k,g}` 的 clean source，只改变 corruption family，定义：

\[
DSA_{a,k,g}
=
\mathbb E_{(x,y):y\notin\mathcal C_{k,g}}
\left[
\frac{1}{|\mathcal O_g|}\sum_{o\in\mathcal O_g}Q_{a,k,g}(T_o(x))
-
\frac{1}{3}\sum_{h\ne g}
\frac{1}{|\mathcal O_h|}\sum_{o\in\mathcal O_h}Q_{a,k,g}(T_o(x))
\right].
\]

客户端级 `DSA_{a,k}` 为四个 family 的等权平均；总体 DSA 为四个客户端的等权平均。

直观含义：若施加 family `g` 后，模型会更倾向输出训练中与 `g` 绑定的类别，即便真实类别
并不属于这些类别，则 DSA 为正。Primary contrast 为：

\[
\Delta DSA=DSA_{\gamma=0.9}-DSA_{\gamma=0.0}.
\]

同一评价图同时输入两组模型，因此该差分控制了 operator 固有难度和评价样本差异。

## 5. 随机绑定对照

推理只执行一次并缓存全部 softmax。随后对每个客户端独立生成 1,000 个随机绑定图：

```text
在 10 个类别之间打乱 dominant family
保持每个 family 原有绑定类别数量不变
permutation_seed = 20260830
```

对每个随机图重新计算 DSA，不重新运行模型。报告：

```text
observed DSA
permutation mean/std
empirical one-sided p-value
observed percentile in the null distribution
```

该对照检验模型的预测移动是否专门对齐真实训练绑定，而不只是任意类别集合。

## 6. Secondary 指标

以下均为解释性指标，不替代 Primary：

```text
paired prediction-flip rate
family-bound top-1 flip bias
fixed-severity Avg / Worst / WCCA / CFG
per-client and per-family DSA
per-operator DSA contribution
gamma0/gamma09 softmax entropy
```

CFG/WCCA 只能证明性能差异，不能单独证明 shortcut；最终结论必须以 DSA 与随机绑定对照为
核心。

## 7. 不确定性与冻结门槛

在 `source_id` 层进行 2,000 次 paired bootstrap。bootstrap 反映评价样本不确定性，不代表
训练种子稳定性；本阶段仍是 seed0 Kill Test。

完整性门槛全部必须 PASS：

```text
8 checkpoints found and loadable
1,000 unique clean source_ids
16 operators per source_id
all paired labels identical
all severities exactly 3
all predictions finite
gamma0/gamma09 use byte-identical evaluation images
```

科学门槛全部必须 PASS：

```text
G1 pooled Delta-DSA >= 0.020 absolute probability
G2 pooled paired-bootstrap 95% CI lower bound > 0
G3 at least 3/4 clients have Delta-DSA > 0
G4 observed gamma09 DSA exceeds the 95th percentile of its shuffled-map null
G5 at least 3/4 clients have one-sided shuffled-map p < 0.05
```

若任一完整性门槛失败，结果为 `INVALID`，修复协议后才可重跑。若完整性通过但任一科学门槛
失败，结果为：

```text
CLE directional-shortcut claim: NO-GO
do not run Local-D/Local-E
do not run Centralized/HFL comparisons
do not design a new method
```

全部通过只表示：

```text
GO to matched-global partition Local-D/Local-E design
not paper GO
not method GO
```

## 8. OpenI 输入、运行与回传设计

已实现：

```text
scripts/audit_cle_shortcut_alignment.py
scripts/openi_cle_shortcut_alignment_phase_a0_entry.py
tests/test_cle_shortcut_alignment.py
```

计划制作单个瘦身输入包：

```text
cle_shortcut_alignment_phase_a0_seed0_inputs.tar.gz
```

只包含：

```text
gamma0/gamma09 eight checkpoints
their two resolved configs
1,000 clean test images and labels
frozen input manifest and SHA256 values
```

实际输入包：

```text
path: local_runs/cle_shortcut_alignment_phase_a0/cle_shortcut_alignment_phase_a0_seed0_inputs.tar.gz
bytes: 184575308
sha256: C1F6823E186DDAF6DB44A38BCBDA300C78B9F1B4702C5E84B7AEE3A485499EFE
members: 13
```

它不包含完整 272.7 MB checkpoint 历史包和 363.2 MB CLE-v2 数据包。OpenI 入口沿用已验证
的 C2NET 行为：从 `context.dataset_path` 查找输入包，安全解包，
生成固定评价网格，执行推理和离线统计，将结果复制到 `context.output_path`，最后调用
`upload_output()`。

预期回传：

```text
cle_shortcut_alignment_phase_a0_seed0_summary.json
cle_shortcut_alignment_phase_a0_seed0_per_client.csv
cle_shortcut_alignment_phase_a0_seed0_per_family.csv
cle_shortcut_alignment_phase_a0_seed0_permutation.csv
cle_shortcut_alignment_phase_a0_seed0_predictions.npz
cle_shortcut_alignment_phase_a0_seed0_outputs.tar.gz
```

## 9. 资源估算

正式任务只包含 16,000 张 32x32 图像在 8 个小模型上的前向推理，不含梯度、优化器、
pretrain 或 communication。保守估计：

```text
V100/A100 model inference:       about 2--8 minutes
corruption generation/statistics: about 2--8 minutes
install/extract/package/upload:   about 5--15 minutes
expected total wall time:         about 10--30 minutes
```

OpenI 环境和依赖缓存会影响总时长。即使采用更慢 GPU，本任务也应远短于一次 12-round 训练。

本地只允许：

```text
archive member/hash audit
CPU manifest construction
tiny synthetic unit test / CLI help / analyzer dry-run
```

不得在本地 RTX 3050 上运行正式 8-checkpoint 推理。

## 10. 实现验证与待运行工作

```text
focused tests:             3 passed
checkpoint strict loading: 8/8 passed on CPU
archive members:           13 exact, no extra payload
formal local inference:    not run
formal OpenI inference:    not started
```

下一步仅为：保持不提交，等待用户明确允许 commit/push；随后用户上传上述输入包并在 OpenI
运行 `scripts/openi_cle_shortcut_alignment_phase_a0_entry.py`（无需额外参数）。回传后由 Codex
独立复算并按 G1--G5 给出 GO/NO-GO。
