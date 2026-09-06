# Calibrated Hard PEW + Hard BER 代码审计

日期：2026-09-06

结论：原实现仍在，当前可复现入口、PEW 模块、BER 模块和 strict AsymHFL-val 接线均未被删除。本次只做静态代码与证据审计，没有修改训练代码、运行实验或加载 GPU。

## 最终方法对象

### 1. Public Environment Witness（PEW）

环境集合为：

```text
clean, noise, blur, weather, digital, unknown
```

公共 CIFAR-100 图像只提供 carrier；环境监督由人工 corruption taxonomy 合成。四个已知环境分别从对应 operator pool 采样，`unknown` 是来自两个不同已知 family 的顺序复合。严重度为 1--5；clean 的严重度目标为 0。

对公共合成样本 `(x~, e, s)`，hard PEW 优化：

```text
L_PEW = CE(h_env(g(x~)), e)
      + 0.25 * 1[s > 0] * CE(h_sev(g(x~)), s - 1)
```

网络是三层卷积 encoder、32 维 embedding、六类环境 head 和五类严重度 head。使用 Adam，学习率 `1e-3`，训练 5 epochs；按公共 validation environment accuracy 恢复最佳 epoch，然后冻结。

`unknown_threshold=auto` 只在公共验证集上，以 `0.00, 0.01, ..., 0.95` 网格最大化环境准确率，平局选择更小阈值。正式 seed-0 配置得到阈值 `0.0`：这并不删除 unknown head；它表示不额外把低置信样本强制改成 unknown，unknown 仍可由六类 softmax 原生 argmax 预测。

冻结 PEW 对每个私有训练样本输出 hard pseudo-environment `e_hat`。私有 `train_corruption_ids.npy` 仅在 `_diagnostic_environment_ids` 中映射成 oracle family，用于打印/报告 group accuracy；该函数的代码注释明确为 `for reporting, never for training`。

### 2. Balanced Environment Risk（BER）

对每个客户端 `k`，只用 strict `fit` 索引统计类别×PEW 环境计数 `n[k,c,e]`。正式参数：

```text
support_gamma = 0.5
count_cap = 32
min_group_count = 2
assignment = hard  # 配置未显式写出，代码默认值
```

有效组满足 `n[k,c,e] >= 2`。类内环境权重为：

```text
a[k,c,e] = min(n[k,c,e], 32)^0.5
           / sum_e' min(n[k,c,e'], 32)^0.5
```

目标是有效类别均匀、类别内按上述 support-shrunk 权重平均组风险：

```text
R[k,c,e] = mean CE on fit samples with (y=c, e_hat=e)
L_BER    = mean_c sum_e a[k,c,e] * R[k,c,e]
```

实现以全 fit 集计数构造 sample weight，并乘以 dataset size，使 minibatch 均值成为该 grouped objective 的随机估计。它不是 max-risk GroupDRO，也不是 CVaR。

### 3. 完整本地目标和通信

正式本地目标为：

```text
L_local = L_BER + 12 * L_JSD + L_DCL
```

JSD 使用 clean 与两个 strong AugMix view；DCL 使用 clean/strong/weak feature。通信保持 strict AsymHFL-val，client-private `audit` 用于路由，`fit` 用于梯度，final-test labels 只用于报告。

## 正式 seed-0 配置

```text
scenario             cle_hfl_v2
private dataset      CIFAR-10
public dataset       CIFAR-100
models               ResNet10, ResNet12, ShuffleNet, MobileNetV2
rounds               12
local epochs         1
private batch size   64
public size          5000
public batch size    128
PEW validation       20%
PEW epochs           5
PEW LR               0.001
PEW severity weight  0.25
PEW threshold        auto
PEW label mode       hard (code default)
BER assignment       hard (code default)
BER gamma/cap/min    0.5 / 32 / 2
lambda JSD           12.0
preserve DCL         true
strict audit ratio   0.15
```

## 文件与入口

- PEW：`fedprime/methods/environment_witness.py`
- BER：`fedprime/methods/balanced_environment_risk.py`
- 本地目标：`fedprime/methods/local_fedease.py`
- PEW 准备和诊断隔离：`fedprime/methods/fedease.py`
- fit-only counts 和 annotated loader：`fedprime/data/fedease.py`
- strict runner 接线：`fedprime/methods/rahfl_asymhfl.py`
- 正式模板：`configs/openi_v100_fedease_pew_asymhfl_val_cle_v2_probe.yaml`
- 12 轮入口：`scripts/openi_strict_pew_asymhfl_entry.py`
- strict LOO 入口：`scripts/openi_cle_pew_loo_entry.py`

## 审计警告

1. 历史表格中的 `A1 BER-only` 命名不准确；A1 实际同时启用 calibrated hard PEW 和 hard BER，只是关闭 CDep。
2. 当前 YAML 没有显式写 `label_mode: hard` 和 `assignment: hard`，它们由代码默认值决定。写论文时应明确写出，不要让读者误以为使用 soft 分配。
3. PEW 依赖人工 corruption taxonomy，不能称 taxonomy-free、unsupervised environment discovery 或无环境监督。
4. 轮内时间不包含 PEW 训练和私有伪标注准备时间，因此效率表必须同时披露 PEW setup 口径。
5. 当前实现中的 soft/multi-label 分支属于后续冻结负路线，不是本文最终方法。
