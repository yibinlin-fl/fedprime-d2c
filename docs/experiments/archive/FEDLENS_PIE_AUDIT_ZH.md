# FedLENS-PIE 表征审计 A

Updated: 2026-08-10

## 目的

这一阶段只验证一个新的、本地的 taxonomy-free 表征模块：Paired
Intervention Encoder（PIE）。它不是对已冻结 handcrafted continuous witness
的调参，也暂不接入联邦训练、BER 或 AsymHFL 通信。

PIE 从公共图像构造跨内容成对样本：两张不同语义图像接受同一个随机干预程序，
但腐蚀随机数相互独立。训练仅要求成对表征一致、批内可区分且不坍缩。具体算子名
只作为图像生成指令；训练数据契约不含 `noise/blur/weather/digital/unknown` 家族标签，
也不训练环境分类头。公共语义标签只在审计阶段用于检测内容泄漏，不参与训练。

## 与冻结负结果的边界

冻结的 continuous witness 使用人工设计的 22 维统计量和后续 covariance risk/CDep。
PIE 使用可学习的跨内容干预不变量，不复用旧特征、旧 covariance objective、旧阈值或
旧 runner。Audit A 不通过时直接停止，不允许通过只调 loss 权重、维度或阈值复活。

## 严格数据角色

```text
PIE train:    公共 CIFAR train 子集；不读取语义标签；只使用 seen concrete operators
Audit seen:  与训练图像不重叠的公共子集；seen operators
Audit holdout: 同一独立公共子集；4 个 held-out concrete operators
Family labels: 全流程禁用
Private CLE fit/audit/test: 本阶段完全不读取
```

默认 held-out operators 为：

```text
impulse_noise, zoom_blur, fog, pixelate
```

它们只用于 Audit A，不能进入 PIE 训练干预池。

## 预注册门槛

以下门槛全部通过才允许讨论 Phase 2 的 propensity-balanced risk（PBR）接入：

```text
seen cross-content retrieval lift       >= 5.0
held-out cross-content retrieval lift   >= 3.0
seen and held-out severity Spearman    >= 0.50 (each)
active dimensions (std >= 0.05)         >= 75%
seen and held-out content probe accuracy <= max(5%, 2 x own probe chance)
```

`retrieval lift` 使用同干预程序的跨内容配对检索准确率除以经验随机命中率。门槛不是
最终论文结论，只是阻止无效表征进入昂贵联邦实验的 promotion gate。结果返回后不得
修改窗口或阈值追认成功。

## 入口

仅做 execution smoke：

```powershell
python scripts/audit_fedlens_pie.py --smoke
```

完整 Audit A（尚未运行）：

```powershell
python scripts/audit_fedlens_pie.py `
  --public_dataset=cifar100 `
  --public_size=5000 `
  --audit_size=1000 `
  --epochs=5 `
  --seed=0
```

默认输出：

```text
local_runs/fedlens_pie_audit/audit_report.json
local_runs/fedlens_pie_audit/pie_encoder.pt
```

Smoke 强制只用 8 个训练样本、8 个审计样本、单 batch 和轻量算子子集；其指标、
门槛结果和 accuracy 都不是科学证据。

## 当前状态

PIE 模块、独立 CLI、结构化报告、checkpoint 元数据和 5 个聚焦测试已实现。相关
协议回归合计 `18 passed`。一个 8+8 样本 CUDA smoke 已完整生成 JSON 和 checkpoint；
它未通过表征门槛是预期现象且不是负科研结果。完整 Audit A 的结果如下；PBR 尚未
实现，当前 CLE-HFL runner 也未修改。

## 完整 Audit A 结果（2026-08-10）

冻结的 seed-0 Audit A 已使用 5,000 个训练样本和 1,000 个不重叠审计样本完成：

```text
metric                              value      gate
seen retrieval lift                5.325035   PASS (>= 5.0)
held-out retrieval lift            5.213998   PASS (>= 3.0)
seen severity Spearman             0.588224   PASS (>= 0.5)
held-out severity Spearman         0.498202   FAIL (>= 0.5)
active dimension fraction          1.000000   PASS (>= 0.75)
seen content probe accuracy        0.011745   PASS (<= 0.05)
held-out content probe accuracy    0.020134   PASS (<= 0.05)
```

严格结论：`NO-GO (6/7)`。唯一失败项距离阈值 `-0.001798`，因此这是非常接近
边界的 held-out ordinal-generalization 失败，不是表征坍缩、内容泄漏或完全无法
识别 unseen intervention。按照预注册纪律，不修改 `0.5` 阈值，也不通过追加随机
种子追认成功；当前 checkpoint 仅作为审计证据，不进入 PBR 或联邦 runner。

结果文件：

```text
local_runs/fedlens_pie_audit_seed0/audit_report.json
local_runs/fedlens_pie_audit_seed0/pie_encoder.pt
```
