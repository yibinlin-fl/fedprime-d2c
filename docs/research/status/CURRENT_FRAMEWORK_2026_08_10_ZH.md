# FedPRIME-D2C 当前项目框架（一屏版）

Updated: 2026-08-10

## 研究问题

项目研究模型异构联邦学习中的三重耦合：不同客户端模型结构不同、类别分布
label-skew Non-IID、图像损坏环境与类别同时不均衡。因为模型结构不同，不能直接
FedAvg 参数；通信主要依靠无标签公共图像上的 logits 知识蒸馏。

## 当前主流程

```text
prepared CLE-HFL v2 data
  -> per-client 85% fit / 15% private audit
  -> four heterogeneous client models
  -> local update
       AugMix + JSD + DCL
       + optional hard PEW annotations
       + optional hard BER class-environment balancing
  -> public-logit communication
       none / symmetric HFL / audit-routed AsymHFL-val
  -> final CLE test reporting
       Avg / Worst / WCCA / CFG / seen-unseen operator metrics
```

严格角色：`fit` 只用于本地梯度；client-private `audit` 只用于 AsymHFL teacher
排序；final test 标签只用于最终报告，不能参与 routing、选择、early stopping 或调参。

## 数据与模型

```text
private benchmark: CLE-HFL v2, 4 clients
heterogeneous models: config 中冻结的四种不同 backbone
public data: CIFAR-100 train images, normally unlabeled for communication
corruption protocol: 15 concrete operators, standard split 11 seen / 4 held-out
```

具体 operator 元数据主要用于数据生成和最终审计。当前 hard PEW 是历史上仍存在的
例外：它把公共合成损坏监督为 clean/noise/blur/weather/digital/unknown，再把私有
样本标成离散 environment id。

## 本地模块

基础本地目标：

```text
classification + AugMix JSD + DCL
```

当前选中的增强是：

```text
calibrated hard PEW + hard BER
```

PEW 预测离散损坏环境；BER 在 class x environment group 上平衡逐样本 CE。现有
A0--A6 消融证明 BER 是主要正增益来源。CDep-v1/v2 已被 matched 实验否定并从当前
local path 移除；EBST 等旧模块也被冻结。

需要区分两层证据：历史 strict 多种子和 40-round seed-0 强结果来自当时的
`PEW+BER+CDep` package；后续共享-PEW 消融证明 CDep 没有正因果贡献，因此论文当前
计划使用的核心是 `hard PEW+hard BER`，其最终完整证据表仍需按 paper-evidence 计划
整理，不能把旧 package 的全部稳定性自动转写成 exact final-method claim。

## 通信模块

当前默认通信是 strict `AsymHFL-val`：

```text
none:      不通信，仅本地训练
HFL:       每个客户端向所有其他客户端做公共 logits KD
AsymHFL:   接收方只向 private-audit accuracy 不低于自己的客户端学习
```

每轮在公共图像上收集各模型 softmax，接收方对允许的 teachers 做 KL 蒸馏。它不做
模型参数聚合，因此兼容异构 backbone。AsymHFL 是项目采用的通信骨架/基线，不应包装
成项目原创贡献。PRAC-HFL、FedCARA v1 等旧通信创新已冻结为负结果。

## 指标

```text
Avg:    四客户端平均准确率，越高越好
Worst:  最差客户端准确率，越高越好
WCCA:   所有有效 class x environment 单元中的最低准确率，越高越好
CFG:    各类别跨环境 accuracy gap 的平均，越低越好
```

## 当前证据结论

```text
hard PEW + hard BER:              当前本地核心，强正消融信号
Multi-label PEW + Soft-BER:       NO-GO, 0/4 gates
PIE-v1 taxonomy-free audit:       NO-GO, 6/7，边界 held-out severity 失败
MPIE-v2 radial ordinal audit:     NO-GO，held-out severity 比 matched v1 低 0.073579
PBR:                              未实现、无资格进入 runner
AsymHFL-val:                      当前通信骨架，不是原创创新
```

因此，taxonomy-free 连续本地路线已经经过两次结构检验但未获得 promotion。不能把
它们继续调参复活。项目当前真正未解决的论文问题，是在保留已验证 hard PEW+BER
性能的同时，形成一个归因清楚、理论成立且不同于冻结负路线的原创贡献；下一轮应在
重新评估通信创新与完全不同的本地机制之间做一次明确选择，而不是继续混合调参。

## 代码地图

```text
fedprime/methods/rahfl_asymhfl.py          unified experiment runner
fedprime/methods/local_fedease.py          current PEW+BER local epoch
fedprime/methods/balanced_environment_risk.py  hard/soft BER losses
fedprime/communication/public_logits.py    none/HFL/AsymHFL strategies
fedprime/data/strict_fit_audit.py          persisted fit/audit protocol
fedprime/engine/cle_metrics.py             Avg/Worst/WCCA/CFG

fedprime/methods/latent_environment.py             isolated PIE-v1 audit code
fedprime/methods/monotone_latent_environment.py    isolated MPIE-v2 negative probe
```
