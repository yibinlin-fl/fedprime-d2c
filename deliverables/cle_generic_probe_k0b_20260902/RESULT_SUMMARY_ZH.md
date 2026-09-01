# CLE K0-B v2 Taxonomy-Free Generic Probe 正式结果

日期：2026-09-02
结论：`GO_TO_K1_CHECKPOINT_SURGERY`

## 1. 输入与完整性

```text
archive: cle_generic_probe_k0b_seed0_formal_outputs.tar.gz
bytes:   234888047
sha256:  1E02A16C765D8AB976A692D444FA9DAEBE38C30F8279CD6DCCFC49D1BFF88608
```

- 压缩包共34个成员、30个文件，无不安全路径、符号链接或硬链接。
- `primary_artifact_manifest.json` 中全部文件的长度与SHA256均匹配。
- `blind_response_manifest.json` 中16份response文件的SHA256均匹配。
- 1,000个公共样本index互不重复，array hash为
  `731B8CFFDCBD241474D33B261E323F9EC11C2EA59BC7705261140A3B8572F6CA`，与冻结协议一致。
- 16份响应张量均为有限值，`centered_response` shape均为`(1000,128,10)`。
- 返回的两份PRIME bank state与仓库冻结资产逐字节一致。manifest内容相同；原始字节hash差异仅来自
  Windows checkout的CRLF与OpenI产物的LF换行差异。
- 16个checkpoint hash全部不同，且与blind manifest记录一致。

## 2. 独立重算

从16份原始`centered_response`张量独立重算`S/Dcf/K/R`，与返回结果的最大绝对误差为：

```text
5.551115123125783e-17
```

使用冻结的bootstrap seed分别重跑HFL与Local的1,000次paired-carrier bootstrap，重新执行冻结
决策函数，得到：

```text
reported verdict:   GO_TO_K1_CHECKPOINT_SURGERY
recomputed verdict: GO_TO_K1_CHECKPOINT_SURGERY
generic fragility kill: false
```

CPU重算与OpenI报告的置信区间只有浮点计算路径导致的末位差异，不影响任何门槛。

## 3. Arm指标

| arm | S | Dcf | K | R |
|---|---:|---:|---:|---:|
| H0 | 8.142875 | 0.899299 | 0.076059 | 0.053017 |
| H9 | 25.695404 | 12.627097 | 0.328786 | 0.259867 |
| L0 | 13.907224 | 1.762881 | 0.090685 | 0.052454 |
| L9 | 32.620350 | 15.310866 | 0.323437 | 0.230052 |

这里真正的主证据不是`S`变大，而是`K`和`R`同时显著变大：gamma9模型不只是对generic probe
更敏感，其响应还能够跨两组互斥公共carrier保持方向一致，并集中指向特定类别分量。

## 4. 冻结gate

### HFL：H9对H0

| gate量 | 结果 | 冻结门槛 | 判定 |
|---|---:|---:|---:|
| Dcf delta | 11.727798，CI95 `[11.399661,12.039224]` | lower > 0 | PASS |
| K delta | 0.252727，CI95 `[0.247124,0.258539]` | delta >= 0.03且lower > 0 | PASS |
| combined R ratio | 4.901569 | >= 1.20 | PASS |
| R delta CI95 | `[0.200498,0.212592]` | lower > 0 | PASS |
| positive R clients | 4/4 | >= 3/4 | PASS |
| Bank A R ratio | 5.739226 | >= 1.10 | PASS |
| Bank B R ratio | 4.317300 | >= 1.10 | PASS |

四个client的R delta分别为：

```text
+0.097401, +0.391802, +0.089175, +0.249022
```

### Local：L9对L0

| gate量 | 结果 | 冻结门槛 | 判定 |
|---|---:|---:|---:|
| Dcf delta | 13.547985，CI95 `[13.193105,13.915590]` | lower > 0 | PASS |
| K delta | 0.232752，CI95 `[0.227374,0.238252]` | delta >= 0.03且lower > 0 | PASS |
| combined R ratio | 4.385780 | >= 1.20 | PASS |
| R delta CI95 | `[0.171834,0.183316]` | lower > 0 | PASS |
| positive R clients | 4/4 | >= 3/4 | PASS |
| Bank A R ratio | 5.166668 | >= 1.10 | PASS |
| Bank B R ratio | 4.094945 | >= 1.10 | PASS |

四个client的R delta分别为：

```text
+0.121450, +0.314705, +0.021863, +0.252373
```

HFL与Local各自8/8 gates通过。虽然两者的`S`也显著上升，但`K/R`全部通过，因此冻结的
`NO_GO_GENERIC_DIRECTIONAL_SIGNAL` kill rule没有触发。

## 5. 科学结论边界

本结果支持：

1. 不使用真实corruption type、family、severity、CLE binding或private corruption metadata时，
   两套预冻结PRIME generic probe bank仍能区分gamma0与gamma0.9模型。
2. 被识别的不是单纯generic fragility，而是`carrier-stable + class-selective directional response`。
3. 两套独立bank均复现，四种异构backbone上方向一致，效应远高于冻结门槛。
4. Local与HFL都通过，继续支持“shortcut首先由本地CLE监督形成”的local-first解释；该结果不支持
   把communication amplification写成主机制。

本结果尚不支持：

1. K0-B不是最终训练方法，不能证明移除该方向会提高WCCA、CFG或任务准确率。
2. 它没有证明PRIME以外的任意generic generator都有效；A/B只是同一maximum-entropy PRIME
   distribution中的独立冻结bank。
3. 它不证明真实世界部署泛化，也不证明跨数据集、跨seed稳定。
4. 它不允许直接进入完整训练或论文主实验；只允许设计最小K1 checkpoint surgery因果验证。

## 6. 下一步

按照预注册流程，下一阶段是`K1 checkpoint surgery`：保持checkpoint冻结或只做明确的参数/响应方向
干预，验证降低K0-B风险是否会定向改善CLE counterfactual指标，同时尽量保持clean与原分布性能。
在K1设计、对照组、恢复门槛和停止规则冻结前，不启动训练，不恢复PNCB、PEW/BER，也不修改通信。
