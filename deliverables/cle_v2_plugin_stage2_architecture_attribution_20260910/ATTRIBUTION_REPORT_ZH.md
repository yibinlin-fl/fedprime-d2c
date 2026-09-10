# CLE-v2 Stage-2 PEW+BER 架构归因（零训练）

本报告仅使用冻结的 Stage-2 预测、固定训练划分与 PEW 注释，不重新训练。

## 客户端概览

| client | model | grid acc Δ | DSA reduction | PEW family acc | unknown | BER p95 multiplier | BER max multiplier |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | ResNet10 | +2.1533 | +0.111312 | 59.83% | 11.37% | 4.1876 | 134.8889 |
| 1 | ResNet12 | +6.0800 | +0.011317 | 59.27% | 5.01% | 2.7139 | 314.8148 |
| 2 | ShuffleNet | -5.1733 | +0.115762 | 52.49% | 3.09% | 4.6428 | 73.5234 |
| 3 | Mobilenetv2 | -1.5000 | +0.075178 | 76.70% | 2.06% | 4.9709 | 106.2125 |

最受损客户端为 client 2（ShuffleNet），operator-grid accuracy 变化 -5.1733。

## 该客户端下降最大的 operator

| operator | family | split | acc Δ | DSA reduction | PEW family acc | fit support |
|---|---|---|---:|---:|---:|---:|
| elastic_transform | digital | seen | -9.9000 | +0.079658 | 0.43% | 1635 |
| glass_blur | blur | seen | -9.9000 | +nan | 67.90% | 81 |
| zoom_blur | blur | unseen | -9.5000 | +nan | nan% | 0 |
| defocus_blur | blur | seen | -9.3000 | +0.007624 | 71.08% | 83 |
| jpeg_compression | digital | seen | -8.3000 | -0.012317 | 3.60% | 111 |

## 该客户端下降最大的类别

| class | fit share | acc Δ | prediction share Δ | PEW family acc | unknown | effective groups | max multiplier |
|---|---:|---:|---:|---:|---:|---:|---:|
| truck | 20.52% | -37.2667 | -12.2400 | 6.19% | 0.63% | 5.8040 | 10.6112 |
| automobile | 18.57% | -36.1333 | -14.8267 | 79.97% | 0.95% | 5.8682 | 9.2456 |
| ship | 26.25% | -34.2000 | -9.6467 | 38.05% | 0.94% | 5.9681 | 7.0536 |
| cat | 9.31% | -2.6000 | -6.0067 | 81.16% | 14.54% | 5.4406 | 23.7105 |
| dog | 0.01% | +0.0000 | +0.0000 | 100.00% | 0.00% | nan | 0.0000 |

## 当前归因结论

- client 2 的 DSA 仍下降 0.115762，因此不是 shortcut 抑制失效。
- 损失是类别选择性的：truck, automobile, ship 的预测召回降为0；同时 airplane 和 horse 分别提升 +29.60、+21.20。
- PEW 误分不能单独解释：automobile 的 PEW family accuracy 约80%，但其召回仍降为0。
- BER 全局权重过大也不能单独解释：client 2 的最大样本乘数反而是四个客户端中最低。
- 当前只能定位到 ShuffleNet/client-2 组合上的类别决策重分配；架构和非 IID 数据固定绑定，不能归因为“小模型容量”。

### client 2 按 corruption family 汇总

| family | mean grid acc Δ | mean PEW family acc on seen operators |
|---|---:|---:|
| blur | -8.8250 | 73.92% |
| digital | -7.3800 | 21.68% |
| noise | +2.6333 | 87.76% |
| weather | -4.4333 | 77.07% |

## 证据边界

- 这是描述性归因，不把相关性写成因果。
- 四个模型不足以支持跨架构相关性检验。
- 本报告未重新训练、未选择 checkpoint、未使用测试标签调参。
