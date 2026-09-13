# CLE代理不可识别性、CVRS反例与DSA必要性验证

日期：2026-09-13

## 结论

```text
protocol: cle_proxy_nonidentifiability_cache_validation_v1
training_or_inference_run: false
N0/N1/N2/N3/N4: all PASS
```

仅观察类别`Y`与代理环境`Z`，在不约束代理误差通道时，无法非平凡地识别真实环境`E`与类别
之间的依赖。本次构造使用Stage-1缓存的1000个均衡source标签和五个均衡proxy组：同一份
`P(Y,Z)`可以扩展成真实依赖TV为0的潜在世界，也可以扩展成真实依赖TV为0.8的潜在世界。

| 检查 | 数值 | 结果 |
|---|---:|---|
| 可观测`TV(Y,Z)` | 0.000000 | N0 PASS |
| 同一观测下世界A `TV(Y,E)` | 0.000000 | N1 PASS |
| 同一观测下世界B `TV(Y,E)` | 0.800000 | N2 PASS |
| 冻结缓存 identical-view JSD | 0.000000 | N3 PASS |
| 同一缓存 cross-operator DSA | 0.119644 | N3 PASS |
| CVRS相对Public-JSD proxy变化 | -0.088854 | N4 PASS |
| CVRS相对Public-JSD DSA变化 | +0.013000 | N4 PASS |

CVRS数值来自Formal `result.json`而不是手抄汇总：MobileNetV2上routing proxy从`0.244061`
下降到`0.155208`，oracle DSA却从`0.116098`上升到`0.129098`。Stage-1 JSD反例从冻结
`STAGE1_PREDICTIONS.npz`重新计算，得到JSD严格为0、DSA为`0.1196442241`。

## 理论含义

该验证支持三层区分：

1. BER定理直接控制伪环境支持结构；
2. 传递到真实环境需要PEW误差、稳定通道或独立审计等额外条件；
3. 没有这些条件时，必须以paired DSA或等价的target-aligned estimand直接验证真实CLE行为。

这里的“DSA必要”是针对本文证据协议，不是声称DSA是所有shortcut研究唯一合法指标。

## 输入完整性

```text
STAGE1_PREDICTIONS.npz
SHA256 FE8212372A53BC01E5BB51B9B10249ABDF33D2E215170FB021D80EEA930F11C3

CVRS result.json
SHA256 D57AE3E3B08BD93D1C8D93F05F5B228B91E7384B8C5B4E2B6C58D0C593737D3A
```

## 产物

```text
PROXY_NONIDENTIFIABILITY_VALIDATION.json
PROXY_NONIDENTIFIABILITY_THEORY.png
PROXY_NONIDENTIFIABILITY_THEORY.pdf
```

完整定理、构造证明、条件传递界和论文表述边界位于：

```text
docs/research/status/CLE_PROXY_NONIDENTIFIABILITY_THEORY_2026_09_13_ZH.md
```
