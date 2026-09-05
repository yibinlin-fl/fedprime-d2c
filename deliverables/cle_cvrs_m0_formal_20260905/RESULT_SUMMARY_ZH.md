# CVRS M0 seed-0 Formal 结果审计

日期：2026-09-05

结论：`NO_GO_CVRS`
完整 HFL：未获授权

## 完整性

```text
archive bytes: 146448795
archive SHA256: 5916858CC71A7AFF1F18B4EEB90F7D3D0C9A13E0B695BFD2F7F7B278861DAB27
input bytes: 109142359
input SHA256: E9427A55DBE2545AF9D5A1EBD8BEA5B18C41C84D7FE89D06674165F4109E3818
taxonomy-free seal: PASS
six output checkpoint hashes: PASS
private stochastic paths matched: PASS
communication modified: false
```

`taxonomy_free_result.json` 声明 `oracle_assets_opened=false`，其 SHA256 与
`pre_oracle_seal.json` 一致。之后才在 `result.json` 中写入 oracle 指标。

## 结果

| Architecture | Arm | Routing strength | DSA | Avg (%) | Worst (%) |
|---|---|---:|---:|---:|---:|
| ResNet10 | Baseline | 0.304155 | 0.293632 | 37.3938 | 25.9000 |
| ResNet10 | Public-JSD | 0.308994 | 0.255995 | 37.0000 | 28.5000 |
| ResNet10 | CVRS | 0.215922 | 0.219023 | 37.8938 | 28.7000 |
| MobileNetV2 | Baseline | 0.376179 | 0.172066 | 50.6063 | 37.2000 |
| MobileNetV2 | Public-JSD | 0.244061 | 0.116098 | 47.6000 | 36.6000 |
| MobileNetV2 | CVRS | 0.155208 | 0.129098 | 52.0438 | 43.3000 |

## 冻结门槛重算

| Architecture | CVRS DSA 相对降幅 | JSD DSA - CVRS DSA | Avg 保留 | Worst 保留 | 结果 |
|---|---:|---:|---:|---:|---|
| ResNet10 | 25.409% | +0.036972 | +0.500 pp | +2.800 pp | PASS |
| MobileNetV2 | 24.972% | -0.013000 | +1.4375 pp | +6.100 pp | FAIL |

门槛要求每个架构分别满足 DSA 相对降幅至少 20%、`DSA_JSD-DSA_CVRS>=0.02`，且 Avg/Worst
相对 baseline 下降不超过 1 pp。MobileNetV2 的第二项不仅没有达到 `+0.02`，而且方向相反：
普通 Public-JSD 的 DSA 比 CVRS 低 `0.013000`。因此最终 verdict 必须为 `NO_GO_CVRS`。

## 科学解释

CVRS 不是完全无效：它在两个架构上都相对 baseline 降低约 25% DSA，并提高任务指标。
但 M0 的关键问题不是“CVRS 能否比什么都不加更好”，而是它是否比普通公共一致性正则提供
稳定、独有且跨架构的 CLE 抑制。MobileNetV2 给出了否定答案。

更关键的是，MobileNetV2 上 taxonomy-free routing strength 从 JSD 的 `0.244061` 进一步降到
CVRS 的 `0.155208`，oracle DSA 却从 `0.116098` 反升至 `0.129098`。这说明 CVRS 能有效优化
自己定义的公共响应 proxy，但该 proxy 的下降不保证真实 CLE shortcut 同步下降。

不得用两个架构的 pooled 均值覆盖逐架构门槛，也不得通过改阈值、调 lambda、补 seed 或完整
HFL 训练救回方法。CVRS 保留为有部分正信号但缺乏独特跨架构优势的负结果。

## 审计边界

结果包保存了六个 checkpoint、封存清单和汇总指标，但没有保存逐 source/operator 的原始
oracle logits。因此本次独立审计核验了 archive、seal、checkpoint 哈希和全部 gate 算术；
没有从原始预测张量重新生成 DSA。若未来需要论文级逐样本复算，只能重新运行只读 evaluator，
不能重新训练或改变当前 verdict。
