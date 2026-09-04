# P3-A 路由身份匹配因果审计

Updated: 2026-09-04

## 目的

P3-A 检验一个单独的机制问题：严格保持 corruption response 的幅度和几何，只交换 response
进入哪一个类别坐标，真实 CLE 的 Directional Shortcut Alignment（DSA）是否随之变化；以及
P2 的 taxonomy-free generic routing profile 是否比随机类别交换更会命中有害路由。

这不是训练方法，不修改 checkpoint，不运行通信，也不生成新的 corruption 或 PRIME view。

## 输入

- Phase-A1a round-40：H0/H9/L0/L9 × 4 clients × 1000 sources × 16 corruptions 的概率；
- P3-A clean-base completion：同一 1000 sources 和同一 16 checkpoints 的 clean logits；
- K0-B Bank-A、carrier-half Ua 的 frozen generic response；
- Phase-A1a binding/family 只在 taxonomy-free intervention 封存后用于 DSA 评分。

## Intervention

对每个 corruption response：

```text
d        = P_C [z_corrupt - z_clean]
d_target = P_pi d
z_target = z_clean + d_target
```

`pi` 完全由 K0-B Bank-A + Ua 上的 P2 class routing strength `g_c` 排序后做 rank reversal 得到。
绑定、corruption family、severity 和 DSA 均未参与 permutation 设计。另以 seed `20260904`
预先生成每个 arm/client 的 1000 个独立、唯一 random derangements。

## Frozen gates

- norm / pairwise Gram / singular-spectrum similarity / chi / energy / K0-B risk 误差 `<=1e-8`；
- H9 与 L9 targeted DSA absolute reduction 均 `>=0.05`；
- H9 与 L9 均至少 3/4 clients 同向；
- H9 与 L9 targeted DSA 均不高于 random-null q10；
- H0/L0 不新增超过 `0.02` 的 DSA。

## 执行

先用 8 个 random derangements 做 smoke；正式审计固定 1000 个：

```powershell
python scripts/analyze_post_no_go_p3a_routing_identity.py --mode formal --confirm-formal
```

正式结果为：

```text
CLASS_IDENTITY_CAUSAL_BUT_GENERIC_PROFILE_NOT_TARGETING
status: NO_GO_TO_METHOD
```

H9 targeted 虽将 DSA 从 `0.204270` 降至 `-0.013220`，但只位于 random null 的第
`60.1%` 百分位，未达到底部 10%；L9 位于第 `0.0%` 百分位并通过。由于 HFL 主对象 H9 的
targeting gate 失败，不能进入 P3-B，也不能据此设计训练方法。
