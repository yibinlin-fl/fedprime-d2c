# P4 Routing Targetability Gap Audit

Updated: 2026-09-04

P4 解释 P3-A 中 H9 targeted percentile `60.1%`、L9 `0.0%` 的分叉。它不救 P2，不搜索新
permutation，也不产生方法；全过程只读取已封存输出并做 CPU 计算。

真实 harmful routing 定义与 DSA 使用同一 binding-family 和 valid-source 条件。对每个
client/class 构造 `M_harm[a,b]` 和正向边际 `g_harm[a]`，再与 K0-B Bank-A + Ua 的 P2
`g_generic` 比较。H9 的 mean cosine/Spearman 为 `0.8496/0.7091`，Top-3 overlap
`0.8333`；L9 为 `0.8459/0.4667` 和 `0.8333`。因此没有证据支持“通信令 generic profile
与 harmful routing 解耦”。

真正分叉发生在 frozen rank-reversal 对完整矩阵的作用。H9 signed destructive percentile
只有 `42.5%`，L9 为 `100.0%`；对应 P3-A DSA percentile 分别为 `60.1%/0.0%`。随机控制中
destructive score 与 DSA reduction 的相关也为中高强度。允许结论：

```text
GENERIC_ROUTING_ALIGNS_WITH_HARMFUL_ROUTING_BUT_TARGET_RULE_FAILED
INFORMATION_MAY_EXIST_IDENTIFIABILITY_AUDIT_REQUIRED
METHOD_GO = false
```

边界是：generic marginal class salience 并不等价于可用于 intervention 的 pairwise routing
map。不得把 oracle `M_harm` 直接作为训练目标，不得调 P3-A rank reversal。若继续，必须先在
纸面上证明 taxonomy-free observables 是否可能识别缺失的 pairwise information。
