# CLE-HFL 最终论文定位（2026-09-25）

## 一句话定位

整篇论文是一项关于 CLE-HFL 的“诊断—归因—缓解”研究；PEW+BER 是由 local-first 发现导出的、
taxonomy-assisted、communication-agnostic 的本地缓解模块。它可以在不重设计通信协议的情况下
嵌入 HFL 训练，但本文不声称它对任意 HFL 算法都无损、普适或保持任务效用。

## 最终证据链

```text
存在
-> 形成归因
-> 匹配方法比较
-> 跨通信复现
-> HFL领域位置
-> 外部有效性边界
```

1. **存在**：source-paired operator counterfactual 与 DSA 证明 strong CLE 会产生沿训练 binding
   方向的类别概率移动，而不仅是普通 corruption 难度。
2. **形成归因**：matched HFL-vs-Local factorial 表明，该现象在当前受控场景中主要形成于客户端
   本地优化；联邦通信没有可靠地把它洗掉。论文不得写成“通信不重要”或“问题与联邦无关”。
3. **匹配方法比较**：固定 strict AsymHFL-val 通信，仅改变本地目标，比较 ERM、CVaR-DRO、
   PEW+GroupDRO 与 PEW+BER；held-out map2、40轮、training seeds 0/1/2 已完成并通过门槛。
4. **跨通信复现**：固定 FedMD 通信后重复同一四目标比较；FedDF 两臂结果保留为已有边界证据，
   不代替 FedMD 四臂的目标归因。
5. **HFL领域位置**：在同一 held-out map2、seed 0、40轮协议中比较十种代表性 HFL 机制，
   并复用完全匹配的 AsymHFL--ERM 与 AsymHFL+PEW+BER 两行形成十二行领域表。
6. **外部边界**：第二受控私有数据集、bounded taxonomy stress、Oracle 粒度审计和诚实 limitations。

## 为什么领域表使用 map2

map1 并非无效，它已经作为“换 binding map 后仍能降低 DSA”的 cross-binding 复现实验完成。
但最终领域比较优先使用 map2，理由是：

1. map2 没有参与早期五臂筛选，方法选择与最终比较隔离；
2. map2 已有 40 轮、training seeds 0/1/2 的主方法结果；
3. 领域表可以直接复用 seed-0 的 AsymHFL--ERM 和 AsymHFL+PEW+BER 两行；
4. 若改用 map1，需要在 map1 上重跑这两行，既增加算力，也不能增加新的科学问题。

因此：map1 负责跨 binding 复现；map2 负责 held-out 主比较、跨通信复现和最终领域表。两者职责不同，
不是 map1 “不能用”。

## 审稿表述边界

推荐写法：

> Client-specific CLE is formed primarily during local optimization and persists through heterogeneous
> federated collaboration; the evaluated communication mechanisms do not reliably wash it out.

禁止写法：

- “CLE 与联邦学习无关”；
- “PEW+BER 是适用于所有 HFL 方法的通用无损插件”；
- “我们首次研究联邦学习中的 spurious correlation”；
- “FedDF 两臂证明跨通信普遍提升效用”。
