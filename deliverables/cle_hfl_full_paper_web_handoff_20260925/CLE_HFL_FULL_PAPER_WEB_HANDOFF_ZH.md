# CLE-HFL 网页端论文讨论交接（2026-10-13协议同步版）

> V0.6是不可变的网页端多Skill润色快照。当前事实与实验槽位在V0.7内部骨架中维护；
> 新结果完成并审计后才生成V0.8。不得直接覆写V0.6。

## 当前论文身份

论文不是“一个万能插件论文”，也不是“一个新通信框架论文”。统一定位为：

> A diagnosis--attribution--mitigation study of class--corruption entanglement in
> model-heterogeneous federated learning.

PEW+BER 是其中由 local-first 发现导出的 taxonomy-assisted、communication-agnostic local
mitigation module。它无需重设计服务器通信，但当前证据不支持“对任意 HFL 方法无损有效”。

## 六段证据链

```text
1. Existence: paired counterfactual + DSA
2. Attribution: matched HFL-vs-Local, predominantly local-first
3. Mitigation: ERM/CVaR/PEW+GroupDRO/PEW+BER, held-out map2, 40 rounds, 3 seeds
4. Cross-communication: completed FedDF boundary evidence + pending FedMD four-objective replication
5. Field position: six representative HFL rows + two matched AsymHFL rows on the moderate protocol
6. External boundary: second controlled dataset, bounded taxonomy stress, Oracle granularity, limitations
```

这六段链条现在使用“双难度、同一研究对象”的协议：

- strong stress：`alpha=0.5/rho=1.0/gamma=0.9`，继续承担existence、DSA、local-first和
  Axis I shortcut-objective attribution；
- moderate utility：`alpha=1.0/rho=0.5/gamma={0,0.5,0.9}`，承担待补dose response和
  Axis II HFL领域比较；
- 两者都基于CIFAR-10与受控synthetic corruptions，因此moderate协议不是第二private dataset，
  也不关闭现实外部有效性缺口。

## map1 与 map2 的职责

- map1 已完成，用于证明改变 client-specific binding map 后，PEW+BER 的 DSA suppression 仍复现。
- map2 未参与早期五臂筛选，并已有 40 轮三 training-seed 主方法结果，因此用于最终 matched
  strong-objective comparison。新moderate协议也固定同一map2，使binding身份不变，只改变
  label-skew、corruption exposure与CLE强度；它用于dose response和HFL领域表。两个协议的结果
  必须分表报告。
- map1 不是不能用于领域表；若使用它，必须重跑主方法锚点，增加成本而不增加新科学问题。

## 已完成、可写入正文的核心事实

1. strong CLE directional shortcut 存在；普通 accuracy 不能替代 DSA。
2. 当前 matched factorial 支持 predominantly local-first，不支持“通信完全无关”。
3. held-out map2 40轮三seed中，PEW+BER 相对 ERM 和同PEW分组的 GroupDRO 均稳定降低 DSA；
   pooled DSA 与 utility trade-off 也优于 CVaR，但并非每个客户端 DSA 都支配 CVaR。
4. map1 复现 DSA suppression。
5. FedDF-fidelity 下 DSA suppression 复现，但 utility gate 与 learning floor 失败，只能作为边界证据。
6. Oracle family/operator gap 很小，不支持升级 operator-level PEW。
7. PEW 是 coarse family-level、taxonomy-assisted witness；未知、复合、连续环境属于外部有效性边界。

## 待补实验，禁止写成已完成

1. FedMD 固定通信的四目标复现；
2. moderate held-out-map2剂量响应：ERM的`gamma=0/0.5/0.9`及`gamma=0.9` PEW+BER；
3. moderate held-out-map2八行HFL领域表（六个context方法加两条AsymHFL锚点）；
4. local-first training seeds 1/2 的成本重构后复验；
5. 真正第二受控私有数据集；
6. bounded taxonomy stress；
7. JTT Formal 仅在正文要直接声称胜过 JTT 时需要。

## 当前 LaTeX

V0.7 内部骨架：

```text
deliverables/cle_hfl_latex_v0_7_20260925/source/main.tex
```

V0.7 已同步最终定位与实验槽位，但不是 submission-ready，不得把 `\evidenceneeded{}` 占位符
删除或伪造数字。2026-10-13已加入strong/moderate双难度职责、moderate剂量响应表和moderate
Axis II表头。V0.6保留为不可变历史版本；待新Formal回填后再生成V0.8。

## 给网页端模型的约束

1. 只能改善论证结构、语言、篇幅与可读性；不得改变实验数字和证据状态。
2. 不得把 local-first 改写为 “federation is unnecessary”。正确含义是 shortcut 在本地形成并
   在异构联邦协作中持续存在，现有通信没有可靠洗掉它。
3. 不得把 PEW+BER 写成 universal plug-in、taxonomy-free 或 utility-preserving。
4. 不得把 FedDF 边界实验写成完整跨通信胜利。
5. HFL领域表是 protocol-matched mechanism-family context，不是 untouched official leaderboard。
6. 保留四项贡献：问题定义、paired DSA与识别边界、local-vs-communication机制归因、
   taxonomy-assisted mitigation与受控验证。
