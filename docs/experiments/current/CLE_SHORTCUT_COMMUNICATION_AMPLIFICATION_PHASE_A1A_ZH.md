# CLE Shortcut Communication Amplification Phase-A1a 归因与 OpenI 运行规范

Updated: 2026-08-30

## 0. 状态与目的

```text
status: COMPLETE; NO_GO_FL_SPECIFIC_AMPLIFICATION
implementation: COMPLETE; INTEGRITY PASS; INDEPENDENT RECOMPUTATION MATCH
training: H0/H9/L0/L9 40-ROUND COMPLETE ON OPENI
commit/push: 6199dd8 / origin/main
```

Phase-A0 已在历史 CLE-v1 diagnostic RAHFL checkpoint 上证明：`gamma=0.9` 会产生强方向性
corruption-to-class shortcut。Phase-A1a 不重复证明该现象，而是判断联邦协作是否在匹配的
本地训练之上进一步放大它。

本阶段不研究 PEW/BER、坏教师、teacher selection、攻击、标签噪声、Centralized 或新方法。

## 1. 资产与版本审计结论

历史 Phase-A0 checkpoint 来自：

```text
diag_rahfl_cle_alpha05_gamma00_seed0
diag_rahfl_cle_alpha05_gamma09_seed0
pretrain_epochs=0
rounds=40
local_epochs=1
AugMix/JSD + DCL + AsymHFL
```

当前没有可复用的严格匹配 CLE-v1 Local-only checkpoint。已有 Local-only 资产分别属于普通
CIFAR-10-C、PEW/BER 消融，或 CLE-v2 `gamma=0.9` 12-round external baseline；后者没有
`gamma=0`，且 `save_final=false`，不能计算 DSA。

`scripts/prepare_cle_data.py` 与 diagnostic configs 自提交 `5870fd7` 后未修改，但其依赖的
loaders 及 RAHFL runner 后续发生了实质扩展。历史本地 gamma00/gamma09 prepared archives
已不在当前工作区，无法把新 Local 训练输入与旧 HFL 输入做逐字节审计。即使重新生成参数
相同的数据，也可能混入依赖版本、corruption 实现环境和训练 runner 差异。

因此不采用：

```text
new Local-only checkpoints vs historical HFL checkpoints
```

推荐在同一个新冻结数据包、同一个代码提交和同一个 OpenI job 中联合产生 HFL 与 Local。

## 2. 数学对象

令 `m in {H, L}` 分别表示 HFL 和 Local-only，`g in {0, 0.9}` 表示 CLE 强度，`k` 表示
客户端。使用 Phase-A0 完全相同的配对干预网格定义：

```text
S[m,g,k] = DSA of client k under training arm (m,g)
E[m,k]   = S[m,0.9,k] - S[m,0,k]
A[k]     = E[H,k] - E[L,k]
A_pool   = mean_k A[k]
```

`E[m,k]` 去除模型在 independent corruption 下的固有 family response；`A[k]` 是差分中的
差分，只在完整匹配成立时解释为 communication-induced shortcut amplification。

## 3. 最小四个 arm

```text
H0: HFL        gamma=0.0
H9: HFL        gamma=0.9
L0: Local-only gamma=0.0
L9: Local-only gamma=0.9
```

四个 arm 固定：

```text
dataset: CIFAR-10 CLE-v1
alpha: 0.5
training seed: 0
clients: 4
models: ResNet10 / ResNet12 / ShuffleNet / MobileNetV2
pretrain_epochs: 0
rounds: 40
local_epochs: 1
batch_size: 64
optimizer: Adam
lr: 0.001
weight_decay: 0
local modules: AugMix/JSD + DCL
fit/audit protocol: strict persisted client-private split
```

唯一允许的 H/L 差异：

```text
H0/H9: strict AsymHFL-val public-logit communication enabled
L0/L9: communication is an exact no-op
```

旧 diagnostic `asymhfl` 使用最终测试准确率做路由，只能作为历史 failure diagnostic，不能
进入新的因果归因实验。四个 arm 必须共享同一份持久化 fit/audit split：

```text
HFL and Local private gradients: fit only
HFL teacher routing: client-private audit only
Local audit: held out and unused by optimization
final test: reporting only for every arm
```

因此本阶段估计的是完整、合规的 strict AsymHFL collaboration 相对匹配 Local-only 的放大
效应，不再把 test-label routing 带入结论。

## 4. 必须隔离随机性

只把配置中的 seed 写成相同并不足够。协作阶段可能消耗全局 RNG，进而改变后续 AugMix 和
private-loader 顺序，使 H/L 差异混入不同的数据增强。

实现 harness 时必须：

```text
1. 每个 client 的初始 state_dict 只生成一次，四个 arm 逐字节复用；
2. private sampler 使用固定 client/round seed；
3. AugMix 使用固定 arm-independent client/round/batch seed；
4. communication 使用独立 RNG stream；
5. H/L 每轮每个 client 的首个 local batch 记录全部 AugMix/DCL views 与标签的 checksum；
6. H/L 对应 arm 的首批标签与全部增强视图 checksum 必须一致。
```

通信更新改变模型后产生的后续优化差异属于目标因果路径；通信额外消耗 RNG 导致的数据变化
不属于目标路径。

## 5. 数据与完整性清单

新数据包必须保存：

```text
canonical CIFAR-10 archive SHA256
generator Git commit
dependency/version manifest
per-client source indices
per-client true labels
per-client corruption family/operator/severity
class-to-family map
persisted per-client fit/audit indices
all generated array SHA256
four shared initial checkpoint SHA256
```

完整性门槛：

```text
I1 H/L within each gamma use byte-identical private arrays and fit/audit indices
I2 gamma0/gamma9 use identical source indices, true labels and fit/audit roles
I3 all four arms use byte-identical initial model states per client
I4 H/L 每轮、每客户端首个 local batch 的标签和全部增强视图 checksum 完全一致
I5 configs differ only in gamma data root and communication switch
I6 HFL routing reads audit only; Local never optimizes on audit
I7 evaluation grid is Phase-A0 seed20260830, 1000x16, severity3
I8 final-test labels never participate in training, routing, selection or stopping
I9 all round-40 final checkpoints and round-12 diagnostic checkpoints are persisted
```

任一完整性门槛失败，结果为 `INVALID`，不得解释通信效应。

## 6. 评价与统计

在 round 12 和 round 40 checkpoint 上分别运行 Phase-A0 分析器：

```text
1000 balanced clean sources
16 CLE-v1 operators
severity=3
same source/operator grid for every arm
```

主结果为 round 40 的 `A_pool`。round 12 只用于观察放大是否早期出现，不替代最终判定。

统计：

```text
2000 paired source bootstraps
paired over source, operator, client and all four arms
report A_pool, CI95, A[k]
report probability DSA and family-bound top-1 bias
secondary: Avg, Worst, WCCA, CFG, paired flip rate, entropy
```

## 7. 冻结晋级门槛（建议）

完整性全部通过后，通信放大进入下一阶段需同时满足：

```text
G1 A_pool >= 0.020
G2 paired bootstrap CI95 lower bound > 0
G3 at least 3/4 clients have A[k] > 0
G4 family-bound top-1 bias amplification > 0
G5 HFL gamma09 DSA remains above its group-size-preserving shuffled-map p95
```

解释：

```text
any integrity failure -> INVALID
valid but any G1--G5 failure -> NO-GO for FL-specific amplification
all G1--G5 pass -> GO only to Centralized/Homogeneous-FL attribution design
```

通过不等于论文 GO，也不允许直接设计防御方法。

## 8. 为什么 HFL 与 Local 要一起重跑

联合重跑不是为了再次证明 CLE shortcut 存在，而是确保：

```text
same data
same code
same initialization
same local stochastic path
same training budget
same fit/audit/test roles
only strict AsymHFL collaboration differs
```

这样 `A_pool` 才能解释为通信放大。新 Local 与旧 HFL 的拼接比较会同时改变代码版本、输入
归档、初始化轨迹和 RNG consumption，即使数值不同，也不能知道差异是否来自通信。

## 9. 已实现入口与输入包

本地已生成自包含 OpenI 输入包：

```text
local_runs/cle_shortcut_amplification_phase_a1a/
  cle_shortcut_amplification_phase_a1a_seed0.tar.gz

bytes: 408228487
SHA256: 6322F16513C6980CDC5904D7EF91204A241205BC76DCCE8BC450E635519B4202
```

代码入口：

```text
数据构造：scripts/prepare_cle_shortcut_amplification_phase_a1a_data.py
OpenI：    scripts/openi_cle_shortcut_amplification_phase_a1a_entry.py
分析器：   scripts/analyze_cle_shortcut_amplification_phase_a1a.py
```

正式 OpenI 参数只需：

```text
--device=auto
```

不要额外传 `--arms`；默认 `all` 按 `H0 -> H9 -> L0 -> L9` 依次运行，每个 arm 完成后立即
独立打包，四臂齐全后自动进行 round12/round40 DSA DiD 分析。输入包已包含 paired private
arrays、持久化 strict split、Phase-A0 的 1000 张 clean source、CIFAR-100 public tar 和全部哈希。

共享初始模型不预先打包；OpenI job 内只生成一次并由四臂逐字节复用，同时保存 SHA256
清单。这样共享初始化成立，且不要求本机与 OpenI 的 PyTorch 初始化实现逐字节相同。

Focused verification on 2026-08-30：

```text
py_compile: PASS
Phase-A0/Phase-A1a/communication focused tests: 13 passed
paired data manifest/hash verification: PASS
shared initialization/config preflight: PASS
formal training: H0/H9/L0/L9 COMPLETE
```

正式训练与推理均在 OpenI 完成，本地 RTX 3050 未运行正式训练。

## 10. 正式结果与判定

全部预注册完整性门槛通过。OpenI summary 与本地从 round12/round40 原始概率缓存的独立复算
完全一致。round 40 正式结果：

```text
H0 DSA:                 0.0015228341
H9 DSA:                 0.2042704937
L0 DSA:                 0.0008234010
L9 DSA:                 0.2051892788
HFL CLE effect:         0.2027476596
Local CLE effect:       0.2043658778
A_pool:                -0.0016182182
CI95:                  [-0.0033365882, 0.0001283891]
A_client:              [-0.0027690681, -0.0023939108, -0.0016801117, +0.0003702177]
top1 amplification:     0.0025799851
H9 shuffled p:          0.000999001
```

```text
G1 minimum amplification: FAIL
G2 positive CI:           FAIL
G3 >=3 positive clients:  FAIL (1/4)
G4 top-1 amplification:   PASS
G5 H9 binding-specific:   PASS
formal verdict:           NO_GO_FL_SPECIFIC_AMPLIFICATION
```

Round 12 的 `A_pool=-0.0169168048`、CI `[-0.0183210229,-0.0155504985]` 表示早期短暂抑制，
但到 round 40 已收敛到近零，因此不能主张通信具有持久防御效果。可支持的机制结论是：CLE
在 HFL 与 Local 中都产生约 `0.204` 的强方向性 shortcut，主要在本地训练阶段形成；strict
AsymHFL 既没有额外放大，也没有持久消除。

```text
result archive: cle_shortcut_amplification_phase_a1a_seed0_analysis_outputs.tar.gz
result bytes:   19322309
result SHA256:  FDF1BEC2395334DD3816BC9C3F594B01D814DB22C0CC245CD1378A45180F397C
```

用户决定继续 CLE，但通信放大假设已经冻结为 NO-GO。下一步只允许先定义并查重一个
local-first、无环境/损坏标签、无 clean counterpart/source-index oracle 的 directional
shortcut suppression 数学对象；不得通过增加 seed、轮数、路由调参或改名来挽救本假设。
